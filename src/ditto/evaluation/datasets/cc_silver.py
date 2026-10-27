"""Common Crawl silver set: labels from the sites' own JSON-LD, and the evaluation adapter.

The dataset is built by the scripts in ``eval/datasets/commoncrawl_silver/`` (``select.py``,
``download.py``, ``labels.py``, ``noise_check.py``). They write, per selected domain,
``data/cc/<domain>/*.warc.gz`` (the captures, as fetched from Common Crawl) and
``data/cc/<domain>/labels.jsonl`` (one JSON object per labeled page, keyed by the
``WARC-Target-URI``). This module holds the pure parts they share:

* :data:`VERTICALS` -- per vertical, the schema.org target types and the label fields;
* :func:`page_labels` -- JSON-LD objects of the target type (via ``extruct``), flattened to the
  fields of ``fields/<vertical>.yaml``;
* :func:`visible_text` / :func:`label_agrees` -- the markup-noise check's comparison;
* :func:`load_sites` -- the evaluation adapter (``ditto.evaluation.datasets.base`` protocol).

Scalar labels are a string or a list of acceptable alternative spellings of the same value
(e.g. ``["PT15M", "15 mins", "15 minutes", ...]`` for an ISO-8601 duration, ``["12.5", "12.50"]``
for a price, ``["2026-10-31", "31 October 2026", ...]`` for a date, ``["GBP", "£"]`` for a
currency, and an identifier's displayed spelling such as ``334/2679`` for ``sku`` ``3342679``),
so that a prediction copied from the visible page can match a value the site encoded in machine
form. ``ingredients`` is a real list field. Fields the JSON-LD object lacks are ``None`` in
``labels.jsonl``; the adapter leaves them out of the page's labels, so they are not scored on
that page (the JSON-LD omitting a property does not show that the page lacks it).
"""

from __future__ import annotations

import html as html_lib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import yaml
from extruct.jsonld import JsonLdExtractor
from lxml import html as lxml_html

from ditto.evaluation.datasets.base import (
    DatasetUnavailable,
    EvalSite,
    LabelKind,
    Selection,
    is_empty,
    select_sites,
)
from ditto.fingerprint.dom import clean_tree, parse_html
from ditto.ingest.encoding import charset_from_content_type, detect_encoding

SCHEMA_ORG = re.compile(r"^https?://(?:www\.)?schema\.org/", re.IGNORECASE)
DATASET_SUBDIR = Path("eval/datasets/commoncrawl_silver")
DEFAULT_DATA_ROOT = Path("data/cc")
LABELS_FILE = "labels.jsonl"

_EVENT_TYPES = frozenset(
    {
        "Event",
        "BusinessEvent",
        "ChildrensEvent",
        "ComedyEvent",
        "CourseInstance",
        "DanceEvent",
        "EducationEvent",
        "EventSeries",
        "ExhibitionEvent",
        "Festival",
        "FoodEvent",
        "Hackathon",
        "LiteraryEvent",
        "MusicEvent",
        "SaleEvent",
        "ScreeningEvent",
        "SocialEvent",
        "SportsEvent",
        "TheaterEvent",
        "VisualArtsEvent",
    }
)
_NEWS_TYPES = frozenset(
    {
        "NewsArticle",
        "AnalysisNewsArticle",
        "AskPublicNewsArticle",
        "BackgroundNewsArticle",
        "OpinionNewsArticle",
        "ReportageNewsArticle",
        "ReviewNewsArticle",
    }
)


@dataclass(frozen=True)
class Vertical:
    """A target vertical: accepted schema.org ``@type`` values and the label fields."""

    name: str
    types: frozenset[str]
    fields: tuple[str, ...]
    list_fields: frozenset[str] = frozenset()
    identifiers: frozenset[str] = frozenset()
    """Code-like fields whose displayed spelling may differ (see :func:`add_display_forms`)."""

    @property
    def primary_type(self) -> str:
        return {
            "product": "Product",
            "job": "JobPosting",
            "recipe": "Recipe",
            "event": "Event",
            "news": "NewsArticle",
            "book": "Book",
        }[self.name]

    def kinds(self) -> dict[str, LabelKind]:
        return {name: "list" for name in self.fields if name in self.list_fields}


VERTICALS: dict[str, Vertical] = {
    v.name: v
    for v in (
        Vertical(
            "product",
            frozenset({"Product", "ProductGroup", "IndividualProduct", "ProductModel"}),
            ("name", "price", "priceCurrency", "availability", "brand", "sku", "image"),
            identifiers=frozenset({"sku"}),
        ),
        Vertical(
            "job",
            frozenset({"JobPosting"}),
            ("title", "hiringOrganization", "location", "datePosted", "employmentType", "salary"),
        ),
        Vertical(
            "recipe",
            frozenset({"Recipe"}),
            ("name", "prepTime", "cookTime", "recipeYield", "ingredients", "author"),
            frozenset({"ingredients"}),
        ),
        Vertical(
            "event",
            _EVENT_TYPES,
            ("name", "startDate", "endDate", "location", "address", "organizer", "performer"),
        ),
        Vertical(
            "news",
            _NEWS_TYPES,
            ("headline", "author", "datePublished", "articleSection", "description", "image"),
        ),
        Vertical(
            "book",
            frozenset({"Book"}),
            ("name", "author", "isbn", "publisher", "datePublished", "numberOfPages"),
            identifiers=frozenset({"isbn"}),
        ),
    )
}

# --------------------------------------------------------------------------- decoding / parsing


def decode_body(body: bytes, content_type: str | None = None) -> str:
    """Decode page bytes exactly as ``ditto ingest`` does (WHATWG-style sniffing)."""
    encoding = detect_encoding(body, charset_from_content_type(content_type))
    return body.decode(encoding, errors="replace")


_EXTRACTOR = JsonLdExtractor()
_JSONLD_SCRIPTS = "//script[translate(@type, 'LDJSON', 'ldjson')='application/ld+json']"


def jsonld_items(root: lxml_html.HtmlElement) -> list[Any]:
    """Top-level JSON-LD items of a parsed page, one ``<script>`` block at a time.

    Uses ``extruct``'s JSON-LD extractor per block so one malformed block does not lose the
    others.
    """
    items: list[Any] = []
    for script in root.xpath(_JSONLD_SCRIPTS):
        try:
            items.extend(_EXTRACTOR.extract_items(script))
        except Exception:  # malformed JSON in a site's markup: skip that block only
            continue
    return items


def type_names(node: dict[str, Any]) -> set[str]:
    """``@type`` of a JSON-LD node as bare schema.org names."""
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {SCHEMA_ORG.sub("", v).strip() for v in values if isinstance(v, str)}


def find_typed(items: Iterable[Any], types: frozenset[str]) -> list[dict[str, Any]]:
    """JSON-LD nodes whose ``@type`` is in ``types``, outermost first.

    Walks ``@graph`` and nested values, but not into a node that already matched (so a
    ``ProductGroup``'s variants or a ``Recipe``'s nested objects are not separate hits).
    """
    found: list[dict[str, Any]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 12:
            return
        if isinstance(value, list):
            for item in value:
                walk(item, depth + 1)
        elif isinstance(value, dict):
            if type_names(value) & types:
                found.append(value)
                return
            for item in value.values():
                walk(item, depth + 1)

    for item in items:
        walk(item, 0)
    return found


# --------------------------------------------------------------------------- value helpers


def clean_text(value: Any) -> str | None:
    """A JSON-LD literal as display text: HTML entities and tags resolved, whitespace collapsed."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, dict):
        for key in ("name", "@value", "text", "value"):
            if key in value:
                return clean_text(value[key])
        return None
    if isinstance(value, list):
        for item in value:
            text = clean_text(item)
            if text:
                return text
        return None
    text = html_lib.unescape(str(value))
    if "<" in text and ">" in text:
        text = re.sub(r"<[^>]+>", " ", text)
        text = html_lib.unescape(text)
    text = " ".join(text.split())
    return text or None


def _number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def names(value: Any) -> list[str]:
    """Every name in a value that may be a string, an object with ``name``, or a list of those."""
    out: list[str] = []
    for item in value if isinstance(value, list) else [value]:
        text = clean_text(item.get("name") if isinstance(item, dict) else item)
        if text and text not in out:
            out.append(text)
    return out


def first(value: Any) -> Any:
    if isinstance(value, list):
        return next((v for v in value if v not in (None, "", [], {})), None)
    return value


def scalar(alternatives: Iterable[str | None]) -> str | list[str] | None:
    """A scalar label: one string, or the de-duplicated list of acceptable alternatives."""
    out: list[str] = []
    for alt in alternatives:
        if alt and alt not in out:
            out.append(alt)
    if not out:
        return None
    return out[0] if len(out) == 1 else out


def enum_name(value: Any) -> str | list[str] | None:
    """schema.org enumeration (``https://schema.org/InStock``, ``FULL_TIME``) plus readable forms."""
    text = clean_text(first(value))
    if not text:
        return None
    bare = SCHEMA_ORG.sub("", text)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", bare).replace("_", " ").strip()
    return scalar([bare, spaced, spaced.replace(" ", "-")])


def price_forms(value: Any) -> str | list[str] | None:
    """A price plus its usual spellings: ``1299.5`` -> ``1299.50``, ``1,299.50``, ``1299,50``."""
    text = clean_text(first(value))
    if not text:
        return None
    alts = [text]
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text
    alts += _amount_forms(number)
    return scalar(alts)


def _amount_forms(number: float) -> list[str]:
    """Spellings of an amount: plain, two decimals, thousands-grouped, decimal comma.

    Dot-grouped integers (``1.000``) are left out: a reader takes them for decimals.
    """
    two = f"{number:.2f}"
    grouped = f"{number:,.2f}"
    alts = [_number(number), two, grouped, two.replace(".", ",")]
    if abs(number) >= 1000:
        alts.append(grouped.translate(str.maketrans(",.", ".,")))
    if number.is_integer():
        alts.append(f"{int(number):,}")
    return alts


CURRENCY_SYMBOLS: dict[str, str] = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
    "JPY": "¥",
    "CNY": "¥",
    "INR": "₹",
    "KRW": "₩",
    "RUB": "₽",
    "AUD": "$",
    "CAD": "$",
    "NZD": "$",
    "SGD": "$",
    "HKD": "$",
    "MXN": "$",
}
"""Display symbol of common ISO 4217 codes (a page usually shows the symbol, not the code)."""


def currency_forms(value: Any) -> str | list[str] | None:
    """A currency as given (usually its ISO code) plus its display symbol when known."""
    text = clean_text(first(value))
    if not text:
        return None
    return scalar([text, CURRENCY_SYMBOLS.get(text.upper())])


_DURATION = re.compile(
    r"^P(?:(?P<d>\d+(?:\.\d+)?)D)?(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)S)?)?$",
    re.IGNORECASE,
)


def duration_forms(value: Any) -> str | list[str] | None:
    """ISO-8601 duration plus the usual human spellings (``1 hr 30 mins``, ``90 minutes``)."""
    text = clean_text(first(value))
    if not text:
        return None
    match = _DURATION.match(text.strip())
    if not match or not any(match.groupdict().values()):
        return text
    parts = {k: float(v) if v else 0.0 for k, v in match.groupdict().items()}
    total = round(parts["d"] * 1440 + parts["h"] * 60 + parts["m"] + parts["s"] / 60)
    hours, minutes = divmod(total, 60)
    alts = [text]
    if hours == 0:
        alts += [f"{minutes} mins", f"{minutes} minutes", f"{minutes} min", f"{minutes}m"]
        if minutes == 1:
            alts.append("1 minute")
    else:
        hr = "hr" if hours == 1 else "hrs"
        hour = "hour" if hours == 1 else "hours"
        if minutes == 0:
            alts += [f"{hours} {hr}", f"{hours} {hour}", f"{hours}h", f"{hours} hr"]
        else:
            mins = "min" if minutes == 1 else "mins"
            minute = "minute" if minutes == 1 else "minutes"
            alts += [
                f"{hours} {hr} {minutes} {mins}",
                f"{hours} {hour} {minutes} {minute}",
                f"{hours} hr {minutes} min",
                f"{hours}h {minutes}m",
            ]
        alts += [f"{total} mins", f"{total} minutes"]
    return scalar(alts)


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def date_forms(value: Any) -> str | list[str] | None:
    """An ISO date/datetime plus its date part and the usual English display spellings.

    ``2026-10-31T19:30`` also matches ``31 October 2026``, ``31 Oct 2026``, ``October 31, 2026``,
    ``Oct 31, 2026``, ``Oct 31 2026`` and ``10/31/2026``; ``31/10/2026`` is added only when the
    day/month order is unambiguous (day above 12).
    """
    text = clean_text(first(value))
    if not text:
        return None
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return text
    year, month, day = (int(g) for g in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return scalar([text, text[:10]])
    full = _MONTHS[month - 1]
    abbr = full[:3]
    alts = [
        text,
        text[:10],
        f"{day} {full} {year}",
        f"{day} {abbr} {year}",
        f"{full} {day}, {year}",
        f"{abbr} {day}, {year}",
        f"{abbr} {day} {year}",
        f"{month:02d}/{day:02d}/{year}",
    ]
    if day > 12:
        alts.append(f"{day:02d}/{month:02d}/{year}")
    return scalar(alts)


def image_url(value: Any, base_url: str | None) -> str | None:
    item = first(value)
    if isinstance(item, dict):
        item = item.get("url") or item.get("contentUrl") or item.get("@id")
        item = first(item)
    text = clean_text(item)
    if not text:
        return None
    return urljoin(base_url, text) if base_url else text


def address_forms(value: Any) -> str | list[str] | None:
    """A PostalAddress (or plain string) as ``street, locality, region`` and shorter forms."""
    addr = first(value)
    if isinstance(addr, str):
        return clean_text(addr)
    if not isinstance(addr, dict):
        return None
    country = addr.get("addressCountry")
    parts = {
        key: clean_text(addr.get(key))
        for key in ("streetAddress", "addressLocality", "addressRegion", "postalCode")
    }
    country_text = clean_text(country)
    locality, region = parts["addressLocality"], parts["addressRegion"]
    full = ", ".join(
        p
        for p in (parts["streetAddress"], locality, region, parts["postalCode"], country_text)
        if p
    )
    return scalar(
        [
            full,
            ", ".join(p for p in (locality, region, country_text) if p),
            ", ".join(p for p in (locality, region) if p),
            locality,
        ]
    )


# --------------------------------------------------------------------------- flatteners


def _offer(obj: dict[str, Any]) -> dict[str, Any]:
    """The first concrete offer of a product (``AggregateOffer`` / ``ProductGroup`` aware)."""
    offers = first(obj.get("offers"))
    if not isinstance(offers, dict) and isinstance(obj.get("hasVariant"), list):
        for variant in obj["hasVariant"]:
            if isinstance(variant, dict) and variant.get("offers"):
                offers = first(variant["offers"])
                break
    if not isinstance(offers, dict):
        return {}
    if "AggregateOffer" in type_names(offers) and offers.get("price") is None:
        merged = dict(offers)
        merged["price"] = offers.get("lowPrice")
        return merged
    spec = first(offers.get("priceSpecification"))
    if offers.get("price") is None and isinstance(spec, dict):
        merged = dict(offers)
        merged["price"] = spec.get("price")
        merged.setdefault("priceCurrency", spec.get("priceCurrency"))
        return merged
    return offers


def _product(obj: dict[str, Any], url: str | None) -> dict[str, Any]:
    offer = _offer(obj)
    return {
        "name": clean_text(obj.get("name")),
        "price": price_forms(offer.get("price")),
        "priceCurrency": currency_forms(offer.get("priceCurrency")),
        "availability": enum_name(offer.get("availability")),
        "brand": scalar(names(obj.get("brand"))[:1]),
        "sku": clean_text(obj.get("sku")),
        "image": image_url(obj.get("image"), url),
    }


_RANGE_SEPARATORS = (" - ", "-", " to ", "\u2013", " \u2013 ")


def _salary(value: Any) -> str | list[str] | None:
    amount = first(value)
    if not isinstance(amount, dict):
        return clean_text(amount)
    inner = amount.get("value", amount)
    if not isinstance(inner, dict):
        return price_forms(inner)
    low, high = clean_text(inner.get("minValue")), clean_text(inner.get("maxValue"))
    single = clean_text(inner.get("value"))
    if low and high and low != high:
        return scalar(_range_forms(low, high))
    return price_forms(single or low or high)


def _range_forms(low: str, high: str) -> list[str]:
    """``low - high`` with the usual separators (hyphen, en dash, "to"); each bound as given, in
    plain form and, when either bound has a fraction, at two decimals (``14 - 14.5`` also matches
    ``14–14.50``)."""
    lows, highs = [low], [high]
    try:
        lo, hi = float(low.replace(",", "")), float(high.replace(",", ""))
    except ValueError:
        pass
    else:
        lows.append(_number(lo))
        highs.append(_number(hi))
        if not (lo.is_integer() and hi.is_integer()):
            lows.append(f"{lo:.2f}")
            highs.append(f"{hi:.2f}")
    return [
        f"{a}{sep}{b}"
        for a in dict.fromkeys(lows)
        for b in dict.fromkeys(highs)
        for sep in _RANGE_SEPARATORS
    ]


def _job(obj: dict[str, Any], url: str | None) -> dict[str, Any]:
    loc = first(obj.get("jobLocation"))
    location = address_forms(loc.get("address") if isinstance(loc, dict) else loc)
    if location is None and "TELECOMMUTE" in str(obj.get("jobLocationType", "")).upper():
        location = "Remote"
    return {
        "title": clean_text(obj.get("title")) or clean_text(obj.get("name")),
        "hiringOrganization": scalar(names(obj.get("hiringOrganization"))[:1]),
        "location": location,
        "datePosted": date_forms(obj.get("datePosted")),
        "employmentType": _employment(obj.get("employmentType")),
        "salary": _salary(obj.get("baseSalary") or obj.get("estimatedSalary")),
    }


def _employment(value: Any) -> str | list[str] | None:
    values = value if isinstance(value, list) else [value]
    texts = [clean_text(v) for v in values]
    present = [t for t in texts if t]
    if not present:
        return None
    if len(present) == 1:
        return enum_name(present[0])
    return ", ".join(present)


def _recipe(obj: dict[str, Any], url: str | None) -> dict[str, Any]:
    raw = obj.get("recipeIngredient") or obj.get("ingredients") or []
    ingredients = [
        t for t in (clean_text(i) for i in (raw if isinstance(raw, list) else [raw])) if t
    ]
    yields = obj.get("recipeYield")
    return {
        "name": clean_text(obj.get("name")),
        "prepTime": duration_forms(obj.get("prepTime")),
        "cookTime": duration_forms(obj.get("cookTime")),
        "recipeYield": scalar(
            clean_text(y) for y in (yields if isinstance(yields, list) else [yields])
        ),
        "ingredients": ingredients,
        "author": scalar(names(obj.get("author"))),
    }


def _event(obj: dict[str, Any], url: str | None) -> dict[str, Any]:
    place = first(obj.get("location"))
    location = address = None
    if isinstance(place, dict):
        location = clean_text(place.get("name"))
        address = address_forms(place.get("address"))
    elif isinstance(place, str):
        location = clean_text(place)
    return {
        "name": clean_text(obj.get("name")),
        "startDate": date_forms(obj.get("startDate")),
        "endDate": date_forms(obj.get("endDate")),
        "location": location,
        "address": address,
        "organizer": scalar(names(obj.get("organizer"))[:1]),
        "performer": scalar(names(obj.get("performer"))),
    }


def _news(obj: dict[str, Any], url: str | None) -> dict[str, Any]:
    return {
        "headline": clean_text(obj.get("headline")) or clean_text(obj.get("name")),
        "author": scalar(names(obj.get("author"))),
        "datePublished": date_forms(obj.get("datePublished")),
        "articleSection": clean_text(obj.get("articleSection")),
        "description": clean_text(obj.get("description")),
        "image": image_url(obj.get("image"), url),
    }


def _book(obj: dict[str, Any], url: str | None) -> dict[str, Any]:
    edition = first(obj.get("workExample"))
    ed = edition if isinstance(edition, dict) else {}
    return {
        "name": clean_text(obj.get("name")),
        "author": scalar(names(obj.get("author"))),
        "isbn": clean_text(obj.get("isbn") or ed.get("isbn")),
        "publisher": scalar(names(obj.get("publisher") or ed.get("publisher"))[:1]),
        "datePublished": date_forms(obj.get("datePublished") or ed.get("datePublished")),
        "numberOfPages": clean_text(obj.get("numberOfPages") or ed.get("numberOfPages")),
    }


_FLATTENERS = {
    "product": _product,
    "job": _job,
    "recipe": _recipe,
    "event": _event,
    "news": _news,
    "book": _book,
}


def flatten(obj: dict[str, Any], vertical: str, url: str | None = None) -> dict[str, Any]:
    """Flatten one JSON-LD object of the vertical's target type to its label fields."""
    values = _FLATTENERS[vertical](obj, url)
    return {name: values.get(name) for name in VERTICALS[vertical].fields}


def _same_page(obj: dict[str, Any], url: str) -> bool:
    def norm(u: str) -> str:
        return re.sub(r"^https?://(www\.)?", "", u.split("#")[0]).rstrip("/").lower()

    target = norm(url)
    for key in ("url", "@id", "mainEntityOfPage"):
        value = obj.get(key)
        if isinstance(value, dict):
            value = value.get("@id") or value.get("url")
        if isinstance(value, str) and norm(urljoin(url, value)) == target:
            return True
    return False


def main_object(objs: list[dict[str, Any]], url: str | None) -> dict[str, Any] | None:
    """The page's own object among the target-type JSON-LD objects, or ``None`` if ambiguous.

    One object: that one. Several: the one whose ``url``/``@id`` is the page URL, else the first
    when they all share one name (variants of one item); otherwise the page is a listing and gets
    no label.
    """
    if not objs:
        return None
    if len(objs) == 1:
        return objs[0]
    if url:
        own = [o for o in objs if _same_page(o, url)]
        if len(own) == 1:
            return own[0]
    labels = {clean_text(o.get("name") or o.get("title") or o.get("headline")) for o in objs}
    return objs[0] if len(labels) == 1 else None


@dataclass
class PageLabel:
    """JSON-LD findings for one page."""

    n_objects: int
    """Target-type JSON-LD objects found on the page."""
    type: str | None
    labels: dict[str, Any] | None
    """Flattened fields of the page's own object (``None``: no object, or ambiguous)."""
    display_forms: list[str] = field(default_factory=list)
    """Identifier fields that got the page's displayed spelling as an alternative."""


def page_labels(root: lxml_html.HtmlElement, vertical: str, url: str | None) -> PageLabel:
    """Labels for a parsed (uncleaned) page. Does not modify ``root``."""
    objs = find_typed(jsonld_items(root), VERTICALS[vertical].types)
    obj = main_object(objs, url)
    if obj is None:
        return PageLabel(len(objs), None, None)
    kind = sorted(type_names(obj) & VERTICALS[vertical].types)
    return PageLabel(len(objs), kind[0] if kind else None, flatten(obj, vertical, url))


_ID_SEPARATOR = "[-/ .]?"


def identifier_display_form(value: str, visible: str) -> str | None:
    """How the page shows an identifier that the JSON-LD gives in another spelling.

    Argos's ``sku`` ``3342679`` is displayed as ``334/2679``; ISBNs come with and without
    hyphens. Returns the displayed spelling (the value's characters with ``-``, ``/``, space or
    ``.`` between them, as a whole token of the visible text), or ``None`` when the value itself
    is shown, is shorter than 4 characters, or no spelling of it is found.
    """
    compact = re.sub(r"[-/ .]", "", value)
    if len(compact) < 4 or not compact.isalnum():
        return None
    edge_l, edge_r = r"(?<![0-9A-Za-z])", r"(?![0-9A-Za-z])"
    if re.search(edge_l + re.escape(value) + edge_r, visible, re.IGNORECASE):
        return None
    pattern = _ID_SEPARATOR.join(re.escape(c) for c in compact)
    match = re.search(edge_l + pattern + edge_r, visible, re.IGNORECASE)
    return match.group(0) if match else None


def add_display_forms(labels: dict[str, Any], vertical: str, visible: str) -> list[str]:
    """Add the displayed spelling of each identifier field as an alternative (in place).

    Returns the fields that got one.
    """
    changed: list[str] = []
    for name in sorted(VERTICALS[vertical].identifiers):
        value = labels.get(name)
        alternatives = value if isinstance(value, list) else [value]
        if not alternatives or not isinstance(alternatives[0], str):
            continue
        shown = identifier_display_form(alternatives[0], visible)
        if shown is not None and shown not in alternatives:
            labels[name] = [*alternatives, shown]
            changed.append(name)
    return changed


def label_page(body: bytes, content_type: str | None, vertical: str, url: str) -> PageLabel:
    """Labels of one captured page: :func:`page_labels` on the original page, then the
    identifier display forms looked up in its script-stripped text."""
    root = parse_page(body, content_type)
    found = page_labels(root, vertical, url)
    if found.labels is not None and VERTICALS[vertical].identifiers:
        found.display_forms = add_display_forms(found.labels, vertical, visible_text(root))
    return found


# --------------------------------------------------------------------------- selection checks


@dataclass
class RobotsRules:
    """Allow/disallow rules of one robots.txt group (Google-style ``*`` and ``$`` patterns)."""

    rules: list[tuple[bool, str]]

    def allowed(self, path: str) -> bool:
        """Longest matching rule wins; ``Allow`` wins ties; no match means allowed."""
        best_len, verdict = -1, True
        for allow, pattern in self.rules:
            if not pattern:
                continue
            if _robots_match(pattern, path):
                length = len(pattern)
                if length > best_len or (length == best_len and allow):
                    best_len, verdict = length, allow
        return verdict


def _robots_match(pattern: str, path: str) -> bool:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = "".join(".*" if ch == "*" else re.escape(ch) for ch in body)
    return re.match(regex + ("$" if anchored else ""), path) is not None


def parse_robots(text: str, agent: str) -> RobotsRules:
    """The rules that apply to ``agent``: the group naming it (case-insensitive product-token
    match), else the ``*`` group, else none (everything allowed)."""
    groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []
    in_rules = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if in_rules:
                groups.append((agents, rules))
                agents, rules, in_rules = [], [], False
            agents.append(value.lower())
        elif key in ("allow", "disallow"):
            in_rules = True
            rules.append((key == "allow", value))
    if agents:
        groups.append((agents, rules))
    token = agent.lower()
    named = [r for a, r in groups if any(x != "*" and x in token for x in a)]
    if named:
        return RobotsRules([rule for group in named for rule in group])
    star = [r for a, r in groups if "*" in a]
    return RobotsRules([rule for group in star for rule in group])


def has_password_input(root: lxml_html.HtmlElement) -> bool:
    """Whether the page shows a password field (login wall heuristic)."""
    return bool(root.xpath("//input[translate(@type, 'PASWORD', 'pasword')='password']"))


# --------------------------------------------------------------------------- visible text / noise


def parse_page(body: bytes, content_type: str | None = None) -> lxml_html.HtmlElement:
    return parse_html(decode_body(body, content_type))


def visible_text(root: lxml_html.HtmlElement) -> str:
    """Text of the script-stripped page (cleans ``root`` in place: call after label extraction)."""
    cleaned = clean_tree(root)
    return " ".join(" ".join(cleaned.itertext()).split())


def attribute_values(root: lxml_html.HtmlElement) -> list[str]:
    """``src``/``href``/``srcset``/``content``/``data-src`` values of a cleaned page."""
    out: list[str] = []
    for el in root.iter():
        attrib = getattr(el, "attrib", None)
        if not attrib:
            continue
        for key in ("src", "href", "srcset", "content", "data-src", "data-srcset"):
            value = attrib.get(key)
            if value:
                out.append(value)
    return out


_CURRENCY = re.compile(r"[$€£¥₹₩₽¢]")


def normalize_for_match(text: str) -> str:
    """Case, whitespace, currency symbols, thousands separators and trailing punctuation."""
    text = html_lib.unescape(text).lower()
    text = _CURRENCY.sub("", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    return text.strip(" .,;:!?")


def _url_tail(url: str) -> str:
    """The file part of a URL without query (image CDNs rewrite hosts and size params)."""
    path = url.split("?")[0].split("#")[0].rstrip("/")
    return path.rsplit("/", 1)[-1].lower()


def label_agrees(
    value: Any,
    text_norm: str,
    attrs: list[str] | None = None,
    text_raw: str | None = None,
) -> bool | None:
    """Whether a scalar label (a string or a list of alternatives) appears on the page.

    ``None`` when the label is empty. Text must occur in the normalized visible text; URL labels
    are looked up by file name in the page's ``src``/``href``/``content`` attributes when
    ``attrs`` is given; an alternative that normalizes to nothing (a currency symbol) is looked
    up in ``text_raw``, the un-normalized visible text, when given. Any alternative agreeing is
    enough.
    """
    if value is None or value == "" or value == []:
        return None
    alternatives = value if isinstance(value, list) else [value]
    return any(_one_agrees(str(v), text_norm, attrs, text_raw) for v in alternatives)


def _one_agrees(
    value: str, text_norm: str, attrs: list[str] | None, text_raw: str | None = None
) -> bool:
    if value.startswith(("http://", "https://", "//")) and attrs is not None:
        tail = _url_tail(value)
        return bool(tail) and any(tail in a.lower() for a in attrs)
    needle = normalize_for_match(value)
    if not needle and text_raw is not None and value.strip():
        return value.strip() in text_raw
    return bool(needle) and needle in text_norm


def list_agreement(items: list[str], text_norm: str) -> float | None:
    """Share of list items found in the visible text (``None`` for an empty list)."""
    if not items:
        return None
    return sum(_one_agrees(i, text_norm, None) for i in items) / len(items)


# --------------------------------------------------------------------------- adapter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = _repo_root() / path
    return candidate if candidate.exists() else path


def iter_labels(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row: dict[str, Any] = json.loads(line)
                yield row


@dataclass
class CCSilverDataset:
    """The selected domains (``domains.yaml``) with their WARC captures and JSON-LD labels."""

    data_root: Path = DEFAULT_DATA_ROOT
    dataset_dir: Path = DATASET_SUBDIR

    def __post_init__(self) -> None:
        self.data_root = _resolve(Path(self.data_root))
        self.dataset_dir = _resolve(Path(self.dataset_dir))

    def domains(self) -> list[dict[str, Any]]:
        path = self.dataset_dir / "domains.yaml"
        if not path.exists():
            raise DatasetUnavailable(f"{path} not found: run select.py and download.py first")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        selected: list[dict[str, Any]] = data.get("selected") or []
        return selected

    def fields_path(self, vertical: str) -> Path:
        return self.dataset_dir / "fields" / f"{vertical}.yaml"

    def site(self, entry: dict[str, Any]) -> EvalSite:
        domain = str(entry["domain"])
        vertical = str(entry["vertical"])
        site_dir = self.data_root / domain
        warcs = sorted(site_dir.glob("*.warc.gz"))
        if not warcs:
            raise DatasetUnavailable(f"no WARC files under {site_dir}: run download.py")
        labels_path = site_dir / LABELS_FILE
        if not labels_path.exists():
            raise DatasetUnavailable(f"{labels_path} not found: run labels.py")
        labels: dict[str, dict[str, Any]] = {}
        for row in iter_labels(labels_path):
            # A property the JSON-LD omits is not evidence that the page lacks it: unscored.
            labels[str(row["url"])] = {k: v for k, v in row["labels"].items() if not is_empty(v)}
        meta = {k: v for k, v in entry.items() if k not in ("domain", "vertical")}
        meta["labeled_pages"] = len(labels)
        return EvalSite(
            name=domain,
            vertical=vertical,
            input_paths=warcs,
            input_kind="warc",
            fields_schema_path=self.fields_path(vertical),
            labels=labels,
            url_to_key={url: url for url in labels},
            field_kinds=VERTICALS[vertical].kinds(),
            meta=meta,
        )

    def sites(self, names: Iterable[str] | None = None) -> list[EvalSite]:
        wanted = set(names) if names is not None else None
        return [
            self.site(entry)
            for entry in self.domains()
            if wanted is None or entry["domain"] in wanted
        ]


def load_sites(selection: Selection, work_dir: Path) -> list[EvalSite]:
    """Evaluation adapter: one :class:`EvalSite` per selected Common Crawl domain.

    Options (``dataset_options``): ``data_root`` (default ``data/cc``), ``dataset_dir``
    (default ``eval/datasets/commoncrawl_silver``). ``split`` is ignored (the set has no
    tuning split; it is evaluated once). Only the WARC files and labels of the sites that the
    selection keeps are read.
    """
    dataset = CCSilverDataset(
        data_root=Path(selection.options.get("data_root", DEFAULT_DATA_ROOT)),
        dataset_dir=Path(selection.options.get("dataset_dir", DATASET_SUBDIR)),
    )
    entries = dataset.domains()
    stubs = [
        EvalSite(str(e["domain"]), str(e["vertical"]), [], "warc", None, {}, {}) for e in entries
    ]
    keep = {s.name for s in select_sites(stubs, selection)}
    return select_sites(dataset.sites(keep), selection)
