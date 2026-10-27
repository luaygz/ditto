"""Common Crawl silver set: JSON-LD labels, noise-check matching, robots rules, the evaluation
adapter, and the (network-free parts of the) dataset scripts."""

from __future__ import annotations

import gzip
import importlib.util
import io
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from ditto.evaluation.datasets.base import DatasetUnavailable, Selection
from ditto.evaluation.datasets.cc_silver import (
    VERTICALS,
    CCSilverDataset,
    attribute_values,
    clean_text,
    currency_forms,
    date_forms,
    decode_body,
    duration_forms,
    enum_name,
    has_password_input,
    identifier_display_form,
    label_agrees,
    label_page,
    list_agreement,
    load_sites,
    main_object,
    normalize_for_match,
    page_labels,
    parse_page,
    parse_robots,
    price_forms,
    visible_text,
)
from ditto.fingerprint.dom import parse_html
from ditto.schema import FieldsSchema

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "eval" / "datasets" / "commoncrawl_silver"


def ld(*objs: Any) -> str:
    return "".join(f'<script type="application/ld+json">{json.dumps(o)}</script>' for o in objs)


def page(head: str, body: str) -> str:
    return f"<!doctype html><html><head><title>t</title>{head}</head><body>{body}</body></html>"


PRODUCT_PAGE = page(
    ld(
        {
            "@context": "https://schema.org",
            "@graph": [
                {"@type": "Organization", "name": "Shop"},
                {"@type": "BreadcrumbList", "itemListElement": []},
                {
                    "@type": "http://schema.org/Product",
                    "name": "Trail Runner 2 &amp; Co",
                    "sku": 12345,
                    "brand": {"@type": "Brand", "name": "Acme"},
                    "image": ["/img/trail-runner.jpg", "/img/other.jpg"],
                    "offers": {
                        "@type": "Offer",
                        "price": 1299.5,
                        "priceCurrency": "USD",
                        "availability": "https://schema.org/InStock",
                    },
                },
            ],
        }
    )
    + '<script type="application/ld+json">{"broken": </script>',
    '<h1>Trail Runner 2 &amp; Co</h1><p class="price">$1,299.50</p><p>In stock</p>'
    '<img src="https://cdn.example.com/img/trail-runner.jpg?w=600"><p>SKU 12345 by Acme</p>'
    "<script>var hidden = 'not visible';</script>",
)


def labels_of(html: str, vertical: str, url: str = "https://shop.example.com/p/1") -> Any:
    return page_labels(parse_html(html), vertical, url)


# --------------------------------------------------------------------------- labels


def test_product_labels_from_graph_skip_broken_block() -> None:
    found = labels_of(PRODUCT_PAGE, "product")
    assert found.n_objects == 1
    assert found.type == "Product"
    assert found.labels == {
        "name": "Trail Runner 2 & Co",
        "price": ["1299.5", "1299.50", "1,299.50", "1299,50", "1.299,50"],
        "priceCurrency": ["USD", "$"],
        "availability": ["InStock", "In Stock", "In-Stock"],
        "brand": "Acme",
        "sku": "12345",
        "image": "https://shop.example.com/img/trail-runner.jpg",
    }


def test_product_aggregate_offer_and_product_group_variants() -> None:
    agg = page(
        ld(
            {
                "@type": "Product",
                "name": "Desk",
                "offers": {"@type": "AggregateOffer", "lowPrice": "199", "highPrice": "299"},
            }
        ),
        "",
    )
    assert labels_of(agg, "product").labels["price"] == ["199", "199.00", "199,00"]
    group = page(
        ld(
            {
                "@type": "ProductGroup",
                "name": "Tee",
                "brand": "Basic",
                "hasVariant": [
                    {
                        "@type": "Product",
                        "name": "Tee S",
                        "offers": {"price": 20, "priceCurrency": "EUR"},
                    },
                    {"@type": "Product", "name": "Tee M", "offers": {"price": 22}},
                ],
            }
        ),
        "",
    )
    found = labels_of(group, "product")
    assert found.n_objects == 1  # variants are not separate hits
    assert found.labels["price"] == ["20", "20.00", "20,00"]
    assert found.labels["priceCurrency"] == ["EUR", "€"]
    assert found.labels["brand"] == "Basic"


def test_price_specification_fallback() -> None:
    html = page(
        ld(
            {
                "@type": "Product",
                "name": "Cam",
                "offers": {"priceSpecification": {"price": "49.99", "priceCurrency": "GBP"}},
            }
        ),
        "",
    )
    labels = labels_of(html, "product").labels
    assert labels["price"] == ["49.99", "49,99"]
    assert labels["priceCurrency"] == ["GBP", "£"]


def test_listing_page_is_ambiguous_unless_one_object_is_the_page() -> None:
    items = [{"@type": "Product", "name": f"Item {i}", "url": f"/p/{i}"} for i in range(3)]
    html = page(ld({"@type": "ItemList", "itemListElement": items}), "")
    found = labels_of(html, "product", "https://shop.example.com/category")
    assert found.n_objects == 3
    assert found.labels is None
    own = labels_of(html, "product", "https://www.shop.example.com/p/2/")
    assert own.labels is not None and own.labels["name"] == "Item 2"
    same_name = [{"@type": "Product", "name": "Same"}, {"@type": "Product", "name": "Same"}]
    assert main_object(same_name, None) is same_name[0]
    assert main_object([], None) is None


def test_job_labels() -> None:
    html = page(
        ld(
            {
                "@type": "JobPosting",
                "title": "Senior Nurse",
                "hiringOrganization": {"@type": "Organization", "name": "City Hospital"},
                "jobLocation": [
                    {
                        "@type": "Place",
                        "address": {
                            "@type": "PostalAddress",
                            "addressLocality": "Leeds",
                            "addressRegion": "West Yorkshire",
                            "addressCountry": {"@type": "Country", "name": "GB"},
                        },
                    }
                ],
                "datePosted": "2026-10-01T09:00:00Z",
                "employmentType": "FULL_TIME",
                "baseSalary": {
                    "@type": "MonetaryAmount",
                    "currency": "GBP",
                    "value": {"@type": "QuantitativeValue", "minValue": 30000, "maxValue": 35000.0},
                },
            }
        ),
        "",
    )
    labels = labels_of(html, "job").labels
    assert labels["title"] == "Senior Nurse"
    assert labels["hiringOrganization"] == "City Hospital"
    assert labels["location"] == ["Leeds, West Yorkshire, GB", "Leeds, West Yorkshire", "Leeds"]
    assert labels["datePosted"][:3] == ["2026-10-01T09:00:00Z", "2026-10-01", "1 October 2026"]
    assert labels["employmentType"] == ["FULL_TIME", "FULL TIME", "FULL-TIME"]
    assert labels["salary"] == [
        "30000 - 35000",
        "30000-35000",
        "30000 to 35000",
        "30000–35000",
        "30000 – 35000",
    ]


def test_job_remote_multi_type_and_single_salary() -> None:
    html = page(
        ld(
            {
                "@type": "JobPosting",
                "name": "Engineer",
                "jobLocationType": "TELECOMMUTE",
                "employmentType": ["FULL_TIME", "CONTRACTOR"],
                "baseSalary": {"value": {"value": 55}},
            }
        ),
        "",
    )
    labels = labels_of(html, "job").labels
    assert labels["title"] == "Engineer"
    assert labels["location"] == "Remote"
    assert labels["employmentType"] == "FULL_TIME, CONTRACTOR"
    assert labels["salary"] == ["55", "55.00", "55,00"]
    assert labels["hiringOrganization"] is None


def test_recipe_labels() -> None:
    html = page(
        ld(
            {
                "@context": "https://schema.org/",
                "@type": ["Recipe", "NewsArticle"],
                "name": "Chana Masala",
                "prepTime": "PT15M",
                "cookTime": "PT1H30M",
                "recipeYield": ["4", "4 servings"],
                "recipeIngredient": ["2 cups chickpeas", "1 tbsp garam masala &amp; salt", ""],
                "author": [{"@type": "Person", "name": "A. Cook"}, {"name": "B. Chef"}],
            }
        ),
        "",
    )
    labels = labels_of(html, "recipe").labels
    assert labels["name"] == "Chana Masala"
    assert "15 mins" in labels["prepTime"] and labels["prepTime"][0] == "PT15M"
    assert "1 hr 30 mins" in labels["cookTime"] and "90 minutes" in labels["cookTime"]
    assert labels["recipeYield"] == ["4", "4 servings"]
    assert labels["ingredients"] == ["2 cups chickpeas", "1 tbsp garam masala & salt"]
    assert labels["author"] == ["A. Cook", "B. Chef"]
    assert VERTICALS["recipe"].kinds() == {"ingredients": "list"}


def test_event_news_book_labels() -> None:
    event = page(
        ld(
            {
                "@type": "MusicEvent",
                "name": "Band Live",
                "startDate": "2026-11-02T20:00",
                "location": {
                    "@type": "MusicVenue",
                    "name": "The Hall",
                    "address": {"streetAddress": "1 Main St", "addressLocality": "Bristol"},
                },
                "performer": [{"name": "Band"}, {"name": "Support"}],
                "organizer": "Promo Ltd",
            }
        ),
        "",
    )
    labels = labels_of(event, "event").labels
    assert labels["name"] == "Band Live"
    assert labels["startDate"][:2] == ["2026-11-02T20:00", "2026-11-02"]
    assert "Nov 2, 2026" in labels["startDate"]
    assert labels["endDate"] is None
    assert labels["location"] == "The Hall"
    assert labels["address"] == ["1 Main St, Bristol", "Bristol"]
    assert labels["performer"] == ["Band", "Support"]
    assert labels["organizer"] == "Promo Ltd"
    string_place = page(ld({"@type": "Event", "name": "Talk", "location": "Online"}), "")
    assert labels_of(string_place, "event").labels["location"] == "Online"

    news = page(
        ld(
            {
                "@type": "NewsArticle",
                "headline": "Rates <b>rise</b>",
                "author": {"name": "Reporter"},
                "datePublished": "2026-10-20",
                "articleSection": ["Economy"],
                "image": {"@type": "ImageObject", "url": "https://img.example.com/a.jpg"},
            }
        ),
        "",
    )
    labels = labels_of(news, "news").labels
    assert labels["headline"] == "Rates rise"
    assert labels["articleSection"] == "Economy"
    assert labels["image"] == "https://img.example.com/a.jpg"

    book = page(
        ld(
            {
                "@type": "Book",
                "name": "Novel",
                "author": "Writer",
                "workExample": {"isbn": "9780000000001", "numberOfPages": 320, "publisher": "Pub"},
            }
        ),
        "",
    )
    labels = labels_of(book, "book").labels
    assert labels["isbn"] == "9780000000001"
    assert labels["numberOfPages"] == "320"
    assert labels["publisher"] == "Pub"


def test_value_helpers() -> None:
    assert duration_forms("PT2H") == [
        "PT2H",
        "2 hrs",
        "2 hours",
        "2h",
        "2 hr",
        "120 mins",
        "120 minutes",
    ]
    assert duration_forms("P0DT0H1M")[:2] == ["P0DT0H1M", "1 mins"]
    assert duration_forms("about 10 minutes") == "about 10 minutes"
    assert duration_forms(None) is None
    assert price_forms("free") == "free"
    assert price_forms(1000) == ["1000", "1000.00", "1,000.00", "1000,00", "1.000,00", "1,000"]
    assert enum_name("http://schema.org/OutOfStock") == [
        "OutOfStock",
        "Out Of Stock",
        "Out-Of-Stock",
    ]
    assert enum_name(None) is None
    assert date_forms("October 3") == "October 3"
    assert clean_text({"@value": " a  b "}) == "a b"
    assert clean_text(True) is None
    assert clean_text([None, "", "x"]) == "x"
    assert clean_text({"other": 1}) is None


# --------------------------------------------------------------------------- noise check helpers


def test_visible_text_and_label_agreement() -> None:
    root = parse_html(PRODUCT_PAGE)
    labels = page_labels(root, "product", "https://shop.example.com/p/1").labels
    assert labels is not None
    text = visible_text(root)
    assert "not visible" not in text and "In stock" in text
    norm = normalize_for_match(text)
    attrs = attribute_values(root)
    assert label_agrees(labels["name"], norm) is True
    assert label_agrees(labels["price"], norm) is True  # "$1,299.50" ~ "1299.50"
    assert label_agrees(labels["availability"], norm) is True  # "In Stock" ~ "In stock"
    assert label_agrees(labels["image"], norm, attrs) is True  # matched by file name
    assert label_agrees("https://x.example.com/missing.png", norm, attrs) is False
    assert label_agrees("Nike", norm) is False
    assert label_agrees(None, norm) is None
    assert list_agreement(["trail runner", "absent item"], norm) == 0.5
    assert list_agreement([], norm) is None


def test_decode_and_password_heuristic() -> None:
    body = '<html><head><meta charset="windows-1252"></head><body>caf\xe9</body></html>'
    raw = body.encode("cp1252")
    assert "café" in decode_body(raw, None)
    assert "café" in decode_body(raw, "text/html; charset=windows-1252")
    login = parse_page(b'<html><body><form><input TYPE="Password"></form></body></html>')
    assert has_password_input(login)
    assert not has_password_input(parse_html(PRODUCT_PAGE))


def test_robots_rules() -> None:
    robots = """
User-agent: *
Disallow: /search
Disallow: /*?sort=
Allow: /search/help$

User-agent: CCBot
User-agent: GPTBot
Disallow: /private/

User-agent: BadBot
Disallow: /
"""
    star = parse_robots(robots, "ditto-benchmark")
    assert star.allowed("/product/1")
    assert not star.allowed("/search?q=x")
    assert not star.allowed("/list?sort=price")
    assert star.allowed("/search/help")
    assert not star.allowed("/search/help/more")
    cc = parse_robots(robots, "CCBot")
    assert cc.allowed("/search") and not cc.allowed("/private/x")
    assert parse_robots("", "CCBot").allowed("/anything")
    assert not parse_robots("User-agent: *\nDisallow: /\n", "CCBot").allowed("/x")


# --------------------------------------------------------------------------- adapter


def warc_bytes(pages: list[tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    writer = WARCWriter(buf, gzip=True)
    for url, html in pages:
        headers = StatusAndHeaders(
            "200 OK", [("Content-Type", "text/html; charset=utf-8")], protocol="HTTP/1.1"
        )
        writer.write_record(
            writer.create_warc_record(
                url, "response", payload=io.BytesIO(html.encode()), http_headers=headers
            )
        )
    return buf.getvalue()


def make_dataset(tmp_path: Path, *, warc: bool = True, labels: bool = True) -> Path:
    ds = tmp_path / "ds"
    (ds / "fields").mkdir(parents=True)
    for vertical in ("recipe", "product"):
        (ds / "fields" / f"{vertical}.yaml").write_text(
            (SCRIPTS / "fields" / f"{vertical}.yaml").read_text()
        )
    doc = {
        "crawls": ["CC-MAIN-2026-43", "CC-MAIN-2026-39"],
        "selected": [
            {"domain": "recipes.example", "vertical": "recipe", "captures": {"downloaded": 1}},
            {"domain": "shop.example", "vertical": "product"},
        ],
    }
    (ds / "domains.yaml").write_text(yaml.safe_dump(doc))
    data = tmp_path / "data"
    for domain in ("recipes.example", "shop.example"):
        site = data / domain
        site.mkdir(parents=True)
        if warc:
            (site / "captures-00000.warc.gz").write_bytes(
                warc_bytes([(f"https://{domain}/a", "<html><body>x</body></html>")])
            )
        if labels:
            row = {
                "url": f"https://{domain}/a",
                "type": "Recipe",
                "n_objects": 1,
                "labels": {"name": "A", "ingredients": ["x"], "author": None, "prepTime": ""},
            }
            (site / "labels.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path


def test_load_sites(tmp_path: Path) -> None:
    root = make_dataset(tmp_path)
    opts = {"data_root": str(root / "data"), "dataset_dir": str(root / "ds")}
    sites = load_sites(Selection(options=opts), tmp_path / "work")
    assert [s.name for s in sites] == ["shop.example", "recipes.example"]  # by (vertical, name)
    recipe = sites[1]
    assert recipe.input_kind == "warc"
    assert [p.name for p in recipe.input_paths] == ["captures-00000.warc.gz"]
    assert recipe.labels == {"https://recipes.example/a": {"name": "A", "ingredients": ["x"]}}
    assert recipe.key_for_url("https://recipes.example/a#frag") == "https://recipes.example/a"
    assert recipe.kind("ingredients") == "list" and recipe.kind("name") == "scalar"
    assert recipe.fields_schema_path == root / "ds" / "fields" / "recipe.yaml"
    assert recipe.meta["labeled_pages"] == 1 and recipe.meta["captures"] == {"downloaded": 1}
    only = load_sites(Selection(verticals=("recipe",), options=opts), tmp_path / "work")
    assert [s.name for s in only] == ["recipes.example"]
    assert (
        CCSilverDataset(root / "data", root / "ds").sites(["shop.example"])[0].vertical == "product"
    )


@pytest.mark.parametrize(("warc", "labels"), [(False, True), (True, False)])
def test_load_sites_unavailable(tmp_path: Path, warc: bool, labels: bool) -> None:
    root = make_dataset(tmp_path, warc=warc, labels=labels)
    opts = {"data_root": str(root / "data"), "dataset_dir": str(root / "ds")}
    with pytest.raises(DatasetUnavailable):
        load_sites(Selection(options=opts), tmp_path)


def test_load_sites_without_domains_yaml(tmp_path: Path) -> None:
    opts = {"data_root": str(tmp_path), "dataset_dir": str(tmp_path / "nope")}
    with pytest.raises(DatasetUnavailable):
        load_sites(Selection(options=opts), tmp_path)


def test_committed_fields_match_verticals() -> None:
    for vertical, spec in VERTICALS.items():
        path = SCRIPTS / "fields" / f"{vertical}.yaml"
        if not path.exists():
            continue
        schema = FieldsSchema.load(path)
        assert tuple(schema.names()) == spec.fields
        for name in spec.list_fields:
            field = schema.get(name)
            assert field is not None and field.type == "list"


# --------------------------------------------------------------------------- dataset scripts


def script(name: str) -> ModuleType:
    """Import a dataset script by path (``select.py`` would clash with the stdlib module)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"cc_silver_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


class FakeHttp:
    """Serves byte ranges of local files as if they were on data.commoncrawl.org."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def download(self, filename: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[filename])
        return dest

    def fetch_range(self, filename: str, offset: int, length: int) -> bytes:
        return self.files[filename][offset : offset + length]


def test_cdx_index_prefix_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cc = script("cc")
    monkeypatch.setattr(cc, "INDEX_CACHE", tmp_path / "cache")
    lines = [
        ("com,example)/a", {"url": "https://example.com/a", "status": "200", "mime": "text/html"}),
        (
            "com,example)/b",
            {"url": "https://www.example.com/b", "status": "301", "mime": "text/html"},
        ),
        (
            "com,example)/robots.txt",
            {"url": "https://example.com/robots.txt", "status": "200", "mime": "text/plain"},
        ),
        (
            "com,example,shop)/x",
            {"url": "https://shop.example.com/x", "status": "200", "mime": "text/html"},
        ),
        (
            "com,examplez)/y",
            {"url": "https://examplez.com/y", "status": "200", "mime": "text/html"},
        ),
    ]
    blocks: list[bytes] = []
    for chunk in (lines[:2], lines[2:]):
        text = "".join(f"{k} 20261005000000 {json.dumps(v)}\n" for k, v in chunk)
        blocks.append(gzip.compress(text.encode()))
    cdx = blocks[0] + blocks[1]
    cluster = (
        f"com,example)/a 20261005000000\tcdx-00000.gz\t0\t{len(blocks[0])}\t1\n"
        f"com,example)/robots.txt 20261005000000\tcdx-00000.gz\t{len(blocks[0])}\t{len(blocks[1])}\t2\n"
    )
    root = "cc-index/collections/CC-MAIN-2026-43/indexes"
    http = FakeHttp({f"{root}/cluster.idx": cluster.encode(), f"{root}/cdx-00000.gz": cdx})
    idx = cc.CdxIndex(http, "CC-MAIN-2026-43")
    assert cc.surt_host("www.Example.com") == "com,example)"
    assert [r["url"] for r in idx.domain_rows("www.example.com")] == ["https://example.com/a"]
    assert len(idx.domain_rows("example.com", html_only=False)) == 3
    assert [r["url"] for r in idx.robots_rows("example.com")] == ["https://example.com/robots.txt"]
    assert idx.domain_rows("nothing.org") == []


def test_warc_record_parsing_and_bounded_iteration(tmp_path: Path) -> None:
    cc = script("cc")
    raw = warc_bytes([("https://a.example/1", "<p>one</p>"), ("https://a.example/2", "<p>two</p>")])
    first = warc_bytes([("https://a.example/1", "<p>one</p>")])
    record = cc.parse_record(first)
    assert (
        record.url == "https://a.example/1"
        and record.status == 200
        and record.body == b"<p>one</p>"
    )
    path = tmp_path / "x.warc.gz"
    path.write_bytes(raw + b"\x1f\x8b partial garbage")
    assert [r.url for r in cc.iter_warc_pages(path, limit=len(raw))] == [
        "https://a.example/1",
        "https://a.example/2",
    ]
    assert [r.url for r in cc.iter_warc_pages(path, limit=len(first))] == ["https://a.example/1"]
    with pytest.raises(ValueError):
        cc.parse_record(b"")
    assert (
        cc.crawl_of({"filename": "crawl-data/CC-MAIN-2026-39/segments/x.warc.gz"})
        == "CC-MAIN-2026-39"
    )


def test_download_plan_cap_and_journal_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = script("download")

    class FakeIndex:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def domain_rows(self, domain: str) -> list[dict[str, Any]]:
            return self.rows

    def row(
        i: int, kind: str, ts: str = "20261005000000", digest: str | None = None
    ) -> dict[str, Any]:
        path = f"/recipe/{i}" if kind == "detail" else f"/tag/{i}"
        return {
            "urlkey": f"com,x){path}",
            "url": f"https://x.com{path}",
            "timestamp": ts,
            "digest": digest or f"D{kind}{i}",
            "filename": "f",
            "offset": "0",
            "length": "1",
        }

    new = [row(i, "detail", "20261010000000") for i in range(30)] + [
        row(i, "other") for i in range(10)
    ]
    old = [row(i, "detail") for i in range(30)] + [row(99, "other", digest="Ddetail0")]
    entry = {"domain": "x.com", "detail_pattern": r"^/recipe/\d+$"}
    indexes = {"CC-MAIN-2026-43": FakeIndex(new), "CC-MAIN-2026-39": FakeIndex(old)}
    plan, counts = download.plan_domain(entry, indexes, cap=20)
    assert counts["distinct_urls"] == 41
    assert counts["distinct_pages"] == 40  # /tag/99 duplicates /recipe/0's payload
    assert (counts["planned_detail"], counts["planned_other"]) == (16, 4)  # 20% tail reserve
    assert [p["kind"] for p in plan] == ["detail"] * 16 + ["other"] * 4
    assert all(p["crawl"] == "CC-MAIN-2026-43" for p in plan if p["kind"] == "detail")
    big, big_counts = download.plan_domain(entry, indexes, cap=1000)
    assert big_counts["planned"] == 40
    smaller, _ = download.plan_domain(entry, indexes, cap=10)
    assert {p["url"] for p in smaller} <= {p["url"] for p in plan} <= {p["url"] for p in big}

    site = tmp_path / "site"
    site.mkdir()
    warc = site / "captures-00000.warc.gz"
    warc.write_bytes(b"A" * 10 + b"B" * 5)  # second record torn by a crash
    (site / "captures-00001.warc.gz").write_bytes(b"orphan")
    (site / "journal.jsonl").write_text(
        json.dumps({"i": 0, "file": warc.name, "end": 10, "url": "u0"})
        + "\n"
        + json.dumps({"i": 1, "error": "boom"})
        + "\n"
        + '{"i": 2, "fi'
    )
    journal = download.Journal(site)
    journal.close()
    assert set(journal.done) == {"u0"}
    assert warc.read_bytes() == b"A" * 10
    assert not (site / "captures-00001.warc.gz").exists()


def test_download_domain_writes_warc_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = script("download")
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "ROLL", 2)
    records = [warc_bytes([(f"https://x.com/{i}", f"<p>{i}</p>")]) for i in range(5)]
    files = {"f": b"".join(records)}
    offsets = [sum(len(r) for r in records[:i]) for i in range(5)]
    plan = [
        {
            "url": f"https://x.com/{i}",
            "filename": "f",
            "offset": offsets[i],
            "length": len(records[i]),
            "kind": "detail",
        }
        for i in range(5)
    ]
    plan.append(
        {
            "url": "https://x.com/wrong",
            "filename": "f",
            "offset": 0,
            "length": len(records[0]),
            "kind": "other",
        }
    )
    stats = download.download_domain(FakeHttp(files), "x.com", plan)
    assert stats["fetched_this_run"] == 5 and stats["errors_this_run"] == 1
    site = tmp_path / "x.com"
    assert sorted(p.name for p in site.glob("*.warc.gz")) == [
        "captures-00000.warc.gz",
        "captures-00001.warc.gz",
        "captures-00002.warc.gz",
    ]
    cc = script("cc")
    urls = [r.url for p in sorted(site.glob("*.warc.gz")) for r in cc.iter_warc_pages(p)]
    assert urls == [f"https://x.com/{i}" for i in range(5)]
    again = download.download_domain(FakeHttp(files), "x.com", plan)
    assert again["fetched_this_run"] == 0 and again["errors_this_run"] == 1
    assert download.downloaded("x.com") == {f"https://x.com/{i}" for i in range(5)}

    labels = script("labels")
    ends = labels.journaled_ends(site)
    items = [("x.com", "recipe", str(site / n), e) for n, e in sorted(ends.items())]
    rows = labels.label_file(items)
    assert [r["url"] for r in rows] == [f"https://x.com/{i}" for i in range(5)]
    assert all(r["labels"] is None and r["n_objects"] == 0 for r in rows)
    stats = labels.summarize(rows, "recipe", r"^/\d$")
    assert stats["pages"] == 5 and stats["labeled_pages"] == 0 and stats["detail_pages"] == 5

    noise = script("noise_check")
    spans = noise.record_spans(site)
    name, start, end = spans["https://x.com/3"]
    with (site / name).open("rb") as fh:
        fh.seek(start)
        assert cc.parse_record(fh.read(end - start)).url == "https://x.com/3"


def test_noise_check_page() -> None:
    noise = script("noise_check")
    labels = labels_of(PRODUCT_PAGE, "product").labels
    result = noise.check_page(PRODUCT_PAGE.encode(), "text/html", labels, "product")
    assert result["name"] == {"agree": True}
    assert result["image"] == {"agree": True}
    assert result["brand"] == {"agree": True}
    recipe = noise.check_page(
        b"<html><body><li>2 cups flour</li></body></html>",
        None,
        {"name": "Bread", "ingredients": ["2 cups flour", "1 egg"], "author": None},
        "recipe",
    )
    assert recipe == {"name": {"agree": False}, "ingredients": {"items": 2, "item_share": 0.5}}


def test_select_choose_prefers_two_per_main_vertical() -> None:
    select = script("select")

    def sample(
        domain: str, vertical: str, rate: float, detail: int = 100, ok: bool = True
    ) -> dict[str, Any]:
        return {
            "domain": domain,
            "vertical": vertical,
            "jsonld_rate": rate,
            "passes": ok,
            "distinct_detail_urls": detail,
            "reasons": [] if ok else ["no JSON-LD"],
        }

    samples = {
        s["domain"]: s
        for s in [
            sample("p1", "product", 0.9),
            sample("p2", "product", 0.95, 5),
            sample("p3", "product", 0.7),
            sample("j1", "job", 1.0),
            sample("j2", "job", 0.8),
            sample("r1", "recipe", 0.9),
            sample("e1", "event", 0.9),
            sample("e2", "event", 0.9, 50),
            sample("e3", "event", 0.3, ok=False),
            sample("n1", "news", 0.9),
            sample("b1", "book", 0.9),
            sample("b2", "book", 0.8),
        ]
    }
    counts = {
        "crawls": ["a", "b"],
        "domains": {d: {"passes": True, "distinct_urls": 9000} for d in samples},
    }
    counts["domains"]["small"] = {"passes": False, "distinct_urls": 120}
    counts["domains"]["broken"] = {"error": "FetchError('x')"}
    candidates = (
        *(select.Candidate(d, s["vertical"], ".") for d, s in samples.items()),
        select.Candidate("small", "job", "."),
        select.Candidate("broken", "job", "."),
        select.Candidate("unsampled", "job", "."),
    )
    counts["domains"]["unsampled"] = {"passes": True, "distinct_urls": 6000}
    chosen, rejected = select.choose(counts, samples, candidates)
    assert [s["domain"] for s in chosen] == ["p1", "p2", "j1", "j2", "r1", "e1", "e2", "b1"]
    reasons = {r["domain"]: r["reason"] for r in rejected}
    assert reasons["e3"] == "no JSON-LD"
    assert reasons["small"].startswith("120 distinct")
    assert reasons["broken"].startswith("index query failed")
    assert reasons["unsampled"] == "not sampled"
    assert set(reasons) == {"p3", "e3", "n1", "b2", "small", "broken", "unsampled"}
    assert "ranked below" in reasons["p3"]

    twins = tuple(
        select.Candidate(c.domain, c.vertical, ".", "same platform", operator="op")
        if c.domain in ("j1", "j2")
        else c
        for c in candidates
    )
    chosen, rejected = select.choose(counts, samples, twins)
    assert [s["domain"] for s in chosen] == ["p1", "p2", "j1", "r1", "e1", "e2", "b1", "b2"]
    reasons = {r["domain"]: r["reason"] for r in rejected}
    assert "shares its operator/platform with the selected j1" in reasons["j2"]


# --------------------------------------------------------------------------- polite downloader


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def polite_http(statuses: list[int], body: bytes = b"0123456789") -> tuple[Any, Any, list[float]]:
    """An ``Http`` over a mock transport answering ``statuses`` in turn (206 serves ``body``)."""
    import httpx

    cc = script("cc")
    fake = FakeClock()
    starts: list[float] = []
    answers = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(fake.now)
        status = next(answers)
        if status == 206:
            first, last = request.headers["Range"].removeprefix("bytes=").split("-")
            return httpx.Response(206, content=body[int(first) : int(last) + 1])
        if status == 200:
            return httpx.Response(200, headers={"Content-Length": str(len(body))})
        return httpx.Response(status, content=b"Request blocked")

    http = cc.Http(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=fake.sleep,
        clock=fake.clock,
        max_attempts=2,
        log=lambda msg: None,
    )
    return cc, http, starts


def test_http_waits_out_slow_down_without_spending_attempts() -> None:
    cc, http, starts = polite_http([403, 403, 503, 206, 206, 403, 206])
    assert http.fetch_range("crawl-data/x.warc.gz", 2, 4) == b"2345"  # 3 slow-downs, 2 attempts
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert 45 <= gaps[0] <= 75 and 90 <= gaps[1] <= 150 and 180 <= gaps[2] <= 300
    assert http.fetch_range("crawl-data/x.warc.gz", 0, 2) == b"01"
    assert starts[4] - starts[3] >= cc.DATA_INTERVAL_S  # paced even without a block
    assert http.fetch_range("crawl-data/x.warc.gz", 0, 1) == b"0"
    assert 45 <= starts[6] - starts[5] <= 75  # the success reset the backoff level
    assert http.stats["slow_downs"] == 4 and http.stats["retries"] == 0


def test_http_backoff_is_capped_and_other_errors_still_fail() -> None:
    cc, http, starts = polite_http([403] * 8 + [206, 404])
    assert http.fetch_range("crawl-data/x.warc.gz", 0, 3) == b"012"
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert max(gaps) <= cc.SLOW_DOWN_MAX_S and gaps[-1] >= 0.75 * cc.SLOW_DOWN_MAX_S
    with pytest.raises(cc.FetchError, match="HTTP 404"):
        http.fetch_range("crawl-data/x.warc.gz", 0, 3)
    _, head, _ = polite_http([429, 200])
    assert head.size("crawl-data/x.warc.gz") == 10
    with pytest.raises(ValueError):
        cc.Http(data_concurrency=2)


def test_plan_keeps_downloaded_pages_and_fills_the_rest() -> None:
    download = script("download")

    class FakeIndex:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def domain_rows(self, domain: str) -> list[dict[str, Any]]:
            return self.rows

    rows = [
        {
            "urlkey": f"com,x)/{kind}/{i}",
            "url": f"https://x.com/{kind}/{i}",
            "timestamp": "20261005000000",
            "digest": f"{kind}{i}",
            "filename": "f",
            "offset": "0",
            "length": "1",
        }
        for kind in ("p", "tag")
        for i in range(30)
    ]
    entry = {"domain": "x.com", "detail_pattern": r"^/p/\d+$"}
    indexes = {"CC-MAIN-2026-43": FakeIndex(rows)}
    fresh, _ = download.plan_domain(entry, indexes, cap=10)
    have = {f"https://x.com/p/{i}" for i in range(20)}  # more details than the quota of 8
    plan, counts = download.plan_domain(entry, indexes, cap=10, have=have)
    details = [p["url"] for p in plan if p["kind"] == "detail"]
    assert set(details) == have and counts["planned_detail"] == 20
    assert [p for p in plan if p["kind"] == "other"] == [p for p in fresh if p["kind"] == "other"]
    assert counts["kept_from_earlier_runs"] == 20 and counts["planned"] == 22
    few = {f"https://x.com/p/{i}" for i in range(3)}
    topped, _ = download.plan_domain(entry, indexes, cap=10, have=few)
    topped_details = {p["url"] for p in topped if p["kind"] == "detail"}
    assert few <= topped_details and len(topped_details) == 8


def test_download_all_round_robin_and_retries_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = script("download")
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    records = {
        d: [warc_bytes([(f"https://{d}/{i}", f"<p>{i}</p>")]) for i in range(3)] for d in "ab"
    }
    files = {d: b"".join(recs) for d, recs in records.items()}

    def plan(d: str) -> list[dict[str, Any]]:
        offsets = [sum(len(r) for r in records[d][:i]) for i in range(3)]
        return [
            {"url": f"https://{d}/{i}", "filename": d, "offset": offsets[i], "length": len(r)}
            for i, r in enumerate(records[d])
        ]

    class Flaky(FakeHttp):
        def __init__(self, files: dict[str, bytes], fail: set[tuple[str, int]]) -> None:
            super().__init__(files)
            self.fail = fail
            self.calls: list[tuple[str, int]] = []

        def fetch_range(self, filename: str, offset: int, length: int) -> bytes:
            self.calls.append((filename, offset))
            if (filename, offset) in self.fail:
                raise RuntimeError("transient")
            return super().fetch_range(filename, offset, length)

    plans = {"a": plan("a"), "b": plan("b")}
    first = Flaky(files, {("b", plans["b"][1]["offset"])})
    stats = download.download_all(first, plans)
    assert [c[0] for c in first.calls] == ["a", "b", "a", "b", "a", "b"]  # interleaved
    assert stats["b"]["errors_this_run"] == 1 and stats["a"]["fetched_this_run"] == 3
    second = Flaky(files, set())
    again = download.download_all(second, plans)
    assert second.calls == [("b", plans["b"][1]["offset"])]  # only the failed capture again
    assert again["b"]["fetched_this_run"] == 1
    assert download.downloaded("b") == {f"https://b/{i}" for i in range(3)}
    assert download.policy(2500)["cap_per_domain"] == 2500


# --------------------------------------------------------------------------- label alternatives


def test_display_forms_dates_currency_and_ranges() -> None:
    assert identifier_display_form("3342679", "Curtains 334/2679 £28.50") == "334/2679"
    assert identifier_display_form("3342679", "Cat. no. 3342679") is None  # shown as is
    assert identifier_display_form("12", "1/2") is None  # too short to be safe
    assert identifier_display_form("3342679", "13342679 elsewhere") is None
    html = page(
        ld({"@type": "Product", "name": "Cam", "sku": "4352321", "offers": {"price": "2.49"}}),
        "<h1>Cam</h1><p>435/2321</p><p>2,49 €</p>",
    )
    found = label_page(html.encode(), "text/html", "product", "https://x.example/p/1")
    assert found.display_forms == ["sku"] and found.labels is not None
    assert found.labels["sku"] == ["4352321", "435/2321"]
    assert "2,49" in found.labels["price"]
    assert price_forms(1299.5) == ["1299.5", "1299.50", "1,299.50", "1299,50", "1.299,50"]
    assert currency_forms("gbp") == ["gbp", "£"] and currency_forms("CHF") == "CHF"
    assert currency_forms(None) is None
    dates = date_forms("2026-10-31T19:30:00+00:00")
    assert isinstance(dates, list)
    assert {"2026-10-31", "31 October 2026", "Oct 31, 2026", "10/31/2026", "31/10/2026"} <= set(
        dates
    )
    assert "05/06/2026" not in (date_forms("2026-06-05") or [])  # ambiguous day/month order
    assert date_forms("2026-13-45") == "2026-13-45"  # not a calendar date: kept as given
    norm = normalize_for_match("Total £28.50")
    assert label_agrees(["GBP", "£"], norm, None, "Total £28.50") is True
    assert label_agrees(["GBP", "£"], norm) is False


def test_job_salary_range_forms() -> None:
    html = page(
        ld(
            {
                "@type": "JobPosting",
                "title": "Clerk",
                "baseSalary": {"value": {"minValue": 14, "maxValue": 14.5}},
            }
        ),
        "<h1>Clerk</h1><p>$14–$14.50 Per Hour</p>",
    )
    labels = labels_of(html, "job").labels
    assert labels is not None and "14–14.50" in labels["salary"]
    norm = normalize_for_match(visible_text(parse_html(html)))
    assert label_agrees(labels["salary"], norm) is True
