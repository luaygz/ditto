"""Value parsing and type checks shared by transforms (``cast``), analysis (``value_type``),
verification (``type_check_pass``) and probe.

* :func:`parse_number` reads the first number in a string, understanding ``1,234.50``,
  ``1.234,56``, ``1 234,56`` (NBSP / narrow NBSP), ``1'234.50`` and Indian ``1,23,456``.
  A single ``.`` is always a decimal mark; a single ``,`` is a thousands separator only when
  exactly three digits follow it (``1,234`` -> 1234, ``0,5`` -> 0.5).
* :func:`parse_date` finds the first date in a string and returns it in ISO 8601 form:
  ``YYYY-MM-DD``; ``YYYY-MM`` for month-and-year values; ``YYYY`` when the whole value is a
  year. It understands ISO dates and datetimes, ``March 3, 2021``, ``3 Mar 2021``,
  ``12-Mar-2021``, ``3 de marzo de 2021``, ``3. März 2021`` (English, German, French,
  Spanish, Italian, Portuguese and Dutch month names) and numeric ``d/m/y`` forms (``/`` is read
  month-first, ``.`` and ``-`` day-first, unless one part is above 12).
* :func:`check_value` is strict: the *whole* value must have the type (``"Email: a@b.org"`` is
  not an email).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import date
from decimal import Decimal
from typing import Any, Literal

ValueType = Literal["email", "url", "number", "date", "price", "text"]
VALUE_TYPES: tuple[ValueType, ...] = ("email", "url", "number", "date", "price", "text")

# --------------------------------------------------------------------------------------- numbers

_GROUP_SPACES = "\u00a0\u202f\u2009 "
_NUM_BODY = rf"\d+(?:(?:[.,'’]|[{_GROUP_SPACES}](?=\d{{3}}(?!\d)))\d+)*"
_NUM_TOKEN = re.compile(rf"(?P<sign>[-+\u2212])?(?P<num>{_NUM_BODY})")
_NUMBER_FULL = re.compile(rf"^[-+\u2212]?\s?(?:{_NUM_BODY})\s?%?$")


def _normalize_number(num: str) -> str | None:
    """``1.234,56`` -> ``1234.56``; ``None`` when the separators are inconsistent."""
    if num.isdigit():
        return num
    seps = re.findall(r"\D", num)
    groups = re.split(r"\D", num)
    marks = [s for s in seps if s in ".,"]
    decimal: str | None = None
    if "." in marks and "," in marks:
        decimal = seps[-1]
        if decimal not in ".," or marks.count(decimal) > 1:
            return None
    elif len(marks) == 1 and seps[-1] == marks[0]:
        tail = groups[-1]
        thousands = marks[0] == "," and len(tail) == 3 and groups[0].strip("0") != ""
        decimal = None if thousands else marks[0]
    inner = groups[1:-1] if decimal else groups[1:]
    sizes = [len(g) for g in inner]
    # Thousands groups have three digits (Indian lakh grouping: twos, then a final three).
    if sizes and not (
        all(s == 3 for s in sizes)
        or ("," in seps and all(s == 2 for s in sizes[:-1]) and sizes[-1] == 3)
    ):
        return None
    out = [groups[0], *inner]
    return "".join(out) + ("." + groups[-1] if decimal else "")


def number_string(text: str) -> str | None:
    """The first number in ``text`` as a plain decimal string (``"-1234.56"``), or ``None``."""
    m = _NUM_TOKEN.search(text)
    if not m:
        return None
    body = _normalize_number(m.group("num"))
    if body is None:
        return None
    return ("-" if m.group("sign") in ("-", "\u2212") else "") + body


def parse_number(text: str) -> float | None:
    """The first number in ``text`` (see module docstring for separator rules)."""
    s = number_string(text)
    return float(s) if s is not None else None


def parse_int(text: str) -> int | None:
    """The first number in ``text`` truncated toward zero, exact for long digit strings."""
    s = number_string(text)
    if s is None:
        return None
    return int(s) if "." not in s else int(Decimal(s))


# ----------------------------------------------------------------------------------------- dates

_MONTH_WORDS: dict[int, tuple[str, ...]] = {
    1: ("january", "jan", "januar", "jänner", "janvier", "janv", "enero", "ene", "gennaio",
        "janeiro", "januari"),
    2: ("february", "feb", "febr", "februar", "février", "fevrier", "févr", "fevr", "febrero",
        "febbraio", "fevereiro", "fev", "februari"),
    3: ("march", "mar", "märz", "maerz", "mär", "mrz", "mars", "marzo", "março", "marco",
        "maart", "mrt"),
    4: ("april", "apr", "avril", "avr", "abril", "abr", "aprile"),
    5: ("may", "mai", "mayo", "maggio", "maio", "mei"),
    6: ("june", "jun", "juni", "juin", "junio", "giugno", "junho"),
    7: ("july", "jul", "juli", "juillet", "juil", "julio", "luglio", "julho"),
    8: ("august", "aug", "août", "aout", "agosto", "ago", "augustus"),
    9: ("september", "sep", "sept", "septembre", "septiembre", "setiembre", "settembre",
        "setembro", "set"),
    10: ("october", "oct", "oktober", "okt", "octobre", "octubre", "ottobre", "outubro", "out"),
    11: ("november", "nov", "novembre", "noviembre", "novembro"),
    12: ("december", "dec", "dezember", "dez", "décembre", "decembre", "déc", "diciembre", "dic",
         "dicembre", "dezembro"),
}  # fmt: skip
_MONTHS = {word: num for num, words in _MONTH_WORDS.items() for word in words}
# Any word, checked against _MONTHS afterwards: much faster than a 100-way alternation.
_MON = r"(?P<mon>[^\W\d_]{3,10})\b\.?"
_ORD = r"(?:st|nd|rd|th|er|º|\.)?"
_TIME = (
    r"(?:[t\s]+\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?\s*(?:[ap]\.?m\.?)?"
    r"\s*(?:z|utc|gmt|[+-]\d{2}:?\d{2})?)?"
)

# Full dates (day precision), tried together; the earliest match in the text wins.
_FULL_DATES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ymd", re.compile(rf"(?<!\d)(?P<y>\d{{4}})([-/.])(?P<m>\d{{1,2}})\2(?P<d>\d{{1,2}})(?!\d){_TIME}")),
    ("mdy", re.compile(rf"\b{_MON}\s*(?P<d>\d{{1,2}}){_ORD},?\s+(?P<y>\d{{4}})(?!\d)")),
    (
        "dmy",
        re.compile(
            rf"(?<!\d)(?P<d>\d{{1,2}}){_ORD}(?:\s+(?:de|of))?[\s\-/]+{_MON}"
            rf"(?:\s+(?:de|del))?[\s\-/,]+(?P<y>\d{{4}})(?!\d)"
        ),
    ),
    ("num", re.compile(r"(?<![\d.])(?P<a>\d{1,2})(?P<sep>[-/.])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4}|\d{2})(?![\d])")),
)  # fmt: skip
_MONTH_YEAR: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("my", re.compile(rf"\b{_MON},?\s+(?P<y>\d{{4}})(?!\d)")),
    ("ym", re.compile(r"(?<![\d-])(?P<y>\d{4})-(?P<m>\d{2})(?![\d-])")),
)
_YEAR_ONLY = re.compile(r"^[12]\d{3}$")
_WEEKDAYS = (
    r"mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?|fri(?:day)?|sat(?:urday)?"
    r"|sun(?:day)?|montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|lundi|mardi"
    r"|mercredi|jeudi|vendredi|samedi|dimanche|lunes|martes|miércoles|jueves|viernes|sábado"
    r"|domingo"
)
_DATE_NOISE = re.compile(
    rf"\b(?:{_WEEKDAYS})\b\.?"
    r"|\b\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:\s*[ap]\.?m\.?)?"
    r"|\b(?:utc|gmt|[ecmp][sd]t|cet|cest|bst|z)\b|[+-]\d{2}:?\d{2}\b"
    r"|\b(?:at|on|um|le|el|the)\b"
)


def _full_iso(kind: str, m: re.Match[str]) -> str | None:
    y = int(m.group("y"))
    if kind == "num":
        a, b = int(m.group("a")), int(m.group("b"))
        if len(m.group("y")) == 2:
            y += 2000 if y < 69 else 1900
        if a > 12 >= b or (b <= 12 and m.group("sep") != "/"):
            d, mo = a, b
        else:
            mo, d = a, b
    else:
        mo = _MONTHS.get(m.group("mon"), 0) if kind in ("mdy", "dmy") else int(m.group("m"))
        d = int(m.group("d"))
    return date(y, mo, d).isoformat() if _valid(y, mo, d) else None


def _month_iso(kind: str, m: re.Match[str]) -> str | None:
    y = int(m.group("y"))
    mo = _MONTHS.get(m.group("mon"), 0) if kind == "my" else int(m.group("m"))
    return f"{y:04d}-{mo:02d}" if _valid(y, mo) else None


def _valid(y: int, mo: int, d: int = 1) -> bool:
    try:
        date(y, mo, d)
    except ValueError:
        return False
    return 1000 <= y <= 2999


def _earliest(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
    build: Callable[[str, re.Match[str]], str | None],
) -> tuple[str, int, int] | None:
    best: tuple[str, int, int] | None = None
    for kind, pattern in patterns:
        for m in pattern.finditer(text):
            if best is not None and m.start() >= best[1]:
                break
            iso = build(kind, m)
            if iso is not None:
                best = (iso, m.start(), m.end())
                break
        if best is not None and best[1] == 0:
            break  # nothing can start earlier
    return best


def _date_match(text: str) -> tuple[str, int, int] | None:
    """``(iso, start, end)`` of the first date in lower-cased, whitespace-normalized ``text``;
    day-precision dates win over month-and-year ones."""
    return _earliest(text, _FULL_DATES, _full_iso) or _earliest(text, _MONTH_YEAR, _month_iso)


def _prep(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split()).lower()


def parse_date(text: str) -> str | None:
    """The first date in ``text`` as ISO 8601 (``YYYY-MM-DD``, ``YYYY-MM`` or ``YYYY``)."""
    t = _prep(text)
    if _YEAR_ONLY.match(t):
        return t
    found = _date_match(t)
    return found[0] if found else None


def is_date(text: str) -> bool:
    """True when the whole of ``text`` is a date (day or month precision), optionally with a
    weekday, a time and a time zone. Bare years are numbers, not dates."""
    t = _prep(text)
    found = _date_match(t)
    if not found:
        return False
    rest = _DATE_NOISE.sub(" ", t[: found[1]] + " " + t[found[2] :])
    return not rest.strip(" ,.-–—|()/")


# ------------------------------------------------------------------------------- emails and URLs

_EMAIL = re.compile(
    r"^(?:mailto:)?<?[\w.!#$%&'*+/=?^`{|}~-]+@(?:[^\W_](?:[\w-]*[^\W_])?\.)+"
    r"(?:[^\W\d_]{2,}|xn--[\w-]+)>?(?:\?\S*)?$",
    re.IGNORECASE,
)
_URL_FORMS = (
    re.compile(r"^(?:https?|ftp)://[^\s/?#]+\S*$", re.IGNORECASE),  # absolute
    re.compile(r"^//[^\s/]+\S*$"),  # protocol-relative
    re.compile(r"^(?:/|\./|\.\./)\S*$"),  # root- or dot-relative
    re.compile(r"^\?\S+$"),  # query-only reference
    re.compile(r"^[\w.~%+-]+(?:/[\w.~%+-]*)+(?:[?#]\S*)?$"),  # relative path a/b
    re.compile(  # bare file name
        r"^[\w.~%+-]+\.(?:html?|php\d?|aspx?|jsp|cgi|pdf|jpe?g|png|gif|webp|svg|xml|json|txt)"
        r"(?:[?#]\S*)?$",
        re.IGNORECASE,
    ),
    re.compile(  # bare domain, lower-case TLD (keeps "Mr.Smith" out)
        r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[a-z]{2,24}(?::\d+)?(?:[/?#]\S*)?$"
    ),
)


def is_email(text: str) -> bool:
    return bool(_EMAIL.match(text.strip()))


def is_url(text: str) -> bool:
    t = text.strip()
    if not t or any(ch.isspace() for ch in t) or not re.search(r"[^\W\d_]", t):
        return False
    if re.match(r"^(?:javascript|mailto|tel|data):", t, re.IGNORECASE):
        return False
    return any(p.match(t) for p in _URL_FORMS)


# ------------------------------------------------------------------------------ numbers / prices

_CURRENCY = (
    r"(?:US\$|CA\$|C\$|AU\$|A\$|NZ\$|HK\$|S\$|NT\$|MX\$|R\$|JP¥|CN¥"
    r"|[$€£¥₹₩₽¢₺₴₦฿₫₱₪₡₲₵₸₭₮₼₾]"
    r"|\b(?:USD|EUR|GBP|JPY|CNY|RMB|CAD|AUD|NZD|CHF|INR|KRW|RUB|BRL|MXN|SEK|NOK|DKK|PLN|CZK|HUF"
    r"|TRY|ZAR|SGD|HKD|TWD|THB|IDR|MYR|PHP|VND|AED|SAR|ILS|EGP|NGN|KES|ARS|CLP|COP|PEN|UAH|RON"
    r"|BGN|ISK)\b"
    r"|\bRs\.?|\bkr\.?|zł|Kč|\bFr\.|\bRM\b|\blei\b)"
)
_AMOUNT = rf"[-+\u2212]?\s?(?:{_NUM_BODY})(?:,-|\.[-–])?"
_ONE_PRICE = rf"(?:{_CURRENCY}\s?)?{_AMOUNT}(?:\s?{_CURRENCY})?"
_PRICE = re.compile(
    rf"^(?:(?:from|only|now|ab|desde|dès)\s+)?{_ONE_PRICE}(?:\s*[-–—]\s*{_ONE_PRICE})?$",
    re.IGNORECASE,
)
_HAS_CURRENCY = re.compile(_CURRENCY, re.IGNORECASE)


def is_number(text: str) -> bool:
    t = text.strip()
    return bool(_NUMBER_FULL.match(t)) and number_string(t) is not None


def is_price(text: str) -> bool:
    t = " ".join(text.split())
    return bool(_PRICE.match(t)) and number_string(t) is not None


def has_currency(text: str) -> bool:
    return bool(_HAS_CURRENCY.search(text))


# ------------------------------------------------------------------------------------ public API


def check_value(type_check: ValueType | str | None, value: Any) -> bool:
    """True when ``value`` has type ``type_check``. ``None``/``text`` accept any non-empty value;
    numeric values pass ``number`` and ``price``; unknown type names accept any value."""
    if value is None:
        return False
    if isinstance(value, bool):
        return type_check in (None, "text")
    if isinstance(value, (int, float)):
        return type_check in (None, "text", "number", "price")
    text = str(value).strip()
    if not text:
        return False
    if type_check == "email":
        return is_email(text)
    if type_check == "url":
        return is_url(text)
    if type_check == "number":
        return is_number(text)
    if type_check == "price":
        return is_price(text)
    if type_check == "date":
        return is_date(text)
    return True


def infer_value_type(values: Iterable[Any], *, threshold: float = 0.9) -> ValueType | None:
    """Most specific type matched by at least ``threshold`` of the non-empty ``values``.

    Order: email, url, date, price (only when at least half the values carry a currency
    marker), number. Returns ``"text"`` when values exist but none qualifies, ``None`` when
    there are no values.
    """
    vals = [s for s in (str(v).strip() for v in values if v is not None) if s]
    if not vals:
        return None
    need = threshold * len(vals)
    for kind in ("email", "url", "date"):
        if sum(check_value(kind, v) for v in vals) >= need:
            return kind
    if sum(is_price(v) for v in vals) >= need and sum(has_currency(v) for v in vals) >= 0.5 * len(
        vals
    ):
        return "price"
    if sum(is_number(v) for v in vals) >= need:
        return "number"
    return "text"
