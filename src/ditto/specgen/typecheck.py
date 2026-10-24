"""Value type checks shared by analysis (value_type), verification (type_check_pass) and probe."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

ValueType = Literal["email", "url", "number", "date", "price", "text"]

_EMAIL = re.compile(r"^(?:mailto:)?[\w.+'-]+@[\w-]+(?:\.[\w-]+)+$", re.IGNORECASE)
_URL = re.compile(r"^(?:https?://|//|/|\./|\.\./|www\.)\S+$|^[\w-]+(?:/[\w.%-]+)+(?:\?\S*)?$", re.IGNORECASE)
_NUMBER = re.compile(r"^[-+]?(?:\d{1,3}(?:[,\s.\u00a0]\d{3})+|\d+)(?:[.,]\d+)?\s*%?$")
_CURRENCY = r"(?:[$€£¥₹₩₽¢]|US\$|USD|EUR|GBP|JPY|CAD|AUD|CHF|INR|Rs\.?)"
_PRICE = re.compile(
    rf"^(?:{_CURRENCY}\s*)?[-+]?\d[\d,.\s\u00a0]*(?:\s*{_CURRENCY})?$|^{_CURRENCY}\s*\d", re.IGNORECASE
)
_HAS_CURRENCY = re.compile(_CURRENCY, re.IGNORECASE)
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(
    rf"^(?:\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}(?:[T ]\d{{1,2}}:\d{{2}}(?::\d{{2}})?\S*)?"
    rf"|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}"
    rf"|{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS},?\s+\d{{4}}"
    rf"|{_MONTHS}\s+\d{{4}})$",
    re.IGNORECASE,
)


def check_value(type_check: ValueType | str | None, value: Any) -> bool:
    """True when ``value`` looks like ``type_check``. ``None``/``text`` accept any non-empty value."""
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return type_check in (None, "text", "number", "price")
    text = str(value).strip()
    if not text:
        return False
    if type_check in (None, "text"):
        return True
    if type_check == "email":
        return bool(_EMAIL.match(text))
    if type_check == "url":
        return bool(_URL.match(text)) and " " not in text
    if type_check == "number":
        return bool(_NUMBER.match(text))
    if type_check == "price":
        return bool(_PRICE.match(text)) and any(ch.isdigit() for ch in text)
    if type_check == "date":
        return bool(_DATE.match(text))
    return True


def infer_value_type(values: Iterable[str], *, threshold: float = 0.9) -> ValueType | None:
    """Most specific type matched by at least ``threshold`` of the non-empty ``values``.

    Order: email, url, date, price (only when a currency marker appears), number. Returns
    ``"text"`` when values exist but none qualifies, ``None`` when there are no values.
    """
    vals = [v.strip() for v in values if v and v.strip()]
    if not vals:
        return None
    n = len(vals)
    for kind in ("email", "url", "date"):
        if sum(check_value(kind, v) for v in vals) >= threshold * n:
            return kind
    if sum(check_value("price", v) for v in vals) >= threshold * n and any(
        _HAS_CURRENCY.search(v) for v in vals
    ):
        return "price"
    if sum(check_value("number", v) for v in vals) >= threshold * n:
        return "number"
    return "text"
