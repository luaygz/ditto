"""URL patterns: a URL with its variable parts (digit runs, slugs, ids, query values) replaced by
placeholders. Used only as a tie-breaker when clustering, never as the primary signal."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

_DIGITS = re.compile(r"\d+")
_ALL_DIGITS = re.compile(r"^\d+$")
_HEX_ID = re.compile(r"^(?=[0-9a-fA-F-]*\d)(?=[0-9a-fA-F-]*[a-fA-F])[0-9a-fA-F-]{8,}$")
_EXT = re.compile(r"^(?P<stem>.+)\.(?P<ext>[A-Za-z][A-Za-z0-9]{0,4})$")
_SLUG_SEP = re.compile(r"[-_+]|%20")
_WORD = re.compile(r"[A-Za-z]")
_LONG_TOKEN = 24


def _segment(seg: str) -> str:
    """Pattern of one path segment (or query value)."""
    if not seg:
        return seg
    m = _EXT.match(seg)
    if m and not _ALL_DIGITS.match(m.group("ext")):
        return _segment(m.group("stem")) + "." + m.group("ext").lower()
    if _ALL_DIGITS.match(seg):
        return "{n}"
    if _HEX_ID.match(seg):
        return "{id}"
    parts = [p for p in _SLUG_SEP.split(seg) if p]
    if len(parts) >= 2 and any(_WORD.search(p) for p in parts):
        return "{slug}"
    if len(seg) >= _LONG_TOKEN:
        return "{slug}"
    return _DIGITS.sub("{n}", seg)


def url_pattern(url: str) -> str:
    """Normalize a URL to its pattern, e.g.
    ``https://shop.example.com/p/blue-widget-42.html?page=3`` ->
    ``shop.example.com/p/{slug}.html?page={n}``.

    The scheme, port, fragment, ``www.`` prefix and query-key order are ignored; path segments
    keep literal words; URLs that are file paths keep their directory structure."""
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return _DIGITS.sub("{n}", url)
    host = host.removeprefix("www.")
    path = "/".join(_segment(seg) for seg in parts.path.split("/"))
    out = host + path
    if parts.query:
        query = sorted(
            f"{key}={'{n}' if _ALL_DIGITS.match(value) else '{v}' if value else ''}"
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        )
        if query:
            out += "?" + "&".join(query)
    return out
