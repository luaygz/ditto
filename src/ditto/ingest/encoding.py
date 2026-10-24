"""Character-encoding detection and HTML sniffing for raw page bytes.

Precedence follows the WHATWG encoding-sniffing algorithm: a byte-order mark, then the transport
charset (HTTP ``Content-Type``), then a ``<meta charset>`` / ``http-equiv`` / XML declaration in
the first 4 KB, then strict UTF-8 validation, then ``charset_normalizer``, then windows-1252.
Encoding names are normalized to Python codec names with :func:`codecs.lookup`.
"""

from __future__ import annotations

import codecs
import re

import charset_normalizer

PRESCAN_BYTES = 4096
SNIFF_BYTES = 1024
FALLBACK_ENCODING = "cp1252"

# Checked in order; the UTF-8 BOM is kept in the name so decoding strips it.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF16_LE, "utf-16"),
)

# Labels browsers decode with a superset encoding (WHATWG encoding standard, "labels").
_SUPERSETS = {
    "ascii": "cp1252",
    "iso8859-1": "cp1252",
    "iso8859-9": "cp1254",
    "iso8859-11": "cp874",
    "tis-620": "cp874",
    "gb2312": "gbk",
    "shift_jis": "cp932",
    "euc_kr": "cp949",
}

# Python codecs that are text encodings but never a web page charset.
_REJECTED = frozenset(
    {"idna", "punycode", "raw-unicode-escape", "unicode-escape", "undefined", "utf-7"}
)

_LABEL = rb"""["']?\s*([A-Za-z0-9._:\-]+)"""
_META_CHARSET = re.compile(rb"<meta\b[^>]*?\bcharset\s*=\s*" + _LABEL, re.IGNORECASE)
_XML_ENCODING = re.compile(rb"^\s*<\?xml\b[^>]*?\bencoding\s*=\s*" + _LABEL, re.IGNORECASE)
_HEADER_CHARSET = re.compile(r"""charset\s*=\s*["']?\s*([^\s;"']+)""", re.IGNORECASE)
_HTML_MARKERS = ("<!doctype html", "<html", "<head", "<body")


def normalize_encoding(label: str | None) -> str | None:
    """Python codec name for an encoding label, or ``None`` if it is unknown or not a text
    encoding usable for HTML (e.g. ``base64``, ``rot13``, ``utf-7``)."""
    if not label:
        return None
    label = label.strip().strip("\"'").strip()
    if not label:
        return None
    try:
        info = codecs.lookup(label)
    except LookupError:
        return None
    if not getattr(info, "_is_text_encoding", True) or info.name in _REJECTED:
        return None
    return _SUPERSETS.get(info.name, info.name)


def charset_from_content_type(content_type: str | None) -> str | None:
    """The raw ``charset`` parameter of a ``Content-Type`` header value, if any."""
    if not content_type:
        return None
    match = _HEADER_CHARSET.search(content_type)
    return match.group(1) if match else None


def bom_encoding(data: bytes) -> str | None:
    for bom, name in _BOMS:
        if data.startswith(bom):
            return name
    return None


def declared_encoding(data: bytes) -> str | None:
    """Encoding declared in the document itself (``<meta>``, then XML declaration) within the
    first :data:`PRESCAN_BYTES`. A UTF-16/32 declaration means UTF-8: the declaration was
    readable as ASCII, so the bytes cannot be UTF-16 (WHATWG rule)."""
    head = data[:PRESCAN_BYTES]
    for pattern in (_META_CHARSET, _XML_ENCODING):
        match = pattern.search(head)
        if match is None:
            continue
        name = normalize_encoding(match.group(1).decode("ascii"))
        if name is None:
            continue
        if name.startswith(("utf-16", "utf-32")):
            return "utf-8"
        return name
    return None


def detect_encoding(data: bytes, http_charset: str | None = None) -> str:
    """Python codec name to decode ``data`` with (always succeeds with ``errors='replace'``)."""
    name = bom_encoding(data) or normalize_encoding(http_charset) or declared_encoding(data)
    if name is not None:
        return name
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return "utf-8"
    return guess_encoding(data) or FALLBACK_ENCODING


def guess_encoding(data: bytes) -> str | None:
    """``charset_normalizer``'s best guess. Among equally good guesses (same mess and
    coherence, or the same decoded text) windows-1252 wins: it is the web's default legacy
    encoding, while the detector's tie order is arbitrary (cp1250 for French text, say)."""
    matches = charset_normalizer.from_bytes(data)
    best = matches.best()
    if best is None:
        return None
    for match in matches:
        same = match.chaos == best.chaos and match.coherence == best.coherence
        if same and FALLBACK_ENCODING in match.could_be_from_charset:
            return FALLBACK_ENCODING
    return normalize_encoding(best.encoding)


def sniff_html(data: bytes) -> bool:
    """Whether bytes without a telling file name look like an HTML document: the first
    non-blank character is ``<`` and a doctype/html/head/body tag occurs in the first 1 KB."""
    head = data[:SNIFF_BYTES]
    encoding = bom_encoding(head) or "latin-1"
    text = head.decode(encoding, errors="replace").lstrip("\ufeff \t\r\n\f").lower()
    return text.startswith("<") and any(marker in text for marker in _HTML_MARKERS)
