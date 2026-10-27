"""Parse, clean and describe pages structurally.

This module defines the *structural path* format every later stage relies on:

    step  := tag ['#' id] ('.' class)* ('[' attr '=' value ']')*
    path  := step (' > ' step)*            e.g.  "html > body > div.product > dl.specs > dd"
    attribute path := path ' @' attrname   e.g.  "html > body > div.product > a.more @href"

* ``tag`` is lower case.
* ``id`` is kept only when it carries no digit and passes the class rules below.
* classes are canonicalized (utility / state / hashed classes dropped), sorted, at most three.
* bracketed attributes appear in the fixed order ``itemprop``, ``data-testid``, ``role``.
* there are no positional indices, so repeated siblings share one path.

The cleaned tree is what the extractor, the analysis and the LLM see: ``<script>``, ``<style>``,
``<noscript>``, ``<svg>``, ``<iframe>``, ``<template>``, comments, processing instructions and
everything in ``<head>`` except ``<title>`` are removed, and so are characters XML forbids
(see ``parse_html``), so every text and attribute value can be written back with lxml.
"""

from __future__ import annotations

import codecs
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar, cast

from lxml import etree, html

from ditto.textutil import normalize_ws

_E = TypeVar("_E", bound=etree._Element)

STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe", "template")
KEPT_ATTRS = ("itemprop", "data-testid", "role")
LINK_ATTRS = ("href", "src")
MAX_CLASSES = 3
PATH_SEP = " > "
ATTR_SEP = " @"

# Utility-framework prefixes (Tailwind, Bootstrap, etc.). A class matching any prefix is dropped.
UTILITY_PREFIXES: tuple[str, ...] = (
    "m-",
    "mt-",
    "mb-",
    "ml-",
    "mr-",
    "mx-",
    "my-",
    "ms-",
    "me-",
    "p-",
    "pt-",
    "pb-",
    "pl-",
    "pr-",
    "px-",
    "py-",
    "ps-",
    "pe-",
    "w-",
    "h-",
    "min-w-",
    "min-h-",
    "max-w-",
    "max-h-",
    "size-",
    "text-",
    "bg-",
    "col-",
    "row-",
    "flex",
    "grid",
    "gap-",
    "space-",
    "order-",
    "border",
    "rounded",
    "shadow",
    "font-",
    "leading-",
    "tracking-",
    "justify-",
    "items-",
    "content-",
    "self-",
    "place-",
    "z-",
    "top-",
    "left-",
    "right-",
    "bottom-",
    "inset-",
    "overflow",
    "d-",
    "float-",
    "align-",
    "opacity-",
    "cursor-",
    "transition",
    "duration-",
    "ease-",
    "delay-",
    "animate-",
    "transform",
    "translate-",
    "scale-",
    "rotate-",
    "ring-",
    "outline-",
    "fill-",
    "stroke-",
    "object-",
    "aspect-",
    "basis-",
    "grow",
    "shrink",
    "decoration-",
    "underline",
    "line-clamp-",
    "truncate",
    "whitespace-",
    "break-",
    "list-",
    "pointer-events-",
    "select-",
    "sr-only",
    "not-sr-only",
    "visible",
    "invisible",
    "offset-",
    "g-",
    "gx-",
    "gy-",
    "fs-",
    "fw-",
    "lh-",
    "u-",
    "is-",
    "has-",
)
# Whole-word utility and state classes (state classes vary per page or per position).
UTILITY_WORDS: frozenset[str] = frozenset(
    {
        "row",
        "col",
        "block",
        "inline",
        "hidden",
        "clearfix",
        "relative",
        "absolute",
        "fixed",
        "sticky",
        "static",
        "container-fluid",
        "active",
        "selected",
        "current",
        "open",
        "closed",
        "odd",
        "even",
        "first",
        "last",
        "disabled",
        "focus",
        "hover",
        "show",
        "fade",
        "in",
        "collapse",
        "collapsed",
        "expanded",
        "loaded",
        "loading",
        "visible",
        "invisible",
        "pull-left",
        "pull-right",
        "left",
        "right",
        "center",
    }
)
_HASHED_PREFIX = re.compile(r"^(css-|sc-|jsx-|emotion-|svelte-|astro-|_)")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{5,}")
_SAFE_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)


def looks_hashed(token: str) -> bool:
    """True for generated class names: css-/sc-/jsx-/_ prefixes, or a 5+ alnum run mixing
    letters and digits (``a8f3k``, ``Xy7Qz1``)."""
    if _HASHED_PREFIX.match(token):
        return True
    for run in _ALNUM_RUN.findall(token):
        if any(ch.isdigit() for ch in run) and any(ch.isalpha() for ch in run):
            return True
    return False


def is_utility_class(token: str) -> bool:
    low = token.lower()
    if low in UTILITY_WORDS:
        return True
    return any(low.startswith(prefix) for prefix in UTILITY_PREFIXES)


def canonical_classes(class_attr: str | None) -> list[str]:
    """Canonical classes of a node: drop utility, state, hashed and unsafe tokens; dedupe;
    sort; keep at most three."""
    if not class_attr:
        return []
    kept = {
        tok
        for tok in class_attr.split()
        if _SAFE_TOKEN.match(tok) and not is_utility_class(tok) and not looks_hashed(tok)
    }
    return sorted(kept)[:MAX_CLASSES]


def keep_identifier(value: str | None) -> str | None:
    """Return a stripped ``id``/``data-testid`` value if it is stable enough to anchor on.

    Values containing any digit (``post-1234``, ``:r1:``) or looking hashed are dropped.
    """
    if not value:
        return None
    value = value.strip()
    if not value or not _SAFE_TOKEN.match(value) or any(ch.isdigit() for ch in value):
        return None
    if looks_hashed(value):
        return None
    return value


def _clean_attr_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = normalize_ws(value)
    if not value or any(ch in value for ch in "[]=>'\""):
        return None
    return value


@dataclass(frozen=True)
class Step:
    tag: str
    id: str | None = None
    classes: tuple[str, ...] = ()
    attrs: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def label(self) -> str:
        out = self.tag
        if self.id:
            out += "#" + self.id
        for cls in self.classes:
            out += "." + cls
        for name, value in self.attrs:
            out += f"[{name}={value}]"
        return out

    def xpath(self) -> str:
        """XPath 1.0 step matching this structural step (anchored on id/classes/attrs)."""
        preds: list[str] = []
        if self.id:
            preds.append(f"@id='{self.id}'")
        for cls in self.classes:
            preds.append(f"contains(concat(' ', normalize-space(@class), ' '), ' {cls} ')")
        for name, value in self.attrs:
            preds.append(f"@{name}='{value}'")
        return self.tag + "".join(f"[{p}]" for p in preds)


def _step(
    tag: str,
    raw_id: str | None,
    raw_class: str | None,
    itemprop: str | None,
    testid: str | None,
    role: str | None,
) -> Step:
    ident = keep_identifier(raw_id)
    classes = tuple(canonical_classes(raw_class))
    attrs: list[tuple[str, str]] = []
    for name, raw in zip(KEPT_ATTRS, (itemprop, testid, role), strict=True):
        value = keep_identifier(raw) if name == "data-testid" else _clean_attr_value(raw)
        if value:
            attrs.append((name, value))
    return Step(tag.lower(), ident, classes, tuple(attrs))


_cached_step = lru_cache(maxsize=1 << 16)(_step)


@lru_cache(maxsize=1 << 16)
def _cached_label(
    tag: str,
    raw_id: str | None,
    raw_class: str | None,
    itemprop: str | None,
    testid: str | None,
    role: str | None,
) -> str:
    return _step(tag, raw_id, raw_class, itemprop, testid, role).label()


def step_of(el: etree._Element) -> Step:
    """The structural step of an element (memoized on its tag and relevant attributes)."""
    get = el.get
    return _cached_step(
        str(el.tag), get("id"), get("class"), get("itemprop"), get("data-testid"), get("role")
    )


# Attributes a step label depends on; an element with none of them is labelled by its tag.
STEP_ATTRS = ("id", "class", *KEPT_ATTRS)
_STEP_ATTR_SET = frozenset(STEP_ATTRS)
_TAG_LABELS: dict[str, str] = {}


def tag_label(tag: str) -> str:
    """Step label of an element that has none of ``STEP_ATTRS`` (its lower-cased tag)."""
    label = _TAG_LABELS.get(tag)
    if label is None:
        label = _TAG_LABELS[tag] = tag.lower()
    return label


def step_label(el: etree._Element) -> str:
    """``step_of(el).label()``, memoized: pages of one site repeat a few thousand step kinds."""
    keys = el.keys()
    if not keys or _STEP_ATTR_SET.isdisjoint(keys):
        return tag_label(str(el.tag))
    get = el.get
    return _cached_label(
        str(el.tag), get("id"), get("class"), get("itemprop"), get("data-testid"), get("role")
    )


_STEP_RE = re.compile(
    r"^(?P<tag>[^#.\[\s]+)(?:#(?P<id>[^.\[\s]+))?(?P<classes>(?:\.[^.\[\s]+)*)(?P<attrs>(?:\[[^\]]+\])*)$"
)
_ATTR_RE = re.compile(r"\[([^=\]]+)=([^\]]*)\]")


def parse_step(label: str) -> Step:
    m = _STEP_RE.match(label)
    if not m:
        raise ValueError(f"not a structural step: {label!r}")
    classes = tuple(c for c in (m.group("classes") or "").split(".") if c)
    attrs = tuple((a, v) for a, v in _ATTR_RE.findall(m.group("attrs") or ""))
    return Step(m.group("tag"), m.group("id"), classes, attrs)


def split_path(path: str) -> tuple[list[Step], str | None]:
    """Split a structural path (or attribute path) into steps and the attribute name."""
    attr: str | None = None
    if ATTR_SEP in path:
        path, attr = path.rsplit(ATTR_SEP, 1)
    return [parse_step(s) for s in path.split(PATH_SEP)], attr


_PARSERS = threading.local()

# ASCII characters that ``\s`` matches in a str pattern (``str.isspace``), as bytes: whitespace
# before an XML declaration looks the same to ``_XML_DECL`` in text and to this in UTF-8 bytes.
_LEADING_SPACE = re.compile(rb"[ \t\n\r\x0b\x0c\x1c-\x1f]*")

# XML-invalid characters that lxml's HTML parser keeps (it turns NUL into U+FFFD itself) but no
# lxml setter accepts afterwards ("All strings must be XML compatible"). Control characters
# that are whitespace to Python become spaces, so whitespace-normalized text stays the same;
# the other controls, U+FFFE, U+FFFF and lone surrogates are dropped.
_CONTROL_SPACES = b"\x0b\x0c\x1c\x1d\x1e\x1f"
_CONTROL_DROP = bytes([*range(0x01, 0x09), *range(0x0E, 0x1C)])
_TO_SPACE = bytes.maketrans(_CONTROL_SPACES, b" " * len(_CONTROL_SPACES))
_FLAG_CONTROLS = bytes.maketrans(
    _CONTROL_SPACES + _CONTROL_DROP, bytes(len(_CONTROL_SPACES) + len(_CONTROL_DROP))
)
_XML_INVALID = re.compile("[\x01-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff\ud800-\udfff]")
_SURROGATES = re.compile("[\ud800-\udfff]")

# libxml2 resolves numeric character references the HTML5 way: NUL, surrogates and code points
# past U+10FFFF become U+FFFD, but the other C0 controls, U+FFFE and U+FFFF are kept (SWDE
# movie-amctv writes ``O&#27;Toole``), so a page referring to one is sanitized after parsing.
# Digits are capped: a longer number is past U+10FFFF anyway (a harmless false alarm).
_CHAR_REF = re.compile(rb"&#(?:[xX]0*([0-9a-fA-F]{1,7})|0*([0-9]{1,8}))")


def _xml_forbidden(code: int) -> bool:
    """Whether XML 1.0 forbids the character with code point ``code``."""
    return (
        (code < 0x20 and code not in (0x9, 0xA, 0xD))
        or 0xD800 <= code <= 0xDFFF
        or code in (0xFFFE, 0xFFFF)
        or code > 0x10FFFF
    )


def _refers_to_forbidden(source: bytes) -> bool:
    """Whether ``source`` has a numeric character reference to a character XML forbids."""
    for m in _CHAR_REF.finditer(source):
        hexa, dec = m.groups()
        if _xml_forbidden(int(hexa, 16) if hexa is not None else int(dec)):
            return True
    return False


def _xml_safe(data: bytes, nonchars: bool) -> bytes:
    """UTF-8 ``data`` without XML-invalid characters; ``nonchars``: it may hold U+FFFE/U+FFFF
    (whose UTF-8 sequences cannot straddle characters, nor can an ASCII control byte)."""
    if data.translate(_FLAG_CONTROLS) != data:
        data = data.translate(_TO_SPACE, _CONTROL_DROP)
    if nonchars:
        data = data.replace(b"\xef\xbf\xbe", b"").replace(b"\xef\xbf\xbf", b"")
    return data


def _xml_safe_text(text: str) -> str:
    """``_xml_safe`` for one string value of an already parsed tree."""
    return _XML_INVALID.sub(lambda m: " " if m.group().isspace() else "", text)


def _sanitize_tree(root: etree._Element) -> None:
    """Apply ``_xml_safe_text`` to every text, tail and attribute value (for pages lxml decoded
    itself, which never pass through ``_xml_safe``, and pages whose character references
    resolve to XML-invalid characters)."""
    find = _XML_INVALID.search
    for el in root.iter():
        if el.text and find(el.text):
            el.text = _xml_safe_text(el.text)
        if el.tail and find(el.tail):
            el.tail = _xml_safe_text(el.tail)
        for name, value in el.items():
            text = str(value)  # HTML trees hold str values; the stubs also allow bytes
            if find(text):
                el.set(name, _xml_safe_text(text))


def _parser(plain: bool, utf8: bool = False) -> etree.HTMLParser:
    """Per-thread reusable parser (lxml parsers must not be shared between threads). ``utf8``
    parsers read bytes as UTF-8, ignoring any ``<meta charset>``."""
    key = ("plain" if plain else "html") + ("-utf8" if utf8 else "")
    parser: etree.HTMLParser | None = getattr(_PARSERS, key, None)
    if parser is None:
        cls = etree.HTMLParser if plain else html.HTMLParser
        if utf8:
            parser = cls(remove_comments=True, remove_pis=True, encoding="utf-8")
        else:
            parser = cls(remove_comments=True, remove_pis=True)
        setattr(_PARSERS, key, parser)
    return parser


@lru_cache(maxsize=256)
def _codec_name(encoding: str) -> str | None:
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None


def _utf8_page(data: bytes, encoding: str) -> bytes | None:
    """``data`` made XML-safe when it can go to libxml2 without decoding: valid UTF-8 declared
    as UTF-8 (``utf-8-sig``: its one leading BOM removed, as the codec does) that is not blank
    and opens (after ASCII whitespace) with neither ``<?`` nor a non-ASCII character, so the
    decoded text has no XML declaration to remove and the same blankness. Otherwise None."""
    codec = _codec_name(encoding)
    if codec == "utf-8-sig":
        if data.startswith(codecs.BOM_UTF8):
            data = data[len(codecs.BOM_UTF8) :]
    elif codec != "utf-8":
        return None
    start = cast(re.Match[bytes], _LEADING_SPACE.match(data)).end()
    if start == len(data) or data[start] >= 0x80 or data.startswith(b"<?", start):
        return None
    nonchars = False
    if not data.isascii():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        nonchars = "\ufffe" in text or "\uffff" in text
    return _xml_safe(data, nonchars)


def _utf8_text(text: str) -> bytes:
    """``text`` as UTF-8 for libxml2: XML declaration removed, XML-safe, empty when blank."""
    text = _XML_DECL.sub("", text, count=1)
    if not text.strip():
        return b""
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates
        text = _SURROGATES.sub("", text)
        data = text.encode("utf-8")
    return _xml_safe(data, "\ufffe" in text or "\uffff" in text)


def _source(data: bytes | str, encoding: str | None) -> tuple[bytes, bool]:
    """What ``_parse`` hands libxml2, and whether lxml must sniff its encoding (bytes without
    ``encoding``; everything else is XML-safe UTF-8)."""
    if isinstance(data, str):
        return _utf8_text(data), False
    if not encoding:
        return data, True
    fast = _utf8_page(data, encoding)
    return (fast if fast is not None else _utf8_text(data.decode(encoding, "replace"))), False


def _parse(data: bytes | str, encoding: str | None, plain: bool) -> etree._Element:
    """Parse bytes (decoded with ``encoding``, or sniffed by lxml without one) or text into a
    tree free of XML-invalid characters (see ``_xml_safe``; those written as character
    references, or in pages lxml decodes itself, are removed from the parsed tree).

    Text and decodable bytes reach libxml2 as UTF-8 with the encoding stated. That yields the
    same tree as handing lxml the decoded string (lxml passes its internal Latin-1/UCS-2/UCS-4
    buffer with the encoding stated, which libxml2 then converts) at less cost, and valid UTF-8
    pages skip the Python decode altogether."""
    source, sniffed = _source(data, encoding)
    parser = _parser(plain, utf8=not sniffed)
    root: etree._Element | None = None
    if source.strip():
        try:
            root = etree.fromstring(source, parser)
        except etree.XMLSyntaxError:  # e.g. a document that is only a comment
            root = None
    if root is None:
        root = etree.fromstring("<html></html>", parser)
    elif sniffed or _refers_to_forbidden(source):
        _sanitize_tree(root)
    return root


def parse_html(data: bytes | str, encoding: str | None = None) -> html.HtmlElement:
    """Parse a page with lxml's HTML parser. ``data`` may be bytes (decoded with ``encoding``
    when given, else by lxml's own sniffing) or an already-decoded string. Empty or
    unparseable input yields an empty ``<html>`` document. XML-invalid characters, which no
    lxml setter would accept again, are removed, whether written as such or as character
    references (``&#27;``): control characters that are whitespace to Python (``\\v``, ``\\f``,
    ``\\x1c``-``\\x1f``) become spaces; other controls, U+FFFE, U+FFFF and lone surrogates are
    dropped (lxml itself turns NUL into U+FFFD)."""
    return cast(html.HtmlElement, _parse(data, encoding, plain=False))


def _clean(root: _E, comments: bool) -> _E:
    etree.strip_elements(root, *STRIP_TAGS, with_tail=False)
    if comments:
        etree.strip_tags(root, etree.Comment, etree.ProcessingInstruction)
    for head in root.iterchildren("head"):
        for child in list(head):
            if not isinstance(child.tag, str) or child.tag.lower() != "title":
                head.remove(child)
        head.text = None
    return root


def clean_tree(root: _E) -> _E:
    """Strip non-content elements in place and return ``root``."""
    return _clean(root, comments=True)


# Trees from ``_parse`` are cleaned with ``comments=False``: its parsers never create comment or
# processing-instruction nodes (lxml unsets those SAX callbacks), so there is nothing to strip.


def clean_html(data: bytes | str, encoding: str | None = None) -> html.HtmlElement:
    """Parse and clean a page: the exact tree the extractor, analysis and LLM operate on."""
    return _clean(parse_html(data, encoding), comments=False)


def clean_etree(data: bytes | str, encoding: str | None = None) -> etree._Element:
    """``clean_html`` with plain ``lxml.etree`` elements instead of ``lxml.html`` ones: the same
    tags, attributes, text and structural paths, without the per-node Python element-class
    lookup (faster to traverse and to return from XPath). For bulk passes (fingerprinting,
    extraction) whose results do not depend on the element class."""
    return _clean(_parse(data, encoding, plain=True), comments=False)


def own_text(el: etree._Element) -> str:
    """Whitespace-normalized text directly inside ``el`` (its text plus children's tails)."""
    parts = [el.text or ""]
    parts.extend(child.tail or "" for child in el)
    return normalize_ws("".join(parts))


def iter_paths(root: etree._Element) -> Iterator[tuple[etree._Element, str]]:
    """Yield ``(element, structural_path)`` for every element in document order."""
    stack: list[tuple[etree._Element, str]] = [(root, step_label(root))]
    while stack:
        el, path = stack.pop()
        yield el, path
        children = [c for c in el if isinstance(c.tag, str)]
        for child in reversed(children):
            stack.append((child, path + PATH_SEP + step_label(child)))


@dataclass(frozen=True)
class TextItem:
    """One text-bearing node (``attr is None``) or one ``href``/``src`` attribute value."""

    path: str  # structural path; for attributes ends with " @href" / " @src"
    text: str
    element: etree._Element
    attr: str | None = None


def iter_text_items(root: etree._Element) -> Iterator[TextItem]:
    """Yield a ``TextItem`` for every element with own text and every href/src attribute."""
    for el, path in iter_paths(root):
        text = own_text(el)
        if text:
            yield TextItem(path, text, el)
        for attr in LINK_ATTRS:
            value = el.get(attr)
            if value and value.strip():
                yield TextItem(path + ATTR_SEP + attr, value.strip(), el, attr)


def node_text(el: etree._Element) -> str:
    """Whitespace-normalized full string value of an element (what a selector extracts)."""
    return normalize_ws("".join(str(t) for t in el.itertext()))
