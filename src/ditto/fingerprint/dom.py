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
everything in ``<head>`` except ``<title>`` are removed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from lxml import etree, html

from ditto.textutil import normalize_ws

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


def step_of(el: etree._Element) -> Step:
    tag = str(el.tag).lower()
    ident = keep_identifier(el.get("id"))
    classes = tuple(canonical_classes(el.get("class")))
    attrs: list[tuple[str, str]] = []
    for name in KEPT_ATTRS:
        raw = el.get(name)
        value = keep_identifier(raw) if name == "data-testid" else _clean_attr_value(raw)
        if value:
            attrs.append((name, value))
    return Step(tag, ident, classes, tuple(attrs))


def step_label(el: etree._Element) -> str:
    return step_of(el).label()


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


def parse_html(data: bytes | str, encoding: str | None = None) -> html.HtmlElement:
    """Parse a page with lxml's HTML parser. ``data`` may be bytes (decoded with ``encoding``
    when given, else by lxml's own sniffing) or an already-decoded string."""
    if isinstance(data, bytes):
        if encoding:
            data = data.decode(encoding, errors="replace")
        else:
            parser = html.HTMLParser(remove_comments=True, remove_pis=True)
            root = html.document_fromstring(data or b"<html></html>", parser=parser)
            return root
    text = _XML_DECL.sub("", data, count=1)
    if not text.strip():
        text = "<html></html>"
    parser = html.HTMLParser(remove_comments=True, remove_pis=True)
    return html.document_fromstring(text, parser=parser)


def clean_tree(root: html.HtmlElement) -> html.HtmlElement:
    """Strip non-content elements in place and return ``root``."""
    etree.strip_elements(root, *STRIP_TAGS, with_tail=False)
    etree.strip_tags(root, etree.Comment, etree.ProcessingInstruction)
    for head in root.findall("head"):
        for child in list(head):
            if not isinstance(child.tag, str) or child.tag.lower() != "title":
                head.remove(child)
        head.text = None
    return root


def clean_html(data: bytes | str, encoding: str | None = None) -> html.HtmlElement:
    """Parse and clean a page: the exact tree the extractor, analysis and LLM operate on."""
    return clean_tree(parse_html(data, encoding))


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
