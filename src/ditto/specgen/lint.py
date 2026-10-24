"""XPath 1.0 selector lint enforced at spec validation time.

Rules:
* must compile with ``lxml.etree.XPath``;
* only XPath 1.0 core functions (no ``matches()``, ``lower-case()``, ``upper-case()``,
  ``ends-with()``, nor any other XPath 2.0/EXSLT function);
* no ``comment()`` node test and no ``data-x-`` attribute (those exist only in the annotated
  union page shown to the LLM);
* absolute selectors start with ``/`` or ``(``; relative ones start with ``.``;
* no purely positional paths: a selector whose every step is a bare tag is rejected when it is
  rooted at ``/``, carries a positional index ``[n]``, or ends on a generic container tag.
  Descendant-anchored semantic tags such as ``//header//h1`` are allowed.
"""

from __future__ import annotations

import re
from functools import lru_cache

from lxml import etree

XPATH1_FUNCTIONS = frozenset(
    {
        "last",
        "position",
        "count",
        "id",
        "local-name",
        "namespace-uri",
        "name",
        "string",
        "concat",
        "starts-with",
        "contains",
        "substring-before",
        "substring-after",
        "substring",
        "string-length",
        "normalize-space",
        "translate",
        "boolean",
        "not",
        "true",
        "false",
        "lang",
        "number",
        "sum",
        "floor",
        "ceiling",
        "round",
    }
)
NODE_TESTS = frozenset({"text", "node", "processing-instruction"})
BANNED_FUNCTIONS = frozenset({"matches", "lower-case", "upper-case", "ends-with", "comment"})
GENERIC_TAGS = frozenset(
    {
        "div",
        "span",
        "p",
        "li",
        "ul",
        "ol",
        "td",
        "tr",
        "th",
        "tbody",
        "table",
        "a",
        "section",
        "article",
        "dd",
        "dt",
        "dl",
        "b",
        "strong",
        "i",
        "em",
        "font",
        "*",
    }
)

_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
_FUNC = re.compile(r"(?<![\w@$:-])([A-Za-z_][\w.-]*(?::[A-Za-z_][\w.-]*)?)\s*\(")
_BARE_STEP = re.compile(r"^(?:child::)?(?:[A-Za-z][\w-]*|\*)(?:\[\s*\d+\s*\])*$")
_HAS_INDEX = re.compile(r"\[\s*\d+\s*\]")
_COMMENT_TEST = re.compile(r"comment\s*\(")


def _strip_strings(xpath: str) -> str:
    return _STRING.sub("''", xpath)


def split_steps(xpath: str) -> list[str]:
    """Split a location path on ``/`` outside brackets, parentheses and string literals."""
    steps: list[str] = []
    depth = 0
    buf: list[str] = []
    quote: str | None = None
    for ch in xpath:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "/" and depth == 0:
            steps.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    steps.append("".join(buf).strip())
    return steps


def is_positional_only(xpath: str) -> bool:
    """True when every step is a bare tag and the path is rooted, indexed, or ends generically."""
    expr = xpath.strip()
    if expr.startswith("(") or "|" in _strip_strings(expr):
        return False
    steps = [s for s in split_steps(expr) if s not in ("", ".")]
    if not steps or not all(_BARE_STEP.match(s) for s in steps):
        return False
    rooted = expr.startswith("/") and not expr.startswith("//")
    indexed = any(_HAS_INDEX.search(s) for s in steps)
    last_tag = re.sub(r"\[.*", "", steps[-1]).removeprefix("child::").lower()
    return rooted or indexed or last_tag in GENERIC_TAGS


@lru_cache(maxsize=65536)
def _lint_cached(xpath: str, relative: bool, allow_positional: bool) -> tuple[str, ...]:
    errors: list[str] = []
    expr = xpath.strip()
    if not expr:
        return ("empty selector",)
    try:
        etree.XPath(expr)
    except etree.XPathSyntaxError as exc:
        return (f"does not compile as XPath 1.0: {exc}",)
    bare = _strip_strings(expr)
    if "data-x-" in expr:
        errors.append("references a data-x- attribute (exists only in the annotated union page)")
    if _COMMENT_TEST.search(bare):
        errors.append("uses comment() which is not allowed")
    for name in _FUNC.findall(bare):
        low = name.lower()
        if low in BANNED_FUNCTIONS:
            if low != "comment":
                errors.append(f"uses {name}() which is not allowed")
        elif low not in XPATH1_FUNCTIONS and low not in NODE_TESTS:
            errors.append(f"uses non-XPath-1.0 function {name}()")
    if relative:
        if not expr.startswith("."):
            errors.append("relative selector must start with '.'")
    elif not (expr.startswith("/") or expr.startswith("(")):
        errors.append("absolute selector must start with '/' or '('")
    if not allow_positional and is_positional_only(expr):
        errors.append(
            "purely positional path; anchor it on id, class, itemprop, data-*, or label text"
        )
    return tuple(errors)


def lint_selector(
    xpath: str, *, relative: bool = False, allow_positional: bool = False
) -> list[str]:
    """Return a list of human-readable problems (empty when the selector is acceptable)."""
    return list(_lint_cached(xpath, relative, allow_positional))
