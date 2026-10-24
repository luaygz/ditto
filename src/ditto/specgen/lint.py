"""XPath 1.0 selector lint enforced at spec validation time.

Rules (each violation yields one message):

* the selector compiles with ``lxml.etree.XPath``;
* only the XPath 1.0 core function library: ``matches()``, ``lower-case()``, ``upper-case()``
  and ``ends-with()`` are named explicitly in the error; any other function (XPath 2.0, XSLT,
  EXSLT), any namespace prefix and any ``$variable`` is rejected too (lxml compiles some of these
  and only fails when the selector runs);
* no ``comment()`` node test and no ``data-x-`` anywhere: both exist only in the annotated
  union page shown to the LLM, never in the pages the spec runs on;
* absolute selectors are location paths starting with ``/``, optionally parenthesized as in
  ``(//h1)[1]``; relative selectors (record columns, key/value parts) start with ``.``; every
  branch of a ``|`` union obeys the same rule;
* no purely positional paths. After ignoring attribute, ``text()``/``node()`` and ``.``/``..``
  steps, a selector whose every step is a bare tag (any axis, or ``*``) with at most positional
  predicates (``[2]``, ``[last()]``, ``[position()=2]``) is unanchored. An unanchored absolute
  selector is rejected when it is rooted at ``/``, carries a positional predicate, or ends on a
  generic tag (``div``, ``span``, ``td``, ...); descendant-anchored semantic tags such as
  ``//header//h1`` pass. An unanchored relative selector (anchored by its row) is rejected when
  it carries a positional predicate or steps through a cell or wildcard (``td``, ``th``, ``*``):
  columns are addressed by header text, so ``./td[2]`` fails while ``.//a/@href`` passes.
  Anchors are ``@id``, classes, ``data-*``, ``itemprop`` and other attributes, text predicates
  such as ``normalize-space()='Email'``, or structural proximity to an anchored step, so
  ``//dt[normalize-space()='Email']/following-sibling::dd[1]`` passes. Headerless tables and
  key/value parts may be positional (``allow_positional``).
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
        "html",
        "body",
        "div",
        "span",
        "p",
        "li",
        "ul",
        "ol",
        "td",
        "tr",
        "th",
        "thead",
        "tbody",
        "tfoot",
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
        "u",
        "small",
        "font",
        "center",
        "*",
    }
)
AXES = frozenset(
    {
        "ancestor",
        "ancestor-or-self",
        "attribute",
        "child",
        "descendant",
        "descendant-or-self",
        "following",
        "following-sibling",
        "namespace",
        "parent",
        "preceding",
        "preceding-sibling",
        "self",
    }
)
OPERATOR_NAMES = frozenset({"and", "or", "div", "mod"})
_CELL_TAGS = frozenset({"td", "th", "*"})

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<str>"[^"]*"|'[^']*')
      | (?P<num>\d+(?:\.\d*)?|\.\d+)
      | (?P<op>//|::|\.\.|!=|<=|>=|[/()\[\]@,|+\-=<>*.$])
      | (?P<name>[^\W\d][\w.\-]*(?::(?:[^\W\d][\w.\-]*|\*))?)
    )""",
    re.VERBOSE,
)
# After these tokens a name is a name test / function name, not an operator (XPath 1.0 §3.7).
_OPERAND_FOLLOWS = frozenset(
    {"@", "::", "(", "[", ",", "/", "//", "|", "+", "-", "=", "!=", "<", "<=", ">", ">=", "$"}
)
_AXIS = "(?:" + "|".join(sorted(AXES, key=len, reverse=True)) + r")\s*::\s*"
_BARE_HEAD = re.compile(rf"^(?:{_AXIS})?(?:[^\W\d][\w.-]*|\*)$")
_NON_ELEMENT_HEAD = re.compile(
    rf"^(?:\.|\.\.|@.*|attribute\s*::.*|(?:{_AXIS})?"
    r"(?:text|node|processing-instruction)\s*\(.*\))$",
    re.DOTALL,
)
_POSITIONAL_PRED = re.compile(
    r"^\s*(?:\d+|last\(\s*\)(?:\s*-\s*\d+)?"
    r"|position\(\s*\)\s*(?:[<>]=?|!?=)\s*(?:\d+|last\(\s*\)(?:\s*-\s*\d+)?)"
    r"|\d+\s*(?:[<>]=?|!?=)\s*position\(\s*\))\s*$"
)


def _tokens(xpath: str) -> list[tuple[str, str]]:
    """XPath 1.0 tokens of an expression lxml already compiled."""
    return [(m.lastgroup or "", m.group(m.lastgroup or 0)) for m in _TOKEN.finditer(xpath)]


def _name_problems(xpath: str) -> list[str]:
    """Functions, node types, prefixes and variables that XPath 1.0 over HTML does not allow."""
    toks = _tokens(xpath)
    problems: list[str] = []
    expect_operand = True
    for i, (kind, val) in enumerate(toks):
        nxt = toks[i + 1][1] if i + 1 < len(toks) else ""
        if kind == "name":
            if not expect_operand and val in OPERATOR_NAMES:
                expect_operand = True
                continue
            prev = toks[i - 1][1] if i else ""
            if prev == "$":
                problems.append(f"uses variable ${val}")
            elif nxt == "(":
                low = val.lower()
                if low == "comment":
                    problems.append("uses comment() which is not allowed")
                elif low in BANNED_FUNCTIONS:
                    problems.append(f"uses {val}() which is not XPath 1.0")
                elif ":" in val:
                    problems.append(f"uses prefixed function {val}() (EXSLT/XPath 2.0)")
                elif val not in XPATH1_FUNCTIONS and val not in NODE_TESTS:
                    problems.append(f"uses non-XPath-1.0 function {val}()")
            elif ":" in val and nxt != "::":
                problems.append(f"uses namespace prefix in {val!r} (HTML has no namespaces)")
            expect_operand = False  # a following "(" or "::" token sets it again
        elif kind == "op" and val == "*":
            expect_operand = not expect_operand  # name test ends an operand; multiply does not
        else:
            expect_operand = kind == "op" and val in _OPERAND_FOLLOWS
    return problems


def _split_top(expr: str, sep: str) -> list[str]:
    """Split on ``sep`` outside brackets, parentheses and string literals."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    quote: str | None = None
    for ch in expr:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def split_steps(xpath: str) -> list[str]:
    """Split a location path on ``/`` outside brackets, parentheses and string literals."""
    return _split_top(xpath, "/")


def _closing_paren(expr: str) -> int | None:
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(expr):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None  # pragma: no cover - lxml compiled the expression, so parentheses balance


def location_paths(xpath: str) -> list[str]:
    """Union branches with grouping parentheses folded away: ``(//a | //b)[1]`` becomes
    ``["//a[1]", "//b[1]"]``. An approximation used only for lint checks."""
    expr = xpath.strip()
    branches = _split_top(expr, "|")
    if len(branches) > 1:
        return [p for b in branches for p in location_paths(b)]
    if expr.startswith("("):
        close = _closing_paren(expr)
        if close is not None:
            return [p + expr[close + 1 :].strip() for p in location_paths(expr[1:close])]
    return [expr]


def _step_parts(step: str) -> tuple[str, list[str]]:
    """``td[@class='x'][2]`` -> ``("td", ["@class='x'", "2"])``."""
    head_end = len(step)
    preds: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    for i, ch in enumerate(step):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "[":
            if depth == 0:
                head_end = min(head_end, i)
                start = i + 1
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                preds.append(step[start:i])
    return step[:head_end].strip(), preds


def _path_positional(path: str) -> bool:
    elements: list[tuple[str, bool]] = []
    for step in split_steps(path):
        if not step:
            continue
        head, preds = _step_parts(step)
        if _NON_ELEMENT_HEAD.match(head):
            continue
        if not _BARE_HEAD.match(head) or not all(_POSITIONAL_PRED.match(p) for p in preds):
            return False  # anchored (attribute/text predicate, function, ...)
        elements.append((head, bool(preds)))
    if not elements:
        return False
    indexed = any(has_index for _, has_index in elements)
    tags = [head.split("::")[-1].strip().lower() for head, _ in elements]
    if path.startswith("."):
        return indexed or any(tag in _CELL_TAGS for tag in tags)
    rooted = path.startswith("/") and not path.startswith("//")
    return rooted or indexed or tags[-1] in GENERIC_TAGS


def is_positional_only(xpath: str) -> bool:
    """True when some union branch is unanchored and positional (see module docstring)."""
    return any(_path_positional(p) for p in location_paths(xpath))


@lru_cache(maxsize=65536)
def _lint_cached(xpath: str, relative: bool, allow_positional: bool) -> tuple[str, ...]:
    expr = xpath.strip()
    if not expr:
        return ("empty selector",)
    try:
        etree.XPath(expr)
    except etree.XPathError as exc:
        return (f"does not compile as XPath 1.0: {exc}",)
    errors: list[str] = []
    if "data-x-" in expr.lower():
        errors.append("references a data-x- attribute (exists only in the annotated union page)")
    errors.extend(dict.fromkeys(_name_problems(expr)))
    paths = location_paths(expr)
    if relative:
        if not all(p.startswith(".") for p in paths):
            errors.append("relative selector must start with '.'")
    elif not all(p.startswith("/") for p in paths):
        errors.append("absolute selector must be a location path starting with '/' or '('")
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
