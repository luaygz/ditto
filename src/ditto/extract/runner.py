"""Run one spec over one cleaned page. No LLM, no I/O: this is what ``Spec.apply`` calls.

What each field type extracts:

* ``scalar``: evaluate ``candidates[selected]``, apply ``transforms`` to each match in document
  order and keep the first non-null value. When the selected candidate yields nothing, try the
  ``fallbacks`` indices in order. Nothing found gives ``null``.
* ``list``: every non-null transformed match of the selected candidate, in document order; when
  that is empty, the first fallback that yields a non-empty list. Nothing found gives ``[]``.
* ``record_list``: every element matched by ``container`` is a row; each column XPath runs
  relative to the row and is reduced like a scalar (first non-null value after the column's
  transforms). Rows whose columns are all null are skipped. Nothing found gives ``[]``.
* ``key_value``: every element matched by ``pairs`` yields a key (``key``, default ``.``) and a
  value (``value``), each relative to the pair element and reduced like a scalar with its
  transforms. Keys are whitespace-normalized with a trailing colon removed; pairs without a
  key are skipped; the first occurrence of a key wins (its value may be null). Nothing found
  gives ``{}``.
* ``container_alternates``/``pairs_alternates`` are candidates for verification; extraction
  uses ``container``/``pairs`` only.

Values are strings (elements read as whitespace-normalized text, ``text()``/``@attr`` results
trimmed), numbers only from ``cast``. Selectors are compiled once per process; a selector that
fails at evaluation time yields no matches. ``normalize-space`` treats U+00A0 (no-break space)
as a space, like the rest of ditto. A field is *missing* on a page when its value is ``null``,
``[]`` or ``{}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from lxml import etree

from ditto.fingerprint.dom import clean_etree
from ditto.specgen.transforms import apply_transforms
from ditto.textutil import normalize_ws

if TYPE_CHECKING:
    from ditto.specgen.models import (
        KeyValueField,
        ListField,
        RecordListField,
        ScalarField,
        Spec,
        SpecField,
    )

# Speed: libxml2 only turns ``//name`` into one descendant scan when the step has no predicate;
# ``//div[@id='x']`` visits every node and evaluates the child axis under each one.
# ``fast_xpath`` rewrites such steps to ``/descendant::div[@id='x']`` when every predicate
# provably filters node by node (a boolean or node-set test that never reads ``position()`` or
# ``last()``), which selects the same nodes in the same order at a fraction of the cost.

# XPath 1.0 tokens. Names may contain '-' and '.'; a QName's local part may be '*'.
_XPATH_TOKEN = re.compile(
    r"""\s*(?:
      (?P<lit>"[^"]*"|'[^']*')
    | (?P<num>\d+(?:\.\d*)?|\.\d+)
    | (?P<op>//|::|\.\.|!=|<=|>=|[/()\[\]@,|+\-=<>*.$])
    | (?P<name>[A-Za-z_][\w.\-]*(?::(?:[A-Za-z_][\w.\-]*|\*))?)
    )""",
    re.VERBOSE,
)
# Tokens after which a name is a name and '*' a name test (XPath 1.0 section 3.7); after any
# other token they are the operators and/or/div/mod and multiplication.
_OPERAND_START = frozenset(
    {"@", "::", "(", "[", ",", "/", "//", "|", "+", "-", "=", "!=", "<", "<=", ">", ">=", "$"}
)
_OPERATOR_NAMES = frozenset({"and", "or", "div", "mod"})
_BOOLEAN_OPERATORS = frozenset({"and", "or", "=", "!=", "<", "<=", ">", ">="})
_NODE_TYPES = frozenset({"node", "text", "comment", "processing-instruction"})
# Core functions whose result is a string or boolean, never a number.
_NON_NUMERIC_FUNCTIONS = frozenset(
    {
        "boolean",
        "not",
        "true",
        "false",
        "lang",
        "contains",
        "starts-with",
        "string",
        "concat",
        "substring",
        "substring-before",
        "substring-after",
        "normalize-space",
        "translate",
        "local-name",
        "name",
        "namespace-uri",
    }
)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # lit | num | op | name | operator (and/or/div/mod, multiplication '*')
    text: str
    start: int
    end: int


def _tokenize(expr: str) -> list[_Token] | None:
    """XPath 1.0 tokens of ``expr``, or None when it does not tokenize."""
    tokens: list[_Token] = []
    pos, n = 0, len(expr)
    while pos < n and not expr[pos:].isspace():
        m = _XPATH_TOKEN.match(expr, pos)
        if m is None:
            return None
        group = m.lastgroup or ""
        kind, text, start = group, m.group(group), m.start(group)
        if kind == "name" or text == "*":
            prev = tokens[-1] if tokens else None
            after_operand = prev is not None and not (
                prev.kind == "operator" or (prev.kind == "op" and prev.text in _OPERAND_START)
            )
            if after_operand:
                if text == "*" or text in _OPERATOR_NAMES:
                    kind = "operator"
                else:
                    return None  # a name right after an operand: not XPath
        tokens.append(_Token(kind, text, start, m.end()))
        pos = m.end()
    return tokens


def _closing(tokens: list[_Token], i: int) -> int:
    """Index of the token closing the bracket or parenthesis opened at ``i``, or -1."""
    depth = 0
    for j in range(i, len(tokens)):
        t = tokens[j]
        if t.kind == "op" and t.text in ("(", "["):
            depth += 1
        elif t.kind == "op" and t.text in (")", "]"):
            depth -= 1
            if depth == 0:
                return j
    return -1


def _node_test_end(tokens: list[_Token], i: int) -> int:
    """End of the name test or node-type test starting at ``i`` (child axis only), or -1."""
    if i >= len(tokens):
        return -1
    t = tokens[i]
    if t.kind == "op" and t.text == "*":
        return i + 1
    if t.kind != "name":
        return -1
    nxt = tokens[i + 1].text if i + 1 < len(tokens) else ""
    if nxt == "::":
        return -1
    if nxt == "(":
        if t.text not in _NODE_TYPES:
            return -1
        close = _closing(tokens, i + 1)
        return -1 if close < 0 else close + 1
    return i + 1


def _skip_predicates(tokens: list[_Token], i: int) -> int:
    while i < len(tokens) and tokens[i].kind == "op" and tokens[i].text == "[":
        close = _closing(tokens, i)
        if close < 0:
            return -1
        i = close + 1
    return i


def _is_location_path(tokens: list[_Token]) -> bool:
    """Whether ``tokens`` are a location path (or a union of paths): a node-set, whatever its
    predicates contain."""
    i, n = 0, len(tokens)
    if n and tokens[0].kind == "op" and tokens[0].text in ("/", "//"):
        i = 1
        if i == n:
            return tokens[0].text == "/"
    while True:
        if i >= n:
            return False
        t = tokens[i]
        if t.kind == "op" and t.text in (".", ".."):
            i += 1
        else:
            if t.kind == "op" and t.text == "@":
                i += 1
            elif t.kind == "name" and i + 1 < n and tokens[i + 1].text == "::":
                i += 2
            i = _node_test_end(tokens, i)
            if i < 0:
                return False
            i = _skip_predicates(tokens, i)
            if i < 0:
                return False
        if i == n:
            return True
        t = tokens[i]
        if not (t.kind == "op" and t.text in ("/", "//", "|")):
            return False
        i += 1
        if t.text == "|" and i < n and tokens[i].kind == "op" and tokens[i].text in ("/", "//"):
            i += 1


def _filters_node_by_node(pred: list[_Token]) -> bool:
    """Whether a predicate's value depends only on the node it tests: it never reads the
    context position or size and is never a number (a number means "position = n")."""
    if not pred:
        return False
    for k, t in enumerate(pred):
        if t.kind == "op" and t.text == "$":
            return False
        if (
            t.kind == "name"
            and t.text in ("position", "last")
            and k + 1 < len(pred)
            and pred[k + 1].text == "("
        ):
            return False
    depth = 0
    for t in pred:
        if t.kind == "op" and t.text in (")", "]"):
            depth -= 1
        elif t.kind == "op" and t.text in ("(", "["):
            depth += 1
        elif depth == 0 and t.kind in ("op", "operator") and t.text in _BOOLEAN_OPERATORS:
            return True  # the lowest-precedence operator at top level yields a boolean
    first = pred[0]
    if (
        first.kind == "name"
        and first.text not in _NODE_TYPES
        and len(pred) > 1
        and pred[1].text == "("
        and _closing(pred, 1) == len(pred) - 1
    ):
        return first.text in _NON_NUMERIC_FUNCTIONS
    return _is_location_path(pred)


# ``contains(concat(' ', normalize-space(@class), ' '), ' TOKEN ')``, token by token.
_CLASS_TEST = (
    ("name", "contains"),
    ("op", "("),
    ("name", "concat"),
    ("op", "("),
    ("lit", " "),
    ("op", ","),
    ("name", "normalize-space"),
    ("op", "("),
    ("op", "@"),
    ("name", "class"),
    ("op", ")"),
    ("op", ","),
    ("lit", " "),
    ("op", ")"),
    ("op", ","),
)
_XML_SPACE = frozenset(" \t\r\n")


def _class_token(pred: list[_Token]) -> str | None:
    """The literal ``' TOKEN '`` (with its quotes) when ``pred`` is exactly the class-token
    test with a non-empty, whitespace-free token; else None."""
    if len(pred) != len(_CLASS_TEST) + 2 or pred[-1].text != ")":
        return None
    for t, (kind, text) in zip(pred, _CLASS_TEST, strict=False):
        if t.kind != kind or (t.text[1:-1] if kind == "lit" else t.text) != text:
            return None
    lit = pred[-2]
    inner = lit.text[1:-1]
    if lit.kind != "lit" or len(inner) < 3 or inner[0] != " " or inner[-1] != " ":
        return None
    token = inner[1:-1]
    if _XML_SPACE.intersection(token):
        return None
    return lit.text[0] + token + lit.text[0]


@lru_cache(maxsize=16384)
def fast_xpath(expr: str) -> str:
    """``expr`` rewritten to select the same nodes in the same order, faster:

    * every ``//step[p]...`` whose predicates all filter node by node becomes
      ``/descendant::step[p]...`` (every descendant of the context is a child of a node in its
      descendant-or-self set), so libxml2 no longer evaluates the child axis under every node;
    * every class-token predicate ``[contains(concat(' ', normalize-space(@class), ' '),
      ' x ')]`` gets the cheap ``[contains(@class, 'x')]`` in front of it: it holds whenever
      the token test does (the token has no whitespace), so the pair keeps exactly the nodes,
      and the positions later predicates see, of the token test alone.

    Anything that does not tokenize is returned unchanged."""
    tokens = _tokenize(expr)
    if tokens is None:
        return expr
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement) in ``expr``
    for i, t in enumerate(tokens):
        if t.kind != "op":
            continue
        if t.text == "[":
            close = _closing(tokens, i)
            token = _class_token(tokens[i + 1 : close]) if close > 0 else None
            if token is not None:
                edits.append((t.start, t.start, f"[contains(@class,{token})]"))
            continue
        if t.text != "//":
            continue
        j = _node_test_end(tokens, i + 1)
        if j < 0 or j >= len(tokens) or tokens[j].text != "[":
            continue  # no predicate: libxml2 already scans descendants once
        ok = True
        while ok and j < len(tokens) and tokens[j].kind == "op" and tokens[j].text == "[":
            close = _closing(tokens, j)
            ok = close > 0 and _filters_node_by_node(tokens[j + 1 : close])
            j = close + 1
        if ok:
            edits.append((t.start, t.end, "/descendant::"))
    if not edits:
        return expr
    parts: list[str] = []
    last = 0
    for start, end, text in sorted(edits):
        parts.append(expr[last:start])
        parts.append(text)
        last = end
    parts.append(expr[last:])
    return "".join(parts)


# XPath's ``normalize-space`` collapses ASCII whitespace only, but the text the LLM reads (and
# every value ditto extracts) treats U+00A0 as a space: a label written ``Type:&nbsp;&nbsp;``
# must still equal ``'Type:'``. Every ``normalize-space`` call therefore reads its argument
# with no-break spaces translated to spaces.
_NBSP_TO_SPACE = ",'\u00a0',' ')"


@lru_cache(maxsize=16384)
def nbsp_as_space(expr: str) -> str:
    """``expr`` with every ``normalize-space(x)`` call outside string literals turned into
    ``normalize-space(translate(x,'\\u00a0',' '))`` (``normalize-space()``: ``x`` is ``.``).
    On text without U+00A0 it selects exactly what ``expr`` does. Anything that does not
    tokenize is returned unchanged."""
    tokens = _tokenize(expr)
    if tokens is None:
        return expr
    edits: list[tuple[int, str]] = []  # (position in ``expr``, inserted text)
    for i, t in enumerate(tokens[:-1]):
        opener = tokens[i + 1]
        if t.kind != "name" or t.text != "normalize-space" or opener.text != "(":
            continue
        close = _closing(tokens, i + 1)
        if close < 0 or tokens[close].text != ")":
            return expr
        if close == i + 2:
            edits.append((opener.end, "translate(." + _NBSP_TO_SPACE))
        else:
            edits += [(opener.end, "translate("), (tokens[close].start, _NBSP_TO_SPACE)]
    if not edits:
        return expr
    parts: list[str] = []
    last = 0
    for pos, text in sorted(edits):
        parts += [expr[last:pos], text]
        last = pos
    parts.append(expr[last:])
    return "".join(parts)


@lru_cache(maxsize=16384)
def compile_xpath(expr: str) -> etree.XPath:
    """Compiled XPath (via :func:`fast_xpath` and :func:`nbsp_as_space`), cached per process.
    String results stay lxml "smart" strings, so callers such as the review checkpoint can
    reach the owning element via ``getparent()``. EXSLT regular expressions are not
    registered: the lint rejects them anyway. A selector that does not compile raises as
    written."""
    compiled = etree.XPath(expr, regexp=False)
    rewritten = nbsp_as_space(fast_xpath(expr))
    if rewritten == expr:
        return compiled
    try:
        return etree.XPath(rewritten, regexp=False)
    except etree.XPathError:  # pragma: no cover - the rewrites keep valid syntax valid
        return compiled


def evaluate(expr: str, node: etree._Element) -> list[Any]:
    """Evaluate an XPath and always return a list of result items (elements or strings;
    a number or ``true`` for scalar XPath expressions)."""
    try:
        result = compile_xpath(expr)(node)
    except etree.XPathError:
        return []
    if isinstance(result, list):
        return [r for r in result if not isinstance(r, etree._Comment)]
    if isinstance(result, bool):
        return [result] if result else []
    if isinstance(result, float):
        return [] if result != result else [result]  # NaN -> nothing
    if isinstance(result, str):
        return [result] if result.strip() else []
    return [result]  # pragma: no cover - lxml returns only the types above


# Speed: most generated selectors start at an element with an id (``//div[@id='main']//h1``),
# and each one scans the whole page for that anchor. ``apply_spec`` finds the anchors of all its
# selectors in a single scan (``Anchors``) and evaluates each selector's rest from its anchor:
# ``//tag[@id='x']<rest>`` selects ``<rest>`` evaluated from each such element, so with one
# anchor it is ``.<rest>`` from that anchor and with none it is empty. A selector whose anchor
# matches several elements is evaluated as written.


@dataclass(frozen=True, slots=True)
class _Anchor:
    tag: str  # element name, or "*"
    id: str
    rest: str | None  # the path after the anchor, relative to it ("./h1"); None: the anchor


_ANCHOR_SHAPE = (("op", "["), ("op", "@"), ("name", "id"), ("op", "="))


def _top_level_union(tokens: list[_Token]) -> bool:
    depth = 0
    for t in tokens:
        if t.kind == "op" and t.text in ("(", "["):
            depth += 1
        elif t.kind == "op" and t.text in (")", "]"):
            depth -= 1
        elif depth == 0 and t.kind == "op" and t.text == "|":
            return True
    return False


@lru_cache(maxsize=16384)
def _id_anchor(expr: str) -> _Anchor | None:
    """``//tag[@id='x']<rest>`` split at its anchor, when ``<rest>`` is empty or the steps of
    one location path (no union or operator, no other predicate on the anchor); else None."""
    tokens = _tokenize(expr)
    if tokens is None or len(tokens) < 8 or (tokens[0].kind, tokens[0].text) != ("op", "//"):
        return None
    tag = tokens[1]
    simple_name = tag.kind == "name" and ":" not in tag.text
    if not (simple_name or (tag.kind, tag.text) == ("op", "*")):
        return None
    shape = tuple((t.kind, t.text) for t in tokens[2:6])
    if shape != _ANCHOR_SHAPE or tokens[6].kind != "lit" or tokens[7].text != "]":
        return None
    ident = tokens[6].text[1:-1]
    if len(tokens) == 8:
        return _Anchor(tag.text, ident, None)
    if tokens[8].kind != "op" or tokens[8].text not in ("/", "//"):
        return None
    rest = "." + expr[tokens[8].start :]
    rest_tokens = _tokenize(rest)
    if rest_tokens is None or _top_level_union(rest_tokens):
        return None
    return _Anchor(tag.text, ident, rest) if _is_location_path(rest_tokens) else None


# Every ``id`` attribute of the page, in document order: one scan, cheaper than a scan per
# anchor (and far cheaper than one predicate testing every id the spec anchors on).
_ID_ATTRIBUTES = etree.XPath("/descendant::*/@id")


class Anchors:
    """The elements of one page that a spec's id-anchored selectors start from, found in one
    scan on first use. The tree must not change while this is in use."""

    __slots__ = ("_found", "_root", "ids")

    def __init__(self, root: etree._Element, ids: frozenset[str]) -> None:
        self._root = root
        self.ids = ids
        self._found: dict[str, list[etree._Element]] | None = None

    @classmethod
    def for_spec(cls, spec: Spec, root: etree._Element) -> Anchors:
        """Anchors for every selector of ``spec`` on ``root``'s page."""
        exprs: list[str] = []
        for field in spec.fields.values():
            if field.type == "record_list":
                exprs += [field.container, *field.columns.values()]
            elif field.type == "key_value":
                exprs += [field.pairs, field.key, field.value]
            else:
                exprs += field.candidates
        anchors = (_id_anchor(e) for e in exprs)
        return cls(root, frozenset(a.id for a in anchors if a is not None))

    def elements(self, anchor: _Anchor) -> list[etree._Element]:
        if self._found is None:
            found: dict[str, list[etree._Element]] = {}
            ids = self.ids
            for value in cast("list[Any]", _ID_ATTRIBUTES(self._root) if ids else []):
                if value in ids:  # a "smart" string: its parent is the element
                    found.setdefault(str(value), []).append(value.getparent())
            self._found = found
        elements = self._found.get(anchor.id, [])
        return [e for e in elements if anchor.tag == "*" or e.tag == anchor.tag]


def _evaluate(expr: str, node: etree._Element, anchors: Anchors | None) -> list[Any]:
    """:func:`evaluate`, starting id-anchored selectors from ``anchors`` (same result)."""
    if anchors is not None:
        anchor = _id_anchor(expr)
        if anchor is not None and anchor.id in anchors.ids:
            found = anchors.elements(anchor)
            if not found:
                return []
            if len(found) == 1:
                return found if anchor.rest is None else evaluate(anchor.rest, found[0])
    return evaluate(expr, node)


def _first_value(items: list[Any], transforms: list[str]) -> Any:
    for item in items:
        value = apply_transforms(item, transforms)
        if value is not None:
            return value
    return None


def candidate_values(
    expr: str, node: etree._Element, transforms: list[str], *, anchors: Anchors | None = None
) -> list[Any]:
    """Every non-null transformed value ``expr`` yields on ``node``, in document order."""
    values = (apply_transforms(item, transforms) for item in _evaluate(expr, node, anchors))
    return [v for v in values if v is not None]


def _order(field: ScalarField | ListField) -> list[int]:
    return [field.selected, *(i for i in field.fallbacks if i != field.selected)]


def scalar_value(
    field: ScalarField, root: etree._Element, *, anchors: Anchors | None = None
) -> Any:
    for idx in _order(field):
        items = _evaluate(field.candidates[idx], root, anchors)
        value = _first_value(items, field.transforms)
        if value is not None:
            return value
    return None


def list_value(
    field: ListField, root: etree._Element, *, anchors: Anchors | None = None
) -> list[Any]:
    for idx in _order(field):
        kept = candidate_values(field.candidates[idx], root, field.transforms, anchors=anchors)
        if kept:
            return kept
    return []


# The brief's header-text column, ``./td[count(ancestor::table//th[<header test>]
# /preceding-sibling::th)+1]<rest>``. Evaluated literally, the predicate rescans the whole table
# for every cell of every row (quadratic in table size); its value only depends on the row's
# ancestors, so it is computed once per row parent and the cell is picked by position.
_HEADER_COLUMN = re.compile(
    r"^\s*\./(?P<cell>td|th)\[\s*(?P<index>count\(\s*ancestor::table(?:\[1\])?\s*//\s*th"
    r"\[[^\[\]]*\]\s*/\s*preceding-sibling::th\s*\)\s*\+\s*1)\s*\](?P<rest>(?:/[^|]*)?)\s*$"
)


@dataclass(frozen=True)
class _HeaderColumn:
    cell: str  # "td" or "th"
    index: str  # XPath number expression, evaluated on the row
    rest: str | None  # remaining path relative to the cell ("./a/@href"), None for the cell


@lru_cache(maxsize=4096)
def _header_column(expr: str) -> _HeaderColumn | None:
    m = _HEADER_COLUMN.match(expr)
    if not m:
        return None
    rest = m.group("rest")
    return _HeaderColumn(m.group("cell"), m.group("index"), "." + rest if rest else None)


def _header_cells(
    plan: _HeaderColumn, row: etree._Element, positions: dict[tuple[Any, str], int]
) -> list[Any]:
    """Exactly what the column XPath returns on ``row``. The index expression is evaluated
    from a cell, as in the literal XPath; cells of rows sharing a parent have the same
    ancestors, so the result is cached per parent (per row when the row is itself a table)."""
    cells = [c for c in row if c.tag == plan.cell]
    if not cells:
        return []
    key = (row if row.tag == "table" else row.getparent(), plan.index)
    pos = positions.get(key)
    if pos is None:
        found = evaluate(plan.index, cells[0])
        pos = int(found[0]) if found and float(found[0]).is_integer() else 0
        positions[key] = pos
    if not 1 <= pos <= len(cells):
        return []
    cell = cells[pos - 1]
    return [cell] if plan.rest is None else evaluate(plan.rest, cell)


def record_rows(
    field: RecordListField, root: etree._Element, *, anchors: Anchors | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = [
        (name, xpath, field.transforms.get(name, []), _header_column(xpath))
        for name, xpath in field.columns.items()
    ]
    positions: dict[tuple[Any, str], int] = {}  # header position per (row parent, column)
    for row in _evaluate(field.container, root, anchors):
        if not isinstance(row, etree._Element):
            continue
        record = {
            name: _first_value(
                _evaluate(xpath, row, anchors)
                if plan is None
                else _header_cells(plan, row, positions),
                chain,
            )
            for name, xpath, chain, plan in columns
        }
        if any(v is not None for v in record.values()):
            rows.append(record)
    return rows


def clean_key(key: Any) -> str:
    """``"  Email :"`` -> ``"Email"`` (whitespace normalized, trailing colons removed)."""
    return normalize_ws(str(key)).rstrip(":：").rstrip()


def key_values(
    field: KeyValueField, root: etree._Element, *, anchors: Anchors | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key_t = field.transforms.get("key", [])
    value_t = field.transforms.get("value", [])
    for pair in _evaluate(field.pairs, root, anchors):
        if not isinstance(pair, etree._Element):
            continue
        raw = _first_value(_evaluate(field.key, pair, anchors), key_t)
        key = clean_key(raw) if raw is not None else ""
        if key and key not in out:
            out[key] = _first_value(_evaluate(field.value, pair, anchors), value_t)
    return out


def field_value(field: SpecField, root: etree._Element, *, anchors: Anchors | None = None) -> Any:
    """The value of ``field`` on ``root``. ``anchors`` (for ``root``'s tree, unchanged since)
    lets id-anchored selectors skip their own scan of the page; the value is the same."""
    if field.type == "scalar":
        return scalar_value(field, root, anchors=anchors)
    if field.type == "list":
        return list_value(field, root, anchors=anchors)
    if field.type == "record_list":
        return record_rows(field, root, anchors=anchors)
    return key_values(field, root, anchors=anchors)


def is_missing(value: Any) -> bool:
    return value is None or value == [] or value == {}


def apply_spec(spec: Spec, root: etree._Element) -> dict[str, Any]:
    """Apply every field of ``spec`` to an already-cleaned tree."""
    out: dict[str, Any] = {}
    missing: list[str] = []
    anchors = Anchors.for_spec(spec, root)
    for name, field in spec.fields.items():
        value = field_value(field, root, anchors=anchors)
        out[name] = value
        if is_missing(value):
            missing.append(name)
    out["_meta"] = {"missing": missing, "spec_version": spec.spec_version}
    return out


def apply_spec_html(spec: Spec, html: str | bytes, encoding: str | None = None) -> dict[str, Any]:
    """Clean ``html`` exactly as the pipeline does (bytes decoded with ``encoding``, or sniffed
    by lxml without one), then :func:`apply_spec`. The tree uses plain lxml elements: values
    are the same as on ``clean_html`` output, which only differs in its Python element class."""
    return apply_spec(spec, clean_etree(html, encoding))
