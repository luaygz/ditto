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
fails at evaluation time yields no matches. A field is *missing* on a page when its value is
``null``, ``[]`` or ``{}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from lxml import etree

from ditto.fingerprint.dom import clean_html
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


@lru_cache(maxsize=16384)
def compile_xpath(expr: str) -> etree.XPath:
    """Compiled XPath, cached per process. String results stay lxml "smart" strings, so callers
    such as the review checkpoint can reach the owning element via ``getparent()``. EXSLT
    regular expressions are not registered: the lint rejects them anyway."""
    return etree.XPath(expr, regexp=False)


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


def _first_value(items: list[Any], transforms: list[str]) -> Any:
    for item in items:
        value = apply_transforms(item, transforms)
        if value is not None:
            return value
    return None


def candidate_values(expr: str, node: etree._Element, transforms: list[str]) -> list[Any]:
    """Every non-null transformed value ``expr`` yields on ``node``, in document order."""
    values = (apply_transforms(item, transforms) for item in evaluate(expr, node))
    return [v for v in values if v is not None]


def _order(field: ScalarField | ListField) -> list[int]:
    return [field.selected, *(i for i in field.fallbacks if i != field.selected)]


def scalar_value(field: ScalarField, root: etree._Element) -> Any:
    for idx in _order(field):
        value = _first_value(evaluate(field.candidates[idx], root), field.transforms)
        if value is not None:
            return value
    return None


def list_value(field: ListField, root: etree._Element) -> list[Any]:
    for idx in _order(field):
        kept = candidate_values(field.candidates[idx], root, field.transforms)
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


def record_rows(field: RecordListField, root: etree._Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = [
        (name, xpath, field.transforms.get(name, []), _header_column(xpath))
        for name, xpath in field.columns.items()
    ]
    positions: dict[tuple[Any, str], int] = {}  # header position per (row parent, column)
    for row in evaluate(field.container, root):
        if not isinstance(row, etree._Element):
            continue
        record = {
            name: _first_value(
                evaluate(xpath, row) if plan is None else _header_cells(plan, row, positions),
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


def key_values(field: KeyValueField, root: etree._Element) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key_t = field.transforms.get("key", [])
    value_t = field.transforms.get("value", [])
    for pair in evaluate(field.pairs, root):
        if not isinstance(pair, etree._Element):
            continue
        raw = _first_value(evaluate(field.key, pair), key_t)
        key = clean_key(raw) if raw is not None else ""
        if key and key not in out:
            out[key] = _first_value(evaluate(field.value, pair), value_t)
    return out


def field_value(field: SpecField, root: etree._Element) -> Any:
    if field.type == "scalar":
        return scalar_value(field, root)
    if field.type == "list":
        return list_value(field, root)
    if field.type == "record_list":
        return record_rows(field, root)
    return key_values(field, root)


def is_missing(value: Any) -> bool:
    return value is None or value == [] or value == {}


def apply_spec(spec: Spec, root: etree._Element) -> dict[str, Any]:
    """Apply every field of ``spec`` to an already-cleaned tree."""
    out: dict[str, Any] = {}
    missing: list[str] = []
    for name, field in spec.fields.items():
        value = field_value(field, root)
        out[name] = value
        if is_missing(value):
            missing.append(name)
    out["_meta"] = {"missing": missing, "spec_version": spec.spec_version}
    return out


def apply_spec_html(spec: Spec, html: str | bytes) -> dict[str, Any]:
    """Clean ``html`` exactly as the pipeline does, then :func:`apply_spec`."""
    return apply_spec(spec, clean_html(html))
