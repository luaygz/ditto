"""Find a template's candidates on one cleaned page (union page, probe, verification)."""

from __future__ import annotations

from lxml import etree

from ditto.analyze.models import CandidateTable
from ditto.analyze.walk import TABLE_KIND, StructPlan, is_split, resolve, walk

_SPLIT_LAYOUTS = frozenset({"dl", "alternating", "pairs", "table"})


def struct_plan(table: CandidateTable) -> StructPlan:
    """The accepted structures of a candidate table, keyed as :mod:`ditto.analyze.walk` keys
    them, mapped to the final table ids."""
    kv: dict[str, str] = {}
    tables: dict[str, str] = {}
    for t in table.tables:
        if t.layout == "dl":
            kv["dl|" + t.container_path] = t.id
        elif t.layout == "alternating":
            kv["alt|" + t.container_path] = t.id
        elif t.layout == "pairs":
            kv["pair|" + t.row_path] = t.id
        elif t.layout == "table":
            tables[TABLE_KIND + "|" + t.container_path] = t.id
    return StructPlan(kv=kv, tables=tables)


def locate_candidates(
    root: etree._Element, table: CandidateTable
) -> dict[str, list[etree._Element]]:
    """Candidate id -> elements on this page (``root`` from ``clean_html``), document order.

    A path candidate matches every node with its path. A key-value candidate (``table_role``
    ``kv_value``) matches nodes with its path whose row key text equals ``column``; a record
    column candidate matches nodes with its path inside cells whose header text (or ``colN``
    for headerless tables) equals ``column``. Attribute candidates map to the element carrying
    the attribute. Nodes of keys or columns the analysis never saw match nothing.
    """
    plan = struct_plan(table)
    split_layout = {t.id for t in table.tables if t.layout in _SPLIT_LAYOUTS}
    split: dict[tuple[str, str, str], str] = {}
    plain: dict[str, str] = {}
    for c in table.candidates:
        if (
            c.table_id in split_layout
            and c.table_role in ("kv_value", "record_column")
            and c.column is not None
            and c.table_id is not None
        ):
            split[(c.path, c.table_id, c.column)] = c.id
        else:
            plain.setdefault(c.path, c.id)
    out: dict[str, list[etree._Element]] = {}
    for item in walk(root).items:
        q = resolve(item, plan)
        if is_split(q) and q.table_id is not None and q.column is not None:
            cid = split.get((item.path, q.table_id, q.column))
        else:
            cid = plain.get(item.path)
        if cid is not None:
            out.setdefault(cid, []).append(item.element)
    return out
