"""One-pass page walk shared by cluster analysis and candidate location.

:func:`walk` yields exactly the text items of :func:`ditto.fingerprint.dom.iter_text_items` (same
paths, texts and document order) and additionally records, per item, the key-value cell and the
table cell it sits in. Analysis uses that to split one shared path such as ``dl.specs > dd`` into
one candidate per key, or ``table > tr > td`` into one candidate per column.

Structures detected on a single page (cross-page acceptance happens in analysis):

* ``dl``: children of ``<dl>``; every ``dd`` pairs with the nearest preceding ``dt``.
* ``alt``: a parent whose element children alternate between two step labels ``(A B+)+`` with at
  least two ``A`` (dl-like markup built from divs or spans).
* ``pair``: a row element with two text-bearing element children (key, value) and no own text,
  ignoring up to two text-less siblings such as spacer cells (``<tr><td><img></td><td>Phone</td>
  <td>555</td></tr>``); one leading key child followed by tail text (``<li><b>Brand:</b> Acme
  </li>``); or leading key text followed by one value child (``<li>Genre: <span>Drama</span>
  </li>``, the row's own text is the key).
* ``tbl``: a ``<table>``: an optional header row (a ``thead`` row, or a first row of at least two
  ``th``) over body rows; column indexes are expanded by ``colspan``.

Structure keys are ``"<kind>|<path>"`` where the path is the container (dl, alt, tbl) or the row
element (pair).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

from lxml import etree

from ditto.fingerprint.dom import ATTR_SEP, LINK_ATTRS, PATH_SEP, own_text, step_label
from ditto.textutil import normalize_ws

KEY_MAX_CHARS = 80
"""Longest text accepted as a header cell or a label."""
KV_KEY_MAX_CHARS = 50
"""Longest text accepted as a key-value key (``Price:``, ``Number of pages``); longer texts in
the key position are headlines of a widget, not keys."""
KEY_MAX_ELEMENTS = 6
"""A key cell spans at most this many elements (``<th><b>Email</b></th>``)."""
VALUE_PREFIX_CHARS = 200
"""Value cells are compared across pages by this many leading characters."""
VALUE_MAX_ELEMENTS = 60
"""A value or table cell with more elements is a page section (layout), not a cell."""
PAIR_MAX_CHILDREN = 4
"""A pair row holds at most this many element children; all but key and value are text-less."""
NO_STRUCT_TAGS = frozenset(
    {
        "html",
        "head",
        "body",
        "title",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "ul",
        "ol",
        "dl",
        "select",
    }
)
"""Elements never treated as a pair row or an alternating key/value parent."""
CELL_TAGS = frozenset({"td", "th"})
LABEL_LEVELS = 4
LABEL_SIBLINGS = 3
LABEL_MAX_ITEMS = 6
"""A sibling subtree with more text items than this is a section, not a label."""
SECTION_TAGS = frozenset(
    {"nav", "header", "footer", "aside", "form", "menu", "ul", "ol", "dl", "table", "select"}
    | {"section", "article", "main", "figure"}
)
"""Elements that are never a label (nor contain one) for a following node."""

TABLE_KIND = "tbl"


class KvRef(NamedTuple):
    """The key-value cell an item sits in: ``role`` is ``"k"`` (key cell) or ``"v"`` (value)."""

    struct: str
    key: str
    role: str
    depth: int


class CellRef(NamedTuple):
    """The table cell an item sits in: ``role`` is ``"h"`` (header cell) or ``"c"`` (body)."""

    struct: str
    col: int
    header: str | None
    role: str
    depth: int


class Item(NamedTuple):
    """One text-bearing node or ``href``/``src`` value (see ``iter_text_items``)."""

    path: str
    text: str
    element: etree._Element
    attr: str | None
    kv: KvRef | None
    cell: CellRef | None


class KvRow(NamedTuple):
    """One key-value row observed on a page."""

    struct: str
    row_path: str  # dl/alt: key element path; pair: row element path
    key_path: str
    key: str
    value: str  # value text prefix
    table: str | None  # enclosing table structure for two-cell table rows


class TableObs(NamedTuple):
    """One ``<table>`` observed on a page."""

    struct: str
    header: tuple[str, ...] | None
    row_path: str | None
    n_rows: int
    n_cols: int
    body: str  # body text prefix


@dataclass(slots=True)
class PageWalk:
    """Everything analysis and location need from one cleaned page."""

    items: list[Item] = field(default_factory=list)
    start: dict[etree._Element, int] = field(default_factory=dict)
    end: dict[etree._Element, int] = field(default_factory=dict)
    own: dict[etree._Element, int] = field(default_factory=dict)
    header_of: dict[etree._Element, etree._Element] = field(default_factory=dict)
    kv_rows: list[KvRow] = field(default_factory=list)
    tables: list[TableObs] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    """Element paths repeated as siblings -> number of such elements on the page."""


_PendKv = tuple[str, str, str]
_PendCell = tuple[str, int, str | None, str]


@dataclass(slots=True)
class _Ctx:
    """Mutable per-walk scratch state."""

    page: PageWalk
    pend_kv: dict[etree._Element, _PendKv] = field(default_factory=dict)
    pend_cell: dict[etree._Element, _PendCell] = field(default_factory=dict)
    tr_table: dict[etree._Element, str] = field(default_factory=dict)


def _tag(el: etree._Element) -> str:
    return str(el.tag).lower()


def text_prefix(el: etree._Element, limit: int) -> str:
    """Whitespace-normalized leading ``limit`` characters of the element's full text."""
    parts: list[str] = []
    seen = 0
    for chunk in el.itertext():
        parts.append(str(chunk))
        seen += len(chunk)
        if seen > 4 * limit:
            break
    return normalize_ws("".join(parts))[:limit]


# Size checks run in C (XPath) instead of iterating Python proxies.
_KEY_TOO_BIG = etree.XPath(f"boolean(descendant::*[{KEY_MAX_ELEMENTS}])")
_VALUE_TOO_BIG = etree.XPath(f"boolean(descendant::*[{VALUE_MAX_ELEMENTS}])")
_LAYOUT_TABLE = etree.XPath(
    "boolean((tr|thead/tr|tbody/tr|tfoot/tr)/*[self::td or self::th]"
    f"[descendant::table or descendant::*[{VALUE_MAX_ELEMENTS}]])"
)


def _key_text(el: etree._Element) -> str | None:
    """Text of a key cell: short (``KV_KEY_MAX_CHARS``) and made of a few nodes, else ``None``."""
    if len(el) >= KEY_MAX_ELEMENTS or _KEY_TOO_BIG(el):
        return None
    text = text_prefix(el, KV_KEY_MAX_CHARS + 1)
    if not text or len(text) > KV_KEY_MAX_CHARS:
        return None
    return text


def _value_text(el: etree._Element) -> str | None:
    """Prefix of a value cell's text; ``None`` when empty or when the cell is a page section."""
    if len(el) >= VALUE_MAX_ELEMENTS or _VALUE_TOO_BIG(el):
        return None
    return text_prefix(el, VALUE_PREFIX_CHARS) or None


def _colspan(el: etree._Element) -> int:
    raw = el.get("colspan")
    if raw and raw.strip().isdigit():
        return max(1, min(int(raw.strip()), 50))
    return 1


def _element_children(el: etree._Element) -> list[etree._Element]:
    return [c for c in el if isinstance(c.tag, str)]


def walk(root: etree._Element, *, for_labels: bool = True) -> PageWalk:
    """Walk a cleaned page once (see module docstring). ``for_labels=False`` skips the element
    index (``start`` / ``own``) that only label search needs."""
    page = PageWalk()
    ctx = _Ctx(page)
    items = page.items
    start = page.start
    own = page.own
    stack: list[tuple[etree._Element, str, KvRef | None, CellRef | None, int]] = [
        (root, step_label(root), None, None, 0)
    ]
    while stack:
        el, path, kv, cell, depth = stack.pop()
        if ctx.pend_kv:
            pend_kv = ctx.pend_kv.pop(el, None)
            if pend_kv is not None:
                kv = KvRef(pend_kv[0], pend_kv[1], pend_kv[2], depth)
        if ctx.pend_cell:
            pend_cell = ctx.pend_cell.pop(el, None)
            if pend_cell is not None:
                cell = CellRef(pend_cell[0], pend_cell[1], pend_cell[2], pend_cell[3], depth)
        if for_labels:
            start[el] = len(items)
        children = _element_children(el)  # first: keeps child proxies alive for own_text
        text = own_text(el)
        own_kv = kv
        if children:
            labels = [step_label(c) for c in children]
            child_paths = [path + PATH_SEP + label for label in labels]
            if len(children) > 1:
                _count_siblings(page.rows, child_paths)
            tag = _tag(el)
            if tag == "dl":
                _detect_dl(ctx, path, children, child_paths)
            elif tag == "table":
                _detect_table(ctx, el, path, children, child_paths)
            elif tag not in NO_STRUCT_TAGS and not _detect_alt(
                ctx, path, children, child_paths, labels
            ):
                own_kv = _detect_pair(ctx, el, path, text, children, child_paths, depth, kv)
            for i in range(len(children) - 1, -1, -1):
                stack.append((children[i], child_paths[i], kv, cell, depth + 1))
        if text:
            if for_labels:
                own[el] = len(items)
            items.append(Item(path, text, el, None, own_kv, cell))
        for attr in LINK_ATTRS:
            value = el.get(attr)
            if value and value.strip():
                items.append(Item(path + ATTR_SEP + attr, value.strip(), el, attr, own_kv, cell))
    return page


def subtree_end(page: PageWalk, el: etree._Element) -> int:
    """Index one past the last item of ``el``'s subtree (memoized in ``page.end``)."""
    pending: list[etree._Element] = []
    node: etree._Element | None = el
    end = len(page.items)
    while node is not None:
        memo = page.end.get(node)
        if memo is not None:
            end = memo
            break
        pending.append(node)
        nxt = node.getnext()
        while nxt is not None and not isinstance(nxt.tag, str):
            nxt = nxt.getnext()
        if nxt is not None and nxt in page.start:
            end = page.start[nxt]
            break
        node = node.getparent()
    for done in pending:
        page.end[done] = end
    return end


def _count_siblings(rows: dict[str, int], child_paths: list[str]) -> None:
    counts: dict[str, int] = {}
    for cp in child_paths:
        counts[cp] = counts.get(cp, 0) + 1
    for cp, n in counts.items():
        if n > 1:
            rows[cp] = rows.get(cp, 0) + n


def _key_value_runs(
    ctx: _Ctx,
    struct: str,
    children: list[etree._Element],
    child_paths: list[str],
    is_key: Callable[[int], bool],
    is_value: Callable[[int], bool],
) -> None:
    """Pair every value child with the nearest preceding key child (dl and alternating)."""
    rows: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None
    for i, child in enumerate(children):
        if is_key(i):
            key = _key_text(child)
            current = None
            if key is not None:
                ctx.pend_kv[child] = (struct, key, "k")
                current = (key, child_paths[i], [])
                rows.append(current)
        elif is_value(i) and current is not None:
            value = _value_text(child)
            if value is not None:
                ctx.pend_kv[child] = (struct, current[0], "v")
                current[2].append(value)
    for key, key_path, values in rows:
        if values:
            ctx.page.kv_rows.append(KvRow(struct, key_path, key_path, key, " ".join(values), None))


def _detect_dl(
    ctx: _Ctx, path: str, children: list[etree._Element], child_paths: list[str]
) -> None:
    tags = [_tag(c) for c in children]
    _key_value_runs(
        ctx,
        "dl|" + path,
        children,
        child_paths,
        lambda i: tags[i] == "dt",
        lambda i: tags[i] == "dd",
    )


def _detect_alt(
    ctx: _Ctx,
    path: str,
    children: list[etree._Element],
    child_paths: list[str],
    labels: list[str],
) -> bool:
    if len(children) < 4:
        return False
    first = labels[0]
    if len(set(labels)) != 2 or labels.count(first) < 2:
        return False
    for i in range(1, len(labels)):
        if labels[i] == first and labels[i - 1] == first:
            return False
    if labels[-1] == first:
        return False
    _key_value_runs(
        ctx,
        "alt|" + path,
        children,
        child_paths,
        lambda i: labels[i] == first,
        lambda i: labels[i] != first,
    )
    return True


def _detect_pair(
    ctx: _Ctx,
    el: etree._Element,
    path: str,
    text: str,
    children: list[etree._Element],
    child_paths: list[str],
    depth: int,
    inherited: KvRef | None,
) -> KvRef | None:
    """Detect a pair row; returns the KvRef for the row's own text (the value in the tail-value
    pattern, the key in the leading-text pattern)."""
    struct = "pair|" + path
    if not text:
        pair = _pair_children(children)
        if pair is None:
            return inherited
        k, v = pair
        key = _key_text(children[k])
        if key is None:
            return inherited
        value = _value_text(children[v])
        if value is None:
            return inherited
        ctx.pend_kv[children[k]] = (struct, key, "k")
        ctx.pend_kv[children[v]] = (struct, key, "v")
        ctx.page.kv_rows.append(
            KvRow(struct, path, child_paths[k], key, value, ctx.tr_table.get(el))
        )
        return inherited
    if len(children) != 1:
        return inherited
    if not (el.text or "").strip():
        key = _key_text(children[0])
        if key is None:
            return inherited
        ctx.pend_kv[children[0]] = (struct, key, "k")
        ctx.page.kv_rows.append(
            KvRow(
                struct, path, child_paths[0], key, text[:VALUE_PREFIX_CHARS], ctx.tr_table.get(el)
            )
        )
        return KvRef(struct, key, "v", depth)
    if (children[0].tail or "").strip() or len(text) > KV_KEY_MAX_CHARS:
        return inherited
    value = _value_text(children[0])
    if value is None:
        return inherited
    ctx.pend_kv[children[0]] = (struct, text, "v")
    ctx.page.kv_rows.append(KvRow(struct, path, path, text, value, ctx.tr_table.get(el)))
    return KvRef(struct, text, "k", depth)


def _pair_children(children: list[etree._Element]) -> tuple[int, int] | None:
    """Indexes of the key and value children of a pair row: both children of a two-child row,
    else the only two text-bearing children of a row of up to ``PAIR_MAX_CHILDREN`` small
    children."""
    if len(children) == 2:
        return 0, 1
    if not 2 < len(children) <= PAIR_MAX_CHILDREN:
        return None
    texted: list[int] = []
    for i, child in enumerate(children):
        if len(child) >= VALUE_MAX_ELEMENTS or _VALUE_TOO_BIG(child):
            return None
        if any(chunk.strip() for chunk in child.itertext()):
            texted.append(i)
    return (texted[0], texted[1]) if len(texted) == 2 else None


def _detect_table(
    ctx: _Ctx,
    table: etree._Element,
    path: str,
    children: list[etree._Element],
    child_paths: list[str],
) -> None:
    """Record a data table; layout tables (nested tables or section-sized cells) are ignored."""
    if _LAYOUT_TABLE(table):
        return
    struct = TABLE_KIND + "|" + path
    rows: list[tuple[str, list[etree._Element], bool, etree._Element]] = []
    for child, child_path in zip(children, child_paths, strict=True):
        tag = _tag(child)
        if tag == "tr":
            rows.append((child_path, _cells(child), False, child))
        elif tag in ("thead", "tbody", "tfoot"):
            for row in _element_children(child):
                if _tag(row) == "tr":
                    row_path = child_path + PATH_SEP + step_label(row)
                    rows.append((row_path, _cells(row), tag == "thead", row))
    if not rows:
        return
    header_idx = next((i for i, entry in enumerate(rows) if entry[2]), None)
    if header_idx is None:
        first = rows[0][1]
        if len(first) >= 2 and all(_tag(c) == "th" for c in first):
            header_idx = 0
    header_texts: list[str | None] = []
    header_cells: list[etree._Element] = []
    if header_idx is not None:
        for cell in rows[header_idx][1]:
            text = text_prefix(cell, KEY_MAX_CHARS) or None
            ctx.pend_cell[cell] = (struct, len(header_texts), text, "h")
            for _ in range(_colspan(cell)):
                header_texts.append(text)
                header_cells.append(cell)
    page = ctx.page
    row_paths: dict[str, int] = {}
    n_rows = 0
    n_cols = 0
    body: list[str] = []
    for i, (row_path, cells, in_head, row) in enumerate(rows):
        if i == header_idx or in_head or not cells:
            continue
        ctx.tr_table[row] = struct
        n_rows += 1
        n_cols = max(n_cols, len(cells))
        row_paths[row_path] = row_paths.get(row_path, 0) + 1
        if len(body) < 8:
            body.append(text_prefix(row, 80))
        col = 0
        for cell in cells:
            col_header = header_texts[col] if col < len(header_texts) else None
            ctx.pend_cell[cell] = (struct, col, col_header, "c")
            if col < len(header_cells):
                page.header_of[cell] = header_cells[col]
            col += _colspan(cell)
    header = tuple(h or "" for h in header_texts) if header_idx is not None else None
    main_row = max(row_paths, key=lambda rp: row_paths[rp]) if row_paths else None
    page.tables.append(TableObs(struct, header, main_row, n_rows, n_cols, "\x1f".join(body)))


def _cells(row: etree._Element) -> list[etree._Element]:
    return [c for c in row if isinstance(c.tag, str) and _tag(c) in CELL_TAGS]


# -- qualification -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructPlan:
    """Accepted structures: structure key -> table id (analysis uses interim ids)."""

    kv: Mapping[str, str]
    tables: Mapping[str, str]


class Qualifier(NamedTuple):
    table_id: str | None
    column: str | None
    role: str | None  # kv_key | kv_value | record_header | record_column
    named: bool  # column comes from a key cell or a header cell (not synthesized "colN")


NO_QUALIFIER = Qualifier(None, None, None, False)


def resolve(item: Item, plan: StructPlan) -> Qualifier:
    """Table role of ``item`` under the accepted structures; the nearest structure wins."""
    kv = item.kv if item.kv is not None and item.kv.struct in plan.kv else None
    cell = item.cell if item.cell is not None and item.cell.struct in plan.tables else None
    if kv is not None and (cell is None or kv.depth > cell.depth):
        tid = plan.kv[kv.struct]
        if kv.role == "k":
            return Qualifier(tid, None, "kv_key", True)
        return Qualifier(tid, kv.key, "kv_value", True)
    if cell is not None:
        tid = plan.tables[cell.struct]
        if cell.role == "h":
            return Qualifier(tid, None, "record_header", True)
        if cell.header:
            return Qualifier(tid, cell.header, "record_column", True)
        return Qualifier(tid, f"col{cell.col + 1}", "record_column", False)
    return NO_QUALIFIER


def is_split(q: Qualifier) -> bool:
    """Whether the qualifier splits a path into per-key / per-column candidates."""
    return q.role == "kv_value" or q.role == "record_column"


# -- labels ---------------------------------------------------------------------------------------


_EMPTY = -1  # the subtree shows no text: look further
_SECTION = -2  # a section (list, nav, table, big block): not a label, stop looking this way


def _nearest_text(page: PageWalk, el: etree._Element, *, forward: bool) -> int:
    """Index of the first (``forward``) or last visible text item in ``el``'s subtree, or
    ``_EMPTY`` / ``_SECTION``."""
    if _tag(el) in SECTION_TAGS:
        return _SECTION
    start = page.start.get(el)
    if start is None:
        return _EMPTY
    end = subtree_end(page, el)
    if end - start > LABEL_MAX_ITEMS:
        return _SECTION
    order = range(start, end) if forward else range(end - 1, start - 1, -1)
    for j in order:
        if page.items[j].attr is None:
            return j
    return _EMPTY


def _first_child(el: etree._Element) -> etree._Element | None:
    for child in el:
        if isinstance(child.tag, str):
            return child
    return None


def find_label(page: PageWalk, idx: int, ok: Callable[[int], bool]) -> int | None:
    """Index of the item labelling item ``idx``: the nearest text accepted by ``ok`` (an
    invariant text), or ``None``.

    For an attribute, its element's own text comes first; for own text that follows a leading
    child (``<li><b>Brand:</b> Acme</li>``), that child. Then, climbing at most
    ``LABEL_LEVELS`` ancestors (never into ``body``): the nearest preceding sibling showing
    text (looking past at most ``LABEL_SIBLINGS`` empty ones), the table header cell, the
    parent's leading own text, and the parent's first child. Only the text *adjacent* to the
    node counts: when the nearest preceding text is itself a value, the search ends after this
    level's header / parent checks, so a label never comes from beyond another field. Lists,
    navigation, tables and blocks of more than ``LABEL_MAX_ITEMS`` items are never labels.
    """
    item = page.items[idx]
    el = item.element
    if item.attr is not None:
        j = page.own.get(el)
        if j is not None and ok(j):
            return j
    elif not (el.text or "").strip():
        first = _first_child(el)
        if first is not None:
            j = _nearest_text(page, first, forward=True)
            if j >= 0 and ok(j):
                return j
    node = el
    for _ in range(LABEL_LEVELS):
        blocked = False
        sib = node.getprevious()
        seen = 0
        while sib is not None and seen < LABEL_SIBLINGS:
            if isinstance(sib.tag, str):
                seen += 1
                j = _nearest_text(page, sib, forward=False)
                if j >= 0:
                    if ok(j):
                        return j
                    blocked = True
                    break
                if j == _SECTION:
                    blocked = True
                    break
            sib = sib.getprevious()
        header = page.header_of.get(node)
        if header is not None:
            j = _nearest_text(page, header, forward=True)
            if j >= 0 and ok(j):
                return j
        parent = node.getparent()
        if parent is None or _tag(parent) in ("body", "html"):
            return None
        if (parent.text or "").strip():
            j = page.own.get(parent, -1)
            if j >= 0 and ok(j):
                return j
        first = _first_child(parent)
        if first is not None and first is not node:
            j = _nearest_text(page, first, forward=True)
            if j >= 0 and ok(j):
                return j
        if blocked:
            return None
        node = parent
    return None
