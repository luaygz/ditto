"""Structural paths over a mutable lxml tree and candidate location for the union page.

``Tree`` caches each element's structural path (the ``ditto.fingerprint.dom`` format), so a tree
can be edited (spliced into, pruned) and re-indexed cheaply. ``locate`` finds field candidates
with the analysis' own matcher (``ditto.analyze.locate_candidates``), so per-key and per-column
candidates that share one path resolve exactly as the analysis counted them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from lxml import etree

from ditto.analyze.locate import locate_candidates
from ditto.analyze.models import Candidate, CandidateTable, Table
from ditto.fingerprint.dom import ATTR_SEP, PATH_SEP, node_text, step_label
from ditto.textutil import normalize_ws

Element = etree._Element

_LABEL_ATTRS = ("id", "class", "itemprop", "data-testid", "role")
_LABEL_CACHE: dict[tuple[str | None, ...], str] = {}
_LABEL_CACHE_MAX = 200_000
# Table layouts whose candidates split one path by key / column (as the analysis does);
# "rows" tables and tables without a layout keep path-level candidates.
SPLIT_LAYOUTS = frozenset({"dl", "alternating", "pairs", "table"})
# Table roles of value cells: shown on the union page even when the analysis classed them noise
SHOWN_TABLE_ROLES = frozenset({"record_column", "kv_value"})


def shown_candidate(c: Candidate) -> bool:
    """Whether the union page shows (annotates, splices) candidate ``c``: every field candidate,
    plus value cells of a detected table that are too rare to be fields (a record column or a
    key-value key on under 2% of pages; the generation prompt lists them too)."""
    return c.is_field or (c.table_id is not None and c.table_role in SHOWN_TABLE_ROLES)


def splits_columns(table: Table | None) -> bool:
    return table is not None and table.layout in SPLIT_LAYOUTS


def iter_elements(root: Element) -> Iterator[Element]:
    """All elements under ``root`` (inclusive) in document order."""
    for el in root.iter():
        if isinstance(el.tag, str):
            yield el


def child_elements(el: Element) -> list[Element]:
    return [c for c in el if isinstance(c.tag, str)]


def label_of(el: Element) -> str:
    """``step_label(el)`` memoized on the attributes that determine it."""
    tag = str(el.tag)
    key = (tag, *(el.get(a) for a in _LABEL_ATTRS)) if el.keys() else (tag,)
    label = _LABEL_CACHE.get(key)
    if label is None:
        label = step_label(el)
        if len(_LABEL_CACHE) < _LABEL_CACHE_MAX:
            _LABEL_CACHE[key] = label
    return label


def short_text(el: Element, limit: int = 200) -> str:
    return normalize_ws(node_text(el))[:limit]


def label_key(text: str) -> str:
    """Comparison key for label text: whitespace-normalized, case-folded, trailing colon cut."""
    return normalize_ws(text).casefold().rstrip(":：").strip()


def split_candidate_path(path: str) -> tuple[str, str | None]:
    """``"a > b @href"`` -> ``("a > b", "href")``; plain paths have no attribute."""
    if ATTR_SEP in path:
        node, attr = path.rsplit(ATTR_SEP, 1)
        return node, attr
    return path, None


class Tree:
    """A parsed page plus a structural-path cache. Call ``invalidate`` after inserting nodes."""

    def __init__(self, root: Element) -> None:
        self.root = root
        self._paths: dict[Element, str] = {}
        self._by_path: dict[str, list[Element]] | None = None

    def path(self, el: Element) -> str:
        cached = self._paths.get(el)
        if cached is not None:
            return cached
        pending: list[Element] = []
        node: Element | None = el
        prefix = ""
        while node is not None:
            hit = self._paths.get(node)
            if hit is not None:
                prefix = hit
                break
            pending.append(node)
            node = node.getparent()
        for item in reversed(pending):
            label = label_of(item)
            prefix = prefix + PATH_SEP + label if prefix else label
            self._paths[item] = prefix
        return prefix

    def by_path(self) -> dict[str, list[Element]]:
        """Path -> elements in document order (one depth-first pass that fills the cache)."""
        if self._by_path is None:
            index: dict[str, list[Element]] = {}
            paths = self._paths
            stack = [(self.root, self.path(self.root))]
            while stack:
                el, path = stack.pop()
                paths[el] = path
                bucket = index.get(path)
                if bucket is None:
                    index[path] = [el]
                else:
                    bucket.append(el)
                prefix = path + PATH_SEP
                kids = list(el.iterchildren(etree.Element))
                for child in reversed(kids):
                    stack.append((child, prefix + label_of(child)))
            self._by_path = index
        return self._by_path

    def invalidate(self) -> None:
        self._by_path = None


@dataclass
class CandIndex:
    """Lookup structures derived once from a candidate table."""

    table: CandidateTable
    fields: dict[str, Candidate]
    """The candidates the union page shows (``shown_candidate``), by id."""
    tables: dict[str, Table]
    key_paths: dict[str, str]
    container_paths: dict[str, str]
    path_class: dict[str, str]

    @classmethod
    def build(cls, table: CandidateTable) -> CandIndex:
        by_id = {c.id: c for c in table.candidates}
        tables = {t.id: t for t in table.tables}
        key_paths: dict[str, str] = {}
        container_paths: dict[str, str] = {}
        for t in table.tables:
            container_paths[t.id] = split_candidate_path(t.container_path)[0]
            key = by_id.get(t.key_candidate) if t.key_candidate else None
            if key is not None:
                key_paths[t.id] = split_candidate_path(key.path)[0]
        path_class: dict[str, str] = {}
        for c in table.candidates:
            # a path holding any shown candidate counts as a field path
            shown = shown_candidate(c)
            if shown or c.path not in path_class:
                path_class[c.path] = "field" if shown else c.cls
        fields = {c.id: c for c in table.candidates if shown_candidate(c)}
        return cls(
            table=table,
            fields=fields,
            tables=tables,
            key_paths=key_paths,
            container_paths=container_paths,
            path_class=path_class,
        )


def preceding_elements(el: Element) -> Iterator[Element]:
    sib = el.getprevious()
    while sib is not None:
        if isinstance(sib.tag, str):
            yield sib
        sib = sib.getprevious()


def ancestor_at(tree: Tree, el: Element, path: str) -> Element | None:
    node: Element | None = el
    while node is not None:
        if tree.path(node) == path:
            return node
        node = node.getparent()
    return None


def key_in(tree: Tree, el: Element, key_path: str) -> Element | None:
    """``el`` itself or its last descendant at ``key_path``."""
    if tree.path(el) == key_path:
        return el
    found: Element | None = None
    for d in iter_elements(el):
        if d is not el and tree.path(d) == key_path:
            found = d
    return found


def kv_key(
    tree: Tree,
    el: Element,
    key_path: str | None,
    container_path: str | None,
    columns: Iterable[str] = (),
    max_up: int = 5,
) -> Element | None:
    """The key element of the key-value row holding ``el``: the nearest preceding sibling (of
    ``el`` or an ancestor below the container, at most ``max_up`` levels up) that is, or
    contains, a node at ``key_path``. Without a key path, a preceding sibling whose text is one
    of ``columns`` qualifies."""
    wanted = {label_key(c) for c in columns}
    node: Element | None = el
    for _ in range(max_up + 1):
        if node is None or (container_path is not None and tree.path(node) == container_path):
            return None
        for sib in preceding_elements(node):
            if key_path is not None:
                hit = key_in(tree, sib, key_path)
                if hit is not None:
                    return hit
            elif label_key(node_text(sib)) in wanted:
                return sib
        node = node.getparent()
    return None


def locate(tree: Tree, index: CandIndex, only: set[str] | None = None) -> dict[str, list[Element]]:
    """Field candidate id -> elements carrying it on this page, in document order (attribute
    candidates map to the element carrying the attribute); ``only`` restricts the ids."""
    found = locate_candidates(tree.root, index.table)
    return {
        cid: els
        for cid, els in found.items()
        if cid in index.fields and (only is None or cid in only)
    }


def find_label(el: Element, hint: str | None, max_prev: int = 3) -> Element | None:
    """The element holding ``hint``: one of the nearest preceding siblings, else the parent's
    first child."""
    if not hint:
        return None
    want = label_key(hint)
    if not want:
        return None
    for n, sib in enumerate(preceding_elements(el)):
        if n >= max_prev:
            break
        if label_key(node_text(sib)) == want:
            return sib
    parent = el.getparent()
    if parent is not None:
        first = next((c for c in parent if isinstance(c.tag, str)), None)
        if first is not None and first is not el and label_key(node_text(first)) == want:
            return first
    return None
