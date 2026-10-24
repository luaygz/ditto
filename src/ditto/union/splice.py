"""Base page choice and splicing of field candidates the base page lacks.

A candidate missing from the base page is copied from a donor page as a minimal unit: the node
plus its label sibling (a key-value candidate: its row, or its key and value cells; a record
column: its table). The unit is inserted under the base element with the same path as the
unit's parent, after the nearest preceding sibling both pages share, else appended; missing
ancestors are created as shallow copies of the donor's.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from ditto.analyze.models import Candidate
from ditto.fingerprint.dom import clean_html, own_text
from ditto.union.prepare import INVARIANT_CLASSES, X_SPLICED, add_mark, prune_attrs, up_from
from ditto.union.tree import (
    CandIndex,
    Element,
    Tree,
    ancestor_at,
    child_elements,
    find_label,
    iter_elements,
    key_in,
    kv_key,
    label_of,
    locate,
    preceding_elements,
    short_text,
    splits_columns,
)

MATCH_CANDIDATES_MAX = 64  # base elements scored when several share the unit's parent path


@dataclass(frozen=True)
class PageSurvey:
    page_id: int
    present: frozenset[str]  # field candidate ids located on the page
    size: int  # raw HTML length


def cid_key(cid: str) -> tuple[int, str]:
    """Sort key putting ``c2`` before ``c10``."""
    digits = cid[1:]
    return (int(digits), cid) if digits.isdigit() else (1 << 30, cid)


def survey(pages: Mapping[int, str], index: CandIndex) -> list[PageSurvey]:
    out: list[PageSurvey] = []
    for pid in sorted(pages):
        html = pages[pid]
        located = locate(Tree(clean_html(html)), index)
        out.append(PageSurvey(pid, frozenset(located), len(html)))
    return out


def choose_base(surveys: list[PageSurvey], focus: set[str] | None = None) -> PageSurvey:
    """Most covered field candidates (focused ones first); ties: smaller page, lower id."""

    def key(s: PageSurvey) -> tuple[int, int, int, int]:
        focused = len(s.present & focus) if focus else 0
        return (-focused, -len(s.present), s.size, s.page_id)

    return min(surveys, key=key)


def choose_donors(
    surveys: list[PageSurvey], base_id: int, missing: Iterable[str]
) -> list[tuple[int, frozenset[str]]]:
    """Greedy set cover: repeatedly the page holding the most still-missing candidates."""
    remaining = set(missing)
    pool = [s for s in surveys if s.page_id != base_id]
    donors: list[tuple[int, frozenset[str]]] = []
    while remaining and pool:
        best = max(pool, key=lambda s: (len(s.present & remaining), -s.size, -s.page_id))
        gain = best.present & remaining
        if not gain:
            break
        donors.append((best.page_id, frozenset(gain)))
        remaining -= gain
        pool.remove(best)
    return donors


def splice_missing(
    base: Tree,
    base_id: int,
    pages: Mapping[int, str],
    surveys: list[PageSurvey],
    index: CandIndex,
    missing: list[str],
    log: Callable[[str], None],
) -> tuple[list[str], list[int]]:
    """Splice every candidate of ``missing`` some sampled page has, donor by donor in document
    order. Returns the spliced ids and the donor page ids used."""
    remaining = set(missing)
    spliced: list[str] = []
    used: list[int] = []
    for donor_id, gain in choose_donors(surveys, base_id, missing):
        todo = gain & remaining
        if not todo:
            continue
        donor = Tree(clean_html(pages[donor_id]))
        located = locate(donor, index, only=set(todo))
        position = {el: n for n, el in enumerate(iter_elements(donor.root))}
        order = sorted(
            (position[els[0]], cid) for cid, els in located.items() if cid in todo and els
        )
        contributed = False
        for _, cid in order:
            if cid not in remaining:
                continue
            splice_candidate(base, donor, index.fields[cid], located[cid][0], index)
            spliced.append(cid)
            contributed = True
            now = set(locate(base, index, only=remaining))
            if cid not in now:
                log(f"spliced {cid} from page {donor_id}, but it does not locate on the union page")
            remaining -= now
            remaining.discard(cid)
        if contributed:
            used.append(donor_id)
    for cid in sorted(remaining, key=cid_key):
        log(f"cannot splice {cid}: no sampled page has it")
    return spliced, used


def splice_candidate(
    base: Tree, donor: Tree, cand: Candidate, node: Element, index: CandIndex
) -> Element:
    """Copy the minimal unit carrying ``cand`` (``node`` on the donor page) into the base page;
    returns the top inserted element (marked ``data-x-spliced``)."""
    unit, label, fixed = _unit(donor, cand, node, index)
    by_path = base.by_path()
    chain: list[Element] = []
    anchor_parent = unit.getparent()
    while anchor_parent is not None and donor.path(anchor_parent) not in by_path:
        chain.append(anchor_parent)
        anchor_parent = anchor_parent.getparent()
    if anchor_parent is None:
        # not even the root path matches: the donor root stands for the base root
        if chain:
            chain.pop()
        target = base.root
    else:
        target = _best_match(by_path[donor.path(anchor_parent)], anchor_parent)
    levels = [unit, *chain]  # bottom-up; each level's donor parent is the next level
    label_at = 0
    if label is None and not fixed:
        for i, level in enumerate(levels):
            label = find_label(level, cand.label_hint)
            if label is not None:
                label_at = i
                break
    copies = [_deep_copy(unit), *(_shallow_copy(el) for el in chain)]
    for i in range(1, len(copies)):
        copies[i].append(copies[i - 1])
    ident = _Identity(index)
    first_donor = levels[-1]
    items = [copies[-1]]
    if label is not None:
        if label_at < len(levels) - 1:
            copies[label_at].addprevious(_deep_copy(label))
        elif _match_child(base, target, donor, label, ident, exact=True) is None:
            items.insert(0, _deep_copy(label))
            first_donor = label
    _insert_aligned(base, target, donor, first_donor, items, ident)
    add_mark(items[0], X_SPLICED, [cand.id])
    base.invalidate()
    return items[0]


def _unit(
    donor: Tree, cand: Candidate, node: Element, index: CandIndex
) -> tuple[Element, Element | None, bool]:
    """(unit, label sibling, fixed): ``fixed`` means the unit already includes its label."""
    table = index.tables.get(cand.table_id) if cand.table_id else None
    if not splits_columns(table):
        table = None  # "rows" tables: path-level candidates splice like any other node
    if table is not None and cand.table_role == "kv_value":
        key_path = index.key_paths.get(table.id)
        columns = [cand.column] if cand.column else []
        key = kv_key(donor, node, key_path, index.container_paths.get(table.id), columns)
        if key is not None:
            common = _common_ancestor(key, node)
            if common is not None and common.getparent() is not None:
                if _is_container(donor, common, key, key_path):
                    unit = _child_toward(common, node)
                    label = _child_toward(common, key)
                    return unit, (None if label is unit else label), True
                return common, None, True
        elif key_path is not None and key_in(donor, node, key_path) not in (None, node):
            # a "pairs" row whose own text is the value: <li><b>Brand:</b> Acme</li>
            return node, None, True
    if table is not None and cand.table_role == "record_column" and cand.column:
        container = ancestor_at(donor, node, index.container_paths[table.id])
        if container is None:
            container = next((n for n in up_from(node, 64) if str(n.tag) == "table"), None)
        if container is not None and container.getparent() is not None:
            return container, None, True
    return node, None, False


def _is_container(donor: Tree, common: Element, key: Element, key_path: str | None) -> bool:
    """Whether ``common`` holds several key-value rows (dl-like) rather than being one row."""
    if key_path is None:
        return len(child_elements(common)) > 2
    return any(n is not key and donor.path(n) == key_path for n in iter_elements(common))


def _common_ancestor(a: Element, b: Element) -> Element | None:
    ancestors = set(up_from(b, 1 << 16))
    node: Element | None = a
    while node is not None:
        if node in ancestors:
            return node
        node = node.getparent()
    return None


def _child_toward(ancestor: Element, el: Element) -> Element:
    node = el
    parent = node.getparent()
    while parent is not None and parent is not ancestor:
        node = parent
        parent = node.getparent()
    return node


def _deep_copy(el: Element) -> Element:
    out = copy.deepcopy(el)
    out.tail = None
    prune_attrs(out)
    return out


def _shallow_copy(el: Element) -> Element:
    out = el.makeelement(el.tag, {str(k): str(v) for k, v in el.attrib.items()})
    out.text = el.text
    prune_attrs(out)
    return out


def _best_match(options: list[Element], donor_el: Element) -> Element:
    """The base element (of several with one path) most like ``donor_el``: shared child
    labels, plus shared short child texts counted double."""
    if len(options) == 1:
        return options[0]
    kids = child_elements(donor_el)
    labels = {label_of(c) for c in kids}
    texts = {t for t in (short_text(c, 60) for c in kids) if t}
    best, best_score = options[0], -1
    for option in options[:MATCH_CANDIDATES_MAX]:
        okids = child_elements(option)
        score = len(labels & {label_of(c) for c in okids})
        score += 2 * len(texts & {short_text(c, 60) for c in okids})
        if score > best_score:
            best, best_score = option, score
    return best


class _Identity:
    """How a sibling is recognized across pages: its invariant (label or boilerplate) text
    when it has some, else its whole text."""

    def __init__(self, index: CandIndex) -> None:
        self.index = index
        self.cache: dict[Element, str] = {}

    def __call__(self, tree: Tree, el: Element) -> str:
        hit = self.cache.get(el)
        if hit is None:
            parts = [
                text
                for n in iter_elements(el)
                if self.index.path_class.get(tree.path(n)) in INVARIANT_CLASSES
                and (text := own_text(n))
            ]
            hit = "label:" + " | ".join(parts)[:200] if parts else "text:" + short_text(el)
            self.cache[el] = hit
        return hit


def _match_child(
    base: Tree, target: Element, donor: Tree, sib: Element, ident: _Identity, *, exact: bool
) -> Element | None:
    """The child of ``target`` corresponding to donor sibling ``sib``: same step label and
    same identity (the last such child); without ``exact``, also the only child with that
    step label."""
    label = label_of(sib)
    same = [c for c in child_elements(target) if label_of(c) == label]
    if not same:
        return None
    want = ident(donor, sib)
    hits = [c for c in same if ident(base, c) == want]
    if hits:
        return hits[-1]
    if not exact and len(same) == 1:
        return same[0]
    return None


def _insert_aligned(
    base: Tree,
    target: Element,
    donor: Tree,
    first_donor: Element,
    items: list[Element],
    ident: _Identity,
) -> None:
    """Insert ``items`` under ``target`` after the base sibling matching the nearest preceding
    donor sibling of ``first_donor`` (moved past base siblings that align with the donor
    siblings in between), else append them."""
    prev = list(preceding_elements(first_donor))
    donor_parent = first_donor.getparent()
    donor_ids = (
        {ident(donor, c) for c in child_elements(donor_parent)}
        if donor_parent is not None
        else set()
    )
    anchor: Element | None = None
    for i, sib in enumerate(prev):
        match = _match_child(base, target, donor, sib, ident, exact=False)
        if match is None:
            continue
        anchor = match
        for between in reversed(prev[:i]):
            nxt = anchor.getnext()
            while nxt is not None and not isinstance(nxt.tag, str):
                nxt = nxt.getnext()
            if nxt is None or label_of(nxt) != label_of(between):
                break
            nxt_id = ident(base, nxt)
            if nxt_id != ident(donor, between) and nxt_id in donor_ids:
                break
            anchor = nxt
        break
    if anchor is None:
        for item in items:
            target.append(item)
        return
    for item in items:
        anchor.addnext(item)
        anchor = item
