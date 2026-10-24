"""Cluster analysis (stage 4): classify every text path of a template and detect tables.

Two passes over the sampled pages; each pass accumulates per chunk of pages and merges, so both
parallelize (:mod:`ditto.analyze.stage`) and run in-process here (:func:`analyze_pages`).

1. :class:`Survey`: per structural path the presence, the modal share of the per-page text
   sequence, the repeat range and per-text recurrence; per key-value structure and table the key /
   header recurrence and whether values vary.
2. :func:`decide` accepts tables and fixes the invariant texts, then :class:`Collect` gathers the
   final statistics per *candidate*: a path, or a path split per key (key-value tables) or per
   column (record tables), with examples and label hints.

:func:`build_table` turns both into the :class:`~ditto.analyze.models.CandidateTable`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol, TypeVar

from ditto.analyze.models import (
    FIELD_CLASSES,
    Candidate,
    CandidateClass,
    CandidateTable,
    Table,
    TableColumn,
    TableLayout,
    TableRole,
)
from ditto.analyze.walk import (
    KEY_MAX_CHARS,
    TABLE_KIND,
    KvRow,
    PageWalk,
    Qualifier,
    StructPlan,
    TableObs,
    find_label,
    is_split,
    resolve,
    walk,
)
from ditto.fingerprint.dom import ATTR_SEP, PATH_SEP, clean_html, parse_step
from ditto.specgen.typecheck import infer_value_type
from ditto.textutil import hash64

BOILERPLATE_M = 0.98
"""Modal share at or above which a path is invariant (boilerplate or label)."""
MANDATORY_P = 0.95
"""Presence at or above which a varying path is mandatory."""
NOISE_P = 0.02
NOISE_MIN_PAGES = 20
"""A path under ``NOISE_P`` presence is noise unless present on this many sampled pages."""
KEY_RECUR_P = 0.02
"""A key / header text recurs when present on at least max(2, 2% of the structure's pages)."""
KEY_RECUR_SHARE = 0.7
"""A key column / header row is invariant when this share of its cells hold recurring texts."""
N_EXAMPLES = 3
EXAMPLE_CHARS = 160
TYPE_SAMPLES = 50
TYPE_SAMPLES_PER_PAGE = 5
LABEL_PAGES = 8
"""Label searches per candidate per chunk of pages (the hint is the mode over these)."""
SEQ_SEP = "\x1f"
SURVEY_SAMPLE = 500
"""Pass 1 (table acceptance, invariant texts) looks at most at this many sampled pages."""
_BIG = 1 << 30
_T = TypeVar("_T")

CandKey = tuple[str, str | None, str | None]
"""(path, table structure, column or key): the identity of a candidate within a template."""


def classify(
    p: float, m: float, pages_present: int, repeat_min: int, repeat_max: int
) -> CandidateClass:
    """Brief stage-4 rules. Invariance is checked before repetition (a menu whose length varies
    on 1% of pages is still boilerplate); ``label`` is assigned later, to invariant texts that
    serve as the label hint of a field candidate."""
    if p < NOISE_P and pages_present < NOISE_MIN_PAGES:
        return "noise"
    if m >= BOILERPLATE_M:
        return "boilerplate"
    if repeat_min != repeat_max:
        return "list"
    if p >= MANDATORY_P:
        return "mandatory"
    return "optional"


def _recurring_share(text_df: Counter[str], n_pages: int) -> float:
    occurrences = sum(text_df.values())
    if occurrences == 0:
        return 0.0
    min_df = max(2, math.ceil(KEY_RECUR_P * n_pages))
    return sum(c for c in text_df.values() if c >= min_df) / occurrences


# -- pass 1: survey --------------------------------------------------------------------------------


@dataclass(slots=True)
class _PathAcc:
    df: int = 0
    rmin: int = _BIG
    rmax: int = 0
    joined: Counter[int] = field(default_factory=Counter)
    multi: Counter[int] = field(default_factory=Counter)  # per-text df on pages with repeats

    def add(self, texts: list[str]) -> None:
        n = len(texts)
        self.df += 1
        self.rmin = min(self.rmin, n)
        self.rmax = max(self.rmax, n)
        if n == 1:
            self.joined[hash64(texts[0])] += 1
        else:
            self.joined[hash64(SEQ_SEP.join(texts))] += 1
            for text in set(texts):
                self.multi[hash64(text)] += 1

    def merge(self, other: _PathAcc) -> None:
        self.df += other.df
        self.rmin = min(self.rmin, other.rmin)
        self.rmax = max(self.rmax, other.rmax)
        self.joined.update(other.joined)
        self.multi.update(other.multi)


@dataclass(slots=True)
class _KvAcc:
    df: int = 0
    rmin: int = _BIG
    rmax: int = 0
    table: str | None = None
    row_paths: Counter[str] = field(default_factory=Counter)
    key_paths: Counter[str] = field(default_factory=Counter)
    key_df: Counter[str] = field(default_factory=Counter)
    values: Counter[tuple[str, int]] = field(default_factory=Counter)

    def add(self, rows: list[KvRow]) -> None:
        self.df += 1
        self.rmin = min(self.rmin, len(rows))
        self.rmax = max(self.rmax, len(rows))
        per_key: dict[str, list[str]] = {}
        for row in rows:
            per_key.setdefault(row.key, []).append(row.value)
            self.row_paths[row.row_path] += 1
            self.key_paths[row.key_path] += 1
            if self.table is None:
                self.table = row.table
        for key, values in per_key.items():
            self.key_df[key] += 1
            self.values[(key, hash64(SEQ_SEP.join(values)))] += 1

    def merge(self, other: _KvAcc) -> None:
        self.df += other.df
        self.rmin = min(self.rmin, other.rmin)
        self.rmax = max(self.rmax, other.rmax)
        self.table = self.table or other.table
        self.row_paths.update(other.row_paths)
        self.key_paths.update(other.key_paths)
        self.key_df.update(other.key_df)
        self.values.update(other.values)


@dataclass(slots=True)
class _TableAcc:
    df: int = 0
    rmin: int = _BIG
    rmax: int = 0
    ncols: int = 0
    header_pages: int = 0
    row_paths: Counter[str] = field(default_factory=Counter)
    header_seq: Counter[tuple[str, ...]] = field(default_factory=Counter)
    header_df: Counter[str] = field(default_factory=Counter)
    body: Counter[int] = field(default_factory=Counter)

    def add(self, tables: list[TableObs]) -> None:
        n_rows = sum(t.n_rows for t in tables)
        self.df += 1
        self.rmin = min(self.rmin, n_rows)
        self.rmax = max(self.rmax, n_rows)
        self.ncols = max(self.ncols, *(t.n_cols for t in tables))
        header = next((t.header for t in tables if t.header is not None), None)
        if header is not None:
            self.header_pages += 1
            self.header_seq[header] += 1
            for text in set(header):
                if text:
                    self.header_df[text] += 1
        for t in tables:
            if t.row_path is not None:
                self.row_paths[t.row_path] += t.n_rows
        self.body[hash64(SEQ_SEP.join(t.body for t in tables))] += 1

    def merge(self, other: _TableAcc) -> None:
        self.df += other.df
        self.rmin = min(self.rmin, other.rmin)
        self.rmax = max(self.rmax, other.rmax)
        self.ncols = max(self.ncols, other.ncols)
        self.header_pages += other.header_pages
        self.row_paths.update(other.row_paths)
        self.header_seq.update(other.header_seq)
        self.header_df.update(other.header_df)
        self.body.update(other.body)


@dataclass(slots=True)
class _Range:
    df: int = 0
    rmin: int = _BIG
    rmax: int = 0

    def add(self, n: int) -> None:
        self.df += 1
        self.rmin = min(self.rmin, n)
        self.rmax = max(self.rmax, n)

    def merge(self, other: _Range) -> None:
        self.df += other.df
        self.rmin = min(self.rmin, other.rmin)
        self.rmax = max(self.rmax, other.rmax)


@dataclass(slots=True)
class Survey:
    """Pass-1 accumulator for one template (mergeable across chunks of pages)."""

    n: int = 0
    paths: dict[str, _PathAcc] = field(default_factory=dict)
    kv: dict[str, _KvAcc] = field(default_factory=dict)
    tables: dict[str, _TableAcc] = field(default_factory=dict)
    rows: dict[str, _Range] = field(default_factory=dict)

    def add(self, page_id: int, page: PageWalk) -> None:
        self.n += 1
        by_path: dict[str, list[str]] = {}
        for item in page.items:
            texts = by_path.get(item.path)
            if texts is None:
                by_path[item.path] = [item.text]
            else:
                texts.append(item.text)
        for path, texts in by_path.items():
            acc = self.paths.get(path)
            if acc is None:
                acc = self.paths[path] = _PathAcc()
            acc.add(texts)
        kv_rows: dict[str, list[KvRow]] = {}
        for row in page.kv_rows:
            kv_rows.setdefault(row.struct, []).append(row)
        for struct, rows in kv_rows.items():
            kv_acc = self.kv.get(struct)
            if kv_acc is None:
                kv_acc = self.kv[struct] = _KvAcc()
            kv_acc.add(rows)
        tables: dict[str, list[TableObs]] = {}
        for table in page.tables:
            tables.setdefault(table.struct, []).append(table)
        for struct, obs in tables.items():
            t_acc = self.tables.get(struct)
            if t_acc is None:
                t_acc = self.tables[struct] = _TableAcc()
            t_acc.add(obs)
        for row_path, count in page.rows.items():
            r_acc = self.rows.get(row_path)
            if r_acc is None:
                r_acc = self.rows[row_path] = _Range()
            r_acc.add(count)

    def merge(self, other: Survey) -> None:
        self.n += other.n
        _merge_into(self.paths, other.paths)
        _merge_into(self.kv, other.kv)
        _merge_into(self.tables, other.tables)
        _merge_into(self.rows, other.rows)


class _Mergeable(Protocol):
    def merge(self, other: Any) -> None: ...


_A = TypeVar("_A", bound=_Mergeable)


def _merge_into(mine: dict[str, _A], theirs: dict[str, _A]) -> None:
    for key, acc in theirs.items():
        current = mine.get(key)
        if current is None:
            mine[key] = acc
        else:
            current.merge(acc)


# -- decisions ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    """What pass 2 needs: accepted structures (interim table id = structure key) and the
    invariant texts for label search."""

    structs: StructPlan
    headered: frozenset[str]
    inv_paths: frozenset[str]
    inv_pairs: frozenset[tuple[str, int]]


def _accept_kv(acc: _KvAcc) -> bool:
    """Key column invariant across pages (keys recur), values vary, and rows repeat."""
    if acc.df < 2 or acc.rmax < 2 or _recurring_share(acc.key_df, acc.df) < KEY_RECUR_SHARE:
        return False
    min_df = max(2, math.ceil(KEY_RECUR_P * acc.df))
    mode: dict[str, int] = {}
    for (key, _), count in acc.values.items():
        mode[key] = max(mode.get(key, 0), count)
    return any(
        acc.key_df[key] >= min_df and top < BOILERPLATE_M * acc.key_df[key]
        for key, top in mode.items()
    )


def _table_header(acc: _TableAcc) -> bool | None:
    """``True``: record table with an invariant ``<th>`` header row; ``False``: headerless record
    table of repeated multi-cell rows; ``None``: not a record table (static or single cell)."""
    if acc.df < 2 or acc.rmax < 1 or max(acc.body.values()) >= BOILERPLATE_M * acc.df:
        return None
    if (
        acc.header_pages * 2 >= acc.df
        and _recurring_share(acc.header_df, acc.header_pages) >= KEY_RECUR_SHARE
    ):
        return True
    if acc.ncols >= 2 and acc.rmax >= 2:
        return False
    return None


def decide(survey: Survey) -> Plan:
    """Accept key-value and record tables and fix the invariant texts used for label search.

    A table with an invariant header row is a record table even when its rows have two cells;
    a headerless table whose two-cell rows form an accepted key-value structure is key-value.
    """
    kv = {s for s, acc in survey.kv.items() if _accept_kv(acc)}
    headered: set[str] = set()
    headerless: set[str] = set()
    for struct, t_acc in survey.tables.items():
        verdict = _table_header(t_acc)
        if verdict is True:
            headered.add(struct)
        elif verdict is False:
            headerless.add(struct)
    kv_tables = {t for s in kv if (t := survey.kv[s].table) is not None}
    kv -= {s for s in kv if survey.kv[s].table in headered}
    headerless -= kv_tables
    tables = headered | headerless
    single = survey.n == 1
    inv_paths = frozenset(
        path
        for path, acc in survey.paths.items()
        if (acc.df >= 2 or single) and max(acc.joined.values()) >= BOILERPLATE_M * acc.df
    )
    inv_pairs = frozenset(
        (path, h)
        for path, acc in survey.paths.items()
        if acc.multi and path not in inv_paths
        for h, count in acc.multi.items()
        if (total := count + acc.joined.get(h, 0)) >= 2 and total >= BOILERPLATE_M * acc.df
    )
    return Plan(
        structs=StructPlan(kv={s: s for s in kv}, tables={s: s for s in tables}),
        headered=frozenset(headered),
        inv_paths=inv_paths,
        inv_pairs=inv_pairs,
    )


# -- pass 2: collect ---------------------------------------------------------------------------------


@dataclass(slots=True)
class _CandAcc:
    df: int = 0
    rmin: int = _BIG
    rmax: int = 0
    n_items: int = 0
    first: tuple[int, int] = (_BIG, 0)  # (page_id, item index) of the first appearance
    named: bool = False
    label_pages: int = 0
    joined: Counter[int] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    roles: Counter[tuple[str, str]] = field(default_factory=Counter)
    labels: Counter[tuple[CandKey, str]] = field(default_factory=Counter)

    def add(self, texts: list[str], first: tuple[int, int]) -> None:
        n = len(texts)
        self.df += 1
        self.n_items += n
        self.rmin = min(self.rmin, n)
        self.rmax = max(self.rmax, n)
        self.joined[hash64(texts[0] if n == 1 else SEQ_SEP.join(texts))] += 1
        if first < self.first:
            self.first = first
        if len(self.examples) < N_EXAMPLES:
            for text in texts:
                clipped = _clip(text)
                if clipped not in self.examples:
                    self.examples.append(clipped)
                    if len(self.examples) == N_EXAMPLES:
                        break
        if len(self.samples) < TYPE_SAMPLES:
            room = min(TYPE_SAMPLES - len(self.samples), TYPE_SAMPLES_PER_PAGE)
            self.samples.extend(t[:200] for t in texts[:room])

    def merge(self, other: _CandAcc) -> None:
        self.df += other.df
        self.n_items += other.n_items
        self.rmin = min(self.rmin, other.rmin)
        self.rmax = max(self.rmax, other.rmax)
        self.first = min(self.first, other.first)
        self.named = self.named or other.named
        self.label_pages += other.label_pages
        self.joined.update(other.joined)
        for example in other.examples:
            if len(self.examples) >= N_EXAMPLES:
                break
            if example not in self.examples:
                self.examples.append(example)
        self.samples.extend(other.samples[: max(0, TYPE_SAMPLES - len(self.samples))])
        self.roles.update(other.roles)
        self.labels.update(other.labels)


def _clip(text: str) -> str:
    return text if len(text) <= EXAMPLE_CHARS else text[: EXAMPLE_CHARS - 1] + "…"


def candidate_key(path: str, q: Qualifier) -> CandKey:
    return (path, q.table_id, q.column) if is_split(q) else (path, None, None)


_ROLE_ONLY = ("kv_key", "record_header")


@dataclass(slots=True)
class Collect:
    """Pass-2 accumulator for one template (mergeable across chunks of pages)."""

    n: int = 0
    cands: dict[CandKey, _CandAcc] = field(default_factory=dict)
    base: tuple[int, int, int] | None = None
    """Best base page so far: (varying candidates present, candidates present, -page_id)."""

    @property
    def base_page_id(self) -> int | None:
        """The sampled page covering the most candidates (varying ones first)."""
        return None if self.base is None else -self.base[2]

    def add(self, page_id: int, page: PageWalk, plan: Plan) -> None:
        self.n += 1
        items = page.items
        quals = [resolve(item, plan.structs) for item in items]
        keys = [candidate_key(item.path, q) for item, q in zip(items, quals, strict=True)]
        groups: dict[CandKey, list[int]] = {}
        for i, key in enumerate(keys):
            idxs = groups.get(key)
            if idxs is None:
                groups[key] = [i]
            else:
                idxs.append(i)
        varying = sum(1 for k in groups if k[1] is not None or k[0] not in plan.inv_paths)
        cover = (varying, len(groups), -page_id)
        if self.base is None or cover > self.base:
            self.base = cover
        inv_cache: dict[int, bool] = {}

        def is_invariant(j: int, own: CandKey) -> bool:
            if keys[j] == own:
                return False
            cached = inv_cache.get(j)
            if cached is not None:
                return cached
            item = items[j]
            q = quals[j]
            if item.attr is not None or len(item.text) > KEY_MAX_CHARS or is_split(q):
                result = False
            elif q.role in _ROLE_ONLY:
                result = True
            else:
                result = item.path in plan.inv_paths or (
                    (item.path, hash64(item.text)) in plan.inv_pairs
                )
            inv_cache[j] = result
            return result

        for key, idxs in groups.items():
            acc = self.cands.get(key)
            if acc is None:
                acc = self.cands[key] = _CandAcc()
            acc.add([items[i].text for i in idxs], (page_id, idxs[0]))
            q = quals[idxs[0]]
            if q.role in _ROLE_ONLY:
                for i in idxs:
                    role = quals[i]
                    if role.role is not None and role.table_id is not None:
                        acc.roles[(role.role, role.table_id)] += 1
                continue
            if is_split(q) and q.named:
                acc.named = True
                continue
            if acc.label_pages >= LABEL_PAGES or (key[1] is None and key[0] in plan.inv_paths):
                continue
            acc.label_pages += 1
            j = find_label(page, idxs[0], partial(is_invariant, own=key))
            if j is not None:
                acc.labels[(keys[j], items[j].text)] += 1

    def merge(self, other: Collect) -> None:
        self.n += other.n
        if other.base is not None and (self.base is None or other.base > self.base):
            self.base = other.base
        for key, acc in other.cands.items():
            current = self.cands.get(key)
            if current is None:
                self.cands[key] = acc
            else:
                current.merge(acc)


# -- assembly ------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Draft:
    key: CandKey
    acc: _CandAcc
    order: int
    cls: CandidateClass
    role: TableRole | None = None
    struct: str | None = None
    column: str | None = None
    label_hint: str | None = None

    @property
    def cid(self) -> str:
        return f"c{self.order}"

    @property
    def is_attr(self) -> bool:
        return ATTR_SEP in self.key[0]


def _struct_kind(struct: str) -> str:
    return struct.split("|", 1)[0]


def _struct_path(struct: str) -> str:
    return struct.split("|", 1)[1]


def _parent_path(path: str) -> str:
    return path.rsplit(PATH_SEP, 1)[0]


def _drafts(collect: Collect, base_pos: dict[CandKey, int]) -> list[_Draft]:
    """Classified drafts numbered in document order of the base page, then by first appearance
    on the other pages (page id, position), then path."""
    n = max(collect.n, 1)

    def order_key(entry: tuple[CandKey, _CandAcc]) -> tuple[int, int, int, str, str, str]:
        key, acc = entry
        pos = base_pos.get(key)
        rank = (0, pos, 0) if pos is not None else (1, *acc.first)
        return (*rank, key[0], key[1] or "", key[2] or "")

    ordered = sorted(collect.cands.items(), key=order_key)
    drafts: list[_Draft] = []
    for order, (key, acc) in enumerate(ordered):
        p = acc.df / n
        m = max(acc.joined.values()) / acc.df
        draft = _Draft(key, acc, order, classify(p, m, acc.df, acc.rmin, acc.rmax))
        if key[1] is not None:
            draft.struct = key[1]
            draft.column = key[2]
            draft.role = "record_column" if _struct_kind(key[1]) == TABLE_KIND else "kv_value"
        elif acc.roles:
            (role, struct), count = acc.roles.most_common(1)[0]
            if 2 * count >= acc.n_items:
                draft.role = "kv_key" if role == "kv_key" else "record_header"
                draft.struct = struct
                if draft.cls != "noise":
                    draft.cls = "label"
        drafts.append(draft)
    return drafts


def _assign_labels(drafts: list[_Draft]) -> None:
    """Label hints; invariant texts serving as the hint of a field candidate become ``label``."""
    sources: set[CandKey] = set()
    for draft in drafts:
        if draft.cls not in FIELD_CLASSES and draft.cls != "noise":
            continue
        if draft.column is not None and draft.acc.named:
            draft.label_hint = draft.column
        elif draft.acc.labels:
            (source, text), _ = draft.acc.labels.most_common(1)[0]
            draft.label_hint = text
            if draft.cls in FIELD_CLASSES:
                sources.add(source)
    for draft in drafts:
        if draft.key in sources and draft.cls == "boilerplate":
            draft.cls = "label"


def _column_name(path: str) -> str:
    elem, _, attr = path.partition(ATTR_SEP)
    last = elem.rsplit(PATH_SEP, 1)[-1]
    try:
        step = parse_step(last)
    except ValueError:
        base = last.split(".", 1)[0].split("#", 1)[0].split("[", 1)[0]
    else:
        attrs = dict(step.attrs)
        base = (
            attrs.get("itemprop")
            or attrs.get("data-testid")
            or (step.classes[0] if step.classes else None)
            or step.id
            or step.tag
        )
    return f"{base}_{attr}" if attr else base


def _nearest_row(elem_path: str, rows: dict[str, _Range]) -> str | None:
    steps = elem_path.split(PATH_SEP)
    prefixes: list[str] = []
    current = ""
    for step in steps:
        current = step if not current else current + PATH_SEP + step
        prefixes.append(current)
    for prefix in reversed(prefixes):
        if prefix in rows:
            return prefix
    return None


@dataclass(slots=True)
class _TableDraft:
    struct: str
    table: Table
    members: list[_Draft]

    @property
    def order(self) -> int:
        return min((d.order for d in self.members), default=_BIG)


def _primary(drafts: list[_Draft]) -> _Draft:
    """The representative candidate of a column: text over attribute, most present, earliest."""
    return min(drafts, key=lambda d: (d.is_attr, -d.acc.df, d.order))


def _kv_table(struct: str, acc: _KvAcc, members: list[_Draft], n: int) -> Table:
    kind = _struct_kind(struct)
    layout: TableLayout = "dl" if kind == "dl" else "alternating" if kind == "alt" else "pairs"
    if kind == "pair":
        container, row_path = _parent_path(_struct_path(struct)), _struct_path(struct)
    else:
        container = _struct_path(struct)
        row_path = acc.row_paths.most_common(1)[0][0]
    keys = [d for d in members if d.role == "kv_key"]
    seen: list[str] = []
    for draft in members:
        if draft.role == "kv_value" and draft.column is not None and draft.column not in seen:
            seen.append(draft.column)
    rest = sorted((k for k in acc.key_df if k not in seen), key=lambda k: (-acc.key_df[k], k))
    return Table(
        id="",
        kind="key_value",
        layout=layout,
        container_path=container,
        row_path=row_path,
        p=round(acc.df / n, 4),
        rows_min=acc.rmin,
        rows_max=acc.rmax,
        key_candidate=_primary(keys).cid if keys else None,
        value_candidate=None,
        keys_seen=(seen + rest)[:200],
    )


def _record_table(
    struct: str, acc: _TableAcc, members: list[_Draft], n: int, headered: bool
) -> Table:
    names: list[str] = []
    if headered and acc.header_seq:
        for text in acc.header_seq.most_common(1)[0][0]:
            if text and text not in names:
                names.append(text)
    by_name: dict[str, list[_Draft]] = {}
    for draft in members:
        if draft.role == "record_column" and draft.column is not None:
            by_name.setdefault(draft.column, []).append(draft)
            if draft.column not in names:
                names.append(draft.column)
    columns: list[TableColumn] = []
    for name in names:
        group = by_name.get(name)
        if not group:
            continue
        best = _primary(group)
        columns.append(
            TableColumn(
                name=name, candidate_id=best.cid, header_source="th" if best.acc.named else "none"
            )
        )
    return Table(
        id="",
        kind="record_list",
        layout="table",
        container_path=_struct_path(struct),
        row_path=acc.row_paths.most_common(1)[0][0] if acc.row_paths else _struct_path(struct),
        p=round(acc.df / n, 4),
        rows_min=acc.rmin,
        rows_max=acc.rmax,
        columns=columns,
    )


def _row_tables(survey: Survey, drafts: list[_Draft], n: int) -> list[_TableDraft]:
    """Repeated sibling rows (``ul > li``, ``div.card``) holding two or more varying sub-paths."""
    rows = {p: r for p, r in survey.rows.items() if r.df >= 2 or survey.n == 1}
    groups: dict[str, list[_Draft]] = {}
    for draft in drafts:
        if draft.struct is not None or draft.cls not in FIELD_CLASSES or draft.acc.rmax < 2:
            continue
        row = _nearest_row(draft.key[0].split(ATTR_SEP, 1)[0], rows)
        if row is not None:
            groups.setdefault(row, []).append(draft)
    out: list[_TableDraft] = []
    for row_path, members in groups.items():
        if len({d.key[0].split(ATTR_SEP, 1)[0] for d in members}) < 2:
            continue
        struct = "rows|" + row_path
        used: set[str] = set()
        columns: list[TableColumn] = []
        for draft in sorted(members, key=lambda d: d.order):
            base = name = _column_name(draft.key[0])
            k = 2
            while name in used:
                name = f"{base}_{k}"
                k += 1
            used.add(name)
            draft.struct = struct
            draft.role = "record_column"
            draft.column = name
            columns.append(TableColumn(name=name, candidate_id=draft.cid, header_source="none"))
        r = rows[row_path]
        table = Table(
            id="",
            kind="record_list",
            layout="rows",
            container_path=_parent_path(row_path),
            row_path=row_path,
            p=round(r.df / n, 4),
            rows_min=r.rmin,
            rows_max=r.rmax,
            columns=columns,
        )
        out.append(_TableDraft(struct, table, members))
    return out


def base_positions(base_page: PageWalk | None, plan: Plan) -> dict[CandKey, int]:
    """Document position of each candidate's first node on the base page."""
    if base_page is None:
        return {}
    out: dict[CandKey, int] = {}
    for i, item in enumerate(base_page.items):
        out.setdefault(candidate_key(item.path, resolve(item, plan.structs)), i)
    return out


def build_table(
    template_id: str,
    cluster_size: int,
    survey: Survey,
    plan: Plan,
    collect: Collect,
    base_page: PageWalk | None,
) -> CandidateTable:
    """Classify the collected candidates, attach label hints and tables, number everything.

    ``base_page`` is the walk of ``collect.base_page_id``; candidate ids follow its document
    order."""
    n = max(collect.n, 1)
    n_survey = max(survey.n, 1)
    drafts = _drafts(collect, base_positions(base_page, plan))
    _assign_labels(drafts)
    members: dict[str, list[_Draft]] = {}
    for draft in drafts:
        if draft.struct is not None:
            members.setdefault(draft.struct, []).append(draft)
    table_drafts: list[_TableDraft] = []
    for struct in plan.structs.kv:
        group = members.get(struct, [])
        if group:
            table = _kv_table(struct, survey.kv[struct], group, n_survey)
            table_drafts.append(_TableDraft(struct, table, group))
    for struct in plan.structs.tables:
        group = members.get(struct, [])
        if group:
            headered = struct in plan.headered
            table = _record_table(struct, survey.tables[struct], group, n_survey, headered)
            table_drafts.append(_TableDraft(struct, table, group))
    table_drafts.extend(_row_tables(survey, drafts, n_survey))
    table_drafts.sort(key=lambda t: (t.order, t.struct))
    table_ids: dict[str, str] = {}
    tables: list[Table] = []
    for i, td in enumerate(table_drafts):
        table_ids[td.struct] = td.table.id = f"tb{i}"
        tables.append(td.table)
    candidates = [_candidate(d, n, table_ids) for d in drafts]
    return CandidateTable(
        template_id=template_id,
        cluster_size=cluster_size,
        n_sampled=collect.n,
        base_page_id=collect.base_page_id,
        candidates=candidates,
        tables=tables,
    )


def _candidate(draft: _Draft, n: int, table_ids: dict[str, str]) -> Candidate:
    acc = draft.acc
    path = draft.key[0]
    attr = path.rsplit(ATTR_SEP, 1)[1] if ATTR_SEP in path else None
    table_id = table_ids.get(draft.struct) if draft.struct is not None else None
    return Candidate(
        id=draft.cid,
        path=path,
        attr=attr,
        cls=draft.cls,
        p=round(acc.df / n, 4),
        m=round(max(acc.joined.values()) / acc.df, 4),
        pages_present=acc.df,
        repeat_min=acc.rmin,
        repeat_max=acc.rmax,
        label_hint=draft.label_hint,
        examples=acc.examples,
        value_type=infer_value_type(acc.samples) if draft.cls != "noise" else None,
        table_id=table_id,
        table_role=draft.role if table_id is not None else None,
        column=draft.column if table_id is not None else None,
    )


# -- in-process entry point -------------------------------------------------------------------------


def analyze_pages(
    template_id: str,
    pages: Sequence[tuple[int, str | bytes]],
    *,
    cluster_size: int,
) -> CandidateTable:
    """Analyze the sampled pages ``(page_id, html)`` of one template, in-process.

    Pass 1 (decisions) parses an evenly spaced subset of at most ``SURVEY_SAMPLE`` pages, pass 2
    every page, matching the parallel stage. Examples are first-seen in the given page order.
    """
    survey = Survey()
    for page_id, html in survey_subset(pages):
        survey.add(page_id, walk_html(html, for_labels=False))
    plan = decide(survey)
    collect = Collect()
    for page_id, html in pages:
        collect.add(page_id, walk_html(html), plan)
    base_id = collect.base_page_id
    base_html = next((html for page_id, html in pages if page_id == base_id), None)
    base_page = walk_html(base_html) if base_html is not None else None
    return build_table(template_id, cluster_size, survey, plan, collect, base_page)


def walk_html(html: str | bytes, *, for_labels: bool = True) -> PageWalk:
    """Clean and walk one page."""
    return walk(clean_html(html), for_labels=for_labels)


def survey_subset(items: Sequence[_T]) -> Sequence[_T]:
    """Evenly spaced subset of at most ``SURVEY_SAMPLE`` sampled pages for pass 1 (table and
    invariance decisions); pass 2 statistics use every sampled page."""
    step = math.ceil(len(items) / SURVEY_SAMPLE) if items else 1
    return items[::step] if step > 1 else items
