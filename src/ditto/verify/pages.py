"""Evaluate selector probes over every page of a cluster, in parallel.

A *probe* is one thing measured on every page:

* ``PathProbe`` — how many nodes sit at an analysis candidate's exact structural path (with
  text). This is "the analysis says the path is present on this page".
* ``SelectorProbe`` — one candidate selector of a ``scalar``/``list`` field: nodes matched,
  values extracted (after transforms), type-check pass, a hash of the value(s) for agreement.
* ``RowsProbe`` — one container/pairs selector of a ``record_list``/``key_value`` field (the
  field variant carries the selector): rows extracted, typed-cell pass rate, a hash of the rows,
  the real sibling count of the matched rows (on sampled pages), and the keys seen.

Every page is parsed and cleaned once (``clean_html``) and all probes run on that tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from lxml import etree

from ditto.extract.runner import evaluate, key_values, record_rows
from ditto.fingerprint.dom import ATTR_SEP, PATH_SEP, clean_html, own_text, step_label
from ditto.parallel import pmap, start_method
from ditto.specgen.models import KeyValueField, RecordListField
from ditto.specgen.transforms import apply_transforms, to_text
from ditto.specgen.typecheck import check_value
from ditto.textutil import hash64, normalize_ws

PAGES_PER_CHUNK = 64
# Without fork, a worker pool costs a process start plus imports per template: templates this
# small are verified in-process instead.
SPAWN_MIN_PAGES = 500
NO_VALUE = -1.0


@dataclass(frozen=True)
class PathProbe:
    path: str


@dataclass(frozen=True)
class SelectorProbe:
    xpath: str
    transforms: tuple[str, ...] = ()
    value_type: str | None = None
    plural: bool = False


@dataclass(frozen=True)
class RowsProbe:
    field: RecordListField | KeyValueField
    column_types: tuple[tuple[str, str], ...] = ()


Probe = PathProbe | SelectorProbe | RowsProbe


@dataclass(frozen=True)
class Observation:
    """One probe on one page."""

    nodes: int = 0
    values: int = 0
    type_ok: float = NO_VALUE
    vhash: int = 0
    siblings: int = -1
    keys: tuple[str, ...] = ()


@dataclass
class ProbeMatrix:
    """Observations of P probes on N pages (row i = ``page_ids[i]``)."""

    page_ids: np.ndarray[Any, np.dtype[np.int64]]
    nodes: np.ndarray[Any, np.dtype[np.int32]]
    values: np.ndarray[Any, np.dtype[np.int32]]
    type_ok: np.ndarray[Any, np.dtype[np.float32]]
    vhash: np.ndarray[Any, np.dtype[np.uint64]]
    siblings: np.ndarray[Any, np.dtype[np.int32]]
    keys: list[set[str]]

    @property
    def n_pages(self) -> int:
        return int(self.page_ids.shape[0])


# -- single-page observations ---------------------------------------------------------------


def _split_attr(path: str) -> tuple[str, str | None]:
    base, sep, attr = path.partition(ATTR_SEP)
    return base, (attr or None) if sep else None


def nodes_at_paths(paths: Sequence[str], tree: etree._Element) -> list[list[str]]:
    """Per path, the texts (or attribute values) of the nodes whose structural path is
    exactly that path and that carry text — the analysis's notion of "present".

    One walk of the page computes structural paths with ``step_label`` (the fingerprint's own
    definition), descending only into subtrees that lead to a wanted path."""
    wanted: dict[str, list[tuple[int, str | None]]] = {}
    prefixes: set[str] = set()
    for j, path in enumerate(paths):
        base, attr = _split_attr(path)
        wanted.setdefault(base, []).append((j, attr))
        parts = base.split(PATH_SEP)
        prefixes.update(PATH_SEP.join(parts[: i + 1]) for i in range(len(parts)))
    out: list[list[str]] = [[] for _ in paths]
    if not wanted:
        return out
    stack: list[tuple[etree._Element, str]] = [(tree, step_label(tree))]
    while stack:
        el, path = stack.pop()
        if path not in prefixes:
            continue
        for j, attr in wanted.get(path, ()):
            text = (el.get(attr) or "").strip() if attr is not None else own_text(el)
            if text:
                out[j].append(text)
        for child in reversed(el):
            if isinstance(child.tag, str):
                stack.append((child, path + PATH_SEP + step_label(child)))
    return out


def observe_path(path: str, tree: etree._Element) -> int:
    """Number of nodes at exactly ``path`` that carry text (or the attribute)."""
    return len(nodes_at_paths([path], tree)[0])


def path_texts(path: str, tree: etree._Element, limit: int = 3) -> list[str]:
    """Texts at exactly ``path``: the values the analysis expects a field mapped to that
    candidate to produce on this page."""
    return nodes_at_paths([path], tree)[0][:limit]


def _norm(value: Any) -> str:
    text = value if isinstance(value, str) else to_text(value)
    return normalize_ws(text or "").casefold()


def observe_selector(probe: SelectorProbe, tree: etree._Element) -> Observation:
    items = evaluate(probe.xpath, tree)
    if not items:
        return Observation()
    transforms = list(probe.transforms)
    values = [v for v in (apply_transforms(it, transforms) for it in items) if v is not None]
    if not values:
        return Observation(nodes=len(items))
    if probe.plural:
        ok = sum(check_value(probe.value_type, v) for v in values) / len(values)
        digest = hash64("\x1f".join(_norm(v) for v in values))
    else:
        ok = float(check_value(probe.value_type, values[0]))
        digest = hash64(_norm(values[0]))
    return Observation(nodes=len(items), values=len(values), type_ok=ok, vhash=digest)


def _row_signature(el: etree._Element) -> tuple[str, frozenset[str]]:
    return step_label(el), frozenset(str(c.tag) for c in el if isinstance(c.tag, str))


def sibling_count(rows: Sequence[etree._Element]) -> int:
    """Real number of row siblings: children of the rows' parents that look like a matched row
    (same structural step and same set of child tags), so header rows do not count."""
    signatures = {_row_signature(r) for r in rows}
    parents: dict[int, etree._Element] = {}
    for r in rows:
        parent = r.getparent()
        if parent is not None:
            parents[id(parent)] = parent
    return sum(
        1
        for parent in parents.values()
        for child in parent
        if isinstance(child.tag, str) and _row_signature(child) in signatures
    )


def observe_rows(probe: RowsProbe, tree: etree._Element, sampled: bool) -> Observation:
    field = probe.field
    xpath = field.container if isinstance(field, RecordListField) else field.pairs
    elements = [e for e in evaluate(xpath, tree) if isinstance(e, etree._Element)]
    if not elements:
        return Observation()
    siblings = sibling_count(elements) if sampled else -1
    if isinstance(field, RecordListField):
        rows = record_rows(field, tree)
        if not rows:
            return Observation(nodes=len(elements), siblings=siblings)
        total = ok = 0
        for row in rows:
            for name, value_type in probe.column_types:
                value = row.get(name)
                if value is not None:
                    total += 1
                    ok += check_value(value_type, value)
        type_ok = ok / total if total else 1.0
        digest = hash64(orjson.dumps(rows, option=orjson.OPT_SORT_KEYS).decode())
        return Observation(len(elements), len(rows), type_ok, digest, siblings)
    pairs = key_values(field, tree)
    if not pairs:
        return Observation(nodes=len(elements), siblings=siblings)
    filled = sum(v is not None for v in pairs.values()) / len(pairs)
    digest = hash64(orjson.dumps(pairs, option=orjson.OPT_SORT_KEYS).decode())
    return Observation(len(elements), len(pairs), filled, digest, siblings, tuple(pairs))


def observe(probe: Probe, tree: etree._Element, sampled: bool = False) -> Observation:
    if isinstance(probe, PathProbe):
        n = observe_path(probe.path, tree)
        return Observation(nodes=n, values=n)
    if isinstance(probe, SelectorProbe):
        return observe_selector(probe, tree)
    return observe_rows(probe, tree, sampled)


# -- parallel evaluation ----------------------------------------------------------------------

_STATE: dict[str, Any] = {}
MAX_KEYS = 2000


def _init(run_root: str, probes: tuple[Probe, ...]) -> None:
    _STATE["root"] = Path(run_root)
    _STATE["probes"] = probes


def load_tree(run_root: Path, page_id: int) -> etree._Element:
    from ditto.ingest import read_html

    tree: etree._Element = clean_html(read_html(run_root, page_id))
    return tree


def _eval_chunk(items: Sequence[tuple[int, bool]]) -> ProbeMatrix:
    probes: tuple[Probe, ...] = _STATE["probes"]
    root: Path = _STATE["root"]
    n, p = len(items), len(probes)
    out = _empty(n, p)
    path_probes = [(j, pr.path) for j, pr in enumerate(probes) if isinstance(pr, PathProbe)]
    path_cols = [j for j, _ in path_probes]
    paths = [path for _, path in path_probes]
    other = [j for j, probe in enumerate(probes) if not isinstance(probe, PathProbe)]
    for i, (pid, sampled) in enumerate(items):
        out.page_ids[i] = pid
        tree = load_tree(root, pid)
        for j, found in zip(path_cols, nodes_at_paths(paths, tree), strict=True):
            out.nodes[i, j] = out.values[i, j] = len(found)
        for j in other:
            obs = observe(probes[j], tree, sampled)
            out.nodes[i, j] = obs.nodes
            out.values[i, j] = obs.values
            out.type_ok[i, j] = obs.type_ok
            out.vhash[i, j] = obs.vhash
            out.siblings[i, j] = obs.siblings
            if obs.keys and len(out.keys[j]) < MAX_KEYS:
                out.keys[j].update(obs.keys)
    return out


def _empty(n: int, p: int) -> ProbeMatrix:
    return ProbeMatrix(
        page_ids=np.zeros(n, np.int64),
        nodes=np.zeros((n, p), np.int32),
        values=np.zeros((n, p), np.int32),
        type_ok=np.full((n, p), NO_VALUE, np.float32),
        vhash=np.zeros((n, p), np.uint64),
        siblings=np.full((n, p), -1, np.int32),
        keys=[set() for _ in range(p)],
    )


def evaluate_probes(
    run_root: Path,
    page_ids: Sequence[int],
    probes: Sequence[Probe],
    *,
    sampled: frozenset[int] = frozenset(),
    workers: int | None = None,
) -> ProbeMatrix:
    """Run every probe on every page (parse + clean once per page) with ``pmap``."""
    items = [(int(pid), pid in sampled) for pid in page_ids]
    if not items or not probes:
        out = _empty(len(items), len(probes))
        out.page_ids[:] = [pid for pid, _ in items]
        return out
    if len(items) < SPAWN_MIN_PAGES and start_method() != "fork":
        workers = 1
    parts = list(
        pmap(
            _eval_chunk,
            items,
            workers=workers,
            chunk_size=PAGES_PER_CHUNK,
            initializer=_init,
            initargs=(str(run_root), tuple(probes)),
        )
    )
    keys: list[set[str]] = [set() for _ in probes]
    for part in parts:
        for j, ks in enumerate(part.keys):
            if len(keys[j]) < MAX_KEYS:
                keys[j].update(ks)
    return ProbeMatrix(
        page_ids=np.concatenate([p.page_ids for p in parts]),
        nodes=np.concatenate([p.nodes for p in parts]),
        values=np.concatenate([p.values for p in parts]),
        type_ok=np.concatenate([p.type_ok for p in parts]),
        vhash=np.concatenate([p.vhash for p in parts]),
        siblings=np.concatenate([p.siblings for p in parts]),
        keys=keys,
    )
