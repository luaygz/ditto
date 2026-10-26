"""Stage 3 step 5, applied after verification: merge and split clusters using the specs.

* **Merge** — cluster B joins cluster A when A's spec reaches its own presence rates on B's
  pages (every field within a sampling tolerance of its verified presence, and A has at least
  one anchor field present on >= 90% of its own pages). When B has a spec of its own, B's spec
  must also reach its rates on A's pages; B's fields that A lacks are carried over.
* **Split** — the pages of a signature group leave their cluster when a field's failures
  concentrate there: the field is present on < 50% of the group's pages while > 95% elsewhere.
  Groups with the same set of such fields form one new cluster (``split_from`` = parent); it
  starts from a copy of the parent's spec and is re-verified on its own pages.

Bounded: one split pass and one merge pass per verification run, merge evidence from at most
``MERGE_SAMPLE`` pages per cluster and the ``MAX_MERGE_CLUSTERS`` largest clusters.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ditto.analyze.models import CandidateTable
from ditto.cluster.models import Cluster, Clusters
from ditto.parallel import pmap
from ditto.specgen.models import Spec
from ditto.textutil import hash64

SPLIT_LOW = 0.5
SPLIT_HIGH = 0.95
MIN_SPLIT_PAGES = 3
MERGE_SAMPLE = 20
MERGE_ANCHOR = 0.9
MERGE_MIN_TOLERANCE = 0.05
MAX_MERGE_CLUSTERS = 500
MAX_URL_EXAMPLES = 10


# -- split ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPlan:
    template_id: str
    new_template_id: str
    group_ids: tuple[int, ...]
    page_ids: tuple[int, ...]
    fields: tuple[str, ...]
    evidence: dict[str, dict[str, float]] = field(default_factory=dict, hash=False)

    def event(self) -> dict[str, Any]:
        return {
            "action": "split",
            "template_id": self.template_id,
            "new_template_id": self.new_template_id,
            "signature_groups": list(self.group_ids),
            "pages": len(self.page_ids),
            "fields": list(self.fields),
            "evidence": self.evidence,
        }


def split_template_id(parent: str, sig_hashes: Sequence[str]) -> str:
    """Stable id for a cluster split off ``parent`` (``t_`` + 8 hex chars)."""
    digest = hash64(parent + ":" + ",".join(sorted(sig_hashes)))
    return f"t_{digest:016x}"[:10]


def plan_splits(
    template_id: str,
    page_ids: Sequence[int],
    groups: Sequence[int],
    presence: Mapping[str, np.ndarray[Any, np.dtype[np.bool_]]],
    sig_hash: Mapping[int, str],
    *,
    min_pages: int = MIN_SPLIT_PAGES,
) -> list[SplitPlan]:
    """Signature groups whose pages lack a field the rest of the cluster has.

    ``groups[i]`` is the signature group of ``page_ids[i]`` (-1 = unknown, never split);
    ``presence[field][i]`` says whether extraction filled the field on that page."""
    pids = np.asarray(page_ids, np.int64)
    gids = np.asarray(groups, np.int64)
    n = int(pids.shape[0])
    profiles: dict[tuple[str, ...], list[int]] = {}
    evidence: dict[str, dict[str, float]] = {}
    for g in sorted({int(x) for x in gids.tolist()} - {-1}):
        in_g = gids == g
        size = int(in_g.sum())
        if size < min_pages or n - size < min_pages:
            continue
        hit: list[str] = []
        for name, pres in presence.items():
            inside = float(pres[in_g].mean())
            outside = float(pres[~in_g].mean())
            if inside < SPLIT_LOW and outside > SPLIT_HIGH:
                hit.append(name)
                evidence[f"{g}:{name}"] = {
                    "in_group": round(inside, 4),
                    "elsewhere": round(outside, 4),
                }
        if hit:
            profiles.setdefault(tuple(sorted(hit)), []).append(g)
    plans: list[SplitPlan] = []
    remaining = n
    for fields, members in sorted(profiles.items()):
        mask = np.isin(gids, members)
        size = int(mask.sum())
        if remaining - size < min_pages:
            continue
        remaining -= size
        ev = {k: v for k, v in evidence.items() if int(k.split(":", 1)[0]) in members}
        plans.append(
            SplitPlan(
                template_id=template_id,
                new_template_id=split_template_id(
                    template_id, [sig_hash.get(g, str(g)) for g in members]
                ),
                group_ids=tuple(members),
                page_ids=tuple(int(p) for p in pids[mask]),
                fields=fields,
                evidence=ev,
            )
        )
    return plans


def describe_urls(
    page_ids: Sequence[int], url_of: Callable[[int], str] | None
) -> tuple[list[str], list[str]]:
    """URL patterns (most common first) and up to 10 example URLs of a page set."""
    if url_of is None or not page_ids:
        return [], []
    from ditto.cluster.urls import url_pattern

    urls = [url_of(p) for p in sorted(page_ids)]
    patterns = [p for p, _ in Counter(url_pattern(u) for u in urls).most_common(5)]
    return patterns, urls[:MAX_URL_EXAMPLES]


def _rebuild(data: Clusters) -> Clusters:
    """A fresh ``Clusters`` (its cached lookups reflect the edited membership)."""
    return Clusters.model_validate(data.model_dump())


def apply_splits(
    clusters: Clusters,
    plans: Sequence[SplitPlan],
    *,
    url_of: Callable[[int], str] | None = None,
) -> Clusters:
    data = clusters.model_copy(deep=True)
    by_id = {c.template_id: c for c in data.clusters}
    for plan in plans:
        parent = by_id[plan.template_id]
        moved = set(plan.page_ids)
        parent.page_ids = [p for p in parent.page_ids if p not in moved]
        parent.signature_groups = [g for g in parent.signature_groups if g not in plan.group_ids]
        parent.size = len(parent.page_ids)
        parent.is_tail = parent.size < data.min_cluster_size
        if url_of is not None:
            parent.url_patterns, parent.url_examples = describe_urls(parent.page_ids, url_of)
        patterns, examples = describe_urls(sorted(moved), url_of)
        child = Cluster(
            template_id=plan.new_template_id,
            size=len(moved),
            is_tail=len(moved) < data.min_cluster_size,
            signature_groups=list(plan.group_ids),
            url_patterns=patterns,
            url_examples=examples,
            page_ids=sorted(moved),
            split_from=parent.template_id,
        )
        data.clusters.append(child)
        by_id[child.template_id] = child
        for g in data.signature_groups:
            if g.id in plan.group_ids:
                g.template_id = child.template_id
    return _rebuild(data)


# -- merge ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MergePlan:
    target: str
    absorbed: str
    evidence: dict[str, Any] = field(default_factory=dict, hash=False)

    def event(self) -> dict[str, Any]:
        return {
            "action": "merge",
            "target": self.target,
            "absorbed": self.absorbed,
            "evidence": self.evidence,
        }


def tolerance(p: float, n: int) -> float:
    """Allowed shortfall of a presence rate measured on ``n`` sampled pages (two binomial
    standard errors, at least 0.05)."""
    return max(MERGE_MIN_TOLERANCE, 2.0 * math.sqrt(p * (1.0 - p) / max(n, 1)))


def own_presence(spec: Spec) -> dict[str, float]:
    return {name: float(f.stats.presence_rate or 0.0) for name, f in spec.fields.items()}


def reaches(own: Mapping[str, float], observed: Mapping[str, float], n: int) -> bool:
    """True when every field's observed presence is within tolerance of its own rate and the
    spec has an anchor field (own presence >= 0.9)."""
    if n <= 0 or not own or max(own.values()) < MERGE_ANCHOR:
        return False
    return all(observed.get(f, 0.0) >= p - tolerance(p, n) for f, p in own.items())


_STATE: dict[str, Any] = {}


def _init(run_root: str, specs: tuple[tuple[str, Spec, str], ...]) -> None:
    _STATE["root"] = Path(run_root)
    _STATE["specs"] = specs


def _presence_chunk(
    items: Sequence[tuple[int, str]],
) -> list[tuple[str, dict[str, tuple[bool, ...] | None]]]:
    from ditto.extract.runner import field_value, is_missing
    from ditto.fingerprint.dom import clean_html
    from ditto.ingest import read_html

    root: Path = _STATE["root"]
    specs: tuple[tuple[str, Spec, str], ...] = _STATE["specs"]
    out: list[tuple[str, dict[str, tuple[bool, ...] | None]]] = []
    for pid, cluster_tid in items:
        tree = clean_html(read_html(root, pid))
        per: dict[str, tuple[bool, ...] | None] = {}
        for tid, spec, anchor in specs:
            if tid == cluster_tid:
                continue
            if anchor and is_missing(field_value(spec.fields[anchor], tree)):
                per[tid] = None  # anchor absent: the spec does not apply to this page
                continue
            per[tid] = tuple(not is_missing(field_value(f, tree)) for f in spec.fields.values())
        out.append((cluster_tid, per))
    return out


def spec_presence(
    run_root: Path,
    specs: Mapping[str, Spec],
    samples: Mapping[str, Sequence[int]],
    *,
    workers: int | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Field presence of every spec on every other cluster's sampled pages:
    ``(spec template, cluster template) -> field -> rate``."""
    entries = []
    for tid, spec in specs.items():
        own = own_presence(spec)
        anchor = max(own, key=lambda k: (own[k], k)) if own else ""
        entries.append((tid, spec, anchor))
    items = [(int(pid), ctid) for ctid, pids in samples.items() for pid in pids]
    sums: dict[tuple[str, str], np.ndarray[Any, np.dtype[np.int64]]] = {}
    for part in pmap(
        _presence_chunk,
        items,
        workers=workers,
        chunk_size=32,
        initializer=_init,
        initargs=(str(run_root), tuple(entries)),
    ):
        for ctid, per in part:
            for stid, flags in per.items():
                key = (stid, ctid)
                if key not in sums:
                    sums[key] = np.zeros(len(specs[stid].fields), np.int64)
                if flags is not None:
                    sums[key] += np.asarray(flags, np.int64)
    rates: dict[tuple[str, str], dict[str, float]] = {}
    for (stid, ctid), counts in sums.items():
        n = max(len(samples[ctid]), 1)
        names = list(specs[stid].fields)
        rates[(stid, ctid)] = {name: float(c) / n for name, c in zip(names, counts, strict=True)}
    return rates


def merge_samples(
    clusters: Clusters,
    *,
    templates: set[str] | None = None,
    sample: int = MERGE_SAMPLE,
    max_clusters: int = MAX_MERGE_CLUSTERS,
) -> dict[str, list[int]]:
    """Sampled page ids per cluster considered for merging (largest clusters first)."""
    pool = clusters.templates(include_tail=True)
    if templates is not None:
        pool = [c for c in pool if c.template_id in templates]
    return {
        c.template_id: clusters.sample(c.template_id, sample, seed=0)
        for c in pool[:max_clusters]
        if c.size > 0
    }


def plan_merges(
    clusters: Clusters,
    specs: Mapping[str, Spec],
    rates: Mapping[tuple[str, str], Mapping[str, float]],
    samples: Mapping[str, Sequence[int]],
) -> list[MergePlan]:
    """Choose merges: each cluster B joins at most one target A (the largest qualifying);
    a target is never absorbed and an absorbed cluster never absorbs."""
    own = {tid: own_presence(spec) for tid, spec in specs.items()}
    size = {c.template_id: c.size for c in clusters.clusters}
    targets = [t for t in specs if own[t] and max(own[t].values()) >= MERGE_ANCHOR]
    absorbed: set[str] = set()
    used: set[str] = set()
    plans: list[MergePlan] = []
    for b in sorted(samples, key=lambda t: (size.get(t, 0), t)):
        if b in used:
            continue
        options = []
        for a in targets:
            if a == b or a in absorbed or a not in samples:
                continue
            if (size.get(a, 0), a) < (size.get(b, 0), b):
                continue  # the larger cluster absorbs; the pair is seen from the other side
            if not reaches(own[a], rates.get((a, b), {}), len(samples[b])):
                continue
            if b in specs and not reaches(own[b], rates.get((b, a), {}), len(samples[a])):
                continue
            options.append(a)
        if not options:
            continue
        a = max(options, key=lambda t: (size.get(t, 0), t))
        evidence = {
            "pages_sampled": len(samples[b]),
            "own_presence": {k: round(v, 4) for k, v in own[a].items()},
            "observed_presence": {k: round(v, 4) for k, v in rates.get((a, b), {}).items()},
        }
        plans.append(MergePlan(target=a, absorbed=b, evidence=evidence))
        absorbed.add(b)
        used.add(a)
    return plans


def apply_merges(clusters: Clusters, plans: Sequence[MergePlan]) -> Clusters:
    data = clusters.model_copy(deep=True)
    by_id = {c.template_id: c for c in data.clusters}
    for plan in plans:
        a, b = by_id[plan.target], by_id.pop(plan.absorbed)
        a.page_ids = sorted(a.page_ids + b.page_ids)
        a.signature_groups = a.signature_groups + b.signature_groups
        a.size = len(a.page_ids)
        a.is_tail = a.size < data.min_cluster_size
        a.merged_from = list(dict.fromkeys([*a.merged_from, b.template_id, *b.merged_from]))
        a.url_patterns = list(dict.fromkeys([*a.url_patterns, *b.url_patterns]))
        a.url_examples = (a.url_examples + b.url_examples)[:MAX_URL_EXAMPLES]
        for g in data.signature_groups:
            if g.template_id == b.template_id:
                g.template_id = a.template_id
    data.clusters = [c for c in data.clusters if c.template_id in by_id]
    return _rebuild(data)


def translate_sources(
    ids: Sequence[str], source: CandidateTable | None, target: CandidateTable | None
) -> list[str]:
    """Map candidate ids of one template's table to the other's by structural path."""
    if source is None or target is None:
        return []
    by_path = {c.path: c.id for c in target.candidates}
    src = {c.id: c for c in source.candidates}
    return [by_path[src[i].path] for i in ids if i in src and src[i].path in by_path]


def merge_spec_fields(
    target: Spec,
    absorbed: Spec,
    target_table: CandidateTable | None,
    absorbed_table: CandidateTable | None,
) -> tuple[Spec, list[str]]:
    """Carry the absorbed spec's fields that the target lacks into the target spec."""
    fields = dict(target.fields)
    not_present = dict(target.not_present)
    added: list[str] = []
    for name, f in absorbed.fields.items():
        if name in fields:
            continue
        sources = translate_sources(f.source_candidates, absorbed_table, target_table)
        fields[name] = f.model_copy(update={"source_candidates": sources})
        not_present.pop(name, None)
        added.append(name)
    return target.model_copy(update={"fields": fields, "not_present": not_present}), added
