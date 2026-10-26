"""Stage 3, steps 1-4: group pages into templates by structural signature.

1. Pages with the same signature form a *signature group*.
2. Paths present on more than ``chrome_df`` of pages (site chrome) are dropped.
3. A 128-permutation MinHash per group over the remaining paths; ``datasketch`` LSH proposes
   candidate pairs; exact Jaccard on the filtered path sets confirms pairs at or above
   ``threshold``; union-find yields clusters.
4. Pairs in the ``[band_low, threshold)`` band merge only when both sides' dominant URL patterns
   agree (and that pattern does not cover nearly the whole site).

Chrome fallback: when one template dominates a site (a single-template crawl, say) its own
structure is on more than 90% of pages and step 2 would strip it, leaving its pages comparable
only on their optional parts. So a pair whose non-chrome paths make up less than
``CHROME_FALLBACK_SHARE`` of its path union is judged on full signatures instead; a second LSH
index over full signatures proposes such pairs among the groups that are mostly chrome.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.cluster.urls import url_pattern
from ditto.parallel import pmap

U64 = NDArray[np.uint64]
I64 = NDArray[np.int64]

# A pair is judged on full signatures when its non-chrome union is below this share of its union.
CHROME_FALLBACK_SHARE = 0.25
# Groups whose own (non-chrome) paths are below this share also go into the full-signature index.
CHROME_FALLBACK_INDEX_SHARE = 0.5
# Candidates whose MinHash estimate is this far below ``band_low`` skip the exact check
# (about 3.5 standard errors at 128 permutations).
ESTIMATE_MARGIN = 0.15
# A URL pattern shared by more than this share of all pages says nothing about templates.
URL_PATTERN_MAX_SHARE = 0.9
URL_EXAMPLES = 10
URL_PATTERNS_KEPT = 10
ID_PREFIX = "t_"
ID_HEX = 4


@dataclass(frozen=True)
class ClusterParams:
    min_cluster_size: int = 20
    threshold: float = 0.75
    band_low: float = 0.6
    chrome_df: float = 0.9
    num_perm: int = 128
    seed: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.band_low <= self.threshold <= 1.0:
            raise ValueError("need 0 < band_low <= threshold <= 1")
        if not 0.0 < self.chrome_df <= 1.0:
            raise ValueError("chrome_df must be in (0, 1]")
        if self.num_perm < 16:
            raise ValueError("num_perm must be at least 16")


@dataclass(frozen=True)
class PairEdge:
    """An exact-checked pair of signature groups."""

    a: int
    b: int
    jaccard: float
    full_signature: bool  # judged on full signatures (chrome fallback)


@dataclass
class ClusterStats:
    n_pages: int = 0
    n_groups: int = 0
    n_paths: int = 0
    chrome_paths: list[tuple[int, float]] = field(default_factory=list)  # (path hash, df)
    n_fallback_groups: int = 0
    lsh_bands: int = 0
    lsh_rows: int = 0
    n_candidates: int = 0
    n_exact: int = 0
    strong_edges: int = 0
    band_merges: list[tuple[PairEdge, str]] = field(default_factory=list)
    band_rejects: list[tuple[PairEdge, str, str]] = field(default_factory=list)
    nearest: dict[str, tuple[str, float]] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class ClusterResult:
    clusters: Clusters
    stats: ClusterStats


@lru_cache(maxsize=16)
def _lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """datasketch's optimal (bands, rows) for a threshold. Computing it integrates the LSH
    S-curve for every (b, r) pair (about a second), so it is done once per process."""
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    return int(lsh.b), int(lsh.r)


def _identity(x: Any) -> Any:
    return x


def _patterns_chunk(urls: Sequence[str]) -> list[str]:
    return [url_pattern(u) for u in urls]


class _Components:
    """Union-find over group ids with explicit member lists (small-to-large relabeling), so the
    label array can filter candidate arrays without per-item ``find`` calls."""

    def __init__(self, n: int) -> None:
        self.label = np.arange(n, dtype=np.int64)
        self.members: dict[int, list[int]] = {i: [i] for i in range(n)}

    def union(self, a: int, b: int) -> int:
        la, lb = int(self.label[a]), int(self.label[b])
        if la == lb:
            return la
        if len(self.members[la]) < len(self.members[lb]):
            la, lb = lb, la
        moved = self.members.pop(lb)
        self.label[moved] = la
        self.members[la].extend(moved)
        return la


def _inter_union(a: I64, b: I64) -> tuple[int, int]:
    """(|a ∩ b|, |a ∪ b|) of two sorted unique id arrays."""
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    return inter, len(a) + len(b) - inter


def _top_pattern(counter: Counter[str]) -> str:
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counter else ""


def cluster_signatures(
    sig_hashes: U64,
    signature_of: Callable[[int], U64],
    urls: Sequence[str],
    params: ClusterParams | None = None,
    *,
    workers: int = 1,
) -> ClusterResult:
    """Cluster pages given per-page signature hashes, a page-id -> signature accessor (called
    once per signature group) and per-page URLs. Pure computation, deterministic."""
    from datasketch import MinHash, MinHashLSH  # deferred: pulls in scipy (~1 s to import)

    p = params or ClusterParams()
    stats = ClusterStats()
    t0 = time.perf_counter()
    n = len(sig_hashes)
    stats.n_pages = n
    if len(urls) != n:
        raise ValueError(f"{n} signatures but {len(urls)} urls")

    # -- 1. exact signature groups, numbered by their first page ------------------------------
    uniq, first, inverse, counts = np.unique(
        np.asarray(sig_hashes, dtype=np.uint64),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    page_group = rank[inverse.reshape(-1)]
    g_first = first[order]
    g_size = counts[order].astype(np.int64)
    g_hash = uniq[order]
    n_groups = len(order)
    stats.n_groups = n_groups
    pages_by_group = np.argsort(page_group, kind="stable")
    bounds = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(g_size, out=bounds[1:])

    def pages_of(g: int) -> I64:
        return pages_by_group[bounds[g] : bounds[g + 1]]

    # -- 2. chrome: paths on more than chrome_df of pages -------------------------------------
    sigs = [np.asarray(signature_of(int(pid)), dtype=np.uint64) for pid in g_first]
    lens = np.array([len(s) for s in sigs], dtype=np.int64)
    all_paths = np.concatenate(sigs) if sigs else np.empty(0, dtype=np.uint64)
    vocab, path_ids = np.unique(all_paths, return_inverse=True)
    path_ids = path_ids.reshape(-1).astype(np.int64)
    stats.n_paths = len(vocab)
    weights = np.repeat(g_size, lens).astype(np.float64)
    df = np.bincount(path_ids, weights=weights, minlength=len(vocab)) / max(n, 1)
    chrome = df > p.chrome_df
    chrome_order = np.flatnonzero(chrome)
    stats.chrome_paths = [(int(vocab[i]), float(df[i])) for i in chrome_order]
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    full_ids = [path_ids[offsets[g] : offsets[g + 1]] for g in range(n_groups)]
    filt_ids = [ids[~chrome[ids]] for ids in full_ids]
    own_share = np.array(
        [len(f) / len(a) if len(a) else 0.0 for f, a in zip(filt_ids, full_ids, strict=True)]
    )
    fallback = own_share < CHROME_FALLBACK_INDEX_SHARE
    stats.n_fallback_groups = int(fallback.sum())
    stats.seconds["groups_chrome"] = time.perf_counter() - t0

    # -- 3. MinHash + LSH proposals, exact Jaccard, union-find --------------------------------
    t1 = time.perf_counter()
    mh_kwargs: dict[str, Any] = {
        "num_perm": p.num_perm,
        "seed": p.seed,
        "scheme": "affine64",
        "hashfunc": _identity,
    }
    mh_filt = MinHash.bulk([vocab[ids] for ids in filt_ids], **mh_kwargs)
    fb_groups = np.flatnonzero(fallback)
    mh_full = dict(
        zip(
            (int(g) for g in fb_groups),
            MinHash.bulk([vocab[full_ids[g]] for g in fb_groups], **mh_kwargs),
            strict=True,
        )
    )
    bands = _lsh_params(p.band_low, p.num_perm)
    lsh_filt = MinHashLSH(threshold=p.band_low, num_perm=p.num_perm, params=bands)
    lsh_full = MinHashLSH(threshold=p.band_low, num_perm=p.num_perm, params=bands)
    stats.lsh_bands, stats.lsh_rows = lsh_filt.b, lsh_filt.r
    with lsh_filt.insertion_session() as session:
        for g, mh in enumerate(mh_filt):
            session.insert(g, mh, check_duplication=False)
    with lsh_full.insertion_session() as session:
        for g, mh in mh_full.items():
            session.insert(g, mh, check_duplication=False)
    hv_filt = (
        np.stack([m.hashvalues for m in mh_filt])
        if mh_filt
        else np.empty((0, p.num_perm), np.uint64)
    )
    hv_full = np.full((n_groups, p.num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for g, mh in mh_full.items():
        hv_full[g] = mh.hashvalues
    stats.seconds["minhash_lsh"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    comps = _Components(n_groups)
    band_edges: list[PairEdge] = []
    checked: list[PairEdge] = []
    cutoff = p.band_low - ESTIMATE_MARGIN
    for g in range(n_groups):
        found = set(lsh_filt.query(mh_filt[g]))
        if fallback[g]:
            found.update(lsh_full.query(mh_full[g]))
        cands = np.array(sorted(c for c in found if c > g), dtype=np.int64)
        stats.n_candidates += len(cands)
        if not len(cands):
            continue
        cands = cands[comps.label[cands] != comps.label[g]]
        if not len(cands):
            continue
        est_f = np.count_nonzero(hv_filt[cands] == hv_filt[g], axis=1) / p.num_perm
        keep = est_f >= cutoff
        if fallback[g]:
            both = fallback[cands]
            est_full = np.count_nonzero(hv_full[cands] == hv_full[g], axis=1) / p.num_perm
            keep |= both & (est_full >= cutoff)
        for c in cands[keep]:
            c = int(c)
            if comps.label[c] == comps.label[g]:
                continue
            stats.n_exact += 1
            inter_f, union_f = _inter_union(filt_ids[g], filt_ids[c])
            inter_a, union_a = _inter_union(full_ids[g], full_ids[c])
            use_full = union_a > 0 and union_f < CHROME_FALLBACK_SHARE * union_a
            if not use_full:
                inter_a, union_a = inter_f, union_f
            jac = inter_a / union_a if union_a else 1.0
            edge = PairEdge(g, c, jac, use_full)
            if jac >= p.threshold:
                comps.union(g, c)
                stats.strong_edges += 1
            elif jac >= p.band_low:
                band_edges.append(edge)
            elif jac >= p.band_low - 0.1:
                checked.append(edge)
    stats.seconds["pairs"] = time.perf_counter() - t2

    # -- 4. URL patterns break ties in the band ----------------------------------------------
    t3 = time.perf_counter()
    patterns: list[str] = []
    chunk = max(1, -(-n // max(workers, 1)))
    use_workers = workers if n >= 20_000 else 1
    for part in pmap(_patterns_chunk, list(urls), workers=use_workers, chunk_size=chunk):
        patterns.extend(part)
    site_patterns = Counter(patterns)

    def group_patterns(g: int) -> Counter[str]:
        return Counter(patterns[int(pid)] for pid in pages_of(g))

    comp_patterns: dict[int, Counter[str]] = {}

    def patterns_of_label(label: int) -> Counter[str]:
        cnt = comp_patterns.get(label)
        if cnt is None:
            cnt = Counter()
            for g in comps.members[label]:
                cnt.update(group_patterns(g))
            comp_patterns[label] = cnt
        return cnt

    band_edges.sort(key=lambda e: (-e.jaccard, e.a, e.b))
    for edge in band_edges:
        la, lb = int(comps.label[edge.a]), int(comps.label[edge.b])
        if la == lb:
            continue
        pa, pb = _top_pattern(patterns_of_label(la)), _top_pattern(patterns_of_label(lb))
        informative = site_patterns[pa] <= URL_PATTERN_MAX_SHARE * n
        if pa == pb and informative:
            merged = patterns_of_label(la) + patterns_of_label(lb)
            comp_patterns.pop(la, None)
            comp_patterns.pop(lb, None)
            comp_patterns[comps.union(edge.a, edge.b)] = merged
            stats.band_merges.append((edge, pa))
        else:
            stats.band_rejects.append((edge, pa, pb))
    stats.seconds["url_band"] = time.perf_counter() - t3

    # -- clusters, ids, output -----------------------------------------------------------------
    t4 = time.perf_counter()
    comp_groups = [sorted(members) for members in comps.members.values()]
    hex_of = [f"{int(h):016x}" for h in g_hash]

    def base_hex(groups: list[int]) -> str:
        top = min(groups, key=lambda g: (-int(g_size[g]), hex_of[g]))
        return hex_of[top]

    comp_info = [(groups, int(g_size[groups].sum()), base_hex(groups)) for groups in comp_groups]
    comp_info.sort(key=lambda t: (-t[1], t[2]))
    used: set[str] = set()
    group_tid: dict[int, str] = {}
    clusters: list[Cluster] = []
    for groups, size, hx in comp_info:
        tid = _assign_id(hx, used)
        page_ids: list[int] = np.sort(np.concatenate([pages_of(g) for g in groups])).tolist()
        page_patterns = [patterns[pid] for pid in page_ids]
        pats = Counter(page_patterns)
        top = sorted(pats.items(), key=lambda kv: (-kv[1], kv[0]))[:URL_PATTERNS_KEPT]
        first_url: dict[str, str] = {}
        for pid, pat in zip(page_ids, page_patterns, strict=True):
            first_url.setdefault(pat, urls[pid])
        examples = [first_url[pat] for pat, _ in top]
        seen = set(examples)
        for pid in page_ids:
            if len(examples) >= URL_EXAMPLES:
                break
            if urls[pid] not in seen:
                examples.append(urls[pid])
                seen.add(urls[pid])
        for g in groups:
            group_tid[g] = tid
        clusters.append(
            Cluster(
                template_id=tid,
                size=size,
                is_tail=size < p.min_cluster_size,
                signature_groups=groups,
                url_patterns=[pat for pat, _ in top],
                url_examples=examples[:URL_EXAMPLES],
                page_ids=page_ids,
            )
        )
    sig_groups = [
        SignatureGroup(
            id=g,
            sig_hash=hex_of[g],
            size=int(g_size[g]),
            template_id=group_tid[g],
            page_ids=pages_of(g).tolist(),
        )
        for g in range(n_groups)
    ]
    for edge in checked + band_edges:
        ta, tb = group_tid[edge.a], group_tid[edge.b]
        if ta == tb:
            continue
        for x, y in ((ta, tb), (tb, ta)):
            if x not in stats.nearest or stats.nearest[x][1] < edge.jaccard:
                stats.nearest[x] = (y, edge.jaccard)
    out = Clusters(
        n_pages=n,
        min_cluster_size=p.min_cluster_size,
        params={
            "threshold": p.threshold,
            "band_low": p.band_low,
            "chrome_df": p.chrome_df,
            "num_perm": p.num_perm,
            "seed": p.seed,
            "chrome_fallback_share": CHROME_FALLBACK_SHARE,
            "n_signature_groups": n_groups,
            "n_chrome_paths": len(stats.chrome_paths),
            "n_fallback_groups": stats.n_fallback_groups,
            "lsh_bands": stats.lsh_bands,
            "lsh_rows": stats.lsh_rows,
            "n_band_merges": len(stats.band_merges),
        },
        clusters=clusters,
        signature_groups=sig_groups,
    )
    stats.seconds["output"] = time.perf_counter() - t4
    return ClusterResult(out, stats)


def _assign_id(hex16: str, used: set[str]) -> str:
    """``t_`` + the shortest prefix (>= 4 hex chars) of ``hex16`` not taken yet."""
    for length in range(ID_HEX, len(hex16) + 1):
        tid = ID_PREFIX + hex16[:length]
        if tid not in used:
            used.add(tid)
            return tid
    i = 2
    while f"{ID_PREFIX}{hex16}_{i}" in used:
        i += 1
    tid = f"{ID_PREFIX}{hex16}_{i}"
    used.add(tid)
    return tid
