"""Site plan: template blueprints, page counts, the shell, and structural calibration.

Structural similarity is *measured*, not assumed: exemplar pages are rendered, cleaned with
:func:`ditto.fingerprint.dom.clean_html` and reduced to their set of structural element paths
(:func:`ditto.fingerprint.dom.iter_paths`), exactly what the clusterer signs. Paths of the base
shell (present on every page) are removed, like the clusterer's chrome filter.

* Regular templates are re-rolled until their content-path Jaccard to every earlier template is
  at most ``MAX_CROSS_JACCARD`` (best of ``MAX_ATTEMPTS``).
* A near-duplicate partner is derived from its template by block-level edits, then single-class
  edits, until the *worst case* Jaccard (max over nav variants, variant-specific chrome kept)
  reaches ``near_duplicate_jaccard``.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ditto.fingerprint.dom import clean_html, iter_paths
from ditto.synth.blueprint import (
    NameRegistry,
    TemplateDef,
    apply_op,
    build_template,
    class_is_hashed,
    near_duplicate_ops,
)
from ditto.synth.page import make_page_data
from ditto.synth.params import MarkupStyle, SynthParams
from ditto.synth.render import (
    Chrome,
    Writer,
    assemble,
    content_tree,
    esc_plain,
    head_text,
    shell_text,
    size_chrome,
)
from ditto.synth.values import Pools, build_pools, slugify
from ditto.synth.vocab import KIND_BY_NAME, KINDS

MAX_CROSS_JACCARD = 0.5
MAX_ATTEMPTS = 6
CALIBRATION_TOL = 0.01


@dataclass
class SitePlan:
    params: SynthParams
    seed: int
    site_name: str
    host: str
    chrome: Chrome
    templates: list[TemplateDef]
    page_template: list[int]  # template index of every page
    variant_weights: list[float]
    near_duplicates: list[dict[str, Any]] = field(default_factory=list)
    cross_jaccard: dict[str, Any] = field(default_factory=dict)
    hashed_class_share: float = 0.0
    content_elements_median: float = 0.0

    @property
    def n_pages(self) -> int:
        return len(self.page_template)


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def page_paths(html_text: str) -> set[str]:
    return {p for _, p in iter_paths(clean_html(html_text))}


def allocate(total: int, weights: Sequence[float], lo: int, hi: int) -> list[int]:
    """Integer sizes proportional to ``weights`` summing to ``total`` within ``[lo, hi]``."""
    n = len(weights)
    if n == 0:
        return []
    if not (n * lo <= total <= n * hi):
        raise ValueError(f"cannot split {total} pages into {n} templates of {lo}..{hi}")
    fixed: dict[int, float] = {}
    while True:
        free = [i for i in range(n) if i not in fixed]
        rem = total - sum(fixed.values())
        sw = sum(weights[i] for i in free) or 1.0
        alloc = {i: rem * weights[i] / sw for i in free}
        viol = {i: float(lo) for i in free if alloc[i] < lo}
        viol.update({i: float(hi) for i in free if alloc[i] > hi})
        if not viol:
            break
        fixed.update(viol)
    exact = {**fixed, **alloc}
    sizes = [math.floor(exact[i]) for i in range(n)]
    short = total - sum(sizes)
    order = sorted(range(n), key=lambda i: (-(exact[i] - sizes[i]), i))
    for i in order[:short]:
        sizes[i] += 1
    return sizes


def plan_sizes(params: SynthParams, seed: int) -> tuple[list[int], list[int]]:
    """Page counts of the regular templates and of the tail templates."""
    rng = random.Random(f"{seed}|sizes")
    mu = math.log(params.pages_median)
    raw = [rng.lognormvariate(mu, params.pages_sigma) for _ in range(params.templates)]
    tails = [rng.randint(*params.tail_pages) for _ in range(params.tail_templates)]
    if params.total_pages is None:
        main = [min(max(round(x), params.pages_min), params.pages_max) for x in raw]
        return main, tails
    main_total = params.total_pages - sum(tails)
    n = params.templates
    if main_total < n:
        raise ValueError(
            f"total_pages={params.total_pages} leaves {main_total} pages for {n} templates"
        )
    lo = min(params.pages_min, main_total // n)
    hi = max(params.pages_max, -(-main_total // n))
    return allocate(main_total, raw, lo, hi), tails


class _Measure:
    """Renders exemplar pages and returns their content-path sets."""

    def __init__(self, params: SynthParams, pools: Pools, chrome: Chrome, host: str) -> None:
        self.params = params
        self.pools = pools
        self.chrome = chrome
        self.host = host
        n_var = params.boilerplate_variants
        self.shells = [shell_text(chrome, v, "pretty") for v in range(n_var)]
        shell_paths = [page_paths(self.page_html("", v)) for v in range(n_var)]
        self.base = shell_paths[0]
        self.variant_extra = [p - self.base for p in shell_paths]

    def page_html(self, content: str, variant: int = 0) -> str:
        head = head_text("Exemplar", f"https://{self.host}/", "utf-8", "pretty", esc_plain)
        return assemble(head, self.shells[variant], content, "pretty")

    def exemplar(self, t: TemplateDef) -> str:
        rng = random.Random(f"exemplar|{t.tid}")
        d = make_page_data(t, self.params, self.pools, rng, host=self.host, exemplar=True)
        w = Writer("pretty")
        w.write(content_tree(t, d), self.shells[0].depth)
        return self.page_html(w.text())

    def content(self, t: TemplateDef) -> set[str]:
        return page_paths(self.exemplar(t)) - self.base

    def worst_case(self, a: set[str], b: set[str]) -> float:
        return max(jaccard(a | v, b | v) for v in self.variant_extra)

    def class_stats(self, t: TemplateDef) -> tuple[int, int]:
        root = clean_html(self.exemplar(t))
        total = hashed = 0
        for el in root.iter():
            value = el.get("class") if isinstance(el.tag, str) else None
            if value and value.split():
                total += 1
                hashed += class_is_hashed(value)
        return total, hashed

    def content_elements(self, t: TemplateDef, n: int = 3) -> float:
        counts = []
        for k in range(n):
            rng = random.Random(f"estimate|{t.tid}|{k}")
            d = make_page_data(t, self.params, self.pools, rng, host=self.host)
            w = Writer("minified")
            w.write(content_tree(t, d), 0)
            counts.append(w.count)
        return sum(counts) / len(counts)


def _calibrate(
    a: TemplateDef, b: TemplateDef, target: float, m: _Measure, rng: random.Random
) -> tuple[TemplateDef, float, float]:
    ca = m.content(a)
    cur = b
    j = m.worst_case(ca, m.content(cur))
    coarse, fine = near_duplicate_ops(a, rng)
    for op in coarse:
        if j <= target + CALIBRATION_TOL:
            break
        cand = apply_op(cur, op, rng)
        jc = m.worst_case(ca, m.content(cand))
        if jc < target - CALIBRATION_TOL:
            continue  # this block edit overshoots; finer edits follow
        cur, j = cand, jc
    for op in fine:
        if j <= target + CALIBRATION_TOL:
            break
        cand = apply_op(cur, op, rng)
        jc = m.worst_case(ca, m.content(cand))
        if target - CALIBRATION_TOL <= jc < j:
            cur, j = cand, jc
    return cur, j, jaccard(ca, m.content(cur))


def build_plan(params: SynthParams, seed: int) -> SitePlan:
    pools = build_pools(seed)
    rng = random.Random(f"{seed}|site")
    site_name = params.site_name or rng.choice(pools.companies)
    host = params.host or f"www.{slugify(site_name)}.com"
    site_style: MarkupStyle = "server" if params.markup_style == "mixed" else params.markup_style
    chrome = Chrome(site_style, site_name, seed, params.hashed_class_share)
    size_chrome(chrome, 40)
    measure = _Measure(params, pools, chrome, host)

    main_sizes, tail_sizes = plan_sizes(params, seed)
    sizes = main_sizes + tail_sizes
    n_main, n_all = len(main_sizes), len(sizes)
    width = max(2, len(str(n_all)))
    tids = [f"T{i + 1:0{width}d}" for i in range(n_all)]
    kinds = list(KINDS)
    rng.shuffle(kinds)
    pair_idx = rng.sample(range(n_main), 2 * params.near_duplicate_pairs)
    pairs = [(pair_idx[2 * k], pair_idx[2 * k + 1]) for k in range(params.near_duplicate_pairs)]
    derived = {b for _, b in pairs}

    names = NameRegistry()
    built: dict[int, TemplateDef] = {}
    content: dict[int, set[str]] = {}
    kind_cursor = 0
    for i in range(n_all):
        if i in derived:
            continue
        kind = kinds[kind_cursor % len(kinds)]
        kind_cursor += 1
        best: tuple[float, TemplateDef, set[str], NameRegistry] | None = None
        for attempt in range(MAX_ATTEMPTS):
            reg = copy.deepcopy(names)
            t = build_template(
                params,
                seed=seed,
                index=i,
                tid=tids[i],
                kind=kind,
                is_tail=i >= n_main,
                names=reg,
                attempt=attempt,
            )
            c = measure.content(t)
            worst = max((jaccard(c, other) for other in content.values()), default=0.0)
            if best is None or worst < best[0]:
                best = (worst, t, c, reg)
            if worst <= MAX_CROSS_JACCARD:
                break
        assert best is not None
        _, t, c, names = best
        t.pages = sizes[i]
        built[i], content[i] = t, c

    near: list[dict[str, Any]] = []
    for a_i, b_i in pairs:
        a = built[a_i]
        drng = random.Random(f"{seed}|neardup|{b_i}")
        base = a.clone()
        base.tid, base.index, base.pages = tids[b_i], b_i, sizes[b_i]
        base.near_duplicate_of = a.tid
        base.segment = names.segment(KIND_BY_NAME[a.kind], drng, skip_first=1)
        b, j, jc = _calibrate(a, base, params.near_duplicate_jaccard, measure, drng)
        built[b_i], content[b_i] = b, measure.content(b)
        near.append(
            {
                "a": a.tid,
                "b": b.tid,
                "target": params.near_duplicate_jaccard,
                "jaccard": round(j, 4),
                "content_jaccard": round(jc, 4),
                "segments": [a.segment, b.segment],
            }
        )
    templates = [built[i] for i in range(n_all)]

    pair_set = {frozenset((tids[a], tids[b])) for a, b in pairs}
    worst_pair: tuple[float, list[str]] = (0.0, [])
    for i in range(n_all):
        for k in range(i + 1, n_all):
            if frozenset((tids[i], tids[k])) in pair_set:
                continue
            jv = jaccard(content[i], content[k])
            if jv > worst_pair[0]:
                worst_pair = (jv, [tids[i], tids[k]])

    total_cls = hashed_cls = 0
    weights: list[float] = []
    elems: list[float] = []
    for t in templates:
        n_cls, n_hashed = measure.class_stats(t)
        total_cls += n_cls * t.pages
        hashed_cls += n_hashed * t.pages
        weights.append(t.pages)
        elems.append(measure.content_elements(t))
    median_elems = _weighted_median(elems, weights)
    share = params.boilerplate_share
    size_chrome(chrome, round(share / (1 - share) * median_elems))

    page_template = [i for i, t in enumerate(templates) for _ in range(t.pages)]
    random.Random(f"{seed}|assign").shuffle(page_template)
    raw_w = [1 / (k + 1) for k in range(params.boilerplate_variants)]
    return SitePlan(
        params=params,
        seed=seed,
        site_name=site_name,
        host=host,
        chrome=chrome,
        templates=templates,
        page_template=page_template,
        variant_weights=[w / sum(raw_w) for w in raw_w],
        near_duplicates=near,
        cross_jaccard={"max": round(worst_pair[0], 4), "pair": worst_pair[1]},
        hashed_class_share=hashed_cls / total_cls if total_cls else 0.0,
        content_elements_median=median_elems,
    )


def _weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights, strict=True))
    half = sum(weights) / 2
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return v
    return pairs[-1][0] if pairs else 0.0
