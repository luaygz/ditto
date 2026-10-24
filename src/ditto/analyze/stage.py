"""Stage 4 over a run directory: :func:`analyze_run`, parallel over sampled pages."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ditto.analyze.core import (
    Collect,
    Plan,
    Survey,
    build_table,
    decide,
    survey_subset,
    walk_html,
)
from ditto.analyze.models import CandidateTable
from ditto.cluster.models import Clusters
from ditto.parallel import default_workers, pmap
from ditto.run import RunDir

DEFAULT_SAMPLE_SIZE = 2000


@dataclass
class _WorkerState:
    root: Path = field(default_factory=Path)
    plans: dict[str, Plan] = field(default_factory=dict)


_STATE = _WorkerState()


def _init_worker(root: str, plans: dict[str, Plan]) -> None:
    _STATE.root = Path(root)
    _STATE.plans = plans


def read_page(root: Path, page_id: int) -> str:
    """HTML of one stored page (per-process cached page store)."""
    from ditto.ingest import read_html

    return read_html(root, page_id)


def _survey_chunk(chunk: Sequence[tuple[str, int]]) -> dict[str, Survey]:
    out: dict[str, Survey] = {}
    for template_id, page_id in chunk:
        survey = out.get(template_id)
        if survey is None:
            survey = out[template_id] = Survey()
        survey.add(page_id, walk_html(read_page(_STATE.root, page_id), for_labels=False))
    return out


def _collect_chunk(chunk: Sequence[tuple[str, int]]) -> dict[str, Collect]:
    out: dict[str, Collect] = {}
    for template_id, page_id in chunk:
        collect = out.get(template_id)
        if collect is None:
            collect = out[template_id] = Collect()
        page = walk_html(read_page(_STATE.root, page_id))
        collect.add(page_id, page, _STATE.plans[template_id])
    return out


def analyze_run(
    run: RunDir,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    template_ids: Sequence[str] | None = None,
    include_tail: bool = False,
    workers: int | None = None,
) -> dict[str, CandidateTable]:
    """Analyze every template (largest first; tail clusters only with ``include_tail``) or the
    given ``template_ids`` on up to ``sample_size`` pages each, stratified by signature group,
    and write ``candidates/<template_id>.json``.

    All templates share one worker pool per pass: pass 1 surveys paths and structures on an
    evenly spaced subset of at most ``SURVEY_SAMPLE`` pages per template and accepts tables;
    pass 2 collects candidate statistics on every sampled page under those decisions.
    """
    started = time.perf_counter()
    clusters = Clusters.load(run.clusters_json)
    if template_ids is None:
        tids = [c.template_id for c in clusters.templates(include_tail=include_tail)]
    else:
        tids = list(dict.fromkeys(template_ids))
        for tid in tids:
            clusters.get(tid)  # KeyError for an unknown template id
    samples = {tid: clusters.sample(tid, sample_size) for tid in tids}
    items = [(tid, pid) for tid in tids for pid in samples[tid]]
    survey_items = [(tid, pid) for tid in tids for pid in survey_subset(samples[tid])]
    n_workers = default_workers(len(items)) if workers is None else max(1, workers)
    root = str(run.root)

    surveys = {tid: Survey() for tid in tids}
    for part in pmap(
        _survey_chunk,
        survey_items,
        workers=n_workers,
        chunk_size=_chunk_size(len(survey_items), n_workers),
        initializer=_init_worker,
        initargs=(root, {}),
    ):
        for tid, survey in part.items():
            surveys[tid].merge(survey)
    plans = {tid: decide(surveys[tid]) for tid in tids}

    collects = {tid: Collect() for tid in tids}
    for c_part in pmap(
        _collect_chunk,
        items,
        workers=n_workers,
        chunk_size=_chunk_size(len(items), n_workers),
        initializer=_init_worker,
        initargs=(root, plans),
    ):
        for tid, collect in c_part.items():
            collects[tid].merge(collect)

    out: dict[str, CandidateTable] = {}
    for tid in tids:
        collect = collects[tid]
        base_id = collect.base_page_id
        base = walk_html(read_page(run.root, base_id)) if base_id is not None else None
        table = build_table(tid, clusters.get(tid).size, surveys[tid], plans[tid], collect, base)
        table.save(run.candidates(tid))
        out[tid] = table
    run.record_timing("analyze", time.perf_counter() - started, len(items), n_workers)
    return out


def _chunk_size(n_items: int, workers: int) -> int:
    """About four chunks per worker for load balance, 4..256 pages each."""
    return max(4, min(256, math.ceil(n_items / (workers * 4)))) if n_items else 1
