"""Stage 5 over a run directory: ``union/<tid>.html`` and ``union/<tid>.json`` per template."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

from ditto.analyze.models import CandidateTable
from ditto.analyze.stage import DEFAULT_SAMPLE_SIZE as ANALYSIS_SAMPLE_SIZE
from ditto.cluster.models import Clusters
from ditto.ingest import PageStore, read_html
from ditto.parallel import default_workers, pmap
from ditto.run import RunDir, write_json
from ditto.union.page import UnionPage, build_union_page

DEFAULT_SAMPLE_SIZE = 100  # sampled pages per template (the brief's range is 50-200)
DEFAULT_SEED = 0


@dataclass(frozen=True)
class _Job:
    run_root: str
    template_id: str
    page_ids: tuple[int, ...]
    focus: tuple[str, ...] | None
    min_tokens: int
    max_tokens: int
    donor_pool: tuple[int, ...] = ()


def _build(chunk: Sequence[_Job]) -> list[UnionPage]:
    out: list[UnionPage] = []
    for job in chunk:
        run = RunDir(Path(job.run_root))
        table = CandidateTable.load(run.candidates(job.template_id))
        pages = {pid: read_html(job.run_root, pid) for pid in job.page_ids}
        more = ((pid, read_html(job.run_root, pid)) for pid in job.donor_pool)
        out.append(
            build_union_page(
                table,
                pages,
                min_tokens=job.min_tokens,
                max_tokens=job.max_tokens,
                focus=set(job.focus) if job.focus is not None else None,
                more_pages=more,
            )
        )
    return out


def build_union_run(
    run: RunDir,
    *,
    min_tokens: int = 8000,
    max_tokens: int = 30000,
    template_ids: Iterable[str] | None = None,
    focus: Mapping[str, Set[str]] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    workers: int | None = None,
) -> dict[str, UnionPage]:
    """Build the union page of every non-tail template (or of ``template_ids``) from
    ``candidates/<tid>.json`` and ``sample_size`` pages drawn with ``Clusters.sample``; a field
    candidate none of them has is looked for in the rest of the analysis sample (donors).

    ``focus`` maps template id -> candidate ids (from ``mapping.json``); a template without an
    entry gets the full union page. Writes ``union/<tid>.html`` and ``union/<tid>.json``."""
    start = time.perf_counter()
    clusters = Clusters.load(run.clusters_json)
    if template_ids is None:
        tids = [c.template_id for c in clusters.templates()]
    else:
        tids = list(template_ids)
    jobs: list[_Job] = []
    for tid in tids:
        if not run.candidates(tid).exists():
            raise FileNotFoundError(f"{run.candidates(tid)} is missing: run the analysis first")
        wanted = focus.get(tid) if focus is not None else None
        sample = clusters.sample(tid, sample_size, seed)
        chosen = set(sample)
        jobs.append(
            _Job(
                run_root=str(run.root),
                template_id=tid,
                page_ids=tuple(sample),
                focus=tuple(sorted(wanted)) if wanted is not None else None,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                donor_pool=tuple(
                    p for p in clusters.sample(tid, ANALYSIS_SAMPLE_SIZE) if p not in chosen
                ),
            )
        )
    n_workers = default_workers(len(jobs)) if workers is None else max(1, workers)
    store = PageStore.open(run) if jobs else None
    results: dict[str, UnionPage] = {}
    for chunk in pmap(_build, jobs, workers=n_workers, chunk_size=1):
        for page in chunk:
            if store is not None:
                page.base_url = store.url(page.base_page_id)
                page.donor_urls = [store.url(pid) for pid in page.donor_page_ids]
            html_path = run.union_html(page.template_id)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(page.html, encoding="utf-8")
            write_json(run.union_meta(page.template_id), page.meta())
            results[page.template_id] = page
    run.record_timing(
        "union",
        time.perf_counter() - start,
        pages=sum(len(j.page_ids) for j in jobs),
        workers=n_workers,
    )
    return results
