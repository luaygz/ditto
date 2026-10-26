"""Stage 8: run the final specs over every page of their clusters (no LLM) into JSONL.

One row per page: ``{url, page_id, template_id, <fields...>, _meta: {missing, spec_version}}``
in ``out/<site>/<template_id>.jsonl``. Pages are parsed, cleaned and extracted in worker
processes; the parent only writes the serialized rows, in page order.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import orjson

from ditto.cluster.models import Clusters
from ditto.extract.runner import apply_spec, is_missing
from ditto.fingerprint.dom import clean_html
from ditto.parallel import chunked, default_workers, pmap
from ditto.run import RunDir
from ditto.specgen.models import Spec

PAGES_PER_CHUNK = 256


@dataclass
class TemplateExtract:
    template_id: str
    pages: int = 0
    path: str = ""
    filled: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractSummary:
    site: str
    pages: int = 0
    seconds: float = 0.0
    workers: int = 1
    cpu_seconds: float = 0.0
    pages_per_sec: float | None = None
    pages_per_sec_per_core: float | None = None
    pages_per_cpu_sec: float | None = None
    templates: dict[str, TemplateExtract] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def row_for(spec: Spec, page_id: int, url: str, values: dict[str, Any]) -> dict[str, Any]:
    """Output row: page identity first, then the fields, then ``_meta`` (the spec model
    rejects field names that collide with these keys)."""
    return {"url": url, "page_id": page_id, "template_id": spec.template_id, **values}


_STATE: dict[str, Any] = {}


def _init(run_root: str, specs: tuple[Spec, ...]) -> None:
    _STATE["root"] = Path(run_root)
    _STATE["specs"] = specs


@dataclass
class _Chunk:
    data: dict[int, bytes]
    filled: dict[int, list[int]]
    pages: dict[int, int]
    cpu: float


def _extract_chunk(items: Sequence[tuple[int, int, str]]) -> _Chunk:
    from ditto.ingest import read_html

    start = time.process_time()
    root: Path = _STATE["root"]
    specs: tuple[Spec, ...] = _STATE["specs"]
    parts: dict[int, list[bytes]] = {}
    filled: dict[int, list[int]] = {}
    pages: dict[int, int] = {}
    for pid, si, url in items:
        spec = specs[si]
        values = apply_spec(spec, clean_html(read_html(root, pid)))
        counts = filled.setdefault(si, [0] * len(spec.fields))
        for i, name in enumerate(spec.fields):
            counts[i] += not is_missing(values[name])
        parts.setdefault(si, []).append(orjson.dumps(row_for(spec, pid, url, values)))
        pages[si] = pages.get(si, 0) + 1
    data = {si: b"\n".join(rows) + b"\n" for si, rows in parts.items()}
    return _Chunk(data, filled, pages, time.process_time() - start)


def _final_specs(
    run: RunDir, clusters: Clusters
) -> tuple[list[tuple[Spec, list[int]]], dict[str, str]]:
    jobs: list[tuple[Spec, list[int]]] = []
    skipped: dict[str, str] = {}
    for cluster in clusters.templates(include_tail=True):
        path = run.spec(cluster.template_id)
        if not path.exists():
            continue
        spec = Spec.load(path)
        if spec.status != "final":
            skipped[cluster.template_id] = "spec is not final (run `ditto verify`)"
            continue
        jobs.append((spec, sorted(cluster.page_ids)))
    return jobs, skipped


class _Writer:
    """Writes each template's rows to ``<tid>.jsonl.tmp`` and renames it when done. Rows
    arrive grouped by template in order, so at most one file is open at a time."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)
        self.current: tuple[int, IO[bytes]] | None = None
        self.done: set[int] = set()

    def _tmp(self, si: int) -> Path:
        return self.paths[si].with_suffix(".jsonl.tmp")

    def _close(self) -> None:
        if self.current is not None:
            si, fh = self.current
            fh.close()
            self._tmp(si).replace(self.paths[si])
            self.done.add(si)
            self.current = None

    def write(self, si: int, data: bytes) -> None:
        if self.current is None or self.current[0] != si:
            self._close()
            self.current = (si, self._tmp(si).open("wb"))
        self.current[1].write(data)

    def finish(self) -> None:
        self._close()
        for si, path in enumerate(self.paths):
            if si not in self.done:  # a cluster with no pages still gets its (empty) file
                path.write_bytes(b"")
                self.done.add(si)

    def abort(self) -> None:
        if self.current is not None:
            si, fh = self.current
            fh.close()
            self._tmp(si).unlink(missing_ok=True)
            self.current = None


def extract_run(run: RunDir, *, workers: int | None = None) -> ExtractSummary:
    """Apply every final spec to every page of its cluster and write the JSONL files.

    Only ``out/<site>/<tid>.jsonl`` of extracted templates are (re)written; other files in
    ``out/<site>/`` (e.g. the long-tail ``_tail.jsonl``) are left alone. Records timing as
    stage ``extract`` and stores the summary under ``extract`` in ``run.json``."""
    from ditto.ingest import PageStore

    t0 = time.perf_counter()
    site = run.site
    clusters = Clusters.load(run.clusters_json)
    jobs, skipped = _final_specs(run, clusters)
    store = PageStore.open(run)
    specs = tuple(spec for spec, _ in jobs)
    items = [(pid, si, store.url(pid)) for si, (_, pids) in enumerate(jobs) for pid in pids]
    n_chunks = len(chunked(items, PAGES_PER_CHUNK))
    used = default_workers(n_chunks) if workers is None else max(1, min(workers, n_chunks))
    summary = ExtractSummary(site=site, workers=used, skipped=skipped)
    for spec in specs:
        path = run.out_jsonl(site, spec.template_id)
        summary.templates[spec.template_id] = TemplateExtract(
            spec.template_id,
            path=str(path.relative_to(run.root)),
            filled=dict.fromkeys(spec.fields, 0),
        )
    run.out_dir(site).mkdir(parents=True, exist_ok=True)
    writer = _Writer([run.out_jsonl(site, spec.template_id) for spec in specs])
    try:
        for chunk in pmap(
            _extract_chunk,
            items,
            workers=used,
            chunk_size=PAGES_PER_CHUNK,
            initializer=_init,
            initargs=(str(run.root), specs),
        ):
            summary.cpu_seconds += chunk.cpu
            for si in sorted(chunk.data):
                writer.write(si, chunk.data[si])
            for si, counts in chunk.filled.items():
                te = summary.templates[specs[si].template_id]
                te.pages += chunk.pages[si]
                for name, c in zip(specs[si].fields, counts, strict=True):
                    te.filled[name] += c
        writer.finish()
    finally:
        writer.abort()
    seconds = time.perf_counter() - t0
    summary.pages = len(items)
    summary.seconds = round(seconds, 4)
    summary.cpu_seconds = round(summary.cpu_seconds, 4)
    if seconds > 0 and items:
        summary.pages_per_sec = round(len(items) / seconds, 2)
        summary.pages_per_sec_per_core = round(len(items) / seconds / used, 2)
    if summary.cpu_seconds > 0:
        summary.pages_per_cpu_sec = round(len(items) / summary.cpu_seconds, 2)
    run.record_timing("extract", seconds, len(items), used)
    run.write_meta(extract=summary.to_dict())
    return summary
