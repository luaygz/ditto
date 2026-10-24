"""Stage orchestration shared by the CLI, the Python API and the evaluation harness.

Each function runs one user-visible stage on a run directory. ``run_all`` chains them with the
two checkpoints (mapping confirmation or schema proposal, then review).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ditto.run import RunDir

Mode = Literal["prompted", "unprompted"]
Tail = Literal["skip", "direct", "spec"]


class CheckpointRequired(RuntimeError):
    """A checkpoint needs a terminal (or ``--yes``) and none is available."""


def _log(msg: str) -> None:
    print(f"[ditto] {msg}", file=sys.stderr, flush=True)


def stage_ingest(
    src: Path | Sequence[Path],
    out: Path,
    *,
    manifest: Path | None = None,
    site: str | None = None,
    workers: int | None = None,
) -> RunDir:
    from ditto.ingest import ingest

    result = ingest(src, out, manifest=manifest, site=site, workers=workers)
    _log(f"ingest: {result}")
    return RunDir(out)


def stage_cluster(run: RunDir, *, min_cluster_size: int = 20, workers: int | None = None) -> Any:
    from ditto.cluster import cluster_run
    from ditto.fingerprint import fingerprint_run

    fingerprint_run(run, workers=workers)
    clusters = cluster_run(run, min_cluster_size=min_cluster_size, workers=workers)
    templates = clusters.templates()
    _log(
        f"cluster: {clusters.n_pages} pages -> {len(clusters.clusters)} clusters, "
        f"{len(templates)} templates with >= {min_cluster_size} pages"
    )
    return clusters


def stage_analyze(
    run: RunDir,
    *,
    include_tail: bool = False,
    workers: int | None = None,
    min_tokens: int = 8000,
    max_tokens: int = 30000,
) -> dict[str, Any]:
    from ditto.analyze import analyze_run
    from ditto.union import build_union_run

    tables = analyze_run(run, include_tail=include_tail, workers=workers)
    template_ids = list(tables)
    unions = build_union_run(
        run, min_tokens=min_tokens, max_tokens=max_tokens, template_ids=template_ids
    )
    _log(f"analyze: {len(tables)} candidate tables, {len(unions)} union pages")
    return tables


@dataclass
class RunOptions:
    mode: Mode = "prompted"
    schema: Path | None = None
    yes: bool = False
    tail: Tail = "skip"
    min_cluster_size: int = 20
    probe_llm: bool = False
    max_repair_calls: int = 3
    no_cache: bool = False
    workers: int | None = None
    concurrency: int = 4
    input_fn: Callable[[str], str] = input
    settings: Any = None  # ditto.llm Settings (pinned per-role models); None = from env
    timings: dict[str, float] = field(default_factory=dict)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _require_terminal(opts: RunOptions, what: str) -> None:
    if not opts.yes and not _interactive():
        raise CheckpointRequired(
            f"{what} needs an interactive terminal; rerun with --yes to accept proposals"
        )


def run_all(src: Path | Sequence[Path], out: Path, opts: RunOptions) -> RunDir:
    """All stages in order, pausing at the two checkpoints unless ``opts.yes``."""
    from ditto.schema import FieldsSchema

    if opts.mode == "prompted" and opts.schema is None:
        raise ValueError("prompted mode needs --schema fields.yaml")
    if opts.tail == "direct" and opts.mode != "prompted":
        raise ValueError("--tail direct extracts the prompted schema; use --mode prompted")
    schema = FieldsSchema.load(opts.schema) if opts.schema else None

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        result = fn()
        opts.timings[name] = time.perf_counter() - t0
        return result

    run = timed("ingest", lambda: stage_ingest(src, out, workers=opts.workers))
    timed(
        "cluster",
        lambda: stage_cluster(run, min_cluster_size=opts.min_cluster_size, workers=opts.workers),
    )
    timed(
        "analyze",
        lambda: stage_analyze(run, include_tail=opts.tail == "spec", workers=opts.workers),
    )
    return continue_run(run, opts, schema)


def continue_run(run: RunDir, opts: RunOptions, schema: Any) -> RunDir:
    """Stages after analysis: checkpoint 1, generation, verification, review, extraction."""
    from ditto.checkpoints import confirm_run, probe_run, review_run, show_schema_proposal
    from ditto.extract import extract_run, write_report
    from ditto.llm import get_llm
    from ditto.specgen import generate_run
    from ditto.verify import verify_run

    mapping = None
    if opts.mode == "prompted":
        probe_llm = (
            get_llm("probe", settings=opts.settings, run=run, no_cache=opts.no_cache)
            if opts.probe_llm
            else None
        )
        probe_run(run, schema, use_llm=opts.probe_llm, llm=probe_llm)
        _require_terminal(opts, "the mapping checkpoint (ditto confirm)")
        mapping = confirm_run(run, yes=opts.yes, interactive=not opts.yes)
    gen_llm = get_llm("generation", settings=opts.settings, run=run, no_cache=opts.no_cache)
    generate_run(
        run,
        mode=opts.mode,
        schema=schema,
        mapping=mapping,
        llm=gen_llm,
        min_cluster_size=opts.min_cluster_size,
        tail="spec" if opts.tail == "spec" else "skip",
        concurrency=opts.concurrency,
    )
    if opts.mode == "unprompted":
        _require_terminal(opts, "the schema checkpoint")
        show_schema_proposal(run, yes=opts.yes, input_fn=opts.input_fn)
    repair_llm = get_llm("repair", settings=opts.settings, run=run, no_cache=opts.no_cache)
    verify_run(run, max_repair_calls=opts.max_repair_calls, llm=repair_llm, workers=opts.workers)
    _require_terminal(opts, "the review checkpoint (ditto review)")
    review_run(run, yes=opts.yes, input_fn=opts.input_fn)
    extract_run(run, workers=opts.workers)
    if opts.tail == "direct" and schema is not None:
        from ditto.specgen.tail import tail_direct_run

        tail_llm = get_llm("tail_direct", settings=opts.settings, run=run, no_cache=opts.no_cache)
        tail_direct_run(run, schema, llm=tail_llm, concurrency=opts.concurrency)
    write_report(run)
    _log(f"done: report at {run.report_md}")
    return run
