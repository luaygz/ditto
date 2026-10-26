"""``ditto`` command-line interface (Typer).

Expected failures (budget, missing terminal at a checkpoint, invalid inputs, provider errors)
print one ``error:`` line on stderr and exit with status 1; usage errors exit with status 2.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from ditto.run import RunDir

if TYPE_CHECKING:
    from ditto.llm.base import BudgetExceeded

app = typer.Typer(
    name="ditto",
    help="Extract a whole downloaded website into JSON with ~one LLM call per page template.",
    no_args_is_help=True,
    add_completion=False,
)
eval_app = typer.Typer(help="Benchmark evaluation.", no_args_is_help=True)
synth_app = typer.Typer(help="Synthetic sites with ground truth.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
app.add_typer(synth_app, name="synth")

RunArg = Annotated[Path, typer.Argument(help="Run directory created by `ditto ingest`.")]
SrcArg = Annotated[
    list[Path], typer.Argument(help="HTML directory, or WARC/WARC.gz files.", exists=True)
]
Workers = Annotated[int | None, typer.Option(help="Worker processes (default: all cores).")]
MinSize = Annotated[
    int, typer.Option("--min-cluster-size", help="Templates below this size are the long tail.")
]
NoCache = Annotated[bool, typer.Option("--no-cache", help="Bypass the LLM response cache.")]
ProbeLLM = Annotated[
    bool, typer.Option("--probe-llm", help="Resolve ambiguous fields with the small model.")
]

MODES = ("prompted", "unprompted")


def _run_dir(path: Path) -> RunDir:
    run = RunDir(path)
    if not run.pages_db.exists():
        raise typer.BadParameter(f"{path} is not a ditto run directory (no pages.sqlite)")
    return run


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise typer.BadParameter("--mode must be prompted or unprompted")


def _fail(message: object) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def budget_message(exc: BudgetExceeded) -> str:
    """One line with the budget totals (the LLM layer's message already states them; a bare
    ``BudgetExceeded("...")`` with keyword totals gets them appended)."""
    text = " ".join(str(exc).split()) or "LLM budget exceeded"
    totals = [
        (name, value)
        for name, value in (
            ("projected", exc.projected_usd),
            ("remaining", exc.remaining_usd),
            ("budget", exc.budget_usd),
            ("spent", exc.spent_usd),
        )
        if value is not None
    ]
    if totals and "projected $" not in text:
        text += " (" + ", ".join(f"{name} ${value:.4f}" for name, value in totals) + ")"
    return text


@contextmanager
def _guard() -> Iterator[None]:
    """Turn expected failures into one ``error:`` line and exit status 1 (no traceback)."""
    from ditto.llm.base import BudgetExceeded, LLMError
    from ditto.pipeline import CheckpointRequired

    try:
        yield
    except BudgetExceeded as exc:
        _fail(budget_message(exc))
    except (CheckpointRequired, LLMError, ValueError, FileNotFoundError) as exc:
        _fail(exc)


@app.command()
def ingest(
    src: SrcArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Run directory to create.")],
    manifest: Annotated[
        Path | None, typer.Option(help="urls.tsv manifest (path<TAB>url) for a directory.")
    ] = None,
    site: Annotated[str | None, typer.Option(help="Site name (default: inferred).")] = None,
    workers: Workers = None,
) -> None:
    """Read an HTML directory (optional urls.tsv) or WARC files into a run directory."""
    from ditto.pipeline import stage_ingest

    source: Path | list[Path] = src[0] if len(src) == 1 else src
    with _guard():
        stage_ingest(source, out, manifest=manifest, site=site, workers=workers)


@app.command()
def cluster(run: RunArg, min_cluster_size: MinSize = 20, workers: Workers = None) -> None:
    """Fingerprint and cluster pages into templates; writes clusters.json and a report."""
    from ditto.pipeline import stage_cluster

    rd = _run_dir(run)
    with _guard():
        stage_cluster(rd, min_cluster_size=min_cluster_size, workers=workers)


@app.command()
def analyze(
    run: RunArg,
    include_tail: Annotated[bool, typer.Option(help="Also analyze tail clusters.")] = False,
    max_tokens: Annotated[int, typer.Option(help="Union page token cap.")] = 30000,
    workers: Workers = None,
) -> None:
    """Build candidate tables and one union page per template."""
    from ditto.pipeline import stage_analyze

    rd = _run_dir(run)
    with _guard():
        stage_analyze(rd, include_tail=include_tail, workers=workers, max_tokens=max_tokens)


@app.command()
def probe(
    run: RunArg,
    schema: Annotated[Path, typer.Option(help="fields.yaml with the requested fields.")],
    probe_llm: ProbeLLM = False,
    no_cache: NoCache = False,
) -> None:
    """Map requested fields to candidates per template (no LLM by default)."""
    from ditto.checkpoints import probe_run
    from ditto.schema import FieldsSchema

    rd = _run_dir(run)
    with _guard():
        fields = FieldsSchema.load(schema)
        llm = None
        if probe_llm:
            from ditto.llm import get_llm

            llm = get_llm("probe", run=rd, no_cache=no_cache)
        probe_run(rd, fields, use_llm=probe_llm, llm=llm)
    typer.echo(f"wrote {rd.probe_json} and {rd.probe_yaml}")


@app.command()
def confirm(
    run: RunArg,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept the proposal.")] = False,
) -> None:
    """Edit or accept the probe mapping; writes mapping.json."""
    from ditto.checkpoints import confirm_run
    from ditto.pipeline import require_terminal

    rd = _run_dir(run)
    with _guard():
        require_terminal(yes, "ditto confirm")
        confirm_run(rd, yes=yes, interactive=not yes)
    typer.echo(f"wrote {rd.mapping_json}")


@app.command()
def generate(
    run: RunArg,
    mode: Annotated[str, typer.Option(help="prompted or unprompted.")] = "prompted",
    schema: Annotated[
        Path | None, typer.Option(help="fields.yaml (required in prompted mode).")
    ] = None,
    min_cluster_size: MinSize = 20,
    tail: Annotated[str, typer.Option(help="skip | spec (direct runs at extraction).")] = "skip",
    concurrency: Annotated[int, typer.Option(help="Concurrent LLM calls.")] = 4,
    no_cache: NoCache = False,
) -> None:
    """The LLM step: one spec-generation call per template; writes draft specs."""
    from ditto.llm import get_llm
    from ditto.schema import FieldsSchema, Mapping
    from ditto.specgen import generate_run

    _check_mode(mode)
    if tail not in ("skip", "spec"):
        raise typer.BadParameter("--tail must be skip or spec (use `ditto extract --tail direct`)")
    if mode == "prompted" and schema is None:
        raise typer.BadParameter("prompted mode needs --schema fields.yaml")
    rd = _run_dir(run)
    with _guard():
        fields = FieldsSchema.load(schema) if schema else None
        mapping = (
            Mapping.load(rd.mapping_json)
            if mode == "prompted" and rd.mapping_json.exists()
            else None
        )
        result = generate_run(
            rd,
            mode=mode,  # type: ignore[arg-type]
            schema=fields,
            mapping=mapping,
            llm=get_llm("generation", run=rd, no_cache=no_cache),
            min_cluster_size=min_cluster_size,
            tail=tail,  # type: ignore[arg-type]
            concurrency=concurrency,
        )
    counts = [
        f"{len(tids)} {status}"
        for status in ("ok", "partial", "failed", "skipped", "unprocessed")
        if (tids := result.by_status(status))
    ]
    typer.echo(f"generate: {', '.join(counts) or 'no templates'}; wrote {rd.generation_json}")
    if result.aborted:
        _fail(f"generation stopped early: {' '.join(result.aborted.split())}")


@app.command()
def verify(
    run: RunArg,
    max_repair_calls: Annotated[int, typer.Option(help="Repair calls per template.")] = 3,
    workers: Workers = None,
    no_cache: NoCache = False,
) -> None:
    """Score every candidate on every page, repair within budget, finalize specs."""
    from ditto.llm import get_llm
    from ditto.verify import verify_run

    rd = _run_dir(run)
    with _guard():
        llm = get_llm("repair", run=rd, no_cache=no_cache) if max_repair_calls > 0 else None
        reports = verify_run(rd, max_repair_calls=max_repair_calls, llm=llm, workers=workers)
    stopped = next(
        (
            r.error
            for rep in reports.values()
            for r in rep.repairs
            if r.error and r.error.startswith("budget exceeded")
        ),
        None,
    )
    if stopped:
        typer.secho(f"warning: repairs stopped, {' '.join(stopped.split())}", fg="yellow", err=True)
    typer.echo(f"verify: {len(reports)} templates; wrote {rd.specs_dir}")


@app.command()
def review(
    run: RunArg, yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip sign-off.")] = False
) -> None:
    """Show five extracted records per template for sign-off; saves golden samples."""
    from ditto.checkpoints import review_run
    from ditto.pipeline import require_terminal

    rd = _run_dir(run)
    with _guard():
        require_terminal(yes, "ditto review")
        review_run(rd, yes=yes)


@app.command()
def extract(
    run: RunArg,
    workers: Workers = None,
    tail: Annotated[
        str, typer.Option(help="skip | direct (small-model call per tail page).")
    ] = "skip",
    schema: Annotated[Path | None, typer.Option(help="fields.yaml for --tail direct.")] = None,
    no_cache: NoCache = False,
) -> None:
    """Run final specs over all pages (no LLM); writes JSONL and the report."""
    from ditto.extract import extract_run, write_report

    if tail not in ("skip", "direct"):
        raise typer.BadParameter("--tail must be skip or direct")
    if tail == "direct" and schema is None:
        raise typer.BadParameter("--tail direct needs --schema fields.yaml")
    rd = _run_dir(run)
    with _guard():
        extract_run(rd, workers=workers)
        if tail == "direct":
            from ditto.llm import get_llm
            from ditto.schema import FieldsSchema
            from ditto.specgen.tail import tail_direct_run

            assert schema is not None
            tail_direct_run(
                rd,
                FieldsSchema.load(schema),
                llm=get_llm("tail_direct", run=rd, no_cache=no_cache),
            )
        write_report(rd)
    typer.echo(f"wrote {rd.report_md}")


@app.command("run")
def run_cmd(
    src: SrcArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Run directory to create.")],
    mode: Annotated[str, typer.Option(help="prompted or unprompted.")] = "prompted",
    schema: Annotated[Path | None, typer.Option(help="fields.yaml (prompted mode).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept both checkpoints.")] = False,
    tail: Annotated[str, typer.Option(help="skip | direct | spec.")] = "skip",
    min_cluster_size: MinSize = 20,
    probe_llm: ProbeLLM = False,
    max_repair_calls: Annotated[int, typer.Option(help="Repair calls per template.")] = 3,
    workers: Workers = None,
    no_cache: NoCache = False,
) -> None:
    """All stages in order, pausing at the two checkpoints unless --yes."""
    from ditto.pipeline import RunOptions, run_all

    _check_mode(mode)
    if tail not in ("skip", "direct", "spec"):
        raise typer.BadParameter("--tail must be skip, direct or spec")
    if mode == "prompted" and schema is None:
        raise typer.BadParameter("prompted mode needs --schema fields.yaml")
    opts = RunOptions(
        mode=mode,  # type: ignore[arg-type]
        schema=schema,
        yes=yes,
        tail=tail,  # type: ignore[arg-type]
        min_cluster_size=min_cluster_size,
        probe_llm=probe_llm,
        max_repair_calls=max_repair_calls,
        no_cache=no_cache,
        workers=workers,
    )
    with _guard():
        rd = run_all(src[0] if len(src) == 1 else src, out, opts)
    typer.echo(f"wrote {rd.report_md}")


@eval_app.command("run")
def eval_run(
    config: Annotated[Path, typer.Argument(help="eval/configs/<name>.yaml")],
    results_dir: Annotated[Path, typer.Option(help="Where results are written.")] = Path(
        "eval/results"
    ),
    dry_run: Annotated[
        bool, typer.Option(help="Validate the config and list the planned sites.")
    ] = False,
) -> None:
    """Regenerate benchmark numbers into eval/results/."""
    from ditto.evaluation.runner import run_eval

    with _guard():
        out = run_eval(config, results_dir=results_dir, dry_run=dry_run)
    typer.echo(f"results: {out}")


@synth_app.command("generate")
def synth_generate(
    params: Annotated[Path, typer.Option(help="Parameter YAML, e.g. eval/synth/default.yaml.")],
    seed: Annotated[int, typer.Option(help="Random seed.")] = 1,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("synth"),
    workers: Workers = None,
) -> None:
    """Build a synthetic site with per-page ground truth."""
    try:
        import faker  # noqa: F401
    except ImportError:
        _fail("`ditto synth` needs Faker: pip install 'ditto-extract[synth]'")
    from ditto.synth import synth_main

    with _guard():
        synth_main(params, seed, out, workers=workers)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
