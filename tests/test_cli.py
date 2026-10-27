"""Every ``ditto`` command through Typer's CliRunner on the 200-page fixture site, with a fake
LLM (no network): the happy paths and the clean one-line failures (exit 1) and usage errors
(exit 2)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import dotenv
import pytest
from pipeline_fakes import FIELD_NAMES, FIELDS, SITE, FakeLLMs, copy_run
from typer.testing import CliRunner, Result

import ditto.checkpoints
import ditto.cli
import ditto.llm
import ditto.llm.client
import ditto.verify
from ditto.cli import app, budget_message
from ditto.cluster.models import Clusters
from ditto.llm.base import BudgetExceeded
from ditto.run import RunDir, read_json
from ditto.verify.models import RepairRecord, VerifyReport

REPO = Path(__file__).resolve().parents[1]
BROKE = BudgetExceeded(
    "LLM budget exceeded: projected $0.0500, but only $0.0100 of $40.00 remains "
    "(spent $39.9900; ledger). Raise LLM_BUDGET_USD or reduce the work.",
    budget_usd=40.0,
    spent_usd=39.99,
    remaining_usd=0.01,
    projected_usd=0.05,
)


def invoke(*args: str | Path) -> Result:
    return CliRunner().invoke(app, [str(a) for a in args])


def ok(*args: str | Path) -> Result:
    result = invoke(*args)
    assert result.exit_code == 0, result.output
    return result


def one_line_error(result: Result, *needles: str) -> str:
    """Exit 1, no traceback, exactly one ``error:`` line on stderr containing ``needles``."""
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    lines = [ln for ln in result.stderr.splitlines() if ln.startswith("error:")]
    assert len(lines) == 1, result.stderr
    assert "Traceback" not in result.output
    for needle in needles:
        assert needle in lines[0], lines[0]
    return lines[0]


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """Rich output without colour codes or box/wrap whitespace (CI forces colour on)."""
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


def usage_error(result: Result, message: str) -> None:
    """Exit 2 with ``message`` in Typer's (possibly wrapped, boxed) error panel."""
    assert result.exit_code == 2, result.output
    text = plain(result.output)
    assert message in text, text


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolated ledger and budget; building a real provider adapter fails the test."""
    monkeypatch.setenv("DITTO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_BUDGET_USD", "40")
    monkeypatch.delenv("DITTO_RUN_BUDGET_USD", raising=False)

    def no_network(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a real LLM adapter was requested")

    monkeypatch.setattr(ditto.llm.client, "make_adapter", no_network)


@pytest.fixture
def llms(monkeypatch: pytest.MonkeyPatch) -> FakeLLMs:
    fakes = FakeLLMs()
    monkeypatch.setattr(ditto.llm, "get_llm", fakes)
    return fakes


@pytest.fixture
def broke_llms(monkeypatch: pytest.MonkeyPatch) -> FakeLLMs:
    fakes = FakeLLMs(fail=BROKE)
    monkeypatch.setattr(ditto.llm, "get_llm", fakes)
    return fakes


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """ingest -> cluster -> analyze through the CLI, once per module."""
    root = tmp_path_factory.mktemp("cli") / "run"
    out = ok("ingest", SITE, "--out", root, "--workers", "1")
    assert out.stdout == ""  # progress goes to stderr
    ok("cluster", root, "--workers", "1")
    ok("analyze", root, "--workers", "1")
    return root


@pytest.fixture
def run(analyzed: Path, tmp_path: Path) -> RunDir:
    return copy_run(analyzed, tmp_path / "run")


@pytest.fixture
def confirmed(run: RunDir) -> RunDir:
    ok("probe", run.root, "--schema", FIELDS)
    ok("confirm", run.root, "--yes")
    return run


# -- happy paths ---------------------------------------------------------------------------------
def test_help_lists_every_command() -> None:
    out = plain(ok("--help").stdout)
    for cmd in (
        "ingest",
        "cluster",
        "analyze",
        "probe",
        "confirm",
        "generate",
        "verify",
        "review",
        "extract",
        "run",
        "eval",
        "synth",
    ):
        assert cmd in out
    assert "--probe-llm" in plain(ok("run", "--help").stdout)


def test_ingest_cluster_analyze_outputs(analyzed: Path) -> None:
    rd = RunDir(analyzed)
    assert rd.read_meta()["n_pages"] == 200
    clusters = Clusters.load(rd.clusters_json)
    templates = clusters.templates()
    assert len(templates) == 4 and all(c.size >= 20 for c in templates)
    assert rd.cluster_report.exists()
    for c in templates:
        assert rd.candidates(c.template_id).exists() and rd.union_html(c.template_id).exists()


def test_main_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded: list[bool] = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: loaded.append(True))  # keep env
    monkeypatch.setattr(sys, "argv", ["ditto", "--help"])
    with pytest.raises(SystemExit) as exc:
        ditto.cli.main()
    assert exc.value.code == 0 and "ingest" in capsys.readouterr().out and loaded == [True]


def test_prompted_flow_stage_by_stage(confirmed: RunDir, llms: FakeLLMs) -> None:
    rd = confirmed
    assert rd.probe_json.exists() and rd.probe_yaml.exists() and rd.mapping_json.exists()
    out = ok("generate", rd.root, "--schema", FIELDS)
    assert "4 ok" in out.stdout
    assert len(llms.calls("generation")) == 4  # one call per template
    templates = Clusters.load(rd.clusters_json).templates()
    for c in templates:
        assert rd.draft_spec(c.template_id).exists()

    out = ok("verify", rd.root, "--max-repair-calls", "0", "--workers", "1")
    assert "verify:" in out.stdout
    assert not llms.calls("repair")  # --max-repair-calls 0 never builds a repair LLM
    specs = sorted(rd.specs_dir.glob("t_*.json"))
    assert specs

    ok("review", rd.root, "--yes")
    assert all(rd.golden(p.stem).exists() for p in specs)

    out = ok("extract", rd.root, "--workers", "1")
    assert str(rd.report_md) in out.stdout
    rows = [
        json.loads(line)
        for path in sorted((rd.root / "out").glob("*/*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert len(rows) >= sum(c.size for c in templates) * 0.9
    assert all(row["title"] for row in rows)  # every template's title spec works
    assert {"url", "page_id", "template_id", "_meta"} <= set(rows[0])
    assert rd.report_json.exists()
    assert llms.calls("tail_direct") == []


def test_generate_unprompted_writes_schema_proposal(run: RunDir, llms: FakeLLMs) -> None:
    out = ok("generate", run.root, "--mode", "unprompted")
    assert "4 ok" in out.stdout
    proposal = read_json(run.schema_proposal)
    assert proposal and run.generation_json.exists()


def test_probe_llm_resolves_ambiguous_fields(run: RunDir, llms: FakeLLMs) -> None:
    ok("probe", run.root, "--schema", FIELDS, "--probe-llm")
    assert len(llms.calls("probe")) >= 1  # the fixture has one ambiguous field
    assert read_json(run.probe_json)["llm"]["used"] is True


def test_extract_tail_direct(confirmed: RunDir, llms: FakeLLMs) -> None:
    ok("generate", confirmed.root, "--schema", FIELDS)
    ok("verify", confirmed.root, "--max-repair-calls", "0")
    ok("extract", confirmed.root, "--tail", "direct", "--schema", FIELDS)
    assert llms.calls("tail_direct")  # one small-model call per tail page


def test_run_end_to_end(tmp_path: Path, llms: FakeLLMs) -> None:
    out_dir = tmp_path / "fx"
    result = ok("run", SITE, "--out", out_dir, "--schema", FIELDS, "--yes", "--workers", "1")
    rd = RunDir(out_dir)
    assert str(rd.report_md) in result.stdout and rd.report_md.exists()
    assert len(llms.calls("generation")) == 4
    assert list((out_dir / "out").glob("*/*.jsonl"))
    assert set(FIELD_NAMES) <= set(read_json(rd.mapping_json)["schema_fields"])


def test_eval_run_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    out = ok(
        "eval", "run", "eval/configs/synth_default.yaml", "--dry-run", "--results-dir", tmp_path
    )
    results = Path(out.stdout.split("results: ", 1)[1].strip())
    assert results.is_relative_to(tmp_path) and (results / "plan.json").exists()
    assert read_json(results / "meta.json")["status"] == "dry_run"


def test_synth_generate_then_ingest_and_cluster(tmp_path: Path) -> None:
    params = tmp_path / "tiny.yaml"
    params.write_text(
        "templates: 2\npages_median: 25\npages_sigma: 0.1\npages_min: 22\npages_max: 28\n"
        "total_pages: 50\ntail_templates: 0\n"
    )
    out = tmp_path / "synth"
    ok("synth", "generate", "--params", params, "--seed", "101", "--out", out, "--workers", "1")
    assert (out / "site" / "urls.tsv").exists() and (out / "truth" / "templates.json").exists()
    n_pages = len(list((out / "site").glob("*.html")))
    assert n_pages >= 40
    ok("ingest", out / "site", "--out", tmp_path / "run", "--workers", "1")
    ok("cluster", tmp_path / "run", "--workers", "1")
    clusters = Clusters.load(RunDir(tmp_path / "run").clusters_json)
    assert clusters.n_pages == n_pages and len(clusters.templates()) == 2


def test_synth_without_faker_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "faker", None)  # import faker -> ImportError
    result = invoke("synth", "generate", "--params", "x.yaml", "--out", tmp_path)
    one_line_error(result, "needs Faker", "[synth]")


# -- usage errors (exit 2) -----------------------------------------------------------------------
def test_prompted_mode_needs_a_schema(run: RunDir, tmp_path: Path) -> None:
    for result in (
        invoke("generate", run.root),
        invoke("run", SITE, "--out", tmp_path / "x", "--yes"),
    ):
        usage_error(result, "needs --schema")
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("generate", "{run}", "--mode", "fancy"), "--mode must be"),
        (("generate", "{run}", "--schema", "{fields}", "--tail", "direct"), "--tail must be"),
        (("extract", "{run}", "--tail", "spec"), "--tail must be"),
        (("extract", "{run}", "--tail", "direct"), "--tail direct needs --schema"),
        (("run", "{site}", "--out", "{tmp}/r", "--mode", "fancy"), "--mode must be"),
        (("run", "{site}", "--out", "{tmp}/r", "--schema", "{fields}", "--tail", "x"), "--tail"),
    ],
)
def test_bad_mode_and_tail(
    run: RunDir, tmp_path: Path, args: tuple[str, ...], message: str
) -> None:
    subst = {"run": run.root, "fields": FIELDS, "site": SITE, "tmp": tmp_path}
    result = invoke(*(a.format(**subst) for a in args))
    usage_error(result, message)
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("cluster",),
        ("analyze",),
        ("probe", "--schema", str(FIELDS)),
        ("confirm", "--yes"),
        ("generate", "--mode", "unprompted"),
        ("verify",),
        ("review", "--yes"),
        ("extract",),
    ],
)
def test_non_run_directory(tmp_path: Path, args: tuple[str, ...]) -> None:
    usage_error(invoke(args[0], tmp_path, *args[1:]), "is not a ditto run directory")


def test_missing_source_is_a_usage_error(tmp_path: Path) -> None:
    usage_error(invoke("ingest", tmp_path / "nope", "--out", tmp_path / "run"), "does not exist")


# -- clean failures (exit 1) ---------------------------------------------------------------------
def test_checkpoints_need_a_terminal_or_yes(run: RunDir, tmp_path: Path) -> None:
    ok("probe", run.root, "--schema", FIELDS)
    one_line_error(invoke("confirm", run.root), "interactive terminal", "--yes")
    assert not run.mapping_json.exists()
    one_line_error(invoke("review", run.root), "interactive terminal", "--yes")
    out_dir = tmp_path / "fx"
    one_line_error(invoke("run", SITE, "--out", out_dir, "--schema", FIELDS), "--yes")
    assert not out_dir.exists()  # refused before any work


def test_confirm_without_probe_is_a_clean_error(run: RunDir) -> None:
    one_line_error(invoke("confirm", run.root, "--yes"), "run the probe first")


def test_bad_fields_yaml_is_a_clean_error(run: RunDir, tmp_path: Path) -> None:
    bad = tmp_path / "fields.yaml"
    bad.write_text("fields: 12\n")
    assert invoke("probe", run.root, "--schema", bad).exit_code == 1


def test_budget_refusal_before_generation(
    confirmed: RunDir, llms: FakeLLMs, monkeypatch: Any
) -> None:
    monkeypatch.setenv("LLM_BUDGET_USD", "0.000001")
    result = invoke("generate", confirmed.root, "--schema", FIELDS)
    one_line_error(result, "LLM budget exceeded", "projected $", "remains", "spent $")
    assert llms.calls() == []


def test_budget_exhausted_during_generation(confirmed: RunDir, broke_llms: FakeLLMs) -> None:
    result = invoke("generate", confirmed.root, "--schema", FIELDS, "--concurrency", "1")
    one_line_error(result, "generation stopped early", "projected $0.0500")
    gen = read_json(confirmed.generation_json)
    assert gen["templates"] and all(t["status"] == "unprocessed" for t in gen["templates"].values())


def test_budget_on_probe_llm(run: RunDir, broke_llms: FakeLLMs) -> None:
    result = invoke("probe", run.root, "--schema", FIELDS, "--probe-llm")
    one_line_error(result, "LLM budget exceeded", "remains")


def test_budget_on_verify(confirmed: RunDir, llms: FakeLLMs, monkeypatch: Any) -> None:
    ok("generate", confirmed.root, "--schema", FIELDS)

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise BROKE

    monkeypatch.setattr(ditto.verify, "verify_run", refuse)
    one_line_error(invoke("verify", confirmed.root), "LLM budget exceeded")


def test_verify_warns_when_repairs_stop_on_budget(
    confirmed: RunDir, llms: FakeLLMs, monkeypatch: Any
) -> None:
    record = RepairRecord(call=1, page_id=0, fields=["title"], error=f"budget exceeded: {BROKE}")
    report = VerifyReport(template_id="t_x", repairs=[record])
    monkeypatch.setattr(ditto.verify, "verify_run", lambda *a, **k: {"t_x": report})
    result = ok("verify", confirmed.root)
    assert "warning: repairs stopped, budget exceeded" in result.stderr


def test_budget_on_extract_tail_direct(confirmed: RunDir, llms: FakeLLMs, monkeypatch: Any) -> None:
    ok("generate", confirmed.root, "--schema", FIELDS)
    ok("verify", confirmed.root, "--max-repair-calls", "0")
    monkeypatch.setenv("LLM_BUDGET_USD", "0.000001")
    result = invoke("extract", confirmed.root, "--tail", "direct", "--schema", FIELDS)
    one_line_error(result, "LLM budget exceeded")
    assert llms.calls("tail_direct") == []


def test_budget_on_run(tmp_path: Path, llms: FakeLLMs, monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_BUDGET_USD", "0.000001")
    result = invoke("run", SITE, "--out", tmp_path / "fx", "--schema", FIELDS, "--yes")
    one_line_error(result, "LLM budget exceeded", "projected $")
    assert llms.calls() == []


def test_budget_on_review_repair(confirmed: RunDir, monkeypatch: Any) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise BROKE

    monkeypatch.setattr(ditto.checkpoints, "review_run", refuse)
    one_line_error(invoke("review", confirmed.root, "--yes"), "LLM budget exceeded")


def test_budget_message_appends_totals_to_bare_messages() -> None:
    bare = BudgetExceeded("out of money", budget_usd=1.0, spent_usd=0.9, projected_usd=0.2)
    assert budget_message(bare) == "out of money (projected $0.2000, budget $1.0000, spent $0.9000)"
    assert budget_message(BudgetExceeded("")) == "LLM budget exceeded"
    assert budget_message(BROKE).count("projected $") == 1
    assert "\n" not in budget_message(BudgetExceeded("two\nlines"))
