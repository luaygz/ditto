"""``ditto.pipeline``: ``run_all`` on the fixture site with fake LLMs, both checkpoints answered
through ``input_fn`` as in a terminal, the unprompted path, ``--tail direct``, a budget stop
during generation, and the refusals that happen before any work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pipeline_fakes import FIELDS, SITE, FakeLLMs

import ditto.llm
import ditto.llm.client
import ditto.pipeline
from ditto.llm.base import BudgetExceeded
from ditto.pipeline import CheckpointRequired, RunOptions, require_terminal, run_all
from ditto.run import RunDir, read_json


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DITTO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_BUDGET_USD", "40")
    monkeypatch.delenv("DITTO_RUN_BUDGET_USD", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)

    def no_network(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a real LLM adapter was requested")

    monkeypatch.setattr(ditto.llm.client, "make_adapter", no_network)


def use_fakes(monkeypatch: pytest.MonkeyPatch, fail: Exception | None = None) -> FakeLLMs:
    fakes = FakeLLMs(fail=fail)
    monkeypatch.setattr(ditto.llm, "get_llm", fakes)
    return fakes


class Answers:
    """Scripted terminal: records every prompt and accepts the default (empty answer)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return ""


def outputs(rd: RunDir) -> list[Path]:
    return sorted((rd.root / "out").glob("*/*.jsonl"))


def test_run_all_yes_with_tail_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = use_fakes(monkeypatch)
    opts = RunOptions(
        schema=FIELDS, yes=True, tail="direct", workers=1, max_repair_calls=0, settings="pinned"
    )
    rd = run_all(SITE, tmp_path / "fx", opts)
    assert rd.report_md.exists() and outputs(rd)
    assert set(opts.timings) == {"ingest", "cluster", "analyze"}
    # no repair LLM at 0 repair calls; the small model reads each long-tail page
    assert [m.role for m in fakes.made] == ["generation", "tail_direct"]
    assert len(fakes.calls("generation")) == 4 and fakes.calls("tail_direct")
    assert read_json(rd.review_json)["signed_off_by"] == "yes"


def test_run_all_interactive_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = use_fakes(monkeypatch)
    monkeypatch.setattr(ditto.pipeline, "interactive", lambda: True)
    answers = Answers()
    opts = RunOptions(schema=FIELDS, workers=1, max_repair_calls=1, input_fn=answers)
    rd = run_all(SITE, tmp_path / "fx", opts)
    assert rd.mapping_json.exists() and outputs(rd)
    assert any("approve" in p for p in answers.prompts)  # review asked in the "terminal"
    assert read_json(rd.review_json)["signed_off_by"] == "user"
    assert "repair" in [m.role for m in fakes.made]


def test_run_all_unprompted_shows_the_schema_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fakes(monkeypatch)
    monkeypatch.setattr(ditto.pipeline, "interactive", lambda: True)
    answers = Answers()
    opts = RunOptions(mode="unprompted", workers=1, max_repair_calls=0, input_fn=answers)
    rd = run_all(SITE, tmp_path / "fx", opts)
    assert rd.schema_proposal.exists() and not rd.probe_json.exists() and outputs(rd)
    assert answers.prompts  # the schema checkpoint and the review both asked


def test_budget_stop_during_generation_keeps_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    broke = BudgetExceeded("LLM budget exceeded: projected $0.05, but only $0.01 remains")
    use_fakes(monkeypatch, fail=broke)
    opts = RunOptions(schema=FIELDS, yes=True, workers=1, max_repair_calls=0, concurrency=1)
    rd = run_all(SITE, tmp_path / "fx", opts)
    err = capsys.readouterr().err
    assert "warning: generation stopped early (budget exhausted: LLM budget exceeded" in err
    assert "unprocessed: t_" in err
    assert rd.report_md.exists()


def test_refusals_before_any_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fakes(monkeypatch)
    out = tmp_path / "fx"
    with pytest.raises(ValueError, match="needs --schema"):
        run_all(SITE, out, RunOptions(yes=True))
    with pytest.raises(ValueError, match="--tail direct"):
        run_all(SITE, out, RunOptions(mode="unprompted", tail="direct", yes=True))
    monkeypatch.setattr(ditto.pipeline, "interactive", lambda: False)
    with pytest.raises(CheckpointRequired, match="rerun with --yes"):
        run_all(SITE, out, RunOptions(schema=FIELDS))
    assert not out.exists()


def test_require_terminal() -> None:
    require_terminal(True, "x")  # --yes never needs a terminal
    if not ditto.pipeline.interactive():  # pytest captures stdin: no terminal here
        with pytest.raises(CheckpointRequired, match="x needs an interactive terminal"):
            require_terminal(False, "x")
