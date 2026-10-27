"""``ditto.pipeline``: ``run_all`` on the fixture site with fake LLMs, both checkpoints answered
through ``input_fn`` as in a terminal, the unprompted path, ``--tail direct`` and ``--tail spec``,
review-triggered repairs, a budget stop during generation, and the refusals that happen before
any work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pipeline_fakes import FIELDS, SITE, FakeLLM, FakeLLMs

import ditto.llm
import ditto.llm.client
import ditto.pipeline
from ditto.cluster.models import Clusters
from ditto.llm.base import BudgetExceeded
from ditto.pipeline import (
    CheckpointRequired,
    RunOptions,
    require_terminal,
    run_all,
    stage_analyze,
    stage_cluster,
    stage_ingest,
)
from ditto.run import RunDir, read_json
from ditto.schema import FieldsSchema
from ditto.specgen.models import Spec
from ditto.specgen.tail import tail_direct_run


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


def test_tail_spec_covers_the_tail_and_tail_direct_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = use_fakes(monkeypatch)
    opts = RunOptions(schema=FIELDS, yes=True, tail="spec", workers=1, max_repair_calls=0)
    rd = run_all(SITE, tmp_path / "fx", opts)
    tail = [c for c in Clusters.load(rd.clusters_json).clusters if c.is_tail]
    assert len(tail) == 3 and all(3 <= c.size <= 5 for c in tail)
    generated = {c["template_id"] for c in fakes.calls("generation")}
    for c in tail:  # generated, verified and extracted like any template
        assert c.template_id in generated
        assert Spec.load(rd.spec(c.template_id)).status == "final"
        lines = rd.out_jsonl(rd.site, c.template_id).read_text().splitlines()
        rows = [json.loads(line) for line in lines]
        assert len(rows) == c.size and all(row["title"] for row in rows)

    # --tail direct afterwards: only tail clusters without a final spec go to the small model
    schema = FieldsSchema.load(FIELDS)
    llm = FakeLLM(rd, "tail_direct")
    assert tail_direct_run(rd, schema, llm=llm).pages == 0 and llm.calls == []
    rd.spec(tail[0].template_id).unlink()
    result = tail_direct_run(rd, schema, llm=llm)
    assert result.pages == tail[0].size == len(llm.calls)
    assert {call["template_id"] for call in llm.calls} == {tail[0].template_id}


class RejectTitle(Answers):
    """Names ``title`` at the first review sign-off prompt, accepts everything else."""

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "what is wrong" in prompt:
            return "wrong node"
        if sum("fields to reject" in p for p in self.prompts) == 1 and "fields to reject" in prompt:
            return "title"
        return ""


@pytest.mark.parametrize("max_repair_calls", [0, 1])
def test_review_repairs_use_the_run_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_repair_calls: int
) -> None:
    fakes = use_fakes(monkeypatch)
    monkeypatch.setattr(ditto.pipeline, "interactive", lambda: True)
    opts = RunOptions(
        schema=FIELDS,
        workers=1,
        max_repair_calls=max_repair_calls,
        no_cache=True,
        settings="pinned",
        input_fn=RejectTitle(),
    )
    rd = run_all(SITE, tmp_path / "fx", opts)
    assert read_json(rd.review_json)["repair"]["called"] is True
    # verification's repair LLM when there is one, else one built at review: same options
    repairs = [r for r in fakes.requests if r["role"] == "repair"]
    assert repairs == [{"role": "repair", "settings": "pinned", "no_cache": True}]


def test_stage_analyze_passes_workers_to_union_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DITTO_WORKERS", "3")  # what union pages used without the option
    rd = stage_ingest(SITE, tmp_path / "fx", workers=1)
    stage_cluster(rd, workers=1)
    stage_analyze(rd, workers=1)
    assert read_json(rd.timings_json)["union"]["workers"] == 1


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
