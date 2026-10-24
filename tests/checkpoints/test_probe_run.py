"""probe_run end to end on a hand-built run directory: probe.json, probe.yaml, --probe-llm."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from checkpoint_builders import clusters, job_schema, job_table, listing_table, make_run

from ditto.checkpoints import ProbeResult, probe_run
from ditto.checkpoints.probe import answer_schema
from ditto.llm.base import Completion, Message, Usage
from ditto.run import RunDir


@dataclass
class FakeLLM:
    """Scripted LLM: returns the queued answers in order and records every request."""

    answers: list[str]
    model: str = "fake/probe-mini"
    role: str = "probe"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": messages,
                "json_schema": json_schema,
                "prompt_version": prompt_version,
                "template_id": template_id,
            }
        )
        return Completion(text=self.answers.pop(0), usage=Usage(100, 20), model=self.model)


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    cl = clusters(
        {"t_job1": 120, "t_list1": 40, "t_nocand": 25},
        tail={"t_tail": 3},
    )
    return make_run(tmp_path / "run", [job_table(), listing_table()], cl)


def test_probe_writes_json_and_yaml(run: RunDir) -> None:
    result = probe_run(run, job_schema())
    assert set(result.templates) == {"t_job1", "t_list1"}  # tail clusters are not probed
    assert result.skipped == {"t_nocand": "no candidate table (run analysis first)"}

    job = result.templates["t_job1"]
    assert job.size == 120
    assert job.url_examples == [f"https://example.org/t_job1/{i}" for i in range(3)]
    assert job.include
    assert set(job.fields_held) == {"title", "company", "location", "date_posted", "salary"}
    company = job.fields["company"]
    assert (company.status, company.use) == ("found", "c4")
    assert company.examples == ["Acme Corp", "Globex Inc", "Initech"]
    assert company.candidates[0].id == "c4"
    assert job.fields["salary"].status == "ambiguous"
    assert job.fields["remote_policy"].status == "missing"

    listing = result.templates["t_list1"]
    assert not listing.include
    assert listing.exclude_reason == "holds no requested field"
    assert listing.fields_held == []

    on_disk = json.loads(run.probe_json.read_text())
    assert on_disk["templates"]["t_job1"]["fields"]["company"]["candidates"][0]["class"] == (
        "mandatory"
    )
    assert ProbeResult.load(run.probe_json).counts() == result.counts()
    assert result.counts()["excluded"] == 1

    text = run.probe_yaml.read_text()
    assert text.startswith("# ditto probe")
    data = yaml.safe_load(text)
    assert data["fields"] == job_schema().names()
    assert data["templates"]["t_list1"]["include"] is False
    tj = data["templates"]["t_job1"]
    assert tj["include"] is True
    assert tj["fields"]["company"]["use"] == "c4"
    assert tj["fields"]["remote_policy"]["use"] == "absent"
    assert tj["fields"]["company"]["candidates"][0].startswith("c4 (1.00) · mandatory")

    timings = json.loads(run.timings_json.read_text())
    assert timings["probe"]["pages"] == 160


def test_probe_llm_resolves_ambiguous_fields(run: RunDir) -> None:
    llm = FakeLLM([json.dumps({"fields": {"salary": {"candidate": "c7", "reason": "pay"}}})])
    result = probe_run(run, job_schema(), use_llm=True, llm=llm)
    assert len(llm.calls) == 1  # only t_job1 has an ambiguous field
    call = llm.calls[0]
    assert call["template_id"] == "t_job1"
    assert call["prompt_version"] == "probe.v1"
    assert call["json_schema"] == answer_schema(["salary"])
    user = call["messages"][-1]["content"]
    assert "c7 | optional | p=0.60" in user
    assert "- salary: the pay" in user
    # the candidate table alone: labels and boilerplate are not offered
    assert "c3 |" not in user and "c0 |" not in user
    salary = result.templates["t_job1"].fields["salary"]
    assert (salary.status, salary.use, salary.resolved_by) == ("found", "c7", "llm")
    assert salary.note == "pay"
    assert salary.examples == ["$120,000", "$95,000", "$40,000"]
    assert result.llm.used and result.llm.calls == 1 and result.llm.resolved == 1
    assert result.llm.model == "fake/probe-mini"
    assert json.loads(run.probe_json.read_text())["llm"]["resolved"] == 1


def test_probe_llm_absent_answer_marks_field_missing(run: RunDir) -> None:
    llm = FakeLLM(['{"fields": {"salary": {"candidate": "absent", "reason": "no pay shown"}}}'])
    result = probe_run(run, job_schema(), use_llm=True, llm=llm)
    salary = result.templates["t_job1"].fields["salary"]
    assert (salary.status, salary.use, salary.resolved_by) == ("missing", "absent", "llm")


def test_probe_llm_invalid_answers_keep_field_ambiguous(run: RunDir) -> None:
    bad = '{"fields": {"salary": {"candidate": "c3", "reason": "label"}}}'
    llm = FakeLLM([bad, '{"fields": {"salary": {"candidate": "c99", "reason": "?"}}}'])
    result = probe_run(run, job_schema(), use_llm=True, llm=llm)
    assert len(llm.calls) == 2  # one retry with the validation errors
    retry_note = llm.calls[1]["messages"][-1]["content"]
    assert "c3 is a label" in retry_note
    assert result.templates["t_job1"].fields["salary"].status == "ambiguous"
    assert "c99" in result.llm.errors["t_job1"]
    assert result.llm.resolved == 0


def test_probe_without_ambiguity_makes_no_call(tmp_path: Path) -> None:
    run = make_run(tmp_path / "r", [listing_table()], clusters({"t_list1": 30}))
    llm = FakeLLM([])
    result = probe_run(run, job_schema(), use_llm=True, llm=llm)
    assert llm.calls == []
    assert result.llm.used and result.llm.calls == 0


def test_llm_rows_trim_to_most_relevant(monkeypatch: pytest.MonkeyPatch) -> None:
    from ditto.checkpoints import probe as probe_mod
    from ditto.checkpoints.match import prepare

    monkeypatch.setattr(probe_mod, "MAX_LLM_ROWS", 2)
    rows = probe_mod.llm_rows(prepare(job_table()), [job_schema().get("company")])  # type: ignore[list-item]
    assert [r.split(" | ")[0] for r in rows] == ["c4", "c7"]  # kept in id order


def test_probe_llm_can_pick_a_table(tmp_path: Path) -> None:
    from checkpoint_builders import kv_table

    from ditto.schema import FieldsSchema

    run = make_run(tmp_path / "r", [kv_table()], clusters({"t_kv1": 30}))
    schema = FieldsSchema.from_obj({"feedback": {"type": "record_list"}})
    first = probe_run(run, schema)
    assert first.templates["t_kv1"].fields["feedback"].status == "missing"
    # force ambiguity so the LLM is asked, then answer with a table id
    first.templates["t_kv1"].fields["feedback"].status = "ambiguous"
    from ditto.analyze.models import CandidateTable
    from ditto.checkpoints.probe import resolve_ambiguous

    llm = FakeLLM(['{"fields": {"feedback": {"candidate": "tb1", "reason": "reviews"}}}'])
    resolve_ambiguous(first, {"t_kv1": CandidateTable.load(run.candidates("t_kv1"))}, schema, llm)
    fp = first.templates["t_kv1"].fields["feedback"]
    assert (fp.status, fp.use) == ("found", ["c4", "c5"])
    assert "tb1 | record_list table" in llm.calls[0]["messages"][-1]["content"]


@pytest.mark.live
def test_live_probe_llm_resolves_salary(run: RunDir) -> None:
    from ditto.llm import get_llm

    llm = get_llm("probe", run=run)
    result = probe_run(run, job_schema(), use_llm=True, llm=llm)
    assert result.llm.calls == 1
    assert result.llm.errors == {}
    salary = result.templates["t_job1"].fields["salary"]
    assert salary.resolved_by == "llm"
    assert (salary.status, salary.use) == ("found", "c7")
