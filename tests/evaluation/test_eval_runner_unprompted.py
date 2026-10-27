from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _eval_tiny_site import FakePipeline, make_tiny_site, truth_predictions

from ditto.evaluation.metrics import MODES
from ditto.evaluation.runner import Hooks, read_proposed_schema, run_eval
from ditto.run import RunDir, write_json

RENAME = {"title": "name", "price": "cost", "headline": "heading", "author": "writer"}
PROPOSED = {
    "t_T01": {"name": "scalar", "cost": "scalar", "share_count": "scalar"},
    "t_T02": {"heading": "scalar", "writer": "scalar"},
}


class ProposingPipeline(FakePipeline):
    """The tiny fake pipeline plus the ``schema_proposal.json`` an unprompted run writes."""

    def continue_run(self, run: RunDir, opts: SimpleNamespace, schema: Any) -> None:
        super().continue_run(run, opts, schema)
        write_json(
            run.schema_proposal,
            {
                "version": 1,
                "mode": "unprompted",
                "templates": {
                    tid: {"fields": {f: {"type": t} for f, t in fields.items()}}
                    for tid, fields in PROPOSED.items()
                },
            },
        )


def _setup(tmp: Path) -> tuple[Path, Hooks, Path]:
    site_dir = make_tiny_site(tmp / "synth")
    preds = {
        k: {RENAME.get(f, f): v for f, v in row.items()} for k, row in truth_predictions().items()
    }
    fake = ProposingPipeline(site_dir, preds)
    hooks = Hooks(
        ingest=fake.ingest,
        cluster=fake.cluster,
        analyze=fake.analyze,
        continue_run=fake.continue_run,
        run_options=fake.run_options,
        page_urls=fake.page_urls,
        read_html=fake.read_html,
        settings=fake.settings,
        resolve_model=fake.resolve_model,
        estimate_cost=fake.estimate_cost,
    )
    cfg = {
        "name": "tiny_unprompted",
        "dataset": "synthetic",
        "dataset_options": {"site_dir": str(site_dir), "name": "tiny"},
        "method": {"kind": "ditto", "mode": "unprompted", "params": {"min_cluster_size": 2}},
        "models": {"generation": "openai/gpt-test", "repair": "openai/gpt-test"},
        "budget_usd": 1.0,
    }
    path = tmp / "tiny_unprompted.yaml"
    path.write_text(json.dumps(cfg))
    datasets = tmp / "datasets"
    (datasets / "synthetic").mkdir(parents=True)
    return path, hooks, datasets


def _judgment(verdicts: dict[tuple[str, str], str]) -> dict[str, Any]:
    return {
        "judge": "build agent (not a human)",
        "source": {"results": "somewhere"},
        "site_rule": "the only site",
        "fields": [
            {"site": "tiny", "template": t, "field": f, "verdict": v, "reason": "r"}
            for (t, f), v in verdicts.items()
        ],
    }


def _run(tmp: Path, judgments: Path) -> dict[str, Any]:
    cfg, hooks, datasets = _setup(tmp)
    out = run_eval(
        cfg,
        results_dir=tmp / "results",
        work_root=tmp / "work",
        datasets_dir=datasets,
        judgments_dir=judgments,
        hooks=hooks,
    )
    results: dict[str, Any] = json.loads((out / "results.json").read_text())
    return results


def test_unprompted_summary_has_recall_per_vertical_and_judged_precision(tmp_path: Path) -> None:
    judgments = tmp_path / "judgments"
    verdicts = {(t, f): "entity" for t, fields in PROPOSED.items() for f in fields}
    verdicts[("t_T01", "share_count")] = "noise"
    write_json(judgments / "synthetic_all_schema_precision.json", _judgment(verdicts))
    results = _run(tmp_path, judgments)
    summary = results["summary"]
    assert summary["unprompted_recall"]["norm"] == pytest.approx(1.0)
    assert summary["unprompted_recall_per_vertical"] == {
        "synthetic": {mode: pytest.approx(1.0) for mode in MODES}
    }
    sp = summary["schema_precision"]
    assert sp["judge"] == "build agent (not a human)"
    assert sp["fields"] == 5 and sp["entity"] == 4
    assert sp["precision"] == pytest.approx(0.8)
    assert sp["gate"] == {"threshold": 0.8, "passed": True}
    assert sp["run_check"]["matches_run"] is True
    (site,) = results["sites"]
    assert site["proposed_schema"] == PROPOSED


def test_judgment_that_does_not_match_the_run_is_flagged(tmp_path: Path) -> None:
    judgments = tmp_path / "judgments"
    verdicts = {("t_T01", "name"): "entity", ("t_T01", "rating"): "noise"}
    write_json(judgments / "synthetic_all_schema_precision.json", _judgment(verdicts))
    sp = _run(tmp_path, judgments)["summary"]["schema_precision"]
    assert sp["precision"] == pytest.approx(0.5)
    assert sp["gate"]["passed"] is False
    check = sp["run_check"]
    assert check["matches_run"] is False
    assert check["stale"] == ["tiny t_T01.rating"]
    assert "tiny t_T02.writer" in check["unjudged"]


def test_no_judgment_file_no_precision(tmp_path: Path) -> None:
    summary = _run(tmp_path, tmp_path / "none")["summary"]
    assert "schema_precision" not in summary
    assert summary["unprompted_recall"]["norm"] == pytest.approx(1.0)


def test_read_proposed_schema(tmp_path: Path) -> None:
    run = RunDir(tmp_path).ensure()
    assert read_proposed_schema(run) == {}
    write_json(
        run.schema_proposal,
        {"templates": {"t_b": {"fields": {"x": {"type": "list"}}}, "t_a": {"fields": {}}}},
    )
    assert read_proposed_schema(run) == {"t_a": {}, "t_b": {"x": "list"}}
