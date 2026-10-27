"""Baseline pages whose LLM call failed: all failed -> the site fails; some -> ok with a note."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _eval_tiny_site import FakePipeline, make_tiny_site

from ditto.evaluation.runner import Hooks, run_eval

# A baseline whose calls all fail on site "allfail" and fail on one page on site "somefail".
FLAKY = """
from ditto.evaluation.baseline_api import BaselineOutput


def run_baseline(pages, schema, *, out_dir, settings=None, max_pages=None, seed=0, site=None):
    n = len(pages)
    failed = n if site == "allfail" else 1
    preds = {p.key: {"title": None} for p in pages}
    return BaselineOutput(
        predictions=preds, calls=n, input_tokens=100 * n, output_tokens=0,
        cost_usd=0.001 * n, model="openai/gpt-test", spend_usd=0.002,
        notes=[f"{failed} pages: provider error 429"], pages_failed=failed,
        prompt_version="flaky_prompt.v1",
    )
"""


def _run(tmp: Path, site_root: Path, name: str) -> dict[str, Any]:
    bdir = tmp / "baselines"
    bdir.mkdir(exist_ok=True)
    (bdir / "flaky.py").write_text(FLAKY)
    datasets = tmp / "datasets"
    datasets.mkdir(exist_ok=True)
    cfg = {
        "name": f"flaky_{name}",
        "dataset": "synthetic",
        "dataset_options": {"site_dir": str(site_root), "name": name},
        "method": {"kind": "baseline", "baseline": "flaky", "params": {"min_cluster_size": 2}},
        "models": {"baseline": "openai/gpt-test"},
        "budget_usd": 1.0,
        "sample": {"pages_per_site": 4},
    }
    path = tmp / f"{name}.yaml"
    path.write_text(json.dumps(cfg))
    fake = FakePipeline(site_root, {})
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
    out = run_eval(
        path,
        results_dir=tmp / "results",
        work_root=tmp / "work",
        datasets_dir=datasets,
        baselines_dir=bdir,
        hooks=hooks,
    )
    return dict(json.loads((out / "results.json").read_text()))


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    return make_tiny_site(tmp_path / "synth")


def test_every_baseline_page_failed_fails_the_site(tiny: Path, tmp_path: Path) -> None:
    results = _run(tmp_path, tiny, "allfail")
    (site,) = results["sites"]
    assert site["status"] == "failed"
    assert site["reason"] == "all 4 baseline pages failed (see notes)"
    assert site["accuracy"] is None  # not scored as F1 0
    assert site["cost"]["pages_failed"] == 4
    assert site["spend_usd"] == pytest.approx(0.002)  # what the failed calls billed still counts
    assert "4 pages: provider error 429" in site["notes"]
    assert results["status"] == "failed"
    assert results["summary"]["n_ok"] == 0


def test_partial_baseline_failures_stay_ok_with_a_note(tiny: Path, tmp_path: Path) -> None:
    results = _run(tmp_path, tiny, "somefail")
    (site,) = results["sites"]
    assert site["status"] == "ok", site["reason"]
    assert site["accuracy"]["pages_scored"] == 4
    assert site["cost"]["pages_failed"] == 1
    assert "1 of 4 baseline pages failed (scored as empty)" in site["notes"]
    assert site["prompt_versions"] == ["flaky_prompt.v1"]
