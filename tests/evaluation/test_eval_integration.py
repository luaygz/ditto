"""The eval harness on the real LLM-free stages (ingest, fingerprint, cluster) and generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _eval_tiny_site import BASELINE, make_tiny_site

from ditto.evaluation.datasets.base import EvalSite
from ditto.evaluation.runner import (
    Hooks,
    _estimate_cost,
    _resolve_model,
    _settings,
    interleave_verticals,
    run_eval,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.mark.slow
def test_synthetic_clustering_eval_end_to_end(tmp_path: Path) -> None:
    params = _write(tmp_path / "params.yaml", "templates: 3\ntail_templates: 1\n")
    cfg = _write(
        tmp_path / "synth_tiny.yaml",
        f"""
name: synth_tiny
dataset: synthetic
dataset_options:
  params: {params}
  pages: 150
  truth_only: true
  cache_root: {tmp_path / "cache"}
  sweep: {{near_duplicate_pairs: [0, 1]}}
method: {{params: {{until: cluster, min_cluster_size: 20}}}}
keep_runs: false
""",
    )
    out = run_eval(
        cfg,
        results_dir=tmp_path / "results",
        work_root=tmp_path / "work",
        datasets_dir=tmp_path,
    )
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "ok"
    assert [s["site"] for s in results["sites"]] == [
        "near_duplicate_pairs=0",
        "near_duplicate_pairs=1",
    ]
    for site in results["sites"]:
        assert site["n_pages"] == 150
        assert 0.0 <= site["clustering"]["ari"] <= 1.0
        assert site["clustering"]["n_true_templates"] == 4
        assert site["speed"]["cluster"]["pages"] == 150
        assert "run directory deleted" in site["notes"][-1]
    curve = results["summary"]["curves"]["near_duplicate_pairs"]
    assert [p["value"] for p in curve] == [0, 1]
    assert not any((tmp_path / "work").rglob("pages.sqlite"))
    # the generated sites are cached: a rerun reuses them
    cached = sorted(p.name for p in (tmp_path / "cache").iterdir())
    assert len(cached) == 2
    run_eval(
        cfg, results_dir=tmp_path / "results", work_root=tmp_path / "work", datasets_dir=tmp_path
    )
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == cached


def test_baseline_on_real_ingested_pages(tmp_path: Path) -> None:
    tiny = make_tiny_site(tmp_path / "synth")
    (tmp_path / "baselines").mkdir()
    (tmp_path / "baselines" / "echo.py").write_text(BASELINE)
    cfg = _write(
        tmp_path / "tiny_echo.yaml",
        f"""
name: tiny_echo
dataset: synthetic
dataset_options: {{site_dir: {tiny}, name: tiny}}
method: {{kind: baseline, baseline: echo, params: {{min_cluster_size: 2}}}}
budget_usd: 0.01
""",
    )
    hooks = Hooks(settings=lambda models, budget: None, resolve_model=lambda r, s: "")
    out = run_eval(
        cfg,
        results_dir=tmp_path / "results",
        work_root=tmp_path / "work",
        datasets_dir=tmp_path,
        baselines_dir=tmp_path / "baselines",
        hooks=hooks,
    )
    results = json.loads((out / "results.json").read_text())
    (site,) = results["sites"]
    assert site["status"] == "ok", site["reason"]
    assert site["n_pages"] == 8
    title = site["accuracy"]["fields"]["title"]["raw"]
    assert (title["tp"], title["fp"], title["fn"]) == (4, 0, 0)
    assert site["cost"]["usd_per_1k_pages"] == pytest.approx(
        _estimate_cost("openai/gpt-test", 500 * 8, 20 * 8)[0] * 1000 / 8
    )


def test_default_llm_hooks_pin_models_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings({"generation": "openai/gpt-pinned"}, 1.5)
    assert settings.models["generation"] == "openai/gpt-pinned"
    assert settings.run_budget_usd == 1.5
    assert _resolve_model("generation", settings) == "openai/gpt-pinned"
    with pytest.raises(ValueError, match="cannot resolve"):
        _resolve_model("nonsense", settings)
    usd, estimated = _estimate_cost("openai/gpt-6.1-sol", 1_000_000, 100_000)
    assert usd > 0 and isinstance(estimated, bool)


def test_interleave_verticals_round_robin() -> None:
    def site(name: str, vertical: str) -> EvalSite:
        return EvalSite(name, vertical, [], "dir", None, {}, {})

    sites = [site("a1", "auto"), site("a2", "auto"), site("a3", "auto"), site("b1", "book")]
    assert [s.name for s in interleave_verticals(sites)] == ["a1", "b1", "a2", "a3"]
    assert interleave_verticals([]) == []
