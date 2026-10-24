from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPORT = Path(__file__).resolve().parents[2] / "eval" / "report.py"


def _load_report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ditto_eval_report", REPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(
    root: Path,
    name: str,
    stamp: str,
    *,
    dataset: str,
    label: str,
    f1: float | None,
    usd: float | None,
    split: str = "test",
    status: str = "ok",
    curves: dict[str, Any] | None = None,
) -> None:
    run = root / name / stamp
    run.mkdir(parents=True)
    summary: dict[str, Any] = {
        "accuracy": {"overall": {"norm": {"f1": f1}, "raw": {"f1": f1 and f1 - 0.1}}}
        if f1 is not None
        else None,
        "cost": {
            "usd_per_1k_pages": usd,
            "input_tokens_per_1k_pages": 1234.0,
            "calls_per_template": 1.5,
            "extrapolated": label == "naive_llm",
        },
        "speed_pages_per_sec_per_core": {"extract": 812.5},
        "curves": curves or {},
    }
    results = {
        "name": name,
        "dataset": dataset,
        "split": split,
        "method_label": label,
        "status": status,
        "summary": summary,
    }
    (run / "results.json").write_text(json.dumps(results))
    (run / "meta.json").write_text(json.dumps({"config_path": f"eval/configs/{name}.yaml"}))


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    _result(
        root,
        "swde_test_ditto",
        "20261001T000000Z",
        dataset="swde",
        label="ditto (prompted)",
        f1=0.5,
        usd=0.2,
    )
    _result(
        root,
        "swde_test_ditto",
        "20261024T000000Z",
        dataset="swde",
        label="ditto (prompted)",
        f1=0.91,
        usd=0.05,
    )
    (root / "swde_test_ditto" / "20261025T000000Z-dryrun").mkdir()  # no results.json: ignored
    _result(
        root,
        "swde_test_naive",
        "20261024T000000Z",
        dataset="swde",
        label="naive_llm",
        f1=0.94,
        usd=21.0,
    )
    _result(
        root,
        "swde_test_handwritten",
        "20261024T000000Z",
        dataset="swde",
        label="handwritten",
        f1=0.97,
        usd=None,
    )
    _result(
        root,
        "swde_dev_ditto",
        "20261024T000000Z",
        dataset="swde",
        label="ditto (prompted)",
        f1=0.99,
        usd=0.01,
        split="dev",
    )
    _result(
        root,
        "cc_silver_ditto",
        "20261024T000000Z",
        dataset="cc_silver",
        label="ditto (prompted)",
        f1=0.8,
        usd=0.04,
        split="all",
    )
    _result(
        root,
        "cc_silver_crawl4ai",
        "20261024T000000Z",
        dataset="cc_silver",
        label="crawl4ai_schema",
        f1=None,
        usd=None,
        split="all",
        status="not_run",
    )
    curves = {
        "templates": [
            {"value": 2, "status": "ok", "ari": 1.0, "purity": 1.0},
            {"value": 200, "status": "ok", "ari": 0.9, "purity": 0.95},
            {"value": 500, "status": "failed"},
        ],
        "noise": [
            {"value": [], "status": "ok", "ari": 0.99, "purity": 1.0, "f1_norm": 0.9},
            {"value": ["nbsp"], "status": "ok", "ari": 0.98, "purity": 1.0, "f1_norm": None},
        ],
        "markup_style": [{"value": "react", "status": "failed"}],
    }
    _result(
        root,
        "synth_curves",
        "20261024T000000Z",
        dataset="synthetic",
        label="ditto (clustering)",
        f1=None,
        usd=None,
        split="all",
        curves=curves,
    )
    return root


def test_latest_results_picks_newest_completed_run(results_dir: Path) -> None:
    report = _load_report()
    latest = report.latest_results(results_dir)
    assert latest["swde_test_ditto"].path.name == "20261024T000000Z"
    assert set(latest) == {
        "swde_test_ditto",
        "swde_test_naive",
        "swde_test_handwritten",
        "swde_dev_ditto",
        "cc_silver_ditto",
        "cc_silver_crawl4ai",
        "synth_curves",
    }
    assert report.latest_results(results_dir / "missing") == {}


def test_headline_points_use_test_split_and_completed_runs(results_dir: Path) -> None:
    report = _load_report()
    points = report.headline_points(report.latest_results(results_dir))
    swde = {p["name"]: p for p in points["swde"]}
    assert set(swde) == {"swde_test_ditto", "swde_test_naive", "swde_test_handwritten"}
    assert swde["swde_test_ditto"]["f1"] == pytest.approx(0.91)
    assert swde["swde_test_naive"]["extrapolated"] is True
    assert swde["swde_test_handwritten"]["usd_per_1k_pages"] is None
    assert [p["name"] for p in points["cc_silver"]] == ["cc_silver_ditto"]


def test_main_writes_charts_and_prints_commands(
    results_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _load_report()
    out_dir = tmp_path / "img"
    assert report.main(["--results-dir", str(results_dir), "--out-dir", str(out_dir)]) == 0
    written = sorted(p.name for p in out_dir.iterdir())
    assert written == ["headline_cost_f1.png", "synth_noise.png", "synth_templates.png"]
    for name in written:
        assert (out_dir / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    text = capsys.readouterr().out
    assert "uv run ditto eval run eval/configs/swde_test_naive.yaml" in text
    assert "uv run ditto eval run eval/configs/synth_curves.yaml" in text
    assert "uv run python eval/report.py" in text
    assert "| swde_test_ditto | ditto (prompted) | ok | 0.910 | 0.810 |" in text


def test_main_without_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _load_report()
    assert report.main(["--results-dir", str(tmp_path), "--out-dir", str(tmp_path / "img")]) == 1
    assert "no results" in capsys.readouterr().out
