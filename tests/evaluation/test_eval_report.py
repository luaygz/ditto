"""``eval/report.py``: charts and the README/docs result tables, on a fake results tree."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPORT = Path(__file__).resolve().parents[2] / "eval" / "report.py"
SECTIONS = (
    "headline",
    "swde",
    "cc_silver",
    "livewebie",
    "clustering",
    "union",
    "spend",
    "models",
    "reproduce",
    "limitations",
)
PINNED = {"generation": "openai/gpt-6.1-sol", "repair": "openai/gpt-6.1-sol"}
SAME = "↳ ditto, same sites and pages"


@pytest.fixture(scope="module")
def report() -> ModuleType:
    pytest.importorskip("matplotlib")  # the `eval` extra; report.py draws the charts with it
    spec = importlib.util.spec_from_file_location("ditto_eval_report", REPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- fake repository -------------------------------------------------------------------------------


def _config(root: Path, name: str, dataset: str, **extra: Any) -> None:
    data: dict[str, Any] = {"name": name, "dataset": dataset, **extra}
    path = root / "eval" / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _accuracy(f1: float, *, verticals: tuple[str, ...] = ("auto",)) -> dict[str, Any]:
    def modes(v: float) -> dict[str, Any]:
        return {
            "norm": {"precision": v, "recall": v, "f1": v},
            "raw": {"precision": v - 0.1, "recall": v - 0.1, "f1": v - 0.1},
            "canon": {"precision": v, "recall": v, "f1": round(min(1.0, v + 0.02), 3)},
        }

    return {
        "per_vertical": {v: {"n_sites": 2, **modes(f1)} for v in verticals},
        "overall": {"n_sites": 2 * len(verticals), "n_verticals": len(verticals), **modes(f1)},
    }


def _score(pages: int, norm: float, raw: float, canon: float | None, excluded: int = 0) -> dict:
    modes = {"norm": norm, "raw": raw, "canon": canon}
    return {
        "pages_scored": pages,
        "pages_excluded": excluded,
        **{m: {"precision": v, "recall": v, "f1": v} for m, v in modes.items() if v is not None},
    }


def _site_cost(pages: int, templates: int, calls: int, tokens: int, usd: float) -> dict[str, Any]:
    return {
        "pages": pages,
        "templates": templates,
        "calls": calls,
        "input_tokens": tokens,
        "output_tokens": tokens // 40,
        "calls_per_template": calls / templates,
        "usd_per_1k_pages": usd,
    }


def _result(
    root: Path,
    name: str,
    stamp: str,
    *,
    dataset: str,
    split: str = "test",
    label: str = "ditto (prompted)",
    status: str = "ok",
    reason: str | None = None,
    summary: dict[str, Any] | None = None,
    sites: list[dict[str, Any]] | None = None,
    spend: float = 0.0,
    models: list[str] | None = None,
    prompts: list[str] | None = None,
    meta_config: dict[str, Any] | None = None,
) -> Path:
    run = root / "eval" / "results" / name / stamp
    run.mkdir(parents=True)
    results = {
        "name": name,
        "dataset": dataset,
        "split": split,
        "method_label": label,
        "status": status,
        "reason": reason,
        "summary": summary or {},
        "sites": sites or [],
    }
    meta: dict[str, Any] = {
        "name": name,
        "config_path": f"eval/configs/{name}.yaml",
        "status": status,
        "spend_usd": spend,
        "models": {"pinned": {}, "used": models or []},
        "prompt_versions": prompts or [],
        "machine": {"cpu_count": 16, "memory_gb": 62.66},
        "notes": [],
    }
    if meta_config is not None:
        meta["config"] = meta_config
    (run / "results.json").write_text(json.dumps(results))
    (run / "meta.json").write_text(json.dumps(meta))
    return run


def _ditto_summary(f1: float, usd: float, *, sample: bool = False) -> dict[str, Any]:
    return {
        "n_sites": 3,
        "n_ok": 2,
        "status_counts": {"ok": 2, "skipped": 1},
        "accuracy": _accuracy(f1),
        "accuracy_sample": _accuracy(f1 - 0.05) if sample else None,
        "optional_field_recall": {"pages": 40, "norm": {"recall": 0.75}, "raw": {"recall": 0.5}},
        "cost": {
            "pages": 2000,
            "templates": 3,
            "input_tokens_per_1k_pages": 4321.0,
            "output_tokens_per_1k_pages": 210.0,
            "calls_per_template": 1.25,
            "usd_per_1k_pages": usd,
            "extrapolated": False,
            "llm_latency_s_mean": 7.5,
        },
        "speed_pages_per_sec_per_core": {"extract": 812.4, "cluster": 9000.0},
    }


def _ditto_sites() -> list[dict[str, Any]]:
    """Per-site results of the latest swde_test_ditto run: two scored sites, one skipped."""
    return [
        {
            "site": "a",
            "vertical": "auto",
            "status": "ok",
            "accuracy": _score(700, 0.9, 0.8, 0.95, excluded=3),
            "accuracy_sample": _score(19, 0.85, 0.75, 0.9, excluded=1),
            "cost": _site_cost(1000, 1, 1, 40000, 0.05),
            "speed": {"extract": {"pages_per_sec_per_core": 800.0}},
        },
        {
            "site": "b",
            "vertical": "book",
            "status": "ok",
            "accuracy": _score(300, 0.7, 0.6, 0.72, excluded=2),
            "accuracy_sample": _score(20, 0.65, 0.55, 0.7),
            "cost": _site_cost(500, 2, 3, 20000, 0.08),
            "speed": {"extract": {"pages_per_sec_per_core": 600.0}},
        },
        {"site": "c", "vertical": "camera", "status": "skipped"},
    ]


def _baseline_summary(f1: float, usd: float) -> dict[str, Any]:
    return {
        "n_sites": 2,
        "n_ok": 2,
        "accuracy": _accuracy(f1),
        "cost": {
            "pages": 40,
            "input_tokens_per_1k_pages": 11300000.0,
            "output_tokens_per_1k_pages": 90000.0,
            "calls_per_template": 7.0,
            "usd_per_1k_pages": usd,
            "extrapolated": True,
        },
    }


def _baseline_sites() -> list[dict[str, Any]]:
    return [
        {
            "site": s,
            "vertical": vertical,
            "status": "ok",
            "speed": {"baseline": {"seconds": 10.0, "pages": 20}},
            "accuracy": _score(20, norm, norm - 0.05, None),
            "cost": {**_site_cost(20, 1, 20, 200000, usd), "extrapolated": True},
        }
        for s, vertical, norm, usd in (("a", "auto", 0.95, 20.0), ("b", "book", 0.93, 23.0))
    ]


CURVES = {
    "templates": [
        {"value": 12, "status": "ok", "ari": 0.98, "purity": 0.99, "n_pages": 10000},
        {"value": 2, "status": "ok", "ari": 1.0, "purity": 1.0, "n_pages": 10000},
        {"value": 200, "status": "failed"},
    ],
    "noise": [
        {"value": [], "status": "ok", "ari": 0.99, "purity": 1.0, "f1_norm": 0.9},
        {"value": ["nbsp"], "status": "ok", "ari": 0.97, "purity": 1.0, "f1_norm": None},
    ],
    "near_duplicate_jaccard": [
        {"value": 0.9, "status": "ok", "ari": 0.993, "purity": 0.99, "over_merge": 1},
        {"value": 0.6, "status": "ok", "ari": 1.0, "purity": 1.0, "over_merge": 0},
    ],
    "markup_style": [{"value": "react", "status": "failed"}],
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repository root (cwd) with configs, results, dataset files and target docs."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    sample20 = {"sample": {"pages_per_site": 20}, "models": PINNED, "budget_usd": 10.0}

    _config(root, "swde_test_ditto", "swde", selection={"split": "test"}, **sample20)
    _config(
        root,
        "swde_test_naive",
        "swde",
        selection={"split": "test"},
        method={"kind": "baseline", "baseline": "naive_llm"},
        **sample20,
    )
    _config(
        root,
        "swde_test_crawl4ai",
        "swde",
        selection={"split": "test"},
        method={"kind": "baseline", "baseline": "crawl4ai_schema"},
    )
    _config(
        root,
        "swde_test_handwritten",
        "swde",
        selection={"split": "test", "sites": ["a", "z", "y"]},
        method={"kind": "baseline", "baseline": "handwritten"},
        budget_usd=0.01,
    )
    _config(root, "swde_dev_ditto", "swde", selection={"split": "dev"}, models=PINNED)
    _config(
        root,
        "swde_dev_unprompted",
        "swde",
        selection={"split": "dev"},
        method={"kind": "ditto", "mode": "unprompted"},
        models=PINNED,
    )
    _config(root, "cc_silver_ditto", "cc_silver", models=PINNED, budget_usd=6.0)
    _config(
        root,
        "cc_silver_crawl4ai",
        "cc_silver",
        method={"kind": "baseline", "baseline": "crawl4ai_schema"},
    )
    _config(root, "livewebie_ditto", "livewebie", selection={"split": "test"})
    _config(root, "synth_default", "synthetic", method={"params": {"until": "cluster"}})
    _config(
        root,
        "synth_curves",
        "synthetic",
        dataset_options={"params": "eval/synth/default.yaml", "sweep": {"templates": [2, 12]}},
        method={"params": {"until": "cluster"}},
    )
    _config(
        root,
        "union_gate_synth",
        "synthetic",
        method={"params": {"until": "union"}},
    )

    # results: an older and a newer run plus a dry run without results.json
    _result(
        root,
        "swde_test_ditto",
        "20261001T000000Z",
        dataset="swde",
        summary=_ditto_summary(0.5, 0.2),
        spend=4.0,
    )
    _result(
        root,
        "swde_test_ditto",
        "20261024T000000Z",
        dataset="swde",
        summary=_ditto_summary(0.91, 0.0431, sample=True),
        sites=_ditto_sites(),
        spend=1.5,
        models=["openai/gpt-6.1-sol"],
        prompts=["repair.v1", "specgen.v1"],
    )
    dry = root / "eval" / "results" / "swde_test_ditto" / "20261025T000000Z-dryrun"
    dry.mkdir()
    (dry / "meta.json").write_text(json.dumps({"dry_run": True, "spend_usd": 9.0}))
    _result(
        root,
        "swde_test_naive",
        "20261024T000000Z",
        dataset="swde",
        label="naive_llm",
        summary=_baseline_summary(0.94, 21.5),
        sites=_baseline_sites(),
        spend=2.25,
        models=["openai/gpt-6.1-sol"],
        prompts=["naive_baseline.v1"],
    )
    _result(
        root,
        "swde_test_handwritten",
        "20261024T000000Z",
        dataset="swde",
        label="handwritten",
        summary={
            "n_sites": 3,
            "n_ok": 2,
            "status_counts": {"ok": 2, "not_run": 1},
            "accuracy": _accuracy(0.99),
            "cost": {"pages": 900, "usd_per_1k_pages": 0.0, "calls_per_template": 0.0},
        },
        sites=[
            {
                "site": "a",
                "vertical": "auto",
                "status": "ok",
                "accuracy": _score(703, 1.0, 1.0, 1.0),
                "cost": _site_cost(703, 1, 0, 0, 0.0),
            },
            {
                "site": "z",
                "vertical": "auto",
                "status": "ok",
                "accuracy": _score(197, 0.98, 0.98, 0.98),
                "cost": _site_cost(197, 1, 0, 0, 0.0),
            },
        ],
    )
    _result(
        root,
        "swde_dev_ditto",
        "20261024T000000Z",
        dataset="swde",
        split="dev",
        summary=_ditto_summary(0.99, 0.01),
    )
    _result(
        root,
        "swde_dev_unprompted",
        "20261024T000000Z",
        dataset="swde",
        split="dev",
        label="ditto (unprompted)",
        summary={
            **_ditto_summary(0.85, 0.12),
            "unprompted_recall": {"norm": 0.847, "raw": 0.831, "canon": 0.85},
            "schema_precision": {
                "file": "eval/judgments/swde_dev_schema_precision.json",
                "precision": 0.9077,
                "entity": 187,
                "fields": 206,
                "n_sites": 10,
            },
        },
    )
    _result(
        root,
        "cc_silver_ditto",
        "20261024T000000Z",
        dataset="cc_silver",
        split="all",
        summary=_ditto_summary(0.8, 0.04),
        spend=0.75,
    )
    _result(
        root,
        "cc_silver_crawl4ai",
        "20261024T000000Z",
        dataset="cc_silver",
        split="all",
        label="crawl4ai_schema",
        status="not_run",
        reason="data/cc/argos.co.uk/labels.jsonl not found: run labels.py",
    )
    _result(
        root,
        "livewebie_ditto",
        "20261024T000000Z",
        dataset="livewebie",
        summary={
            "n_sites": 9,
            "n_ok": 9,
            "accuracy": _accuracy(0.7),
            "groups": {
                "norm": {
                    "type_1": {"f1": 0.81, "n_sites": 9},
                    "type_3": {"f1": 0.62, "n_sites": 4},
                },
                "raw": {"type_1": {"f1": 0.71, "n_sites": 9}, "type_3": {"f1": 0.52, "n_sites": 4}},
                "canon": {"type_1": {"f1": 0.83, "n_sites": 9}},
            },
            "cost": {"input_tokens_per_1k_pages": 900.0, "calls_per_template": 1.0},
        },
    )
    _result(
        root,
        "synth_default",
        "20261024T000000Z",
        dataset="synthetic",
        split="all",
        label="ditto (clustering)",
        summary={
            "n_sites": 1,
            "n_ok": 1,
            "clustering": {
                "ari": 1.0,
                "purity": 1.0,
                "n_pages": 10000.0,
                "n_true_templates": 15.0,
                "n_clusters": 15.0,
                "over_merge": 0.0,
                "under_merge": 0.0,
            },
            "speed_pages_per_sec_per_core": {"fingerprint": 315.6, "cluster": 11282.6},
        },
    )
    _result(
        root,
        "synth_curves",
        "20261024T000000Z",
        dataset="synthetic",
        split="all",
        label="ditto (clustering)",
        summary={"n_sites": 9, "n_ok": 7, "curves": CURVES},
    )
    _result(
        root,
        "union_gate_synth",
        "20261024T000000Z",
        dataset="synthetic",
        split="all",
        label="ditto (union)",
        summary={
            "union": {
                "n_sites": 1,
                "n_templates": 12,
                "tokens": {"min": 8100, "median": 12000, "max": 29000},
                "over_budget": 0,
                "coverage": {"optional_total": 36, "covered": 36, "rate": 1.0},
                "gate": {"pass": True},
            }
        },
    )
    # a result whose config file is gone: kept, described by the config embedded in meta.json
    _result(
        root,
        "old_eval",
        "20261020T000000Z",
        dataset="swde",
        split="dev",
        label="naive_llm",
        summary=_baseline_summary(0.9, 20.0),
        sites=[{"site": "q", "vertical": "auto", "status": "ok", "accuracy": _score(5, 1, 1, 1)}],
        spend=0.5,
        meta_config={
            "name": "old_eval",
            "dataset": "swde",
            "selection": {"split": "dev"},
            "method": {"kind": "baseline", "baseline": "naive_llm"},
        },
    )

    synth = root / "eval" / "synth" / "default.yaml"
    synth.parent.mkdir(parents=True)
    synth.write_text("templates: 12\nnoise: []\n")
    judged = root / "eval" / "judgments" / "swde_dev_schema_precision.json"
    judged.parent.mkdir(parents=True)
    judged.write_text(json.dumps({"fields": []}))
    analysis = root / "eval" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "swde_dev_misses.json").write_text(
        json.dumps(
            {
                "per_method": {
                    "ditto": {
                        "norm_misses": 1000,
                        "sample": {"classified": 84},
                        "population_estimate": {
                            "formatting": 150,
                            "real_error": 150,
                            "label_problem": 700,
                        },
                    }
                }
            }
        )
    )
    cc = root / "eval" / "datasets" / "commoncrawl_silver"
    cc.mkdir(parents=True)
    (cc / "domains.yaml").write_text(
        yaml.safe_dump(
            {
                "crawls": ["CC-MAIN-2026-43", "CC-MAIN-2026-39"],
                "selection_date": "2026-10-24",
                "download": {"cap_per_domain": 2500},
                "selected": [
                    {
                        "domain": "argos.co.uk",
                        "vertical": "product",
                        "target_type": "Product",
                        "crawls": ["CC-MAIN-2026-43", "CC-MAIN-2026-39"],
                        "evidence": {"distinct_html_urls": 33133},
                        "captures": {
                            "index_html_captures": {
                                "CC-MAIN-2026-39": 15620,
                                "CC-MAIN-2026-43": 17747,
                            },
                            "distinct_urls": 33100,
                            "planned": 7201,
                            "downloaded": 6701,
                            "downloaded_detail": 6701,
                            "downloaded_other": 0,
                        },
                    },
                    {
                        "domain": "dice.com",
                        "vertical": "job",
                        "target_type": "JobPosting",
                        "crawls": ["CC-MAIN-2026-43"],
                        "evidence": {"distinct_html_urls": 22873},
                    },
                ],
            }
        )
    )
    (cc / "noise_check.json").write_text(
        json.dumps(
            {
                "performed_by": "build agent (automated script noise_check.py), not a human",
                "source": "labels",
                "pages": 50,
                "complete": False,
                "domains_checked": ["argos.co.uk"],
                "domains_missing": ["dice.com"],
                "per_vertical_field": {"product": {"name": {"labeled": 50, "rate": 1.0}}},
                "selection_sample": {
                    "source": "selection",
                    "pages": 200,
                    "complete": True,
                    "domains_checked": ["argos.co.uk", "dice.com"],
                    "domains_missing": [],
                    "per_domain_field": {
                        "argos.co.uk": {"name": {"labeled": 25, "rate": 1.0}},
                        "dice.com": {
                            "datePosted": {"labeled": 25, "rate": 0.0},
                            "skills": {"labeled": 20, "rate": 0.6, "item_rate": 0.9},
                        },
                    },
                },
            }
        )
    )
    lw = root / "eval" / "datasets" / "livewebie"
    lw.mkdir(parents=True)
    (lw / "manifest.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "summary": "9 of 15 websites usable",
                "counts": {"websites": 15},
                "sites": {"dp.la": {"excluded_reason": "no page could be fetched"}, "x.org": {}},
            }
        )
    )
    blocks = "\n\n".join(
        f"<!-- results:{s}:start -->\nSTALE\n<!-- results:{s}:end -->" for s in SECTIONS
    )
    (root / "README.md").write_text(f"# ditto\n\nIntro.\n\n{blocks}\n\nLicense.\n")
    (root / "docs").mkdir()
    (root / "docs" / "benchmarks.md").write_text(f"# Benchmarks\n\n{blocks}\n")
    return root


def _context(report: ModuleType) -> Any:
    entries = report.load_entries(Path("eval/configs"), Path("eval/results"))
    return report.Context(
        entries=entries, datasets_dir=Path("eval/datasets"), analysis_dir=Path("eval/analysis")
    )


def _render(report: ModuleType, repo: Path, *, detail: bool) -> dict[str, str]:
    target = report.Target(Path("docs/benchmarks.md") if detail else Path("README.md"), detail)
    return dict(report.render_sections(_context(report), target))


def _row(text: str, first_cell: str, nth: int = 0) -> list[str]:
    rows = [
        [c.strip() for c in line.strip("|").split(" | ")]
        for line in text.splitlines()
        if line.startswith(f"| {first_cell} |")
    ]
    if len(rows) <= nth:
        raise AssertionError(f"no row {first_cell!r} #{nth} in:\n{text}")
    return rows[nth]


def _rows_after(text: str, first_cell: str) -> list[str]:
    """The row right after the row starting with ``first_cell``."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"| {first_cell} |"):
            return [c.strip() for c in lines[i + 1].strip("|").split(" | ")]
    raise AssertionError(f"no row {first_cell!r} in:\n{text}")


# -- loading ---------------------------------------------------------------------------------------


def test_latest_results_picks_newest_completed_run(report: ModuleType, repo: Path) -> None:
    latest = report.latest_results(Path("eval/results"))
    assert latest["swde_test_ditto"].path.name == "20261024T000000Z"
    assert latest["swde_test_ditto"].billed_all_runs == pytest.approx(5.5)  # dry run excluded
    assert latest["swde_test_ditto"].n_runs == 2
    assert "swde_test_crawl4ai" not in latest
    assert report.latest_results(Path("missing")) == {}


def test_load_entries_lists_every_config_and_orphan_results(report: ModuleType, repo: Path) -> None:
    entries = report.load_entries(Path("eval/configs"), Path("eval/results"))
    assert entries["swde_test_crawl4ai"].result is None
    assert entries["swde_test_crawl4ai"].status_text() == "not run yet"
    assert entries["swde_test_naive"].config.label == "naive per-page LLM"
    assert entries["swde_test_naive"].config.pages_per_site == 20
    orphan = entries["old_eval"]
    assert orphan.config.kind == "baseline" and orphan.config.split == "dev"
    assert entries["swde_test_ditto"].status_text() == "ok on 2 of 3 sites (1 skipped)"
    assert entries["cc_silver_crawl4ai"].status_text().startswith("not run: data/cc/argos")


# -- like-for-like comparisons -------------------------------------------------------------------


def test_overall_f1_aggregates_sites_then_verticals(report: ModuleType) -> None:
    sites = [
        {"vertical": "auto", "accuracy": {"norm": {"f1": 1.0}}},
        {"vertical": "auto", "accuracy": {"norm": {"f1": 0.5}}},
        {"vertical": "book", "accuracy": {"norm": {"f1": 0.25}}},
        {"vertical": "job", "accuracy": {"norm": {"f1": None}}},
    ]
    assert report.overall_f1(sites, "accuracy", "norm") == pytest.approx((0.75 + 0.25) / 2)
    assert report.overall_f1(sites, "accuracy", "canon") is None
    assert report.pooled_cost([{"cost": {"pages": 0}}]) == {}


def test_compare_pairs_each_baseline_with_ditto_on_its_sites_and_pages(
    report: ModuleType, repo: Path
) -> None:
    entries = report.load_entries(Path("eval/configs"), Path("eval/results"))
    ditto = entries["swde_test_ditto"]
    naive = report.compare(entries["swde_test_naive"], ditto)
    assert naive.sites == ("a", "b") and naive.ditto_key == "accuracy_sample"
    assert naive.f1("ditto") == pytest.approx(0.75)  # sample F1: (0.85 + 0.65) / 2 verticals
    assert naive.f1("ditto", "canon") == pytest.approx(0.8)
    assert naive.f1("baseline") == pytest.approx(0.94)
    assert naive.pages_scored("ditto") == 39 and not naive.partial
    cost = naive.cost("ditto")
    assert cost["usd_per_1k_pages"] == pytest.approx(0.06)  # (0.05 + 0.04) / 1,500 pages
    assert cost["calls_per_template"] == pytest.approx(1.25)
    assert naive.speed() == pytest.approx(700.0)

    written = report.compare(entries["swde_test_handwritten"], ditto)
    assert written.sites == ("a",) and written.ditto_key == "accuracy" and written.partial
    assert written.f1("ditto") == pytest.approx(0.9)

    assert report.compare(entries["swde_test_crawl4ai"], ditto) is None  # not run
    assert report.compare(entries["swde_test_naive"], None).startswith("no ditto")
    assert report.compare(entries["old_eval"], entries["swde_dev_ditto"]) == (
        "no site scored by both"
    )
    entries["swde_test_ditto"].config = report.ConfigInfo.from_mapping(
        {"name": "swde_test_ditto", "dataset": "swde", "sample": {"pages_per_site": 10}}, "x"
    )
    assert "scores no 20-page samples" in report.compare(entries["swde_test_naive"], ditto)


def test_headline_points_pair_baselines_with_ditto(report: ModuleType, repo: Path) -> None:
    points = report.headline_points(report.load_entries(Path("eval/configs"), Path("eval/results")))
    swde = [(p["kind"], p["group"], p["label"], round(p["f1"], 3)) for p in points["swde"]]
    assert swde == [
        ("ditto", None, "ditto, all 2 sites", 0.91),
        ("baseline", 0, "naive per-page LLM\n2 sites × 20 pages", 0.94),
        ("ditto", 0, "ditto on naive's 2 sites × 20 pages", 0.75),
        ("baseline", 1, "hand-written XPath (build agent)\n1 site", 1.0),
        ("ditto", 1, "ditto on hand-written's 1 site", 0.9),
    ]
    assert points["swde"][1]["extrapolated"] is True
    assert [p["name"] for p in points["cc_silver"]] == ["cc_silver_ditto"]


# -- tables ----------------------------------------------------------------------------------------


def test_swde_table_reads_numbers_from_the_latest_results(report: ModuleType, repo: Path) -> None:
    swde = _render(report, repo, detail=False)["swde"]
    test_part, dev_part = swde.split("**SWDE development slice")
    ditto = _row(test_part, "ditto (prompted)")
    assert ditto[1:10] == [
        "2 sites, all labeled pages the LLM did not see",
        "1,000",
        "0.910",
        "0.810",
        "0.930",
        "4,321",
        "210",
        "1.25",
        "$0.0431",
    ]
    assert ditto[10] == "812 pages/s/core (extract)"
    naive = _row(test_part, "naive per-page LLM")
    assert naive[1] == "2 of 3 sites, seeded 20-page sample per site"
    assert naive[3] == "0.940" and naive[5] == "0.960" and naive[9] == "$21.50 (extrapolated)"
    assert naive[10] == "2.00 pages/s (wall)"
    same = _rows_after(test_part, "naive per-page LLM")
    assert same[0] == SAME
    assert same[1:7] == [
        "the same 2 sites, same samples minus pages the LLM saw",
        "39",
        "0.750",
        "0.650",
        "0.800",
        "40,000",
    ]
    assert same[9:] == ["$0.0600", "700 pages/s/core (extract)", "ok"]
    written = _row(test_part, "hand-written XPath (build agent)")
    assert written[1] == "2 of 3 sites, all labeled pages"
    restricted = _row(test_part, "↳ hand-written XPath (build agent), same sites")
    assert restricted[1:4] == ["the same 1 site (ditto lacks the others)", "703", "1.000"]
    assert _row(test_part, SAME, 1)[1:4] == [
        "the same 1 site, labeled pages the LLM did not see",
        "700",
        "0.900",
    ]
    crawl4ai = _row(test_part, "Crawl4AI generate_schema")
    assert crawl4ai[-1] == "not run yet"
    assert not re.search(r"\d", "".join(crawl4ai[1:])), "a config without results shows no number"
    assert "rows: ditto scored on exactly" not in test_part  # the README explains it once
    assert "the LLM saw 5 of 1,005 labeled pages (0.50%) while generating 3 specs" in test_part
    assert "optional-field recall" in test_part and "normalized 0.750" in test_part
    assert "[swde_test_ditto](eval/results/swde_test_ditto/20261024T000000Z)" in test_part
    assert "`swde_test_crawl4ai` (not run)" in test_part
    assert "uv run ditto eval run eval/configs/<name>.yaml" in test_part
    assert _row(dev_part, "ditto (prompted)")[3] == "0.990"
    assert _row(dev_part, "naive per-page LLM")[3] == "0.900"  # the orphan result
    assert _rows_after(dev_part, "naive per-page LLM")[-1] == "no site scored by both"
    assert "schema precision: 0.908 (187 of 206 proposed fields on 10 sites" in dev_part
    assert "by the build agent, not a human" in dev_part
    assert "canon 0.850" in dev_part


def test_detail_tables_add_verticals_stages_and_per_config_commands(
    report: ModuleType, repo: Path
) -> None:
    swde = _render(report, repo, detail=True)["swde"]
    assert "Per vertical (mean of its sites" in swde
    assert "rows: ditto scored on exactly the sites and pages of the baseline" in swde
    assert _row(swde, "auto")[1:] == [
        "ditto (prompted)",
        "2",
        "0.910",
        "0.910",
        "0.910",
        "0.810",
        "0.930",
    ]
    assert "| swde_test_ditto | – | – | 9,000 |" in swde
    assert "16 cores, 62.66 GB" in swde
    assert (
        "- ditto (prompted) (`swde_test_ditto`): [eval/results/swde_test_ditto/20261024T000000Z/]"
        "(../eval/results/swde_test_ditto/20261024T000000Z); regenerate with "
        "`uv run ditto eval run eval/configs/swde_test_ditto.yaml`"
    ) in swde


def test_cc_silver_section_names_domains_and_noise_check(report: ModuleType, repo: Path) -> None:
    readme = _render(report, repo, detail=False)["cc_silver"]
    assert _row(readme, "`argos.co.uk`")[1:] == [
        "product",
        "Product",
        "CC-MAIN-2026-43: 17,747; CC-MAIN-2026-39: 15,620",
        "33,100",
        "6,701 of 7,201",
    ]
    assert _row(readme, "`dice.com`")[3:] == ["CC-MAIN-2026-43", "22,873", "not downloaded yet"]
    assert "selected 2026-10-24" in readme and "download cap 2,500 pages per domain" in readme
    crawl4ai = _row(readme, "Crawl4AI generate_schema")
    assert crawl4ai[-1] == "not run: data/cc/argos.co.uk/labels.jsonl not found: run labels.py"
    # the downloaded-page check covers one of two domains, so the selection-sample check is shown
    assert "on 200 pages from the domain-selection sample, 2 domains" in readme
    assert "performed by build agent (automated script noise_check.py), not a human" in readme
    assert "2 fields agree on fewer than 80%" in readme and "(#limitations)" in readme
    detail = _render(report, repo, detail=True)["cc_silver"]
    assert _row(detail, "dice.com", 1)[1:] == ["skills", "20", "0.600", "0.900"]
    assert "(6,701 detail, 0 other)" in detail
    assert "`dice.com` datePosted 0.00 (25 pages); `dice.com` skills 0.60 (20 pages)" in detail


def test_livewebie_reports_complexity_types_or_skip_reason(report: ModuleType, repo: Path) -> None:
    text = _render(report, repo, detail=False)["livewebie"]
    assert "Snapshot status: partial. 9 of 15 websites usable" in text
    assert "Excluded: `dp.la` (no page could be fetched)." in text
    detail = _render(report, repo, detail=True)["livewebie"]
    assert "- excluded `dp.la`: no page could be fetched" in detail
    assert _row(text, "type_1 (one single-valued attribute)")[1:] == [
        "0.810",
        "0.710",
        "0.830",
        "9",
    ]
    assert _row(text, "type_3 (one list-valued attribute)")[2:4] == ["0.520", "–"]
    assert _row(text, "all fields")[1:] == ["0.700", "0.600", "0.720", "9"]

    manifest = Path("eval/datasets/livewebie/manifest.json")
    manifest.write_text(json.dumps({"status": "skipped", "reason": "release unavailable"}))
    for run in Path("eval/results/livewebie_ditto").iterdir():
        for f in run.iterdir():
            f.unlink()
        run.rmdir()
    sections = _render(report, repo, detail=False)
    assert "LiveWeb-IE skipped: release unavailable" in sections["livewebie"]
    assert "ditto (prompted) (`livewebie_ditto`): not run yet." in sections["livewebie"]
    assert "- **LiveWeb-IE skipped**: release unavailable." in sections["limitations"]


def test_clustering_section_has_scale_table_and_curves(report: ModuleType, repo: Path) -> None:
    ctx = _context(report)
    ctx.curve_charts = {"templates": Path("docs/img/synth_templates.png")}
    text = report.render_sections(ctx, report.Target(Path("README.md")))["clustering"]
    assert _row(text, "synth_default")[1:10] == [
        "10,000",
        "15",
        "15",
        "1.000",
        "1.000",
        "0",
        "0",
        "316",
        "11,283",
    ]
    templates = _row(text, "templates")
    assert templates[1] == "2: 1.000 · 12 (default): 0.980 · 200: failed"  # numeric order
    assert templates[2:] == ["0.980", "0.990", "[chart](docs/img/synth_templates.png)"]
    assert _row(text, "noise")[1] == "none (default): 0.990 · [nbsp]: 0.970"
    assert _row(text, "markup_style")[2:] == ["–", "–", "–"]
    assert "union_gate_synth" not in text  # gate runs are not clustering evaluations
    detail = report.render_sections(ctx, report.Target(Path("docs/benchmarks.md"), True))
    assert "Every point:" in detail["clustering"]
    assert "[chart](img/synth_templates.png)" in detail["clustering"]


def test_union_section(report: ModuleType, repo: Path) -> None:
    text = _render(report, repo, detail=True)["union"]
    row = _row(text, "`union_gate_synth`")
    assert row[1:] == [
        "synthetic (all)",
        "1",
        "12",
        "8,100 / 12,000 / 29,000",
        "0",
        "36 of 36",
        "pass",
        "ok",
    ]


def test_spend_and_models(report: ModuleType, repo: Path) -> None:
    readme = _render(report, repo, detail=False)
    spend = readme["spend"]
    assert _row(spend, "`swde_test_ditto`")[2:6] == [
        "ok on 2 of 3 sites (1 skipped)",
        "$0.0862",  # 2,000 pages at $0.0431 per 1,000
        "$5.50 (2 runs)",
        "$10.00",
    ]
    assert _row(spend, "`swde_test_naive`")[3] == "$0.8600"
    assert "`synth_default` |" not in spend
    assert (
        "Run without any LLM call: [synth_curves](eval/results/synth_curves/20261024T000000Z), "
        "[synth_default](eval/results/synth_default/20261024T000000Z)"
    ) in spend
    total_cost = 0.0862 + 0.86 + 0.02 + 0.24 + 0.08 + 0.8  # ditto test, naive, dev, unprompted, cc
    assert f"Total LLM cost of the published results: ${total_cost:.2f}" in spend
    assert "Billed in the 13 runs kept under `eval/results/`: $9.00" in spend
    assert "Not run yet: `swde_test_crawl4ai`." in spend
    detail = _render(report, repo, detail=True)["spend"]
    assert _row(detail, "`swde_test_crawl4ai`")[2:5] == ["not run yet", "–", "–"]
    assert _row(detail, "`synth_default`")[3:6] == ["–", "$0", "no LLM"]
    assert "Run without any LLM call" not in detail
    models = readme["models"]
    assert "`openai/gpt-6.1-sol` (2 evaluations)" in models
    assert "`naive_baseline.v1`, `repair.v1`, `specgen.v1`" in models
    assert "generation: `openai/gpt-6.1-sol`; repair: `openai/gpt-6.1-sol`" in models


def test_reproduce_lists_every_config_and_the_judgment_file(report: ModuleType, repo: Path) -> None:
    text = _render(report, repo, detail=True)["reproduce"]
    assert _row(text, "`swde_test_naive`")[1:] == [
        "SWDE test split",
        "naive per-page LLM",
        "all",
        "20 pages / site",
        "$10.00",
        "`uv run ditto eval run eval/configs/swde_test_naive.yaml`",
    ]
    assert _row(text, "`swde_test_handwritten`")[3] == "3 listed"
    assert _row(text, "`union_gate_synth`")[2:6] == ["ditto to `union`", "all", "–", "no LLM"]
    assert _row(text, "`livewebie_ditto`")[4] == "all pages"
    assert "`swde_dev_unprompted` also regenerates the schema precision" in text
    assert "(../eval/judgments/swde_dev_schema_precision.json)" in text


def test_limitations_are_generated_from_results_and_datasets(
    report: ModuleType, repo: Path
) -> None:
    text = _render(report, repo, detail=False)["limitations"]
    bullets = text.splitlines()
    assert all(b.startswith("- **") for b in bullets)
    assert "seeded samples of 20 on the SWDE test split (pages per site) instead of the " in text
    assert "planned 200" in text
    assert (
        "**naive per-page LLM** on the SWDE test split covers 2 of 3 sites, on a seeded "
        "20-page sample per site (budget)"
    ) in text
    assert "**hand-written XPath (build agent)** on the SWDE test split covers 2 of 3" in text
    assert "**Crawl4AI generate_schema** on the Common Crawl silver set: not run: data/cc" in text
    assert "capped at 2,500 pages per domain** (`argos.co.uk` 7,201: pages fetched" in text
    assert "6,701 of the 7,201 planned pages had been downloaded" in text
    assert "`dice.com` datePosted 0.00 (25 pages)" in text
    assert "**LiveWeb-IE on 1 of 15 sites**" in text and "non-commercial research" in text
    assert "minimum ARI 0.970 over 6 points; templates merge only at" in text
    assert "near_duplicate_jaccard = 0.9." in text
    assert "dominate ditto's remaining dev misses**: of its 1,000 misses" in text
    assert "an estimated 70% are label problems" in text and "84 misses" in text
    assert "**Extraction speed in the evaluation runs** meets the target of 500" in text
    assert "812 on `cc_silver_ditto`" in text and "16 logical CPUs" in text


def test_no_results_means_no_numbers(report: ModuleType, tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for name, dataset in (("swde_test_ditto", "swde"), ("cc_silver_ditto", "cc_silver")):
        data = {"name": name, "dataset": dataset, "selection": {"split": "test"}}
        data |= {"models": PINNED, "budget_usd": 5}
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(data))
    entries = report.load_entries(configs, tmp_path / "results")
    ctx = report.Context(
        entries=entries, datasets_dir=tmp_path / "datasets", analysis_dir=tmp_path / "analysis"
    )
    for target in (report.Target(tmp_path / "README.md"), report.Target(tmp_path / "b.md", True)):
        for name, body in report.render_sections(ctx, target).items():
            prose = re.sub(r"`[^`]*`|\]\([^)]*\)", "", body)  # code spans and link targets
            prose = re.sub(r"\bF1\b|1k pages|1,000 pages", "", prose)  # column names and units
            prose = prose.replace("| $5.00 |", "| |")  # the configs' budget cap (not a result)
            assert not re.search(r"\d", prose), f"{name}: {prose}"
            if name in ("headline", "swde", "cc_silver", "limitations"):
                assert "not run yet" in body


# -- spend per build phase ---------------------------------------------------------------------


def test_spend_by_phase_splits_the_ledger_at_tag_dates(report: ModuleType) -> None:
    tags = [
        ("v0.1.0", datetime.fromisoformat("2026-10-26T12:00:00+02:00")),
        ("v0.2.0", datetime.fromisoformat("2026-10-27T12:00:00+02:00")),
    ]
    rows = [
        {"ts": "2026-10-24T09:00:00+00:00", "role": "generation", "cost_usd": 0.5},
        {"ts": "2026-10-26T09:59:59+00:00", "role": "baseline", "cost_usd": 1.0},
        {"ts": "2026-10-26T10:00:00+00:00", "role": "baseline", "cost_usd": 2.0},
        {"ts": "2026-10-27T10:00:00+00:00", "role": "repair", "cost_usd": 4.0},
        {"ts": "2026-10-27T11:00:00+00:00", "role": "repair", "cost_usd": 0.0, "cached": True},
    ]
    for row in rows:
        row.setdefault("list_cost_usd", row["cost_usd"])
    phases = report.spend_by_phase(rows, tags)
    assert [(p["phase"], p["calls"], p["billed"]) for p in phases] == [
        ("before v0.1.0", 2, 1.5),
        ("v0.1.0–v0.2.0", 1, 2.0),
        ("v0.2.0–now", 1, 4.0),
    ]
    table = "\n".join(report.spend_by_phase_table(phases))
    assert _row(table, "total")[3:6] == ["4", "$7.50", "$7.50"]
    assert "generation $0.5000" in _row(table, "before v0.1.0")[6]
    assert "undercounting the provider's charges by about 23%" in table
    assert report.spend_by_phase(rows, [])[0]["phase"] == "all"


def test_main_spend_by_phase(
    report: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert report.main(["--spend-by-phase"]) == 1
    assert "ledger.jsonl: not found" in capsys.readouterr().out
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"ts": "2026-10-24T09:00:00+00:00", "role": "baseline", "cost_usd": 0.25})
        + "\n\n"
    )
    monkeypatch.setattr(
        report,
        "tag_dates",
        lambda: [("v0.1.0", datetime.fromisoformat("2026-10-25T00:00:00+00:00"))],
    )
    assert report.main(["--spend-by-phase", "--ledger", str(ledger)]) == 0
    out = capsys.readouterr().out
    assert "| before v0.1.0 |" in out and "$0.2500" in out
    assert not (tmp_path / "README.md").exists()


# -- markers and main --------------------------------------------------------------------------------


def test_update_markers(report: ModuleType) -> None:
    text = "a\n<!-- results:swde:start -->\nold\n<!-- results:swde:end -->\nb\n"
    text += "<!-- results:other:start -->x<!-- results:other:end -->\n"
    new, unknown = report.update_markers(text, {"swde": "NEW"})
    assert new == (
        "a\n<!-- results:swde:start -->\nNEW\n<!-- results:swde:end -->\nb\n"
        "<!-- results:other:start -->x<!-- results:other:end -->\n"
    )
    assert unknown == ["other"]
    assert report.update_markers(new, {"swde": "NEW"})[0] == new
    with pytest.raises(ValueError, match="unclosed"):
        report.update_markers("<!-- results:swde:start -->\n", {"swde": "x"})


def test_main_writes_charts_tables_and_check(
    report: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report.main([]) == 0
    img = repo / "docs" / "img"
    written = sorted(p.name for p in img.iterdir())
    assert written == [
        "headline_cost_f1.png",
        "synth_near_duplicate_jaccard.png",
        "synth_noise.png",
        "synth_templates.png",
    ]
    for name in written:
        assert (img / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    readme = (repo / "README.md").read_text()
    assert readme.startswith("# ditto\n\nIntro.\n") and readme.endswith("License.\n")
    assert "STALE" not in readme
    assert "![LLM cost per 1,000 pages against F1](docs/img/headline_cost_f1.png)" in readme
    assert "dotted line to ditto (star, same colour)" in readme
    bench = (repo / "docs" / "benchmarks.md").read_text()
    assert "](img/headline_cost_f1.png)" in bench
    assert "results sections rewritten" in capsys.readouterr().out

    assert report.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out
    (repo / "README.md").write_text(readme.replace("0.910", "0.111"))
    assert report.main(["--check"]) == 1
    assert "out of date: README.md" in capsys.readouterr().out


def test_main_without_results_or_docs(
    report: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert report.main([]) == 0
    out = capsys.readouterr().out
    assert "README.md: not found, skipped" in out
    assert "0 of 0 configs have results" in out
    assert not (tmp_path / "docs" / "img").exists()
