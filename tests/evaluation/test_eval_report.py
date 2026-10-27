"""``eval/report.py``: charts and the README/docs result tables, on a fake results tree."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPORT = Path(__file__).resolve().parents[2] / "eval" / "report.py"
SECTIONS = ("headline", "swde", "cc_silver", "livewebie", "clustering", "union", "spend", "models")
PINNED = {"generation": "openai/gpt-6.1-sol", "repair": "openai/gpt-6.1-sol"}


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
        }

    return {
        "per_vertical": {v: {"n_sites": 2, **modes(f1)} for v in verticals},
        "overall": {"n_sites": 2 * len(verticals), "n_verticals": len(verticals), **modes(f1)},
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
            "input_tokens_per_1k_pages": 4321.0,
            "output_tokens_per_1k_pages": 210.0,
            "calls_per_template": 1.25,
            "usd_per_1k_pages": usd,
            "extrapolated": False,
            "llm_latency_s_mean": 7.5,
        },
        "speed_pages_per_sec_per_core": {"extract": 812.4, "cluster": 9000.0},
    }


def _baseline_summary(f1: float, usd: float) -> dict[str, Any]:
    return {
        "n_sites": 2,
        "n_ok": 2,
        "accuracy": _accuracy(f1),
        "cost": {
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
            "status": "ok",
            "speed": {"baseline": {"seconds": 10.0, "pages": 20}},
            "accuracy": {"pages_scored": 20},
        }
        for s in ("a", "b")
    ]


CURVES = {
    "templates": [
        {"value": 2, "status": "ok", "ari": 1.0, "purity": 1.0, "n_pages": 10000},
        {"value": 12, "status": "ok", "ari": 0.98, "purity": 0.99, "n_pages": 10000},
        {"value": 200, "status": "failed"},
    ],
    "noise": [
        {"value": [], "status": "ok", "ari": 0.99, "purity": 1.0, "f1_norm": 0.9},
        {"value": ["nbsp"], "status": "ok", "ari": 0.97, "purity": 1.0, "f1_norm": None},
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
    _config(root, "swde_dev_ditto", "swde", selection={"split": "dev"}, models=PINNED)
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
    )
    _result(
        root,
        "swde_test_ditto",
        "20261024T000000Z",
        dataset="swde",
        summary=_ditto_summary(0.91, 0.0431, sample=True),
        sites=[{"site": "a", "status": "ok", "accuracy": {"pages_scored": 700}}],
        spend=1.5,
        models=["openai/gpt-6.1-sol"],
        prompts=["repair.v1", "specgen.v1"],
    )
    (root / "eval" / "results" / "swde_test_ditto" / "20261025T000000Z-dryrun").mkdir()
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
        "swde_dev_ditto",
        "20261024T000000Z",
        dataset="swde",
        split="dev",
        summary=_ditto_summary(0.99, 0.01),
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
        summary={"n_sites": 7, "n_ok": 5, "curves": CURVES},
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
    cc = root / "eval" / "datasets" / "commoncrawl_silver"
    cc.mkdir(parents=True)
    (cc / "domains.yaml").write_text(
        yaml.safe_dump(
            {
                "crawls": ["CC-MAIN-2026-43", "CC-MAIN-2026-39"],
                "selection_date": "2026-10-24",
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
                            "downloaded": 20000,
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
                "pages": 200,
                "per_vertical_field": {
                    "product": {"name": {"labeled": 50, "rate": 0.98}},
                    "recipe": {"ingredients": {"labeled": 40, "rate": 0.6, "item_rate": 0.9}},
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


def _render(report: ModuleType, repo: Path, *, detail: bool) -> dict[str, str]:
    entries = report.load_entries(Path("eval/configs"), Path("eval/results"))
    ctx = report.Context(entries=entries, datasets_dir=Path("eval/datasets"))
    target = report.Target(Path("docs/benchmarks.md") if detail else Path("README.md"), detail)
    return dict(report.render_sections(ctx, target))


def _row(text: str, first_cell: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(f"| {first_cell} |"):
            return [c.strip() for c in line.strip("|").split(" | ")]
    raise AssertionError(f"no row {first_cell!r} in:\n{text}")


# -- loading ---------------------------------------------------------------------------------------


def test_latest_results_picks_newest_completed_run(report: ModuleType, repo: Path) -> None:
    latest = report.latest_results(Path("eval/results"))
    assert latest["swde_test_ditto"].path.name == "20261024T000000Z"
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


def test_headline_points_use_test_split_and_completed_runs(report: ModuleType, repo: Path) -> None:
    points = report.headline_points(report.latest_results(Path("eval/results")))
    swde = {p["name"]: p for p in points["swde"]}
    assert set(swde) == {"swde_test_ditto", "swde_test_naive"}
    assert swde["swde_test_ditto"]["f1"] == pytest.approx(0.91)
    assert swde["swde_test_naive"]["extrapolated"] is True
    assert [p["name"] for p in points["cc_silver"]] == ["cc_silver_ditto"]


# -- tables ----------------------------------------------------------------------------------------


def test_swde_table_reads_numbers_from_the_latest_results(report: ModuleType, repo: Path) -> None:
    swde = _render(report, repo, detail=False)["swde"]
    test_part, dev_part = swde.split("**SWDE development slice")
    ditto = _row(test_part, "ditto (prompted)")
    assert ditto[1:9] == [
        "all labeled pages the LLM did not see",
        "700",
        "0.910",
        "0.810",
        "4,321",
        "210",
        "1.25",
        "$0.0431",
    ]
    assert ditto[9] == "812 pages/s/core (extract)"
    assert "the baselines' 20-page sample per site" in test_part
    naive = _row(test_part, "naive per-page LLM")
    assert naive[3] == "0.940" and naive[8] == "$21.50 (extrapolated)"
    assert naive[9] == "2.00 pages/s (wall)"
    crawl4ai = _row(test_part, "Crawl4AI generate_schema")
    assert crawl4ai[-1] == "not run yet"
    assert not re.search(r"\d", "".join(crawl4ai[1:])), "a config without results shows no number"
    assert "optional-field recall" in test_part and "normalized 0.750" in test_part
    assert "[swde_test_ditto](eval/results/swde_test_ditto/20261024T000000Z)" in test_part
    assert "`swde_test_crawl4ai` (not run)" in test_part
    assert "uv run ditto eval run eval/configs/<name>.yaml" in test_part
    assert _row(dev_part, "ditto (prompted)")[3] == "0.990"
    assert _row(dev_part, "naive per-page LLM")[3] == "0.900"  # the orphan result


def test_detail_tables_add_verticals_stages_and_per_config_commands(
    report: ModuleType, repo: Path
) -> None:
    swde = _render(report, repo, detail=True)["swde"]
    assert "Per vertical (mean of its sites):" in swde
    assert _row(swde, "auto")[1:] == ["ditto (prompted)", "2", "0.910", "0.910", "0.910", "0.810"]
    assert "| swde_test_ditto | – | – | 9,000 |" in swde
    assert "16 cores, 62.66 GB" in swde
    assert (
        "- ditto (prompted) (`swde_test_ditto`): [eval/results/swde_test_ditto/20261024T000000Z/]"
        "(../eval/results/swde_test_ditto/20261024T000000Z); regenerate with "
        "`uv run ditto eval run eval/configs/swde_test_ditto.yaml`"
    ) in swde


def test_cc_silver_section_names_domains_and_reports_not_run(
    report: ModuleType, repo: Path
) -> None:
    readme = _render(report, repo, detail=False)["cc_silver"]
    assert _row(readme, "`argos.co.uk`")[1:] == [
        "product",
        "Product",
        "CC-MAIN-2026-43, CC-MAIN-2026-39",
        "CC-MAIN-2026-43: 17,747; CC-MAIN-2026-39: 15,620",
        "33,100",
        "20,000",
    ]
    assert _row(readme, "`dice.com`")[4:] == ["–", "22,873", "not downloaded yet"]
    assert "selected 2026-10-24" in readme
    crawl4ai = _row(readme, "Crawl4AI generate_schema")
    assert crawl4ai[-1] == "not run: data/cc/argos.co.uk/labels.jsonl not found: run labels.py"
    assert "performed by build agent (automated script noise_check.py), not a human" in readme
    detail = _render(report, repo, detail=True)["cc_silver"]
    assert _row(detail, "recipe")[1:] == ["ingredients", "40", "0.600", "0.900"]


def test_livewebie_reports_complexity_types_or_skip_reason(report: ModuleType, repo: Path) -> None:
    text = _render(report, repo, detail=False)["livewebie"]
    assert "Snapshot status: partial. 9 of 15 websites usable" in text
    assert "excluded `dp.la`: no page could be fetched" in text
    assert _row(text, "type_1 (one single-valued attribute)")[1:] == ["0.810", "0.710", "9"]
    assert _row(text, "all fields")[1:] == ["0.700", "0.600", "9"]

    manifest = Path("eval/datasets/livewebie/manifest.json")
    manifest.write_text(json.dumps({"status": "skipped", "reason": "release unavailable"}))
    for run in Path("eval/results/livewebie_ditto").iterdir():
        for f in run.iterdir():
            f.unlink()
        run.rmdir()
    text = _render(report, repo, detail=False)["livewebie"]
    assert "LiveWeb-IE skipped: release unavailable" in text
    assert "ditto (prompted) (`livewebie_ditto`): not run yet." in text


def test_clustering_section_has_scale_table_and_curves(report: ModuleType, repo: Path) -> None:
    entries = report.load_entries(Path("eval/configs"), Path("eval/results"))
    ctx = report.Context(
        entries=entries,
        datasets_dir=Path("eval/datasets"),
        curve_charts={"templates": Path("docs/img/synth_templates.png")},
    )
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
    assert templates[1] == "2: 1.000 · 12 (default): 0.980 · 200: failed"
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
    assert _row(spend, "`swde_test_ditto`")[2:5] == [
        "ok on 2 of 3 sites (1 skipped)",
        "$1.50",
        "$10.00",
    ]
    assert "`synth_default` |" not in spend
    assert (
        "Run without any LLM call: [synth_curves](eval/results/synth_curves/20261024T000000Z), "
        "[synth_default](eval/results/synth_default/20261024T000000Z)"
    ) in spend
    assert "Total billed for the latest run of every evaluation: $5.00." in spend
    assert "Not run yet: `swde_test_crawl4ai`." in spend
    detail = _render(report, repo, detail=True)["spend"]
    assert _row(detail, "`swde_test_crawl4ai`")[2:4] == ["not run yet", "–"]
    assert _row(detail, "`synth_default`")[3:5] == ["$0", "no LLM"]
    assert "Run without any LLM call" not in detail
    models = readme["models"]
    assert "`openai/gpt-6.1-sol` (2 evaluations)" in models
    assert "`naive_baseline.v1`, `repair.v1`, `specgen.v1`" in models
    assert "generation: `openai/gpt-6.1-sol`; repair: `openai/gpt-6.1-sol`" in models


def test_no_results_means_no_numbers(report: ModuleType, tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for name, dataset in (("swde_test_ditto", "swde"), ("cc_silver_ditto", "cc_silver")):
        data = {"name": name, "dataset": dataset, "selection": {"split": "test"}}
        data |= {"models": PINNED, "budget_usd": 5}
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(data))
    entries = report.load_entries(configs, tmp_path / "results")
    ctx = report.Context(entries=entries, datasets_dir=tmp_path / "datasets")
    for target in (report.Target(tmp_path / "README.md"), report.Target(tmp_path / "b.md", True)):
        for name, body in report.render_sections(ctx, target).items():
            prose = re.sub(r"`[^`]*`|\]\([^)]*\)", "", body)  # code spans and link targets
            prose = re.sub(r"\bF1\b|1k pages|1,000 pages", "", prose)  # column names and units
            prose = prose.replace(
                "| $5.00 |", "| |"
            )  # the configs' budget cap (config, not result)
            assert not re.search(r"\d", prose), f"{name}: {prose}"
            if name in ("headline", "swde", "cc_silver"):
                assert "not run yet" in body


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
    assert written == ["headline_cost_f1.png", "synth_noise.png", "synth_templates.png"]
    for name in written:
        assert (img / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    readme = (repo / "README.md").read_text()
    assert readme.startswith("# ditto\n\nIntro.\n") and readme.endswith("License.\n")
    assert "STALE" not in readme
    assert "![LLM cost per 1,000 pages against F1](docs/img/headline_cost_f1.png)" in readme
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
