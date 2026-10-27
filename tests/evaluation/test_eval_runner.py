from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from _eval_tiny_site import BASELINE, FakePipeline, make_tiny_site, truth_predictions

from ditto.evaluation.datasets.base import EvalSite, Selection
from ditto.evaluation.datasets.synthetic import read_site
from ditto.evaluation.metrics import METRIC_VERSION
from ditto.evaluation.runner import Hooks, cost_summary, manifest_hashes, run_eval


def _hooks(fake: FakePipeline, **extra: Any) -> Hooks:
    return Hooks(
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
        **extra,
    )


def _config(tmp: Path, site_dir: Path, **overrides: Any) -> Path:
    cfg: dict[str, Any] = {
        "name": "tiny_eval",
        "dataset": "synthetic",
        "dataset_options": {"site_dir": str(site_dir), "name": "tiny"},
        "method": {"kind": "ditto", "mode": "prompted", "params": {"min_cluster_size": 2}},
        "models": {"generation": "openai/gpt-test", "repair": "openai/gpt-test"},
        "budget_usd": 1.0,
        "sample": {"pages_per_site": 4},
    }
    cfg.update(overrides)
    path = tmp / f"{cfg['name']}.yaml"
    path.write_text(json.dumps(cfg))  # JSON is valid YAML
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    return make_tiny_site(tmp_path / "synth")


@pytest.fixture
def datasets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "datasets"
    (d / "synthetic").mkdir(parents=True)
    (d / "synthetic" / "manifest.json").write_text('{"seeds": [1]}\n')
    (d / "swde").mkdir()  # no manifest: not listed
    return d


def _run(tmp: Path, cfg: Path, hooks: Hooks, datasets_dir: Path, **kw: Any) -> Path:
    return run_eval(
        cfg,
        results_dir=tmp / "results",
        work_root=tmp / "work",
        datasets_dir=datasets_dir,
        hooks=hooks,
        **kw,
    )


def test_prompted_run_writes_results_and_meta(
    tiny: Path, tmp_path: Path, datasets_dir: Path
) -> None:
    preds = truth_predictions()
    preds["p000001"]["price"] = "$99"  # wrong value: FP + FN
    preds["p000002"]["title"] = "  WIDGET THREE. "  # normalized match only
    preds["p000003"]["price"] = "$5"  # label is null: FP
    preds["p000006"]["author"] = None  # miss on a field the generating page lacked
    fake = FakePipeline(tiny, preds)
    out = _run(tmp_path, _config(tmp_path, tiny), _hooks(fake), datasets_dir)

    assert out.parent == tmp_path / "results" / "tiny_eval"
    assert {p.name for p in out.iterdir()} == {
        "results.json",
        "results.csv",
        "per_site.csv",
        "meta.json",
    }

    # the pipeline was driven through its entry points with the site's inputs
    stages = [name for name, _ in fake.calls]
    assert stages == ["ingest", "cluster", "analyze", "continue_run"]
    ingest_kw = fake.calls[0][1]
    assert ingest_kw["src"] == tiny / "site"
    assert ingest_kw["manifest"] == tiny / "site" / "urls.tsv"
    assert ingest_kw["site"] == "tiny"
    opts = fake.calls[3][1]["opts"]
    assert opts.yes is True and opts.mode == "prompted"
    assert opts.settings["models"] == {
        "generation": "openai/gpt-test",
        "repair": "openai/gpt-test",
    }
    schema = fake.calls[3][1]["schema"]
    assert schema.names() == ["title", "price", "tags", "sizes", "specs", "headline", "author"]

    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "ok"
    assert results["method_label"] == "ditto (prompted)"
    (site,) = results["sites"]
    acc = site["accuracy"]
    assert acc["pages_excluded"] == 2  # pages_seen "0" and "4"
    assert acc["pages_scored"] == 6
    title = acc["fields"]["title"]
    assert (title["norm"]["tp"], title["norm"]["fp"], title["norm"]["fn"]) == (3, 0, 0)
    assert (title["raw"]["tp"], title["raw"]["fp"], title["raw"]["fn"]) == (2, 1, 1)
    price = acc["fields"]["price"]["norm"]
    assert (price["tp"], price["fp"], price["fn"], price["tn"]) == (1, 2, 1, 0)
    author = acc["fields"]["author"]["norm"]
    assert (author["tp"], author["fp"], author["fn"], author["tn"]) == (1, 0, 1, 1)
    assert acc["fields"]["sizes"]["norm"]["f1"] == 1.0
    assert acc["fields"]["specs"]["norm"]["f1"] == 1.0

    optional = site["optional_field_recall"]
    assert set(optional["fields"]) == {"author"}
    assert optional["norm"]["recall"] == pytest.approx(0.5)
    assert site["accuracy_sample"]["pages_scored"] <= 4

    clustering = site["clustering"]
    assert clustering["ari"] == pytest.approx(1.0)
    assert clustering["n_templates"] == 2 and clustering["n_true_templates"] == 2

    cost = site["cost"]
    assert cost["calls"] == 3 and cost["cached_calls"] == 1
    assert cost["input_tokens_per_1k_pages"] == pytest.approx(30_000 / 8 * 1000)
    assert cost["calls_per_template"] == pytest.approx(1.5)
    assert cost["usd_per_1k_pages"] == pytest.approx((0.03 + 0.012) * 1000 / 8)
    assert cost["llm_latency_s_mean"] == pytest.approx(2.0)
    assert site["spend_usd"] == pytest.approx(0.004)
    assert site["speed"]["extract"]["pages_per_sec_per_core"] == pytest.approx(400.0)

    summary = results["summary"]
    assert summary["accuracy"]["overall"]["n_sites"] == 1
    assert summary["accuracy"]["per_vertical"]["synthetic"]["norm"]["f1"] == pytest.approx(
        site["accuracy"]["norm"]["f1"]
    )
    assert summary["speed_pages_per_sec_per_core"]["cluster"] == pytest.approx(80.0)
    assert summary["clustering"]["ari"] == pytest.approx(1.0)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["models"] == {
        "pinned": {"generation": "openai/gpt-test", "repair": "openai/gpt-test"},
        "used": ["openai/gpt-test"],
    }
    assert meta["prompt_versions"] == ["repair.v1", "specgen.v1"]
    assert len(meta["git"]["commit"]) == 40
    manifest = datasets_dir / "synthetic" / "manifest.json"
    assert meta["dataset_manifests"] == {
        "synthetic": {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    }
    assert meta["machine"]["cpu_count"] >= 1 and meta["machine"]["memory_gb"] > 0
    assert meta["spend_usd"] == pytest.approx(0.004)
    assert meta["spend_by_site"] == {"tiny": pytest.approx(0.004)}
    assert meta["config"]["name"] == "tiny_eval"
    assert meta["status"] == "ok"
    assert meta["metric_version"] == METRIC_VERSION
    assert meta["match_columns"] == ["norm", "raw", "canon"]

    per_site = _read_csv(out / "per_site.csv")
    assert [r["site"] for r in per_site] == ["tiny"]
    assert float(per_site[0]["f1_norm"]) == pytest.approx(site["accuracy"]["norm"]["f1"])
    assert float(per_site[0]["f1_canon"]) == pytest.approx(site["accuracy"]["canon"]["f1"])
    assert float(per_site[0]["ari"]) == pytest.approx(1.0)
    assert per_site[0]["pps_core_extract"] == "400.0"
    rows = _read_csv(out / "results.csv")
    assert [r["scope"] for r in rows] == ["synthetic", "overall"]
    assert float(rows[-1]["f1_canon"]) == pytest.approx(
        summary["accuracy"]["overall"]["canon"]["f1"]
    )
    assert float(rows[-1]["usd_per_1k_pages"]) == pytest.approx(5.25)
    assert float(rows[-1]["cluster_ari"]) == pytest.approx(1.0)


def test_unprompted_mode_aligns_names_then_scores_recall(
    tiny: Path, tmp_path: Path, datasets_dir: Path
) -> None:
    rename = {"title": "name", "price": "cost", "headline": "heading", "author": "writer"}
    preds = {
        k: {rename.get(f, f): v for f, v in row.items()} for k, row in truth_predictions().items()
    }
    fake = FakePipeline(tiny, preds)
    cfg = _config(
        tmp_path,
        tiny,
        method={"kind": "ditto", "mode": "unprompted"},
        sample={"align_pages": 50},
    )
    out = _run(tmp_path, cfg, _hooks(fake), datasets_dir)
    results = json.loads((out / "results.json").read_text())
    (site,) = results["sites"]
    assert site["alignment"]["t_T01"]["title"] == "name"
    assert site["alignment"]["t_T01"]["price"] == "cost"
    assert site["alignment"]["t_T02"] == {"headline": "heading", "author": "writer"}
    assert fake.calls[3][1]["schema"] is None
    assert results["summary"]["unprompted_recall"]["norm"] == pytest.approx(1.0)


def test_clustering_only_run_needs_no_llm(tiny: Path, tmp_path: Path, datasets_dir: Path) -> None:
    fake = FakePipeline(tiny, {}, merge_templates=True)
    hooks = _hooks(fake)
    hooks.settings = _fail  # type: ignore[assignment]
    cfg = _config(
        tmp_path,
        tiny,
        method={"params": {"until": "cluster"}},
        models={},
        budget_usd=0.0,
        dataset_options={"site_dir": str(tiny), "name": "tiny", "truth_only": True},
    )
    out = _run(tmp_path, cfg, hooks, datasets_dir)
    results = json.loads((out / "results.json").read_text())
    (site,) = results["sites"]
    assert site["status"] == "ok"
    assert site["accuracy"] is None
    assert site["clustering"]["ari"] == pytest.approx(0.0)
    assert site["clustering"]["over_merge"] == 1
    assert [name for name, _ in fake.calls] == ["ingest", "cluster"]
    assert site["cost"]["calls"] == 0


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("must not be called")


def test_baseline_method_on_a_page_sample(tiny: Path, tmp_path: Path, datasets_dir: Path) -> None:
    bdir = tmp_path / "baselines"
    bdir.mkdir()
    (bdir / "echo.py").write_text(BASELINE)
    fake = FakePipeline(tiny, {})
    cfg = _config(
        tmp_path,
        tiny,
        method={"kind": "baseline", "baseline": "echo"},
        models={"baseline": "openai/gpt-test"},
    )
    out = _run(tmp_path, cfg, _hooks(fake), datasets_dir, baselines_dir=bdir)
    results = json.loads((out / "results.json").read_text())
    (site,) = results["sites"]
    assert site["status"] == "ok", site["reason"]
    assert results["method_label"] == "echo"
    assert site["accuracy"]["pages_scored"] == 4
    assert site["cost"]["pages"] == 4 and site["cost"]["extrapolated"] is True
    assert site["cost"]["calls_per_page"] == 1.0
    assert site["cost"]["usd_per_1k_pages"] == pytest.approx(0.5 + 0.08)
    assert site["spend_usd"] == 0.0
    assert any("extrapolated from 4 of 8 pages" in n for n in site["notes"])
    assert "site=tiny max_pages=4 fields=7" in site["notes"]
    assert set(site["speed"]) == {"baseline"}
    assert [name for name, _ in fake.calls] == ["ingest", "cluster"]


def test_baseline_not_run_is_recorded(tiny: Path, tmp_path: Path, datasets_dir: Path) -> None:
    bdir = tmp_path / "baselines"
    bdir.mkdir()
    (bdir / "echo.py").write_text(BASELINE)
    cfg = _config(
        tmp_path,
        tiny,
        method={"kind": "baseline", "baseline": "echo"},
        models={},
        dataset_options={"site_dir": str(tiny), "name": "refuse"},
    )
    out = _run(tmp_path, cfg, _hooks(FakePipeline(tiny, {})), datasets_dir, baselines_dir=bdir)
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "not_run"
    assert results["sites"][0]["reason"] == "no hand-written wrapper for refuse"
    assert results["summary"]["accuracy"] is None


def test_unavailable_dataset_is_recorded_as_not_run(tmp_path: Path, datasets_dir: Path) -> None:
    cfg = _config(tmp_path, tmp_path, dataset_options={"params": str(tmp_path / "missing.yaml")})
    out = _run(tmp_path, cfg, Hooks(), datasets_dir)
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "not_run"
    assert "missing.yaml" in results["reason"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["status"] == "not_run"


def test_dry_run_writes_plan_only(tiny: Path, tmp_path: Path, datasets_dir: Path) -> None:
    fake = FakePipeline(tiny, {})
    out = _run(tmp_path, _config(tmp_path, tiny), _hooks(fake), datasets_dir, dry_run=True)
    assert out.name.endswith("-dryrun")
    assert {p.name for p in out.iterdir()} == {"plan.json", "meta.json"}
    plan = json.loads((out / "plan.json").read_text())
    assert plan["sites"][0]["labeled_pages"] == 8
    assert plan["uses_llm"] is True
    assert fake.calls == []


def test_budget_cap_skips_remaining_sites(tiny: Path, tmp_path: Path, datasets_dir: Path) -> None:
    def two_sites(dataset: str, selection: Selection, work_dir: Path) -> list[EvalSite]:
        return [read_site(tiny, name=n, work_dir=work_dir) for n in ("a", "b")]

    fake = FakePipeline(tiny, truth_predictions())
    hooks = _hooks(fake, load_sites=two_sites)
    out = _run(tmp_path, _config(tmp_path, tiny, budget_usd=0.003), hooks, datasets_dir)
    results = json.loads((out / "results.json").read_text())
    assert [s["status"] for s in results["sites"]] == ["ok", "skipped"]
    assert "budget" in results["sites"][1]["reason"]
    assert results["summary"]["status_counts"] == {"ok": 1, "skipped": 1}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["spend_usd"] == pytest.approx(0.004)


def test_failed_site_is_recorded_with_traceback(
    tiny: Path, tmp_path: Path, datasets_dir: Path
) -> None:
    fake = FakePipeline(tiny, {})

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("verification blew up")

    hooks = _hooks(fake)
    hooks.continue_run = broken
    out = _run(tmp_path, _config(tmp_path, tiny), hooks, datasets_dir)
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "failed"
    site = results["sites"][0]
    assert site["reason"] == "RuntimeError: verification blew up"
    log = Path(site["notes"][0].removeprefix("traceback: "))
    assert "verification blew up" in log.read_text()
    assert results["summary"]["accuracy"] is None


def test_pinned_model_mismatch_fails_the_site(
    tiny: Path, tmp_path: Path, datasets_dir: Path
) -> None:
    fake = FakePipeline(tiny, {})
    hooks = _hooks(fake)
    hooks.resolve_model = lambda role, settings: "openai/other"
    out = _run(tmp_path, _config(tmp_path, tiny), hooks, datasets_dir)
    site = json.loads((out / "results.json").read_text())["sites"][0]
    assert site["status"] == "failed"
    assert "pinned openai/gpt-test" in site["reason"]
    assert fake.calls == []


def test_cost_summary_without_pages_or_usage() -> None:
    empty = cost_summary([], pages=0, templates=0, estimate=FakePipeline.estimate_cost)
    assert empty["calls"] == 0 and empty["usd_per_1k_pages"] is None
    assert empty["calls_per_template"] is None


def test_manifest_hashes_ignores_dirs_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "manifest.json").write_bytes(b"{}")
    (tmp_path / "b").mkdir()
    assert list(manifest_hashes(tmp_path)) == ["a"]
