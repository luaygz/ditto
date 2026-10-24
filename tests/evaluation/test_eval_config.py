from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ditto.evaluation.config import EvalConfig
from ditto.synth.params import load_params

CONFIGS = Path(__file__).resolve().parents[2] / "eval" / "configs"
EXPECTED = {
    "synth_default",
    "synth_scale",
    "synth_curves",
    "synth_nightly",
    "swde_dev_ditto",
    "swde_dev_naive",
    "swde_test_ditto",
    "swde_test_naive",
    "swde_test_crawl4ai",
    "swde_test_autoscraper",
    "swde_test_handwritten",
    "swde_dev_unprompted",
    "cc_silver_ditto",
    "cc_silver_naive",
    "cc_silver_crawl4ai",
    "livewebie_ditto",
}


def test_every_config_validates_and_is_named_after_its_file() -> None:
    names = set()
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = EvalConfig.load(path)
        assert cfg.name == path.stem
        names.add(cfg.name)
        if cfg.uses_llm and cfg.method.kind == "ditto":
            assert cfg.models["generation"] == cfg.models["repair"]
            assert cfg.budget_usd > 0
        if cfg.method.kind == "baseline" and cfg.method.baseline != "handwritten":
            assert set(cfg.models) == {"baseline"}
        if cfg.dataset == "synthetic":
            assert cfg.seeds.synth == 1  # frozen evaluation seed
            assert not cfg.uses_llm
    assert names == EXPECTED


def test_naive_and_ditto_share_the_page_sample() -> None:
    for dataset in ("swde_dev", "swde_test", "cc_silver"):
        ditto = EvalConfig.load(CONFIGS / f"{dataset}_ditto.yaml")
        naive = EvalConfig.load(CONFIGS / f"{dataset}_naive.yaml")
        n = ditto.sample.pages_per_site
        assert n is not None and n == naive.sample.pages_per_site and 0 < n <= 200
        assert ditto.seeds.run == naive.seeds.run
        assert ditto.models["generation"] == naive.models["baseline"]


def test_synthetic_sweep_points_are_valid_generator_params() -> None:
    cfg = EvalConfig.load(CONFIGS / "synth_curves.yaml")
    opts = cfg.dataset_options
    base = {"total_pages": opts["pages"]}
    for param, values in opts["sweep"].items():
        extra = opts.get("sweep_overrides", {}).get(param, {})
        for value in values:
            load_params(
                Path(opts["params"]) if Path(opts["params"]).exists() else None,
                {**base, **extra, param: value},
            )


def test_invalid_configs_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ndataset: swde\nmethod: {kind: baseline}\n")
    with pytest.raises(ValidationError, match=r"needs method\.baseline"):
        EvalConfig.load(bad)
    bad.write_text("name: x\ndataset: swde\nmethod: {kind: ditto, baseline: naive_llm}\n")
    with pytest.raises(ValidationError, match="only valid"):
        EvalConfig.load(bad)
    bad.write_text("name: x\ndataset: imdb\n")
    with pytest.raises(ValidationError):
        EvalConfig.load(bad)
    bad.write_text("name: x\ndataset: swde\nbudget: 3\n")
    with pytest.raises(ValidationError):
        EvalConfig.load(bad)
    bad.write_text("- a\n")
    with pytest.raises(ValueError, match="mapping"):
        EvalConfig.load(bad)


def test_selection_passes_dataset_options_and_synth_seed() -> None:
    cfg = EvalConfig.model_validate(
        {
            "name": "x",
            "dataset": "synthetic",
            "dataset_options": {"pages": 10},
            "selection": {"split": "dev", "sites": ["a"], "max_sites": 2},
        }
    )
    sel = cfg.selection_obj()
    assert sel.split == "dev" and sel.sites == ("a",) and sel.max_sites == 2
    assert sel.options == {"pages": 10, "seed": 1}
    assert cfg.method.label == "ditto (prompted)"
