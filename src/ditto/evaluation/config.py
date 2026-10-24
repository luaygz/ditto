"""Evaluation config files (``eval/configs/<name>.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ditto.evaluation.datasets.base import Selection, Split
from ditto.llm.base import Role

DatasetName = Literal["synthetic", "swde", "cc_silver", "livewebie"]
DATASET_MODULES: dict[str, str] = {
    "synthetic": "ditto.evaluation.datasets.synthetic",
    "swde": "ditto.evaluation.datasets.swde",
    "cc_silver": "ditto.evaluation.datasets.cc_silver",
    "livewebie": "ditto.evaluation.datasets.livewebie",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionConfig(_Strict):
    split: Split = "all"
    sites: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)
    max_sites: int | None = None


class DittoParams(_Strict):
    """Pipeline parameters (defaults are the product defaults)."""

    until: Literal["cluster", "extract"] = "extract"
    """``cluster`` stops after clustering (clustering-only evaluations, no LLM)."""
    min_cluster_size: int = 20
    union_min_tokens: int = 8000
    union_max_tokens: int = 30000
    max_repair_calls: int = 3
    tail: Literal["skip", "direct", "spec"] = "skip"
    concurrency: int = 4


class MethodConfig(_Strict):
    kind: Literal["ditto", "baseline"] = "ditto"
    mode: Literal["prompted", "unprompted"] = "prompted"
    baseline: str | None = None
    """Baseline module name: ``eval/baselines/<baseline>.py``."""
    params: DittoParams = Field(default_factory=DittoParams)

    @model_validator(mode="after")
    def _baseline_named(self) -> MethodConfig:
        if self.kind == "baseline" and not self.baseline:
            raise ValueError("method.kind 'baseline' needs method.baseline: <module name>")
        if self.kind == "ditto" and self.baseline:
            raise ValueError("method.baseline is only valid with method.kind 'baseline'")
        return self

    @property
    def label(self) -> str:
        if self.kind == "baseline":
            return str(self.baseline)
        if self.params.until == "cluster":
            return "ditto (clustering)"
        return f"ditto ({self.mode})"


class SampleConfig(_Strict):
    pages_per_site: int | None = None
    """Page sample per site (seeded): what a baseline runs on, and the extra
    ``accuracy_sample`` scored for ditto so both compare on identical pages."""
    align_pages: int = 50
    """Unprompted mode: pages per template used to align LLM field names to label names."""
    align_min_fraction: float = 0.1


class SeedsConfig(_Strict):
    run: int = 1
    """Page samples and baseline sampling."""
    synth: int = 1
    """Synthetic generator seed (frozen evaluation seed: 1)."""


class EvalConfig(_Strict):
    name: str
    description: str = ""
    dataset: DatasetName
    dataset_options: dict[str, Any] = Field(default_factory=dict)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    method: MethodConfig = Field(default_factory=MethodConfig)
    models: dict[Role, str] = Field(default_factory=dict)
    """Pinned ``provider/model`` per LLM role."""
    budget_usd: float = 0.0
    """Spend cap for this evaluation (real, uncached spend)."""
    sample: SampleConfig = Field(default_factory=SampleConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    workers: int | None = None
    keep_runs: bool = True
    """Keep each site's run directory under the work root (``false`` deletes it after scoring,
    for sweeps over many large synthetic sites)."""

    @classmethod
    def load(cls, path: str | Path) -> EvalConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: an eval config is a YAML mapping")
        return cls.model_validate(data)

    @property
    def uses_llm(self) -> bool:
        if self.method.kind == "baseline":
            return True
        return self.method.params.until == "extract"

    def selection_obj(self) -> Selection:
        options = dict(self.dataset_options)
        options.setdefault("seed", self.seeds.synth)
        return Selection(
            split=self.selection.split,
            sites=tuple(self.selection.sites),
            verticals=tuple(self.selection.verticals),
            max_sites=self.selection.max_sites,
            options=options,
        )
