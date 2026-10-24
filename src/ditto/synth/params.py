"""Parameters of the synthetic site generator (``eval/synth/*.yaml``).

Every knob of the brief's stress table is a field here; the defaults are the brief's defaults.
``templates`` counts the regular templates; ``tail_templates`` adds small templates (below the
default ``--min-cluster-size`` of 20 pages) so every site has a long tail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MarkupStyle = Literal["server", "react", "table"]
KVStyle = Literal["dl", "table", "div"]
AbsentRendering = Literal["omitted", "empty", "na", "hidden"]
NoiseKind = Literal["entities", "nbsp", "whitespace", "charset"]

MARKUP_STYLES: tuple[MarkupStyle, ...] = ("server", "react", "table")
KV_STYLES: tuple[KVStyle, ...] = ("dl", "table", "div")
ABSENT_RENDERINGS: tuple[AbsentRendering, ...] = ("omitted", "empty", "na", "hidden")


class SynthParams(BaseModel):
    """One synthetic site configuration. Ranges are inclusive ``[low, high]`` pairs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- size -------------------------------------------------------------------------------
    templates: int = Field(12, ge=1, le=1000, description="regular templates per site")
    pages_median: float = Field(2000, gt=0, description="median of the log-normal page counts")
    pages_sigma: float = Field(1.0, ge=0, le=5, description="sigma of the log-normal page counts")
    pages_min: int = Field(20, ge=1, description="smallest regular template")
    pages_max: int = Field(60000, ge=1, description="largest regular template")
    total_pages: int | None = Field(
        None, ge=1, description="rescale template sizes so the site has exactly this many pages"
    )
    tail_templates: int = Field(3, ge=0, le=1000, description="extra small (long-tail) templates")
    tail_pages: tuple[int, int] = Field((1, 19), description="pages per tail template")
    # --- fields -----------------------------------------------------------------------------
    mandatory_fields: tuple[int, int] = Field((5, 8), description="mandatory scalars per template")
    optional_fields: int = Field(3, ge=0, le=10, description="optional scalars per template")
    optional_presence: tuple[float, float] = Field(
        (0.02, 0.95), description="presence rate range of optional fields and columns"
    )
    list_fields: int = Field(1, ge=0, le=3, description="list fields per template")
    list_items: tuple[int, int] = Field((1, 8), description="items per list")
    record_tables: int = Field(1, ge=0, le=3, description="record tables per template")
    table_rows: tuple[int, int] = Field((0, 12), description="rows per record table")
    optional_columns: int = Field(1, ge=0, le=2, description="optional columns per record table")
    kv_blocks: int = Field(1, ge=0, le=3, description="key-value blocks per template")
    kv_style: KVStyle | Literal["mixed"] = "mixed"
    kv_keys: tuple[int, int] = Field((3, 8), description="keys shown per key-value block")
    # --- markup -----------------------------------------------------------------------------
    hashed_class_share: float = Field(
        0.3,
        ge=0,
        le=1,
        description="share of class-bearing elements with only hashed/utility classes",
    )
    boilerplate_share: float = Field(
        0.4, ge=0.05, le=0.95, description="share of body elements that are nav/header/footer"
    )
    boilerplate_variants: int = Field(3, ge=1, le=6, description="nav/footer variants")
    near_duplicate_pairs: int = Field(1, ge=0, le=10)
    near_duplicate_jaccard: float = Field(0.7, gt=0, lt=1)
    absent_rendering: AbsentRendering | Literal["mixed"] = "omitted"
    markup_style: MarkupStyle | Literal["mixed"] = "server"
    noise: tuple[NoiseKind, ...] = ()
    noise_rate: float = Field(0.5, ge=0, le=1, description="share of pages/values given noise")
    # --- naming -----------------------------------------------------------------------------
    site_name: str | None = None
    host: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SynthParams:
        for name in (
            "tail_pages",
            "mandatory_fields",
            "optional_presence",
            "list_items",
            "table_rows",
            "kv_keys",
        ):
            lo, hi = getattr(self, name)
            if lo > hi or lo < 0:
                raise ValueError(f"{name}: need 0 <= low <= high, got [{lo}, {hi}]")
        if self.tail_pages[0] < 1:
            raise ValueError("tail_pages: a tail template has at least one page")
        if self.mandatory_fields[0] < 1:
            raise ValueError("mandatory_fields: at least one mandatory scalar besides the title")
        if self.list_items[0] < 1:
            raise ValueError("list_items: a list has at least one item")
        if self.kv_keys[0] < 1:
            raise ValueError("kv_keys: a key-value block shows at least one key")
        lo_p, hi_p = self.optional_presence
        if not (0 < lo_p <= hi_p <= 1):
            raise ValueError("optional_presence must lie in (0, 1]")
        if self.pages_min > self.pages_max:
            raise ValueError("pages_min must not exceed pages_max")
        if 2 * self.near_duplicate_pairs > self.templates:
            raise ValueError("near_duplicate_pairs needs two templates per pair")
        if len(set(self.noise)) != len(self.noise):
            raise ValueError("noise kinds must be unique")
        if self.total_pages is not None:
            floor = self.templates + self.tail_templates * self.tail_pages[0]
            if self.total_pages < floor:
                raise ValueError(f"total_pages={self.total_pages} below the minimum {floor}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> SynthParams:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: synth params must be a YAML mapping")
        return cls.model_validate(data)


def load_params(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> SynthParams:
    """Load params from YAML (defaults when ``path`` is None) and apply ``overrides``."""
    base = SynthParams.load(path) if path is not None else SynthParams()
    if not overrides:
        return base
    merged = base.model_dump()
    merged.update(overrides)
    return SynthParams.model_validate(merged)
