"""USD prices per million tokens (``config/prices.yaml``) and cost estimation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_REPO_PRICES = Path(__file__).resolve().parents[3] / "config" / "prices.yaml"
# Dated snapshot suffixes: gpt-4o-2024-08-06, gpt-4-0613, claude-3-5-sonnet-20241022.
_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")


@dataclass(frozen=True)
class Price:
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens
    cached_input: float | None = None  # USD per 1M cached input tokens (default: input)
    estimated: bool = False


@dataclass(frozen=True)
class PriceTable:
    models: dict[str, Price]
    default: Price
    retrieved: str | None = None
    tiers: dict[str, float] | None = None  # service tier -> price multiplier (flex: 0.5)

    def tier_multiplier(self, tier: str | None) -> float:
        """Billing multiplier of an OpenAI service tier; 1.0 for default/unknown tiers."""
        if not tier or not self.tiers:
            return 1.0
        return float(self.tiers.get(tier, 1.0))

    def lookup(self, model: str) -> Price:
        """Price of ``"provider/model"`` (a bare id means ``openai/``).

        Falls back to the undated alias of a dated snapshot, then to the ``default`` entry
        (deliberately expensive and marked estimated, so the budget guard errs safe).
        """
        ref = model if "/" in model else f"openai/{model}"
        if ref in self.models:
            return self.models[ref]
        undated = _SNAPSHOT.sub("", ref)
        if undated in self.models:
            return self.models[undated]
        return self.default


def prices_path() -> Path:
    """``ditto/_data/prices.yaml`` in an installed wheel, else the repo's ``config/prices.yaml``."""
    packaged = resources.files("ditto").joinpath("_data").joinpath("prices.yaml")
    if packaged.is_file():
        return Path(str(packaged))
    return _REPO_PRICES


def load_prices(path: Path | None = None) -> PriceTable:
    return _load_prices(path or prices_path())


@lru_cache(maxsize=8)
def _load_prices(path: Path) -> PriceTable:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = {str(k): _price(v) for k, v in (raw.get("models") or {}).items()}
    default = _price(raw.get("default") or {"input": 10.0, "output": 50.0, "estimated": True})
    retrieved = raw.get("retrieved")
    tiers = {str(k): float(v) for k, v in (raw.get("tiers") or {}).items()}
    return PriceTable(
        models=models,
        default=default,
        retrieved=str(retrieved or "") or None,
        tiers=tiers or None,
    )


def _price(entry: dict[str, Any]) -> Price:
    cached = entry.get("cached_input")
    return Price(
        input=float(entry["input"]),
        output=float(entry["output"]),
        cached_input=float(cached) if cached is not None else None,
        estimated=bool(entry.get("estimated", False)),
    )


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    prices: PriceTable | None = None,
) -> tuple[float, bool]:
    """``(usd, estimated)`` for one call of ``model`` (``"provider/model"``).

    ``input_tokens`` includes ``cached_input_tokens``, which are billed at the cached rate.
    ``estimated`` is true when the price entry is marked estimated or the model is unknown.
    """
    table = prices or load_prices()
    price = table.lookup(model)
    cached = min(max(cached_input_tokens, 0), input_tokens)
    cached_rate = price.cached_input if price.cached_input is not None else price.input
    usd = (
        (input_tokens - cached) * price.input + cached * cached_rate + output_tokens * price.output
    ) / 1_000_000
    return usd, price.estimated or price is table.default
