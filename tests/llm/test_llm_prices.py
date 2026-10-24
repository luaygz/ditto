from __future__ import annotations

from pathlib import Path

import pytest
from llm_fakes import PRICES

from ditto.llm.prices import estimate_cost, load_prices, prices_path


def test_cost_math() -> None:
    usd, estimated = estimate_cost("openai/gpt-test", 1000, 500, prices=PRICES)
    assert usd == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)
    assert estimated is False


def test_cached_input_tokens_use_cached_rate() -> None:
    usd, _ = estimate_cost("openai/gpt-test", 1000, 500, cached_input_tokens=400, prices=PRICES)
    assert usd == pytest.approx((600 * 2.0 + 400 * 0.5 + 500 * 10.0) / 1e6)
    # no cached_input price: cached tokens are billed at the input rate
    usd, _ = estimate_cost("openai/gpt-test-mini", 1000, 0, cached_input_tokens=400, prices=PRICES)
    assert usd == pytest.approx(1000 * 0.5 / 1e6)


def test_estimated_flags_and_fallbacks() -> None:
    assert estimate_cost("openai/gpt-guess", 10, 10, prices=PRICES)[1] is True
    usd, estimated = estimate_cost("openai/unknown-model", 1_000_000, 0, prices=PRICES)
    assert (usd, estimated) == (10.0, True)
    # bare ids mean openai/, dated snapshots fall back to their alias
    assert estimate_cost("gpt-test", 1_000_000, 0, prices=PRICES) == (2.0, False)
    assert estimate_cost("openai/gpt-test-2026-01-31", 1_000_000, 0, prices=PRICES) == (2.0, False)


def test_repo_price_table() -> None:
    path = prices_path()
    assert path.name == "prices.yaml" and path.is_file()
    table = load_prices()
    assert "openai/gpt-6.1-sol" in table.models
    assert "openai/gpt-5.4-mini" in table.models
    assert table.default.estimated is True
    usd, estimated = estimate_cost("openai/gpt-6.1-sol", 1_000_000, 1_000_000)
    price = table.models["openai/gpt-6.1-sol"]
    assert usd == pytest.approx(price.input + price.output)
    assert estimated is price.estimated


def test_load_prices_from_file(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text(
        "models:\n  openai/x:\n    input: 1\n    output: 2\n    estimated: true\n"
        "default:\n  input: 3\n  output: 4\n  estimated: true\n"
    )
    table = load_prices(f)
    assert table.models["openai/x"].estimated is True
    assert estimate_cost("openai/x", 1_000_000, 1_000_000, prices=table) == (3.0, True)
