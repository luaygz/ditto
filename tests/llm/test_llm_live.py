"""Live API tests (``pytest -m live``): one tiny JSON call per model, well under $0.05 total.

They use the real settings (``.env``, default cache dir) so their spend lands in the global
ledger, but bypass the response cache so the call really reaches the provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ditto.llm import (
    complete_json,
    get_llm,
    load_settings,
    resolve_model,
    summarize_usage,
)
from ditto.llm.base import Role
from ditto.run import RunDir

PROMPT = [
    {"role": "user", "content": 'What is 2 + 2? Reply as JSON: {"answer": <integer>}'},
]


def _answer(data: Any) -> int:
    if not isinstance(data, dict) or not isinstance(data.get("answer"), int):
        raise ValueError('expected {"answer": <integer>}')
    return int(data["answer"])


def _one_call(role: Role, tmp_path: Path, settings: Any = None) -> dict[str, Any]:
    settings = settings or load_settings()
    run = RunDir(tmp_path / "run")
    llm = get_llm(role, settings=settings, run=run, no_cache=True)
    value = complete_json(
        llm,
        PROMPT,  # type: ignore[arg-type]
        validate=_answer,
        prompt_version="live_test.v1",
        max_tokens=1024,
    )
    assert value == 4
    summary = summarize_usage(run)
    totals = summary["totals"]
    assert totals["calls"] >= 1 and totals["cached_calls"] == 0
    assert totals["input_tokens"] > 0 and totals["output_tokens"] > 0
    assert totals["cost_usd"] > 0
    print(f"\n{role}: {llm.model} {totals}")
    return summary


@pytest.mark.live
def test_main_model_json_call(tmp_path: Path) -> None:
    summary = _one_call("generation", tmp_path)
    model = next(iter(summary["by_model"]))
    assert model == resolve_model("generation", load_settings())
    assert summary["totals"]["cost_usd"] < 0.02


@pytest.mark.live
def test_small_model_json_call(tmp_path: Path) -> None:
    summary = _one_call("probe", tmp_path)
    assert summary["totals"]["cost_usd"] < 0.01


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_anthropic_adapter_json_call(tmp_path: Path) -> None:
    model = os.environ.get("DITTO_ANTHROPIC_TEST_MODEL", "claude-haiku-4-5")
    settings = load_settings().with_models(baseline=f"anthropic/{model}")
    summary = _one_call("baseline", tmp_path, settings)
    assert next(iter(summary["by_model"])) == f"anthropic/{model}"
