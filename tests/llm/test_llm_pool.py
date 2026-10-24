from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from llm_fakes import PRICES, FakeAdapter, make_settings

from ditto.llm.base import BudgetExceeded, Usage
from ditto.llm.client import CachedLLM
from ditto.llm.pool import run_concurrently


def test_bounded_and_ordered() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def work(i: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01 * (5 - i % 5))
        with lock:
            active -= 1
        return i * i

    result = run_concurrently(work, list(range(12)), concurrency=3)
    assert result.results == [(i, i * i) for i in range(12)]
    assert result.errors == [] and result.not_processed == []
    assert not result.stopped_by_budget
    assert 1 < peak <= 3


def test_budget_stop_lists_unprocessed_templates() -> None:
    def work(tid: str) -> str:
        if tid == "t2":
            raise BudgetExceeded("out of money")
        return f"spec-{tid}"

    result = run_concurrently(work, ["t0", "t1", "t2", "t3", "t4"], concurrency=1)
    assert result.results == [("t0", "spec-t0"), ("t1", "spec-t1")]
    assert result.not_processed == ["t2", "t3", "t4"]
    assert isinstance(result.budget_error, BudgetExceeded)
    assert result.stopped_by_budget


def test_other_errors_are_captured_per_item() -> None:
    def work(i: int) -> int:
        if i % 2:
            raise RuntimeError(f"bad {i}")
        return i

    result = run_concurrently(work, [0, 1, 2, 3], concurrency=4)
    assert result.results == [(0, 0), (2, 2)]
    assert [(i, str(e)) for i, e in result.errors] == [(1, "bad 1"), (3, "bad 3")]
    assert result.not_processed == []


def test_empty_and_oversized_pool() -> None:
    assert run_concurrently(lambda x: x, [], concurrency=8).results == []
    assert run_concurrently(lambda x: x + 1, [1], concurrency=0).results == [(1, 2)]


def test_per_run_budget_through_the_pool(tmp_path: Path) -> None:
    from ditto.run import RunDir

    run = RunDir(tmp_path / "run")
    settings = make_settings(tmp_path, run_budget_usd=0.005)
    llm = CachedLLM(
        model="openai/gpt-test",
        role="generation",
        adapter=FakeAdapter(usage=Usage(10, 100)),
        settings=settings,
        run=run,
        prices=PRICES,
    )
    templates = [f"t_{i}" for i in range(8)]

    def generate(tid: str) -> str:
        return llm.complete(
            [{"role": "user", "content": tid}], prompt_version="p", template_id=tid, max_tokens=100
        ).text

    result = run_concurrently(generate, templates, concurrency=1)
    assert [tid for tid, _ in result.results] == templates[:4]
    assert result.not_processed == templates[4:]
    assert result.budget_error is not None and result.budget_error.scope == "run"


def test_keyboard_interrupt_propagates() -> None:
    def work(_i: int) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_concurrently(work, [1], concurrency=1)
