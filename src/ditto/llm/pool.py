"""Bounded worker pool for LLM calls across templates (threads: the work is network-bound)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ditto.llm.base import BudgetExceeded

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class PoolResult(Generic[T, R]):
    """Outcome of :func:`run_concurrently`, every list in input order."""

    results: list[tuple[T, R]] = field(default_factory=list)
    errors: list[tuple[T, Exception]] = field(default_factory=list)  # fn raised (not budget)
    not_processed: list[T] = field(default_factory=list)  # never started, or hit the budget
    budget_error: BudgetExceeded | None = None

    @property
    def stopped_by_budget(self) -> bool:
        return self.budget_error is not None


def run_concurrently(
    fn: Callable[[T], R], items: Sequence[T], concurrency: int = 4
) -> PoolResult[T, R]:
    """Run ``fn`` over ``items`` with at most ``concurrency`` in flight.

    The first :class:`BudgetExceeded` stops scheduling: calls already running finish, the
    item that hit the budget and every unstarted item are listed in ``not_processed``. Other
    exceptions are captured per item in ``errors`` and do not stop the pool.
    """
    n = len(items)
    ok: dict[int, R] = {}
    failed: dict[int, Exception] = {}
    budget_error: BudgetExceeded | None = None
    next_index = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="ditto-llm") as ex:
        running: dict[Future[R], int] = {}

        def submit_more() -> None:
            nonlocal next_index
            while budget_error is None and next_index < n and len(running) < max(1, concurrency):
                running[ex.submit(fn, items[next_index])] = next_index
                next_index += 1

        submit_more()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                index = running.pop(fut)
                exc = fut.exception()
                if exc is None:
                    ok[index] = fut.result()
                elif isinstance(exc, BudgetExceeded):
                    budget_error = budget_error or exc
                elif isinstance(exc, Exception):
                    failed[index] = exc
                else:  # KeyboardInterrupt / SystemExit from a worker
                    raise exc
            submit_more()

    result: PoolResult[T, R] = PoolResult(budget_error=budget_error)
    for index, item in enumerate(items):
        if index in ok:
            result.results.append((item, ok[index]))
        elif index in failed:
            result.errors.append((item, failed[index]))
        else:
            result.not_processed.append(item)
    return result
