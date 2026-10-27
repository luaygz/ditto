"""Chunked multiprocessing used by every page-level stage."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar

import psutil

T = TypeVar("T")
R = TypeVar("R")


def cpu_count() -> int:
    return os.cpu_count() or 1


def memory_gb() -> float:
    return psutil.virtual_memory().total / 1024**3


def default_workers(n_items: int | None = None) -> int:
    """Worker processes for page-level work: all cores, capped by item count; env override
    ``DITTO_WORKERS``."""
    env = os.environ.get("DITTO_WORKERS")
    workers = int(env) if env else cpu_count()
    if n_items is not None:
        workers = min(workers, max(1, n_items))
    return max(1, workers)


def chunked(items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def start_method() -> str:
    """Multiprocessing start method: fork on Linux (fastest), spawn elsewhere (the macOS
    default). ``DITTO_MP_START`` (fork|spawn|forkserver) overrides it, e.g.
    ``DITTO_MP_START=spawn`` emulates macOS on Linux."""
    method = os.environ.get("DITTO_MP_START", "").strip().lower()
    if not method:
        return "fork" if sys.platform.startswith("linux") else "spawn"
    available = mp.get_all_start_methods()
    if method not in available:
        raise ValueError(
            f"DITTO_MP_START={method!r} is not available (choose from {', '.join(available)})"
        )
    return method


def _context() -> mp.context.BaseContext:
    return mp.get_context(start_method())


def pmap(
    fn: Callable[[Sequence[T]], R],
    items: Sequence[T],
    *,
    workers: int | None = None,
    chunk_size: int = 256,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[object, ...] = (),
) -> Iterator[R]:
    """Apply ``fn`` to consecutive chunks of ``items`` in worker processes, yielding results
    in chunk order. ``fn`` must be a picklable top-level function taking one chunk.
    Runs in-process when one worker is enough (keeps tests and small runs fast)."""
    chunks = chunked(items, max(1, chunk_size))
    n = default_workers(len(chunks)) if workers is None else max(1, min(workers, len(chunks) or 1))
    if n <= 1 or len(chunks) <= 1:
        if initializer is not None:
            initializer(*initargs)
        for chunk in chunks:
            yield fn(chunk)
        return
    with _context().Pool(n, initializer=initializer, initargs=initargs) as pool:
        yield from pool.imap(fn, chunks)


def pmap_flat(
    fn: Callable[[Sequence[T]], Iterable[R]],
    items: Sequence[T],
    **kwargs: object,
) -> Iterator[R]:
    for out in pmap(fn, items, **kwargs):  # type: ignore[arg-type]
        yield from out
