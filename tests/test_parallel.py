from __future__ import annotations

import sys

import pytest

from ditto import parallel
from ditto.parallel import chunked, default_workers, pmap, pmap_flat, start_method


def test_default_start_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DITTO_MP_START", raising=False)
    assert start_method() == ("fork" if sys.platform.startswith("linux") else "spawn")


def test_start_method_override_runs_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DITTO_MP_START", " Spawn ")
    assert start_method() == "spawn"
    # builtins pickle by name, so they work under spawn without importing this module
    assert list(pmap(sum, list(range(8)), workers=2, chunk_size=2)) == [1, 5, 9, 13]


def test_start_method_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DITTO_MP_START", "threads")
    with pytest.raises(ValueError, match="DITTO_MP_START"):
        start_method()


def test_default_workers_env_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DITTO_WORKERS", "3")
    assert default_workers() == 3
    assert default_workers(2) == 2
    assert default_workers(0) == 1
    monkeypatch.delenv("DITTO_WORKERS")
    assert default_workers() == parallel.cpu_count()


def test_chunked_and_in_process_pmap() -> None:
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    seen: list[int] = []
    out = list(pmap(sum, [1, 2, 3], workers=1, initializer=seen.append, initargs=(7,)))
    assert out == [6] and seen == [7]
    assert list(pmap_flat(sorted, [3, 1, 2, 0], workers=2, chunk_size=2)) == [1, 3, 0, 2]
