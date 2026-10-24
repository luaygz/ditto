"""What the input scanners hand back to :func:`ditto.ingest.ingest` for merging."""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

MAX_ERROR_SAMPLES = 20


class Capture(NamedTuple):
    """One HTML page found in the input, before per-URL and duplicate filtering."""

    url: str
    source: str  # blob path relative to the run root, or absolute WARC path
    sha256: str  # of the payload bytes
    fetched_at: str | None
    offset: int | None  # WARC record offset; None for whole-file blobs
    length: int  # WARC record length, or blob size
    encoding: str
    rank: tuple[float, str, int]  # larger means a later capture of the same URL


class ScanResult(NamedTuple):
    captures: list[Capture]
    counts: Counter[str]
    errors: list[str]


def merge_scans(results: list[ScanResult]) -> ScanResult:
    captures: list[Capture] = []
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for result in results:
        captures.extend(result.captures)
        counts.update(result.counts)
        errors.extend(result.errors[: MAX_ERROR_SAMPLES - len(errors)])
    return ScanResult(captures, counts, errors)
