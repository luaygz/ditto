"""Stage 1: read an HTML directory or WARC files into a run directory.

Output: ``pages.sqlite`` (dense ``page_id`` 0..N-1 in URL order), content-addressed blobs for
directory input (WARC input is read in place by record offset), ``run.json`` metadata and the
``ingest`` timing. Re-ingesting into the same run directory replaces all of it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, TypeVar
from urllib.parse import urlsplit

from ditto import __version__
from ditto.ingest.capture import Capture, ScanResult, merge_scans
from ditto.ingest.directory import (
    HTML_SUFFIXES,
    MANIFEST_NAME,
    list_html_files,
    normalize_relpath,
    read_manifest,
    scan_files,
)
from ditto.ingest.store import clear_cache, write_pages_db
from ditto.ingest.warc import find_warc_files, is_warc_path, scan_warc_files
from ditto.parallel import default_workers, pmap
from ditto.run import RunDir

FILES_PER_CHUNK = 256

InputKind = Literal["directory", "warc"]
T = TypeVar("T")

_VOLUME_STATS = ("files", "warc_files", "warc_records", "candidates")


@dataclass(frozen=True)
class IngestResult:
    """Outcome of :func:`ingest`; ``stats`` holds input volume and every skip/drop count."""

    run: RunDir
    kind: InputKind
    site: str
    n_pages: int
    stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    workers: int = 1

    def __str__(self) -> str:
        dropped = ", ".join(
            f"{key} {value}"
            for key, value in self.stats.items()
            if key not in _VOLUME_STATS and value
        )
        return (
            f"{self.n_pages} pages ({self.kind} input, site {self.site}) into {self.run.root}; "
            f"{dropped or 'nothing skipped'}; {self.seconds:.1f}s on {self.workers} workers"
        )


@dataclass(frozen=True)
class _Inputs:
    kind: InputKind
    paths: list[Path]  # the HTML directory, or every WARC file
    fallback_site: str


def ingest(
    src: Path | Sequence[Path],
    out: Path,
    *,
    manifest: Path | None = None,
    site: str | None = None,
    workers: int | None = None,
) -> IngestResult:
    """Ingest one HTML directory (optional ``urls.tsv``) or WARC files/directories into ``out``.

    Keeps the latest capture per URL, then drops byte-identical duplicates (the copy with the
    smallest URL survives). Pages are numbered in URL order.
    """
    start = time.perf_counter()
    sources = [Path(src)] if isinstance(src, str | os.PathLike) else [Path(p) for p in src]
    inputs = _classify(sources)
    if manifest is not None and inputs.kind == "warc":
        raise ValueError("a urls.tsv manifest only applies to directory input, not WARC files")
    run = RunDir(Path(out)).ensure()
    clear_cache()
    if inputs.kind == "warc":
        shutil.rmtree(run.blobs_dir, ignore_errors=True)

    manifest_info: dict[str, object] = {}
    if inputs.kind == "directory":
        scan, n_workers, manifest_info = _scan_directory(inputs.paths[0], run, manifest, workers)
    else:
        scan, n_workers = _scan_warcs(inputs.paths, workers)

    latest, superseded = _latest_per_url(scan.captures)
    pages, duplicates = _drop_duplicates(latest)
    if inputs.kind == "directory":
        _prune_blobs(run.blobs_dir, {c.sha256 for c in pages})
    write_pages_db(run.pages_db, pages)

    counts = scan.counts
    stats: dict[str, int] = (
        {"files": counts["files"]}
        if inputs.kind == "directory"
        else {"warc_files": len(inputs.paths), "warc_records": counts["records"]}
    )
    stats.update(
        candidates=len(scan.captures),
        skipped_non_html=counts["skipped_non_html"],
        skipped_empty=counts["skipped_empty"],
        skipped_status=counts["skipped_status"],
        skipped_revisit=counts["skipped_revisit"],
        skipped_no_url=counts["skipped_no_url"],
        superseded_captures=superseded,
        duplicates=duplicates,
        errors=counts["errors"],
    )
    site_name = site or _guess_site([c.url for c in pages], inputs.fallback_site)
    seconds = time.perf_counter() - start
    run.write_meta(
        site=site_name,
        source=[str(p.resolve()) for p in sources],
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ditto_version=__version__,
        n_pages=len(pages),
        ingest={
            "kind": inputs.kind,
            **manifest_info,
            **stats,
            "content_digest": _content_digest(pages),
            "error_samples": scan.errors,
            "seconds": round(seconds, 3),
            "workers": n_workers,
        },
    )
    run.record_timing("ingest", seconds, len(pages), n_workers)
    return IngestResult(
        run=run,
        kind=inputs.kind,
        site=site_name,
        n_pages=len(pages),
        stats=stats,
        errors=scan.errors,
        seconds=seconds,
        workers=n_workers,
    )


def _classify(sources: list[Path]) -> _Inputs:
    """One HTML directory, or any mix of WARC files and directories holding only WARC files."""
    if not sources:
        raise ValueError("no input paths given")
    html_dirs: list[Path] = []
    warcs: list[Path] = []
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        if path.is_dir():
            found = [] if _has_html_file(path) else find_warc_files(path)
            if found:
                warcs.extend(found)
            else:
                html_dirs.append(path)
        elif is_warc_path(path):
            warcs.append(path)
        else:
            raise ValueError(f"unsupported input file {path}: expected a directory or .warc[.gz]")
    if html_dirs and (warcs or len(html_dirs) > 1):
        raise ValueError("give either one HTML directory or WARC files/directories, not a mix")
    if html_dirs:
        root = html_dirs[0].resolve()
        return _Inputs("directory", [root], root.name)
    files = sorted({p.resolve() for p in warcs})
    first = sources[0].resolve()
    return _Inputs("warc", files, first.name if first.is_dir() else first.parent.name)


def _has_html_file(directory: Path) -> bool:
    for _dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if any(os.path.splitext(name)[1].lower() in HTML_SUFFIXES for name in filenames):
            return True
    return False


def _run_scans(
    fn: Callable[[Sequence[T]], ScanResult],
    items: Sequence[T],
    chunk_size: int,
    workers: int | None,
) -> tuple[ScanResult, int]:
    """Scan ``items`` in chunks with pmap; returns the merged result and the worker count."""
    n_chunks = -(-len(items) // chunk_size)
    wanted = default_workers() if workers is None else max(1, workers)
    n_workers = max(1, min(wanted, n_chunks))
    results = list(pmap(fn, items, workers=n_workers, chunk_size=chunk_size))
    return merge_scans(results), n_workers


def _scan_warcs(files: list[Path], workers: int | None) -> tuple[ScanResult, int]:
    return _run_scans(scan_warc_files, [str(p) for p in files], 1, workers)


def _scan_directory(
    root: Path, run: RunDir, manifest: Path | None, workers: int | None
) -> tuple[ScanResult, int, dict[str, object]]:
    manifest_path = manifest if manifest is not None else root / MANIFEST_NAME
    entries = read_manifest(manifest_path, root) if manifest_path.exists() else None
    if manifest is not None and entries is None:
        raise FileNotFoundError(f"manifest not found: {manifest}")
    urls = entries.urls if entries is not None else {}
    listing = list_html_files(
        root,
        listed=urls.keys(),
        ignore={normalize_relpath(str(manifest_path.resolve()), root)},
        exclude={os.path.realpath(run.root), os.path.realpath(run.blobs_dir)},
    )

    for prefix in range(256):
        (run.blobs_dir / f"{prefix:02x}").mkdir(parents=True, exist_ok=True)
    worker = partial(scan_files, root=str(root), run_root=str(run.root.resolve()))
    scan, n_workers = _run_scans(worker, listing.files, FILES_PER_CHUNK, workers)

    captures = [c._replace(url=urls.get(c.url, c.url)) for c in scan.captures]
    counts = Counter(scan.counts)
    counts["files"] = len(listing.files)
    counts["skipped_non_html"] += listing.skipped_non_html
    seen = {rel for rel, _ in listing.files}
    meta: dict[str, object] = {
        "manifest": str(manifest_path.resolve()) if entries is not None else None,
        "manifest_entries": len(urls),
        "manifest_invalid_lines": entries.invalid_lines if entries is not None else 0,
        "manifest_missing": sum(1 for rel in urls if rel not in seen),
    }
    return ScanResult(captures, counts, scan.errors), n_workers, meta


def _latest_per_url(captures: list[Capture]) -> tuple[list[Capture], int]:
    """Keep the highest-ranked (latest) capture of every URL."""
    best: dict[str, Capture] = {}
    for capture in captures:
        current = best.get(capture.url)
        if current is None or capture.rank > current.rank:
            best[capture.url] = capture
    return list(best.values()), len(captures) - len(best)


def _drop_duplicates(captures: list[Capture]) -> tuple[list[Capture], int]:
    """Sort by URL and keep the first page of every payload hash."""
    seen: set[str] = set()
    pages: list[Capture] = []
    for capture in sorted(captures, key=lambda c: c.url):
        if capture.sha256 not in seen:
            seen.add(capture.sha256)
            pages.append(capture)
    return pages, len(captures) - len(pages)


def _prune_blobs(blobs_dir: Path, keep: set[str]) -> None:
    """Delete every blob (and stray temp file) whose name is not a kept page hash."""
    for sub in os.scandir(blobs_dir):
        if sub.is_dir(follow_symlinks=False):
            for entry in os.scandir(sub.path):
                if entry.name not in keep:
                    os.unlink(entry.path)


def _guess_site(urls: list[str], fallback: str, sample: int = 10_000) -> str:
    """Most common URL host over an evenly spaced sample of at most ``sample`` URLs (ties:
    first in URL order), else ``fallback``."""
    hosts: Counter[str] = Counter()
    for url in urls[:: max(1, -(-len(urls) // sample))]:
        try:
            host = urlsplit(url).hostname
        except ValueError:
            continue
        if host:
            hosts[host] += 1
    return hosts.most_common(1)[0][0] if hosts else fallback


def _content_digest(pages: list[Capture]) -> str:
    """SHA-256 over ``url<TAB>sha256`` lines in page order; changes iff the page set changes."""
    digest = hashlib.sha256()
    for capture in pages:
        digest.update(f"{capture.url}\t{capture.sha256}\n".encode())
    return digest.hexdigest()
