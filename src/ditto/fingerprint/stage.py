"""Stage 2 over a whole run: parse, clean and fingerprint every page in parallel."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ditto.fingerprint.dom import clean_etree
from ditto.fingerprint.store import FingerprintStore, FingerprintWriter
from ditto.fingerprint.tree import U32, U64, PageFingerprint, fingerprint_tree
from ditto.parallel import default_workers, pmap
from ditto.run import RunDir

CHUNK_SIZE = 64

# Worker state, set by ``_init_worker``: the run root and the path hashes this process has
# already shipped to the parent (each path string crosses the process boundary once per worker).
_ROOT: str = ""
_SENT: set[int] = set()


@dataclass
class _Chunk:
    sig_lens: NDArray[np.int64]
    sig_values: U64
    sig_hashes: U64
    text_lens: NDArray[np.int64]
    text_path: U64
    text_hash: U64
    text_count: U32
    new_paths: list[tuple[int, str]] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)


def _init_worker(root: str) -> None:
    global _ROOT
    _ROOT = root
    _SENT.clear()


def fingerprint_html(html: str) -> PageFingerprint:
    """Clean and fingerprint one page's HTML."""
    return fingerprint_tree(clean_etree(html))


def _fingerprint_chunk(page_ids: Sequence[int]) -> _Chunk:
    from ditto.ingest import read_html

    fps: list[PageFingerprint] = []
    errors: dict[int, str] = {}
    for pid in page_ids:
        try:
            fps.append(fingerprint_html(read_html(_ROOT, pid)))
        except Exception as exc:  # one unreadable page must not stop the stage
            errors[pid] = f"{type(exc).__name__}: {exc}"
            fps.append(PageFingerprint.empty())
    new_paths: list[tuple[int, str]] = []
    for fp in fps:
        for h, p in fp.paths.items():
            if h not in _SENT:
                _SENT.add(h)
                new_paths.append((h, p))
    return _Chunk(
        sig_lens=np.array([len(fp.signature) for fp in fps], dtype=np.int64),
        sig_values=np.concatenate([fp.signature for fp in fps]),
        sig_hashes=np.array([fp.signature_hash for fp in fps], dtype=np.uint64),
        text_lens=np.array([len(fp.text_path) for fp in fps], dtype=np.int64),
        text_path=np.concatenate([fp.text_path for fp in fps]),
        text_hash=np.concatenate([fp.text_hash for fp in fps]),
        text_count=np.concatenate([fp.text_count for fp in fps]),
        new_paths=new_paths,
        errors=errors,
    )


def fingerprint_run(run: RunDir, *, workers: int | None = None) -> FingerprintStore:
    """Fingerprint every ingested page into ``run.fingerprints_dir`` and return the store."""
    from ditto.ingest import PageStore

    start = time.perf_counter()
    n_pages = len(PageStore.open(run))
    n_chunks = -(-n_pages // CHUNK_SIZE)
    n_workers = max(1, min(default_workers() if workers is None else workers, n_chunks))
    writer = FingerprintWriter(run.fingerprints_dir)
    try:
        chunks = pmap(
            _fingerprint_chunk,
            range(n_pages),
            workers=n_workers,
            chunk_size=CHUNK_SIZE,
            initializer=_init_worker,
            initargs=(str(run.root),),
        )
        for c in chunks:
            writer.add_chunk(
                c.sig_lens,
                c.sig_values,
                c.sig_hashes,
                c.text_lens,
                c.text_path,
                c.text_hash,
                c.text_count,
                c.new_paths,
                c.errors,
            )
    except BaseException:
        writer.abort()
        raise
    store = writer.close()
    run.record_timing("fingerprint", time.perf_counter() - start, n_pages, n_workers)
    return store
