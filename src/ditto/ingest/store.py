"""The page index (``pages.sqlite``) and raw page access for every later stage."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ditto.ingest.capture import Capture
from ditto.ingest.warc import read_warc_payload
from ditto.run import RunDir

SCHEMA = """
CREATE TABLE pages (
    page_id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    fetched_at TEXT,
    offset INTEGER,
    length INTEGER,
    encoding TEXT NOT NULL
);
CREATE INDEX pages_url ON pages(url);
"""

_COLUMNS = "page_id, url, source, sha256, fetched_at, offset, length, encoding"


@dataclass(frozen=True, slots=True)
class PageRecord:
    """One ingested page. ``source`` is an absolute path: the page's blob (``offset`` is None,
    ``length`` is the byte size) or the WARC file holding the record at ``offset``/``length``."""

    page_id: int
    url: str
    source: str
    sha256: str
    fetched_at: str | None
    offset: int | None
    length: int | None
    encoding: str


def write_pages_db(path: Path, pages: Sequence[Capture]) -> None:
    """Write ``pages.sqlite`` with ``page_id`` = position in ``pages``, replacing any old one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as con:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.executescript(SCHEMA)
        con.executemany(
            f"INSERT INTO pages ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (i, c.url, c.source, c.sha256, c.fetched_at, c.offset, c.length, c.encoding)
                for i, c in enumerate(pages)
            ),
        )
        con.commit()
    for suffix in ("-journal", "-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    os.replace(tmp, path)


class PageStore:
    """Read access to an ingested run: page records (all held in memory, ~0.3 KB each) and raw
    bytes / decoded HTML per page. Holds no open file or database handle, so it is safe to
    share with forked workers."""

    def __init__(self, run: RunDir, records: list[PageRecord]) -> None:
        self.run = run
        self._records = records
        self._by_url: dict[str, int] | None = None

    @classmethod
    def open(cls, run: RunDir | Path | str) -> PageStore:
        run = run if isinstance(run, RunDir) else RunDir(Path(run))
        if not run.pages_db.exists():
            raise FileNotFoundError(f"{run.pages_db} does not exist; run `ditto ingest` first")
        root = os.path.abspath(run.root)
        with closing(sqlite3.connect(run.pages_db)) as con:
            rows = con.execute(f"SELECT {_COLUMNS} FROM pages ORDER BY page_id").fetchall()
        records = [
            PageRecord(pid, url, os.path.join(root, src), sha, fetched, off, length, enc)
            for pid, url, src, sha, fetched, off, length, enc in rows
        ]
        if any(rec.page_id != i for i, rec in enumerate(records)):
            raise ValueError(f"{run.pages_db}: page ids are not dense 0..N-1")
        return cls(run, records)

    @classmethod
    def cached(cls, run_root: Path | str) -> PageStore:
        """The store for ``run_root``, opened once per process (for pmap workers)."""
        key = os.fspath(run_root)
        store = _CACHE.get(key)
        if store is None:
            store = _CACHE[key] = cls.open(key)
        return store

    def __len__(self) -> int:
        return len(self._records)

    def ids(self) -> list[int]:
        return list(range(len(self._records)))

    def record(self, page_id: int) -> PageRecord:
        if not 0 <= page_id < len(self._records):
            raise KeyError(page_id)
        return self._records[page_id]

    def records(self) -> Iterator[PageRecord]:
        return iter(self._records)

    def url(self, page_id: int) -> str:
        return self.record(page_id).url

    def lookup(self, url: str) -> int | None:
        """Page id of an exact URL, or None."""
        if self._by_url is None:
            self._by_url = {rec.url: rec.page_id for rec in self._records}
        return self._by_url.get(url)

    def read_bytes(self, page_id: int) -> bytes:
        """Raw page bytes: the blob, or the WARC record's HTTP payload body."""
        rec = self.record(page_id)
        if rec.offset is None:
            with open(rec.source, "rb") as fh:
                return fh.read()
        return read_warc_payload(rec.source, rec.offset, rec.length or 0)

    def read_html(self, page_id: int) -> str:
        """The page decoded with its detected encoding (undecodable bytes replaced)."""
        return self.read_bytes(page_id).decode(self.record(page_id).encoding, errors="replace")


_CACHE: dict[str, PageStore] = {}


def clear_cache() -> None:
    """Forget per-process stores (ingest calls this after rewriting a run)."""
    _CACHE.clear()


def read_html(run_root: Path | str, page_id: int) -> str:
    """Decoded HTML of one page through the per-process cached store (for pmap workers)."""
    return PageStore.cached(run_root).read_html(page_id)
