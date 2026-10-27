"""Shared Common Crawl plumbing for the silver-set scripts (select, download, labels, noise check).

Network etiquette (Common Crawl asks for it, and the brief requires it):

* every request carries a descriptive ``User-Agent``;
* requests to ``data.commoncrawl.org`` are serialized (:data:`MAX_DATA_CONCURRENCY` = 1) and
  start at least :data:`DATA_INTERVAL_S` apart; at most :data:`MAX_INDEX_CONCURRENCY` concurrent
  requests go to ``index.commoncrawl.org``;
* a 403, 429 or 503 from ``data.commoncrawl.org`` means "slow down" (Common Crawl's CloudFront
  WAF answers 403 "Request blocked" to clients it throttles): every data request then waits an
  exponential backoff with jitter, starting at :data:`SLOW_DOWN_START_S` and doubling up to
  :data:`SLOW_DOWN_MAX_S` while the block lasts, and the next request is the probe (no storm of
  retries). Slow-down answers never exhaust a request's attempts: the request waits as long as
  the block lasts. A success resets the backoff;
* other 5xx and transport errors back off exponentially with jitter (honouring ``Retry-After``)
  for at most ``max_attempts``;
* every download is a byte-range request that resumes from the last received byte when a
  transfer breaks;
* index data is cached on disk under ``data/cc/index_cache/`` so every query is made once.

Index queries. ``index.commoncrawl.org`` serves each crawl's CDX index from ZipNum files: a
``cluster.idx`` that maps the first SURT key of every 3,000-line block to its byte range in one
of the ``cdx-NNNNN.gz`` files. During selection (2026-10-24) the index server answered most
requests with 504 Gateway Time-out, taking ~6 minutes per page with backoff, so
:class:`CdxIndex` reads the same ZipNum files directly from ``data.commoncrawl.org`` and applies
the same prefix match and ``status``/``mime`` filters the server would
(``url=<domain>/*&filter==status:200&filter==mime:text/html``). The rows are the server's rows.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import io
import json
import random
import re
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import yaml
from warcio.archiveiterator import ArchiveIterator

USER_AGENT = "ditto-benchmark/1.0 (+https://github.com/luaygz/ditto)"
INDEX_BASE = "https://index.commoncrawl.org"
COLLINFO_URL = f"{INDEX_BASE}/collinfo.json"
DATA_BASE = "https://data.commoncrawl.org"

MAX_DATA_CONCURRENCY = 1
MAX_INDEX_CONCURRENCY = 2
DATA_INTERVAL_S = 1.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
SLOW_DOWN_STATUS = frozenset({403, 429, 503})
SLOW_DOWN_START_S = 60.0
SLOW_DOWN_MAX_S = 1800.0

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_DIR = REPO / "data" / "cc"
INDEX_CACHE = DATA_DIR / "index_cache"
FIELDS_DIR = HERE / "fields"
DOMAINS_YAML = HERE / "domains.yaml"
CAPTURES_JSONL = HERE / "captures.jsonl.gz"
MANIFEST_JSON = HERE / "manifest.json"
NOISE_JSON = HERE / "noise_check.json"
SELECTION_JSON = HERE / "selection_log.json"


class FetchError(RuntimeError):
    """A request failed permanently (non-retryable status or retries exhausted)."""


class Http:
    """Thread-safe polite client: paced and serialized data requests, slow-down handling,
    backoff, byte-range resume (see the module docstring)."""

    def __init__(
        self,
        *,
        data_concurrency: int = MAX_DATA_CONCURRENCY,
        index_concurrency: int = MAX_INDEX_CONCURRENCY,
        max_attempts: int = 10,
        max_delay: float = 300.0,
        data_interval: float = DATA_INTERVAL_S,
        slow_down_start: float = SLOW_DOWN_START_S,
        slow_down_max: float = SLOW_DOWN_MAX_S,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if data_concurrency > MAX_DATA_CONCURRENCY:
            raise ValueError(f"at most {MAX_DATA_CONCURRENCY} concurrent data requests")
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(180.0, connect=30.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        )
        self._sems = {
            "data": threading.BoundedSemaphore(data_concurrency),
            "index": threading.BoundedSemaphore(index_concurrency),
        }
        self.max_attempts = max_attempts
        self.max_delay = max_delay
        self.data_interval = data_interval
        self.slow_down_start = slow_down_start
        self.slow_down_max = slow_down_max
        self._sleep_fn = sleep
        self._clock = clock
        self._log = log or (lambda msg: print(msg, flush=True))
        self._next_data_start = 0.0
        self._slow_level = 0
        self.lock = threading.Lock()
        self.stats: dict[str, float] = {
            "requests": 0,
            "retries": 0,
            "bytes": 0,
            "slow_downs": 0,
            "slow_down_seconds": 0.0,
        }

    def close(self) -> None:
        self.client.close()

    def _count(self, key: str, value: float = 1) -> None:
        with self.lock:
            self.stats[key] += value

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        delay = min(self.max_delay, 2.0 * 2**attempt) * (0.5 + random.random())
        if retry_after and retry_after.strip().isdigit():
            delay = max(delay, float(retry_after.strip()))
        self._count("retries")
        self._sleep_fn(delay)

    @contextmanager
    def _slot(self, url: str) -> Iterator[None]:
        """Hold the host's concurrency slot; data requests also wait for their start time."""
        data = url.startswith(DATA_BASE)
        with self._sems["data" if data else "index"]:
            if data:
                wait = self._next_data_start - self._clock()
                if wait > 0:
                    self._sleep_fn(wait)
                self._next_data_start = self._clock() + self.data_interval
            yield

    def _slowed(self, url: str, status: int, retry_after: str | None) -> bool:
        """Handle a data-host answer inside its slot: ``True`` when it asks us to slow down.

        Pushes the next data request start out by the jittered backoff of the current block
        level; a non-slow-down answer ends the block."""
        if not url.startswith(DATA_BASE):
            return False
        if status not in SLOW_DOWN_STATUS:
            self._slow_level = 0
            return False
        self._slow_level += 1
        base = min(self.slow_down_max, self.slow_down_start * 2 ** (self._slow_level - 1))
        delay = min(self.slow_down_max, base * (0.75 + 0.5 * random.random()))
        if retry_after and retry_after.strip().isdigit():
            delay = max(delay, float(retry_after.strip()))
        self._next_data_start = self._clock() + delay
        self._count("slow_downs")
        self._count("slow_down_seconds", delay)
        self._log(
            f"data.commoncrawl.org: HTTP {status} (slow down), next request in {delay:.0f}s "
            f"(level {self._slow_level})"
        )
        return True

    def get(self, url: str, *, ok: tuple[int, ...] = (200,), method: str = "GET") -> httpx.Response:
        """GET (or HEAD) with backoff; returns the response when its status is in ``ok``."""
        last = ""
        attempt = 0
        while attempt < self.max_attempts:
            try:
                with self._slot(url):
                    resp = self.client.request(method, url)
                    slowed = self._slowed(url, resp.status_code, resp.headers.get("Retry-After"))
                self._count("requests")
                self._count("bytes", len(resp.content))
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
                self._sleep(attempt)
                attempt += 1
                continue
            if resp.status_code in ok:
                return resp
            if slowed:
                continue  # the slot already waits out the block; not an attempt
            last = f"HTTP {resp.status_code}"
            if resp.status_code in RETRY_STATUS:
                self._sleep(attempt, resp.headers.get("Retry-After"))
                attempt += 1
                continue
            raise FetchError(f"{url}: {last}")
        raise FetchError(f"{url}: gave up after {self.max_attempts} attempts ({last})")

    def size(self, filename: str) -> int:
        resp = self.get(f"{DATA_BASE}/{filename}", method="HEAD")
        return int(resp.headers["Content-Length"])

    def fetch_range(self, filename: str, offset: int, length: int) -> bytes:
        """Bytes ``[offset, offset+length)`` of a Common Crawl file, resuming broken transfers."""
        url = f"{DATA_BASE}/{filename}"
        buf = bytearray()
        attempt = 0
        last = ""
        while len(buf) < length:
            if attempt >= self.max_attempts:
                raise FetchError(f"{url}: gave up after {attempt} attempts ({last})")
            start = offset + len(buf)
            headers = {"Range": f"bytes={start}-{offset + length - 1}"}
            retry_after: str | None = None
            try:
                with self._slot(url), self.client.stream("GET", url, headers=headers) as resp:
                    self._count("requests")
                    if self._slowed(url, resp.status_code, resp.headers.get("Retry-After")):
                        continue  # waited out in the next slot; never counts as an attempt
                    if resp.status_code in RETRY_STATUS:
                        last = f"HTTP {resp.status_code}"
                        retry_after = resp.headers.get("Retry-After")
                    elif resp.status_code != 206:
                        raise FetchError(f"{url}: HTTP {resp.status_code} for a range request")
                    else:
                        for chunk in resp.iter_bytes():
                            buf += chunk
                            self._count("bytes", len(chunk))
                        if len(buf) >= length:
                            break
                        last = f"short body ({len(buf)}/{length} bytes)"
            except httpx.TransportError as exc:
                # keep what arrived; the next attempt asks only for the missing tail
                last = f"{type(exc).__name__}: {exc}"
            self._sleep(attempt, retry_after)
            attempt += 1
        return bytes(buf[:length])

    def download(self, filename: str, dest: Path, *, chunk: int = 16 << 20) -> Path:
        """Whole file by consecutive byte ranges into ``dest`` (resumes a ``.part`` file)."""
        if dest.exists():
            return dest
        total = self.size(filename)
        part = dest.with_name(dest.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open("ab") as fh:
            done = fh.tell()
            while done < total:
                data = self.fetch_range(filename, done, min(chunk, total - done))
                fh.write(data)
                fh.flush()
                done += len(data)
        part.replace(dest)
        return dest


# --------------------------------------------------------------------------- index


def collinfo(http: Http, *, refresh: bool = False) -> list[dict[str, Any]]:
    """The crawl list from index.commoncrawl.org, newest first (cached)."""
    path = INDEX_CACHE / "collinfo.json"
    if refresh or not path.exists():
        data = http.get(COLLINFO_URL).json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
    result: list[dict[str, Any]] = json.loads(path.read_text())
    return result


def surt_host(domain: str) -> str:
    """SURT host prefix of a domain as the CDX index keys it (``www.`` dropped):
    ``www.food.com`` -> ``com,food)``."""
    host = domain.lower().strip().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return ",".join(reversed(host.split("."))) + ")"


def html_filter(row: dict[str, Any]) -> bool:
    """The server's ``filter==status:200&filter==mime:text/html``."""
    return row.get("status") == "200" and row.get("mime") == "text/html"


@dataclass(frozen=True)
class Block:
    file: str
    offset: int
    length: int


class CdxIndex:
    """One crawl's ZipNum CDX index, read from ``data.commoncrawl.org`` and cached on disk."""

    def __init__(self, http: Http, crawl: str) -> None:
        self.http = http
        self.crawl = crawl
        self.root = f"cc-index/collections/{crawl}/indexes"
        self.cache = INDEX_CACHE / crawl
        idx = http.download(f"{self.root}/cluster.idx", self.cache / "cluster.idx")
        self.keys: list[str] = []
        self.blocks: list[Block] = []
        with idx.open("r", encoding="utf-8") as fh:
            for line in fh:
                key, file, offset, length, *_ = line.rstrip("\n").split("\t")
                self.keys.append(key)
                self.blocks.append(Block(file, int(offset), int(length)))

    def blocks_for(self, prefix: str) -> list[Block]:
        """Blocks whose key range intersects keys starting with ``prefix``."""
        start = max(0, bisect.bisect_right(self.keys, prefix) - 1)
        stop = bisect.bisect_left(self.keys, prefix + "\U0010ffff")
        return self.blocks[start : max(stop, start + 1)]

    def block_lines(self, block: Block) -> list[str]:
        path = self.cache / "blocks" / block.file / f"{block.offset}.gz"
        if not path.exists():
            raw = self.http.fetch_range(f"{self.root}/{block.file}", block.offset, block.length)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(raw)
            tmp.replace(path)
        raw = path.read_bytes()
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS).decode("utf-8").splitlines()

    def query(self, prefix: str, *, exact: bool = False) -> list[dict[str, Any]]:
        """Rows whose SURT key starts with ``prefix`` (or equals it when ``exact``)."""
        rows: list[dict[str, Any]] = []
        for block in self.blocks_for(prefix):
            for line in self.block_lines(block):
                key, _, rest = line.partition(" ")
                if not key.startswith(prefix) or (exact and key != prefix):
                    continue
                timestamp, _, payload = rest.partition(" ")
                row: dict[str, Any] = json.loads(payload)
                row["urlkey"] = key
                row["timestamp"] = timestamp
                rows.append(row)
        return rows

    def domain_rows(self, domain: str, *, html_only: bool = True) -> list[dict[str, Any]]:
        """``url=<domain>/*`` (host and ``www.`` host), status-200 text/html rows by default."""
        rows = self.query(surt_host(domain) + "/")
        return [r for r in rows if html_filter(r)] if html_only else rows

    def robots_rows(self, domain: str) -> list[dict[str, Any]]:
        return self.query(surt_host(domain) + "/robots.txt", exact=True)


# --------------------------------------------------------------------------- WARC records


@dataclass
class Record:
    url: str
    status: int
    content_type: str | None
    body: bytes
    warc_date: str | None


def parse_record(raw: bytes) -> Record:
    """Parse one gzip-member WARC response record fetched by byte range."""
    for rec in ArchiveIterator(io.BytesIO(raw)):
        if rec.rec_type != "response":
            continue
        return _record(rec)
    raise ValueError("no WARC response record in the fetched bytes")


def _record(rec: Any) -> Record:
    http_headers = rec.http_headers
    return Record(
        rec.rec_headers.get_header("WARC-Target-URI") or "",
        int(http_headers.get_statuscode() or 0) if http_headers else 0,
        http_headers.get_header("Content-Type") if http_headers else None,
        rec.content_stream().read(),
        rec.rec_headers.get_header("WARC-Date"),
    )


class _Bounded(io.RawIOBase):
    """Read-only view of the first ``limit`` bytes of a file (a WARC still being appended)."""

    def __init__(self, fh: io.BufferedReader, limit: int) -> None:
        self.fh = fh
        self.left = limit

    def readable(self) -> bool:
        return True

    def readinto(self, buf: Any) -> int:
        n = min(len(buf), self.left)
        if n <= 0:
            return 0
        data = self.fh.read(n)
        buf[: len(data)] = data
        self.left -= len(data)
        return len(data)


def iter_warc_pages(path: Path, limit: int | None = None) -> Iterator[Record]:
    """Every response record in a local WARC file (only its first ``limit`` bytes if given)."""
    with path.open("rb") as fh:
        stream: Any = fh if limit is None else io.BufferedReader(_Bounded(fh, limit), 1 << 20)
        for rec in ArchiveIterator(stream):
            if rec.rec_type == "response":
                yield _record(rec)


def crawl_of(row: dict[str, Any]) -> str:
    match = re.search(r"crawl-data/(CC-MAIN-\d{4}-\d{2})/", str(row["filename"]))
    return match.group(1) if match else ""


def url_path(url: str) -> str:
    """Path plus query of a URL (what detail-pattern regexes are matched against)."""
    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Deterministic gzip JSONL (mtime 0) so the committed file hashes reproducibly."""
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(payload)


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def dump_yaml(doc: Any) -> str:
    """Readable YAML for the committed files: key order kept, no anchors/aliases."""
    return yaml.dump(doc, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True, width=100)
