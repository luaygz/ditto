"""Polite snapshot fetcher for the LiveWeb-IE pages (robots.txt, 1 request/s/host, WARC output).

Etiquette (from the ditto brief and RFC 9309):

* every request sends ``User-Agent: ditto-benchmark/1.0 (+https://github.com/luaygz/ditto)``;
* ``robots.txt`` is fetched first and obeyed: the group for our product token
  ``ditto-benchmark`` if one exists, else ``*``; longest match wins, ``allow`` wins ties, ``*``
  and ``$`` wildcards are supported. A 4xx ``robots.txt`` allows everything; a 5xx or network
  failure (after retries) disallows everything (RFC 9309 section 2.3.1);
* ``Crawl-delay`` is honoured when it is longer than the 1 second floor between requests to a
  host (the floor counts retries and redirect hops);
* a ``Content-Signal`` with ``ai-input=no`` excludes the host, because ditto feeds page content to
  an LLM during spec generation;
* 429 and 5xx back off exponentially (honouring ``Retry-After``); redirects are followed by hand,
  same host only, each hop robots-checked and archived.

Each host gets one ``<website>.warc.gz`` (warcio, gzip per record): a ``warcinfo`` record, the
``robots.txt`` exchange, then a ``request``/``response`` pair per fetch. Payloads are stored
decoded: ``Content-Encoding`` is renamed ``X-Ditto-Orig-Content-Encoding``, ``Transfer-Encoding``
is dropped and ``Content-Length`` is the stored length. A JSONL fetch log per website records
every URL's outcome and the WARC byte offset after it, so an interrupted run resumes by truncating
the ``.part`` file to the last logged offset and skipping logged URLs.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

USER_AGENT = "ditto-benchmark/1.0 (+https://github.com/luaygz/ditto)"
PRODUCT_TOKEN = "ditto-benchmark"
MIN_INTERVAL_S = 1.0
MAX_REDIRECTS = 5
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
MAX_BACKOFF_S = 300.0
DISALLOWED = "disallowed by robots.txt"
TIMEOUT_S = 60.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_DROP_HEADERS = frozenset({"transfer-encoding", "content-length", "content-encoding"})


# --------------------------------------------------------------------------- robots.txt


@dataclass
class RobotsGroup:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)  # (allow, pattern)
    crawl_delay: float | None = None
    content_signals: dict[str, str] = field(default_factory=dict)


@dataclass
class Robots:
    """Parsed ``robots.txt`` reduced to the rules that apply to one product token."""

    rules: list[tuple[bool, str]]
    crawl_delay: float | None
    content_signals: dict[str, str]
    mode: str = "parsed"
    """``parsed`` | ``allow_all`` (4xx) | ``disallow_all`` (unreachable)."""

    @classmethod
    def allow_all(cls) -> Robots:
        return cls([], None, {}, "allow_all")

    @classmethod
    def disallow_all(cls) -> Robots:
        return cls([(False, "/")], None, {}, "disallow_all")

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        if parts.path == "/robots.txt":
            return True
        best_len = -1
        best_allow = True
        for allow, pattern in self.rules:
            if not pattern:
                continue
            if _robots_match(pattern, target) or _robots_match(pattern, unquote(target)):
                n = len(pattern)
                if n > best_len or (n == best_len and allow):
                    best_len, best_allow = n, allow
        return best_allow

    def forbids_ai_input(self) -> bool:
        return self.content_signals.get("ai-input", "").lower() == "no"


def parse_robots(text: str, token: str = PRODUCT_TOKEN) -> Robots:
    """Parse ``robots.txt`` per RFC 9309 and keep the group(s) that apply to ``token``."""
    groups: list[RobotsGroup] = []
    current: RobotsGroup | None = None
    in_agents = False
    loose_signals: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (s.strip() for s in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if current is None or not in_agents:
                current = RobotsGroup()
                groups.append(current)
            current.agents.append(value.lower())
            in_agents = True
            continue
        if key in ("allow", "disallow"):
            in_agents = False
            if current is not None:
                current.rules.append((key == "allow", value))
        elif key == "crawl-delay":
            in_agents = False
            if current is not None:
                with contextlib.suppress(ValueError):
                    current.crawl_delay = float(value)
        elif key == "content-signal":
            in_agents = False
            target = current.content_signals if current is not None else loose_signals
            target.update(_parse_signals(value))
    token = token.lower()
    chosen = [g for g in groups if any(a != "*" and _agent_matches(a, token) for a in g.agents)]
    if not chosen:
        chosen = [g for g in groups if "*" in g.agents]
    rules = [r for g in chosen for r in g.rules]
    delays = [g.crawl_delay for g in chosen if g.crawl_delay is not None]
    signals = dict(loose_signals)
    for g in chosen:
        signals.update(g.content_signals)
    return Robots(rules, max(delays) if delays else None, signals)


def _agent_matches(agent: str, token: str) -> bool:
    return agent == token or agent.split("/", 1)[0] == token


def _parse_signals(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in value.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip().lower()
    return out


def _robots_match(pattern: str, target: str) -> bool:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(piece) for piece in body.split("*"))
    return re.match(regex + ("$" if anchored else ""), target) is not None


# --------------------------------------------------------------------------- WARC


@dataclass
class Exchange:
    """One HTTP request/response as archived."""

    url: str
    method: str
    request_headers: list[tuple[str, str]]
    status: int
    reason: str
    http_version: str
    response_headers: list[tuple[str, str]]
    body: bytes
    fetched_at: dt.datetime
    ip: str | None = None


def stored_headers(headers: Sequence[tuple[str, str]], body_len: int) -> list[tuple[str, str]]:
    """Response headers as archived for a decoded payload (see the module docstring)."""
    out: list[tuple[str, str]] = []
    for name, value in headers:
        low = name.lower()
        if low == "content-encoding":
            out.append(("X-Ditto-Orig-Content-Encoding", value))
        elif low not in _DROP_HEADERS:
            out.append((name, value))
    out.append(("Content-Length", str(body_len)))
    return out


def write_warcinfo(writer: WARCWriter, filename: str, info: dict[str, str]) -> None:
    writer.write_record(writer.create_warcinfo_record(filename, info))


def write_exchange(writer: WARCWriter, ex: Exchange) -> None:
    """Append a ``request`` + ``response`` record pair."""
    date = ex.fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = urlsplit(ex.url)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    req_headers = StatusAndHeaders(
        f"{ex.method} {target} {ex.http_version}",
        list(ex.request_headers),
        is_http_request=True,
    )
    resp_headers = StatusAndHeaders(
        f"{ex.status} {ex.reason}".strip(),
        stored_headers(ex.response_headers, len(ex.body)),
        protocol=ex.http_version,
    )
    warc_headers: dict[str, str] = {"WARC-Date": date}
    if ex.ip:
        warc_headers["WARC-IP-Address"] = ex.ip
    resp = writer.create_warc_record(
        ex.url,
        "response",
        payload=io.BytesIO(ex.body),
        length=len(ex.body),
        http_headers=resp_headers,
        warc_headers_dict=dict(warc_headers),
    )
    req = writer.create_warc_record(
        ex.url,
        "request",
        payload=io.BytesIO(b""),
        length=0,
        http_headers=req_headers,
        warc_headers_dict=dict(warc_headers),
    )
    writer.write_request_response_pair(req, resp)


# --------------------------------------------------------------------------- fetching


class HostThrottle:
    """At most one request start per ``interval`` seconds (thread-safe)."""

    def __init__(
        self,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                delay = self._last + self.interval - now
                if delay > 0:
                    self._sleep(delay)
                    now = self._clock()
            self._last = now


@dataclass
class FetchOutcome:
    exchanges: list[Exchange]
    final_url: str
    status: int | None
    error: str | None = None


class PoliteClient:
    """One host's HTTP client: throttled, retried, redirects followed by hand."""

    def __init__(
        self,
        host: str,
        interval: float,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.host = host
        self.throttle = HostThrottle(interval, sleep=sleep)
        self._sleep = sleep
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
            follow_redirects=False,
            timeout=TIMEOUT_S,
        )
        self.requests = 0

    def close(self) -> None:
        self.client.close()

    def get_once(self, url: str) -> Exchange:
        """One GET with retries on 429/5xx/network errors; raises the last error."""
        last_exc: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self.throttle.wait()
            self.requests += 1
            try:
                resp = self.client.get(url)
            except httpx.HTTPError as exc:
                last_exc = exc
                self._sleep(BACKOFF_BASE_S * 2**attempt)
                continue
            ex = _exchange(resp)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS - 1:
                wait = max(BACKOFF_BASE_S * 2**attempt, _retry_after(resp))
                self._sleep(min(wait, MAX_BACKOFF_S))
                continue
            return ex
        assert last_exc is not None
        raise last_exc

    def fetch(self, url: str, robots: Robots) -> FetchOutcome:
        """GET ``url`` following same-host redirects; every hop is returned for archiving.

        ``final_url`` is the URL as sent on the wire (percent-encoded), which is what the WARC
        records as ``WARC-Target-URI``.
        """
        exchanges: list[Exchange] = []
        current = url
        for _hop in range(MAX_REDIRECTS + 1):
            try:
                ex = self.get_once(current)
            except httpx.HTTPError as exc:
                return FetchOutcome(exchanges, current, None, f"{type(exc).__name__}: {exc}")
            exchanges.append(ex)
            location = _header(ex.response_headers, "location")
            if ex.status not in (301, 302, 303, 307, 308) or not location:
                return FetchOutcome(exchanges, ex.url, ex.status)
            nxt = urljoin(ex.url, location)
            if urlsplit(nxt).netloc != self.host:
                return FetchOutcome(exchanges, ex.url, ex.status, f"off-host redirect to {nxt}")
            if not robots.allowed(nxt):
                return FetchOutcome(
                    exchanges, ex.url, ex.status, f"redirect target disallowed: {nxt}"
                )
            current = nxt
        return FetchOutcome(
            exchanges, exchanges[-1].url, exchanges[-1].status, "too many redirects"
        )


def fetch_robots(
    client: PoliteClient, scheme: str = "https"
) -> tuple[Robots, list[Exchange], dict[str, Any]]:
    """Fetch and interpret ``<scheme>://<host>/robots.txt`` (RFC 9309 section 2.3)."""
    url = f"{scheme}://{client.host}/robots.txt"
    outcome = client.fetch(url, Robots.allow_all())
    info: dict[str, Any] = {"url": url, "final_url": outcome.final_url, "status": outcome.status}
    if outcome.status is None:
        info["error"] = outcome.error
        info["mode"] = "disallow_all"
        return Robots.disallow_all(), outcome.exchanges, info
    body = outcome.exchanges[-1].body
    info["sha256"] = hashlib.sha256(body).hexdigest()
    if 200 <= outcome.status < 300:
        robots = parse_robots(body.decode("utf-8", errors="replace"))
    elif 400 <= outcome.status < 500:
        robots = Robots.allow_all()
    else:
        robots = Robots.disallow_all()
    info.update(
        mode=robots.mode,
        crawl_delay=robots.crawl_delay,
        content_signals=robots.content_signals,
        rules=len(robots.rules),
    )
    return robots, outcome.exchanges, info


@dataclass
class SiteResult:
    website: str
    host: str
    warc: Path | None
    robots: dict[str, Any]
    counts: dict[str, int]
    excluded_reason: str | None
    started_at: str
    finished_at: str
    requests: int


def snapshot_site(
    website: str,
    urls: Sequence[str],
    out_dir: Path,
    *,
    client: PoliteClient | None = None,
    log: Callable[[str], None] = print,
) -> SiteResult:
    """Snapshot one website's pages into ``out_dir/<website>.warc.gz`` (resumable)."""
    hosts = {urlsplit(u).netloc for u in urls}
    if len(hosts) != 1:
        raise ValueError(f"{website}: expected one host, got {sorted(hosts)}")
    host = hosts.pop()
    scheme = urlsplit(urls[0]).scheme or "https"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fetchlog").mkdir(exist_ok=True)
    final = out_dir / f"{website}.warc.gz"
    part = out_dir / f"{website}.warc.gz.part"
    log_path = out_dir / "fetchlog" / f"{website}.jsonl"
    done_path = out_dir / "fetchlog" / f"{website}.done.json"
    if final.is_file() and done_path.is_file():
        data = json.loads(done_path.read_text(encoding="utf-8"))
        log(f"{website}: complete snapshot present, skipping")
        return SiteResult(**{**data, "warc": final if data.get("warc") else None})
    started = _now_iso()
    if final.is_file() and not part.exists():
        final.replace(part)  # renamed but summary never written: resume from it
    done = _resume(part, log_path)
    excluded: str | None = None
    own = client is None
    pc = client or PoliteClient(host, MIN_INTERVAL_S)
    try:
        with part.open("ab") as fh:
            writer = WARCWriter(fh, gzip=True)
            if fh.tell() == 0:
                write_warcinfo(writer, final.name, _warcinfo(website))
            robots, robots_ex, robots_info = fetch_robots(pc, scheme)
            for ex in robots_ex:
                write_exchange(writer, ex)
            fh.flush()
            interval = max(MIN_INTERVAL_S, robots.crawl_delay or 0.0)
            pc.throttle.interval = interval
            robots_info["interval_s"] = interval
            if robots.mode == "disallow_all":
                excluded = (
                    f"robots.txt unreachable ({robots_info.get('error') or robots_info['status']}); "
                    "RFC 9309 requires assuming complete disallow"
                )
            elif robots.forbids_ai_input():
                excluded = (
                    "robots.txt Content-Signal ai-input=no; ditto feeds page content to an LLM "
                    "during spec generation"
                )
            todo = [u for u in dict.fromkeys(urls) if u not in done]
            if excluded:
                log(f"{website}: excluded: {excluded}")
                todo = []
            log(
                f"{website}: {len(done)} already logged, {len(todo)} to fetch, interval {interval}s"
            )
            with log_path.open("a", encoding="utf-8") as logf:
                for i, url in enumerate(todo):
                    row: dict[str, Any] = {"url": url}
                    start = fh.tell()
                    if not robots.allowed(url):
                        row.update(status=None, error=DISALLOWED)
                    else:
                        try:
                            outcome = pc.fetch(url, robots)
                            for ex in outcome.exchanges:
                                write_exchange(writer, ex)
                            fh.flush()
                            row.update(_outcome_row(outcome))
                        except Exception as exc:  # one bad page must not end the site
                            fh.seek(start)
                            fh.truncate()
                            row.update(status=None, error=f"{type(exc).__name__}: {exc}")
                            log(f"{website}: {url}: {row['error']}")
                    row["warc_end"] = fh.tell()
                    logf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    logf.flush()
                    if (i + 1) % 100 == 0:
                        log(f"{website}: {i + 1}/{len(todo)}")
    finally:
        if own:
            pc.close()
    rows = _read_log(log_path)
    counts = _count(rows)
    if excluded:
        part.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        warc: Path | None = None
    else:
        part.replace(final)
        warc = final
    result = SiteResult(
        website=website,
        host=host,
        warc=warc,
        robots=robots_info,
        counts=counts,
        excluded_reason=excluded,
        started_at=started,
        finished_at=_now_iso(),
        requests=pc.requests,
    )
    payload = {**result.__dict__, "warc": str(warc) if warc else None}
    done_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result


def _outcome_row(outcome: FetchOutcome) -> dict[str, Any]:
    row: dict[str, Any] = {"final_url": outcome.final_url, "status": outcome.status}
    if outcome.error:
        row["error"] = outcome.error
    if outcome.exchanges:
        last = outcome.exchanges[-1]
        row["content_type"] = _header(last.response_headers, "content-type")
        row["bytes"] = len(last.body)
        row["sha256"] = hashlib.sha256(last.body).hexdigest()
        row["fetched_at"] = last.fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        row["hops"] = len(outcome.exchanges)
    return row


def _resume(part: Path, log_path: Path) -> set[str]:
    """Truncate ``part`` to the last logged record end; return the settled URLs.

    A URL is settled once it has an HTTP answer or was disallowed; network failures and
    exceptions are retried on the next run.
    """
    rows = _read_log(log_path)
    if not part.is_file():
        if rows:
            log_path.unlink()
        return set()
    end = rows[-1]["warc_end"] if rows else 0
    with part.open("r+b") as fh:
        fh.truncate(end)
    return {url for url, r in latest_rows(rows).items() if _settled(r)}


def latest_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last fetch-log row per URL (a retried URL has several)."""
    return {r["url"]: r for r in rows}


def _settled(row: dict[str, Any]) -> bool:
    return row.get("status") is not None or row.get("error") == DISALLOWED


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # torn last line from an interrupted run
    return rows


def _count(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    latest = latest_rows(rows)
    counts = {"urls": len(latest), "ok_html": 0, "non_200": 0, "errors": 0, "disallowed": 0}
    for r in latest.values():
        if r.get("error") == DISALLOWED:
            counts["disallowed"] += 1
        elif r.get("status") is None:
            counts["errors"] += 1
        elif r["status"] == 200 and "html" in str(r.get("content_type", "")).lower():
            counts["ok_html"] += 1
        else:
            counts["non_200"] += 1
    return counts


def _exchange(resp: httpx.Response) -> Exchange:
    req = resp.request
    ip: str | None = None
    stream = resp.extensions.get("network_stream")
    if stream is not None:
        addr = stream.get_extra_info("server_addr")
        if isinstance(addr, tuple) and addr:
            ip = str(addr[0])
    return Exchange(
        url=str(req.url),
        method=req.method,
        request_headers=[(k, v) for k, v in req.headers.multi_items()],
        status=resp.status_code,
        reason=resp.reason_phrase,
        http_version=resp.http_version or "HTTP/1.1",
        response_headers=[(k, v) for k, v in resp.headers.multi_items()],
        body=resp.content,
        fetched_at=dt.datetime.now(dt.UTC),
        ip=ip,
    )


def _retry_after(resp: httpx.Response) -> float:
    value = resp.headers.get("retry-after")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, (when - dt.datetime.now(dt.UTC)).total_seconds())


def _header(headers: Sequence[tuple[str, str]], name: str) -> str | None:
    name = name.lower()
    for k, v in headers:
        if k.lower() == name:
            return v
    return None


def _warcinfo(website: str) -> dict[str, str]:
    return {
        "software": "ditto eval/datasets/livewebie/snapshot.py (warcio)",
        "operator": "ditto benchmark (https://github.com/luaygz/ditto)",
        "http-header-user-agent": USER_AGENT,
        "robots": "obey (RFC 9309); Crawl-delay honoured; >= 1 s between requests per host",
        "description": (
            f"Snapshot of the LiveWeb-IE (arXiv 2603.13773) benchmark pages of {website}; "
            "payloads stored decoded (Content-Encoding renamed X-Ditto-Orig-Content-Encoding)."
        ),
        "format": "WARC File Format 1.0",
    }


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
