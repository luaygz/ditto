from __future__ import annotations

import datetime as dt
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from lwie_fixtures import load_script
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

snapshot: Any = load_script("snapshot")

# Excerpts of real robots.txt files fetched 2026-10-24.
ARXIV = """# robots.txt for http://arxiv.org/ and mirror sites http://*.arxiv.org/
User-agent: *
Crawl-delay: 15
Allow: /abs
Allow: /list
Disallow: /user
Disallow: /search

User-agent: Googlebot
Allow: /abs
Disallow: /api
"""
ICLR = """User-agent: *
Disallow: /static
Disallow: /virtual/*/search
Disallow: /*?*page=

User-agent: GPTBot
Disallow: /virtual/
Crawl-delay: 10
"""
SPORTSDB = """User-agent: *
Allow: /
Disallow: /nfo/
Content-Signal: ai-train=no, search=yes, ai-input=no

User-agent: GPTBot
Disallow: /
"""
DPLA = """User-agent: *
Disallow: /search
Crawl-delay: 2

User-agent: ClaudeBot
Disallow: /
"""


def test_parse_robots_star_group_and_crawl_delay() -> None:
    r = snapshot.parse_robots(ARXIV)
    assert r.crawl_delay == 15.0
    assert r.allowed("https://arxiv.org/abs/2401.00588")
    assert r.allowed("https://arxiv.org/list/cs.AI/2024-01")
    assert not r.allowed("https://arxiv.org/search?query=x")
    assert r.allowed("https://arxiv.org/robots.txt")


def test_parse_robots_wildcards() -> None:
    r = snapshot.parse_robots(ICLR)
    assert r.crawl_delay is None  # GPTBot's delay does not apply to us
    assert r.allowed("https://iclr.cc/virtual/2025/poster/30687")
    assert not r.allowed("https://iclr.cc/virtual/2025/search?q=x")
    assert not r.allowed("https://iclr.cc/virtual/2025/papers.html?filter=t&page=2")
    assert r.allowed("https://iclr.cc/virtual/2025/papers.html?filter=title&search=game")


def test_parse_robots_content_signal_and_other_bots() -> None:
    r = snapshot.parse_robots(SPORTSDB)
    assert r.forbids_ai_input()
    assert r.allowed("https://www.thesportsdb.com/event/1")
    d = snapshot.parse_robots(DPLA)
    assert not d.forbids_ai_input()
    assert d.crawl_delay == 2.0
    assert d.allowed("https://dp.la/item/abc")


def test_parse_robots_specific_group_wins_and_precedence() -> None:
    text = (
        "User-agent: *\nDisallow: /\n\n"
        "User-agent: ditto-benchmark\nUser-agent: other\nDisallow: /private\nAllow: /private/ok$\n"
        "Disallow: /tie\nAllow: /tie\n"
    )
    r = snapshot.parse_robots(text)
    assert r.allowed("https://x.org/page")
    assert not r.allowed("https://x.org/private/a")
    assert r.allowed("https://x.org/private/ok")
    assert not r.allowed("https://x.org/private/ok/more")
    assert r.allowed("https://x.org/tie")  # allow wins ties
    assert snapshot.Robots.disallow_all().allowed("https://x.org/robots.txt")
    assert not snapshot.Robots.disallow_all().allowed("https://x.org/a")
    assert snapshot.Robots.allow_all().allowed("https://x.org/a")


def test_parse_robots_ignores_junk() -> None:
    r = snapshot.parse_robots(
        "garbage\nDisallow: /x\nCrawl-delay: soon\nUser-agent: *\nCrawl-delay: soon\nAllow:\n"
    )
    assert r.allowed("https://x.org/x")
    assert r.crawl_delay is None


def test_throttle_spacing() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    t = snapshot.HostThrottle(1.0, clock=lambda: now[0], sleep=sleep)
    t.wait()
    now[0] += 0.25
    t.wait()
    now[0] += 3.0
    t.wait()
    assert slept == [0.75]


def test_stored_headers_for_decoded_payload() -> None:
    out = snapshot.stored_headers(
        [
            ("Content-Type", "text/html"),
            ("Content-Encoding", "gzip"),
            ("Transfer-Encoding", "chunked"),
            ("Content-Length", "10"),
        ],
        42,
    )
    assert out == [
        ("Content-Type", "text/html"),
        ("X-Ditto-Orig-Content-Encoding", "gzip"),
        ("Content-Length", "42"),
    ]


class Site:
    """Mock HTTP site; records every requested URL."""

    def __init__(self, routes: dict[str, Callable[[], httpx.Response]]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(str(request.url))
        assert request.headers["user-agent"] == snapshot.USER_AGENT
        handler = self.routes.get(request.url.raw_path.decode())
        return handler() if handler else httpx.Response(404, text="nope")


def html(body: str) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(200, html=f"<html><body>{body}</body></html>")


def client_for(site: Site, host: str = "x.org") -> Any:
    http = httpx.Client(
        transport=httpx.MockTransport(site),
        headers={"User-Agent": snapshot.USER_AGENT},
        follow_redirects=False,
    )
    return snapshot.PoliteClient(host, 1.0, client=http, sleep=lambda s: None)


def records(path: Path) -> list[tuple[str, str | None, str | None, bytes]]:
    out = []
    with path.open("rb") as fh:
        for rec in ArchiveIterator(fh):
            status = rec.http_headers.get_statuscode() if rec.http_headers else None
            out.append(
                (
                    rec.rec_type,
                    rec.rec_headers.get_header("WARC-Target-URI"),
                    status,
                    rec.content_stream().read(),
                )
            )
    return out


def test_fetch_retries_and_redirects() -> None:
    flaky = iter(
        [httpx.Response(503, headers={"Retry-After": "1"}), httpx.Response(200, text="ok")]
    )
    site = Site(
        {
            "/flaky": lambda: next(flaky),
            "/old": lambda: httpx.Response(301, headers={"Location": "/new"}),
            "/new": html("new"),
            "/away": lambda: httpx.Response(302, headers={"Location": "https://other.org/x"}),
            "/blocked": lambda: httpx.Response(302, headers={"Location": "/private/x"}),
        }
    )
    pc = client_for(site)
    robots = snapshot.parse_robots("User-agent: *\nDisallow: /private\n")
    out = pc.fetch("https://x.org/flaky", robots)
    assert (out.status, len(out.exchanges)) == (200, 1)
    assert pc.requests == 2
    out = pc.fetch("https://x.org/old", robots)
    assert (out.status, out.final_url, len(out.exchanges)) == (200, "https://x.org/new", 2)
    out = pc.fetch("https://x.org/away", robots)
    assert out.error == "off-host redirect to https://other.org/x"
    out = pc.fetch("https://x.org/blocked", robots)
    assert out.error and "disallowed" in out.error
    assert "https://x.org/private/x" not in site.requested


def test_fetch_network_error_gives_up() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(transport=httpx.MockTransport(boom))
    pc = snapshot.PoliteClient("x.org", 1.0, client=http, sleep=lambda s: None)
    out = pc.fetch("https://x.org/a", snapshot.Robots.allow_all())
    assert out.status is None and out.error and "ConnectError" in out.error
    assert pc.requests == snapshot.MAX_ATTEMPTS
    robots, exchanges, info = snapshot.fetch_robots(pc)
    assert robots.mode == "disallow_all" and exchanges == [] and info["mode"] == "disallow_all"


def test_retry_after_parsing() -> None:
    assert snapshot._retry_after(httpx.Response(429, headers={"Retry-After": "7"})) == 7.0
    assert snapshot._retry_after(httpx.Response(429)) == 0.0
    assert snapshot._retry_after(httpx.Response(429, headers={"Retry-After": "soon"})) == 0.0
    later = (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    assert 20 < snapshot._retry_after(httpx.Response(429, headers={"Retry-After": later})) <= 30


def test_robots_status_handling() -> None:
    pc = client_for(Site({"/robots.txt": lambda: httpx.Response(404)}))
    assert snapshot.fetch_robots(pc)[0].mode == "allow_all"
    pc = client_for(Site({"/robots.txt": lambda: httpx.Response(500)}))
    robots, exchanges, info = snapshot.fetch_robots(pc)
    assert robots.mode == "disallow_all" and info["status"] == 500
    assert len(exchanges) == 1  # retried, only the final answer is archived


def test_write_exchange_roundtrip(tmp_path: Path) -> None:
    ex = snapshot.Exchange(
        url="https://x.org/a?b=1",
        method="GET",
        request_headers=[("User-Agent", "u")],
        status=200,
        reason="OK",
        http_version="HTTP/1.1",
        response_headers=[("Content-Type", "text/html"), ("Content-Encoding", "gzip")],
        body=b"<html>hi</html>",
        fetched_at=dt.datetime(2026, 10, 24, tzinfo=dt.UTC),
        ip="1.2.3.4",
    )
    buf = io.BytesIO()
    writer = WARCWriter(buf, gzip=True)
    snapshot.write_warcinfo(writer, "t.warc.gz", {"software": "test"})
    snapshot.write_exchange(writer, ex)
    path = tmp_path / "t.warc.gz"
    path.write_bytes(buf.getvalue())
    recs = records(path)
    assert [r[0] for r in recs] == ["warcinfo", "response", "request"]
    assert recs[1][1:] == ("https://x.org/a?b=1", "200", b"<html>hi</html>")
    with path.open("rb") as fh:
        resp = next(r for r in ArchiveIterator(fh) if r.rec_type == "response")
        assert resp.http_headers.get_header("X-Ditto-Orig-Content-Encoding") == "gzip"
        assert resp.rec_headers.get_header("WARC-IP-Address") == "1.2.3.4"


def make_site() -> Site:
    return Site(
        {
            "/robots.txt": lambda: httpx.Response(200, text="User-agent: *\nDisallow: /private\n"),
            "/a": html("page a"),
            "/b": lambda: httpx.Response(301, headers={"Location": "/c"}),
            "/c": html("page c"),
            "/d": html("page d"),
            "/gone": lambda: httpx.Response(410, text="gone"),
        }
    )


URLS = [
    "https://x.org/a",
    "https://x.org/b",
    "https://x.org/private/p",
    "https://x.org/gone",
    "https://x.org/a",
    "https://x.org/d",
]


def test_snapshot_site_end_to_end(tmp_path: Path) -> None:
    site = make_site()
    res = snapshot.snapshot_site(
        "x.org", URLS, tmp_path, client=client_for(site), log=lambda m: None
    )
    assert res.warc == tmp_path / "x.org.warc.gz" and res.excluded_reason is None
    assert res.counts == {"urls": 5, "ok_html": 3, "non_200": 1, "errors": 0, "disallowed": 1}
    assert "https://x.org/private/p" not in site.requested
    assert site.requested.count("https://x.org/a") == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "fetchlog" / "x.org.jsonl").read_text().splitlines()
    ]
    b = next(r for r in rows if r["url"] == "https://x.org/b")
    assert (b["final_url"], b["status"], b["hops"]) == ("https://x.org/c", 200, 2)
    recs = records(res.warc)
    responses = [(uri, status) for kind, uri, status, _ in recs if kind == "response"]
    assert responses == [
        ("https://x.org/robots.txt", "200"),
        ("https://x.org/a", "200"),
        ("https://x.org/b", "301"),
        ("https://x.org/c", "200"),
        ("https://x.org/gone", "410"),
        ("https://x.org/d", "200"),
    ]
    done = json.loads((tmp_path / "fetchlog" / "x.org.done.json").read_text())
    assert done["counts"] == res.counts and done["robots"]["interval_s"] == 1.0
    # complete snapshot: a rerun makes no request
    again = make_site()
    res2 = snapshot.snapshot_site(
        "x.org", URLS, tmp_path, client=client_for(again), log=lambda m: None
    )
    assert again.requested == [] and res2.counts == res.counts


def test_snapshot_site_resumes_after_interruption(tmp_path: Path) -> None:
    first = make_site()
    snapshot.snapshot_site(
        "x.org", URLS[:2], tmp_path, client=client_for(first), log=lambda m: None
    )
    # simulate a crash after more bytes were written but before the log/summary caught up
    (tmp_path / "fetchlog" / "x.org.done.json").unlink()
    part = tmp_path / "x.org.warc.gz"
    with part.open("ab") as fh:
        fh.write(b"torn record bytes")
    second = make_site()
    res = snapshot.snapshot_site(
        "x.org", URLS, tmp_path, client=client_for(second), log=lambda m: None
    )
    assert "https://x.org/a" not in second.requested and "https://x.org/c" not in second.requested
    assert "https://x.org/d" in second.requested
    assert res.counts["urls"] == 5
    uris = [uri for kind, uri, _s, _b in records(res.warc) if kind == "response"]
    assert uris.count("https://x.org/a") == 1 and uris.count("https://x.org/robots.txt") == 2


def test_snapshot_site_excludes_ai_input_no(tmp_path: Path) -> None:
    site = Site({"/robots.txt": lambda: httpx.Response(200, text=SPORTSDB)})
    res = snapshot.snapshot_site(
        "x.org", ["https://x.org/a"], tmp_path, client=client_for(site), log=lambda m: None
    )
    assert res.warc is None and res.excluded_reason and "ai-input=no" in res.excluded_reason
    assert site.requested == ["https://x.org/robots.txt"]
    assert not (tmp_path / "x.org.warc.gz").exists()
    assert not (tmp_path / "x.org.warc.gz.part").exists()


def test_snapshot_site_excludes_unreachable_robots(tmp_path: Path) -> None:
    site = Site({"/robots.txt": lambda: httpx.Response(503)})
    res = snapshot.snapshot_site(
        "x.org", ["https://x.org/a"], tmp_path, client=client_for(site), log=lambda m: None
    )
    assert res.warc is None and res.excluded_reason and "RFC 9309" in res.excluded_reason
    assert "https://x.org/a" not in site.requested


def test_snapshot_site_rejects_multiple_hosts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one host"):
        snapshot.snapshot_site("x", ["https://a.org/1", "https://b.org/2"], tmp_path)


def test_non_ascii_url_is_archived_percent_encoded(tmp_path: Path) -> None:
    raw = "https://x.org/drink/12756-autodaf\ufffd"  # as listed in the release
    site = Site({"/drink/12756-autodaf%EF%BF%BD": html("drink")})
    res = snapshot.snapshot_site(
        "x.org", [raw], tmp_path, client=client_for(site), log=lambda m: None
    )
    assert res.counts["ok_html"] == 1
    row = json.loads((tmp_path / "fetchlog" / "x.org.jsonl").read_text())
    assert row["url"] == raw
    assert row["final_url"] == "https://x.org/drink/12756-autodaf%EF%BF%BD"
    uris = [uri for kind, uri, _s, _b in records(res.warc) if kind == "response"]
    assert uris[-1] == row["final_url"]


def test_unexpected_error_is_logged_and_retried(tmp_path: Path) -> None:
    calls = {"n": 0}

    def explode() -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return httpx.Response(200, html="<html>ok</html>")

    routes = {"/a": html("a"), "/x": explode}
    res = snapshot.snapshot_site(
        "x.org",
        ["https://x.org/a", "https://x.org/x"],
        tmp_path,
        client=client_for(Site(routes)),
        log=lambda m: None,
    )
    assert res.counts == {"urls": 2, "ok_html": 1, "non_200": 0, "errors": 1, "disallowed": 0}
    assert len(records(res.warc)) == 1 + 2 * 2  # warcinfo, robots pair, /a pair; /x dropped
    (tmp_path / "fetchlog" / "x.org.done.json").unlink()
    again = Site(routes)
    res = snapshot.snapshot_site(
        "x.org",
        ["https://x.org/a", "https://x.org/x"],
        tmp_path,
        client=client_for(again),
        log=lambda m: None,
    )
    assert "https://x.org/a" not in again.requested and "https://x.org/x" in again.requested
    assert res.counts == {"urls": 2, "ok_html": 2, "non_200": 0, "errors": 0, "disallowed": 0}
