from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from lwie_fixtures import D1, D2, D3, DETAIL_HTML, L1, LISTING_HTML, load_script
from warcio.warcwriter import WARCWriter

from ditto.evaluation.datasets.livewebie import load_release
from ditto.schema import FieldsSchema

download: Any = load_script("download")
build_fields: Any = load_script("build_fields")
snap: Any = download.snapshot


def test_page_index_contains() -> None:
    idx = download.PageIndex(DETAIL_HTML.encode())
    assert idx.contains("A Light in the Attic")
    assert idx.contains("  A Light in\nthe Attic ")  # whitespace-normalised
    assert idx.contains("\u00a351.77")
    assert idx.contains("../../media/cache/fe/72/fe72f0532301ec28892ae79a629a293c.jpg")
    assert idx.contains("")
    assert not idx.contains("Tipping the Velvet")
    empty = download.PageIndex(b"")
    assert not empty.contains("x")


def test_page_index_extractor_input_drops_scripts_and_head() -> None:
    page = (
        b"<html><head><meta property='og:image' content='/a.svg'></head><body><h1>T</h1>"
        b"<noscript><img src='/b.svg'></noscript><script>var x = 'hidden text';</script>"
        b"</body></html>"
    )
    clean = download.PageIndex(page)
    raw = download.PageIndex(page, extractor_input=False)
    for value in ("/a.svg", "/b.svg", "hidden text"):
        assert raw.contains(value) and not clean.contains(value)
    assert clean.contains("T")
    assert not download.PageIndex(b"", extractor_input=False).contains("x")


def _warc(path: Path, pages: dict[str, tuple[int, bytes]]) -> None:
    with path.open("wb") as fh:
        writer = WARCWriter(fh, gzip=True)
        for url, (status, body) in pages.items():
            snap.write_exchange(
                writer,
                snap.Exchange(
                    url=url,
                    method="GET",
                    request_headers=[],
                    status=status,
                    reason="",
                    http_version="HTTP/1.1",
                    response_headers=[("Content-Type", "text/html")],
                    body=body,
                    fetched_at=dt.datetime(2026, 10, 24, tzinfo=dt.UTC),
                ),
            )


def _snapshot_dir(release_root: Path) -> Path:
    data_dir = release_root.parents[1]
    final_d1 = D1.replace("index.html", "")
    _warc(
        data_dir / "books.toscrape.com.warc.gz",
        {
            D1: (301, b""),
            final_d1: (200, DETAIL_HTML.encode()),
            D2: (404, b"not here"),
            L1: (200, LISTING_HTML.encode()),
        },
    )
    (data_dir / "fetchlog").mkdir()
    rows = [
        {"url": D1, "final_url": final_d1, "status": 200},
        {"url": D2, "final_url": D2, "status": 404},
        {"url": D3, "status": None, "error": "disallowed by robots.txt"},
        {"url": L1, "final_url": L1, "status": 200},
    ]
    (data_dir / "fetchlog" / "books.toscrape.com.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return data_dir


def test_snapshot_pages_and_label_presence(release_root: Path) -> None:
    data_dir = _snapshot_dir(release_root)
    pages = download.snapshot_pages(
        data_dir / "books.toscrape.com.warc.gz", data_dir / "fetchlog" / "books.toscrape.com.jsonl"
    )
    assert set(pages) == {D1, L1}
    rel = load_release(release_root)
    pres = download.label_presence(rel, "books.toscrape.com", pages)
    # D1: title + image found; L1: both links found, 1 of 2 titles (the second is truncated
    # on the page and its full form only in a title attribute that differs in apostrophe)
    assert pres["by_field"]["title"] == 1.0
    assert pres["by_field"]["book_image"] == 1.0
    assert pres["by_field"]["book_link"] == 1.0
    assert pres["by_field"]["title_list"] == 0.5
    assert (pres["values"], pres["found"]) == (6, 5)
    assert pres["labeled_pages"] == 2
    assert pres["labeled_pages_not_in_snapshot"] == 2  # D2 (404) and D3 (disallowed)
    assert pres["by_group"] == {"g_000": 1.0, "g_002": 0.75}


def test_site_entry_included_and_excluded(release_root: Path, tmp_path: Path) -> None:
    data_dir = _snapshot_dir(release_root)
    rel = load_release(release_root)
    result = snap.SiteResult(
        website="books.toscrape.com",
        host="books.toscrape.com",
        warc=data_dir / "books.toscrape.com.warc.gz",
        robots={"status": 404},
        counts={"urls": 4},
        excluded_reason=None,
        started_at="a",
        finished_at="b",
        requests=5,
    )
    entry = download.site_entry(rel, "books.toscrape.com", result, data_dir, 0.5)
    assert entry["included"] is True
    assert entry["pages_200"] == 2 and entry["domain"] == "book"
    assert entry["groups"] == ["g_000", "g_002"]
    assert len(entry["warc_sha256"]) == 64
    strict = download.site_entry(rel, "books.toscrape.com", result, data_dir, 0.95)
    assert strict["included"] is False and "threshold 0.95" in strict["excluded_reason"]
    gone = snap.SiteResult(
        website="quotes.toscrape.com",
        host="quotes.toscrape.com",
        warc=None,
        robots={},
        counts={},
        excluded_reason="robots.txt unreachable",
        started_at="a",
        finished_at="b",
        requests=1,
    )
    entry = download.site_entry(rel, "quotes.toscrape.com", gone, data_dir, 0.5)
    assert entry == {**entry, "included": False, "excluded_reason": "robots.txt unreachable"}


def test_tree_digest_and_release_counts(release_root: Path) -> None:
    d1 = download.tree_digest(release_root)
    (release_root / ".DS_Store").write_bytes(b"junk")
    assert download.tree_digest(release_root) == d1
    (release_root / "extra.txt").write_text("x")
    assert download.tree_digest(release_root) != d1
    with pytest.raises(SystemExit, match="websites"):
        download.release_counts(load_release(release_root))


def test_build_fields_render(release_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "fields"
    paths = build_fields.build(release_root, out)
    assert [p.name for p in paths] == ["books.toscrape.com.yaml", "quotes.toscrape.com.yaml"]
    text = paths[0].read_text()
    assert text.startswith("# Prompted-mode fields for books.toscrape.com from LiveWeb-IE")
    assert "non-commercial research" in text
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    schema = FieldsSchema.load(paths[0])
    assert schema.names() == ["title", "book_image", "title_list", "book_link"]
    assert schema.get("book_link") is not None
    data = yaml.safe_load(text)["fields"]
    assert data["book_image"]["type_check"] == "url"
    assert "type_check" not in data["title"]
    assert data["title_list"]["type"] == "list"


def test_build_fields_main(release_root: Path, tmp_path: Path, capsys: Any) -> None:
    assert build_fields.main(["--release", str(release_root), "--out", str(tmp_path / "f")]) == 0
    assert "quotes.toscrape.com.yaml" in capsys.readouterr().out


def _audited_snapshot(tmp_path: Path, *, gap_s: int, extra: str | None = None) -> Path:
    """A WARC with robots.txt (Disallow: /private) and pages ``gap_s`` seconds apart."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    warc = tmp_path / "s.warc.gz"
    base = dt.datetime(2026, 10, 24, 9, 0, 0, tzinfo=dt.UTC)
    urls = ["https://h.example/robots.txt", "https://h.example/a", "https://h.example/b"]
    if extra:
        urls.append(extra)
    bodies = {
        u: (b"User-agent: *\nDisallow: /private\n" if "robots" in u else u.encode()) for u in urls
    }
    with warc.open("wb") as fh:
        writer = WARCWriter(fh, gzip=True)
        for n, url in enumerate(urls):
            snap.write_exchange(
                writer,
                snap.Exchange(
                    url=url,
                    method="GET",
                    request_headers=[],
                    status=200,
                    reason="OK",
                    http_version="HTTP/1.1",
                    response_headers=[("Content-Type", "text/html")],
                    body=bodies[url],
                    fetched_at=base + dt.timedelta(seconds=gap_s * n),
                ),
            )
    rows = [
        {"url": u, "final_url": u, "status": 200, "sha256": hashlib.sha256(bodies[u]).hexdigest()}
        for u in urls[1:]
    ]
    (tmp_path / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return warc


def test_audit_snapshot_checks_robots_rate_settlement_and_hashes(tmp_path: Path) -> None:
    urls = ["https://h.example/a", "https://h.example/b"]
    warc = _audited_snapshot(tmp_path, gap_s=1)
    audit = download.audit_snapshot(warc, tmp_path / "s.jsonl", urls, 1.0)
    assert audit["ok"] and audit["requests"] == 3 and audit["sessions"] == 1
    assert audit["first_request"] == "2026-10-24T09:00:00Z"
    assert audit["response_statuses"] == {"200": 3} and audit["max_requests_in_any_60s"] == 3
    slow = download.audit_snapshot(warc, tmp_path / "s.jsonl", urls, 120.0)
    assert not slow["throttle_ok"] and slow["nominal_max_in_60s"] == 0  # 3 in 2 s, 120 s delay
    missing = download.audit_snapshot(warc, tmp_path / "s.jsonl", [*urls, "https://h.example/c"], 1)
    assert missing["urls_unsettled"] == 1 and not missing["ok"]
    bad = _audited_snapshot(tmp_path / "x", gap_s=1, extra="https://h.example/private/1")
    audit = download.audit_snapshot(bad, tmp_path / "x" / "s.jsonl", urls, 1.0)
    assert audit["requests_disallowed_by_robots"] == 1 and not audit["robots_ok"]
    log = tmp_path / "x" / "s.jsonl"
    log.write_text(log.read_text().replace('"sha256": "', '"sha256": "0'))
    assert download.audit_snapshot(bad, log, urls, 1.0)["payload_mismatches"] == 3


def test_build_manifest_status_and_site_notes(
    release_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"websites": 3}
    good = {"included": True, "audit": {"ok": True}, "fetched_from": "a", "fetched_to": "c"}
    sites = {"x": good, "y": {**good, "fetched_from": "0"}, "z": {"included": False}}
    manifest = download.build_manifest({}, counts, sites, release_root, 0.5)
    assert manifest["status"] == "partial" and manifest["snapshot"]["complete"] is True
    assert manifest["status_note"].startswith("Final. 'partial' because 1 of the 3 websites")
    assert manifest["summary"].startswith("2 of 3 LiveWeb-IE websites usable (x, y); excluded: z")
    assert manifest["snapshot"]["fetched_from"] == "0" and manifest["snapshot"]["fetched_to"] == "c"
    broken = {**sites, "y": {**good, "audit": {"ok": False}}}
    assert download.build_manifest({}, counts, broken, release_root, 0.5)["status_note"] == (
        "Not final: the snapshot audit fails for y."
    )
    assert (
        download.build_manifest({}, {"websites": 1}, {"z": {}}, release_root, 0.5)["status"]
        == "skipped"
    )
    data_dir = _snapshot_dir(release_root)
    monkeypatch.setitem(download.SITE_NOTES, "quotes.toscrape.com", {"diagnosis": "checked"})
    gone = snap.SiteResult(
        website="quotes.toscrape.com",
        host="quotes.toscrape.com",
        warc=None,
        robots={},
        counts={},
        excluded_reason="robots.txt unreachable",
        started_at="2026-10-27T09:00:00Z",
        finished_at="b",
        requests=4,
    )
    entry = download.site_entry(
        load_release(release_root), "quotes.toscrape.com", gone, data_dir, 0.5
    )
    assert entry["diagnosis"] == "checked" and entry["checked_at"] == "2026-10-27T09:00:00Z"


def test_recorded_results_offline(tmp_path: Path) -> None:
    (tmp_path / "fetchlog").mkdir()
    (tmp_path / "w.warc.gz").write_bytes(b"")
    done = {
        "website": "w",
        "host": "w",
        "warc": "/elsewhere/w.warc.gz",
        "robots": {},
        "counts": {},
        "excluded_reason": None,
        "started_at": "a",
        "finished_at": "b",
        "requests": 3,
    }
    (tmp_path / "fetchlog" / "w.done.json").write_text(json.dumps(done))
    results = download.recorded_results(["w"], tmp_path)
    assert results["w"].warc == tmp_path / "w.warc.gz" and results["w"].requests == 3
    assert snap.load_result("nope", tmp_path) is None
    with pytest.raises(SystemExit, match="no finished snapshot"):
        download.recorded_results(["w", "nope"], tmp_path)
