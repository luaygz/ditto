from __future__ import annotations

import datetime as dt
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
