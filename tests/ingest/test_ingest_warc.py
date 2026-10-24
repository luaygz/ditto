from __future__ import annotations

import gzip
import io
from pathlib import Path

import pytest
from ingest_helpers import Rec, write_tree, write_warc
from warcio.archiveiterator import ArchiveIterator

from ditto.ingest import PageStore, ingest
from ditto.ingest.warc import read_warc_payload, scan_warc, warc_timestamp

HOST = "https://shop.example.com"
A_V0 = b"<html><body><h1>A version 0</h1></body></html>"
A_V1 = b"<html><body><h1>A version 1</h1></body></html>"
A_V2 = b"<html><body><h1>A version 2</h1></body></html>"
B_LATIN = "<html><body><p>Café crème, 12 €</p></body></html>".encode("cp1252")
GZ_BODY = b"<html><body><p>was gzip-encoded on the wire</p></body></html>"
CHUNKED_BODY = b"<html><body><p>was chunked</p></body></html>"


def chunked(data: bytes, size: int = 16) -> bytes:
    parts = [
        f"{len(data[i : i + size]):x}\r\n".encode() + data[i : i + size] + b"\r\n"
        for i in range(0, len(data), size)
    ]
    return b"".join(parts) + b"0\r\n\r\n"


def first_file() -> list[Rec]:
    html = [("Content-Type", "text/html")]
    return [
        Rec("", kind="warcinfo"),
        Rec(f"{HOST}/a", kind="request"),
        Rec(f"{HOST}/a", A_V1, date="2024-01-01T00:00:00Z"),
        Rec(f"{HOST}/b", B_LATIN, headers=[("Content-Type", "text/html; charset=ISO-8859-1")]),
        Rec(f"{HOST}/logo.png", b"\x89PNG\r\n\x1a\n....", headers=[("Content-Type", "image/png")]),
        Rec(f"{HOST}/a", kind="revisit", date="2024-03-01T00:00:00Z"),
        Rec(f"{HOST}/missing", b"<html><body>Not found</body></html>", status="404 Not Found"),
        Rec(f"{HOST}/c", B_LATIN, headers=[("Content-Type", "text/html; charset=ISO-8859-1")]),
        Rec(
            f"{HOST}/gz",
            gzip.compress(GZ_BODY),
            headers=[*html, ("Content-Encoding", "gzip")],
        ),
        Rec(
            f"{HOST}/chunked",
            chunked(CHUNKED_BODY),
            headers=[("Content-Type", "application/xhtml+xml"), ("Transfer-Encoding", "chunked")],
        ),
    ]


def second_file() -> list[Rec]:
    return [
        Rec(f"{HOST}/a", A_V2, date="2024-06-01T12:30:00.5Z"),
        Rec(f"{HOST}/d", b"<!DOCTYPE html><html><body>no type</body></html>", headers=[]),
        Rec(f"{HOST}/e", b'{"json": true}', headers=[]),
        Rec(f"{HOST}/a", A_V0, date="2023-01-01T00:00:00Z"),
    ]


EXPECTED_URLS = [f"{HOST}/{p}" for p in ("a", "b", "chunked", "d", "gz")]


@pytest.fixture
def warcs(tmp_path: Path) -> tuple[Path, Path]:
    folder = tmp_path / "crawl"
    folder.mkdir()
    return (
        write_warc(folder / "one.warc.gz", first_file()),
        write_warc(folder / "two.warc", second_file()),
    )


def check_store(store: PageStore, warcs: tuple[Path, Path]) -> None:
    assert [store.url(pid) for pid in store.ids()] == EXPECTED_URLS
    a = store.record(0)
    assert a.source == str(warcs[1].resolve())
    assert a.fetched_at == "2024-06-01T12:30:00.5Z"
    assert store.read_bytes(0) == A_V2

    b = store.record(1)
    assert b.source == str(warcs[0].resolve())
    assert b.encoding == "cp1252"
    assert "Café crème, 12 €" in store.read_html(1)
    assert store.read_bytes(2) == CHUNKED_BODY
    assert store.read_bytes(4) == GZ_BODY
    assert store.read_html(3) == "<!DOCTYPE html><html><body>no type</body></html>"


def test_warc_files_latest_capture_and_skips(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    result = ingest(list(warcs), tmp_path / "run", workers=2)
    assert result.kind == "warc"
    assert result.workers == 2
    assert result.n_pages == 5
    assert result.site == "shop.example.com"
    stats = result.stats
    assert stats["warc_files"] == 2
    assert stats["warc_records"] == 14
    assert stats["superseded_captures"] == 2  # /a v0 and v1 lose to v2
    assert stats["duplicates"] == 1  # /c has the same payload as /b
    assert stats["skipped_non_html"] == 2  # logo.png, /e (no Content-Type, not HTML)
    assert stats["skipped_revisit"] == 1
    assert stats["skipped_status"] == 1
    assert stats["errors"] == 0
    check_store(PageStore.open(result.run), warcs)
    assert not result.run.blobs_dir.exists()


def test_warc_directory_input(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    result = ingest(warcs[0].parent, tmp_path / "run", workers=1)
    assert result.kind == "warc"
    check_store(PageStore.open(result.run), warcs)
    assert result.run.read_meta()["source"] == [str(warcs[0].parent.resolve())]


@pytest.mark.parametrize("index", [0, 1])
def test_offsets_address_single_records(
    warcs: tuple[Path, Path], tmp_path: Path, index: int
) -> None:
    path = warcs[index]
    store = PageStore.open(ingest(path, tmp_path / "run").run)
    assert len(store) >= 2
    for rec in store.records():
        assert rec.offset is not None and rec.length is not None
        with open(path, "rb") as fh:
            fh.seek(rec.offset)
            raw = fh.read(rec.length)
        record = next(iter(ArchiveIterator(io.BytesIO(raw))))
        assert record.rec_headers.get_header("WARC-Target-URI") == rec.url
        assert record.rec_type == "response"
        payload = read_warc_payload(rec.source, rec.offset, rec.length)
        assert payload == store.read_bytes(rec.page_id)
        assert rec.sha256 == __import__("hashlib").sha256(payload).hexdigest()


def test_single_gz_file(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    result = ingest(warcs[0], tmp_path / "run")
    store = PageStore.open(result.run)
    assert store.read_bytes(store.lookup(f"{HOST}/a") or 0) == A_V1
    assert result.site == "shop.example.com"
    assert result.stats["superseded_captures"] == 0


def test_corrupt_warc_is_reported_not_fatal(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    bad = tmp_path / "crawl" / "bad.warc.gz"
    bad.write_bytes(b"\x00\x01\x02 not an archive \xff\xfe" * 20)
    result = ingest(warcs[0].parent, tmp_path / "run")
    assert result.stats["errors"] == 1
    assert result.n_pages == 5
    assert "bad.warc.gz" in result.errors[0]
    assert result.run.read_meta()["ingest"]["error_samples"] == result.errors


def test_truncated_warc_keeps_complete_records(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    raw = warcs[1].read_bytes()
    cut = raw.index(b"WARC/1.0", raw.index(b"WARC-Target-URI: " + HOST.encode() + b"/e")) + 40
    truncated = tmp_path / "cut.warc"
    truncated.write_bytes(raw[:cut])  # ends inside the header of the last record
    result = ingest(truncated, tmp_path / "run")
    store = PageStore.open(result.run)
    assert [store.url(pid) for pid in store.ids()] == [f"{HOST}/a", f"{HOST}/d"]
    assert store.read_bytes(0) == A_V2


def test_warc_input_errors(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest"):
        ingest(warcs[0], tmp_path / "run", manifest=tmp_path / "urls.tsv")
    site = write_tree(tmp_path / "site", {"a.html": b"<html><body>x</body></html>"})
    with pytest.raises(ValueError, match="not a mix"):
        ingest([site, warcs[0]], tmp_path / "run")


def test_directory_with_html_and_warc_is_html(warcs: tuple[Path, Path], tmp_path: Path) -> None:
    folder = warcs[0].parent
    (folder / "index.html").write_bytes(b"<html><body>x</body></html>")
    result = ingest(folder, tmp_path / "run")
    assert result.kind == "directory"
    assert result.n_pages == 1


def test_scan_warc_counts_records(warcs: tuple[Path, Path]) -> None:
    scan = scan_warc(str(warcs[0]))
    assert scan.counts["records"] == 10
    assert scan.counts["other"] == 2  # warcinfo + request
    assert [c.url.rsplit("/", 1)[1] for c in scan.captures] == ["a", "b", "c", "gz", "chunked"]


def test_warc_timestamp() -> None:
    assert warc_timestamp("2024-01-01T00:00:00Z") == warc_timestamp("2024-01-01T00:00:00")
    assert warc_timestamp("2024-01-01T00:00:00.5Z") > warc_timestamp("2024-01-01T00:00:00Z")
    assert warc_timestamp(None) == float("-inf")
    assert warc_timestamp("yesterday") == float("-inf")
