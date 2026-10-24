from __future__ import annotations

import codecs
import json
import os
import sqlite3
from pathlib import Path

import pytest
from ingest_helpers import write_tree

from ditto.ingest import PageStore, ingest, read_html
from ditto.run import RunDir

INDEX = "<!DOCTYPE html><html><head><title>Home</title></head><body><h1>Café</h1></body></html>"
PRODUCT = (
    '<html><body class="product"><h1>Widget {n}</h1>'
    '<span class="price">€{n}.99</span></body></html>'
)
LATIN = (
    '<html><head><meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">'
    "</head><body><p>Crème brûlée \u2019s</p></body></html>"
)
BOM = '<html><head><meta charset="iso-8859-1"></head><body><p>naïve — ok</p></body></html>'


def site_files() -> dict[str, bytes]:
    return {
        "index.html": INDEX.encode(),
        "about.htm": b"<html><body><h1>About us</h1></body></html>",
        "products/p1.html": PRODUCT.format(n=1).encode(),
        "products/p2.html": PRODUCT.format(n=2).encode(),
        "products/deep/p3.xhtml": PRODUCT.format(n=3).encode(),
        "latin.html": LATIN.encode("cp1252"),
        "bom.html": codecs.BOM_UTF8 + BOM.encode(),
        "dup.html": PRODUCT.format(n=1).encode(),  # byte-identical to products/p1.html
        "page": b"\n<!doctype html><html><body><p>extensionless</p></body></html>",
        "README": b"This is not a web page.\n",
        "style.css": b"body { color: red }",
        "logo.png": b"\x89PNG\r\n\x1a\n",
        "empty.html": b"",
        ".hidden/secret.html": b"<html><body>hidden</body></html>",
    }


@pytest.fixture
def site(tmp_path: Path) -> Path:
    return write_tree(tmp_path / "mysite", site_files())


def test_directory_without_manifest(site: Path, tmp_path: Path) -> None:
    result = ingest(site, tmp_path / "run")
    store = PageStore.open(result.run)

    urls = [store.url(pid) for pid in store.ids()]
    assert urls == [
        "about.htm",
        "bom.html",
        "dup.html",
        "index.html",
        "latin.html",
        "page",
        "products/deep/p3.xhtml",
        "products/p2.html",
    ]
    assert result.n_pages == len(store) == 8
    assert store.ids() == list(range(8))
    assert [rec.page_id for rec in store.records()] == list(range(8))
    assert result.stats["duplicates"] == 1  # products/p1.html == dup.html; smallest URL kept
    assert result.stats["skipped_non_html"] == 3  # README, style.css, logo.png
    assert result.stats["skipped_empty"] == 1
    assert result.stats["superseded_captures"] == 0
    assert result.kind == "directory"
    assert result.site == "mysite"
    assert "8 pages" in str(result)

    meta = result.run.read_meta()
    assert meta["site"] == "mysite" == result.run.site
    assert meta["n_pages"] == 8
    assert meta["source"] == [str(site.resolve())]
    assert meta["ingest"]["duplicates"] == 1
    assert meta["ingest"]["manifest"] is None
    assert len(meta["ingest"]["content_digest"]) == 64
    timings = json.loads(result.run.timings_json.read_text())
    assert timings["ingest"]["pages"] == 8


def test_blobs_are_content_addressed_hardlinks(site: Path, tmp_path: Path) -> None:
    run = ingest(site, tmp_path / "run").run
    store = PageStore.open(run)
    rec = store.record(store.lookup("index.html") or 0)
    assert rec.url == "index.html"
    blob = run.blobs_dir / rec.sha256[:2] / rec.sha256
    assert rec.source == str(blob.resolve())
    assert rec.offset is None
    assert rec.length == len(INDEX.encode())
    assert os.path.samefile(blob, site / "index.html")  # hardlinked, not copied
    assert store.read_bytes(rec.page_id) == (site / "index.html").read_bytes()
    assert rec.fetched_at is not None and rec.fetched_at.endswith("Z")
    stored = {p.name for p in run.blobs_dir.glob("*/*")}
    assert stored == {r.sha256 for r in store.records()}

    with sqlite3.connect(run.pages_db) as con:
        columns = [row[1] for row in con.execute("PRAGMA table_info(pages)")]
        indexes = [row[1] for row in con.execute("PRAGMA index_list(pages)")]
        source = con.execute("SELECT source FROM pages WHERE url = 'index.html'").fetchone()[0]
    assert columns == [
        "page_id",
        "url",
        "source",
        "sha256",
        "fetched_at",
        "offset",
        "length",
        "encoding",
    ]
    assert indexes == ["pages_url"]
    assert source == f"blobs/{rec.sha256[:2]}/{rec.sha256}"  # relative: the run dir can move


def test_encodings_round_trip(site: Path, tmp_path: Path) -> None:
    run = ingest(site, tmp_path / "run").run
    store = PageStore.open(run)

    latin = store.record(store.lookup("latin.html") or 0)
    assert latin.encoding == "cp1252"
    assert "Crème brûlée \u2019s" in store.read_html(latin.page_id)

    bom = store.record(store.lookup("bom.html") or 0)
    assert bom.encoding == "utf-8-sig"
    html = store.read_html(bom.page_id)
    assert html.startswith("<html>")
    assert "naïve — ok" in html

    index = store.record(store.lookup("index.html") or 0)
    assert index.encoding == "utf-8"
    assert "<h1>Café</h1>" in read_html(run.root, index.page_id)
    assert read_html(str(run.root), index.page_id) == store.read_html(index.page_id)


def test_urls_tsv_is_detected_at_root(site: Path, tmp_path: Path) -> None:
    (site / "urls.tsv").write_text(
        "path\turl\n"
        "# comment line\n"
        "index.html\thttps://www.example.com/\n"
        "./products/p2.html\thttps://www.example.com/p/2\n"
        "products/deep/p3.xhtml\thttps://www.example.com/p/3\n"
        "latin.html\thttps://shop.example.org/latin\n"
        "missing.html\thttps://www.example.com/missing\n"
        "no tab here\n"
        "\n"
    )
    result = ingest(site, tmp_path / "run")
    store = PageStore.open(result.run)
    urls = {store.url(pid) for pid in store.ids()}
    assert "https://www.example.com/" in urls
    assert "https://www.example.com/p/2" in urls
    assert "about.htm" in urls  # not in the manifest: falls back to the relative path
    assert result.site == "www.example.com"
    assert result.stats["skipped_non_html"] == 3  # urls.tsv itself is not counted
    ingest_meta = result.run.read_meta()["ingest"]
    assert ingest_meta["manifest"] == str((site / "urls.tsv").resolve())
    assert ingest_meta["manifest_entries"] == 5
    assert ingest_meta["manifest_missing"] == 1
    assert ingest_meta["manifest_invalid_lines"] == 1
    assert [store.url(pid) for pid in store.ids()] == sorted(urls)


def test_explicit_manifest_outside_directory(tmp_path: Path) -> None:
    site = write_tree(
        tmp_path / "pages",
        {
            "a.html": b"<html><body>A</body></html>",
            "view.php": b"<html><body>B</body></html>",  # only taken because it is listed
            "other.php": b"<html><body>C</body></html>",
        },
    )
    manifest = tmp_path / "list.tsv"
    manifest.write_text(
        f"{site / 'a.html'}\thttps://ex.com/a\nview.php\thttps://ex.com/view?id=2\n"
    )
    result = ingest(site, tmp_path / "run", manifest=manifest, site="custom")
    store = PageStore.open(result.run)
    assert [store.url(pid) for pid in store.ids()] == [
        "https://ex.com/a",
        "https://ex.com/view?id=2",
    ]
    assert result.site == "custom"
    assert result.stats["skipped_non_html"] == 1


def test_same_url_keeps_latest_file(tmp_path: Path) -> None:
    site = write_tree(
        tmp_path / "s",
        {
            "old/item.html": b"<html><body>old</body></html>",
            "new/item.html": b"<html><body>new</body></html>",
        },
    )
    os.utime(site / "old/item.html", (1_000_000, 1_000_000))
    os.utime(site / "new/item.html", (2_000_000, 2_000_000))
    (site / "urls.tsv").write_text(
        "old/item.html\thttps://ex.com/item\nnew/item.html\thttps://ex.com/item\n"
    )
    result = ingest(site, tmp_path / "run")
    store = PageStore.open(result.run)
    assert len(store) == 1
    assert result.stats["superseded_captures"] == 1
    assert b"new" in store.read_bytes(0)
    assert store.record(0).fetched_at == "1970-01-24T03:33:20Z"
    assert len(list(result.run.blobs_dir.glob("*/*"))) == 1  # the superseded blob is pruned


def test_reingest_replaces_previous_output(site: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    first = ingest(site, out)
    old = PageStore.open(first.run)
    about_sha = old.record(old.lookup("about.htm") or 0).sha256
    first_digest = first.run.read_meta()["ingest"]["content_digest"]
    assert read_html(out, 0).startswith("<html><body><h1>About us")

    index = Path(old.record(old.lookup("index.html") or 0).source)
    index.unlink()
    index.write_bytes(b"stale")  # a damaged blob from the previous ingest gets replaced
    (site / "about.htm").unlink()
    (site / "zzz.html").write_bytes(b"<html><body>new page</body></html>")
    second = ingest(site, out)
    store = PageStore.open(second.run)
    assert len(store) == 8
    assert store.lookup("about.htm") is None
    assert store.url(len(store) - 1) == "zzz.html"
    assert not (out / "blobs" / about_sha[:2] / about_sha).exists()
    assert read_html(out, 0).startswith('<html><head><meta charset="iso-8859-1">')  # bom.html
    assert store.read_bytes(store.lookup("index.html") or 0) == INDEX.encode()
    assert second.run.read_meta()["ingest"]["content_digest"] != first_digest
    assert not list(out.glob("pages.sqlite*.tmp"))
    blobs = {p.name for p in (out / "blobs").glob("*/*")}
    assert blobs == {rec.sha256 for rec in store.records()}


def test_run_dir_inside_source_is_not_ingested(site: Path) -> None:
    out = site / "run"
    first = ingest(site, out)
    second = ingest(site, out)
    assert first.n_pages == second.n_pages == 8
    assert first.stats == second.stats


def test_many_files_with_worker_processes(tmp_path: Path) -> None:
    files = {
        f"d{i % 7}/page{i:04d}.html": f"<html><body><p>page {i}</p></body></html>".encode()
        for i in range(600)
    }
    site = write_tree(tmp_path / "big", files)
    result = ingest(site, tmp_path / "run", workers=2)
    assert result.workers == 2
    assert result.n_pages == 600
    store = PageStore.open(result.run)
    urls = [rec.url for rec in store.records()]
    assert urls == sorted(files)
    assert store.read_html(5) == files[urls[5]].decode()


def test_input_errors(site: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "nope", tmp_path / "run")
    with pytest.raises(ValueError, match="unsupported input file"):
        ingest(site / "index.html", tmp_path / "run")
    with pytest.raises(ValueError, match="no input"):
        ingest([], tmp_path / "run")
    other = write_tree(tmp_path / "other", {"x.html": b"<html></html>"})
    with pytest.raises(ValueError, match="not a mix"):
        ingest([site, other], tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="manifest"):
        ingest(site, tmp_path / "run", manifest=tmp_path / "missing.tsv")


def test_page_store_errors(site: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ditto ingest"):
        PageStore.open(tmp_path / "empty")
    store = PageStore.open(RunDir(ingest(site, tmp_path / "run").run.root))
    with pytest.raises(KeyError):
        store.record(len(store))
    with pytest.raises(KeyError):
        store.record(-1)
    assert store.lookup("https://nowhere.example/") is None


def test_blobs_are_copied_when_hardlinks_fail(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(src: str, dst: str) -> None:
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr("ditto.ingest.directory.os.link", refuse)
    run = ingest(site, tmp_path / "run", workers=1).run
    store = PageStore.open(run)
    rec = store.record(store.lookup("index.html") or 0)
    assert not os.path.samefile(rec.source, site / "index.html")
    assert store.read_bytes(rec.page_id) == (site / "index.html").read_bytes()
    assert not list(run.blobs_dir.glob("*/*.tmp"))


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs file permissions")
def test_unreadable_file_is_reported(site: Path, tmp_path: Path) -> None:
    locked = site / "products" / "p2.html"
    locked.chmod(0)
    try:
        result = ingest(site, tmp_path / "run", workers=1)
    finally:
        locked.chmod(0o644)
    assert result.stats["errors"] == 1
    assert result.n_pages == 7
    assert result.errors[0].startswith("products/p2.html: ")
