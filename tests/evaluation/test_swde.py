"""SWDE adapter: ground-truth parser (on lines copied from the real files), split, fields,
layout check, eval adapter, and the download script's verification/resume logic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from ditto.evaluation.datasets import swde
from ditto.evaluation.datasets.base import DatasetUnavailable, Selection
from ditto.schema import FieldsSchema

REPO = Path(__file__).resolve().parents[2]
SWDE_EVAL_DIR = REPO / "eval" / "datasets" / "swde"
REAL_DATA = REPO / "data" / "swde"

# Copied verbatim from the real ground-truth files (UTF-8 BOM and CRLF line ends included).
REAL_HEADER = "\ufeffcamera\tamazon\tprice\r\n"
REAL_STATS = "1767\t1767\t1767\t723\r\n"
REAL_PRICE = "0000\t1\t$129.00\r\n"  # camera-amazon-price.txt
REAL_NULL = "0000\t0\t<NULL>\r\n"  # auto-aol-engine.txt
REAL_TWO = "0049\t2\tStanley Donen\tGene Kelly\r\n"  # movie-allmovie-director.txt
REAL_THREE = "0000\t3\tHoward Zinn\tPaul Buhle\tMike Konopacki\r\n"  # book-abebooks-author.txt
REAL_ENTITY = "0695\t1\t2010 Chrysler Town &amp; Country\r\n"  # auto-aol-model.txt
REAL_NBSP = "0000\t1\tRay Allen&nbsp;&nbsp;#20&nbsp;&nbsp;SG\r\n"  # nbaplayer-espn-name.txt
REAL_PAGE_HEAD = b'\xef\xbb\xbf<base href="http://www.amazon.com/gp/product/tech-data/B0035FZJKI"/>'


# --------------------------------------------------------------------------- parser


def test_parse_line_single_value() -> None:
    assert swde.parse_groundtruth_line(REAL_PRICE) == ("0000", ["$129.00"])


def test_parse_line_count_zero_is_empty() -> None:
    assert swde.parse_groundtruth_line(REAL_NULL) == ("0000", [])
    assert swde.parse_groundtruth_line("0001\t0") == ("0001", [])


def test_parse_line_multiple_values() -> None:
    assert swde.parse_groundtruth_line(REAL_TWO) == ("0049", ["Stanley Donen", "Gene Kelly"])
    assert swde.parse_groundtruth_line(REAL_THREE)[1] == [
        "Howard Zinn",
        "Paul Buhle",
        "Mike Konopacki",
    ]


def test_parse_line_keeps_raw_entities_and_decode_value() -> None:
    _, values = swde.parse_groundtruth_line(REAL_ENTITY)
    assert values == ["2010 Chrysler Town &amp; Country"]
    assert swde.decode_value(values[0]) == "2010 Chrysler Town & Country"
    _, values = swde.parse_groundtruth_line(REAL_NBSP)
    assert swde.decode_value(values[0]) == "Ray Allen\xa0\xa0#20\xa0\xa0SG"


@pytest.mark.parametrize(
    "line",
    [
        "0000",  # no count
        "00a0\t1\tx",  # bad id
        "0000\tone\tx",  # bad count
        "0000\t2\tonly one",  # count mismatch
        "0000\t1\ta\tb",  # extra value
        "0000\t0\tvalue",  # count 0 with a value
        "0000\t1\t<NULL>",  # NULL counted as a value
        "0000\t-1",  # negative count
    ],
)
def test_parse_line_rejects_malformed(line: str) -> None:
    with pytest.raises(swde.GroundTruthError):
        swde.parse_groundtruth_line(line)


def test_parse_file_with_bom_crlf_and_stats() -> None:
    text = "\ufeffmovie\tallmovie\tdirector\r\n2\t1\t2\t2\r\n" + REAL_NULL + REAL_TWO
    gt = swde.parse_groundtruth(text)
    assert (gt.vertical, gt.site, gt.attribute) == ("movie", "allmovie", "director")
    assert gt.stats == swde.GroundTruthStats(2, 1, 2, 2)
    assert gt.values == {"0000": [], "0049": ["Stanley Donen", "Gene Kelly"]}


def test_parse_file_real_header_and_stats_line() -> None:
    # The real stats line describes all 1767 pages, so a one-line excerpt must disagree with it.
    with pytest.raises(swde.GroundTruthError, match="disagrees"):
        swde.parse_groundtruth(REAL_HEADER + REAL_STATS + REAL_PRICE)
    gt = swde.parse_groundtruth(REAL_HEADER + "1\t1\t1\t1\r\n" + REAL_PRICE)
    assert gt.values == {"0000": ["$129.00"]}


@pytest.mark.parametrize(
    "text",
    [
        "camera\tamazon\tprice\n",  # no stats line
        "camera\tamazon\n1\t1\t1\t1\n0000\t1\tx\n",  # short header
        "camera\tamazon\tprice\n1\t1\t1\n0000\t1\tx\n",  # 3 stats
        "camera\tamazon\tprice\n2\t2\t2\t1\n0000\t1\tx\n0000\t1\tx\n",  # duplicate id
    ],
)
def test_parse_file_rejects_malformed(text: str) -> None:
    with pytest.raises(swde.GroundTruthError):
        swde.parse_groundtruth(text)


def test_read_groundtruth_reports_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("x\ty\tz\n1\t1\t1\t1\n0000\t3\ta\n", encoding="utf-8")
    with pytest.raises(swde.GroundTruthError, match=r"bad\.txt"):
        swde.read_groundtruth(path)


def test_source_url_from_real_page_head() -> None:
    assert swde.source_url(REAL_PAGE_HEAD) == (
        "http://www.amazon.com/gp/product/tech-data/B0035FZJKI"
    )
    assert swde.source_url(b"<html><head></head></html>") is None
    assert swde.source_url(b'\r\n <BASE HREF="http://a.example/x?a=1&amp;b=2">') == (
        "http://a.example/x?a=1&b=2"
    )


# --------------------------------------------------------------------------- split and fields


def test_split_rule() -> None:
    dev = swde.sites("dev")
    test = swde.sites("test")
    assert len(dev) == 16 and len(test) == 64 and len(swde.sites("all")) == 80
    assert not {r.name for r in dev} & {r.name for r in test}
    for vertical in swde.VERTICALS:
        names = sorted(swde.SITES[vertical])
        assert len(names) == 10
        assert [r.site for r in dev if r.vertical == vertical] == names[:2]
    assert swde.split_of(swde.SiteRef("camera", "amazon")) == "dev"
    assert swde.split_of(swde.SiteRef("camera", "buy")) == "test"
    with pytest.raises(ValueError):
        swde.sites("train")  # type: ignore[arg-type]


def test_split_json_matches_rule() -> None:
    path = SWDE_EVAL_DIR / "split.json"
    if not path.is_file():
        pytest.skip("split.json not written yet")
    assert json.loads(path.read_text(encoding="utf-8")) == swde.split_document()


def test_ref_for_name() -> None:
    assert swde.ref_for_name("nbaplayer-wiki") == swde.SiteRef("nbaplayer", "wiki")
    assert swde.ref_for_name("auto-aol").name == "auto-aol"
    with pytest.raises(KeyError):
        swde.ref_for_name("auto-nosuch")


@pytest.mark.parametrize("vertical", swde.VERTICALS)
def test_fields_yaml_covers_attributes(vertical: str) -> None:
    path = swde.fields_path(vertical, SWDE_EVAL_DIR / "fields")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(swde.ATTRIBUTES[vertical])
    schema = FieldsSchema.load(path)
    assert schema.names() == list(swde.ATTRIBUTES[vertical])
    assert all(f.description for f in schema.fields)


# --------------------------------------------------------------------------- fake tree


def _write_gt(root: Path, vertical: str, site: str, attribute: str, rows: list[list[str]]) -> None:
    flat = [v for row in rows for v in row]
    stats = f"{len(rows)}\t{sum(1 for r in rows if r)}\t{len(flat)}\t{len(set(flat))}"
    body = "".join(
        f"{i:04d}\t{len(r)}\t{chr(9).join(r) if r else '<NULL>'}\r\n" for i, r in enumerate(rows)
    )
    path = swde.groundtruth_path(root, vertical, site, attribute)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"\ufeff{vertical}\t{site}\t{attribute}\r\n{stats}\r\n{body}", "utf-8")


def make_tree(root: Path, pages: int = 2) -> Path:
    """A structurally complete miniature SWDE tree: 80 sites, ``pages`` pages each."""
    for vertical in swde.VERTICALS:
        for site in swde.SITES[vertical]:
            sdir = root / vertical / f"{vertical}-{site}({pages})"
            sdir.mkdir(parents=True)
            for i in range(pages):
                url = f"http://www.{site}.example/{vertical}/item?id={i}&amp;x=1"
                (sdir / f"{i:04d}.htm").write_bytes(
                    f'\ufeff<base href="{url}"/>\r\n<html><body><h1>Item {i} &amp; co</h1>'
                    "</body></html>".encode()
                )
            for attribute in swde.ATTRIBUTES[vertical]:
                rows = [[f"{attribute} {i} &amp; co"] if i % 2 == 0 else [] for i in range(pages)]
                _write_gt(root, vertical, site, attribute, rows)
    return root


def test_scan_layout_accepts_complete_tree(tmp_path: Path) -> None:
    scans = swde.scan_layout(make_tree(tmp_path / "swde"))
    assert len(scans) == 80
    first = scans[0]
    assert (first.vertical, first.site, first.pages, first.declared_pages) == ("auto", "aol", 2, 2)
    assert first.attributes["model"] == {
        "pages_with_values": 1,
        "values": 1,
        "unique_values": 1,
        "multi_value_pages": 0,
    }


def test_scan_layout_reports_every_problem(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "swde")
    (root / "camera" / "camera-jr(2)" / "0001.htm").unlink()  # page missing vs. name
    (root / "groundtruth" / "book" / "book-buy-isbn_13.txt").unlink()  # ground truth missing
    (root / "groundtruth" / "book" / "book-buy-color.txt").write_text("x", "utf-8")
    _write_gt(root, "job", "dice", "title", [["a"], ["b"], ["c"]])  # page ids != pages
    (root / "movie" / "movie-imdb(2)").rename(root / "movie" / "movie-imdb-old")
    with pytest.raises(swde.LayoutError) as info:
        swde.scan_layout(root)
    text = "\n".join(info.value.problems)
    assert "camera-jr(2): 1 pages, name says 2" in text
    assert "missing ground-truth file book/book-buy-isbn_13.txt" in text
    assert "unexpected ground-truth file book/book-buy-color.txt" in text
    assert "job-dice-title.txt: page ids differ" in text
    assert "movie: sites" in text and "no SWDE site directory for movie-imdb" in text


def test_scan_layout_missing_root(tmp_path: Path) -> None:
    with pytest.raises(swde.LayoutError, match="verticals"):
        swde.scan_layout(tmp_path / "absent")


def test_load_site_labels(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "swde")
    input_dir, labels = swde.load_site("camera", "amazon", root=root)
    assert input_dir == root / "camera" / "camera-amazon(2)"
    assert labels == {
        "0000": {
            "model": ["model 0 & co"],
            "price": ["price 0 & co"],
            "manufacturer": ["manufacturer 0 & co"],
        },
        "0001": {"model": [], "price": [], "manufacturer": []},
    }
    raw = swde.load_labels(root, "camera", "amazon", decode=False)
    assert raw["0000"]["model"] == ["model 0 &amp; co"]
    with pytest.raises(FileNotFoundError):
        swde.site_dir(root, "camera", "nosuch")


def test_load_labels_rejects_wrong_header(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "swde")
    path = swde.groundtruth_path(root, "camera", "amazon", "price")
    path.write_text(path.read_text("utf-8-sig").replace("amazon", "buy", 1), "utf-8")
    with pytest.raises(swde.GroundTruthError, match="header"):
        swde.load_labels(root, "camera", "amazon")


def test_page_urls_falls_back_on_duplicates(tmp_path: Path) -> None:
    sdir = tmp_path / "camera-buy(3)"
    sdir.mkdir()
    for i, head in enumerate([b'<base href="http://x/1"/>', b'<base href="http://x/1"/>', b""]):
        (sdir / f"{i:04d}.htm").write_bytes(head + b"<html></html>")
    (sdir / "notes.txt").write_text("ignored", "utf-8")
    ref = swde.SiteRef("camera", "buy")
    urls = swde.page_urls(ref, swde.page_paths(sdir))
    assert urls == {
        "0000": "https://buy.swde.invalid/camera/0000.htm",
        "0001": "https://buy.swde.invalid/camera/0001.htm",
        "0002": "https://buy.swde.invalid/camera/0002.htm",
    }


def test_declared_pages() -> None:
    assert swde.declared_pages(Path("camera-amazon(1767)")) == 1767
    with pytest.raises(ValueError):
        swde.declared_pages(Path("camera-amazon"))


def test_load_sites_dev_camera(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "swde")
    work = tmp_path / "work"
    selection = Selection(
        split="dev",
        verticals=("camera",),
        options={"data_root": str(root), "fields_dir": str(SWDE_EVAL_DIR / "fields")},
    )
    found = swde.load_sites(selection, work)
    assert [s.name for s in found] == ["camera-amazon", "camera-beachaudio"]
    site = found[0]
    assert site.vertical == "camera" and site.input_kind == "dir"
    assert site.input_paths == [root / "camera" / "camera-amazon(2)"]
    assert site.fields_schema_path == SWDE_EVAL_DIR / "fields" / "camera.yaml"
    assert site.labels["0000"]["price"] == ["price 0 & co"]
    assert site.labels["0001"]["price"] == []
    assert site.manifest == work / "swde" / "camera-amazon" / "urls.tsv"
    lines = site.manifest.read_text("utf-8").splitlines()
    assert lines == [
        "0000.htm\thttp://www.amazon.example/camera/item?id=0&x=1",
        "0001.htm\thttp://www.amazon.example/camera/item?id=1&x=1",
    ]
    assert site.key_for_url("http://www.amazon.example/camera/item?id=1&x=1") == "0001"
    assert site.meta["split"] == "dev" and site.meta["pages"] == 2
    assert site.meta["fallback_urls"] == 0
    assert site.fields() == ["model", "price", "manufacturer"]


def test_load_sites_test_split_filters_and_limits(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "swde")
    opts = {"data_root": str(root), "fields_dir": str(tmp_path / "nofields")}
    found = swde.load_sites(Selection(split="test", max_sites=3, options=opts), tmp_path / "w")
    assert [s.name for s in found] == ["auto-automotive", "auto-autoweb", "auto-carquotes"]
    assert all(s.fields_schema_path is None for s in found)
    one = swde.load_sites(
        Selection(split="all", sites=("nbaplayer-wiki",), options=opts), tmp_path / "w"
    )
    assert [s.name for s in one] == ["nbaplayer-wiki"]
    assert not swde.load_sites(
        Selection(split="dev", sites=("nbaplayer-wiki",), options=opts), tmp_path / "w"
    )


def test_load_sites_unavailable(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailable, match=r"download\.py"):
        swde.load_sites(Selection(options={"data_root": str(tmp_path)}), tmp_path / "w")
    root = make_tree(tmp_path / "swde")
    (root / "auto" / "auto-aol(2)").rename(tmp_path / "moved")
    with pytest.raises(DatasetUnavailable, match="auto-aol"):
        swde.load_sites(
            Selection(sites=("auto-aol",), options={"data_root": str(root)}), tmp_path / "w"
        )


def test_url_index_roundtrip_and_staleness(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "swde")
    refs = [swde.SiteRef("camera", "amazon"), swde.SiteRef("job", "dice")]
    assert swde.build_url_index(root, refs, workers=2) == 4
    index = swde.url_index_path(root, refs[0])
    assert index == root / "urls" / "camera-amazon.tsv"
    assert swde.read_urls_tsv(index)["0001"] == "http://www.amazon.example/camera/item?id=1&x=1"
    paths = swde.page_paths(swde.site_dir(root, "camera", "amazon"))
    index.write_text("0000.htm\thttp://idx/0\n0001.htm\thttp://idx/1\n\n", "utf-8")
    assert swde.site_urls(root, refs[0], paths)["0001"] == "http://idx/1"  # index is used
    index.write_text("0000.htm\thttp://idx/0\n", "utf-8")  # stale: falls back to the pages
    assert swde.site_urls(root, refs[0], paths)["0001"].startswith("http://www.amazon.example/")
    assert swde.is_fallback_url(swde.fallback_url(refs[1], "0003"))
    assert not swde.is_fallback_url("http://www.dice.com/job/1")


@pytest.mark.slow
@pytest.mark.skipif(not (REAL_DATA / "groundtruth").is_dir(), reason="SWDE not downloaded")
def test_real_dataset_every_site_loads(tmp_path: Path) -> None:
    found = swde.load_sites(Selection(options={"data_root": str(REAL_DATA)}), tmp_path)
    assert len(found) == 80
    assert sum(s.meta["pages"] for s in found) == 124291
    for site in found:
        assert len(site.labels) == site.meta["pages"] == len(site.url_to_key)
        assert site.meta["fallback_urls"] == 0


# --------------------------------------------------------------------------- download script


def _load_download_script() -> ModuleType:
    name = "swde_download_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SWDE_EVAL_DIR / "download.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(dl: ModuleType, data: bytes, name: str = "x.bin") -> object:
    git = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
    return dl.Artifact(name, len(data), git, hashlib.sha256(data).hexdigest(), False)


def test_file_hashes_match_git(tmp_path: Path) -> None:
    dl = _load_download_script()
    path = tmp_path / "hello"
    path.write_bytes(b"hello\n")
    git_sha, sha = dl.file_hashes(path)
    assert git_sha == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object`
    assert sha == hashlib.sha256(b"hello\n").hexdigest()


def test_artifact_table_is_pinned() -> None:
    dl = _load_download_script()
    names = {a.name for a in dl.ARTIFACTS}
    assert {f"{v}.7z" for v in swde.VERTICALS} | {"groundtruth.7z"} <= names
    assert all(len(a.sha256) == 64 and len(a.git_blob_sha1) == 40 for a in dl.ARTIFACTS)
    assert all(a.url.startswith("https://raw.githubusercontent.com/") for a in dl.ARTIFACTS)


def test_verify_file_rejects_mismatches(tmp_path: Path) -> None:
    dl = _load_download_script()
    data = b"0123456789"
    art = _artifact(dl, data)
    path = tmp_path / "x.bin"
    path.write_bytes(data)
    assert dl.verify_file(path, art) == hashlib.sha256(data).hexdigest()
    path.write_bytes(b"0123456780")
    with pytest.raises(dl.VerificationError, match="git blob"):
        dl.verify_file(path, art)
    path.write_bytes(b"short")
    with pytest.raises(dl.VerificationError, match="size"):
        dl.verify_file(path, art)


def _range_transport(data: bytes, seen: list[str | None], honor_range: bool = True) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        seen.append(rng)
        if rng and honor_range:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=data[start:])
        return httpx.Response(200, content=data)

    return httpx.MockTransport(handler)


def test_download_resumes_from_part_file(tmp_path: Path) -> None:
    dl = _load_download_script()
    data = bytes(range(256)) * 10
    art = _artifact(dl, data)
    (tmp_path / "x.bin.part").write_bytes(data[:1000])
    seen: list[str | None] = []
    with httpx.Client(transport=_range_transport(data, seen)) as client:
        hashes = dl.fetch_all(tmp_path, client=client, artifacts=(art,), log=lambda _: None)
    assert seen == ["bytes=1000-"]
    assert (tmp_path / "x.bin").read_bytes() == data
    assert hashes == {"x.bin": hashlib.sha256(data).hexdigest()}
    # Second run: present and verified, no request.
    with httpx.Client(transport=_range_transport(data, seen)) as client:
        dl.fetch_all(tmp_path, client=client, artifacts=(art,), log=lambda _: None)
    assert seen == ["bytes=1000-"]


def test_download_restarts_when_range_ignored(tmp_path: Path) -> None:
    dl = _load_download_script()
    data = b"abcdefghij" * 50
    art = _artifact(dl, data)
    (tmp_path / "x.bin.part").write_bytes(data[:100])
    seen: list[str | None] = []
    logs: list[str] = []
    with httpx.Client(transport=_range_transport(data, seen, honor_range=False)) as client:
        dl.fetch_all(tmp_path, client=client, artifacts=(art,), log=logs.append)
    assert (tmp_path / "x.bin").read_bytes() == data
    assert any("ignored Range" in line for line in logs)


def test_download_bad_content_is_removed(tmp_path: Path) -> None:
    dl = _load_download_script()
    art = _artifact(dl, b"expected!")
    (tmp_path / "x.bin").write_bytes(b"corrupted")  # same size, wrong bytes: refetched
    with (
        httpx.Client(transport=_range_transport(b"tampered!", [])) as client,
        pytest.raises(dl.VerificationError),
    ):
        dl.fetch_all(tmp_path, client=client, artifacts=(art,), log=lambda _: None)
    assert not (tmp_path / "x.bin").exists()


def test_download_http_error(tmp_path: Path) -> None:
    dl = _load_download_script()
    art = _artifact(dl, b"data")
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    with httpx.Client(transport=transport) as client, pytest.raises(dl.VerificationError):
        dl.download(client, art, tmp_path / "x.bin", log=lambda _: None)


def test_manifest_from_fake_tree(tmp_path: Path) -> None:
    dl = _load_download_script()
    scans = swde.scan_layout(make_tree(tmp_path / "swde"))
    hashes = {a.name: a.sha256 for a in dl.ARTIFACTS}
    manifest = dl.build_manifest(scans, hashes, downloaded_on="2026-10-24")
    assert manifest["totals"] == {
        "verticals": 8,
        "sites": 80,
        "pages": 160,
        "groundtruth_files": 320,
    }
    assert manifest["split"]["dev_pages"] == 32 and manifest["split"]["test_pages"] == 128
    assert manifest["verticals"]["camera"]["site_detail"]["amazon"]["split"] == "dev"
    assert set(manifest["fields"]) == set(swde.VERTICALS)
    assert manifest["archives"][0]["sha256"] == dl.ARTIFACTS[0].sha256


def test_extract_all_uses_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dl = _load_download_script()
    hashes = {a.name: a.sha256 for a in dl.ARTIFACTS}
    calls: list[list[str]] = []
    monkeypatch.setattr(dl, "seven_zip", lambda: "7z")
    monkeypatch.setattr(dl.subprocess, "run", lambda cmd, check: calls.append(cmd))
    quiet = {"log": lambda _: None}
    assert dl.extract_all(tmp_path / "a", tmp_path, hashes, **quiet) is True
    n_archives = sum(1 for a in dl.ARTIFACTS if a.extract)
    assert len(calls) == n_archives and calls[0][:2] == ["7z", "x"]
    assert dl.extract_all(tmp_path / "a", tmp_path, hashes, **quiet) is False
    assert dl.extract_all(tmp_path / "a", tmp_path, hashes, force=True, **quiet) is True
    assert len(calls) == 2 * n_archives


def test_main_writes_split_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dl = _load_download_script()
    root = make_tree(tmp_path / "swde")
    hashes = {a.name: a.sha256 for a in dl.ARTIFACTS}
    monkeypatch.setattr(dl, "fetch_all", lambda archive_dir, client: hashes)
    monkeypatch.setattr(dl, "extract_all", lambda *args, **kwargs: False)
    monkeypatch.setattr(dl, "SPLIT_PATH", tmp_path / "split.json")
    monkeypatch.setattr(dl, "MANIFEST_PATH", tmp_path / "manifest.json")
    assert dl.main(["--data-dir", str(root)]) == 0
    assert json.loads((tmp_path / "split.json").read_text("utf-8")) == swde.split_document()
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert manifest["totals"]["pages"] == 160
    assert (root / "urls" / "camera-amazon.tsv").is_file()
    # A rerun keeps the original download date and skips the URL index.
    manifest["downloaded_on"] = "2000-01-01"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    assert dl.main(["--data-dir", str(root)]) == 0
    rerun = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert rerun["downloaded_on"] == "2000-01-01"
    (root / "groundtruth" / "auto" / "auto-aol-price.txt").unlink()
    assert dl.main(["--data-dir", str(root)]) == 1
