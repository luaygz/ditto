from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from lwie_fixtures import D1, D2, D3, L1, Q1

from ditto.evaluation.datasets.base import DatasetUnavailable, Selection
from ditto.evaluation.datasets.livewebie import (
    attribute_category,
    complexity_groups,
    load_release,
    load_sites,
    parse_groundtruth,
    site_fields,
    site_labels,
    site_templates,
)
from ditto.schema import FieldsSchema

# Real lines from groundtruth/academic/huggingface.co-g_000-title.txt (a title wraps a line),
# arxiv.org-g_000-author.txt (tab-separated list) and arxiv.org-g_001-title.txt (count 0).
HF_TITLE = (
    "academic huggingface.co g_000 paper_title\n"
    "2 2 2\n"
    '"https://huggingface.co/papers/2408.10188" 1 LongVILA: Scaling Long-Context Visual Language'
    " Models for Long Videos\n"
    '"https://huggingface.co/papers/2408.10198" 1 MeshFormer: High-Quality Mesh Generation with'
    " 3D-Guided Reconstruction\n"
    "  Model\n"
)
ARXIV_AUTHOR = (
    "academic arxiv.org g_000 paper_author\n"
    "3 10 10\n"
    '"https://arxiv.org/abs/2401.00588" 8 Ying Sheng\tShiyi Cao\tDacheng Li\tBanghua Zhu\t'
    "Zhuohan Li\tDanyang Zhuo\tJoseph E. Gonzalez\tIon Stoica\n"
    '"https://arxiv.org/abs/2401.01150" 2 Alberto Lerner\tGustavo Alonso\n'
    '"https://arxiv.org/list/cs.GL/2024-01" 0 \n'
)


def test_parse_groundtruth_multiline_value() -> None:
    gt = parse_groundtruth(HF_TITLE)
    assert (gt.domain, gt.website, gt.group_id, gt.label) == (
        "academic",
        "huggingface.co",
        "g_000",
        "paper_title",
    )
    assert gt.values["https://huggingface.co/papers/2408.10198"] == [
        "MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction\n  Model"
    ]


def test_parse_groundtruth_lists_and_empty() -> None:
    gt = parse_groundtruth(ARXIV_AUTHOR)
    assert len(gt.values["https://arxiv.org/abs/2401.00588"]) == 8
    assert gt.values["https://arxiv.org/abs/2401.01150"] == ["Alberto Lerner", "Gustavo Alonso"]
    assert gt.values["https://arxiv.org/list/cs.GL/2024-01"] == []


@pytest.mark.parametrize(
    "text",
    [
        "academic arxiv.org g_000\n1 1 1\n",  # short header
        'academic arxiv.org g_000 t\n1 1 1\n"https://a.org/x" 2 only-one\n',  # count mismatch
        'academic arxiv.org g_000 t\n2 1 1\n"https://a.org/x" 1 v\n',  # page count mismatch
        "academic arxiv.org g_000 t\n1 1 1\nstray line\n",  # text before first record
        "just one line",
    ],
)
def test_parse_groundtruth_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        parse_groundtruth(text)


def test_load_release_structure(release_root: Path) -> None:
    rel = load_release(release_root)
    assert rel.websites() == ["books.toscrape.com", "quotes.toscrape.com"]
    assert rel.domains == {"books.toscrape.com": "book", "quotes.toscrape.com": "other"}
    assert rel.groups[("books.toscrape.com", "g_000")] == [D1, D2, D3]  # duplicate D1 dropped
    assert rel.site_urls("books.toscrape.com") == [D1, D2, D3, L1]
    assert rel.queries[2].cardinality == "list"
    with pytest.raises(KeyError):
        rel.label_path("books.toscrape.com", "g_009", "title")


def test_load_release_missing(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailable):
        load_release(tmp_path)


def test_site_fields_rename_list_homonyms(release_root: Path) -> None:
    rel = load_release(release_root)
    fields = {fd.name: fd for fd in site_fields(rel, "books.toscrape.com")}
    assert list(fields) == ["title", "book_image", "title_list", "book_link"]
    assert fields["title"].kind == "scalar" and fields["title"].groups == ["g_000"]
    assert fields["title_list"].kind == "list" and fields["title_list"].attribute == "title"
    assert fields["book_link"].kind == "list"  # list-only attribute keeps its name
    assert fields["book_image"].category == "image"
    assert fields["book_link"].category == "link"


def test_descriptions_come_from_queries(release_root: Path) -> None:
    rel = load_release(release_root)
    fields = {fd.name: fd for fd in site_fields(rel, "books.toscrape.com")}
    assert fields["title"].description() == "Extract the title of the book."
    image = fields["book_image"].description()
    assert image.startswith('Book image (asked for in the query "Scrape the book stock')
    assert "img src" in image
    assert fields["title_list"].description().endswith("One entry per item on the page.")
    assert "a href" in fields["book_link"].description()


@pytest.mark.parametrize(
    ("attr", "cat"),
    [
        ("season_badge", "image"),
        ("thumb_image", "image"),
        ("logo", "image"),
        ("author_profile_link", "link"),
        ("link", "link"),
        ("photo_title", "text"),
        ("upc", "text"),
    ],
)
def test_attribute_category(attr: str, cat: str) -> None:
    assert attribute_category(attr) == cat


def test_site_labels_shapes(release_root: Path) -> None:
    rel = load_release(release_root)
    fields = site_fields(rel, "books.toscrape.com")
    labels = site_labels(rel, "books.toscrape.com", fields)
    assert labels[D1] == {
        "title": "A Light in the Attic",
        "book_image": "../../media/cache/fe/72/fe72f0532301ec28892ae79a629a293c.jpg",
    }
    assert labels[D3] == {"title": None}  # count 0 -> labeled empty; book_image not annotated
    assert labels[L1]["title_list"] == ["It's Only the Himalayas", "Full Moon over Noah's Ark"]
    assert "title" not in labels[L1]


def test_templates_and_complexity_groups(release_root: Path) -> None:
    rel = load_release(release_root)
    fields = site_fields(rel, "books.toscrape.com")
    assert site_templates(rel, "books.toscrape.com") == {
        D1: "g_000",
        D2: "g_000",
        D3: "g_000",
        L1: "g_002",
    }
    assert complexity_groups(rel, "books.toscrape.com", fields) == {
        "type_1": ["title"],
        "type_2": ["title", "book_image"],
        "type_3": ["title_list"],
        "type_4": ["title_list", "book_link"],
    }


def _write_dataset_dir(release_root: Path, *, included: dict[str, bool]) -> tuple[Path, Path, Path]:
    data_dir = release_root.parents[1]
    fields_dir = data_dir.parent / "fields"
    fields_dir.mkdir(exist_ok=True)
    for site in included:
        (fields_dir / f"{site}.yaml").write_text("fields:\n  title: the title\n")
        with gzip.open(data_dir / f"{site}.warc.gz", "wb") as fh:
            fh.write(b"")
    manifest = data_dir.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "partial",
                "sites": {s: {"included": inc, "warc_sha256": "x"} for s, inc in included.items()},
            }
        )
    )
    (data_dir / "fetchlog").mkdir(exist_ok=True)
    final = D2.replace("index.html", "")
    (data_dir / "fetchlog" / "books.toscrape.com.jsonl").write_text(
        json.dumps({"url": D2, "final_url": final, "status": 200})
        + "\n"
        + json.dumps({"url": D1, "final_url": D1, "status": 200})
        + "\n"
    )
    return data_dir, fields_dir, manifest


def _selection(data_dir: Path, fields_dir: Path, manifest: Path, **kw: object) -> Selection:
    opts = {"data_dir": str(data_dir), "fields_dir": str(fields_dir), "manifest": str(manifest)}
    return Selection(options=opts, **kw)  # type: ignore[arg-type]


def test_load_sites(release_root: Path, tmp_path: Path) -> None:
    data_dir, fields_dir, manifest = _write_dataset_dir(
        release_root, included={"books.toscrape.com": True, "quotes.toscrape.com": False}
    )
    sites = load_sites(_selection(data_dir, fields_dir, manifest), tmp_path / "work")
    assert [s.name for s in sites] == ["books.toscrape.com"]
    site = sites[0]
    assert site.vertical == "book"
    assert site.input_kind == "warc"
    assert site.input_paths == [data_dir / "books.toscrape.com.warc.gz"]
    assert site.fields_schema_path == fields_dir / "books.toscrape.com.yaml"
    assert site.kind("title_list") == "list" and site.kind("title") == "scalar"
    assert site.key_for_url(D2.replace("index.html", "")) == D2  # redirected page
    assert site.key_for_url(L1) == L1
    assert site.truth_templates[L1] == "g_002"
    assert site.field_groups["type_4"] == ["title_list", "book_link"]
    queries = site.meta["queries"]
    assert queries[3]["fields"] == ["title_list", "book_link"]
    assert site.meta["snapshot"] == {"included": True, "warc_sha256": "x"}


def test_load_sites_dev_split_is_empty(release_root: Path, tmp_path: Path) -> None:
    data_dir, fields_dir, manifest = _write_dataset_dir(
        release_root, included={"books.toscrape.com": True}
    )
    assert load_sites(_selection(data_dir, fields_dir, manifest, split="dev"), tmp_path) == []


def test_load_sites_unavailable(release_root: Path, tmp_path: Path) -> None:
    data_dir, fields_dir, manifest = _write_dataset_dir(
        release_root, included={"books.toscrape.com": True}
    )
    (data_dir / "books.toscrape.com.warc.gz").unlink()
    with pytest.raises(DatasetUnavailable, match="snapshot missing"):
        load_sites(_selection(data_dir, fields_dir, manifest), tmp_path)
    manifest.write_text(json.dumps({"status": "skipped", "reason": "release gone"}))
    with pytest.raises(DatasetUnavailable, match="release gone"):
        load_sites(_selection(data_dir, fields_dir, manifest), tmp_path)
    with pytest.raises(DatasetUnavailable, match="manifest missing"):
        load_sites(_selection(data_dir, fields_dir, tmp_path / "nope.json"), tmp_path)


def test_load_sites_missing_fields(release_root: Path, tmp_path: Path) -> None:
    data_dir, fields_dir, manifest = _write_dataset_dir(
        release_root, included={"books.toscrape.com": True}
    )
    (fields_dir / "books.toscrape.com.yaml").unlink()
    with pytest.raises(DatasetUnavailable, match=r"fields\.yaml missing"):
        load_sites(_selection(data_dir, fields_dir, manifest), tmp_path)


def test_committed_fields_files_load() -> None:
    fields_dir = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "livewebie" / "fields"
    paths = sorted(fields_dir.glob("*.yaml"))
    assert len(paths) == 15
    for path in paths:
        schema = FieldsSchema.load(path)
        assert schema.fields
        for f in schema.fields:
            assert f.type in ("scalar", "list") and f.description and not f.examples


def test_quotes_site_labels(release_root: Path) -> None:
    rel = load_release(release_root)
    fields = site_fields(rel, "quotes.toscrape.com")
    assert [(f.name, f.kind) for f in fields] == [("author", "scalar")]
    assert site_labels(rel, "quotes.toscrape.com", fields) == {Q1: {"author": "Albert Einstein"}}
