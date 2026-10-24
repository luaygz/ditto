from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _eval_tiny_site import make_tiny_site

from ditto.evaluation.datasets import synthetic
from ditto.evaluation.datasets.base import (
    DatasetUnavailable,
    EvalSite,
    Selection,
    is_empty,
    select_sites,
)
from ditto.schema import FieldsSchema


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    return make_tiny_site(tmp_path / "synth")


def test_read_site_builds_labels_kinds_and_truth(tiny: Path, tmp_path: Path) -> None:
    site = synthetic.read_site(tiny, name="tiny", work_dir=tmp_path / "work")
    assert site.input_kind == "dir"
    assert site.input_paths == [tiny / "site"]
    assert site.manifest == tiny / "site" / "urls.tsv"
    assert len(site.labels) == 8
    assert site.labels["p000003"]["price"] is None
    assert site.labels["p000000"]["sizes"] == [
        {"size": "S", "stock": "3"},
        {"size": "M", "stock": "0"},
    ]
    assert "headline" not in site.labels["p000000"]  # other template's field: not annotated
    assert site.field_kinds == {"tags": "list", "sizes": "record_list", "specs": "key_value"}
    assert site.kind("title") == "scalar"
    assert site.truth_templates["p000005"] == "T02"
    assert site.key_for_url("https://shop.example/news/5") == "p000005"
    assert site.meta["synth"]["seed"] == 1
    assert site.fields_schema_path is not None
    schema = FieldsSchema.load(site.fields_schema_path)
    assert schema.names() == ["title", "price", "tags", "sizes", "specs", "headline", "author"]
    title = schema.get("title")
    assert title is not None and title.description == "the product name"
    specs = schema.get("specs")
    assert specs is not None and specs.type == "key_value"


def test_read_site_truth_only(tiny: Path, tmp_path: Path) -> None:
    site = synthetic.read_site(tiny, name="tiny", work_dir=tmp_path, truth_only=True)
    assert all(row == {} for row in site.labels.values())
    assert len(site.truth_templates) == 8
    assert site.field_kinds == {}


def test_read_site_rejects_other_dirs(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailable):
        synthetic.read_site(tmp_path, name="x", work_dir=tmp_path)
    (tmp_path / "site").mkdir()
    (tmp_path / "truth").mkdir()
    with pytest.raises(DatasetUnavailable, match="no page sidecars"):
        synthetic.read_site(tmp_path, name="x", work_dir=tmp_path)


def test_machine_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synthetic, "cpu_count", lambda: 4)
    monkeypatch.setattr(synthetic, "memory_gb", lambda: 64.0)
    pages, note = synthetic.machine_scaled_pages(100_000)
    assert pages == 20_000
    assert note is not None and "100,000 to 20,000" in note and "4 cores" in note
    assert synthetic.machine_scaled_pages(10_000) == (10_000, None)
    monkeypatch.setattr(synthetic, "cpu_count", lambda: 16)
    assert synthetic.machine_scaled_pages(100_000) == (100_000, None)
    monkeypatch.setattr(synthetic, "memory_gb", lambda: 8.0)
    assert synthetic.machine_scaled_pages(100_000)[0] == 20_000


def test_load_sites_sweep_one_parameter_at_a_time(
    tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated: list[tuple[dict[str, Any], int]] = []

    def fake_generate(
        params: Path | None, overrides: dict[str, Any], seed: int, cache: Path
    ) -> Path:
        generated.append((overrides, seed))
        return tiny

    monkeypatch.setattr(synthetic, "_generate", fake_generate)
    monkeypatch.setattr(synthetic, "cpu_count", lambda: 2)
    params = tmp_path / "default.yaml"
    params.write_text("templates: 12\n")
    selection = Selection(
        options={
            "params": str(params),
            "pages": 100_000,
            "scale_for_machine": True,
            "overrides": {"noise_rate": 0.1},
            "sweep": {"templates": [2, 50], "markup_style": ["react"]},
            "include_default": True,
            "seed": 1,
        }
    )
    sites = synthetic.load_sites(selection, tmp_path / "work")
    assert [s.name for s in sites] == [
        "default",
        "markup_style=react",
        "templates=2",
        "templates=50",
    ]
    assert generated[0] == ({"noise_rate": 0.1, "total_pages": 20_000}, 1)
    assert generated[1][0] == {"noise_rate": 0.1, "total_pages": 20_000, "templates": 2}
    by_name = {s.name: s for s in sites}
    assert by_name["templates=50"].meta["parameter"] == "templates"
    assert by_name["templates=50"].meta["value"] == 50
    assert by_name["templates=50"].meta["index"] == 2
    assert by_name["default"].meta["parameter"] is None
    assert "scaled from 100,000 to 20,000" in by_name["default"].meta["notes"][0]


def test_load_sites_with_site_dir_and_missing_params(tiny: Path, tmp_path: Path) -> None:
    sites = synthetic.load_sites(Selection(options={"site_dir": str(tiny)}), tmp_path)
    assert [s.name for s in sites] == ["synth"]
    with pytest.raises(DatasetUnavailable, match="not found"):
        synthetic.load_sites(Selection(options={"params": str(tmp_path / "nope.yaml")}), tmp_path)


def _site(name: str, vertical: str) -> EvalSite:
    return EvalSite(name, vertical, [], "dir", None, {}, {})


def test_select_sites_filters_and_orders() -> None:
    sites = [_site("b", "movie"), _site("a", "movie"), _site("z", "auto")]
    assert [s.name for s in select_sites(sites, Selection())] == ["z", "a", "b"]
    assert [s.name for s in select_sites(sites, Selection(verticals=("movie",)))] == ["a", "b"]
    assert [s.name for s in select_sites(sites, Selection(sites=("b", "z")))] == ["z", "b"]
    assert [s.name for s in select_sites(sites, Selection(max_sites=1))] == ["z"]


def test_eval_site_helpers() -> None:
    site = EvalSite(
        "s",
        "v",
        [],
        "warc",
        None,
        {"k1": {"a": "1"}, "k2": {"b": None}},
        {"https://x.org/p": "k1", "https://x.org/q/": "k2"},
        field_kinds={"c": "list"},
    )
    assert site.fields() == ["a", "b", "c"]
    assert site.key_for_url("https://x.org/p") == "k1"
    assert site.key_for_url("https://x.org/p#frag") == "k1"
    assert site.key_for_url("https://x.org/p/") == "k1"
    assert site.key_for_url("https://x.org/q") == "k2"
    assert site.key_for_url("https://x.org/zzz") is None
    assert is_empty(None) and is_empty(" ") and is_empty([]) and is_empty({})
    assert not is_empty(0) and not is_empty("x") and not is_empty(["x"])
