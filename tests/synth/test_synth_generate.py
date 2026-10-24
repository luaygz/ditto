"""End-to-end generation: layout, determinism, sidecars, noise, presence rates, calibration."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest
from _synth_checks import page_encoding, sidecar_problems, tree_bytes

from ditto.schema import FieldsSchema
from ditto.synth import SynthParams, generate_site, synth_main
from ditto.synth.plan import MAX_CROSS_JACCARD, build_plan

SMALL = SynthParams(templates=3, tail_templates=1, total_pages=300, table_rows=(0, 4))


@pytest.fixture(scope="module")
def small_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("small")
    generate_site(SMALL, 141, out, workers=1)
    return out


def test_layout_and_metadata(small_site: Path) -> None:
    meta = json.loads((small_site / "synth.json").read_text())
    assert meta["seed"] == 141 and meta["n_pages"] == 300 and meta["n_templates"] == 4
    assert sum(meta["template_pages"].values()) == 300
    pages = sorted(p.name for p in (small_site / "site").glob("*.html"))
    assert len(pages) == 300 and pages[0] == "p000000.html"
    lines = (small_site / "site/urls.tsv").read_text().splitlines()
    assert len(lines) == 300
    paths, urls = zip(*(line.split("\t") for line in lines), strict=True)
    assert list(paths) == pages and len(set(urls)) == 300
    assert all(u.startswith("https://") for u in urls)
    truth = sorted(p.name for p in (small_site / "truth").glob("p*.json"))
    assert [t.removesuffix(".json") for t in truth] == [p.removesuffix(".html") for p in pages]

    doc = json.loads((small_site / "truth/templates.json").read_text())
    tids = [t["template_id"] for t in doc["templates"]]
    assert tids == ["T01", "T02", "T03", "T04"]
    assert [t["is_tail"] for t in doc["templates"]] == [False, False, False, True]
    assert sum(t["near_duplicate_of"] is not None for t in doc["templates"]) == 1
    for t in doc["templates"]:
        types = Counter(f["type"] for f in t["fields"])
        assert types["list"] == 1 and types["record_list"] == 1 and types["key_value"] == 1
        assert sum(f["optional"] for f in t["fields"]) == 3
        schema = FieldsSchema.load(small_site / "truth/fields" / f"{t['template_id']}.yaml")
        assert schema.names() == [f["name"] for f in t["fields"]]


def test_sidecars_match_page_text(small_site: Path) -> None:
    by_template: Counter[str] = Counter()
    for truth_path in sorted((small_site / "truth").glob("p*.json")):
        truth = json.loads(truth_path.read_text())
        raw = (small_site / "site" / f"{truth['page']}.html").read_bytes()
        assert sidecar_problems(raw, truth) == [], truth["page"]
        by_template[truth["template_id"]] += 1
        present = {k for k, v in truth["fields"].items() if v is not None}
        assert set(truth["optional_present"]) <= present
    meta = json.loads((small_site / "synth.json").read_text())
    assert dict(by_template) == meta["template_pages"]


def test_deterministic_across_workers_and_seeds(small_site: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    generate_site(SMALL, 141, again, workers=3)
    assert tree_bytes(again) == tree_bytes(small_site)
    other = tmp_path / "other"
    generate_site(SMALL, 142, other, workers=1)
    assert tree_bytes(other)["site/p000000.html"] != tree_bytes(small_site)["site/p000000.html"]


def test_rerun_in_place_and_refuse_foreign_dirs(tmp_path: Path) -> None:
    params = SynthParams(templates=2, tail_templates=0, near_duplicate_pairs=0, total_pages=40)
    out = tmp_path / "site_out"
    generate_site(params, 143, out, workers=1)
    first = tree_bytes(out)
    (out / "site" / "stale.html").write_text("x")
    generate_site(params, 143, out, workers=1)
    assert tree_bytes(out) == first
    foreign = tmp_path / "foreign"
    (foreign / "site").mkdir(parents=True)
    (foreign / "site" / "keep.html").write_text("mine")
    with pytest.raises(FileExistsError):
        generate_site(params, 143, foreign, workers=1)
    assert (foreign / "site" / "keep.html").read_text() == "mine"


def test_noise_keeps_ground_truth(tmp_path: Path) -> None:
    params = SynthParams(
        templates=3,
        tail_templates=0,
        near_duplicate_pairs=0,
        total_pages=60,
        markup_style="mixed",
        absent_rendering="mixed",
        noise=("entities", "nbsp", "whitespace", "charset"),
        noise_rate=1.0,
    )
    generate_site(params, 144, tmp_path, workers=1)
    raws = []
    for truth_path in sorted((tmp_path / "truth").glob("p*.json")):
        truth = json.loads(truth_path.read_text())
        raw = (tmp_path / "site" / f"{truth['page']}.html").read_bytes()
        assert page_encoding(raw) == "windows-1252"
        assert sidecar_problems(raw, truth) == [], truth["page"]
        raws.append(raw)
    assert all(b"&#" in r for r in raws)
    assert any(b"&nbsp;" in r or b"&#160;" in r or b"\xa0" in r for r in raws)
    minified = [r for r in raws if b"\n" not in r.strip()]
    loose = [r for r in raws if b"\n" in r.strip()]
    assert minified and loose


def test_absent_na_pages_have_null_truth(tmp_path: Path) -> None:
    params = SynthParams(
        templates=2, tail_templates=0, near_duplicate_pairs=0, total_pages=60, absent_rendering="na"
    )
    generate_site(params, 145, tmp_path, workers=1)
    na_pages = 0
    for truth_path in sorted((tmp_path / "truth").glob("p*.json")):
        truth = json.loads(truth_path.read_text())
        raw = (tmp_path / "site" / f"{truth['page']}.html").read_text()
        nulls = [k for k, v in truth["fields"].items() if v is None]
        assert raw.count(">N/A<") == len(nulls)
        na_pages += bool(nulls)
    assert na_pages > 0


def test_presence_rates_and_boilerplate_share_mid_size(tmp_path: Path) -> None:
    params = SynthParams(
        templates=2,
        tail_templates=0,
        near_duplicate_pairs=0,
        total_pages=2000,
        optional_fields=5,
        optional_columns=2,
        optional_presence=(0.05, 0.9),
    )
    generate_site(params, 146, tmp_path)
    doc = json.loads((tmp_path / "truth/templates.json").read_text())
    counted: Counter[tuple[str, str]] = Counter()
    for truth_path in (tmp_path / "truth").glob("p*.json"):
        truth = json.loads(truth_path.read_text())
        for name in truth["optional_present"]:
            counted[truth["template_id"], name] += 1
    checked = 0
    for t in doc["templates"]:
        n = t["pages"]
        units = [f for f in t["fields"] if f["optional"]]
        units += [c for f in t["fields"] for c in f.get("columns", []) if c["optional"]]
        for unit in units:
            p, obs = unit["presence"], unit["observed_presence"]
            assert abs(obs - p) <= 4 * math.sqrt(p * (1 - p) / n) + 0.01, (t["template_id"], unit)
            checked += 1
        for f in t["fields"]:
            if f["optional"]:
                assert counted[t["template_id"], f["name"]] == round(f["observed_presence"] * n)
    assert checked == 2 * (5 + 2)
    meta = json.loads((tmp_path / "synth.json").read_text())
    assert abs(meta["measured"]["boilerplate_share_mean"] - 0.4) < 0.08
    assert meta["generation"]["pages_per_second"] > 0


@pytest.mark.parametrize("share", [0.0, 0.3, 1.0])
def test_hashed_class_share(share: float) -> None:
    plan = build_plan(
        SynthParams(
            templates=4,
            tail_templates=0,
            near_duplicate_pairs=0,
            total_pages=100,
            hashed_class_share=share,
        ),
        147,
    )
    assert abs(plan.hashed_class_share - share) < 0.12


@pytest.mark.parametrize("share", [0.15, 0.6])
def test_boilerplate_share_target(share: float, tmp_path: Path) -> None:
    params = SynthParams(
        templates=3,
        tail_templates=0,
        near_duplicate_pairs=0,
        total_pages=120,
        boilerplate_share=share,
        boilerplate_variants=5,
    )
    generate_site(params, 148, tmp_path, workers=1)
    meta = json.loads((tmp_path / "synth.json").read_text())
    assert abs(meta["measured"]["boilerplate_share_mean"] - share) < 0.08
    assert len(meta["measured"]["shell_elements"]) == 5


@pytest.mark.parametrize("target", [0.6, 0.7, 0.8, 0.9])
def test_near_duplicate_jaccard_calibration(target: float) -> None:
    plan = build_plan(
        SynthParams(templates=4, tail_templates=0, total_pages=400, near_duplicate_jaccard=target),
        149,
    )
    (pair,) = plan.near_duplicates
    assert abs(pair["jaccard"] - target) <= 0.03, pair
    a = next(t for t in plan.templates if t.tid == pair["a"])
    b = next(t for t in plan.templates if t.tid == pair["b"])
    assert b.near_duplicate_of == a.tid and a.segment != b.segment
    assert plan.cross_jaccard["max"] <= MAX_CROSS_JACCARD


def test_default_site_plan_is_separable() -> None:
    plan = build_plan(SynthParams(total_pages=10_000), 1)
    assert plan.n_pages == 10_000
    assert len(plan.templates) == 15 and sum(t.is_tail for t in plan.templates) == 3
    assert all(t.pages < 20 for t in plan.templates if t.is_tail)
    assert all(t.pages >= 20 for t in plan.templates if not t.is_tail)
    (pair,) = plan.near_duplicates
    assert abs(pair["jaccard"] - 0.7) <= 0.02
    assert plan.cross_jaccard["max"] <= MAX_CROSS_JACCARD
    assert len({t.segment for t in plan.templates}) == 15
    assert len({t.prefix for t in plan.templates}) >= 14  # the derived partner shares its prefix


def test_synth_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    params_path = tmp_path / "p.yaml"
    params_path.write_text(
        "templates: 2\ntail_templates: 0\nnear_duplicate_pairs: 0\ntotal_pages: 30\n"
        "site_name: Example Outfitters\nhost: shop.example.org\nlist_fields: 2\n"
        "mandatory_fields: [6, 6]\n"
    )
    summary = synth_main(params_path, 150, tmp_path / "out", workers=1)
    assert summary.n_pages == 30 and summary.n_templates == 2
    assert "30 pages" in capsys.readouterr().out
    doc = json.loads((tmp_path / "out/truth/templates.json").read_text())
    assert doc["site_name"] == "Example Outfitters" and doc["host"] == "shop.example.org"
    for t in doc["templates"]:
        assert t["url_example"].startswith("https://shop.example.org/")
        assert sum(f["type"] == "list" for f in t["fields"]) == 2
        mandatory = [f for f in t["fields"][1:] if f["type"] == "scalar" and not f["optional"]]
        assert len(mandatory) == 6
