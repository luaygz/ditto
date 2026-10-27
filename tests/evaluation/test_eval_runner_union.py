"""``until: union`` evaluations: union-page tokens and optional-field coverage (v0.2 gate)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from _eval_tiny_site import FakePipeline, make_tiny_site

from ditto.analyze.models import Candidate, CandidateTable
from ditto.evaluation.datasets.base import EvalSite, Selection
from ditto.evaluation.datasets.synthetic import read_site
from ditto.evaluation.runner import (
    Hooks,
    label_strings,
    optional_fields,
    run_eval,
    truth_templates_file,
)
from ditto.run import RunDir, write_json

MAIN = "html > body > main.product"
TEMPLATES = {
    "templates": [
        {
            "template_id": "T01",
            "is_tail": False,
            "fields": [
                {"name": "title", "type": "scalar", "optional": False, "presence": 1.0},
                {"name": "price", "type": "scalar", "optional": True, "presence": 0.75},
                {
                    "name": "sizes",
                    "type": "record_list",
                    "optional": False,
                    "presence": 1.0,
                    "label": "Sizes",
                    "columns": [
                        {"name": "size", "label": "Size", "optional": False, "presence": 1.0},
                        {"name": "stock", "label": "Stock", "optional": True, "presence": 0.6},
                    ],
                },
            ],
        },
        {
            "template_id": "T02",
            "is_tail": False,
            "fields": [
                {"name": "headline", "type": "scalar", "optional": False, "presence": 1.0},
                {"name": "author", "type": "scalar", "optional": True, "presence": 0.5},
                {"name": "editor", "type": "scalar", "optional": True, "presence": 0.02},
            ],
        },
        {"template_id": "T09", "is_tail": True, "fields": []},
    ]
}


def _cand(cid: str, path: str, cls: str, **kw: Any) -> Candidate:
    return Candidate.model_validate(
        {"id": cid, "path": path, "class": cls, "p": 0.5, "m": 0.1, **kw}
    )


UNIONS: dict[str, tuple[list[Candidate], str, int]] = {
    "t_T01": (
        [
            _cand("c1", f"{MAIN} > h1", "mandatory"),
            _cand("c2", f"{MAIN} > span.price", "optional"),
            _cand("c3", f"{MAIN} > table > tr > td", "list", column="Size"),
            _cand("c4", f"{MAIN} > table > tr > td", "list", column="Stock"),
        ],
        '<html><body><main class="product"><!-- c1 · mandatory · on 100% of pages -->'
        "<h1>Widget One</h1><!-- c2 · optional · on 75% of pages --><span>$19.99</span>"
        "<table><tr><th>Size</th><th>Stock</th></tr><tr><!-- c3 · list · on 75% of pages -->"
        "<td>S</td><!-- c4 · list · on 75% of pages --><td>3</td></tr></table>"
        "</main></body></html>",
        900,
    ),
    "t_T02": (
        [
            _cand("c1", "html > body > article > h2.headline", "mandatory"),
            _cand("c2", "html > body > article > p.byline", "noise"),
        ],
        "<html><body><article><!-- c1 · mandatory · on 100% of pages -->"
        '<h2 class="headline">Garden tips</h2></article></body></html>',
        31_000,
    ),
}


def _fake_analyze(run: RunDir, **kw: Any) -> None:
    for tid, (cands, html, tokens) in UNIONS.items():
        table = CandidateTable(template_id=tid, cluster_size=4, n_sampled=4, candidates=cands)
        write_json(run.candidates(tid), table.model_dump(by_alias=True))
        run.union_dir.mkdir(parents=True, exist_ok=True)
        run.union_html(tid).write_text(html, encoding="utf-8")
        write_json(
            run.union_meta(tid),
            {
                "template_id": tid,
                "tokens": tokens,
                "under_min": tokens < kw["min_tokens"],
                "base_page_id": 0,
                "donor_page_ids": [1],
                "spliced": ["c2"],
                "budget_skipped": ["c9"],
                "missing": [],
                "shown": ["c1", "c2"],
                "collapse_level": 0,
                "dropped": [],
            },
        )


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("an until: union evaluation must not touch the LLM layer")


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    root = make_tiny_site(tmp_path / "synth")
    (root / "truth" / "templates.json").write_text(json.dumps(TEMPLATES))
    return root


def _run(tmp: Path, site_root: Path, hooks: Hooks, **cfg_extra: Any) -> dict[str, Any]:
    cfg = {
        "name": "union_tiny",
        "dataset": "synthetic",
        "dataset_options": {"site_dir": str(site_root), "name": "tiny"},
        "method": {"params": {"until": "union", "min_cluster_size": 2}},
        **cfg_extra,
    }
    path = tmp / "union_tiny.yaml"
    path.write_text(json.dumps(cfg))
    datasets = tmp / "datasets"
    datasets.mkdir(exist_ok=True)
    out = run_eval(
        path,
        results_dir=tmp / "results",
        work_root=tmp / "work",
        datasets_dir=datasets,
        hooks=hooks,
    )
    results = dict(json.loads((out / "results.json").read_text()))
    with (out / "per_site.csv").open() as fh:
        results["_csv"] = list(csv.DictReader(fh))
    return results


def _hooks(fake: FakePipeline, **extra: Any) -> Hooks:
    return Hooks(
        ingest=fake.ingest,
        cluster=fake.cluster,
        analyze=extra.pop("analyze", _fake_analyze),
        continue_run=_fail,
        run_options=_fail,
        page_urls=fake.page_urls,
        read_html=fake.read_html,
        settings=_fail,
        resolve_model=_fail,
        estimate_cost=fake.estimate_cost,
        **extra,
    )


def test_union_run_reports_tokens_and_optional_field_coverage(tiny: Path, tmp_path: Path) -> None:
    results = _run(tmp_path, tiny, _hooks(FakePipeline(tiny, {})))
    assert results["status"] == "ok" and results["method_label"] == "ditto (union)"
    (site,) = results["sites"]
    union = site["union"]
    assert union["n_templates"] == 2 and union["max_tokens_limit"] == 30_000
    assert (union["max_tokens"], union["median_tokens"], union["min_tokens"]) == (
        31_000,
        15_950,
        900,
    )
    assert union["over_budget"] == 1 and union["under_min"] == 1
    assert union["with_missing_candidates"] == 0
    t01 = union["templates"][0]
    assert t01["template_id"] == "t_T01" and t01["spliced"] == ["c2"] and t01["shown"] == 2
    assert t01["budget_skipped"] == ["c9"]

    cov = union["coverage"]
    assert (cov["covered"], cov["optional_total"], cov["tail_templates_skipped"]) == (2, 3, 1)
    assert cov["never_present"] == 1  # "editor" is on no page: listed, not counted
    by_true = {t["true_template"]: t for t in cov["templates"]}
    assert by_true["T01"]["covered_fields"] == ["price", "sizes.stock"]
    assert by_true["T01"]["template_id"] == "t_T01" and by_true["T01"]["cluster_share"] == 1.0
    assert by_true["T02"]["never_present"] == ["editor"]
    (author,) = by_true["T02"]["missing"]
    assert author["field"] == "author" and author["reason"] == "not_shown"
    assert author["path"] == "html > body > article > p.byline"
    assert author["path_support"] == "2/2" and "c2 noise" in author["detail"]

    gate = results["summary"]["union"]
    assert gate["tokens"] == {"min": 900, "median": 15_950, "max": 31_000}
    assert gate["per_site"]["tiny"]["max_tokens"] == 31_000
    assert gate["coverage"]["missing"] == [
        {
            "site": "tiny",
            "true_template": "T02",
            "template_id": "t_T02",
            "field": "author",
            "reason": "not_shown",
        }
    ]
    assert gate["gate"] == {
        "all_sites_run": True,
        "tokens_under_max": False,
        "coverage_complete": False,
        "pass": False,
    }
    (row,) = results["_csv"]
    assert row["union_max_tokens"] == "31000" and row["union_optional_covered"] == "2"
    assert site["cost"]["calls"] == 0 and site["spend_usd"] == 0.0


def test_template_without_a_union_page_counts_as_missing(tiny: Path, tmp_path: Path) -> None:
    fake = FakePipeline(tiny, {}, merge_templates=True)  # one cluster "t_all", no union page
    results = _run(tmp_path, tiny, _hooks(fake))
    cov = results["sites"][0]["union"]["coverage"]
    assert cov["covered"] == 0 and cov["optional_total"] == 3
    reasons = {m["reason"] for t in cov["templates"] for m in t["missing"]}
    assert reasons == {"no_union_page"}


def test_site_without_truth_templates_reports_tokens_only(tmp_path: Path) -> None:
    root = make_tiny_site(tmp_path / "synth")  # its templates.json has no "templates" list
    results = _run(tmp_path, root, _hooks(FakePipeline(root, {})))
    (site,) = results["sites"]
    assert site["union"]["coverage"] is None
    gate = results["summary"]["union"]
    assert gate["coverage"] is None
    assert gate["gate"]["coverage_complete"] is None
    assert gate["gate"]["tokens_under_max"] is False  # t_T02 is over 30k tokens


def test_failed_site_fails_the_gate(tiny: Path, tmp_path: Path) -> None:
    def two_sites(dataset: str, selection: Selection, work_dir: Path) -> list[EvalSite]:
        return [read_site(tiny, name=n, work_dir=work_dir) for n in ("a", "b")]

    def analyze(run: RunDir, **kw: Any) -> None:
        if run.root.parent.name == "b":
            raise RuntimeError("boom")
        _fake_analyze(run, **kw)

    hooks = _hooks(FakePipeline(tiny, {}), analyze=analyze, load_sites=two_sites)
    results = _run(tmp_path, tiny, hooks, method={"params": {"until": "union"}})
    assert [s["status"] for s in results["sites"]] == ["ok", "failed"]
    gate = results["summary"]["union"]
    assert gate["n_sites"] == 1 and gate["gate"]["all_sites_run"] is False
    assert gate["gate"]["pass"] is False


def test_label_strings_shapes() -> None:
    assert label_strings(None) == [] and label_strings("") == []
    assert label_strings("x") == ["x"] and label_strings(3) == ["3"]
    assert label_strings(["a", None, "b"]) == ["a", "b"]
    assert label_strings({"Color": "Red", "Size": None}) == ["Red"]
    rows = [{"size": "S", "stock": "3"}, {"size": "M", "stock": None}]
    assert label_strings(rows) == ["S", "3", "M"]
    assert label_strings(rows, "stock") == ["3"]


def test_optional_fields_names_values_and_other_labels() -> None:
    tdef = {
        "fields": [
            {"name": "title", "optional": False, "presence": 1.0, "label": None},
            {"name": "rating", "optional": True, "presence": 0.4, "label": "Rating:"},
            {"name": "tags", "type": "list", "optional": False, "presence": 0.9, "label": "Tags"},
            {
                "name": "rows",
                "label": "Schedule",
                "columns": [
                    {"name": "time", "label": "Time", "presence": 1.0},
                    {"name": "room", "label": "Room", "optional": True, "presence": 0.02},
                ],
            },
        ]
    }
    labels = {
        "p1": {"rating": "4.5", "tags": ["a"], "rows": [{"time": "9", "room": "R1"}]},
        "p2": {"rating": None, "tags": [], "rows": [{"time": "10", "room": None}]},
    }
    fields = optional_fields(tdef, labels, ["p1", "p2", "p3"])
    assert [f.name for f in fields] == ["rating", "tags", "rows.room"]
    rating, tags, room = fields
    assert dict(rating.values) == {"p1": ["4.5"]}
    assert rating.other_labels == ("Room", "Schedule", "Tags", "Time")
    assert dict(tags.values) == {"p1": ["a"]}  # presence under 1 makes a list optional
    assert dict(room.values) == {"p1": ["R1"]}
    assert room.other_labels == ("Rating:", "Tags", "Time")


def test_truth_templates_file_is_found_next_to_the_site_input(tiny: Path, tmp_path: Path) -> None:
    site = read_site(tiny, name="tiny", work_dir=tmp_path / "w")
    assert truth_templates_file(site) == tiny / "truth" / "templates.json"
    site.input_paths = [tmp_path / "elsewhere" / "site"]
    assert truth_templates_file(site) is None
    site.meta["site_dir"] = str(tiny)
    assert truth_templates_file(site) == tiny / "truth" / "templates.json"
