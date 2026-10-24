"""build_union_run over a real (ingested) run directory."""

from __future__ import annotations

from pathlib import Path

from union_fixtures import profile_page, profile_table

from ditto.cluster.models import Cluster, Clusters
from ditto.ingest import PageStore, ingest
from ditto.run import RunDir, read_json
from ditto.union import build_union_run


def _make_run(tmp_path: Path) -> RunDir:
    site = tmp_path / "site"
    site.mkdir()
    pages = {
        "ann.html": profile_page("Ann Lee", pubs=["A", "B", "C"]),
        "bo.html": profile_page("Bo Chen", email="bo@example.org"),
        "cy.html": profile_page(
            "Cy Diaz", specs=[("Department", "HR"), ("Fax", "555"), ("Office", "C3")]
        ),
        "di.html": profile_page("Di Ek", awards=["Prize"]),
    }
    for name, html in pages.items():
        (site / name).write_text(html, encoding="utf-8")
    (site / "urls.tsv").write_text(
        "".join(f"{name}\thttps://uni.example/staff/{name[:-5]}\n" for name in pages)
    )
    run = ingest(site, tmp_path / "run", workers=1).run
    ids = PageStore.open(run).ids()
    Clusters(
        n_pages=len(ids),
        min_cluster_size=2,
        clusters=[
            Cluster(
                template_id="t_test",
                size=len(ids),
                is_tail=False,
                signature_groups=[],
                page_ids=ids,
            ),
            Cluster(template_id="t_tail", size=1, is_tail=True, signature_groups=[], page_ids=[]),
        ],
        signature_groups=[],
    ).save(run.clusters_json)
    profile_table().save(run.candidates("t_test"))
    return run


def test_build_union_run_writes_pages_and_meta(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    results = build_union_run(run, min_tokens=0, workers=1)
    assert set(results) == {"t_test"}  # tail templates are skipped
    page = results["t_test"]
    assert run.union_html("t_test").read_text(encoding="utf-8") == page.html
    meta = read_json(run.union_meta("t_test"))
    assert meta["tokens"] == page.tokens
    assert meta["base_page_id"] == page.base_page_id
    assert meta["base_url"].startswith("https://uni.example/staff/")
    assert len(meta["donor_urls"]) == len(meta["donor_page_ids"]) >= 1
    # the base (Bo: email text + href) lacks fax and awards, which come from donors
    assert meta["base_url"] == "https://uni.example/staff/bo"
    assert sorted(meta["spliced"]) == ["c10", "c8"]
    assert meta["missing"] == []
    assert meta["focus"] is None
    assert "union" in read_json(run.timings_json)


def test_build_union_run_with_focus_overwrites_one_template(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    build_union_run(run, min_tokens=0, workers=1)
    results = build_union_run(
        run, template_ids=["t_test"], focus={"t_test": {"c1"}}, min_tokens=0, workers=1
    )
    meta = read_json(run.union_meta("t_test"))
    assert meta["focus"] == ["c1"]
    assert "c1 · mandatory" in results["t_test"].html
    assert "c9 · list" not in run.union_html("t_test").read_text(encoding="utf-8")
