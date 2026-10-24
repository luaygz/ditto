from __future__ import annotations

from pathlib import Path

import pytest
from analyze_sites import product_page

from ditto.analyze import CandidateTable, analyze_pages, analyze_run
from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.ingest import PageStore, ingest
from ditto.run import RunDir, read_json

TID = "t_abcd"
TAIL = "t_tail"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> RunDir:
    """A 36-page run: one template of 32 pages in two signature groups, one tail of 4."""
    src = tmp_path_factory.mktemp("site")
    for i in range(36):
        html = product_page(i) if i < 32 else f"<html><body><h1>Other {i}</h1></body></html>"
        (src / f"p{i:03d}.html").write_text(html)
    out = tmp_path_factory.mktemp("run")
    ingest(src, out, workers=1)
    rundir = RunDir(out)
    store = PageStore.open(rundir)
    by_name = {Path(store.url(pid)).stem: pid for pid in store.ids()}
    main = sorted(by_name[f"p{i:03d}"] for i in range(32))
    tail = sorted(by_name[f"p{i:03d}"] for i in range(32, 36))
    groups = [
        SignatureGroup(id=0, sig_hash="0", size=20, template_id=TID, page_ids=main[:20]),
        SignatureGroup(id=1, sig_hash="1", size=12, template_id=TID, page_ids=main[20:]),
        SignatureGroup(id=2, sig_hash="2", size=4, template_id=TAIL, page_ids=tail),
    ]
    Clusters(
        n_pages=36,
        min_cluster_size=20,
        clusters=[
            Cluster(
                template_id=TID, size=32, is_tail=False, signature_groups=[0, 1], page_ids=main
            ),
            Cluster(template_id=TAIL, size=4, is_tail=True, signature_groups=[2], page_ids=tail),
        ],
        signature_groups=groups,
    ).save(rundir.clusters_json)
    return rundir


def test_analyze_run_writes_candidates_and_timing(run: RunDir) -> None:
    result = analyze_run(run, workers=1)
    assert list(result) == [TID]  # tail clusters only on request
    table = result[TID]
    assert (table.template_id, table.cluster_size, table.n_sampled) == (TID, 32, 32)
    assert CandidateTable.load(run.candidates(TID)) == table
    assert {t.layout for t in table.tables} >= {"dl", "table", "pairs", "rows"}
    timing = read_json(run.timings_json)["analyze"]
    assert timing["pages"] == 32 and timing["workers"] == 1


def test_analyze_run_parallel_matches_in_process(run: RunDir) -> None:
    serial = analyze_run(run, workers=1, sample_size=16)[TID]
    parallel = analyze_run(run, workers=3, sample_size=16)[TID]
    assert serial == parallel
    assert serial.n_sampled == 16
    store = PageStore.open(run)
    clusters = Clusters.load(run.clusters_json)
    sample = clusters.sample(TID, 16)
    pages = [(pid, store.read_html(pid)) for pid in sample]
    assert analyze_pages(TID, pages, cluster_size=32) == serial


def test_analyze_run_selected_and_tail_templates(run: RunDir) -> None:
    result = analyze_run(run, template_ids=[TAIL], workers=1)
    assert list(result) == [TAIL]
    assert run.candidates(TAIL).exists()
    both = analyze_run(run, include_tail=True, workers=1)
    assert sorted(both) == sorted([TID, TAIL])
    with pytest.raises(KeyError):
        analyze_run(run, template_ids=["t_missing"], workers=1)
