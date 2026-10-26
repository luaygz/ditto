from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from cluster_pages import article, product

from ditto.cluster import Clusters, cluster_run
from ditto.evaluation.clustering import clustering_metrics
from ditto.fingerprint import fingerprint_run
from ditto.ingest import PageStore, ingest
from ditto.run import RunDir


def write_site(root: Path) -> tuple[Path, dict[str, str]]:
    rng = random.Random(11)
    site = root / "site"
    site.mkdir()
    rows, truth = [], {}
    for i in range(90):
        kind = "product" if i % 3 else "article"
        html = product(i, rng) if kind == "product" else article(i, rng)
        url = (
            f"https://shop.example/products/thing-{i}"
            if kind == "product"
            else f"https://shop.example/news/{i}"
        )
        (site / f"p{i:03d}.html").write_text(html)
        rows.append(f"p{i:03d}.html\t{url}")
        truth[url] = kind
    (site / "urls.tsv").write_text("\n".join(rows) + "\n")
    return site, truth


def test_cluster_run_end_to_end(tmp_path: Path) -> None:
    site, truth = write_site(tmp_path)
    run = RunDir(tmp_path / "run")
    ingest(site, run.root, workers=1)
    fingerprint_run(run, workers=1)
    clusters = cluster_run(run, min_cluster_size=10, workers=1)
    assert run.clusters_json.exists() and run.cluster_report.exists()
    loaded = Clusters.load(run.clusters_json)
    assert loaded.model_dump() == clusters.model_dump()
    pages = PageStore.open(run)
    true = [truth[pages.url(pid)] for pid in range(len(pages))]
    pred = [loaded.template_of(pid) for pid in range(len(pages))]
    metrics = clustering_metrics(true, pred)
    assert metrics["ari"] == 1.0
    assert metrics["n_pred"] == 2
    assert loaded.params["threshold"] == 0.75
    report = run.cluster_report.read_text()
    assert "## Templates" in report
    assert "html > body.page" in report
    timings = json.loads(run.timings_json.read_text())
    assert timings["cluster"]["pages"] == 90


def test_cluster_run_requires_fresh_fingerprints(tmp_path: Path) -> None:
    site, _ = write_site(tmp_path)
    run = RunDir(tmp_path / "run")
    ingest(site, run.root, workers=1)
    with pytest.raises(FileNotFoundError):
        cluster_run(run)
    fingerprint_run(run, workers=1)
    (site / "extra.html").write_text("<html><body><p>new page</p></body></html>")
    ingest(site, run.root, workers=1)
    with pytest.raises(ValueError, match="rerun the fingerprint stage"):
        cluster_run(run)
