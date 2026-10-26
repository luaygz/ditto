"""Stage 3 over a run: read fingerprints and URLs, cluster, write clusters.json and the report."""

from __future__ import annotations

import time

from ditto.cluster.core import ClusterParams, cluster_signatures
from ditto.cluster.models import Clusters
from ditto.cluster.report import render_cluster_report
from ditto.fingerprint.store import FingerprintStore
from ditto.parallel import default_workers
from ditto.run import RunDir


def cluster_run(
    run: RunDir,
    *,
    min_cluster_size: int = 20,
    threshold: float = 0.75,
    band_low: float = 0.6,
    chrome_df: float = 0.9,
    num_perm: int = 128,
    seed: int = 1,
    workers: int | None = None,
) -> Clusters:
    """Cluster a fingerprinted run into templates; writes ``clusters.json`` and
    ``cluster_report.md``. ``workers`` only parallelizes URL-pattern extraction on big runs."""
    from ditto.ingest import PageStore

    start = time.perf_counter()
    params = ClusterParams(min_cluster_size, threshold, band_low, chrome_df, num_perm, seed)
    store = FingerprintStore.open(run)
    pages = PageStore.open(run)
    if len(pages) != store.n_pages:
        raise ValueError(
            f"{store.n_pages} fingerprinted pages but {len(pages)} ingested; "
            "rerun the fingerprint stage"
        )
    urls = [rec.url for rec in pages.records()]
    n_workers = default_workers() if workers is None else max(1, workers)
    result = cluster_signatures(
        store.signature_hashes, store.signature, urls, params, workers=n_workers
    )
    result.clusters.save(run.clusters_json)
    run.cluster_report.write_text(render_cluster_report(result, store.path_of, run.site))
    run.record_timing("cluster", time.perf_counter() - start, store.n_pages, 1)
    return result.clusters
