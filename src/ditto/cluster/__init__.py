"""Stage 3: template clustering (signature groups, chrome filter, MinHash/LSH, URL tie-break)."""

from __future__ import annotations

from ditto.cluster.core import ClusterParams, ClusterResult, cluster_signatures
from ditto.cluster.models import Cluster, Clusters, SignatureGroup
from ditto.cluster.report import render_cluster_report
from ditto.cluster.stage import cluster_run
from ditto.cluster.urls import url_pattern

__all__ = [
    "Cluster",
    "ClusterParams",
    "ClusterResult",
    "Clusters",
    "SignatureGroup",
    "cluster_run",
    "cluster_signatures",
    "render_cluster_report",
    "url_pattern",
]
