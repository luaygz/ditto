"""Clustering quality against true template labels: adjusted Rand index, purity, over- and
under-merge counts. Pure numpy.

Labels are any hashable values. A predicted label of ``None`` (a page the clustering did not
assign) counts as its own singleton cluster.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray

I64 = NDArray[np.int64]


def _codes(labels: Sequence[Hashable | None], *, none_singleton: bool) -> I64:
    index: dict[object, int] = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        key: object = ("\x00none", i) if label is None and none_singleton else label
        code = index.get(key)
        if code is None:
            code = index[key] = len(index)
        out[i] = code
    return out


def _contingency(
    true_labels: Sequence[Hashable], pred_labels: Sequence[Hashable | None]
) -> tuple[I64, I64, I64, int, int]:
    """Sparse contingency table: (true code, pred code, count) per non-empty cell."""
    if len(true_labels) != len(pred_labels):
        raise ValueError(f"{len(true_labels)} true labels but {len(pred_labels)} predicted")
    t = _codes(true_labels, none_singleton=False)
    p = _codes(pred_labels, none_singleton=True)
    n_true = int(t.max()) + 1 if len(t) else 0
    n_pred = int(p.max()) + 1 if len(p) else 0
    cells, counts = np.unique(t * max(n_pred, 1) + p, return_counts=True)
    return cells // max(n_pred, 1), cells % max(n_pred, 1), counts.astype(np.int64), n_true, n_pred


def _comb2(x: NDArray[np.int64]) -> float:
    xf = x.astype(np.float64)
    return float((xf * (xf - 1.0) / 2.0).sum())


def adjusted_rand_index(
    true_labels: Sequence[Hashable], pred_labels: Sequence[Hashable | None]
) -> float:
    """Hubert-Arabie adjusted Rand index (1.0 = identical partitions, ~0 = random)."""
    ti, pi, counts, n_true, n_pred = _contingency(true_labels, pred_labels)
    n = int(counts.sum())
    if n_true == n_pred and n_true in (0, 1, n):
        return 1.0
    sum_cells = _comb2(counts)
    sum_true = _comb2(np.bincount(ti, weights=counts, minlength=n_true).astype(np.int64))
    sum_pred = _comb2(np.bincount(pi, weights=counts, minlength=n_pred).astype(np.int64))
    expected = sum_true * sum_pred / (n * (n - 1) / 2.0)
    max_index = (sum_true + sum_pred) / 2.0
    if max_index == expected:
        return 1.0 if sum_cells == expected else 0.0
    return float((sum_cells - expected) / (max_index - expected))


def _majority_share(keys: I64, counts: I64, n_keys: int) -> float:
    if not len(counts):
        return 1.0
    best = np.zeros(n_keys, dtype=np.int64)
    np.maximum.at(best, keys, counts)
    return float(best.sum() / counts.sum())


def purity(true_labels: Sequence[Hashable], pred_labels: Sequence[Hashable | None]) -> float:
    """Share of pages whose predicted cluster's majority true label is their own."""
    _, pi, counts, _, n_pred = _contingency(true_labels, pred_labels)
    return _majority_share(pi, counts, n_pred)


def clustering_metrics(
    true_labels: Sequence[Hashable], pred_labels: Sequence[Hashable | None]
) -> dict[str, float | int]:
    """All clustering metrics at once.

    * ``ari``: adjusted Rand index.
    * ``purity``: share of pages in their predicted cluster's majority true template.
    * ``inverse_purity``: share of pages in their true template's majority predicted cluster.
    * ``over_merge``: predicted clusters that mix pages of two or more true templates.
    * ``under_merge``: true templates whose pages are spread over two or more predicted clusters.
    """
    ti, pi, counts, n_true, n_pred = _contingency(true_labels, pred_labels)
    true_per_pred = np.bincount(pi, minlength=n_pred)
    pred_per_true = np.bincount(ti, minlength=n_true)
    return {
        "ari": adjusted_rand_index(true_labels, pred_labels),
        "purity": _majority_share(pi, counts, n_pred),
        "inverse_purity": _majority_share(ti, counts, n_true),
        "over_merge": int((true_per_pred >= 2).sum()),
        "under_merge": int((pred_per_true >= 2).sum()),
        "n_pages": int(counts.sum()),
        "n_true": n_true,
        "n_pred": n_pred,
    }
