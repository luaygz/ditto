from __future__ import annotations

import numpy as np
import pytest

from ditto.evaluation.clustering import adjusted_rand_index, clustering_metrics, purity


def test_identical_partitions_up_to_renaming() -> None:
    true = ["a", "a", "b", "b", "c"]
    pred = ["t_1", "t_1", "t_2", "t_2", "t_3"]
    assert adjusted_rand_index(true, pred) == 1.0
    m = clustering_metrics(true, pred)
    assert m == {
        "ari": 1.0,
        "purity": 1.0,
        "inverse_purity": 1.0,
        "over_merge": 0,
        "under_merge": 0,
        "n_pages": 5,
        "n_true": 3,
        "n_pred": 3,
    }


def test_known_ari_values() -> None:
    # hand-computed: cells C2 sum 2, rows 6, cols 3, expected 6*3/15 = 1.2, max 4.5
    assert adjusted_rand_index([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]) == pytest.approx(0.8 / 3.3)
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    # sklearn documentation example
    assert adjusted_rand_index([0, 0, 1, 2], [0, 0, 1, 1]) == pytest.approx(0.5714285714)
    # symmetric
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(0.5714285714)


def test_degenerate_partitions() -> None:
    assert adjusted_rand_index([], []) == 1.0
    assert adjusted_rand_index([1, 1, 1], ["x", "x", "x"]) == 1.0
    assert adjusted_rand_index([1, 2, 3], ["x", "y", "z"]) == 1.0
    assert adjusted_rand_index([1, 1, 1], ["x", "y", "z"]) == 0.0
    assert adjusted_rand_index(["a"], ["b"]) == 1.0


def test_over_and_under_merge() -> None:
    true = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    pred = ["x"] * 4 + ["x"] * 2 + ["y"] * 2 + ["z"] * 2  # a+b merged, b split
    m = clustering_metrics(true, pred)
    assert m["over_merge"] == 1  # x mixes a and b
    assert m["under_merge"] == 1  # b spread over x and y
    assert m["purity"] == pytest.approx((4 + 2 + 2) / 10)
    assert m["inverse_purity"] == pytest.approx((4 + 2 + 2) / 10)
    assert purity(true, pred) == m["purity"]
    assert 0 < m["ari"] < 1


def test_none_predictions_are_singletons() -> None:
    true = ["a", "a", "a", "b"]
    m = clustering_metrics(true, ["t", None, None, "u"])
    assert m["n_pred"] == 4
    assert m["under_merge"] == 1
    assert m["purity"] == 1.0
    assert m["inverse_purity"] == pytest.approx(2 / 4)


def test_random_labels_score_near_zero() -> None:
    rng = np.random.default_rng(0)
    true = rng.integers(0, 10, 5000).tolist()
    pred = rng.integers(0, 10, 5000).tolist()
    assert abs(adjusted_rand_index(true, pred)) < 0.01


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        clustering_metrics(["a"], [])
