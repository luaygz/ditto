from __future__ import annotations

import random

import numpy as np
import pytest
from cluster_pages import article, page, product

from ditto.cluster import ClusterParams, cluster_signatures, render_cluster_report
from ditto.cluster.core import _assign_id
from ditto.evaluation.clustering import adjusted_rand_index
from ditto.fingerprint import PageFingerprint, fingerprint_html


def build_site(
    n_products: int = 60, n_articles: int = 40, seed: int = 0
) -> tuple[list[PageFingerprint], list[str], list[str]]:
    rng = random.Random(seed)
    fps, urls, truth = [], [], []
    for i in range(n_products + n_articles):
        if i < n_products:
            fps.append(fingerprint_html(product(i, rng)))
            urls.append(f"https://shop.example/products/item-{i}-blue")
            truth.append("product")
        else:
            fps.append(fingerprint_html(article(i, rng)))
            urls.append(f"https://shop.example/news/{2020 + i % 5}/{i}")
            truth.append("article")
    order = list(range(len(fps)))
    rng.shuffle(order)
    return [fps[i] for i in order], [urls[i] for i in order], [truth[i] for i in order]


def run(fps: list[PageFingerprint], urls: list[str], **kw: float):
    hashes = np.array([fp.signature_hash for fp in fps], dtype=np.uint64)
    params = ClusterParams(**kw)  # type: ignore[arg-type]
    return cluster_signatures(hashes, lambda pid: fps[pid].signature, urls, params)


def test_two_templates_and_chrome() -> None:
    fps, urls, truth = build_site()
    result = run(fps, urls, min_cluster_size=5)
    c = result.clusters
    pred = [c.template_of(pid) for pid in range(len(fps))]
    assert adjusted_rand_index(truth, pred) == 1.0
    assert len(c.templates()) == 2
    assert c.n_pages == 100
    # several signature groups (optional parts) but two templates
    assert len(c.signature_groups) > 2
    assert sum(g.size for g in c.signature_groups) == 100
    # chrome (header, nav, footer, html, body) was detected
    assert result.stats.chrome_paths
    for t in c.templates():
        assert t.template_id.startswith("t_") and len(t.template_id) == 6
        assert t.size == len(t.page_ids)
        assert not t.is_tail
        assert t.url_patterns and t.url_examples
    products = next(t for t in c.clusters if "products" in t.url_patterns[0])
    assert products.url_patterns == ["shop.example/products/{slug}"]
    assert products.size == 60
    for g in c.signature_groups:
        assert c.get(g.template_id).signature_groups.count(g.id) == 1


def test_deterministic_and_order_independent() -> None:
    fps, urls, _ = build_site(seed=3)
    a = run(fps, urls).clusters
    b = run(fps, urls).clusters
    assert a.model_dump() == b.model_dump()
    perm = list(range(len(fps)))[::-1]
    c = run([fps[i] for i in perm], [urls[i] for i in perm]).clusters
    assert {t.template_id for t in a.clusters} == {t.template_id for t in c.clusters}
    for new_pid, old_pid in enumerate(perm):
        assert c.template_of(new_pid) == a.template_of(old_pid)


def test_tail_clusters_are_flagged() -> None:
    fps, urls, _ = build_site(n_products=30, n_articles=8)
    c = run(fps, urls, min_cluster_size=20).clusters
    sizes = sorted((t.size, t.is_tail) for t in c.clusters)
    assert sizes == [(8, True), (30, False)]
    assert [t.size for t in c.templates()] == [30]
    assert [t.size for t in c.templates(include_tail=True)] == [30, 8]


def blocks(names: list[str], extra: str = "") -> str:
    """A page whose content is one div per class name (one distinct path each)."""
    return page("".join(f"<div class='{n}'>{n}</div>" for n in names) + extra)


def band_site(same_pattern: bool) -> tuple[list[PageFingerprint], list[str], list[str]]:
    """Groups A and B share 8 of 12 content paths (J = 0.667, inside the 0.6-0.75 band);
    C is unrelated and keeps the shared paths under the chrome cut."""
    a = [f"c{i}" for i in range(10)]
    b = [f"c{i}" for i in range(8)] + ["e0", "e1"]
    c = [f"z{i}" for i in range(10)]
    fps, urls, truth = [], [], []
    for i in range(25):
        fps.append(fingerprint_html(blocks(a)))
        urls.append(f"https://ex.com/item/{i}")
        truth.append("A")
        fps.append(fingerprint_html(blocks(b)))
        urls.append(f"https://ex.com/item/{100 + i}" if same_pattern else f"https://ex.com/n/{i}")
        truth.append("B")
        fps.append(fingerprint_html(blocks(c)))
        urls.append(f"https://ex.com/c/{i}")
        truth.append("C")
    return fps, urls, truth


def test_band_pairs_merge_only_when_url_patterns_agree() -> None:
    fps, urls, _ = band_site(same_pattern=True)
    result = run(fps, urls)
    assert len(result.clusters.clusters) == 2
    assert len(result.stats.band_merges) == 1
    edge, pattern = result.stats.band_merges[0]
    assert edge.jaccard == pytest.approx(8 / 12)
    assert pattern == "ex.com/item/{n}"

    fps, urls, _ = band_site(same_pattern=False)
    result = run(fps, urls)
    assert len(result.clusters.clusters) == 3
    assert not result.stats.band_merges
    assert len(result.stats.band_rejects) == 1
    # above the band no URL agreement is needed; below it nothing merges
    assert len(run(fps, urls, threshold=0.65, band_low=0.6).clusters.clusters) == 2
    assert len(run(*band_site(True)[:2], threshold=0.8, band_low=0.7).clusters.clusters) == 3


def test_uninformative_url_pattern_does_not_merge() -> None:
    fps, _, _ = band_site(same_pattern=True)
    urls = [f"https://ex.com/p{i}.html" for i in range(len(fps))]
    assert len(run(fps, urls).clusters.clusters) == 3


def test_dominant_template_is_not_split_by_chrome_filter() -> None:
    """A single-template site: its own structure is on > 90% of pages, so the chrome cut
    removes it and only optional sections remain; full signatures must decide."""
    rng = random.Random(7)
    base = [f"b{i}" for i in range(20)]
    fps, urls, truth = [], [], []
    for i in range(80):
        opts = [f"o{k}" for k in range(4) if rng.random() < 0.5]
        fps.append(fingerprint_html(blocks(base + opts)))
        urls.append(f"https://ex.com/d/{i}")
        truth.append("detail")
    for i in range(4):
        fps.append(fingerprint_html(blocks([f"x{k}" for k in range(6)])))
        urls.append(f"https://ex.com/x/{i}")
        truth.append("other")
    result = run(fps, urls, min_cluster_size=3)
    assert result.stats.n_fallback_groups > 0
    pred = [result.clusters.template_of(p) for p in range(len(fps))]
    assert adjusted_rand_index(truth, pred) == 1.0


def test_empty_and_identical_signatures() -> None:
    empty = PageFingerprint.empty()
    fps = [empty, empty, fingerprint_html(blocks(["a"]))]
    result = run(fps, ["u0", "u1", "u2"], min_cluster_size=1)
    assert result.clusters.template_of(0) == result.clusters.template_of(1)
    assert result.clusters.template_of(0) != result.clusters.template_of(2)
    none = cluster_signatures(np.empty(0, np.uint64), lambda pid: fps[pid].signature, [])
    assert none.clusters.clusters == [] and none.clusters.n_pages == 0


def test_errors() -> None:
    with pytest.raises(ValueError):
        ClusterParams(threshold=0.5, band_low=0.6)
    with pytest.raises(ValueError):
        ClusterParams(chrome_df=0)
    with pytest.raises(ValueError):
        ClusterParams(num_perm=4)
    fps = [fingerprint_html(blocks(["a"]))]
    with pytest.raises(ValueError):
        run(fps, [])


def test_assign_id_extends_on_collision() -> None:
    used: set[str] = set()
    assert _assign_id("abcd000000000001", used) == "t_abcd"
    assert _assign_id("abcd000000000002", used) == "t_abcd0"
    assert _assign_id("abcd000000000003", used) == "t_abcd00"
    assert _assign_id("abcd000000000001", {"t_abcd", "t_abcd0", "t_abcd00"} | used) != ""
    full = {f"t_{'abcd000000000001'[:k]}" for k in range(4, 17)}
    assert _assign_id("abcd000000000001", full) == "t_abcd000000000001_2"


def test_report_mentions_templates_and_chrome() -> None:
    fps, urls, _ = band_site(same_pattern=False)
    result = run(fps, urls)
    paths = {h: p for fp in fps for h, p in fp.paths.items()}
    text = render_cluster_report(result, paths.__getitem__, site="ex.com")
    assert text.startswith("# Cluster report: ex.com")
    for t in result.clusters.clusters:
        assert t.template_id in text
    assert "html > body.page > header.site" in text
    assert "kept apart" in text
