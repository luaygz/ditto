from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ditto.fingerprint import FingerprintStore, fingerprint_html, fingerprint_run
from ditto.ingest import PageStore, ingest
from ditto.run import RunDir


def _site(root: Path, n: int = 150) -> Path:
    site = root / "site"
    site.mkdir()
    rows = []
    for i in range(n):
        if i % 3 == 0:
            html = (
                f"<html><body><h1>Item {i}</h1><dl><dt>Price</dt><dd>${i}</dd></dl></body></html>"
            )
        else:
            items = "".join(f"<li><a href='/p/{j}'>p{j}</a></li>" for j in range(i % 7 + 1))
            html = f"<html><body><h2 class='cat'>Cat {i}</h2><ul>{items}</ul></body></html>"
        (site / f"page{i:04d}.html").write_text(html)
        rows.append(f"page{i:04d}.html\thttps://ex.com/x/{i}")
    (site / "urls.tsv").write_text("\n".join(rows) + "\n")
    return site


def test_fingerprint_run_matches_per_page_fingerprints(tmp_path: Path) -> None:
    site = _site(tmp_path)
    run = RunDir(tmp_path / "run")
    ingest(site, run.root, workers=1)
    store = fingerprint_run(run, workers=2)
    pages = PageStore.open(run)
    assert store.n_pages == len(pages) == 150
    for pid in (0, 1, 77, 149):
        fp = fingerprint_html(pages.read_html(pid))
        assert np.array_equal(store.signature(pid), fp.signature)
        assert np.array_equal(store.text_items(pid)[1], fp.text_hash)
        for h, path in fp.paths.items():
            assert store.path_of(h) == path
    serial = fingerprint_run(run, workers=1)
    assert np.array_equal(serial.signature_hashes, store.signature_hashes)
    assert len(serial.path_hashes) == len(FingerprintStore.open(run).path_hashes)
    timings = json.loads(run.timings_json.read_text())
    assert timings["fingerprint"]["pages"] == 150
