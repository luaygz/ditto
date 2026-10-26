from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ditto.fingerprint import FingerprintStore, FingerprintWriter, fingerprint_html
from ditto.fingerprint.tree import PageFingerprint
from ditto.run import RunDir
from ditto.textutil import hash64

PAGES = [
    "<html><body><h1>A</h1><p class='x'>one</p><a href='/a'>go</a></body></html>",
    "<html><body><h1>B</h1><ul><li>1</li><li>2</li></ul></body></html>",
    "<html><body></body></html>",
    "<html><head><title>T</title></head><body><div class='card'><b>x</b></div></body></html>",
]


def test_round_trip(tmp_path: Path) -> None:
    fps = [fingerprint_html(h) for h in PAGES]
    store = FingerprintStore.write(tmp_path / "fp", fps)
    again = FingerprintStore.load(tmp_path / "fp")
    for s in (store, again):
        assert s.n_pages == len(PAGES) == len(s)
        for pid, fp in enumerate(fps):
            assert np.array_equal(s.signature(pid), fp.signature)
            assert s.signature_hash(pid) == fp.signature_hash
            tp, th, tc = s.text_items(pid)
            assert np.array_equal(tp, fp.text_path)
            assert np.array_equal(th, fp.text_hash)
            assert np.array_equal(tc, fp.text_count)
            for h, path in fp.paths.items():
                assert s.path_of(h) == path
            back = s.fingerprint(pid)
            assert back.paths == fp.paths
            assert np.array_equal(back.signature, fp.signature)
    assert store.hash_of("html > body") == hash64("html > body")
    assert len(store.paths()) == len(store.path_hashes)
    assert np.array_equal(store.signature_hashes, [fp.signature_hash for fp in fps])


def test_errors_unknown_ids_and_paths(tmp_path: Path) -> None:
    writer = FingerprintWriter(tmp_path / "fp")
    writer.add(fingerprint_html(PAGES[0]))
    writer.add(PageFingerprint.empty(), error="ParserError: boom")
    store = writer.close()
    assert store.errors == {1: "ParserError: boom"}
    assert len(store.signature(1)) == 0
    with pytest.raises(KeyError):
        store.signature(2)
    with pytest.raises(KeyError):
        store.text_items(-1)
    with pytest.raises(KeyError):
        store.path_of(12345)


def test_chunked_writes_match_single_writes(tmp_path: Path) -> None:
    fps = [fingerprint_html(h) for h in PAGES]
    writer = FingerprintWriter(tmp_path / "chunked")
    writer.add_chunk(
        np.array([len(f.signature) for f in fps]),
        np.concatenate([f.signature for f in fps]),
        np.array([f.signature_hash for f in fps], dtype=np.uint64),
        np.array([len(f.text_path) for f in fps]),
        np.concatenate([f.text_path for f in fps]),
        np.concatenate([f.text_hash for f in fps]),
        np.concatenate([f.text_count for f in fps]),
        [(h, p) for f in fps for h, p in f.paths.items()],
    )
    chunked = writer.close()
    single = FingerprintStore.write(tmp_path / "single", fps)
    for pid in range(len(fps)):
        assert np.array_equal(chunked.signature(pid), single.signature(pid))
        assert np.array_equal(chunked.text_items(pid)[1], single.text_items(pid)[1])


def test_empty_store_and_missing_run(tmp_path: Path) -> None:
    store = FingerprintStore.write(tmp_path / "fp", [])
    assert store.n_pages == 0
    assert len(store.sig_values) == 0
    with pytest.raises(FileNotFoundError):
        FingerprintStore.open(RunDir(tmp_path / "nothing"))


def test_rewrite_replaces_old_files(tmp_path: Path) -> None:
    FingerprintStore.write(tmp_path / "fp", [fingerprint_html(h) for h in PAGES])
    store = FingerprintStore.write(tmp_path / "fp", [fingerprint_html(PAGES[1])])
    assert store.n_pages == 1
    assert store.signature_hash(0) == fingerprint_html(PAGES[1]).signature_hash
