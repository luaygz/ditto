"""On-disk fingerprints of every page in a run (CSR layout, memory-mapped on read).

Files under ``run.fingerprints_dir``::

    meta.json         {"version", "n_pages", "n_sig_values", "n_text_items", "n_paths", "errors"}
    sig_offsets.npy   int64[n_pages + 1]   page i's signature is sig_values[off[i]:off[i+1]]
    sig_values.u64    raw little-endian uint64, sorted unique element-path hashes per page
    sig_hash.npy      uint64[n_pages]      hash of each page's signature
    text_offsets.npy  int64[n_pages + 1]
    text_path.u64 / text_hash.u64 / text_count.u32   raw arrays, sorted by path hash per page
    path_hash.npy     uint64[n_paths] sorted
    paths.json        path strings aligned with path_hash.npy

Raw arrays are appended chunk by chunk while pages stream in, so writing never holds the whole
site in memory; reading maps them, so 100k pages cost a few hundred MB of page cache at most.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import orjson
from numpy.typing import NDArray

from ditto.fingerprint.tree import U32, U64, PageFingerprint, signature_hash
from ditto.run import RunDir
from ditto.textutil import hash64

VERSION = 1
I64 = NDArray[np.int64]

_FILES = (
    "meta.json",
    "sig_offsets.npy",
    "sig_values.u64",
    "sig_hash.npy",
    "text_offsets.npy",
    "text_path.u64",
    "text_hash.u64",
    "text_count.u32",
    "path_hash.npy",
    "paths.json",
)
_RAW: dict[str, tuple[str, np.dtype[Any]]] = {
    "sig_values": ("sig_values.u64", np.dtype("<u8")),
    "text_path": ("text_path.u64", np.dtype("<u8")),
    "text_hash": ("text_hash.u64", np.dtype("<u8")),
    "text_count": ("text_count.u32", np.dtype("<u4")),
}


def _map_raw(path: Path, dtype: np.dtype[Any], length: int) -> NDArray[Any]:
    if length == 0:
        return np.empty(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(length,))


class FingerprintWriter:
    """Streams page fingerprints (in page-id order) into a fingerprint directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in _FILES:
            (self.directory / name).unlink(missing_ok=True)
        self._files: dict[str, BinaryIO] = {
            key: open(self.directory / fname, "wb")  # noqa: SIM115 - closed in close()
            for key, (fname, _) in _RAW.items()
        }
        self._sig_lens: list[NDArray[np.int64]] = []
        self._text_lens: list[NDArray[np.int64]] = []
        self._sig_hashes: list[U64] = []
        self._paths: dict[int, str] = {}
        self._errors: dict[int, str] = {}
        self._n_pages = 0

    def add_chunk(
        self,
        sig_lens: NDArray[np.int64],
        sig_values: U64,
        sig_hashes: U64,
        text_lens: NDArray[np.int64],
        text_path: U64,
        text_hash: U64,
        text_count: U32,
        paths: Mapping[int, str] | Iterable[tuple[int, str]] = (),
        errors: Mapping[int, str] | None = None,
    ) -> None:
        """Append the fingerprints of the next ``len(sig_lens)`` pages."""
        self._files["sig_values"].write(np.ascontiguousarray(sig_values, "<u8").tobytes())
        self._files["text_path"].write(np.ascontiguousarray(text_path, "<u8").tobytes())
        self._files["text_hash"].write(np.ascontiguousarray(text_hash, "<u8").tobytes())
        self._files["text_count"].write(np.ascontiguousarray(text_count, "<u4").tobytes())
        self._sig_lens.append(np.asarray(sig_lens, dtype=np.int64))
        self._text_lens.append(np.asarray(text_lens, dtype=np.int64))
        self._sig_hashes.append(np.asarray(sig_hashes, dtype=np.uint64))
        items = paths.items() if isinstance(paths, Mapping) else paths
        self._paths.update(items)
        if errors:
            self._errors.update(errors)
        self._n_pages += len(sig_lens)

    def add(self, fp: PageFingerprint, error: str | None = None) -> None:
        """Append one page (convenience for small runs and tests)."""
        self.add_chunk(
            np.array([len(fp.signature)], dtype=np.int64),
            fp.signature,
            np.array([fp.signature_hash], dtype=np.uint64),
            np.array([len(fp.text_path)], dtype=np.int64),
            fp.text_path,
            fp.text_hash,
            fp.text_count,
            fp.paths,
            {self._n_pages: error} if error else None,
        )

    def close(self) -> FingerprintStore:
        for f in self._files.values():
            f.close()
        sig_offsets = _offsets(self._sig_lens)
        text_offsets = _offsets(self._text_lens)
        sig_hash = np.concatenate(self._sig_hashes) if self._sig_hashes else np.empty(0, np.uint64)
        np.save(self.directory / "sig_offsets.npy", sig_offsets)
        np.save(self.directory / "text_offsets.npy", text_offsets)
        np.save(self.directory / "sig_hash.npy", sig_hash)
        hashes = np.array(sorted(self._paths), dtype=np.uint64)
        np.save(self.directory / "path_hash.npy", hashes)
        (self.directory / "paths.json").write_bytes(
            orjson.dumps([self._paths[int(h)] for h in hashes])
        )
        meta = {
            "version": VERSION,
            "n_pages": self._n_pages,
            "n_sig_values": int(sig_offsets[-1]),
            "n_text_items": int(text_offsets[-1]),
            "n_paths": len(hashes),
            "errors": {str(k): v for k, v in sorted(self._errors.items())},
        }
        (self.directory / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        return FingerprintStore.load(self.directory)

    def abort(self) -> None:
        """Close the partial files after a failure (the directory is left without meta.json)."""
        for f in self._files.values():
            f.close()


def _offsets(lens: list[NDArray[np.int64]]) -> I64:
    out = np.zeros(sum(len(x) for x in lens) + 1, dtype=np.int64)
    if lens:
        np.cumsum(np.concatenate(lens), out=out[1:])
    return out


class FingerprintStore:
    """Read access to a run's fingerprints. Arrays are memory-mapped; per-page accessors return
    views. Safe to share with forked workers."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        meta = orjson.loads((self.directory / "meta.json").read_bytes())
        if meta.get("version") != VERSION:
            raise ValueError(f"unsupported fingerprint store version: {meta.get('version')}")
        self.n_pages: int = int(meta["n_pages"])
        self.errors: dict[int, str] = {int(k): v for k, v in meta.get("errors", {}).items()}
        self.sig_offsets: I64 = np.load(self.directory / "sig_offsets.npy")
        self.text_offsets: I64 = np.load(self.directory / "text_offsets.npy")
        self.signature_hashes: U64 = np.load(self.directory / "sig_hash.npy")
        n_sig, n_text = int(meta["n_sig_values"]), int(meta["n_text_items"])
        lengths = {"sig_values": n_sig, "text_path": n_text, "text_hash": n_text}
        lengths["text_count"] = n_text
        raw = {
            key: _map_raw(self.directory / fname, dtype, lengths[key])
            for key, (fname, dtype) in _RAW.items()
        }
        self.sig_values: U64 = raw["sig_values"]
        self.text_path: U64 = raw["text_path"]
        self.text_hash: U64 = raw["text_hash"]
        self.text_count: U32 = raw["text_count"]
        self._path_hash: U64 = np.load(self.directory / "path_hash.npy")
        self._path_strings: list[str] = orjson.loads((self.directory / "paths.json").read_bytes())

    @classmethod
    def open(cls, run: RunDir) -> FingerprintStore:
        if not (run.fingerprints_dir / "meta.json").exists():
            raise FileNotFoundError(
                f"{run.fingerprints_dir} has no fingerprints; run the fingerprint stage first"
            )
        return cls(run.fingerprints_dir)

    @classmethod
    def load(cls, directory: Path) -> FingerprintStore:
        return cls(directory)

    @classmethod
    def write(cls, directory: Path, fingerprints: Iterable[PageFingerprint]) -> FingerprintStore:
        """Write a store from in-memory fingerprints (page id = position)."""
        writer = FingerprintWriter(directory)
        for fp in fingerprints:
            writer.add(fp)
        return writer.close()

    def __len__(self) -> int:
        return self.n_pages

    def _check(self, page_id: int) -> None:
        if not 0 <= page_id < self.n_pages:
            raise KeyError(page_id)

    def signature(self, page_id: int) -> U64:
        """Sorted unique element-path hashes of a page."""
        self._check(page_id)
        return self.sig_values[self.sig_offsets[page_id] : self.sig_offsets[page_id + 1]]

    def signature_hash(self, page_id: int) -> int:
        self._check(page_id)
        return int(self.signature_hashes[page_id])

    def text_items(self, page_id: int) -> tuple[U64, U64, U32]:
        """``(path_hash, text_hash, count)`` arrays of a page, sorted by path hash."""
        self._check(page_id)
        lo, hi = self.text_offsets[page_id], self.text_offsets[page_id + 1]
        return self.text_path[lo:hi], self.text_hash[lo:hi], self.text_count[lo:hi]

    def fingerprint(self, page_id: int) -> PageFingerprint:
        """A page's fingerprint with its own path vocabulary (copies; for inspection/tests)."""
        sig = np.array(self.signature(page_id))
        tp, th, tc = self.text_items(page_id)
        tp, th, tc = tp.copy(), th.copy(), tc.copy()
        hashes = np.union1d(sig, tp)
        return PageFingerprint(sig, tp, th, tc, {int(h): self.path_of(int(h)) for h in hashes})

    @property
    def path_hashes(self) -> U64:
        """Every path hash in the vocabulary, sorted."""
        return self._path_hash

    def path_of(self, h: int) -> str:
        """Path string of a path hash (``KeyError`` if unknown)."""
        i = int(np.searchsorted(self._path_hash, np.uint64(h)))
        if i >= len(self._path_hash) or int(self._path_hash[i]) != int(h):
            raise KeyError(h)
        return self._path_strings[i]

    @staticmethod
    def hash_of(path: str) -> int:
        return hash64(path)

    def paths(self) -> dict[int, str]:
        """The whole vocabulary as a dict (hash -> path)."""
        return {int(h): p for h, p in zip(self._path_hash, self._path_strings, strict=True)}


__all__ = ["FingerprintStore", "FingerprintWriter", "signature_hash"]
