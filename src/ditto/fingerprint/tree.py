"""Stage 2 per-page fingerprint: the structural signature and the text items of one cleaned tree.

* ``signature``: sorted unique ``hash64`` of every element's structural path. Paths carry no
  positional indices, so repeated siblings (same parent, same child path) collapse to one entry.
* text items, one per distinct text path on the page, sorted by path hash:
  ``text_path = hash64(path)`` where ``path`` is the structural path of a node with own text, or
  that path plus ``" @href"`` / ``" @src"`` for link attributes;
  ``text_hash = hash64("\\x1f".join(texts of those nodes in document order))``;
  ``text_count`` = number of text-bearing nodes (or attribute values) with that path.
* ``paths``: hash -> path string for every hash above (the vocabulary the store keeps once).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from lxml import etree
from numpy.typing import NDArray

from ditto.fingerprint.dom import ATTR_SEP, LINK_ATTRS, PATH_SEP, STEP_ATTRS, step_label, tag_label
from ditto.textutil import hash64, normalize_ws

TEXT_JOIN = "\x1f"

U64 = NDArray[np.uint64]
U32 = NDArray[np.uint32]

_EMPTY_U64: U64 = np.empty(0, dtype=np.uint64)
_EMPTY_U32: U32 = np.empty(0, dtype=np.uint32)


def signature_hash(signature: U64) -> int:
    """Stable 64-bit hash of a signature array (BLAKE2b over its little-endian bytes)."""
    data = np.ascontiguousarray(signature, dtype="<u8").tobytes()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


@dataclass
class PageFingerprint:
    signature: U64
    text_path: U64
    text_hash: U64
    text_count: U32
    paths: dict[int, str] = field(default_factory=dict)

    @property
    def signature_hash(self) -> int:
        return signature_hash(self.signature)

    @classmethod
    def empty(cls) -> PageFingerprint:
        return cls(_EMPTY_U64, _EMPTY_U64, _EMPTY_U64, _EMPTY_U32, {})


class _PathTable:
    """Per-process interning of structural paths: ``(parent path id, step label) -> path id``,
    with each path string built and hashed once. Sites repeat a few thousand distinct paths,
    so per-element work drops to two dict lookups."""

    MAX_PATHS = 1 << 20

    def __init__(self) -> None:
        self.ids: dict[tuple[int, str], int] = {}
        self.paths: list[str] = []
        self.hashes: list[int] = []

    def reset_if_full(self) -> None:
        if len(self.paths) >= self.MAX_PATHS:
            self.ids.clear()
            self.paths.clear()
            self.hashes.clear()

    def child(self, parent: int, label: str) -> int:
        key = (parent, label)
        pid = self.ids.get(key)
        if pid is None:
            path = label if parent < 0 else self.paths[parent] + PATH_SEP + label
            pid = self._add(key, path)
        return pid

    def attr(self, parent: int, attr: str) -> int:
        key = (parent, "\x00" + attr)  # step labels never contain NUL
        pid = self.ids.get(key)
        if pid is None:
            pid = self._add(key, self.paths[parent] + ATTR_SEP + attr)
        return pid

    def _add(self, key: tuple[int, str], path: str) -> int:
        pid = len(self.paths)
        self.paths.append(path)
        self.hashes.append(hash64(path))
        self.ids[key] = pid
        return pid


_TABLE = _PathTable()


_STEP_ATTR_SET = frozenset(STEP_ATTRS)


def fingerprint_tree(root: etree._Element) -> PageFingerprint:
    """Fingerprint a cleaned tree (``dom.clean_html`` or ``dom.clean_etree`` output)."""
    table = _TABLE
    table.reset_if_full()
    child_id, attr_id = table.child, table.attr
    pid_of: dict[etree._Element | None, int] = {root: child_id(-1, step_label(root))}
    parts: dict[etree._Element | None, list[str]] = {}  # raw own-text pieces, doc order
    texts: dict[int, list[str]] = {}  # text path id -> texts in document order
    if root.text:
        parts[root] = [root.text]
    for el in root.iter():
        if el is root:
            keys = el.keys()
            pid = pid_of[root]
        else:
            parent = el.getparent()
            tail = el.tail
            if tail:
                piece = parts.get(parent)
                if piece is None:
                    parts[parent] = [tail]
                else:
                    piece.append(tail)
            tag = el.tag
            if not isinstance(tag, str):
                continue
            keys = el.keys()
            plain = not keys or _STEP_ATTR_SET.isdisjoint(keys)
            pid = pid_of[el] = child_id(pid_of[parent], tag_label(tag) if plain else step_label(el))
            text = el.text
            if text:
                parts[el] = [text]
        if not keys:
            continue
        for attr in LINK_ATTRS:
            if attr in keys:
                value = (el.get(attr) or "").strip()
                if value:
                    key = attr_id(pid, attr)
                    bucket = texts.get(key)
                    if bucket is None:
                        texts[key] = [value]
                    else:
                        bucket.append(value)
    for owner, piece in parts.items():
        raw = piece[0] if len(piece) == 1 else "".join(piece)
        if raw.isspace():
            continue
        key = pid_of[owner]
        bucket = texts.get(key)
        if bucket is None:
            texts[key] = [normalize_ws(raw)]
        else:
            bucket.append(normalize_ws(raw))

    hashes, path_strings = table.hashes, table.paths
    element_ids = set(pid_of.values())
    paths = {hashes[i]: path_strings[i] for i in element_ids}
    signature = np.fromiter(paths.keys(), dtype=np.uint64, count=len(paths))
    signature.sort()
    n = len(texts)
    text_path = np.empty(n, dtype=np.uint64)
    text_hash = np.empty(n, dtype=np.uint64)
    text_count = np.empty(n, dtype=np.uint32)
    for i, (key, values) in enumerate(texts.items()):
        h = hashes[key]
        paths[h] = path_strings[key]
        text_path[i] = h
        text_hash[i] = hash64(TEXT_JOIN.join(values))
        text_count[i] = len(values)
    order = np.argsort(text_path, kind="stable")
    return PageFingerprint(signature, text_path[order], text_hash[order], text_count[order], paths)
