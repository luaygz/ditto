"""Small text and hashing helpers shared by every stage."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run (including NBSP and other Unicode spaces) to one space."""
    return _WS.sub(" ", text.replace("\u00a0", " ")).strip()


def hash64(text: str) -> int:
    """Stable unsigned 64-bit hash (BLAKE2b, 8-byte digest) of a UTF-8 string."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
