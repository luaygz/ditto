"""Stage 2: parse, normalize and fingerprint pages (structural paths, signatures, text items)."""

from __future__ import annotations

from ditto.fingerprint.stage import fingerprint_html, fingerprint_run
from ditto.fingerprint.store import FingerprintStore, FingerprintWriter
from ditto.fingerprint.tree import PageFingerprint, fingerprint_tree, signature_hash

__all__ = [
    "FingerprintStore",
    "FingerprintWriter",
    "PageFingerprint",
    "fingerprint_html",
    "fingerprint_run",
    "fingerprint_tree",
    "signature_hash",
]
