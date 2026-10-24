"""Shared assertions for the synthetic generator tests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ditto.fingerprint.dom import clean_html, node_text

_CHARSET = re.compile(rb"charset=[\"']?([A-Za-z0-9_-]+)")


def page_encoding(raw: bytes) -> str:
    m = _CHARSET.search(raw[:2000])
    assert m, "page declares no charset"
    return m.group(1).decode("ascii")


def element_texts(raw: bytes) -> set[str]:
    """Whitespace-normalized text of every element of the cleaned page."""
    root = clean_html(raw, page_encoding(raw))
    return {node_text(el) for el in root.iter() if isinstance(el.tag, str)}


def sidecar_problems(raw: bytes, truth: dict[str, Any]) -> list[str]:
    """Every ground-truth value must be the exact visible text of some element."""
    texts = element_texts(raw)
    problems: list[str] = []
    for name, value in truth["fields"].items():
        if value is not None and value not in texts:
            problems.append(f"field {name}={value!r}")
    for name, items in truth["lists"].items():
        problems += [f"list {name} item {v!r}" for v in items if v not in texts]
    for name, rows in truth["records"].items():
        for row in rows:
            problems += [
                f"record {name}.{c}={v!r}"
                for c, v in row.items()
                if v is not None and v not in texts
            ]
    for name, pairs in truth["kv"].items():
        for key, value in pairs.items():
            if key not in texts or value not in texts:
                problems.append(f"kv {name} {key!r}={value!r}")
    return problems


def tree_bytes(root: Path) -> dict[str, bytes]:
    """All generated files except synth.json's timing block."""
    out = {
        str(p.relative_to(root)): p.read_bytes()
        for sub in ("site", "truth")
        for p in sorted((root / sub).rglob("*"))
        if p.is_file()
    }
    meta = json.loads((root / "synth.json").read_text())
    meta.pop("generation")
    out["synth.json"] = json.dumps(meta, sort_keys=True).encode()
    return out
