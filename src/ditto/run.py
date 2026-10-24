"""Run directory layout.

Every stage reads its inputs from and writes its outputs into one run directory, so any stage
can be rerun alone. This module is the single place that names those files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson


@dataclass(frozen=True)
class RunDir:
    """Paths inside a run directory. Creating the object touches nothing on disk."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # -- metadata -----------------------------------------------------------------------------
    @property
    def run_json(self) -> Path:
        """Run metadata: site name, source paths, ditto version, created timestamp."""
        return self.root / "run.json"

    # -- stage 1: ingest ----------------------------------------------------------------------
    @property
    def pages_db(self) -> Path:
        return self.root / "pages.sqlite"

    @property
    def blobs_dir(self) -> Path:
        """Content-addressed raw page bytes (``blobs/<sha[:2]>/<sha>``)."""
        return self.root / "blobs"

    # -- stage 2: fingerprint -----------------------------------------------------------------
    @property
    def fingerprints_dir(self) -> Path:
        return self.root / "fingerprints"

    # -- stage 3: cluster ---------------------------------------------------------------------
    @property
    def clusters_json(self) -> Path:
        return self.root / "clusters.json"

    @property
    def cluster_report(self) -> Path:
        return self.root / "cluster_report.md"

    # -- stage 4: analyze ---------------------------------------------------------------------
    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    def candidates(self, template_id: str) -> Path:
        return self.candidates_dir / f"{template_id}.json"

    # -- stage 5: union page ------------------------------------------------------------------
    @property
    def union_dir(self) -> Path:
        return self.root / "union"

    def union_html(self, template_id: str) -> Path:
        return self.union_dir / f"{template_id}.html"

    def union_meta(self, template_id: str) -> Path:
        return self.union_dir / f"{template_id}.json"

    # -- stage 9: probe / confirm ---------------------------------------------------------------
    @property
    def probe_json(self) -> Path:
        return self.root / "probe.json"

    @property
    def probe_yaml(self) -> Path:
        return self.root / "probe.yaml"

    @property
    def mapping_json(self) -> Path:
        return self.root / "mapping.json"

    @property
    def schema_proposal(self) -> Path:
        """Unprompted mode: the proposed schema shown at the checkpoint before selectors."""
        return self.root / "schema_proposal.json"

    # -- stage 6: generate --------------------------------------------------------------------
    @property
    def drafts_dir(self) -> Path:
        return self.root / "specs" / "draft"

    def draft_spec(self, template_id: str) -> Path:
        return self.drafts_dir / f"{template_id}.json"

    @property
    def generation_json(self) -> Path:
        """Per-template generation outcome (ok / failed fields / skipped and why)."""
        return self.root / "generation.json"

    # -- stage 7: verify ----------------------------------------------------------------------
    @property
    def specs_dir(self) -> Path:
        return self.root / "specs"

    def spec(self, template_id: str) -> Path:
        return self.specs_dir / f"{template_id}.json"

    @property
    def verify_dir(self) -> Path:
        return self.root / "verify"

    def verify_report(self, template_id: str) -> Path:
        return self.verify_dir / f"{template_id}.json"

    # -- review -------------------------------------------------------------------------------
    @property
    def golden_dir(self) -> Path:
        return self.root / "golden"

    def golden(self, template_id: str) -> Path:
        return self.golden_dir / f"{template_id}.jsonl"

    # -- stage 8: extract ---------------------------------------------------------------------
    def out_dir(self, site: str) -> Path:
        return self.root / "out" / site

    def out_jsonl(self, site: str, template_id: str | None) -> Path:
        return self.out_dir(site) / f"{template_id or '_tail'}.jsonl"

    @property
    def report_json(self) -> Path:
        return self.root / "report.json"

    @property
    def report_md(self) -> Path:
        return self.root / "report.md"

    # -- accounting -----------------------------------------------------------------------------
    @property
    def usage_jsonl(self) -> Path:
        """One JSON line per LLM call made for this run (role, template, tokens, cost, cached)."""
        return self.root / "usage.jsonl"

    @property
    def timings_json(self) -> Path:
        """Stage name -> {seconds, pages, pages_per_sec_per_core}."""
        return self.root / "timings.json"

    # -- helpers ------------------------------------------------------------------------------
    def ensure(self) -> RunDir:
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def read_meta(self) -> dict[str, Any]:
        if not self.run_json.exists():
            return {}
        data: dict[str, Any] = json.loads(self.run_json.read_text())
        return data

    def write_meta(self, **updates: Any) -> None:
        meta = self.read_meta()
        meta.update(updates)
        write_json(self.run_json, meta)

    @property
    def site(self) -> str:
        """Site name recorded by ingest (most common URL host, else source directory name)."""
        return str(self.read_meta().get("site", self.root.name))

    def record_timing(self, stage: str, seconds: float, pages: int, workers: int) -> None:
        timings: dict[str, Any] = {}
        if self.timings_json.exists():
            timings = json.loads(self.timings_json.read_text())
        per_core = pages / seconds / max(workers, 1) if seconds > 0 else None
        timings[stage] = {
            "seconds": round(seconds, 4),
            "pages": pages,
            "workers": workers,
            "pages_per_sec": round(pages / seconds, 2) if seconds > 0 else None,
            "pages_per_sec_per_core": round(per_core, 2) if per_core is not None else None,
        }
        write_json(self.timings_json, timings)


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (indent 2, keys in insertion order, non-ASCII kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())
