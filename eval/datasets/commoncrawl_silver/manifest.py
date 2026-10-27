"""``manifest.json`` for the Common Crawl silver set: sources, crawl ids, file hashes, dates, counts.

Rewritten from the current state of ``domains.yaml``, ``captures.jsonl.gz``, the downloaded
WARC files, the labels and the noise check by ``download.py``, ``labels.py`` and
``noise_check.py`` (or directly: ``uv run python eval/datasets/commoncrawl_silver/manifest.py``).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc import (
    CAPTURES_JSONL,
    COLLINFO_URL,
    DATA_BASE,
    DATA_DIR,
    DOMAINS_YAML,
    FIELDS_DIR,
    HERE,
    INDEX_BASE,
    INDEX_CACHE,
    MANIFEST_JSON,
    MAX_DATA_CONCURRENCY,
    NOISE_JSON,
    SELECTION_JSON,
    USER_AGENT,
    sha256_file,
)

HASH_CACHE = DATA_DIR / "hash_cache.json"


class Hasher:
    """SHA-256 of files, cached by (size, mtime) so large WARCs are hashed once."""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = (
            json.loads(HASH_CACHE.read_text()) if HASH_CACHE.exists() else {}
        )

    def __call__(self, path: Path) -> str:
        stat = path.stat()
        key = str(path.resolve())
        hit = self.cache.get(key)
        if hit and hit["size"] == stat.st_size and hit["mtime"] == stat.st_mtime:
            return str(hit["sha256"])
        digest = sha256_file(path)
        self.cache[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest}
        return digest

    def save(self) -> None:
        HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        HASH_CACHE.write_text(json.dumps(self.cache, indent=1))


def _journal_counts(site_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    path = site_dir / "journal.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            if "file" in entry:
                counts[entry["file"]] += 1
    return counts


def _runs(site_dir: Path) -> list[dict[str, Any]]:
    path = site_dir / "download_runs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _throughput(site_dir: Path) -> dict[str, Any] | None:
    runs = _runs(site_dir)
    if not runs:
        return None
    fetched = sum(r["fetched_this_run"] for r in runs)
    seconds = sum(r["seconds_this_run"] for r in runs)
    nbytes = sum(r["bytes_this_run"] for r in runs)
    return {
        "runs": len(runs),
        "records": fetched,
        "seconds": round(seconds, 1),
        "records_per_second": round(fetched / seconds, 2) if seconds else None,
        "megabytes_per_second": round(nbytes / seconds / 1e6, 3) if seconds else None,
        "errors": sum(r["errors_this_run"] for r in runs),
        "note": "round-robin runs (rows with 'started') share their wall time across domains",
    }


def _total_seconds(site_dirs: list[Path]) -> float:
    """Wall time of all download runs: a round-robin run (one ``started`` stamp shared by its
    domains) counts once; older per-domain runs count per domain."""
    seen: dict[Any, float] = {}
    for site_dir in site_dirs:
        for n, r in enumerate(_runs(site_dir)):
            key = r.get("started", (site_dir.name, n))
            seen[key] = max(seen.get(key, 0.0), float(r["seconds_this_run"]))
    return sum(seen.values())


def write_manifest(http_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    hasher = Hasher()
    doc = yaml.safe_load(DOMAINS_YAML.read_text()) if DOMAINS_YAML.exists() else {}
    crawls: list[str] = doc.get("crawls", [])
    files = {
        str(p.relative_to(HERE)): hasher(p)
        for p in [
            DOMAINS_YAML,
            CAPTURES_JSONL,
            SELECTION_JSON,
            NOISE_JSON,
            *sorted(FIELDS_DIR.glob("*.yaml")),
        ]
        if p.exists()
    }
    domains: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    for entry in doc.get("selected", []):
        domain = entry["domain"]
        site_dir = DATA_DIR / domain
        records = _journal_counts(site_dir)
        warcs = {
            p.name: {"sha256": hasher(p), "bytes": p.stat().st_size, "records": records[p.name]}
            for p in sorted(site_dir.glob("*.warc.gz"))
        }
        labels_path = site_dir / "labels.jsonl"
        labels_stats_path = site_dir / "labels_stats.json"
        info: dict[str, Any] = {
            "vertical": entry["vertical"],
            "target_type": entry.get("target_type"),
            "captures": entry.get("captures", {}),
            "warc_files": warcs,
            "warc_bytes": sum(w["bytes"] for w in warcs.values()),
            "download": _throughput(site_dir),
        }
        if labels_path.exists():
            stats = json.loads(labels_stats_path.read_text()) if labels_stats_path.exists() else {}
            info["labels"] = {
                "file": f"data/cc/{domain}/labels.jsonl",
                "sha256": hasher(labels_path),
                "labeled_pages": stats.get("labeled_pages"),
                "pages": stats.get("pages"),
            }
            totals["labeled_pages"] += int(stats.get("labeled_pages") or 0)
        domains[domain] = info
        totals["warc_bytes"] += info["warc_bytes"]
        totals["warc_records"] += sum(w["records"] for w in warcs.values())
        totals["planned"] += int(entry.get("captures", {}).get("planned", 0))
    cluster = {
        crawl: hasher(INDEX_CACHE / crawl / "cluster.idx")
        for crawl in crawls
        if (INDEX_CACHE / crawl / "cluster.idx").exists()
    }
    throughput = [d["download"] for d in domains.values() if d["download"]]
    seconds = _total_seconds([DATA_DIR / d for d in domains])
    records = sum(t["records"] for t in throughput)
    manifest = {
        "dataset": "commoncrawl_silver",
        "written_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "selection_date": doc.get("selection_date"),
        "user_agent": USER_AGENT,
        "sources": {
            "crawl_list": COLLINFO_URL,
            "cdx_index": f"{DATA_BASE}/cc-index/collections/<crawl>/indexes/"
            "(ZipNum: cluster.idx + cdx-NNNNN.gz, the files behind "
            f"{INDEX_BASE}/<crawl>-index)",
            "records": f"{DATA_BASE}/<filename> (HTTP byte ranges offset..offset+length-1)",
        },
        "crawls": crawls,
        "download_policy": doc.get("download"),
        "cluster_idx_sha256": cluster,
        "files": files,
        "domains": domains,
        "totals": dict(totals),
        "download_throughput": {
            "records": records,
            "seconds": round(seconds, 1),
            "records_per_second": round(records / seconds, 2) if seconds else None,
            "max_concurrent_requests": MAX_DATA_CONCURRENCY,
        },
    }
    if http_stats:
        manifest["last_run_http_stats"] = http_stats
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    hasher.save()
    return manifest


if __name__ == "__main__":
    out = write_manifest()
    print(json.dumps(out["totals"], indent=1))
