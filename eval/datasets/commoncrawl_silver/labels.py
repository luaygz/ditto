"""Step 5: silver labels from each page's own JSON-LD.

    uv run python eval/datasets/commoncrawl_silver/labels.py [--only D ...] [--workers N]

For every selected domain, reads ``data/cc/<domain>/*.warc.gz`` (only the journaled, complete
part of each file, so it can run while ``download.py`` is still appending), runs ``extruct`` on
the original page, takes the page's own JSON-LD object of the vertical's target type and
flattens it to the fields of ``fields/<vertical>.yaml``
(:func:`ditto.evaluation.datasets.cc_silver.page_labels`). Writes
``data/cc/<domain>/labels.jsonl`` (one row per labeled page, keyed by the WARC-Target-URI) and
``labels_stats.json``. Labels are data: never committed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc import DATA_DIR, DOMAINS_YAML, iter_warc_pages, url_path
from manifest import write_manifest

from ditto.evaluation.datasets.cc_silver import (
    LABELS_FILE,
    VERTICALS,
    decode_body,
    page_labels,
)
from ditto.fingerprint.dom import parse_html
from ditto.parallel import pmap


def journaled_ends(site_dir: Path) -> dict[str, int]:
    """Byte offset after the last complete record of each WARC file (from download.py)."""
    ends: dict[str, int] = {}
    path = site_dir / "journal.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            if "file" in entry:
                ends[entry["file"]] = max(ends.get(entry["file"], 0), entry["end"])
    return ends


def label_file(chunk: Sequence[tuple[str, str, str, int]]) -> list[dict[str, Any]]:
    """pmap worker: label every page of the given WARC files (domain, vertical, path, limit)."""
    out: list[dict[str, Any]] = []
    for _domain, vertical, path, limit in chunk:
        for record in iter_warc_pages(Path(path), limit=limit):
            if record.status != 200:
                continue
            root = parse_html(decode_body(record.body, record.content_type))
            found = page_labels(root, vertical, record.url)
            out.append(
                {
                    "url": record.url,
                    "type": found.type,
                    "n_objects": found.n_objects,
                    "labels": found.labels,
                }
            )
    return out


def summarize(rows: list[dict[str, Any]], vertical: str, detail: str) -> dict[str, Any]:
    rx = re.compile(detail)
    fields = VERTICALS[vertical].fields
    filled: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    for row in rows:
        is_detail = bool(rx.search(url_path(row["url"])))
        by_kind["detail_pages" if is_detail else "other_pages"] += 1
        if row["n_objects"]:
            by_kind["detail_with_jsonld" if is_detail else "other_with_jsonld"] += 1
        if row["labels"] is not None:
            by_kind["detail_labeled" if is_detail else "other_labeled"] += 1
            for name in fields:
                if row["labels"].get(name) not in (None, "", []):
                    filled[name] += 1
    labeled = sum(r["labels"] is not None for r in rows)
    return {
        "pages": len(rows),
        "pages_with_target_jsonld": sum(r["n_objects"] > 0 for r in rows),
        "labeled_pages": labeled,
        "ambiguous_pages": sum(r["n_objects"] > 1 and r["labels"] is None for r in rows),
        **dict(by_kind),
        "field_fill_rate": {n: round(filled[n] / labeled, 3) if labeled else 0.0 for n in fields},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    doc = yaml.safe_load(DOMAINS_YAML.read_text())
    for entry in doc["selected"]:
        domain, vertical = entry["domain"], entry["vertical"]
        if args.only and domain not in args.only:
            continue
        site_dir = DATA_DIR / domain
        ends = journaled_ends(site_dir)
        items = [
            (domain, vertical, str(site_dir / name), end) for name, end in sorted(ends.items())
        ]
        if not items:
            print(f"{domain}: no downloaded WARC files yet", flush=True)
            continue
        started = time.time()
        rows = [
            r for part in pmap(label_file, items, workers=args.workers, chunk_size=1) for r in part
        ]
        elapsed = time.time() - started
        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            seen.setdefault(row["url"], row)
        rows = sorted(seen.values(), key=lambda r: r["url"])
        labeled = [r for r in rows if r["labels"] is not None]
        with (site_dir / LABELS_FILE).open("w", encoding="utf-8") as fh:
            for row in labeled:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stats = summarize(rows, vertical, entry["detail_pattern"])
        stats["seconds"] = round(elapsed, 1)
        stats["pages_per_second"] = round(len(rows) / elapsed, 1) if elapsed else None
        (site_dir / "labels_stats.json").write_text(json.dumps(stats, indent=1))
        print(
            f"{domain}: {stats['pages']} pages, {stats['labeled_pages']} labeled "
            f"({stats['pages_per_second']} pages/s) fill={stats['field_fill_rate']}",
            flush=True,
        )
    write_manifest()


if __name__ == "__main__":
    main()
