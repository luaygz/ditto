"""Step 4 (download) of the Common Crawl silver set.

    uv run python eval/datasets/commoncrawl_silver/download.py [--cap 20000] [--only D ...]

For every domain selected in ``domains.yaml``: list its status-200 ``text/html`` captures in the
two crawls, keep the newest capture per URL and one URL per payload digest, then plan up to
``--cap`` pages -- every detail-pattern URL first (a seeded sample if there are more than the
cap allows), then a seeded sample of the other URLs so the site keeps its long tail (at least
``TAIL_SHARE`` of the cap when the site has that many). Records are fetched by byte range (at most
4 concurrent requests, backoff on 429/503) and their gzip members concatenated into proper WARC
files ``data/cc/<domain>/captures-NNNNN.warc.gz`` (``ROLL`` records per file).

Resumable: ``data/cc/<domain>/journal.jsonl`` records every appended capture with the file's end
offset; a restart truncates each file to its last journaled offset and skips journaled captures.

Writes the committed ``captures.jsonl.gz`` (one row per planned capture: url, crawl, filename,
offset, length, digest, ...), updates ``domains.yaml`` with the capture counts and rewrites
``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc import (
    CAPTURES_JSONL,
    DATA_DIR,
    DOMAINS_YAML,
    CdxIndex,
    Http,
    dump_yaml,
    gzip_jsonl,
    parse_record,
    url_path,
)
from manifest import write_manifest

DEFAULT_CAP = 20_000
TAIL_SHARE = 0.2
ROLL = 5000
IN_FLIGHT = 64


def plan_domain(
    entry: dict[str, Any], indexes: dict[str, CdxIndex], cap: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Captures to download for one domain, detail-pattern URLs first, plus listing counts."""
    domain = entry["domain"]
    detail = re.compile(entry["detail_pattern"])
    counts: dict[str, Any] = {"index_html_captures": {}}
    newest: dict[str, dict[str, Any]] = {}
    for crawl, idx in indexes.items():
        rows = idx.domain_rows(domain)
        counts["index_html_captures"][crawl] = len(rows)
        for row in rows:
            row = row | {"crawl": crawl}
            prev = newest.get(row["urlkey"])
            if prev is None or row["timestamp"] > prev["timestamp"]:
                newest[row["urlkey"]] = row
    counts["distinct_urls"] = len(newest)
    by_digest: dict[str, dict[str, Any]] = {}
    for key in sorted(newest):
        row = newest[key]
        by_digest.setdefault(row.get("digest") or key, row)
    unique = sorted(by_digest.values(), key=lambda r: r["urlkey"])
    counts["distinct_pages"] = len(unique)
    details = [r for r in unique if detail.search(url_path(r["url"]))]
    others = [r for r in unique if not detail.search(url_path(r["url"]))]
    counts["distinct_detail_pages"] = len(details)
    counts["distinct_other_pages"] = len(others)
    # seeded shuffle + prefix: a smaller cap selects a subset of a larger cap's pages
    rng = random.Random(f"download:{domain}")
    rng.shuffle(details)
    rng.shuffle(others)
    tail_reserve = min(len(others), round(cap * TAIL_SHARE))
    n_detail = min(len(details), cap - tail_reserve)
    n_other = min(len(others), cap - n_detail)
    chosen_details = sorted(details[:n_detail], key=lambda r: r["urlkey"])
    chosen_others = sorted(others[:n_other], key=lambda r: r["urlkey"])
    plan = [r | {"kind": "detail"} for r in chosen_details]
    plan += [r | {"kind": "other"} for r in chosen_others]
    counts["cap"] = cap
    counts["planned"] = len(plan)
    counts["planned_detail"] = len(chosen_details)
    counts["planned_other"] = len(chosen_others)
    return plan, counts


def capture_row(domain: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": domain,
        "url": row["url"],
        "crawl": row["crawl"],
        "filename": row["filename"],
        "offset": int(row["offset"]),
        "length": int(row["length"]),
        "digest": row.get("digest"),
        "timestamp": row["timestamp"],
        "kind": row["kind"],
    }


class Journal:
    """Append-only progress log keyed by URL; restores file sizes and the done set on resume."""

    def __init__(self, site_dir: Path) -> None:
        self.path = site_dir / "journal.jsonl"
        self.site_dir = site_dir
        self.done: dict[str, dict[str, Any]] = {}
        self.ends: dict[str, int] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn final line from an interrupted run
                if "error" in entry:
                    continue
                self.done[entry["url"]] = entry
                self.ends[entry["file"]] = max(self.ends.get(entry["file"], 0), entry["end"])
        for name, end in self.ends.items():
            path = site_dir / name
            if path.stat().st_size > end:
                with path.open("r+b") as fh:
                    fh.truncate(end)
        for path in site_dir.glob("captures-*.warc.gz"):
            if path.name not in self.ends:
                path.unlink()  # created but nothing journaled
        self.fh = self.path.open("a")

    def log(self, entry: dict[str, Any]) -> None:
        self.fh.write(json.dumps(entry) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def download_domain(http: Http, domain: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
    site_dir = DATA_DIR / domain
    site_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(site_dir)
    todo = [i for i, row in enumerate(plan) if row["url"] not in journal.done]
    per_file: dict[str, int] = {}
    for entry in journal.done.values():
        per_file[entry["file"]] = per_file.get(entry["file"], 0) + 1
    file_no = max((int(n[9:14]) for n in per_file), default=0)
    current = f"captures-{file_no:05d}.warc.gz"
    print(f"{domain}: {len(plan)} planned, {len(journal.done)} already done", flush=True)

    def fetch(i: int) -> bytes:
        row = plan[i]
        raw = http.fetch_range(row["filename"], int(row["offset"]), int(row["length"]))
        record = parse_record(raw)
        if record.url != row["url"]:
            raise ValueError(f"record URL {record.url!r} != index URL {row['url']!r}")
        return raw

    started = time.time()
    fetched = errors = 0
    nbytes = 0
    out = (site_dir / current).open("ab")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending: deque[tuple[int, Future[bytes]]] = deque()
        queue = iter(todo)
        while True:
            while len(pending) < IN_FLIGHT:
                nxt = next(queue, None)
                if nxt is None:
                    break
                pending.append((nxt, pool.submit(fetch, nxt)))
            if not pending:
                break
            i, fut = pending.popleft()
            try:
                raw = fut.result()
            except Exception as exc:
                errors += 1
                journal.log({"i": i, "error": repr(exc)[:500]})
                continue
            if per_file.get(current, 0) >= ROLL:
                out.close()
                file_no += 1
                current = f"captures-{file_no:05d}.warc.gz"
                out = (site_dir / current).open("ab")
            out.write(raw)
            out.flush()
            per_file[current] = per_file.get(current, 0) + 1
            journal.log({"i": i, "file": current, "end": out.tell(), "url": plan[i]["url"]})
            fetched += 1
            nbytes += len(raw)
            if fetched % 1000 == 0:
                rate = fetched / (time.time() - started)
                print(f"  {domain}: {fetched}/{len(todo)} fetched, {rate:.1f}/s", flush=True)
    out.close()
    journal.close()
    elapsed = time.time() - started
    stats = {
        "fetched_this_run": fetched,
        "errors_this_run": errors,
        "bytes_this_run": nbytes,
        "seconds_this_run": round(elapsed, 1),
        "records_per_second": round(fetched / elapsed, 2) if elapsed else None,
    }
    print(f"{domain}: done {stats}", flush=True)
    return stats


def downloaded(domain: str) -> set[str]:
    """URLs whose records are in the domain's WARC files (from the journal)."""
    path = DATA_DIR / domain / "journal.jsonl"
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            if "error" not in entry:
                done.add(entry["url"])
    return done


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--only", nargs="*", help="download only these domains")
    parser.add_argument("--plan-only", action="store_true", help="plan and write captures only")
    args = parser.parse_args(argv)

    doc = yaml.safe_load(DOMAINS_YAML.read_text())
    http = Http()
    indexes = {crawl: CdxIndex(http, crawl) for crawl in doc["crawls"]}
    plans: dict[str, list[dict[str, Any]]] = {}
    for entry in doc["selected"]:
        plan, counts = plan_domain(entry, indexes, args.cap)
        plans[entry["domain"]] = plan
        entry["captures"] = counts
        print(f"{entry['domain']}: {counts}", flush=True)
        stats_path = DATA_DIR / entry["domain"] / "plan.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(counts, indent=1))

    all_rows = [capture_row(d, r) for d, plan in plans.items() for r in plan]
    gzip_jsonl(CAPTURES_JSONL, all_rows)
    print(f"wrote {CAPTURES_JSONL} ({len(all_rows)} captures)", flush=True)

    if not args.plan_only:
        for entry in doc["selected"]:
            domain = entry["domain"]
            if args.only and domain not in args.only:
                continue
            stats = download_domain(http, domain, plans[domain])
            runs_path = DATA_DIR / domain / "download_runs.jsonl"
            with runs_path.open("a") as fh:
                fh.write(json.dumps(stats | {"finished": time.time()}) + "\n")

    for entry in doc["selected"]:
        done = downloaded(entry["domain"])
        plan = plans[entry["domain"]]
        entry["captures"]["downloaded"] = sum(r["url"] in done for r in plan)
        for kind in ("detail", "other"):
            entry["captures"][f"downloaded_{kind}"] = sum(
                r["url"] in done for r in plan if r["kind"] == kind
            )
    DOMAINS_YAML.write_text(dump_yaml(doc))
    write_manifest(http_stats=http.stats)


if __name__ == "__main__":
    main()
