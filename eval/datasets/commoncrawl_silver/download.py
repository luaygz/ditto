"""Step 4 (download) of the Common Crawl silver set.

    uv run python eval/datasets/commoncrawl_silver/download.py [--cap 2500] [--only D ...]
        [--plan-only]

For every domain selected in ``domains.yaml``: list its status-200 ``text/html`` captures in the
two crawls, keep the newest capture per URL and one URL per payload digest, then plan up to
``--cap`` pages -- every detail-pattern URL first (a seeded sample if there are more than the
cap allows), then a seeded sample of the other URLs so the site keeps its long tail (at least
``TAIL_SHARE`` of the cap when the site has that many). Captures already downloaded by an
earlier run are kept and count toward the quota of their kind; the rest of each quota is filled
in the seeded order (so a domain downloaded under an earlier, larger cap keeps all its pages).

Records are fetched by byte range, one request at a time, at least one second apart, with
Common Crawl's slow-down answers (403/429/503) waited out with growing backoff
(:class:`cc.Http`), and their gzip members concatenated into WARC files
``data/cc/<domain>/captures-NNNNN.warc.gz`` (``ROLL`` records per file). Domains are fetched
round-robin, one capture each in turn, so an interrupted run leaves every domain with a similar
share of its plan.

Resumable: ``data/cc/<domain>/journal.jsonl`` records every appended capture with the file's end
offset and every failed capture as an error row; a restart truncates each file to its last
journaled offset, skips journaled captures and retries the failed ones.

Writes the committed ``captures.jsonl.gz`` (one row per planned capture: url, crawl, filename,
offset, length, digest, ...), updates ``domains.yaml`` with the capture counts, the cap and its
reason, and rewrites ``manifest.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc import (
    CAPTURES_JSONL,
    DATA_DIR,
    DATA_INTERVAL_S,
    DOMAINS_YAML,
    MAX_DATA_CONCURRENCY,
    SLOW_DOWN_MAX_S,
    SLOW_DOWN_START_S,
    CdxIndex,
    Http,
    dump_yaml,
    gzip_jsonl,
    parse_record,
    url_path,
)
from manifest import write_manifest

DEFAULT_CAP = 2_500
TAIL_SHARE = 0.2
ROLL = 5000
CAP_REASON = (
    "Lowered from 20,000 to 2,500 pages per domain on 2026-10-27: at 4 concurrent requests "
    "Common Crawl's CloudFront WAF answered every range request with 403 'Request blocked' after "
    "about 2,250 records (~440 MB) per run, and 160,000 captures at the recommended pace of one "
    "request per second would take ~44 h. Pages downloaded before the change are kept "
    "(argos.co.uk: 6,701 detail pages)."
)


def plan_domain(
    entry: dict[str, Any],
    indexes: dict[str, CdxIndex],
    cap: int,
    have: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Captures to download for one domain, detail-pattern URLs first, plus listing counts.

    ``have``: URLs already downloaded; they stay in the plan and fill their kind's quota first.
    """
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
    chosen_details = sorted(_fill(details, n_detail, have), key=lambda r: r["urlkey"])
    chosen_others = sorted(_fill(others, n_other, have), key=lambda r: r["urlkey"])
    plan = [r | {"kind": "detail"} for r in chosen_details]
    plan += [r | {"kind": "other"} for r in chosen_others]
    counts["cap"] = cap
    counts["planned"] = len(plan)
    counts["planned_detail"] = len(chosen_details)
    counts["planned_other"] = len(chosen_others)
    counts["kept_from_earlier_runs"] = sum(r["url"] in have for r in plan)
    return plan, counts


def _fill(
    rows: list[dict[str, Any]], quota: int, have: set[str] | frozenset[str]
) -> list[dict[str, Any]]:
    """Every already-downloaded row, topped up to ``quota`` in the given (seeded) order."""
    kept = [r for r in rows if r["url"] in have]
    fresh = [r for r in rows if r["url"] not in have]
    return kept + fresh[: max(0, quota - len(kept))]


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
    """Append-only progress log keyed by URL; restores file sizes and the done set on resume.

    Error rows are kept for the record but never count as done: a rerun retries them.
    """

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


class DomainWriter:
    """One domain's WARC files and journal during a download run."""

    def __init__(self, domain: str, plan: list[dict[str, Any]]) -> None:
        self.domain = domain
        self.plan = plan
        self.site_dir = DATA_DIR / domain
        self.site_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.site_dir)
        self.todo = [i for i, row in enumerate(plan) if row["url"] not in self.journal.done]
        self.per_file: dict[str, int] = {}
        for entry in self.journal.done.values():
            self.per_file[entry["file"]] = self.per_file.get(entry["file"], 0) + 1
        self.file_no = max((int(n[9:14]) for n in self.per_file), default=0)
        self.current = f"captures-{self.file_no:05d}.warc.gz"
        self.out: BinaryIO = (self.site_dir / self.current).open("ab")
        self.fetched = self.errors = self.nbytes = 0

    def append(self, i: int, raw: bytes) -> None:
        if self.per_file.get(self.current, 0) >= ROLL:
            self.out.close()
            self.file_no += 1
            self.current = f"captures-{self.file_no:05d}.warc.gz"
            self.out = (self.site_dir / self.current).open("ab")
        self.out.write(raw)
        self.out.flush()
        self.per_file[self.current] = self.per_file.get(self.current, 0) + 1
        self.journal.log(
            {"i": i, "file": self.current, "end": self.out.tell(), "url": self.plan[i]["url"]}
        )
        self.fetched += 1
        self.nbytes += len(raw)

    def fail(self, i: int, exc: Exception) -> None:
        self.errors += 1
        self.journal.log({"i": i, "url": self.plan[i]["url"], "error": repr(exc)[:500]})

    def close(self) -> None:
        self.out.close()
        self.journal.close()


def fetch_capture(http: Http, row: dict[str, Any]) -> bytes:
    """The capture's gzip member, checked to be the record the index row points at."""
    raw = http.fetch_range(row["filename"], int(row["offset"]), int(row["length"]))
    record = parse_record(raw)
    if record.url != row["url"]:
        raise ValueError(f"record URL {record.url!r} != index URL {row['url']!r}")
    return raw


def download_all(http: Http, plans: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Fetch every planned capture not yet journaled, round-robin over the domains."""
    writers = {d: DomainWriter(d, plan) for d, plan in plans.items()}
    total = sum(len(w.todo) for w in writers.values())
    for w in writers.values():
        done = len(w.plan) - len(w.todo)
        print(f"{w.domain}: {len(w.plan)} planned, {done} already done", flush=True)
    started = time.time()
    queues = {d: iter(w.todo) for d, w in writers.items()}
    count = 0
    try:
        while queues:
            for domain in list(queues):
                i = next(queues[domain], None)
                if i is None:
                    del queues[domain]
                    continue
                w = writers[domain]
                try:
                    w.append(i, fetch_capture(http, w.plan[i]))
                except Exception as exc:  # one bad capture must not end the run
                    w.fail(i, exc)
                count += 1
                if count % 250 == 0:
                    rate = count / (time.time() - started)
                    left = (total - count) / rate if rate else 0
                    print(
                        f"  {count}/{total} captures attempted, {rate:.2f}/s, "
                        f"~{left / 3600:.1f} h left, http {http.stats}",
                        flush=True,
                    )
    finally:
        for w in writers.values():
            w.close()
    elapsed = time.time() - started
    stats: dict[str, dict[str, Any]] = {}
    for domain, w in writers.items():
        stats[domain] = {
            "fetched_this_run": w.fetched,
            "errors_this_run": w.errors,
            "bytes_this_run": w.nbytes,
            "seconds_this_run": round(elapsed, 1),
        }
        print(f"{domain}: done {stats[domain]}", flush=True)
    return stats


def download_domain(http: Http, domain: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """:func:`download_all` for one domain."""
    return download_all(http, {domain: plan})[domain]


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


def policy(cap: int) -> dict[str, Any]:
    """The download settings recorded in ``domains.yaml``."""
    return {
        "cap_per_domain": cap,
        "cap_reason": CAP_REASON,
        "tail_share": TAIL_SHARE,
        "data_concurrency": MAX_DATA_CONCURRENCY,
        "min_interval_s": DATA_INTERVAL_S,
        "slow_down": (
            f"403/429/503 from data.commoncrawl.org: backoff with jitter from {SLOW_DOWN_START_S:.0f}"
            f" s doubling to {SLOW_DOWN_MAX_S:.0f} s while blocked, one probe request each time"
        ),
    }


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
        domain = entry["domain"]
        plan, counts = plan_domain(entry, indexes, args.cap, downloaded(domain))
        plans[domain] = plan
        entry["captures"] = counts
        print(f"{domain}: {counts}", flush=True)
        stats_path = DATA_DIR / domain / "plan.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(counts, indent=1))

    all_rows = [capture_row(d, r) for d, plan in plans.items() for r in plan]
    gzip_jsonl(CAPTURES_JSONL, all_rows)
    print(f"wrote {CAPTURES_JSONL} ({len(all_rows)} captures)", flush=True)
    doc["download"] = policy(args.cap)
    _write_counts(doc, plans)

    if not args.plan_only:
        wanted = {d: p for d, p in plans.items() if not args.only or d in args.only}
        run_started = time.time()
        stats = download_all(http, wanted)
        for domain, s in stats.items():
            with (DATA_DIR / domain / "download_runs.jsonl").open("a") as fh:
                fh.write(json.dumps(s | {"started": run_started, "finished": time.time()}) + "\n")
        doc["download"]["last_run"] = {
            "finished": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "http": dict(http.stats),
        }
        _write_counts(doc, plans)
    write_manifest(http_stats=http.stats)


def _write_counts(doc: dict[str, Any], plans: dict[str, list[dict[str, Any]]]) -> None:
    """Record the downloaded share of every plan in ``domains.yaml``."""
    for entry in doc["selected"]:
        done = downloaded(entry["domain"])
        plan = plans[entry["domain"]]
        entry["captures"]["downloaded"] = sum(r["url"] in done for r in plan)
        for kind in ("detail", "other"):
            entry["captures"][f"downloaded_{kind}"] = sum(
                r["url"] in done for r in plan if r["kind"] == kind
            )
    DOMAINS_YAML.write_text(dump_yaml(doc))


if __name__ == "__main__":
    main()
