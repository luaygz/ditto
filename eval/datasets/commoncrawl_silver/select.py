"""Steps 1-4 (selection) of the Common Crawl silver set.

    uv run python eval/datasets/commoncrawl_silver/select.py count     # steps 1-2
    uv run python eval/datasets/commoncrawl_silver/select.py survey D  # path shapes of a domain
    uv run python eval/datasets/commoncrawl_silver/select.py sample    # step 3
    uv run python eval/datasets/commoncrawl_silver/select.py choose    # step 4 -> domains.yaml

Everything network-bound is cached (index pages under ``data/cc/index_cache/``, sampled
records under ``data/cc/selection/records/``), so re-running a phase is cheap. ``choose`` writes
the committed ``domains.yaml`` (selected and rejected candidates with reasons) and
``selection_log.json`` (the per-candidate evidence).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import statistics
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates import CANDIDATES, Candidate
from cc import (
    DATA_DIR,
    DOMAINS_YAML,
    SELECTION_JSON,
    CdxIndex,
    Http,
    Record,
    collinfo,
    crawl_of,
    dump_yaml,
    parse_record,
    surt_host,
    url_path,
)

from ditto.evaluation.datasets.cc_silver import (
    VERTICALS,
    decode_body,
    has_password_input,
    page_labels,
    parse_robots,
    visible_text,
)
from ditto.fingerprint.dom import parse_html

MIN_DISTINCT_URLS = 5000
SAMPLE_SIZE = 50
MIN_JSONLD_RATE = 0.60
MIN_WORDS = 200
MIN_ROBOTS_ALLOWED = 0.90
MAX_LOGIN_SHARE = 0.20
MIN_DETAIL_SAMPLED = 20
INDEX_LOCK = threading.Lock()
INDEXES: dict[str, CdxIndex] = {}
N_SELECT = 8
MAIN_VERTICALS = ("product", "job", "recipe", "event")
EXTRA_VERTICALS = ("news", "book")
ROBOTS_AGENTS = ("CCBot", "ditto-benchmark")

SELECTION_DIR = DATA_DIR / "selection"
COUNTS_JSON = SELECTION_DIR / "counts.json"
SAMPLES_DIR = SELECTION_DIR / "samples"
RECORDS_DIR = SELECTION_DIR / "records"


def latest_crawls(http: Http, n: int = 2) -> list[str]:
    """Step 1: the ``n`` most recent crawls listed in collinfo.json (newest first)."""
    crawls = collinfo(http)
    return [c["id"] for c in crawls[:n]]


def cdx(http: Http, crawl: str) -> CdxIndex:
    """The crawl's ZipNum index, loaded once per process."""
    with INDEX_LOCK:
        if crawl not in INDEXES:
            INDEXES[crawl] = CdxIndex(http, crawl)
        return INDEXES[crawl]


def domain_rows(http: Http, domain: str, crawls: list[str]) -> list[dict[str, Any]]:
    """Status-200 text/html index rows of ``<domain>/*`` in every crawl (``crawl`` key added)."""
    rows: list[dict[str, Any]] = []
    for crawl in crawls:
        rows += [r | {"crawl": crawl} for r in cdx(http, crawl).domain_rows(domain)]
    return rows


# --------------------------------------------------------------------------- step 2


def count_domain(http: Http, cand: Candidate, crawls: list[str]) -> dict[str, Any]:
    """Step 2: distinct status-200 text/html URLs of ``<domain>/*`` across the crawls."""
    info: dict[str, Any] = {"domain": cand.domain, "vertical": cand.vertical, "crawls": {}}
    keys: set[str] = set()
    detail = re.compile(cand.detail)
    detail_keys: set[str] = set()
    for crawl in crawls:
        idx = cdx(http, crawl)
        rows = idx.domain_rows(cand.domain)
        crawl_keys = {r["urlkey"] for r in rows}
        keys |= crawl_keys
        detail_keys |= {r["urlkey"] for r in rows if detail.search(url_path(r["url"]))}
        info["crawls"][crawl] = {
            "index_blocks": len(idx.blocks_for(surt_host(cand.domain) + "/")),
            "html_captures": len(rows),
            "distinct_urls": len(crawl_keys),
        }
    info["distinct_urls"] = len(keys)
    info["distinct_detail_urls"] = len(detail_keys)
    info["passes"] = len(keys) >= MIN_DISTINCT_URLS
    return info


def load_counts() -> dict[str, Any]:
    if not COUNTS_JSON.exists():
        return {"crawls": [], "domains": {}}
    data: dict[str, Any] = json.loads(COUNTS_JSON.read_text())
    return data


def cmd_count(args: argparse.Namespace) -> None:
    http = Http()
    crawls = latest_crawls(http)
    print("crawls:", crawls, flush=True)
    existing: dict[str, Any] = load_counts()["domains"]
    todo = [c for c in CANDIDATES if args.redo or c.domain not in existing]
    if args.only:
        todo = [c for c in CANDIDATES if c.domain in set(args.only)]

    def work(cand: Candidate) -> dict[str, Any]:
        try:
            return count_domain(http, cand, crawls)
        except Exception as exc:  # recorded as a rejection reason, never silently dropped
            return {"domain": cand.domain, "vertical": cand.vertical, "error": repr(exc)}

    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for info in pool.map(work, todo):
            existing[info["domain"]] = info
            print(
                f"{info['domain']:28s} {info['vertical']:8s} "
                f"distinct={info.get('distinct_urls', '-')!s:>6} "
                f"{'PASS' if info.get('passes') else 'fail'} {info.get('error', '')}",
                flush=True,
            )
            COUNTS_JSON.write_text(
                json.dumps({"crawls": crawls, "domains": existing}, indent=1, sort_keys=True)
            )
    print("stats:", http.stats)


def cmd_survey(args: argparse.Namespace) -> None:
    """Most common path shapes of domains' index rows (to check detail regexes)."""
    http = Http()
    crawls = latest_crawls(http)
    for domain in args.domain:
        cand = next(c for c in CANDIDATES if c.domain == domain)
        shapes: Counter[str] = Counter()
        examples: dict[str, str] = {}
        detail = re.compile(args.pattern or cand.detail)
        n_detail = n = 0
        for row in domain_rows(http, cand.domain, crawls):
            path = url_path(row["url"])
            n += 1
            n_detail += bool(detail.search(path))
            shape = re.sub(r"\d+", "N", path.split("?")[0])
            shape = "/".join(seg if len(seg) < 25 else "<long>" for seg in shape.split("/")[:4])
            shapes[shape] += 1
            examples.setdefault(shape, row["url"])
        print(f"\n{cand.domain}: {n} rows, detail regex matches {n_detail}")
        for shape, count in shapes.most_common(args.top):
            print(f"{count:7d}  {shape:50s} {examples[shape][:110]}")


# --------------------------------------------------------------------------- step 3


def detail_pool(http: Http, cand: Candidate, crawls: list[str]) -> list[dict[str, Any]]:
    """Distinct detail-pattern URLs (newest capture of each), sorted by SURT key."""
    rx = re.compile(cand.detail)
    pool: dict[str, dict[str, Any]] = {}
    for row in domain_rows(http, cand.domain, crawls):
        if rx.search(url_path(row["url"])):
            prev = pool.get(row["urlkey"])
            if prev is None or row["timestamp"] > prev["timestamp"]:
                pool[row["urlkey"]] = row
    return [pool[k] for k in sorted(pool)]


def cached_record(http: Http, domain: str, row: dict[str, Any]) -> tuple[bytes, Record]:
    key = hashlib.sha1(f"{row['filename']}:{row['offset']}".encode()).hexdigest()[:20]
    path = RECORDS_DIR / domain / f"{key}.warc.gz"
    if path.exists():
        raw = path.read_bytes()
    else:
        raw = http.fetch_range(row["filename"], int(row["offset"]), int(row["length"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return raw, parse_record(raw)


def check_page(record: Record, vertical: str) -> dict[str, Any]:
    root = parse_html(decode_body(record.body, record.content_type))
    found = page_labels(root, vertical, record.url)
    password = has_password_input(root)
    words = len(visible_text(root).split())
    filled = sum(1 for v in (found.labels or {}).values() if v not in (None, "", []))
    return {
        "url": record.url,
        "status": record.status,
        "jsonld_objects": found.n_objects,
        "labeled": found.labels is not None,
        "fields_filled": filled,
        "words": words,
        "password_input": password,
    }


def robots_check(http: Http, domain: str, crawls: list[str], urls: list[str]) -> dict[str, Any]:
    """The newest captured robots.txt of the domain vs. the sampled detail URLs."""
    rows: list[dict[str, Any]] = []
    for crawl in crawls:
        rows += [r for r in cdx(http, crawl).robots_rows(domain) if r.get("status") == "200"]
    if not rows:
        return {"found": False, "allowed": {a: 1.0 for a in ROBOTS_AGENTS}}
    hosts = Counter(re.sub(r"^https?://", "", u).split("/")[0] for u in urls)
    main_host = hosts.most_common(1)[0][0] if hosts else domain
    rows.sort(
        key=lambda r: (
            re.sub(r"^https?://", "", r["url"]).split("/")[0] == main_host,
            r["timestamp"],
        )
    )
    row = rows[-1]
    _, record = cached_record(http, domain, row)
    text = decode_body(record.body, record.content_type)
    allowed = {}
    for agent in ROBOTS_AGENTS:
        rules = parse_robots(text, agent)
        allowed[agent] = sum(rules.allowed(url_path(u)) for u in urls) / max(1, len(urls))
    return {
        "found": True,
        "url": row["url"],
        "timestamp": row["timestamp"],
        "crawl": crawl_of(row),
        "allowed": allowed,
    }


def sample_domain(http: Http, cand: Candidate, crawls: list[str]) -> dict[str, Any]:
    rng = random.Random(f"sample:{cand.domain}")
    pool = detail_pool(http, cand, crawls)
    sample = rng.sample(pool, min(SAMPLE_SIZE, len(pool)))
    pages: list[dict[str, Any]] = []
    errors: list[str] = []

    def one(row: dict[str, Any]) -> dict[str, Any]:
        _, record = cached_record(http, cand.domain, row)
        return check_page(record, cand.vertical) | {"crawl": row["crawl"]}

    with ThreadPoolExecutor(max_workers=4) as pool_exec:
        futures = [(row, pool_exec.submit(one, row)) for row in sample]
        for row, fut in futures:
            try:
                pages.append(fut.result())
            except Exception as exc:
                errors.append(f"{row['url']}: {exc!r}")
    n = len(pages)
    jsonld = sum(p["jsonld_objects"] > 0 for p in pages)
    words = [p["words"] for p in pages]
    login = sum(p["password_input"] and p["words"] < MIN_WORDS for p in pages)
    robots = robots_check(http, cand.domain, crawls, [p["url"] for p in pages])
    result: dict[str, Any] = {
        "domain": cand.domain,
        "vertical": cand.vertical,
        "detail_pattern": cand.detail,
        "distinct_detail_urls": len(pool),
        "sampled": len(sample),
        "fetched": n,
        "fetch_errors": errors,
        "jsonld_pages": jsonld,
        "jsonld_rate": round(jsonld / n, 3) if n else 0.0,
        "labeled_pages": sum(p["labeled"] for p in pages),
        "median_words": statistics.median(words) if words else 0,
        "share_over_200_words": round(sum(w > MIN_WORDS for w in words) / n, 3) if n else 0.0,
        "login_walled_pages": login,
        "robots": robots,
        "pages": pages,
    }
    reasons = []
    if n < MIN_DETAIL_SAMPLED:
        reasons.append(f"only {n} detail-pattern pages found/fetched (< {MIN_DETAIL_SAMPLED})")
    if n and result["jsonld_rate"] < MIN_JSONLD_RATE:
        reasons.append(
            f"{VERTICALS[cand.vertical].primary_type} JSON-LD on {result['jsonld_rate']:.0%} of "
            f"sampled detail pages (< {MIN_JSONLD_RATE:.0%})"
        )
    if n and result["median_words"] <= MIN_WORDS:
        reasons.append(
            f"median visible text {result['median_words']} words (<= {MIN_WORDS}): JS shell"
        )
    if n and login / n > MAX_LOGIN_SHARE:
        reasons.append(f"login wall on {login}/{n} sampled pages")
    for agent, share in robots["allowed"].items():
        if share < MIN_ROBOTS_ALLOWED:
            reasons.append(f"robots.txt disallows {1 - share:.0%} of detail URLs for {agent}")
    result["passes"] = not reasons
    result["reasons"] = reasons
    return result


def cmd_sample(args: argparse.Namespace) -> None:
    http = Http()
    counts = load_counts()
    crawls = counts["crawls"] or latest_crawls(http)
    by_domain = {c.domain: c for c in CANDIDATES}
    todo = [
        by_domain[d]
        for d, info in sorted(counts["domains"].items())
        if info.get("passes") and d in by_domain
    ]
    if args.only:
        todo = [by_domain[d] for d in args.only]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    def work(cand: Candidate) -> dict[str, Any]:
        out = SAMPLES_DIR / f"{cand.domain}.json"
        if out.exists() and not args.redo:
            cached: dict[str, Any] = json.loads(out.read_text())
            if cached.get("detail_pattern") == cand.detail:
                return cached
        try:
            result = sample_domain(http, cand, crawls)
        except Exception as exc:
            result = {
                "domain": cand.domain,
                "vertical": cand.vertical,
                "detail_pattern": cand.detail,
                "passes": False,
                "reasons": [f"sampling failed: {exc!r}"],
            }
        out.write_text(json.dumps(result, indent=1))
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        for res in pool.map(work, todo):
            print(
                f"{res['domain']:28s} {res['vertical']:8s} "
                f"detail={res.get('distinct_detail_urls', '-')!s:>6} "
                f"n={res.get('fetched', '-')!s:>3} jsonld={res.get('jsonld_rate', '-')!s:>5} "
                f"words={res.get('median_words', '-')!s:>6} "
                f"{'PASS' if res['passes'] else 'fail: ' + '; '.join(res['reasons'])}",
                flush=True,
            )
    print("stats:", http.stats)


# --------------------------------------------------------------------------- step 4 (choice)


def rank_key(sample: dict[str, Any]) -> tuple[float, int, str]:
    """Higher JSON-LD rate (to 0.1) first, then more distinct detail URLs (up to the cap)."""
    detail = min(int(sample.get("distinct_detail_urls", 0)), 20000)
    return (-round(float(sample["jsonld_rate"]), 1), -detail, sample["domain"])


def choose(
    counts: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    candidates: tuple[Candidate, ...] = CANDIDATES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Two passing domains per main vertical where possible; shortfalls are filled from the
    extra vertical (books or news) with the most passing domains, then from any vertical.
    Every other candidate is returned as rejected with its reason."""
    passing: dict[str, list[dict[str, Any]]] = {}
    for sample in samples.values():
        if sample.get("passes"):
            passing.setdefault(sample["vertical"], []).append(sample)
    for items in passing.values():
        items.sort(key=rank_key)
    operator = {c.domain: c.operator or c.domain for c in candidates}
    chosen: list[dict[str, Any]] = []
    shadowed: dict[str, str] = {}

    def take(items: list[dict[str, Any]], n: int) -> None:
        taken = 0
        for item in items:
            if taken >= n or item in chosen or item["domain"] in shadowed:
                continue
            twin = next(
                (
                    c["domain"]
                    for c in chosen
                    if operator.get(c["domain"]) == operator.get(item["domain"])
                ),
                None,
            )
            if twin is not None:
                shadowed[item["domain"]] = twin
                continue
            chosen.append(item)
            taken += 1

    for vertical in MAIN_VERTICALS:
        take(passing.get(vertical, []), 2)
    if len(chosen) < N_SELECT:
        extras = sorted(EXTRA_VERTICALS, key=lambda v: -len(passing.get(v, [])))
        take(passing.get(extras[0], []), N_SELECT - len(chosen))
    if len(chosen) < N_SELECT:
        take(
            sorted((s for items in passing.values() for s in items), key=rank_key),
            N_SELECT - len(chosen),
        )
    chosen_names = {s["domain"] for s in chosen}
    rejected: list[dict[str, Any]] = []
    for cand in candidates:
        if cand.domain in chosen_names:
            continue
        count = counts["domains"].get(cand.domain, {})
        sample = samples.get(cand.domain)
        if "error" in count:
            reason = f"index query failed: {count['error']}"
        elif not count.get("passes"):
            reason = (
                f"{count.get('distinct_urls', 0)} distinct status-200 text/html URLs in the two "
                f"crawls (< {MIN_DISTINCT_URLS})"
            )
        elif sample is None:
            reason = "not sampled"
        elif not sample.get("passes"):
            reason = "; ".join(sample.get("reasons", []))
        elif cand.domain in shadowed:
            reason = (
                f"passed all checks but shares its operator/platform with the selected "
                f"{shadowed[cand.domain]} ({cand.note}); one domain per operator"
            )
        else:
            reason = (
                f"passed all checks but ranked below the chosen {cand.vertical} domains "
                f"(JSON-LD rate {sample['jsonld_rate']:.0%})"
            )
        rejected.append({"domain": cand.domain, "vertical": cand.vertical, "reason": reason})
    return chosen, rejected


def cmd_choose(args: argparse.Namespace) -> None:
    counts = load_counts()
    samples = {p.stem: json.loads(p.read_text()) for p in sorted(SAMPLES_DIR.glob("*.json"))}
    chosen, rejected = choose(counts, samples)
    by_domain = {c.domain: c for c in CANDIDATES}
    today = args.date or dt.date.today().isoformat()
    existing: dict[str, Any] = {}
    if DOMAINS_YAML.exists():
        old = yaml.safe_load(DOMAINS_YAML.read_text()) or {}
        existing = {d["domain"]: d for d in old.get("selected", [])}
    selected = []
    for s in chosen:
        cand = by_domain[s["domain"]]
        prev = existing.get(cand.domain, {})
        selected.append(
            {
                "domain": cand.domain,
                "vertical": cand.vertical,
                "target_type": VERTICALS[cand.vertical].primary_type,
                "crawls": counts["crawls"],
                "detail_pattern": cand.detail,
                "selection_date": prev.get("selection_date", today),
                "evidence": {
                    "distinct_html_urls": counts["domains"][cand.domain]["distinct_urls"],
                    "distinct_detail_urls": s["distinct_detail_urls"],
                    "sampled_detail_pages": s["fetched"],
                    "jsonld_rate": s["jsonld_rate"],
                    "median_visible_words": s["median_words"],
                    "robots_txt": {
                        "found": s["robots"]["found"],
                        "url": s["robots"].get("url"),
                        "crawl": s["robots"].get("crawl"),
                        "detail_urls_allowed": s["robots"]["allowed"],
                    },
                    "login_walled_pages": s["login_walled_pages"],
                },
                **{k: prev[k] for k in ("captures",) if k in prev},
            }
        )
    doc = {
        "description": (
            "Common Crawl silver set: domains whose detail pages carry schema.org JSON-LD of the "
            "vertical's target type. Labels are regenerated from that JSON-LD by labels.py "
            "into data/cc/<domain>/labels.jsonl (never committed)."
        ),
        "crawls": counts["crawls"],
        "selection_date": min(d["selection_date"] for d in selected) if selected else today,
        "criteria": {
            "min_distinct_html_urls": MIN_DISTINCT_URLS,
            "sample_size": SAMPLE_SIZE,
            "sample_from": "distinct detail-pattern URLs (index status 200, mime text/html)",
            "min_jsonld_rate": MIN_JSONLD_RATE,
            "min_median_visible_words": MIN_WORDS,
            "robots_agents": list(ROBOTS_AGENTS),
            "min_robots_allowed_share": MIN_ROBOTS_ALLOWED,
            "max_login_wall_share": MAX_LOGIN_SHARE,
        },
        "selected": selected,
        "rejected": rejected,
    }
    DOMAINS_YAML.write_text(dump_yaml(doc))
    log = {
        "crawls": counts["crawls"],
        "counts": counts["domains"],
        "samples": {d: {k: v for k, v in s.items() if k != "pages"} for d, s in samples.items()},
    }
    SELECTION_JSON.write_text(json.dumps(log, indent=1, sort_keys=True) + "\n")
    for d in selected:
        print(
            f"selected {d['domain']:28s} {d['vertical']:8s} jsonld={d['evidence']['jsonld_rate']}"
        )
    print(f"{len(rejected)} rejected; wrote {DOMAINS_YAML} and {SELECTION_JSON}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("count", help="steps 1-2: distinct HTML URLs per candidate")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_count)
    p = sub.add_parser("survey", help="URL path shapes of one candidate")
    p.add_argument("domain", nargs="+")
    p.add_argument("--pattern", help="detail regex to try instead of the candidate's")
    p.add_argument("--top", type=int, default=30)
    p.set_defaults(fn=cmd_survey)
    p = sub.add_parser("sample", help="step 3: JSON-LD / text / robots / login checks")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("choose", help="step 4: pick 8 domains, write domains.yaml")
    p.add_argument("--date", help="selection date to record (default: today)")
    p.set_defaults(fn=cmd_choose)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
