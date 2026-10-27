"""Download the LiveWeb-IE release, snapshot its pages politely, check labels, write the manifest.

LiveWeb-IE: Yang et al., "LiveWeb-IE: A Benchmark For Online Web Information Extraction",
ICLR 2026, arXiv 2603.13773 (https://arxiv.org/abs/2603.13773). The release (queries, ground
truth, page-group URLs) is https://github.com/sbY99/LiveWeb-IE, pinned here to one commit and
verified by a content digest over every file. It is licensed for non-commercial research only,
with attribution, and may be redistributed only together with its license terms; ditto keeps the
release and every page under ``data/livewebie/`` (gitignored) and commits only scripts, the
derived ``fields/*.yaml`` and ``manifest.json``.

Steps (each is skipped when its output already exists, so a rerun does no redundant network
work):

1. fetch ``codeload.github.com/sbY99/LiveWeb-IE/tar.gz/<commit>`` into ``data/livewebie/`` and
   extract it to ``data/livewebie/release/LiveWeb_IE``; verify the tree digest and structure;
2. snapshot every referenced page per website with :mod:`snapshot` (robots.txt obeyed, >= 1 s
   between requests per host, descriptive User-Agent) into ``data/livewebie/<website>.warc.gz``;
3. label-presence check: the share of ground-truth values found verbatim (whitespace-normalised)
   in the snapshot HTML, per site / group / field: an upper bound on recall from the snapshot,
   which measures drift since the annotation and pages that need JavaScript;
4. write ``eval/datasets/livewebie/manifest.json``.

Usage::

    uv run python eval/datasets/livewebie/download.py [--data-dir data/livewebie]
        [--sites arxiv.org,books.toscrape.com] [--release-only] [--min-presence 0.5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tarfile
import threading
import traceback
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from lxml import etree
from lxml import html as lxml_html
from warcio.archiveiterator import ArchiveIterator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import snapshot  # noqa: E402

from ditto.evaluation.datasets.livewebie import (  # noqa: E402
    CITATION,
    LICENSE_TERMS,
    RELEASE_COMMIT,
    RELEASE_REPO,
    Release,
    fetch_log_entries,
    load_release,
    parse_groundtruth,
    site_fields,
)
from ditto.fingerprint.dom import clean_html  # noqa: E402
from ditto.textutil import normalize_ws  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "livewebie"
MANIFEST_PATH = HERE / "manifest.json"
FIELDS_DIR = HERE / "fields"
TARBALL_URL = f"https://codeload.github.com/{RELEASE_REPO}/tar.gz/{RELEASE_COMMIT}"
RELEASE_COMMIT_DATE = "2026-02-01T08:07:56Z"
# sha256 over sorted "relpath\0sha256(file)\n" of LiveWeb_IE/ (.DS_Store excluded)
RELEASE_TREE_SHA256 = "dceb29577810893a69b4363701f407d70021e221249ccb0a81631b8f3ab71c37"
DATASET_JSON_SHA256 = "122dbef831a6903344b2922529e37267272198a9ad938e49414e1f95693d8fbf"
EXPECTED = {"websites": 15, "groups": 46, "queries": 342, "unique_attributes": 97}
DEFAULT_MIN_PRESENCE = 0.5

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        stamp = dt.datetime.now(dt.UTC).strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)


# --------------------------------------------------------------------------- release


def tree_digest(root: Path) -> str:
    """sha256 over sorted ``relpath\\0sha256(file)\\n`` lines (``.DS_Store`` excluded)."""
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store")
    for p in files:
        rel = p.relative_to(root).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def fetch_release(data_dir: Path, *, offline: bool = False) -> dict[str, Any]:
    """Download + extract the pinned release; verify its digest. Returns manifest fields."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tarball = data_dir / f"LiveWeb-IE-{RELEASE_COMMIT}.tar.gz"
    release_dir = data_dir / "release"
    root = release_dir / "LiveWeb_IE"
    if not tarball.is_file():
        if offline:
            raise SystemExit(f"release tarball missing: {tarball} (run without --offline)")
        log(f"downloading {TARBALL_URL}")
        tmp = tarball.with_suffix(".part")
        with httpx.stream(
            "GET",
            TARBALL_URL,
            headers={"User-Agent": snapshot.USER_AGENT},
            follow_redirects=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        tmp.replace(tarball)
    if not (root / "dataset.json").is_file() or tree_digest(root) != RELEASE_TREE_SHA256:
        if release_dir.exists():
            shutil.rmtree(release_dir)
        tmp_dir = data_dir / "release.part"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(tmp_dir, filter="data")
        (top,) = [p for p in tmp_dir.iterdir() if p.is_dir()]
        top.replace(release_dir)
        shutil.rmtree(tmp_dir)
    digest = tree_digest(root)
    if digest != RELEASE_TREE_SHA256:
        raise SystemExit(f"release tree digest {digest} != pinned {RELEASE_TREE_SHA256}")
    dataset_sha = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
    if dataset_sha != DATASET_JSON_SHA256:
        raise SystemExit(f"dataset.json sha256 {dataset_sha} != pinned {DATASET_JSON_SHA256}")
    return {
        "repo": f"https://github.com/{RELEASE_REPO}",
        "commit": RELEASE_COMMIT,
        "commit_date": RELEASE_COMMIT_DATE,
        "tarball_url": TARBALL_URL,
        "tarball_sha256": hashlib.sha256(tarball.read_bytes()).hexdigest(),
        "tree_sha256": digest,
        "dataset_json_sha256": dataset_sha,
        "path": _rel(root),
        "license_url": f"https://github.com/{RELEASE_REPO}#license-and-terms-of-use",
        "license_terms": LICENSE_TERMS,
        "citation": CITATION,
    }


def release_counts(release: Release) -> dict[str, Any]:
    """Structure counts; raises if the release does not match the paper's description."""
    gt_files = {release.root / p for q in release.queries for p in q.labels.values()}
    for path in sorted(gt_files):
        parse_groundtruth(path.read_text(encoding="utf-8"))
    all_urls = [u for urls in release.groups.values() for u in urls]
    types: dict[str, int] = {}
    for q in release.queries:
        types[q.type] = types.get(q.type, 0) + 1
    counts: dict[str, Any] = {
        "websites": len(release.websites()),
        "domains": len(set(release.domains.values())),
        "groups": len(release.groups),
        "queries": len(release.queries),
        "unique_attributes": len({a for q in release.queries for a in q.attributes}),
        "groundtruth_files": len(gt_files),
        "group_urls": len(all_urls),
        "unique_urls": len(set(all_urls)),
        "queries_by_type": dict(sorted(types.items())),
    }
    for key, want in EXPECTED.items():
        if counts[key] != want:
            raise SystemExit(f"release {key} = {counts[key]}, expected {want}")
    return counts


# --------------------------------------------------------------------------- label presence


class PageIndex:
    """Whitespace-normalised text and attribute values of one page, for substring checks.

    ``extractor_input=True`` indexes the tree every method works on (``clean_html``: scripts,
    styles, ``noscript``, ``svg``, ``iframe``, ``template`` and ``<head>`` except ``<title>``
    removed); ``False`` indexes the raw HTML, script text included.
    """

    def __init__(self, body: bytes, *, extractor_input: bool = True) -> None:
        root: etree._Element
        if extractor_input:
            root = clean_html(body)
        else:
            try:
                root = lxml_html.fromstring(body)
            except (etree.ParserError, ValueError):  # empty or unparsable payload: no content
                root = etree.Element("html")
        texts = [_text(t) for t in root.itertext()]
        self.spaced = normalize_ws(" ".join(texts))
        self.joined = normalize_ws("".join(texts))
        self.attrs = {normalize_ws(_text(v)) for el in root.iter() for v in el.attrib.values()}

    def contains(self, value: str) -> bool:
        v = normalize_ws(value)
        if not v:
            return True
        return v in self.attrs or v in self.spaced or v in self.joined


def _text(value: str | bytes) -> str:
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def snapshot_pages(warc: Path, fetchlog: Path) -> dict[str, bytes]:
    """Benchmark URL -> HTML payload of its final 200 response in the snapshot."""
    by_final: dict[str, bytes] = {}
    with warc.open("rb") as fh:
        for rec in ArchiveIterator(fh):
            if rec.rec_type != "response" or rec.http_headers is None:
                continue
            if rec.http_headers.get_statuscode() != "200":
                continue
            uri = rec.rec_headers.get_header("WARC-Target-URI")
            by_final[uri] = rec.content_stream().read()
    pages: dict[str, bytes] = {}
    for row in fetch_log_entries(fetchlog):
        final = row.get("final_url")
        if row.get("status") == 200 and final in by_final:
            pages[row["url"]] = by_final[final]
    return pages


def label_presence(
    release: Release,
    website: str,
    pages: dict[str, bytes],
    *,
    extractor_input: bool = True,
) -> dict[str, Any]:
    """Share of ground-truth values found in the snapshot, per field, group and overall."""
    index: dict[str, PageIndex] = {}
    per_field: dict[str, dict[str, int]] = {}
    per_group: dict[str, dict[str, int]] = {}
    total = {"values": 0, "found": 0, "labeled_pages": 0, "pages_missing": 0}
    seen_pages: set[str] = set()
    missing_pages: set[str] = set()
    for fd in site_fields(release, website):
        for gid in fd.groups:
            gt = parse_groundtruth(
                release.label_path(website, gid, fd.attribute).read_text(encoding="utf-8")
            )
            for url, vals in gt.values.items():
                body = pages.get(url)
                if body is None:
                    missing_pages.add(url)
                    continue
                seen_pages.add(url)
                idx = index.get(url)
                if idx is None:
                    idx = index[url] = PageIndex(body, extractor_input=extractor_input)
                found = sum(idx.contains(v) for v in vals)
                for bucket in (
                    per_field.setdefault(fd.name, {"values": 0, "found": 0}),
                    per_group.setdefault(gid, {"values": 0, "found": 0}),
                    total,
                ):
                    bucket["values"] += len(vals)
                    bucket["found"] += found
    total["labeled_pages"] = len(seen_pages)
    total["pages_missing"] = len(missing_pages - seen_pages)
    return {
        "overall": _rate(total),
        "values": total["values"],
        "found": total["found"],
        "labeled_pages": total["labeled_pages"],
        "labeled_pages_not_in_snapshot": total["pages_missing"],
        "by_group": {g: _rate(c) for g, c in sorted(per_group.items())},
        "by_field": {f: _rate(c) for f, c in sorted(per_field.items())},
    }


def _rate(c: dict[str, int]) -> float | None:
    return round(c["found"] / c["values"], 4) if c["values"] else None


# --------------------------------------------------------------------------- snapshot audit


def audit_snapshot(
    warc: Path, log_path: Path, urls: Sequence[str], interval: float
) -> dict[str, Any]:
    """Re-check a finished snapshot from its WARC and fetch log alone.

    Every request record is checked against the ``robots.txt`` archived before it in the same
    file (its session), every benchmark URL must have a settled fetch-log row, and every logged
    200 must match the archived payload's SHA-256. ``WARC-Date`` has one-second resolution and is
    stamped when the response arrives, so the archive bounds the request rate only per window:
    ``max_requests_in_any_60s`` may exceed ``60 / interval`` by the requests whose responses
    straddle the window edges (``throttle_ok`` allows two).
    """
    stamps: list[dt.datetime] = []
    robots = snapshot.Robots.allow_all()
    sessions = disallowed = 0
    statuses: Counter[str] = Counter()
    payloads: dict[str, str] = {}
    with warc.open("rb") as fh:
        for rec in ArchiveIterator(fh):
            uri = rec.rec_headers.get_header("WARC-Target-URI") or ""
            is_robots = urlsplit(uri).path == "/robots.txt"
            if rec.rec_type == "request":
                date = rec.rec_headers.get_header("WARC-Date") or ""
                stamps.append(dt.datetime.fromisoformat(date.replace("Z", "+00:00")))
                if is_robots:
                    sessions += 1
                elif not robots.allowed(uri):
                    disallowed += 1
            elif rec.rec_type == "response" and rec.http_headers is not None:
                code = rec.http_headers.get_statuscode()
                statuses[code] += 1
                body = rec.content_stream().read()
                if is_robots:
                    robots = _archived_robots(int(code), body)
                elif code == "200":
                    payloads[uri] = hashlib.sha256(body).hexdigest()
    latest = snapshot.latest_rows(fetch_log_entries(log_path))
    unsettled = [
        u
        for u in dict.fromkeys(urls)
        if u not in latest
        or (latest[u].get("status") is None and latest[u].get("error") != snapshot.DISALLOWED)
    ]
    mismatched = [
        r["url"]
        for r in latest.values()
        if r.get("status") == 200 and payloads.get(r.get("final_url", "")) != r.get("sha256")
    ]
    stamps.sort()
    window = start = 0
    for end, stamp in enumerate(stamps):
        while (stamp - stamps[start]).total_seconds() >= 60:
            start += 1
        window = max(window, end - start + 1)
    nominal = int(60 // interval)
    checks = {
        "robots_ok": disallowed == 0,
        "throttle_ok": window <= nominal + 2,
        "all_urls_settled": not unsettled,
        "payloads_match_log": not mismatched,
    }
    return {
        "requests": len(stamps),
        "sessions": sessions,
        "first_request": _iso(stamps[0]) if stamps else None,
        "last_request": _iso(stamps[-1]) if stamps else None,
        "response_statuses": dict(sorted(statuses.items())),
        "interval_s": interval,
        "max_requests_in_any_60s": window,
        "nominal_max_in_60s": nominal,
        "requests_disallowed_by_robots": disallowed,
        "urls_unsettled": len(unsettled),
        "payload_mismatches": len(mismatched),
        **checks,
        "ok": all(checks.values()),
    }


def _archived_robots(status: int, body: bytes) -> snapshot.Robots:
    """The rules a fetched ``robots.txt`` stands for (as :func:`snapshot.fetch_robots`)."""
    if 200 <= status < 300:
        return snapshot.parse_robots(body.decode("utf-8", errors="replace"))
    if 400 <= status < 500:
        return snapshot.Robots.allow_all()
    return snapshot.Robots.disallow_all()


def _iso(stamp: dt.datetime) -> str:
    return stamp.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- main


def run_snapshots(
    release: Release, websites: Sequence[str], data_dir: Path
) -> dict[str, snapshot.SiteResult]:
    def one(website: str) -> snapshot.SiteResult:
        try:
            return snapshot.snapshot_site(website, release.site_urls(website), data_dir, log=log)
        except Exception:
            log(f"{website}: snapshot failed:\n{traceback.format_exc()}")
            raise

    with ThreadPoolExecutor(max_workers=max(1, len(websites))) as pool:
        results = list(pool.map(one, websites))
    return {r.website: r for r in results}


def site_entry(
    release: Release,
    website: str,
    result: snapshot.SiteResult,
    data_dir: Path,
    min_presence: float,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "domain": release.domains[website],
        "host": result.host,
        "groups": sorted(g for (s, g) in release.groups if s == website),
        "urls": len(release.site_urls(website)),
        "robots": result.robots,
        "fetch_counts": result.counts,
    }
    fields_path = FIELDS_DIR / f"{website}.yaml"
    if fields_path.is_file():
        entry["fields_yaml"] = _rel(fields_path)
        entry["fields_sha256"] = hashlib.sha256(fields_path.read_bytes()).hexdigest()
    if result.warc is None:
        entry.update(
            requests=result.requests,
            checked_at=result.started_at,
            included=False,
            excluded_reason=result.excluded_reason,
        )
        entry.update(SITE_NOTES.get(website, {}))
        return entry
    warc = result.warc
    log_path = data_dir / "fetchlog" / f"{website}.jsonl"
    audit = audit_snapshot(
        warc, log_path, release.site_urls(website), float(result.robots.get("interval_s", 1.0))
    )
    entry.update(
        requests=audit["requests"],
        fetched_from=audit["first_request"],
        fetched_to=audit["last_request"],
        audit=audit,
    )
    entry["warc"] = _rel(warc)
    entry["warc_bytes"] = warc.stat().st_size
    entry["warc_sha256"] = _sha256_file(warc)
    pages = snapshot_pages(warc, log_path)
    entry["pages_200"] = len(pages)
    presence = label_presence(release, website, pages)
    raw = label_presence(release, website, pages, extractor_input=False)
    entry["label_presence"] = presence
    entry["label_presence_raw_html"] = {"overall": raw["overall"], "by_field": raw["by_field"]}
    overall = presence["overall"]
    if not pages:
        statuses = Counter(
            str(r.get("status") or r.get("error")) for r in fetch_log_entries(log_path)
        )
        entry.update(
            included=False,
            excluded_reason=f"no page fetched with HTTP 200 (outcomes: {dict(statuses)})",
        )
    elif overall is None or overall < min_presence:
        low = {f: r for f, r in presence["by_field"].items() if r is not None and r < min_presence}
        entry.update(
            included=False,
            excluded_reason=(
                f"only {overall} of ground-truth values appear in the extractor input of the "
                f"fetched pages (threshold {min_presence}); fields below it: {low}"
            ),
        )
    else:
        entry["included"] = True
    entry.update(SITE_NOTES.get(website, {}))
    return entry


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--sites", default="", help="comma-separated websites (default: all)")
    ap.add_argument("--release-only", action="store_true", help="download + verify only")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="no network: rebuild the manifest from the release, WARCs and fetch logs on disk",
    )
    ap.add_argument("--min-presence", type=float, default=DEFAULT_MIN_PRESENCE)
    args = ap.parse_args(list(argv) if argv is not None else None)
    data_dir: Path = args.data_dir.resolve()

    release_info = fetch_release(data_dir, offline=args.offline)
    release = load_release(data_dir / "release" / "LiveWeb_IE")
    counts = release_counts(release)
    log(f"release ok: {counts}")
    if args.release_only:
        return 0
    websites = [w for w in release.websites() if not args.sites or w in args.sites.split(",")]
    if args.offline:
        results = recorded_results(websites, data_dir)
    else:
        results = run_snapshots(release, websites, data_dir)
    previous = _load_manifest()
    sites = dict(previous.get("sites", {})) if args.sites else {}
    for website in websites:
        sites[website] = site_entry(release, website, results[website], data_dir, args.min_presence)
        log(
            f"{website}: included={sites[website]['included']} "
            f"{sites[website].get('excluded_reason') or ''}"
        )
    manifest = build_manifest(release_info, counts, sites, data_dir, args.min_presence)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {MANIFEST_PATH}: {manifest['summary']}")
    return 0


def recorded_results(websites: Sequence[str], data_dir: Path) -> dict[str, snapshot.SiteResult]:
    """Every site's recorded snapshot outcome (``--offline``); fails if a site was never run."""
    results: dict[str, snapshot.SiteResult] = {}
    for website in websites:
        recorded = snapshot.load_result(website, data_dir)
        if recorded is None:
            raise SystemExit(
                f"{website}: no finished snapshot under {data_dir} (run without --offline)"
            )
        results[website] = recorded
    return results


def build_manifest(
    release_info: dict[str, Any],
    counts: dict[str, Any],
    sites: dict[str, dict[str, Any]],
    data_dir: Path,
    min_presence: float,
) -> dict[str, Any]:
    """The manifest document from the per-site entries."""
    included = sorted(w for w, s in sites.items() if s.get("included"))
    excluded = sorted(w for w, s in sites.items() if not s.get("included"))
    failing = sorted(w for w in included if not sites[w].get("audit", {}).get("ok"))
    total = counts["websites"]
    status = "ok" if len(included) == total else "partial" if included else "skipped"
    if failing:
        note = f"Not final: the snapshot audit fails for {', '.join(failing)}."
    elif status == "partial":
        note = STATUS_NOTE.format(excluded=len(excluded), total=total)
    else:
        note = "Final."
    spans = [s for s in sites.values() if s.get("fetched_from")]
    return {
        "dataset": "livewebie",
        "status": status,
        "status_note": note,
        "summary": (
            f"{len(included)} of {total} LiveWeb-IE websites usable ({', '.join(included)}); "
            f"excluded: {', '.join(excluded) or 'none'} (see each site's excluded_reason and "
            "diagnosis)"
        ),
        "paper": {
            "title": "LiveWeb-IE: A Benchmark For Online Web Information Extraction",
            "arxiv": "https://arxiv.org/abs/2603.13773",
            "venue": "ICLR 2026",
            "openreview": "https://openreview.net/forum?id=bIODmQ1TNd",
        },
        "release": release_info,
        "permissions": PERMISSIONS,
        "counts": counts,
        "paper_reported": PAPER_REPORTED,
        "snapshot": {
            "user_agent": snapshot.USER_AGENT,
            "min_interval_s_per_host": snapshot.MIN_INTERVAL_S,
            "robots": "RFC 9309; Crawl-delay honoured; Content-Signal ai-input=no excludes host",
            "fetched_from": min((s["fetched_from"] for s in spans), default=None),
            "fetched_to": max((s["fetched_to"] for s in spans), default=None),
            "complete": not failing,
            "audit": (
                "per site under 'audit', recomputed from the WARC and fetch log at manifest time: "
                "every request checked against the robots.txt archived before it, the request "
                "rate per 60 s window (WARC-Date has 1 s resolution and is stamped at response "
                "time), every benchmark URL settled, every logged 200 matching its archived "
                "payload hash"
            ),
            "manifest_written": _now(),
            "data_dir": _rel(data_dir),
            "min_label_presence": min_presence,
        },
        "evaluation_protocol": PROTOCOL_NOTES,
        "known_issues": KNOWN_ISSUES,
        "sites": dict(sorted(sites.items())),
    }


PERMISSIONS: dict[str, Any] = {
    "checked_at": "2026-10-24",
    "assessment": (
        "Usable by third parties for non-commercial research. The release README grants use to "
        "whoever downloads it ('By downloading or using this dataset, you agree to the following "
        "terms') under non-commercial-research, attribution and license-on-redistribution terms "
        "that it says comply with the agreements made with the website providers; the paper "
        "requires benchmark users' systems to access the listed URLs (live evaluation), so "
        "fetching them is the intended use. ditto uses it for non-commercial research evaluation, "
        "keeps release and pages under gitignored data/, and commits only derived field "
        "definitions (with the license terms) and this manifest. Each host's robots.txt is "
        "obeyed at fetch time; a host whose robots.txt carries Content-Signal ai-input=no is "
        "excluded because ditto feeds page content to an LLM."
    ),
    "evidence": [
        {
            "url": "https://arxiv.org/abs/2603.13773",
            "finding": "Abstract: 'Based on trusted and permission-granted websites'; footnote "
            "links the dataset at https://github.com/sbY99/LiveWeb-IE.",
        },
        {
            "url": "https://arxiv.org/html/2603.13773v1",
            "finding": "Ethics statement: robots.txt and Terms of Use reviewed, administrators "
            "contacted for explicit permission 'to include their website in our benchmark'; "
            "'We will release our benchmark under a license that restricts its application to "
            "research purposes only'. Sec. 1/3: 'LiveWeb-IE mandates evaluation directly on live "
            "websites ... by requiring WIE systems to access the target URL'.",
        },
        {
            "url": "https://github.com/sbY99/LiveWeb-IE",
            "finding": "README 'License and Terms of Use' (non-commercial research only, "
            "attribution, no redistribution without the license); GitHub API reports no SPDX "
            "license; repository holds dataset.json, group.jsonl per site and 179 ground-truth "
            "files.",
        },
    ],
}

PAPER_REPORTED: dict[str, Any] = {
    "_label": "numbers reported by the LiveWeb-IE paper (arXiv 2603.13773v1), not measured by ditto",
    "domains": 8,
    "websites": 15,
    "page_groups": 46,
    "queries": 342,
    "unique_attributes": 97,
    "web_pages_per_website_tables_7_9": {
        "arxiv.org": 87,
        "iclr.cc": 292,
        "neurips.cc": 311,
        "icml.cc": 225,
        "huggingface.co": 230,
        "marinespecies.org": 73,
        "fueleconomy.gov": 69,
        "thesportsdb.com": 261,
        "dp.la": 184,
        "themealdb.com": 298,
        "thecocktaildb.com": 495,
        "books.toscrape.com": 1100,
        "sandbox.oxylabs.io": 1143,
        "scrapethissite.com": 32,
        "quotes.toscrape.com": 70,
    },
    "static_vs_live_gap": "Appendix F: HTML-based LLM wrapper generation loses >15% F1 on the "
    "Nov 2025 versions of SWDE pages vs. their snapshots (GPT-4o: -15.14% avg).",
}

PROTOCOL_NOTES: list[str] = [
    "The paper evaluates systems on the live sites at evaluation time; ditto evaluates on the "
    "frozen snapshot recorded here, so ditto's numbers are not directly comparable with the "
    "paper's.",
    "The paper gives systems free-form queries; ditto is given per-site prompted-mode "
    "fields.yaml files translated mechanically from those queries "
    "(eval/datasets/livewebie/build_fields.py), so attribute identification is not tested.",
    "Field kind follows the query type: type_1/type_2 attributes are scalar, type_3/type_4 "
    "attributes are lists; a site using one attribute name both ways gets '<attr>_list' for the "
    "list-valued one.",
    "Complexity breakdown: mean field F1 over the fields asked for by each type's queries "
    "(field-level), not the paper's per-query precision/recall.",
    "Image and link ground truth are raw @src/@href values (often relative), compared as such.",
    "Pages are labeled only with the fields annotated for their page group.",
]

KNOWN_ISSUES: list[str] = [
    "Release README swaps the Type II/III descriptions; dataset.json follows the paper "
    "(type_2 = several single-valued attributes, type_3 = one list-valued attribute).",
    "iclr.cc, icml.cc and neurips.cc g_000 'author' is queried as a list (type_3) but annotated "
    "as one ' · '-joined string per page; kept as published.",
    "Some group.jsonl files repeat URLs (4915 listed, 4874 unique); duplicates are fetched once.",
    "Ground-truth values may contain newlines (huggingface.co titles); the parser keeps them.",
    "Four release URLs contain U+FFFD or U+2019 (thecocktaildb.com 12756/178368/178369, "
    "themealdb.com 53062); they are requested percent-encoded, the WARC-Target-URI is the "
    "encoded form and the fetch log maps it back to the release URL.",
    "Label presence ('label_presence') is measured on the extractor input (clean_html: no "
    "script, style, noscript, svg, iframe, template or head except title); "
    "'label_presence_raw_html' is the same check on the raw HTML, script text included.",
]

_CONFERENCE = (
    "Excluded: the shared virtual-conference site (neurips.cc / iclr.cc / icml.cc) was "
    "redesigned after annotation (checked on the archived pages 2026-10-27). Poster pages "
    "(g_000): the author list is now one div.event-organizers text joined by '⋅' (U+22C5) "
    "where the ground truth joins with ' · ' (U+00B7), and the '#abstract_details' link that "
    "every abstract_link label holds no longer exists (the abstract is inline); only the title "
    "still matches. Search pages (g_001, papers.html?filter=title&search=...): the labeled, "
    "filtered list is rendered by JavaScript; the server HTML carries the complete unfiltered "
    "paper list ({papers} poster links) inside <noscript>, which the extractor input drops, and "
    "no author names or 'papers.html?filter=author' links at all."
)

SITE_NOTES: dict[str, dict[str, Any]] = {
    "dp.la": {
        "diagnosis": (
            "Excluded: every page request (184 URLs, robots.txt Crawl-delay 2 s honoured, paths "
            "allowed) was answered by CloudFront with HTTP 403 'Request blocked' "
            "(x-cache: Error from cloudfront) while robots.txt itself was served; the block sits "
            "in front of the site, and getting past it would mean disguising the client, which "
            "the fetch policy rules out. Re-probed 2026-10-27 (robots.txt and one page): still "
            "403."
        ),
    },
    "fueleconomy.gov": {
        "diagnosis": (
            "Excluded: www.fueleconomy.gov closes the connection without an HTTP response to "
            "GET /robots.txt (4 attempts with backoff per run) on every run, 2026-10-24 and "
            "2026-10-27; RFC 9309 section 2.3.1.4 treats an unreachable robots.txt as complete "
            "disallow, so no page was requested."
        ),
    },
    "thesportsdb.com": {
        "diagnosis": (
            "Excluded: robots.txt carries 'Content-Signal: ai-train=no, search=yes, "
            "ai-input=no'; ditto sends page content to an LLM during spec generation, which is "
            "ai-input, so no page was requested."
        ),
    },
    "iclr.cc": {"diagnosis": _CONFERENCE.format(papers="3,828")},
    "icml.cc": {"diagnosis": _CONFERENCE.format(papers="3,340")},
    "neurips.cc": {"diagnosis": _CONFERENCE.format(papers="4,538")},
    "thecocktaildb.com": {
        "field_notes": {
            "cocktail_link": "letter-browse pages now link drinks as /drink/<id>; the annotation "
            "has /drink/<id>-<slug>",
            "cocktail_image_list": "letter-browse thumbnails now end in a size suffix "
            "(<image url>/small); the annotation has the bare image URL",
        },
    },
    "themealdb.com": {
        "field_notes": {
            "meal_image_list": "letter-browse thumbnails are now relative (/images/media/meals/"
            "...); the annotation has absolute https://www.themealdb.com/... URLs",
        },
    },
    "sandbox.oxylabs.io": {
        "field_notes": {
            "image": "Next.js lazy images: the <img> src in the HTML is a data: placeholder; the "
            "/assets/*.svg path is only in <noscript>, script data and the og:image <meta>, all "
            "dropped from the extractor input",
            "image_list": "as image",
            "platform": "the lower-case values ('playstation') come from the og:platform <meta> "
            "in <head>; the page body shows other spellings ('Playstation platform')",
        },
    },
    "scrapethissite.com": {
        "field_notes": {
            "title": "g_000 (pages/ajax-javascript/#<year>): the film table is filled by "
            "JavaScript; the server HTML has an empty hidden table",
            "turtle_name": "g_003 (pages/frames/): the content is a separate document loaded in "
            "an <iframe> (/pages/frames/?frame=i), not part of the page",
            "turtle_image": "as turtle_name",
            "turtle_link": "as turtle_name",
        },
    },
}
"""Verified explanations, added to the site entries: why a site is excluded, and why an included
site's field has (almost) no label value in the extractor input."""

STATUS_NOTE = (
    "Final. 'partial' because {excluded} of the {total} websites are excluded; each exclusion is "
    "a property of "
    "the site at snapshot time (robots.txt policy, a CDN block of the client, or a redesign "
    "since annotation), recorded per site in excluded_reason and diagnosis, not an unfinished "
    "fetch. Every included site's snapshot is complete: every benchmark URL has a settled fetch, "
    "and the audit (robots.txt, request rate, payload hashes) passes."
)


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        data: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data
    return {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
