"""Step 6: markup-noise check of the JSON-LD silver labels.

    uv run python eval/datasets/commoncrawl_silver/noise_check.py [--source labels|selection]
        [--n 200] [--seed 0] [--no-manifest]

Samples ``--n`` labeled pages across the selected domains (an equal share per domain, the
remainder to the first domains in ``domains.yaml``; seeded), strips scripts from each page exactly
as the extractor does (:func:`ditto.evaluation.datasets.cc_silver.visible_text`) and checks
whether every label value appears on the page: scalar labels (or any of their alternative
spellings) must occur in the normalized visible text, a currency symbol in the raw visible text;
URL labels (``image``) must appear by file name in a ``src``/``href``/``content`` attribute; list
labels (``ingredients``) are scored per item. Empty labels are not counted.

Sources:

* ``labels`` (default; the reported check): the downloaded set. Labels come from
  ``data/cc/<domain>/labels.jsonl`` (run ``labels.py`` first), page bytes are re-read from the
  WARC files through ``download.py``'s journal. Domains without labels yet are listed under
  ``domains_missing`` (``complete`` is true only when every selected domain was sampled);
  labels older than the journal are listed under ``labels_stale``. The result is the top level
  of the committed ``noise_check.json``.
* ``selection``: the selection-phase sample records (``data/cc/selection/records/<domain>/``,
  about 50 detail pages per domain cached by ``select.py sample``), labeled on the fly with the
  same code; it works before the download has finished. The result goes under the
  ``selection_sample`` key of ``noise_check.json``.

The check is automated and was run by the build agent, not by a human; the JSON holds rates
only, no page content. ``manifest.json`` is rewritten afterwards unless ``--no-manifest`` (use that
while ``download.py`` is still running: it rewrites the manifest when it finishes).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc import DATA_DIR, DOMAINS_YAML, NOISE_JSON, Record, parse_record
from manifest import write_manifest

from ditto.evaluation.datasets.cc_silver import (
    LABELS_FILE,
    VERTICALS,
    attribute_values,
    iter_labels,
    label_agrees,
    label_page,
    list_agreement,
    normalize_for_match,
    parse_page,
    visible_text,
)

URL_FIELDS = frozenset({"image"})
SOURCES = ("labels", "selection")
SELECTION_RECORDS = DATA_DIR / "selection" / "records"
PERFORMED_BY = "build agent (automated script noise_check.py), not a human"
METHOD = (
    "Each label value is compared with the page the extractor sees (all <script>, <style>, "
    "<noscript>, <svg>, <iframe>, <template> and <head> content except <title> removed). Text is "
    "normalized (case, whitespace, HTML entities, currency symbols, thousands separators, "
    "trailing punctuation) and a label agrees when it occurs as a substring of the normalized "
    "visible text; for scalar labels with alternative spellings (ISO durations, dates, "
    "schema.org enumerations, prices, salary ranges, currency code or symbol, an identifier's "
    "displayed spelling) any alternative counts, and a currency symbol is looked up in the "
    "un-normalized visible text. URL labels agree when their file name occurs in a src/href/"
    "srcset/content attribute. List labels (ingredients) report the share of pages whose every "
    "item agrees and the item-level rate. Empty labels are not counted. Substring matching "
    "is lenient for short values (a currency symbol or code anywhere on the page agrees)."
)


@dataclass
class Sample:
    """One labeled page of the pool: its labels and how to get its captured bytes."""

    url: str
    labels: dict[str, Any]
    load: Callable[[], Record]


def record_spans(site_dir: Path) -> dict[str, tuple[str, int, int]]:
    """URL -> (file, start, end) of its record, from download.py's journal."""
    spans: dict[str, tuple[str, int, int]] = {}
    last_end: dict[str, int] = {}
    path = site_dir / "journal.jsonl"
    if not path.exists():
        return spans
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break
        if "file" not in entry:
            continue
        start = last_end.get(entry["file"], 0)
        spans[entry["url"]] = (entry["file"], start, entry["end"])
        last_end[entry["file"]] = entry["end"]
    return spans


def check_page(
    body: bytes, ctype: str | None, labels: dict[str, Any], vertical: str
) -> dict[str, Any]:
    """Per-field agreement of one page's labels with its script-stripped content."""
    root = parse_page(body, ctype)
    raw = visible_text(root)
    text = normalize_for_match(raw)
    attrs = attribute_values(root)
    result: dict[str, Any] = {}
    for name in VERTICALS[vertical].fields:
        value = labels.get(name)
        if name in VERTICALS[vertical].list_fields:
            share = list_agreement(list(value or []), text)
            if share is not None:
                result[name] = {"items": len(value), "item_share": share}
        else:
            agree = label_agrees(value, text, attrs if name in URL_FIELDS else None, raw)
            if agree is not None:
                result[name] = {"agree": agree}
    return result


def downloaded_pool(domain: str) -> tuple[list[Sample], dict[str, Any]]:
    """Labeled pages of a downloaded domain whose record is journaled, plus pool facts."""
    site_dir = DATA_DIR / domain
    labels_path = site_dir / LABELS_FILE
    journal = site_dir / "journal.jsonl"
    if not labels_path.exists():
        return [], {"labels": False}
    spans = record_spans(site_dir)

    def loader(url: str) -> Callable[[], Record]:
        def load() -> Record:
            name, start, end = spans[url]
            with (site_dir / name).open("rb") as fh:
                fh.seek(start)
                return parse_record(fh.read(end - start))

        return load

    rows = sorted(iter_labels(labels_path), key=lambda r: r["url"])
    pool = [Sample(r["url"], r["labels"], loader(r["url"])) for r in rows if r["url"] in spans]
    stale = journal.exists() and journal.stat().st_mtime > labels_path.stat().st_mtime
    return pool, {"labels": True, "labeled_pages": len(pool), "labels_stale": stale}


def selection_pool(domain: str, vertical: str) -> tuple[list[Sample], dict[str, Any]]:
    """Labeled pages among the selection-phase sample records of a domain."""
    pool: list[Sample] = []
    records = sorted((SELECTION_RECORDS / domain).glob("*.warc.gz"))
    for path in records:
        try:
            record = parse_record(path.read_bytes())
        except ValueError:
            continue
        if record.status != 200:
            continue
        found = label_page(record.body, record.content_type, vertical, record.url)
        if found.labels is not None:
            pool.append(Sample(record.url, found.labels, lambda record=record: record))
    pool.sort(key=lambda s: s.url)
    return pool, {"labels": bool(pool), "records": len(records), "labeled_pages": len(pool)}


def quotas(domains: list[str], n: int) -> dict[str, int]:
    """Equal share of ``n`` per domain; the remainder goes to the first domains."""
    if not domains:
        return {}
    share = {d: n // len(domains) for d in domains}
    for d in domains[: n % len(domains)]:
        share[d] += 1
    return share


def run(source: str, n: int, seed: int) -> dict[str, Any]:
    """The noise check over one source; returns the report section."""
    doc = yaml.safe_load(DOMAINS_YAML.read_text())
    selected: list[dict[str, Any]] = doc["selected"]
    pools: dict[str, list[Sample]] = {}
    facts: dict[str, dict[str, Any]] = {}
    for entry in selected:
        domain, vertical = entry["domain"], entry["vertical"]
        if source == "labels":
            pool, info = downloaded_pool(domain)
        else:
            pool, info = selection_pool(domain, vertical)
        facts[domain] = info
        if pool:
            pools[domain] = pool
    verticals = {e["domain"]: e["vertical"] for e in selected}
    checked = [e["domain"] for e in selected if e["domain"] in pools]
    rng = random.Random(seed)
    field_tot: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    domain_tot: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    sampled: dict[str, int] = {}
    for domain, want in quotas(checked, n).items():
        vertical = verticals[domain]
        chosen = rng.sample(pools[domain], min(want, len(pools[domain])))
        sampled[domain] = len(chosen)
        for sample in chosen:
            record = sample.load()
            result = check_page(record.body, record.content_type, sample.labels, vertical)
            for field, res in result.items():
                for bucket in (field_tot[vertical][field], domain_tot[domain][field]):
                    _add(bucket, res)
    missing = [e["domain"] for e in selected if e["domain"] not in pools]
    return {
        "performed_by": PERFORMED_BY,
        "date": dt.date.today().isoformat(),
        "source": source,
        "seed": seed,
        "requested_pages": n,
        "pages": sum(sampled.values()),
        "pages_per_domain": sampled,
        "domains_checked": checked,
        "domains_missing": missing,
        "complete": not missing,
        "labels_stale": sorted(d for d, f in facts.items() if f.get("labels_stale")),
        "pool": facts,
        "method": METHOD,
        "per_vertical_field": _rates(field_tot),
        "per_domain_field": _rates(domain_tot),
    }


def _add(bucket: dict[str, float], res: dict[str, Any]) -> None:
    bucket["labeled"] = bucket.get("labeled", 0) + 1
    if "agree" in res:
        bucket["agree"] = bucket.get("agree", 0) + int(res["agree"])
        return
    bucket["items"] = bucket.get("items", 0) + res["items"]
    bucket["items_agree"] = bucket.get("items_agree", 0) + round(res["item_share"] * res["items"])
    bucket["agree"] = bucket.get("agree", 0) + int(res["item_share"] == 1.0)


def _rates(table: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, fields in sorted(table.items()):
        out[key] = {}
        for field, b in fields.items():
            item: dict[str, Any] = {
                "labeled": int(b["labeled"]),
                "agree": int(b.get("agree", 0)),
                "rate": round(b.get("agree", 0) / b["labeled"], 3),
            }
            if "items" in b:
                item["items"] = int(b["items"])
                item["item_rate"] = round(b["items_agree"] / b["items"], 3) if b["items"] else None
                item["rate_definition"] = "share of pages where every list item agrees"
            out[key][field] = item
    return out


def merge_report(existing: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    """``noise_check.json`` with one source's section replaced (the other one is kept)."""
    if section["source"] == "selection":
        return {**existing, "selection_sample": section}
    kept = (
        {"selection_sample": existing["selection_sample"]} if "selection_sample" in existing else {}
    )
    return {**section, **kept}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="labels")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not rewrite manifest.json (use while download.py is still running)",
    )
    args = parser.parse_args(argv)
    section = run(args.source, args.n, args.seed)
    existing = json.loads(NOISE_JSON.read_text()) if NOISE_JSON.exists() else {}
    NOISE_JSON.write_text(json.dumps(merge_report(existing, section), indent=1) + "\n")
    print(
        f"{args.source}: {section['pages']} pages from {section['pages_per_domain']}; "
        f"missing: {section['domains_missing'] or 'none'}"
        + (f"; stale labels: {section['labels_stale']}" if section["labels_stale"] else ""),
        flush=True,
    )
    for vertical, fields in section["per_vertical_field"].items():
        print(" ", vertical, {f: v["rate"] for f, v in fields.items()}, flush=True)
    if not args.no_manifest:
        write_manifest()


if __name__ == "__main__":
    main()
