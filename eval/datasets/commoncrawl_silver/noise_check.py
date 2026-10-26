"""Step 6: markup-noise check of the JSON-LD silver labels.

    uv run python eval/datasets/commoncrawl_silver/noise_check.py [--n 200] [--seed 0]

Samples ``--n`` labeled pages across the selected domains (equal share per domain, seeded),
re-reads each page from its WARC file, strips scripts exactly as the extractor does
(:func:`ditto.evaluation.datasets.cc_silver.visible_text`) and checks whether every label value
appears on the page: scalar labels (or any of their alternative spellings) must occur in the
normalized visible text; URL labels (``image``) must appear by file name in a ``src``/``href``/
``content`` attribute; list labels (``ingredients``) are scored per item. Writes per-field and
per-domain agreement rates to the committed ``noise_check.json``. This check is automated and
was run by the build agent, not by a human; it contains rates only, no page content.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc import DATA_DIR, DOMAINS_YAML, NOISE_JSON, parse_record
from manifest import write_manifest

from ditto.evaluation.datasets.cc_silver import (
    LABELS_FILE,
    VERTICALS,
    attribute_values,
    decode_body,
    iter_labels,
    label_agrees,
    list_agreement,
    normalize_for_match,
    visible_text,
)
from ditto.fingerprint.dom import parse_html

URL_FIELDS = frozenset({"image"})


def record_spans(site_dir: Path) -> dict[str, tuple[str, int, int]]:
    """URL -> (file, start, end) of its record, from download.py's journal."""
    spans: dict[str, tuple[str, int, int]] = {}
    last_end: dict[str, int] = {}
    path = site_dir / "journal.jsonl"
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
    root = parse_html(decode_body(body, ctype))
    text = normalize_for_match(visible_text(root))
    attrs = attribute_values(root)
    result: dict[str, Any] = {}
    for name in VERTICALS[vertical].fields:
        value = labels.get(name)
        if name in VERTICALS[vertical].list_fields:
            share = list_agreement(list(value or []), text)
            if share is not None:
                result[name] = {"items": len(value), "item_share": share}
        else:
            agree = label_agrees(value, text, attrs if name in URL_FIELDS else None)
            if agree is not None:
                result[name] = {"agree": agree}
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    doc = yaml.safe_load(DOMAINS_YAML.read_text())
    entries = [e for e in doc["selected"] if (DATA_DIR / e["domain"] / LABELS_FILE).exists()]
    rng = random.Random(args.seed)
    per_domain_n = {e["domain"]: args.n // len(entries) for e in entries}
    for e in entries[: args.n % len(entries)]:
        per_domain_n[e["domain"]] += 1

    field_tot: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    domain_tot: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    sampled: dict[str, int] = {}
    for entry in entries:
        domain, vertical = entry["domain"], entry["vertical"]
        site_dir = DATA_DIR / domain
        rows = sorted(iter_labels(site_dir / LABELS_FILE), key=lambda r: r["url"])
        spans = record_spans(site_dir)
        rows = [r for r in rows if r["url"] in spans]
        chosen = rng.sample(rows, min(per_domain_n[domain], len(rows)))
        sampled[domain] = len(chosen)
        handles: dict[str, io.BufferedReader] = {}
        for row in chosen:
            name, start, end = spans[row["url"]]
            fh = handles.setdefault(name, (site_dir / name).open("rb"))
            fh.seek(start)
            record = parse_record(fh.read(end - start))
            result = check_page(record.body, record.content_type, row["labels"], vertical)
            for field, res in result.items():
                for bucket in (field_tot[vertical][field], domain_tot[domain][field]):
                    bucket["labeled"] = bucket.get("labeled", 0) + 1
                    if "agree" in res:
                        bucket["agree"] = bucket.get("agree", 0) + int(res["agree"])
                    else:
                        bucket["items"] = bucket.get("items", 0) + res["items"]
                        bucket["items_agree"] = bucket.get("items_agree", 0) + round(
                            res["item_share"] * res["items"]
                        )
                        bucket["agree"] = bucket.get("agree", 0) + int(res["item_share"] == 1.0)
        for fh in handles.values():
            fh.close()

    def rates(table: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
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
                    item["item_rate"] = (
                        round(b["items_agree"] / b["items"], 3) if b["items"] else None
                    )
                    item["rate_definition"] = "share of pages where every list item agrees"
                out[key][field] = item
        return out

    report = {
        "performed_by": "build agent (automated script noise_check.py), not a human",
        "date": dt.date.today().isoformat(),
        "seed": args.seed,
        "pages": sum(sampled.values()),
        "pages_per_domain": sampled,
        "method": (
            "Each label value is compared with the page the extractor sees (all <script>, "
            "<style>, <noscript>, <svg>, <iframe>, <template> and <head> content except <title> "
            "removed). Text is normalized (case, whitespace, HTML entities, currency symbols, "
            "thousands separators, trailing punctuation) and a label agrees when it occurs as a "
            "substring of the normalized visible text; for scalar labels with alternative "
            "spellings (ISO durations, schema.org enumerations, prices, ISO dates) any "
            "alternative counts. URL labels agree when their file name occurs in a src/href/"
            "srcset/content attribute. List labels (ingredients) report the share of pages "
            "whose every item agrees and the item-level rate. Empty labels are not counted."
        ),
        "per_vertical_field": rates(field_tot),
        "per_domain_field": rates(domain_tot),
    }
    NOISE_JSON.write_text(json.dumps(report, indent=1) + "\n")
    for vertical, fields in report["per_vertical_field"].items():
        print(vertical, {f: v["rate"] for f, v in fields.items()})
    write_manifest()


if __name__ == "__main__":
    main()
