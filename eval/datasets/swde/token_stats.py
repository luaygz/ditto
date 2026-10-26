"""Cleaned-HTML token statistics for the SWDE development slice (LLM budget planning).

For each development site: a seeded random sample of pages, each cleaned with
``ditto.fingerprint.dom.clean_html`` (the tree the extractor and LLM see), serialized with
``lxml.html.tostring``, and counted with tiktoken ``o200k_base``. Writes
``eval/datasets/swde/token_stats.json`` and prints a table.

Usage::

    uv run python eval/datasets/swde/token_stats.py [--data-dir data/swde] [--sample 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

import tiktoken
from lxml import html

from ditto.evaluation.datasets import swde
from ditto.fingerprint.dom import clean_html

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUT_PATH = HERE / "token_stats.json"
ENCODING = "o200k_base"


def summarize(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]
    return {
        "mean": round(statistics.fmean(ordered), 1),
        "median": float(statistics.median(ordered)),
        "p90": float(p90),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def measure_site(ref: swde.SiteRef, root: Path, sample: int, seed: int, enc: Any) -> dict[str, Any]:
    paths = swde.page_paths(swde.site_dir(root, ref.vertical, ref.site))
    rng = random.Random(f"{seed}:{ref.name}")
    chosen = sorted(rng.sample(paths, min(sample, len(paths))))
    raw_bytes: list[int] = []
    cleaned_chars: list[int] = []
    tokens: list[int] = []
    for path in chosen:
        data = path.read_bytes()
        text = html.tostring(clean_html(data), encoding="unicode")
        raw_bytes.append(len(data))
        cleaned_chars.append(len(text))
        tokens.append(len(enc.encode_ordinary(text)))
    return {
        "site": ref.name,
        "pages": len(paths),
        "sampled": [swde.page_key(p) for p in chosen],
        "raw_bytes": summarize(raw_bytes),
        "cleaned_chars": summarize(cleaned_chars),
        "cleaned_tokens": summarize(tokens),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "swde")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    enc = tiktoken.get_encoding(ENCODING)
    rows = [
        measure_site(ref, args.data_dir, args.sample, args.seed, enc) for ref in swde.sites("dev")
    ]
    all_means = [r["cleaned_tokens"]["mean"] for r in rows]
    out = {
        "dataset": "swde",
        "split": "dev",
        "encoding": ENCODING,
        "cleaning": "ditto.fingerprint.dom.clean_html, serialized with lxml.html.tostring",
        "sample_per_site": args.sample,
        "seed": args.seed,
        "sites": rows,
        "overall": {
            "dev_sites": len(rows),
            "dev_pages": sum(r["pages"] for r in rows),
            "mean_of_site_mean_tokens": round(statistics.fmean(all_means), 1),
            "median_of_site_mean_tokens": float(statistics.median(all_means)),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{'site':28} {'pages':>6} {'raw KB':>8} {'tok mean':>9} {'tok med':>8} {'tok p90':>8}")
    for r in rows:
        t = r["cleaned_tokens"]
        print(
            f"{r['site']:28} {r['pages']:>6} {r['raw_bytes']['mean'] / 1024:>8.0f} "
            f"{t['mean']:>9.0f} {t['median']:>8.0f} {t['p90']:>8.0f}"
        )
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
