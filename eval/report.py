"""Benchmark charts from the latest results under ``eval/results/``.

Usage::

    uv run python eval/report.py [--results-dir eval/results] [--out-dir docs/img]

Writes ``docs/img/headline_cost_f1.png`` (cost per 1,000 pages, log scale, against normalized F1
for ditto and every baseline on the SWDE test split and the Common Crawl silver set) and
``docs/img/synth_<parameter>.png`` (one clustering curve per synthetic stress parameter), then
prints a summary table per dataset and the exact commands that regenerate every figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # the Agg backend must be chosen before this import

HEADLINE_DATASETS: dict[str, str] = {
    "swde": "SWDE (test split)",
    "cc_silver": "Common Crawl silver",
}
CURVES_CONFIG = "synth_curves"
CURVE_METRICS = (("ari", "adjusted Rand index"), ("purity", "purity"))
NO_COST_FLOOR = 1e-3


@dataclass
class Result:
    name: str
    path: Path
    results: dict[str, Any]
    meta: dict[str, Any]

    @property
    def config_path(self) -> str:
        return str(self.meta.get("config_path") or f"eval/configs/{self.name}.yaml")

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self.results.get("summary") or {})


def latest_results(results_dir: Path) -> dict[str, Result]:
    """Newest completed run per config name (dry runs and runs without results are skipped)."""
    out: dict[str, Result] = {}
    if not results_dir.is_dir():
        return out
    for config_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        runs = sorted(
            (p for p in config_dir.iterdir() if (p / "results.json").is_file()),
            key=lambda p: p.name,
        )
        if not runs:
            continue
        run = runs[-1]
        meta_path = run / "meta.json"
        out[config_dir.name] = Result(
            name=config_dir.name,
            path=run,
            results=json.loads((run / "results.json").read_text()),
            meta=json.loads(meta_path.read_text()) if meta_path.is_file() else {},
        )
    return out


def _overall_f1(result: Result, mode: str = "norm") -> float | None:
    accuracy = result.summary.get("accuracy") or {}
    value = ((accuracy.get("overall") or {}).get(mode) or {}).get("f1")
    return float(value) if isinstance(value, (int, float)) else None


def headline_points(results: dict[str, Result]) -> dict[str, list[dict[str, Any]]]:
    """Per headline dataset: one point per method (label, $/1k pages, F1, extrapolated)."""
    points: dict[str, list[dict[str, Any]]] = {k: [] for k in HEADLINE_DATASETS}
    for r in results.values():
        dataset = r.results.get("dataset")
        if dataset not in HEADLINE_DATASETS or r.results.get("status") != "ok":
            continue
        if dataset == "swde" and r.results.get("split") != "test":
            continue
        f1 = _overall_f1(r)
        if f1 is None:
            continue
        cost = r.summary.get("cost") or {}
        usd = cost.get("usd_per_1k_pages")
        points[str(dataset)].append(
            {
                "label": r.results.get("method_label") or r.name,
                "name": r.name,
                "usd_per_1k_pages": float(usd) if isinstance(usd, (int, float)) else None,
                "f1": f1,
                "extrapolated": bool(cost.get("extrapolated")),
            }
        )
    return points


def plot_headline(points: dict[str, list[dict[str, Any]]], out: Path) -> Path | None:
    panels = [(k, v) for k, v in points.items() if v]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.8), squeeze=False)
    for ax, (dataset, pts) in zip(axes[0], panels, strict=True):
        for p in pts:
            usd = p["usd_per_1k_pages"]
            x = usd if usd and usd > 0 else NO_COST_FLOOR
            marker = "*" if str(p["label"]).startswith("ditto") else "o"
            ax.scatter([x], [p["f1"]], s=140 if marker == "*" else 60, marker=marker, zorder=3)
            note = p["label"]
            if not usd:
                note += " (no LLM cost)"
            elif p["extrapolated"]:
                note += " (cost extrapolated)"
            ax.annotate(note, (x, p["f1"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("LLM cost per 1,000 pages (USD, log scale)")
        ax.set_ylabel("F1 (normalized match)")
        ax.set_title(HEADLINE_DATASETS[dataset])
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _value_label(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "none"
    return str(value)


def plot_curves(result: Result, out_dir: Path) -> list[Path]:
    """One PNG per stress parameter: clustering metrics against the parameter's values."""
    written: list[Path] = []
    curves: dict[str, list[dict[str, Any]]] = result.summary.get("curves") or {}
    for param, raw_points in sorted(curves.items()):
        pts = [p for p in raw_points if p.get("status") == "ok"]
        if not pts:
            continue
        xs = list(range(len(pts)))
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, label in CURVE_METRICS:
            ys = [p.get(key) for p in pts]
            if any(isinstance(y, (int, float)) for y in ys):
                ax.plot(
                    xs,
                    [y if isinstance(y, (int, float)) else float("nan") for y in ys],
                    marker="o",
                    label=label,
                )
        if any(isinstance(p.get("f1_norm"), (int, float)) for p in pts):
            ax.plot(
                xs,
                [p.get("f1_norm") or float("nan") for p in pts],
                marker="s",
                label="extraction F1 (normalized)",
            )
        ax.set_xticks(xs, [_value_label(p.get("value")) for p in pts], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(param)
        ax.set_title(f"Synthetic sites: {param} (other parameters at defaults)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        out = out_dir / f"synth_{param}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or value is None:
        return "–" if value is None else str(value)
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def summary_tables(results: dict[str, Result]) -> str:
    """Markdown table per dataset: F1 (normalized, raw), tokens/1k pages, calls/template, speed."""
    by_dataset: dict[str, list[Result]] = {}
    for r in results.values():
        by_dataset.setdefault(str(r.results.get("dataset")), []).append(r)
    lines: list[str] = []
    for dataset in sorted(by_dataset):
        lines += [
            f"### {dataset}",
            "",
            "| config | method | status | F1 norm | F1 raw | input tok/1k pages "
            "| output tok/1k pages | calls/template | $/1k pages | extract pages/s/core |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sorted(by_dataset[dataset], key=lambda r: r.name):
            cost = r.summary.get("cost") or {}
            speed = r.summary.get("speed_pages_per_sec_per_core") or {}
            lines.append(
                f"| {r.name} | {r.results.get('method_label')} | {r.results.get('status')} "
                f"| {_fmt(_overall_f1(r))} | {_fmt(_overall_f1(r, 'raw'))} "
                f"| {_fmt(cost.get('input_tokens_per_1k_pages'), 0)} "
                f"| {_fmt(cost.get('output_tokens_per_1k_pages'), 0)} "
                f"| {_fmt(cost.get('calls_per_template'), 2)} "
                f"| {_fmt(cost.get('usd_per_1k_pages'), 4)} | {_fmt(speed.get('extract'), 1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    args = parser.parse_args(argv)

    results = latest_results(args.results_dir)
    if not results:
        print(f"no results under {args.results_dir}; run `uv run ditto eval run <config>` first")
        return 1
    figures: list[tuple[Path, list[Result]]] = []
    points = headline_points(results)
    headline = plot_headline(points, args.out_dir / "headline_cost_f1.png")
    if headline is not None:
        used = [results[p["name"]] for pts in points.values() for p in pts]
        figures.append((headline, used))
    if CURVES_CONFIG in results:
        for path in plot_curves(results[CURVES_CONFIG], args.out_dir):
            figures.append((path, [results[CURVES_CONFIG]]))

    print(summary_tables(results))
    print("Figures and the commands that regenerate them:")
    for path, used in figures:
        print(f"\n{path}")
        for r in sorted(used, key=lambda r: r.name):
            print(f"  uv run ditto eval run {r.config_path}    # latest: {r.path}")
        print(
            f"  uv run python eval/report.py --results-dir {args.results_dir} "
            f"--out-dir {args.out_dir}"
        )
    if not figures:
        print("  (none: no completed SWDE test, Common Crawl or synthetic-curve results yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
