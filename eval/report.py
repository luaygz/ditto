"""Benchmark charts and the README/docs result tables, from the latest results of every config.

Usage::

    uv run python eval/report.py [--results-dir eval/results] [--configs-dir eval/configs]
                                 [--out-dir docs/img] [--readme README.md]
                                 [--benchmarks docs/benchmarks.md] [--check]

Reads only the newest ``eval/results/<name>/<timestamp>/`` (with a ``results.json``) of each
config in ``eval/configs/`` and writes:

- ``docs/img/headline_cost_f1.png``: cost per 1,000 pages (log scale) against normalized F1 for
  ditto and every baseline on the SWDE test split and the Common Crawl silver set;
- ``docs/img/synth_<parameter>.png``: one clustering curve per synthetic stress parameter;
- every ``<!-- results:<section>:start -->`` ... ``<!-- results:<section>:end -->`` block of
  ``README.md`` (compact tables) and ``docs/benchmarks.md`` (with per-vertical, per-stage and
  per-point detail). Sections: headline, swde, cc_silver, livewebie, clustering, spend, models.
  Each table cites the results directory it was read from and the command that regenerates it.
  A config without results reads "not run yet", never a number.

``--check`` writes nothing and exits 1 when a marker block is out of date.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # the Agg backend must be chosen before this import

HEADLINE_DATASETS: dict[str, str] = {
    "swde": "SWDE (test split)",
    "cc_silver": "Common Crawl silver",
}
HEADLINE_FILE = "headline_cost_f1.png"
CURVES_CONFIG = "synth_curves"
CURVE_METRICS = (("ari", "adjusted Rand index"), ("purity", "purity"))
NO_COST_FLOOR = 1e-3

SECTIONS = (
    "headline",
    "swde",
    "cc_silver",
    "livewebie",
    "clustering",
    "union",
    "spend",
    "models",
)
MARKER = re.compile(
    r"(?P<start><!-- results:(?P<name>[a-z0-9_]+):start -->)(?P<body>.*?)"
    r"(?P<end><!-- results:(?P=name):end -->)",
    re.S,
)
NOT_RUN_YET = "not run yet"
DASH = "–"
REPORT_CMD = "uv run python eval/report.py"
BASELINE_ORDER = ("naive_llm", "crawl4ai_schema", "autoscraper", "handwritten")
METHOD_NAMES = {
    "naive_llm": "naive per-page LLM",
    "crawl4ai_schema": "Crawl4AI generate_schema",
    "autoscraper": "AutoScraper",
    "handwritten": "hand-written XPath (build agent)",
}
STAGES = ("ingest", "fingerprint", "cluster", "analyze", "union", "generate", "verify", "extract")
COMPLEXITY_TYPES = {
    "type_1": "one single-valued attribute",
    "type_2": "several single-valued attributes",
    "type_3": "one list-valued attribute",
    "type_4": "several list-valued attributes",
}


# -- loading ---------------------------------------------------------------------------------------


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

    @property
    def status(self) -> str:
        return str(self.results.get("status") or self.meta.get("status") or "unknown")

    @property
    def reason(self) -> str | None:
        reason = self.results.get("reason") or self.meta.get("reason")
        return str(reason) if reason else None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def sites(self) -> list[dict[str, Any]]:
        return [s for s in self.results.get("sites") or [] if isinstance(s, dict)]


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


@dataclass(frozen=True)
class ConfigInfo:
    """What the tables need from one evaluation config."""

    name: str
    path: str
    dataset: str
    split: str = "all"
    kind: str = "ditto"
    mode: str = "prompted"
    baseline: str | None = None
    until: str = "extract"
    pages_per_site: int | None = None
    budget_usd: float | None = None
    models: dict[str, str] = field(default_factory=dict)
    params_file: str | None = None
    sweep: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], path: str) -> ConfigInfo:
        method = data.get("method") or {}
        params = method.get("params") or {}
        options = data.get("dataset_options") or {}
        pages = (data.get("sample") or {}).get("pages_per_site")
        budget = data.get("budget_usd")
        return cls(
            name=str(data["name"]),
            path=path,
            dataset=str(data.get("dataset")),
            split=str((data.get("selection") or {}).get("split") or "all"),
            kind=str(method.get("kind") or "ditto"),
            mode=str(method.get("mode") or "prompted"),
            baseline=method.get("baseline"),
            until=str(params.get("until") or "extract"),
            pages_per_site=int(pages) if pages else None,
            budget_usd=float(budget) if isinstance(budget, (int, float)) else None,
            models={str(k): str(v) for k, v in (data.get("models") or {}).items()},
            params_file=str(options["params"]) if options.get("params") else None,
            sweep=bool(options.get("sweep")),
        )

    @property
    def label(self) -> str:
        if self.kind == "baseline":
            return METHOD_NAMES.get(self.baseline or "", self.baseline or "baseline")
        if self.until == "cluster":
            return "ditto (clustering)"
        if self.until != "extract":
            return f"ditto ({self.until})"
        return f"ditto ({self.mode})"

    @property
    def uses_llm(self) -> bool:
        return self.kind == "baseline" or self.until == "extract"


@dataclass
class Entry:
    """One evaluation config and its latest result (``None``: not run yet)."""

    config: ConfigInfo
    result: Result | None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    def status_text(self) -> str:
        r = self.result
        if r is None:
            return NOT_RUN_YET
        if r.ok:
            n, n_ok = r.summary.get("n_sites"), r.summary.get("n_ok")
            if isinstance(n, int) and isinstance(n_ok, int) and n != n_ok:
                counts = r.summary.get("status_counts") or {}
                others = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()) if k != "ok")
                return f"ok on {n_ok} of {n} sites ({others})" if others else "ok"
            return "ok"
        label = "not run" if r.status == "not_run" else r.status
        return f"{label}: {_short(r.reason)}" if r.reason else label


def _display(path: Path | str) -> str:
    p = Path(path)
    if p.is_absolute() and p.is_relative_to(Path.cwd()):
        p = p.relative_to(Path.cwd())
    return p.as_posix()


def load_entries(configs_dir: Path, results_dir: Path) -> dict[str, Entry]:
    """Every config under ``configs_dir`` with its latest result; results whose config file is
    gone are kept too (their ``meta.json`` embeds the config)."""
    results = latest_results(results_dir)
    entries: dict[str, Entry] = {}
    for path in sorted(configs_dir.glob("*.yaml")) if configs_dir.is_dir() else []:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "name" not in data:
            continue
        info = ConfigInfo.from_mapping(data, _display(path))
        entries[info.name] = Entry(info, results.get(info.name))
    for name, r in results.items():
        if name in entries:
            continue
        data = r.meta.get("config") or {
            "name": name,
            "dataset": r.results.get("dataset"),
            "selection": {"split": r.results.get("split")},
            "method": r.results.get("method") or {},
        }
        entries[name] = Entry(ConfigInfo.from_mapping({**data, "name": name}, r.config_path), r)
    return entries


# -- charts ----------------------------------------------------------------------------------------


def _overall_f1(result: Result, mode: str = "norm", key: str = "accuracy") -> float | None:
    accuracy = result.summary.get(key) or {}
    return _num(((accuracy.get("overall") or {}).get(mode) or {}).get("f1"))


def headline_points(results: Mapping[str, Result]) -> dict[str, list[dict[str, Any]]]:
    """Per headline dataset: one point per method (label, $/1k pages, F1, extrapolated)."""
    points: dict[str, list[dict[str, Any]]] = {k: [] for k in HEADLINE_DATASETS}
    for r in results.values():
        dataset = r.results.get("dataset")
        if dataset not in HEADLINE_DATASETS or not r.ok:
            continue
        if dataset == "swde" and r.results.get("split") != "test":
            continue
        f1 = _overall_f1(r)
        if f1 is None:
            continue
        cost = r.summary.get("cost") or {}
        points[str(dataset)].append(
            {
                "label": r.results.get("method_label") or r.name,
                "name": r.name,
                "usd_per_1k_pages": _num(cost.get("usd_per_1k_pages")),
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


def _curves(result: Result) -> dict[str, list[dict[str, Any]]]:
    curves = result.summary.get("curves") or {}
    return {str(k): [p for p in v if isinstance(p, dict)] for k, v in sorted(curves.items())}


def curve_params(result: Result) -> list[str]:
    """Stress parameters with at least one completed point (one chart each)."""
    return [p for p, pts in _curves(result).items() if any(x.get("status") == "ok" for x in pts)]


def plot_curves(result: Result, out_dir: Path) -> list[Path]:
    """One PNG per stress parameter: clustering metrics against the parameter's values."""
    written: list[Path] = []
    curves = _curves(result)
    for param in curve_params(result):
        pts = [p for p in curves[param] if p.get("status") == "ok"]
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


# -- formatting ------------------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def _fmt(value: Any, digits: int = 3) -> str:
    v = _num(value)
    return DASH if v is None else f"{v:,.{digits}f}"


def _usd(value: Any) -> str:
    v = _num(value)
    if v is None:
        return DASH
    if v == 0:
        return "$0"
    return f"${v:,.2f}" if v >= 1 else f"${v:.4f}"


def _seconds(value: Any) -> str:
    v = _num(value)
    if v is None:
        return DASH
    if v < 120:
        return f"{v:.1f} s"
    if v < 7200:
        return f"{v / 60:.1f} min"
    return f"{v / 3600:.1f} h"


def _short(text: Any, limit: int = 140) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _get(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
    return data


# -- output targets --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A markdown file with marker blocks; links are made relative to it."""

    path: Path
    detail: bool = False

    def href(self, path: Path | str) -> str:
        rel = os.path.relpath(Path(path).resolve(), self.path.resolve().parent)
        return Path(rel).as_posix()

    def link(self, path: Path | str, text: str | None = None) -> str:
        shown = text if text is not None else _display(path) + ("/" if Path(path).is_dir() else "")
        return f"[{shown}]({self.href(path)})"


@dataclass
class Context:
    entries: dict[str, Entry]
    datasets_dir: Path
    headline: Path | None = None
    curve_charts: dict[str, Path] = field(default_factory=dict)

    def select(
        self, dataset: str, split: str | None = None, *, until: str = "extract"
    ) -> list[Entry]:
        """Configs of ``dataset`` (and ``split``) whose ditto runs stop at ``until``; baselines
        count as ``extract`` (end-to-end extraction)."""
        chosen = [
            e
            for e in self.entries.values()
            if e.config.dataset == dataset
            and (split is None or e.config.split == split)
            and (e.config.until if e.config.kind == "ditto" else "extract") == until
        ]
        return sorted(chosen, key=_method_key)


def _method_key(e: Entry) -> tuple[int, str, str]:
    c = e.config
    if c.kind != "baseline":
        return (0, c.mode, c.name)
    rank = BASELINE_ORDER.index(c.baseline) if c.baseline in BASELINE_ORDER else len(BASELINE_ORDER)
    return (1 + rank, "", c.name)


def _sources(entries: Sequence[Entry], target: Target) -> list[str]:
    """Where every number came from and the commands that regenerate it (one line per config in
    the detailed docs, one line in total in the README)."""
    if not target.detail:
        where = " · ".join(
            target.link(e.result.path, e.config.name)
            if e.result
            else f"`{e.config.name}` (not run)"
            for e in entries
        )
        return [
            f"Sources (latest run per config): {where}. Regenerate with "
            "`uv run ditto eval run eval/configs/<name>.yaml`, then `" + REPORT_CMD + "`."
        ]
    lines = ["Read from the latest run of each config:", ""]
    for e in entries:
        where = target.link(e.result.path) if e.result else NOT_RUN_YET
        lines.append(
            f"- {e.config.label} (`{e.config.name}`): {where}; "
            f"regenerate with `uv run ditto eval run {e.config.path}`"
        )
    lines += ["", f"Then rewrite these tables and charts with `{REPORT_CMD}`."]
    return lines


# -- accuracy tables (SWDE, Common Crawl) ------------------------------------------------------------


def _pages_scored(r: Result, key: str) -> int | None:
    values = [_num(_get(s, key, "pages_scored")) for s in r.sites if s.get("status") == "ok"]
    kept = [v for v in values if v is not None]
    return int(sum(kept)) if kept else None


def _baseline_pages_per_sec(r: Result) -> float | None:
    pages = seconds = 0.0
    for s in r.sites:
        b = _get(s, "speed", "baseline") or {}
        p, t = _num(b.get("pages")), _num(b.get("seconds"))
        if s.get("status") == "ok" and p and t:
            pages, seconds = pages + p, seconds + t
    return pages / seconds if seconds else None


def _speed_cell(e: Entry) -> str:
    assert e.result is not None
    if e.config.kind == "baseline":
        v = _baseline_pages_per_sec(e.result)
        return DASH if v is None else f"{v:,.2f} pages/s (wall)"
    v = _num(_get(e.result.summary, "speed_pages_per_sec_per_core", "extract"))
    return DASH if v is None else f"{v:,.0f} pages/s/core (extract)"


def _cost_cell(cost: Mapping[str, Any]) -> str:
    text = _usd(cost.get("usd_per_1k_pages"))
    notes = [
        note
        for flag, note in (("extrapolated", "extrapolated"), ("price_estimated", "estimated price"))
        if cost.get(flag)
    ]
    return f"{text} ({', '.join(notes)})" if notes and text != DASH else text


def _scope(e: Entry, sample: bool) -> str:
    n = e.config.pages_per_site
    if e.config.kind == "baseline":
        return f"seeded {n}-page sample per site" if n else "all labeled pages"
    if sample:
        return f"the baselines' {n}-page sample per site"
    return "all labeled pages the LLM did not see"


def _accuracy_row(e: Entry, *, sample: bool = False) -> list[str]:
    head = [e.config.label, _scope(e, sample)]
    if not e.ok:
        return [*head, *([DASH] * 9), e.status_text()]
    assert e.result is not None
    key = "accuracy_sample" if sample else "accuracy"
    cost = e.result.summary.get("cost") or {}
    return [
        *head,
        _fmt(_pages_scored(e.result, key), 0),
        _fmt(_overall_f1(e.result, "norm", key)),
        _fmt(_overall_f1(e.result, "raw", key)),
        _fmt(_overall_f1(e.result, "canon", key)),
        _fmt(cost.get("input_tokens_per_1k_pages"), 0),
        _fmt(cost.get("output_tokens_per_1k_pages"), 0),
        _fmt(cost.get("calls_per_template"), 2),
        _cost_cell(cost),
        _speed_cell(e),
        e.status_text(),
    ]


ACCURACY_HEADER = (
    "method",
    "scored on",
    "pages scored",
    "F1 norm",
    "F1 raw",
    "F1 canon",
    "input tokens / 1k pages",
    "output tokens / 1k pages",
    "LLM calls / template",
    "$ / 1k pages",
    "speed",
    "status",
)


def accuracy_table(entries: Sequence[Entry]) -> list[str]:
    rows: list[list[str]] = []
    for e in entries:
        rows.append(_accuracy_row(e))
        has_sample = e.result is not None and e.result.summary.get("accuracy_sample")
        if e.ok and e.config.kind != "baseline" and e.config.pages_per_site and has_sample:
            rows.append(_accuracy_row(e, sample=True))
    return _table(ACCURACY_HEADER, rows)


def _recall_notes(entries: Sequence[Entry]) -> list[str]:
    lines: list[str] = []
    for e in entries:
        if not e.ok or e.config.kind == "baseline":
            continue
        assert e.result is not None
        opt = e.result.summary.get("optional_field_recall") or {}
        if opt:
            lines.append(
                f"- {e.config.label}, optional-field recall (attribute on the page but on none of "
                f"the pages the LLM saw for its template; {_fmt(opt.get('pages'), 0)} pages): "
                f"normalized {_fmt(_get(opt, 'norm', 'recall'))}, "
                f"raw {_fmt(_get(opt, 'raw', 'recall'))}"
            )
        unprompted = e.result.summary.get("unprompted_recall") or {}
        if unprompted:
            lines.append(
                f"- {e.config.label}, recall of the labeled attributes after aligning the "
                f"LLM's field names to label names by value overlap: normalized "
                f"{_fmt(unprompted.get('norm'))}, raw {_fmt(unprompted.get('raw'))}"
            )
    return [*lines, ""] if lines else []


def per_vertical_table(entries: Sequence[Entry]) -> list[str]:
    rows: list[list[str]] = []
    for e in entries:
        if not e.ok:
            continue
        assert e.result is not None
        for vertical, agg in sorted(
            (_get(e.result.summary, "accuracy", "per_vertical") or {}).items()
        ):
            rows.append(
                [
                    vertical,
                    e.config.label,
                    _fmt(agg.get("n_sites"), 0),
                    _fmt(_get(agg, "norm", "precision")),
                    _fmt(_get(agg, "norm", "recall")),
                    _fmt(_get(agg, "norm", "f1")),
                    _fmt(_get(agg, "raw", "f1")),
                    _fmt(_get(agg, "canon", "f1")),
                ]
            )
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])
    header = ("vertical", "method", "sites", "P norm", "R norm", "F1 norm", "F1 raw", "F1 canon")
    return ["Per vertical (mean of its sites):", "", *_table(header, rows), ""]


def stage_speed_table(entries: Sequence[Entry]) -> list[str]:
    rows: list[list[str]] = []
    for e in entries:
        if not e.ok or e.config.kind == "baseline":
            continue
        assert e.result is not None
        speed = e.result.summary.get("speed_pages_per_sec_per_core") or {}
        if not speed:
            continue
        machine = e.result.meta.get("machine") or {}
        rows.append(
            [
                e.config.name,
                *(_fmt(speed.get(s), 0) for s in STAGES),
                _seconds(_get(e.result.summary, "cost", "llm_latency_s_mean")),
                f"{machine.get('cpu_count', DASH)} cores, {machine.get('memory_gb', DASH)} GB",
            ]
        )
    if not rows:
        return []
    header = ("config", *STAGES, "LLM latency (mean)", "machine")
    return [
        "Speed per stage, pages per second per core (mean over sites; LLM wait excluded from "
        "verify):",
        "",
        *_table(header, rows),
        "",
    ]


def _accuracy_block(title: str, entries: Sequence[Entry], target: Target) -> list[str]:
    if not entries:
        return [f"{title}: no evaluation config.", ""]
    lines = [f"**{title}**", "", *accuracy_table(entries), "", *_recall_notes(entries)]
    if target.detail:
        lines += per_vertical_table(entries) + stage_speed_table(entries)
    return [*lines, *_sources(entries, target), ""]


# -- sections --------------------------------------------------------------------------------------


def section_headline(ctx: Context, target: Target) -> list[str]:
    used = [e for d in HEADLINE_DATASETS for e in ctx.select(d) if e.config.split != "dev"]
    if ctx.headline is None:
        return [
            f"Headline chart: {NOT_RUN_YET} (it needs a completed SWDE test-split or Common "
            "Crawl evaluation).",
        ]
    plotted = [
        e for e in used if e.result is not None and e.ok and _overall_f1(e.result) is not None
    ]
    return [
        f"![LLM cost per 1,000 pages against F1]({target.href(ctx.headline)})",
        "",
        "LLM cost per 1,000 pages (log scale; token counts at standard-tier list prices from "
        "`config/prices.yaml`) against normalized F1. Stars are ditto, circles baselines.",
        "",
        *_sources(plotted, target),
    ]


def section_swde(ctx: Context, target: Target) -> list[str]:
    return [
        *_accuracy_block(
            "SWDE test split (frozen, never tuned on)", ctx.select("swde", "test"), target
        ),
        *_accuracy_block(
            "SWDE development slice (tuning allowed)", ctx.select("swde", "dev"), target
        ),
    ]


def _capture_cells(entry: Mapping[str, Any]) -> list[str]:
    captures = entry.get("captures") or {}
    per_crawl = captures.get("index_html_captures") or {}
    index = (
        "; ".join(f"{crawl}: {_fmt(n, 0)}" for crawl, n in sorted(per_crawl.items(), reverse=True))
        if per_crawl
        else DASH
    )
    distinct = captures.get("distinct_urls", _get(entry, "evidence", "distinct_html_urls"))
    downloaded = captures.get("downloaded")
    return [
        index,
        _fmt(distinct, 0),
        _fmt(downloaded, 0) if downloaded is not None else "not downloaded yet",
    ]


def domains_table(datasets_dir: Path, target: Target) -> list[str]:
    path = datasets_dir / "commoncrawl_silver" / "domains.yaml"
    if not path.is_file():
        return [f"Common Crawl domains: `{_display(path)}` not found.", ""]
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = [
        [
            f"`{d.get('domain')}`",
            d.get("vertical", DASH),
            d.get("target_type", DASH),
            ", ".join(d.get("crawls") or doc.get("crawls") or []) or DASH,
            *_capture_cells(d),
        ]
        for d in doc.get("selected") or []
    ]
    header = (
        "domain",
        "vertical",
        "JSON-LD type",
        "crawls",
        "HTML captures in the index",
        "distinct URLs",
        "pages downloaded",
    )
    date = doc.get("selection_date")
    return [
        f"Domains (selected {date}; from {target.link(path)}):" if date else "Domains:",
        "",
        *_table(header, rows),
        "",
    ]


def noise_check_lines(datasets_dir: Path, target: Target) -> list[str]:
    path = datasets_dir / "commoncrawl_silver" / "noise_check.json"
    script = "uv run python eval/datasets/commoncrawl_silver/noise_check.py"
    if not path.is_file():
        return [
            f"Label-noise check (JSON-LD labels against the visible text): {NOT_RUN_YET} "
            f"(`{script}`).",
            "",
        ]
    report = json.loads(path.read_text(encoding="utf-8"))
    who = report.get("performed_by") or "the build agent (automated script), not a human"
    head = (
        f"Label-noise check: each JSON-LD label compared with the text the extractor sees, on "
        f"{_fmt(report.get('pages'), 0)} sampled pages; performed by {who} "
        f"({target.link(path)}, `{script}`)."
    )
    if not target.detail:
        return [head, ""]
    rows = [
        [vertical, name, _fmt(v.get("labeled"), 0), _fmt(v.get("rate")), _fmt(v.get("item_rate"))]
        for vertical, fields in sorted((report.get("per_vertical_field") or {}).items())
        for name, v in fields.items()
    ]
    header = ("vertical", "field", "labeled pages", "agreement", "item agreement (lists)")
    return [head, "", *_table(header, rows), ""]


def section_cc_silver(ctx: Context, target: Target) -> list[str]:
    return [
        *_accuracy_block("Common Crawl silver set", ctx.select("cc_silver"), target),
        *domains_table(ctx.datasets_dir, target),
        *noise_check_lines(ctx.datasets_dir, target),
    ]


def _manifest_lines(datasets_dir: Path, target: Target) -> list[str]:
    path = datasets_dir / "livewebie" / "manifest.json"
    if not path.is_file():
        return [f"LiveWeb-IE snapshot: no manifest at `{_display(path)}`.", ""]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    status = manifest.get("status") or "unknown"
    if status == "skipped":
        return [
            f"LiveWeb-IE skipped: {manifest.get('reason') or 'no reason recorded'} "
            f"({target.link(path)}).",
            "",
        ]
    lines = [
        f"Snapshot status: {status}. {manifest.get('summary') or ''} ({target.link(path)})",
        "",
    ]
    excluded = {
        site: info.get("excluded_reason")
        for site, info in sorted((manifest.get("sites") or {}).items())
        if isinstance(info, dict) and info.get("excluded_reason")
    }
    if excluded:
        lines += [f"- excluded `{site}`: {_short(reason)}" for site, reason in excluded.items()]
        lines.append("")
    return lines


def section_livewebie(ctx: Context, target: Target) -> list[str]:
    entries = ctx.select("livewebie")
    lines = _manifest_lines(ctx.datasets_dir, target)
    for e in entries:
        if not e.ok:
            lines += [f"{e.config.label} (`{e.config.name}`): {e.status_text()}.", ""]
            continue
        assert e.result is not None
        groups = e.result.summary.get("groups") or {}
        rows = [
            [
                f"{t} ({COMPLEXITY_TYPES.get(t, t)})",
                _fmt(_get(groups, "norm", t, "f1")),
                _fmt(_get(groups, "raw", t, "f1")),
                _fmt(_get(groups, "canon", t, "f1")),
                _fmt(_get(groups, "norm", t, "n_sites"), 0),
            ]
            for t in sorted(set(groups.get("norm") or {}) | set(groups.get("raw") or {}))
        ]
        rows.append(
            [
                "all fields",
                _fmt(_overall_f1(e.result)),
                _fmt(_overall_f1(e.result, "raw")),
                _fmt(_overall_f1(e.result, "canon")),
                _fmt(e.result.summary.get("n_ok"), 0),
            ]
        )
        cost = e.result.summary.get("cost") or {}
        lines += [
            f"**{e.config.label}** on the snapshot, mean field F1 per query complexity type:",
            "",
            *_table(("complexity type", "F1 norm", "F1 raw", "F1 canon", "sites"), rows),
            "",
            f"Input tokens per 1k pages {_fmt(cost.get('input_tokens_per_1k_pages'), 0)}, "
            f"LLM calls per template {_fmt(cost.get('calls_per_template'), 2)}, "
            f"$ per 1k pages {_cost_cell(cost)}.",
            "",
        ]
    if entries:
        lines += _sources(entries, target)
    return lines


def _defaults(params_file: str | None) -> dict[str, Any]:
    if not params_file or not Path(params_file).is_file():
        return {}
    data = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _point_value(point: Mapping[str, Any], defaults: Mapping[str, Any], param: str) -> str:
    shown = _value_label(point.get("value"))
    default = defaults.get(param)
    return f"{shown} (default)" if default is not None and point.get("value") == default else shown


def _scale_table(entries: Sequence[Entry]) -> list[str]:
    rows: list[list[str]] = []
    for e in entries:
        if not e.ok:
            rows.append([e.config.name, *([DASH] * 9), e.status_text()])
            continue
        assert e.result is not None
        c = e.result.summary.get("clustering") or {}
        speed = e.result.summary.get("speed_pages_per_sec_per_core") or {}
        rows.append(
            [
                e.config.name,
                _fmt(c.get("n_pages"), 0),
                _fmt(c.get("n_true_templates"), 0),
                _fmt(c.get("n_clusters"), 0),
                _fmt(c.get("ari")),
                _fmt(c.get("purity")),
                _fmt(c.get("over_merge"), 0),
                _fmt(c.get("under_merge"), 0),
                _fmt(speed.get("fingerprint"), 0),
                _fmt(speed.get("cluster"), 0),
                "; ".join([e.status_text(), *map(str, e.result.meta.get("notes") or [])]),
            ]
        )
    header = (
        "evaluation",
        "pages",
        "true templates",
        "clusters found",
        "ARI",
        "purity",
        "over-merged clusters",
        "under-merged templates",
        "fingerprint pages/s/core",
        "cluster pages/s/core",
        "status",
    )
    return _table(header, rows)


def _curve_summary(e: Entry, charts: Mapping[str, Path], target: Target) -> list[str]:
    assert e.result is not None
    defaults = _defaults(e.config.params_file)
    rows: list[list[str]] = []
    for param, points in _curves(e.result).items():
        cells = [
            f"{_point_value(p, defaults, param)}: "
            + (_fmt(p.get("ari")) if p.get("status") == "ok" else str(p.get("status")))
            for p in points
        ]
        ok = [p for p in points if p.get("status") == "ok"]
        chart = target.link(charts[param], "chart") if param in charts else DASH
        rows.append(
            [
                param,
                " · ".join(cells),
                _fmt(min((_num(p.get("ari")) or 0.0 for p in ok), default=None)),
                _fmt(min((_num(p.get("purity")) or 0.0 for p in ok), default=None)),
                chart,
            ]
        )
    header = ("stress parameter", "value: ARI", "min ARI", "min purity", "chart")
    return _table(header, rows)


def _curve_detail(e: Entry) -> list[str]:
    assert e.result is not None
    defaults = _defaults(e.config.params_file)
    rows = [
        [
            param,
            _point_value(p, defaults, param),
            _fmt(p.get("n_pages"), 0),
            _fmt(p.get("n_true_templates"), 0),
            _fmt(p.get("n_clusters"), 0),
            _fmt(p.get("ari")),
            _fmt(p.get("purity")),
            _fmt(p.get("over_merge"), 0),
            _fmt(p.get("under_merge"), 0),
            str(p.get("status")),
        ]
        for param, points in _curves(e.result).items()
        for p in points
    ]
    header = (
        "parameter",
        "value",
        "pages",
        "true templates",
        "clusters",
        "ARI",
        "purity",
        "over-merged",
        "under-merged",
        "status",
    )
    return ["Every point:", "", *_table(header, rows), ""]


def section_clustering(ctx: Context, target: Target) -> list[str]:
    synthetic = ctx.select("synthetic", until="cluster") + ctx.select("synthetic")
    scale = [e for e in synthetic if not e.config.sweep and e.config.name != CURVES_CONFIG]
    curves = [e for e in synthetic if e.config.sweep or e.config.name == CURVES_CONFIG]
    lines: list[str] = []
    if scale:
        lines += ["**Synthetic default site** (frozen seed 1)", "", *_scale_table(scale), ""]
    for e in curves:
        title = f"**Stress curves** (`{e.config.name}`: one parameter at a time from the defaults)"
        if not e.ok:
            lines += [f"{title}: {e.status_text()}.", ""]
            continue
        lines += [title, "", *_curve_summary(e, ctx.curve_charts, target), ""]
        if target.detail:
            lines += _curve_detail(e)
    if not synthetic:
        lines += ["No synthetic evaluation config.", ""]
    return [*lines, *_sources([*scale, *curves], target)]


def _union_row(e: Entry) -> list[str]:
    head = [f"`{e.config.name}`", f"{e.config.dataset} ({e.config.split})"]
    if not e.ok:
        return [*head, *([DASH] * 6), e.status_text()]
    assert e.result is not None
    u = e.result.summary.get("union") or {}
    tokens = u.get("tokens") or {}
    coverage = u.get("coverage") or {}
    covered = (
        f"{_fmt(coverage.get('covered'), 0)} of {_fmt(coverage.get('optional_total'), 0)}"
        if coverage
        else "no ground truth"
    )
    gate = u.get("gate") or {}
    passed = gate.get("pass")
    return [
        *head,
        _fmt(u.get("n_sites"), 0),
        _fmt(u.get("n_templates"), 0),
        " / ".join(_fmt(tokens.get(k), 0) for k in ("min", "median", "max")),
        _fmt(u.get("over_budget"), 0),
        covered,
        DASH if passed is None else ("pass" if passed else "fail"),
        e.status_text(),
    ]


def section_union(ctx: Context, target: Target) -> list[str]:
    entries = [e for d in ("synthetic", "swde") for e in ctx.select(d, until="union")]
    if not entries:
        return ["No union-page evaluation config."]
    header = (
        "evaluation",
        "dataset",
        "sites",
        "templates",
        "union tokens min / median / max",
        "over the token cap",
        "optional fields present in their union page",
        "gate",
        "status",
    )
    return [*_table(header, [_union_row(e) for e in entries]), "", *_sources(entries, target)]


def section_spend(ctx: Context, target: Target) -> list[str]:
    """Billed spend per evaluation. The README lists only evaluations that call an LLM in the
    table; the rest are named in one line each for "not run yet" and "no LLM"."""
    rows: list[list[str]] = []
    pending: list[str] = []
    no_llm: list[str] = []
    total = 0.0
    for e in sorted(ctx.entries.values(), key=lambda e: e.config.name):
        budget = _usd(e.config.budget_usd) if e.config.uses_llm else "no LLM"
        if e.result is None:
            if target.detail:
                rows.append([f"`{e.config.name}`", e.config.label, NOT_RUN_YET, DASH, budget, DASH])
            else:
                pending.append(f"`{e.config.name}`")
            continue
        spend = _num(e.result.meta.get("spend_usd"))
        if spend is None:
            spend = _num(_get(e.result.summary, "cost", "spend_usd"))
        total += spend or 0.0
        if not target.detail and not e.config.uses_llm and not spend:
            no_llm.append(target.link(e.result.path, e.config.name))
            continue
        rows.append(
            [
                f"`{e.config.name}`",
                e.config.label,
                e.status_text(),
                _usd(spend),
                budget,
                target.link(e.result.path, e.result.path.name),
            ]
        )
    header = ("evaluation", "method", "status", "spend (billed)", "budget cap", "results")
    lines = [*_table(header, rows), ""] if rows else []
    if no_llm:
        lines += [f"Run without any LLM call: {', '.join(no_llm)}.", ""]
    if pending:
        lines += [f"Not run yet: {', '.join(pending)}.", ""]
    ran = any(e.result is not None for e in ctx.entries.values())
    total_text = _usd(total) if ran else f"{NOT_RUN_YET} (no evaluation has results)"
    return [
        *lines,
        f"Total billed for the latest run of every evaluation: {total_text}. Spend is what the "
        "provider billed (`meta.json` `spend_usd`): cached calls cost nothing and flex-tier calls "
        "half, so the $ / 1k pages columns, which price every token at standard-tier list prices, "
        "are the ones to compare. Rewrite with `" + REPORT_CMD + "`.",
    ]


def section_models(ctx: Context, target: Target) -> list[str]:
    used: dict[str, set[str]] = {}
    prompts: set[str] = set()
    rows: list[list[str]] = []
    for e in sorted(ctx.entries.values(), key=lambda e: e.config.name):
        if not e.config.uses_llm:
            continue
        pinned = ", ".join(f"{role}: `{m}`" for role, m in sorted(e.config.models.items())) or DASH
        if e.result is None:
            rows.append([f"`{e.config.name}`", pinned, NOT_RUN_YET, DASH])
            continue
        models = sorted(str(m) for m in _get(e.result.meta, "models", "used") or [])
        versions = sorted(str(p) for p in e.result.meta.get("prompt_versions") or [])
        for m in models:
            used.setdefault(m, set()).add(e.config.name)
        prompts.update(versions)
        rows.append(
            [
                f"`{e.config.name}`",
                pinned,
                ", ".join(f"`{m}`" for m in models) or DASH,
                ", ".join(f"`{p}`" for p in versions) or DASH,
            ]
        )

    def evaluations(names: set[str]) -> str:
        return f"{len(names)} evaluation{'s' if len(names) != 1 else ''}"

    summary = (
        "Models used by the published results: "
        + "; ".join(f"`{m}` ({evaluations(names)})" for m, names in sorted(used.items()))
        + ". Prompt versions: "
        + (", ".join(f"`{p}`" for p in sorted(prompts)) or "none recorded")
        + "."
        if used
        else f"Models and prompt versions of the published results: {NOT_RUN_YET} (no LLM "
        "evaluation has results)."
    )
    pinned: dict[str, set[str]] = {}
    for e in ctx.entries.values():
        for role, model in e.config.models.items():
            pinned.setdefault(role, set()).add(model)
    lines = [summary, ""]
    if pinned:
        lines += [
            "Pinned in `eval/configs/*.yaml` (a run fails if the LLM layer resolves another id): "
            + "; ".join(
                f"{role}: " + ", ".join(f"`{m}`" for m in sorted(models))
                for role, models in sorted(pinned.items())
            )
            + ".",
            "",
        ]
    if target.detail and rows:
        header = ("evaluation", "pinned models (config)", "models used", "prompt versions")
        lines += [*_table(header, rows), ""]
    return [*lines, f"Each run records them in `meta.json`; rewrite with `{REPORT_CMD}`."]


SECTION_RENDERERS = {
    "headline": section_headline,
    "swde": section_swde,
    "cc_silver": section_cc_silver,
    "livewebie": section_livewebie,
    "clustering": section_clustering,
    "union": section_union,
    "spend": section_spend,
    "models": section_models,
}


def render_sections(ctx: Context, target: Target) -> dict[str, str]:
    """Markdown body of every results section for ``target``."""
    out: dict[str, str] = {}
    for name in SECTIONS:
        lines = SECTION_RENDERERS[name](ctx, target)
        while lines and not lines[-1]:
            lines.pop()
        out[name] = "\n".join(lines)
    return out


def update_markers(text: str, sections: Mapping[str, str]) -> tuple[str, list[str]]:
    """Replace the body of every known marker block; returns (new text, unknown section names)."""
    unknown: list[str] = []

    def replace(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in sections:
            unknown.append(name)
            return m.group(0)
        return f"{m.group('start')}\n{sections[name]}\n{m.group('end')}"

    new = MARKER.sub(replace, text)
    starts = re.findall(r"<!-- results:([a-z0-9_]+):start -->", text)
    closed = [m.group("name") for m in MARKER.finditer(text)]
    if len(starts) != len(closed):
        raise ValueError(f"unclosed results marker(s): {sorted(set(starts) - set(closed))}")
    return new, unknown


# -- main ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--configs-dir", type=Path, default=Path("eval/configs"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("eval/datasets"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--benchmarks", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if a table is out of date"
    )
    args = parser.parse_args(argv)

    entries = load_entries(args.configs_dir, args.results_dir)
    results = {name: e.result for name, e in entries.items() if e.result is not None}
    ctx = Context(entries=entries, datasets_dir=args.datasets_dir)
    points = headline_points(results)
    if any(points.values()):
        ctx.headline = args.out_dir / HEADLINE_FILE
        if not args.check:
            plot_headline(points, ctx.headline)
    curves = results.get(CURVES_CONFIG)
    if curves is not None and curves.ok:
        ctx.curve_charts = {p: args.out_dir / f"synth_{p}.png" for p in curve_params(curves)}
        if not args.check:
            plot_curves(curves, args.out_dir)

    stale: list[str] = []
    for path, detail in ((args.readme, False), (args.benchmarks, True)):
        if not path.is_file():
            print(f"{path}: not found, skipped")
            continue
        text = path.read_text(encoding="utf-8")
        new, unknown = update_markers(text, render_sections(ctx, Target(path, detail)))
        for name in unknown:
            print(f"{path}: unknown results section {name!r} left unchanged")
        if new == text:
            print(f"{path}: up to date")
        elif args.check:
            stale.append(str(path))
        else:
            path.write_text(new, encoding="utf-8")
            print(f"{path}: results sections rewritten")
    figures = ([ctx.headline] if ctx.headline else []) + list(ctx.curve_charts.values())
    for fig in figures:
        print(f"{'expected' if args.check else 'wrote'} {fig}")
    print(f"{len(results)} of {len(entries)} configs have results under {args.results_dir}")
    if stale:
        print(f"out of date: {', '.join(stale)} (run `{REPORT_CMD}`)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
