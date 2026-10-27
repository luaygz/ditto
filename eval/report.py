"""Benchmark charts and the README/docs result tables, from the latest results of every config.

Usage::

    uv run python eval/report.py [--results-dir eval/results] [--configs-dir eval/configs]
                                 [--datasets-dir eval/datasets] [--analysis-dir eval/analysis]
                                 [--out-dir docs/img] [--readme README.md]
                                 [--benchmarks docs/benchmarks.md] [--check]
    uv run python eval/report.py --spend-by-phase [--ledger .cache/ledger.jsonl]

Reads only the newest ``eval/results/<name>/<timestamp>/`` (with a ``results.json``) of each
config in ``eval/configs/`` and writes:

- ``docs/img/headline_cost_f1.png``: LLM cost per 1,000 pages (log scale) against normalized F1
  on the SWDE test split and the Common Crawl silver set: ditto on every site, every baseline,
  and next to each baseline ditto scored on exactly that baseline's sites and pages;
- ``docs/img/synth_<parameter>.png``: one clustering curve per synthetic stress parameter;
- every ``<!-- results:<section>:start -->`` ... ``<!-- results:<section>:end -->`` block of
  ``README.md`` (compact tables) and ``docs/benchmarks.md`` (with per-vertical, per-stage and
  per-point detail). Sections: headline, swde, cc_silver, livewebie, clustering, union, spend,
  models, reproduce, limitations. Each table cites the results directory it was read from and
  the command that regenerates it. A config without results reads "not run yet", never a number.

Baselines run on fewer sites (budget) or on seeded page samples. Every baseline row is followed
by a "same sites and pages" row: ditto's F1 recomputed from its per-site results on exactly the
sites the baseline scored and, for a sampled baseline, on the same seeded pages
(``accuracy_sample``), aggregated the way the harness aggregates (site, then vertical, then
overall), with ditto's cost pooled over those sites.

``--check`` writes nothing and exits 1 when a marker block is out of date.
``--spend-by-phase`` prints the LLM ledger's spend per build phase (between the commit dates of
the release tags) and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # the Agg backend must be chosen before this import

HEADLINE_DATASETS: dict[str, str] = {
    "swde": "SWDE test split",
    "cc_silver": "Common Crawl silver set",
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
    "reproduce",
    "limitations",
)
MARKER = re.compile(
    r"(?P<start><!-- results:(?P<name>[a-z0-9_]+):start -->)(?P<body>.*?)"
    r"(?P<end><!-- results:(?P=name):end -->)",
    re.S,
)
NOT_RUN_YET = "not run yet"
DASH = "–"
REPORT_CMD = "uv run python eval/report.py"
EVAL_CMD = "uv run ditto eval run"
BASELINE_ORDER = ("naive_llm", "crawl4ai_schema", "autoscraper", "handwritten")
METHOD_NAMES = {
    "naive_llm": "naive per-page LLM",
    "crawl4ai_schema": "Crawl4AI generate_schema",
    "autoscraper": "AutoScraper",
    "handwritten": "hand-written XPath (build agent)",
}
METHOD_SHORT = {
    "naive_llm": "naive",
    "crawl4ai_schema": "Crawl4AI",
    "autoscraper": "AutoScraper",
    "handwritten": "hand-written",
}
COVERAGE_REASONS = {
    "naive_llm": "budget",
    "crawl4ai_schema": "budget: Crawl4AI cost more per site than estimated",
    "autoscraper": "budget: one site per vertical, as it makes many model calls per template",
    "handwritten": "the build agent wrote wrappers for these sites only",
}
SAME_PAGES = "↳ ditto, same sites and pages"
STAGES = ("ingest", "fingerprint", "cluster", "analyze", "union", "generate", "verify", "extract")
COMPLEXITY_TYPES = {
    "type_1": "one single-valued attribute",
    "type_2": "several single-valued attributes",
    "type_3": "one list-valued attribute",
    "type_4": "several list-valued attributes",
}
# Plan figures from the product brief, quoted where results fall short of them.
PLANNED_SAMPLE = 200
PLANNED_CC_CAP = 20_000
SPEED_TARGET = 500.0
NOISE_LOW = 0.8


# -- loading ---------------------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    path: Path
    results: dict[str, Any]
    meta: dict[str, Any]
    billed_all_runs: float = 0.0
    """Sum of ``spend_usd`` over every run of this config kept under the results directory."""
    n_runs: int = 1

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

    def ok_sites(self, key: str = "accuracy") -> dict[str, dict[str, Any]]:
        """Sites that ran and carry a ``key`` score, by site name."""
        return {
            str(s.get("site")): s
            for s in self.sites
            if s.get("status") == "ok" and isinstance(s.get(key), Mapping)
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def latest_results(results_dir: Path) -> dict[str, Result]:
    """Newest completed run per config name (dry runs and runs without results are skipped);
    ``billed_all_runs`` adds up what every kept run of the config was billed."""
    out: dict[str, Result] = {}
    if not results_dir.is_dir():
        return out
    for config_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        runs = sorted((p for p in config_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
        complete = [p for p in runs if (p / "results.json").is_file()]
        if not complete:
            continue
        billed, n_runs = 0.0, 0
        for run in runs:
            meta = _read_json(run / "meta.json")
            if meta is None or meta.get("dry_run"):
                continue
            n_runs += 1
            billed += _num(meta.get("spend_usd")) or 0.0
        run = complete[-1]
        out[config_dir.name] = Result(
            name=config_dir.name,
            path=run,
            results=json.loads((run / "results.json").read_text()),
            meta=_read_json(run / "meta.json") or {},
            billed_all_runs=billed,
            n_runs=max(n_runs, 1),
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
    seed: int | None = None
    budget_usd: float | None = None
    models: dict[str, str] = field(default_factory=dict)
    params_file: str | None = None
    sweep: bool = False
    n_listed_sites: int | None = None
    """Sites named in ``selection.sites`` (``None``: every site of the split)."""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], path: str) -> ConfigInfo:
        method = data.get("method") or {}
        params = method.get("params") or {}
        options = data.get("dataset_options") or {}
        pages = (data.get("sample") or {}).get("pages_per_site")
        seed = (data.get("seeds") or {}).get("run")
        budget = data.get("budget_usd")
        selection = data.get("selection") or {}
        sites = selection.get("sites")
        return cls(
            name=str(data["name"]),
            path=path,
            dataset=str(data.get("dataset")),
            split=str(selection.get("split") or "all"),
            kind=str(method.get("kind") or "ditto"),
            mode=str(method.get("mode") or "prompted"),
            baseline=method.get("baseline"),
            until=str(params.get("until") or "extract"),
            pages_per_site=int(pages) if pages else None,
            seed=int(seed) if isinstance(seed, int) else None,
            budget_usd=float(budget) if isinstance(budget, (int, float)) else None,
            models={str(k): str(v) for k, v in (data.get("models") or {}).items()},
            params_file=str(options["params"]) if options.get("params") else None,
            sweep=bool(options.get("sweep")),
            n_listed_sites=len(sites) if isinstance(sites, list) else None,
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
    def short(self) -> str:
        return METHOD_SHORT.get(self.baseline or "", self.label)

    @property
    def uses_llm(self) -> bool:
        return self.kind == "baseline" or self.until == "extract"

    @property
    def dataset_label(self) -> str:
        if self.dataset == "swde":
            return {"dev": "SWDE dev slice", "test": "SWDE test split"}.get(self.split, "SWDE")
        return {
            "cc_silver": "Common Crawl silver set",
            "livewebie": "LiveWeb-IE snapshot",
            "synthetic": "synthetic sites",
        }.get(self.dataset, self.dataset)


@dataclass
class Entry:
    """One evaluation config and its latest result (``None``: not run yet)."""

    config: ConfigInfo
    result: Result | None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    @property
    def n_ok(self) -> int | None:
        n = self.result.summary.get("n_ok") if self.result else None
        return n if isinstance(n, int) else None

    @property
    def n_sites(self) -> int | None:
        n = self.result.summary.get("n_sites") if self.result else None
        return n if isinstance(n, int) else None

    def status_text(self) -> str:
        r = self.result
        if r is None:
            return NOT_RUN_YET
        if r.ok:
            n, n_ok = self.n_sites, self.n_ok
            if n is not None and n_ok is not None and n != n_ok:
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


# -- formatting ------------------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return fmean(kept) if kept else None


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


def _pct(value: Any) -> str:
    v = _num(value)
    return DASH if v is None else f"{100 * v:.0f}%" if v >= 0.1 else f"{100 * v:.2f}%"


def _seconds(value: Any) -> str:
    v = _num(value)
    if v is None:
        return DASH
    if v < 120:
        return f"{v:.1f} s"
    if v < 7200:
        return f"{v / 60:.1f} min"
    return f"{v / 3600:.1f} h"


def _plural(n: int | float, word: str) -> str:
    return f"{n:,.0f} {word}{'' if n == 1 else 's'}"


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


# -- per-site aggregation (same rules as the harness) ------------------------------------------


def _overall_f1(result: Result, mode: str = "norm", key: str = "accuracy") -> float | None:
    return _num(_get(result.summary, key, "overall", mode, "f1"))


def overall_f1(sites: Iterable[Mapping[str, Any]], key: str, mode: str) -> float | None:
    """Overall F1 of ``sites`` aggregated like the harness: each site's F1 (the mean over its
    fields, as stored in ``results.json``), averaged per vertical, then over verticals."""
    per_vertical: dict[str, list[float | None]] = {}
    for s in sites:
        vertical = str(s.get("vertical") or _get(s, key, "vertical"))
        per_vertical.setdefault(vertical, []).append(_num(_get(s, key, mode, "f1")))
    return _mean(_mean(v) for v in per_vertical.values())


def pooled_cost(sites: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Cost of ``sites`` pooled like the harness's summary: tokens and dollars per 1,000 pages
    over all their pages, LLM calls per template as the mean over sites."""
    costs = [c for s in sites if isinstance(c := s.get("cost"), Mapping) and _num(c.get("pages"))]
    pages = sum(_num(c.get("pages")) or 0.0 for c in costs)
    if not pages:
        return {}
    priced = [c for c in costs if _num(c.get("usd_per_1k_pages")) is not None]
    usd = sum((_num(c["usd_per_1k_pages"]) or 0.0) * (_num(c["pages"]) or 0.0) for c in priced)
    tokens_in = sum(_num(c.get("input_tokens")) or 0.0 for c in costs)
    tokens_out = sum(_num(c.get("output_tokens")) or 0.0 for c in costs)
    return {
        "pages": pages,
        "input_tokens_per_1k_pages": tokens_in * 1000 / pages,
        "output_tokens_per_1k_pages": tokens_out * 1000 / pages,
        "calls_per_template": _mean(_num(c.get("calls_per_template")) for c in costs),
        "usd_per_1k_pages": usd / pages if priced else None,
        "extrapolated": any(c.get("extrapolated") for c in costs),
        "price_estimated": any(c.get("price_estimated") for c in costs),
    }


def _mean_speed(sites: Iterable[Mapping[str, Any]], stage: str = "extract") -> float | None:
    return _mean(_num(_get(s, "speed", stage, "pages_per_sec_per_core")) for s in sites)


@dataclass(frozen=True)
class Comparison:
    """A baseline and ditto scored on exactly the same sites and pages."""

    baseline: Entry
    ditto: Entry
    sites: tuple[str, ...]
    ditto_key: str
    """``accuracy_sample`` (the baseline's seeded page sample) or ``accuracy`` (all pages)."""
    sample: int | None

    def _side(self, side: str) -> tuple[Entry, str]:
        return (self.baseline, "accuracy") if side == "baseline" else (self.ditto, self.ditto_key)

    def site_rows(self, side: str) -> list[dict[str, Any]]:
        entry, key = self._side(side)
        assert entry.result is not None
        rows = entry.result.ok_sites(key)
        return [rows[s] for s in self.sites]

    def f1(self, side: str, mode: str = "norm") -> float | None:
        return overall_f1(self.site_rows(side), self._side(side)[1], mode)

    def pages_scored(self, side: str) -> int:
        key = self._side(side)[1]
        return int(sum(_num(_get(s, key, "pages_scored")) or 0 for s in self.site_rows(side)))

    def cost(self, side: str) -> dict[str, Any]:
        return pooled_cost(self.site_rows(side))

    def speed(self) -> float | None:
        return _mean_speed(self.site_rows("ditto"))

    @property
    def baseline_sites(self) -> int:
        assert self.baseline.result is not None
        return len(self.baseline.result.ok_sites())

    @property
    def partial(self) -> bool:
        """ditto lacks some of the baseline's sites: the baseline is re-scored on the rest."""
        return len(self.sites) < self.baseline_sites


def partner(entry: Entry, entries: Iterable[Entry]) -> Entry | None:
    """The prompted ditto evaluation of ``entry``'s dataset and split."""
    found = [
        e
        for e in entries
        if e.config.kind == "ditto"
        and e.config.mode == "prompted"
        and e.config.until == "extract"
        and e.config.dataset == entry.config.dataset
        and e.config.split == entry.config.split
    ]
    return min(found, key=lambda e: e.config.name) if found else None


def compare(baseline: Entry, ditto: Entry | None) -> Comparison | str | None:
    """ditto on exactly the baseline's sites and pages; a reason when that cannot be computed;
    ``None`` when the baseline itself has no results."""
    if not baseline.ok or baseline.result is None:
        return None
    if ditto is None:
        return "no ditto evaluation of this dataset"
    if ditto.result is None:
        return f"ditto (`{ditto.config.name}`) {NOT_RUN_YET}"
    if not ditto.ok:
        return f"ditto: {ditto.status_text()}"
    n = baseline.config.pages_per_site
    key = "accuracy"
    if n:
        if ditto.config.pages_per_site != n:
            return f"`{ditto.config.name}` scores no {n}-page samples"
        if ditto.config.seed != baseline.config.seed:
            return f"`{ditto.config.name}` samples pages with another seed"
        key = "accuracy_sample"
    scored = ditto.result.ok_sites(key)
    common = tuple(s for s in baseline.result.ok_sites() if s in scored)
    if not common:
        return "no site scored by both"
    return Comparison(baseline, ditto, common, key, n)


# -- charts ----------------------------------------------------------------------------------------


def _point(
    name: str, label: str, kind: str, group: int | None, cost: Mapping[str, Any], f1: float
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "kind": kind,
        "group": group,
        "usd_per_1k_pages": _num(cost.get("usd_per_1k_pages")),
        "f1": f1,
        "extrapolated": bool(cost.get("extrapolated")),
    }


def _sites_phrase(n: int, sample: int | None) -> str:
    return _plural(n, "site") + (f" × {sample} pages" if sample else "")


def headline_points(entries: Mapping[str, Entry]) -> dict[str, list[dict[str, Any]]]:
    """Per headline dataset: ditto on every site, each baseline, and ditto on that baseline's
    sites and pages (same ``group``)."""
    points: dict[str, list[dict[str, Any]]] = {k: [] for k in HEADLINE_DATASETS}
    for dataset in HEADLINE_DATASETS:
        chosen = sorted(
            (
                e
                for e in entries.values()
                if e.config.dataset == dataset
                and e.config.split != "dev"
                and (e.config.kind == "baseline" or e.config.until == "extract")
            ),
            key=_method_key,
        )
        ditto = partner(chosen[0], chosen) if chosen else None
        if ditto is not None and ditto.result is not None and ditto.ok:
            f1 = _overall_f1(ditto.result)
            if f1 is not None:
                label = f"ditto, all {_plural(ditto.n_ok or 0, 'site')}"
                cost = ditto.result.summary.get("cost") or {}
                points[dataset].append(_point(ditto.config.name, label, "ditto", None, cost, f1))
        group = 0
        for b in chosen:
            if b.config.kind != "baseline" or not b.ok or b.result is None:
                continue
            comp = compare(b, ditto)
            if isinstance(comp, Comparison):
                f1, cost, n = comp.f1("baseline"), comp.cost("baseline"), len(comp.sites)
            else:
                f1, cost = _overall_f1(b.result), b.result.summary.get("cost") or {}
                n = b.n_ok or 0
            if f1 is None:
                continue
            label = f"{b.config.label}\n{_sites_phrase(n, b.config.pages_per_site)}"
            points[dataset].append(_point(b.config.name, label, "baseline", group, cost, f1))
            ditto_f1 = comp.f1("ditto") if isinstance(comp, Comparison) else None
            if isinstance(comp, Comparison) and ditto_f1 is not None:
                points[dataset].append(
                    _point(
                        comp.ditto.config.name,
                        f"ditto on {b.config.short}'s {_sites_phrase(n, comp.sample)}",
                        "ditto",
                        group,
                        comp.cost("ditto"),
                        ditto_f1,
                    )
                )
            group += 1
    return points


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _label_points(ax: Any, items: Sequence[tuple[float, float, str, Any]], size: float) -> None:
    """Annotate every point without covering another label or marker: greedy placement in
    points (1/72 inch), right of the marker first, then left, moving away vertically."""
    scale = 72.0 / ax.figure.dpi
    box = ax.get_window_extent()
    frame = (box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale)
    anchors = [tuple(v * scale for v in ax.transData.transform((x, y))) for x, y, _, _ in items]
    taken: list[tuple[float, ...]] = [(px - 6, py - 6, px + 6, py + 6) for px, py in anchors]
    for i in sorted(range(len(items)), key=lambda i: -items[i][1]):
        x, y, text, color = items[i]
        px, py = anchors[i]
        lines = text.split("\n")
        w, h = 0.58 * size * max(len(line) for line in lines), 1.25 * size * len(lines)
        # nearest first: just above or below the marker, right side before left, then farther
        options = [
            (side, dy) for k in range(6) for side in (1, -1) for dy in (2 + 11 * k, -h - 2 - 11 * k)
        ]
        chosen = options[0]
        for side, dy in options:
            x0 = px + 7 if side > 0 else px - 7 - w
            cand = (x0, py + dy, x0 + w, py + dy + h)
            inside = frame[0] <= cand[0] and cand[2] <= frame[2] and frame[1] <= cand[1]
            if inside and cand[3] <= frame[3] and not any(_overlaps(cand, t) for t in taken):
                chosen = (side, dy)
                break
        side, dy = chosen
        x0 = px + 7 if side > 0 else px - 7 - w
        taken.append((x0, py + dy, x0 + w, py + dy + h))
        far = abs(dy) > 14 and abs(dy + h) > 14
        ax.annotate(
            text,
            (x, y),
            xytext=(7 * side, dy),
            textcoords="offset points",
            ha="left" if side > 0 else "right",
            va="bottom",
            fontsize=size,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5} if far else None,
        )


def plot_headline(points: dict[str, list[dict[str, Any]]], out: Path) -> Path | None:
    panels = [(k, v) for k, v in points.items() if v]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.8), squeeze=False)
    palette = plt.get_cmap("tab10").colors
    labels: list[tuple[Any, list[tuple[float, float, str, Any]]]] = []
    for ax, (dataset, pts) in zip(axes[0], panels, strict=True):
        xs = [p["usd_per_1k_pages"] or NO_COST_FLOOR for p in pts]
        ys = [p["f1"] for p in pts]
        colors = ["black" if p["group"] is None else palette[p["group"] % 10] for p in pts]
        for group in sorted({p["group"] for p in pts if p["group"] is not None}):
            pair = [(x, y) for x, y, p in zip(xs, ys, pts, strict=True) if p["group"] == group]
            if len(pair) == 2:
                (x0, y0), (x1, y1) = pair
                ax.plot([x0, x1], [y0, y1], ls=":", lw=1, color=palette[group % 10], zorder=2)
        items: list[tuple[float, float, str, Any]] = []
        for x, y, p, color in zip(xs, ys, pts, colors, strict=True):
            if p["kind"] == "ditto" and p["group"] is None:  # outline: a paired star may sit inside
                ax.scatter(
                    [x], [y], s=320, marker="*", facecolors="none", edgecolors=color, zorder=4
                )
            elif p["kind"] == "ditto":
                ax.scatter([x], [y], s=150, marker="*", color=color, zorder=3)
            else:
                ax.scatter([x], [y], s=55, marker="o", color=color, zorder=3)
            note = p["label"]
            if not p["usd_per_1k_pages"]:
                note += "\nno LLM cost"
            elif p["extrapolated"]:
                note += "\ncost extrapolated"
            items.append((x, y, note, color))
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 6, max(xs) * 8)
        ax.set_ylim(max(0.0, (int(min(ys) * 20) - 2) / 20), 1.07)
        ax.set_xlabel("LLM cost per 1,000 pages (USD, standard list prices, log scale)")
        ax.set_ylabel("F1 (normalized match)")
        ax.set_title(HEADLINE_DATASETS[dataset])
        ax.grid(True, which="both", alpha=0.3)
        labels.append((ax, items))
    fig.tight_layout()
    for ax, items in labels:
        _label_points(ax, items, size=7.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _value_label(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "none"
    return str(value)


def _sorted_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Points in ascending parameter order when every value is a number or a list of numbers."""
    values = [p.get("value") for p in points]
    if all(_num(v) is not None for v in values):
        return sorted(points, key=lambda p: float(p["value"]))
    if all(isinstance(v, list) and all(_num(x) is not None for x in v) for v in values):
        return sorted(points, key=lambda p: [float(x) for x in p["value"]])
    return points


def _curves(result: Result) -> dict[str, list[dict[str, Any]]]:
    curves = result.summary.get("curves") or {}
    return {
        str(k): _sorted_points([p for p in v if isinstance(p, dict)])
        for k, v in sorted(curves.items())
    }


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
    analysis_dir: Path = Path("eval/analysis")

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
            f"`{EVAL_CMD} eval/configs/<name>.yaml`, then `{REPORT_CMD}`."
        ]
    lines = ["Read from the latest run of each config:", ""]
    for e in entries:
        where = target.link(e.result.path) if e.result else NOT_RUN_YET
        lines.append(
            f"- {e.config.label} (`{e.config.name}`): {where}; "
            f"regenerate with `{EVAL_CMD} {e.config.path}`"
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


def _extract_cell(v: float | None) -> str:
    return DASH if v is None else f"{v:,.0f} pages/s/core (extract)"


def _speed_cell(e: Entry) -> str:
    assert e.result is not None
    if e.config.kind == "baseline":
        v = _baseline_pages_per_sec(e.result)
        return DASH if v is None else f"{v:,.2f} pages/s (wall)"
    return _extract_cell(_num(_get(e.result.summary, "speed_pages_per_sec_per_core", "extract")))


def _cost_cell(cost: Mapping[str, Any]) -> str:
    text = _usd(cost.get("usd_per_1k_pages"))
    notes = [
        note
        for flag, note in (("extrapolated", "extrapolated"), ("price_estimated", "estimated price"))
        if cost.get(flag)
    ]
    return f"{text} ({', '.join(notes)})" if notes and text != DASH else text


def _sites_prefix(n: int | None, total: int | None) -> str:
    if n is None:
        return ""
    of = f" of {total}" if total and total > n else ""
    return f"{n}{of} site{'' if n == 1 and not of else 's'}, "


def _scope(e: Entry, total: int | None = None) -> str:
    sites = _sites_prefix(e.n_ok if e.ok else None, total)
    n = e.config.pages_per_site
    if e.config.kind == "baseline":
        return sites + (f"seeded {n}-page sample per site" if n else "all labeled pages")
    return sites + "all labeled pages the LLM did not see"


def _metric_row(
    head: Sequence[str],
    pages: Any,
    f1: Mapping[str, float | None],
    cost: Mapping[str, Any],
    speed: str,
    status: str,
) -> list[str]:
    return [
        *head,
        _fmt(pages, 0),
        _fmt(f1.get("norm")),
        _fmt(f1.get("raw")),
        _fmt(f1.get("canon")),
        _fmt(cost.get("input_tokens_per_1k_pages"), 0),
        _fmt(cost.get("output_tokens_per_1k_pages"), 0),
        _fmt(cost.get("calls_per_template"), 2),
        _cost_cell(cost),
        speed,
        status,
    ]


MODES = ("norm", "raw", "canon")


def _accuracy_row(e: Entry, total: int | None = None) -> list[str]:
    head = [e.config.label, _scope(e, total)]
    if not e.ok:
        return [*head, *([DASH] * 9), e.status_text()]
    assert e.result is not None
    return _metric_row(
        head,
        _pages_scored(e.result, "accuracy"),
        {m: _overall_f1(e.result, m) for m in MODES},
        e.result.summary.get("cost") or {},
        _speed_cell(e),
        e.status_text(),
    )


def _comparison_rows(comp: Comparison) -> list[list[str]]:
    n = len(comp.sites)
    same = f"the same {_plural(n, 'site')}"
    rows: list[list[str]] = []
    if comp.partial:
        rows.append(
            _metric_row(
                [f"↳ {comp.baseline.config.label}, same sites", f"{same} (ditto lacks the others)"],
                comp.pages_scored("baseline"),
                {m: comp.f1("baseline", m) for m in MODES},
                comp.cost("baseline"),
                DASH,
                "ok",
            )
        )
    pages = (
        f"{same}, same samples minus pages the LLM saw"
        if comp.sample
        else f"{same}, labeled pages the LLM did not see"
    )
    rows.append(
        _metric_row(
            [SAME_PAGES, pages],
            comp.pages_scored("ditto"),
            {m: comp.f1("ditto", m) for m in MODES},
            comp.cost("ditto"),
            _extract_cell(comp.speed()),
            "ok",
        )
    )
    return rows


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


def accuracy_table(entries: Sequence[Entry]) -> tuple[list[str], bool]:
    """The table and whether it has a "same sites and pages" row."""
    rows: list[list[str]] = []
    matched = False
    for e in entries:
        if e.config.kind != "baseline":
            rows.append(_accuracy_row(e))
            continue
        ditto = partner(e, entries)
        rows.append(_accuracy_row(e, ditto.n_sites if ditto else None))
        comp = compare(e, ditto)
        if isinstance(comp, Comparison):
            rows += _comparison_rows(comp)
            matched = True
        elif comp is not None:
            rows.append([SAME_PAGES, DASH, *([DASH] * 9), comp])
    return _table(ACCURACY_HEADER, rows), matched


def _ditto_notes(entries: Sequence[Entry], target: Target) -> list[str]:
    lines: list[str] = []
    for e in entries:
        if not e.ok or e.config.kind == "baseline":
            continue
        assert e.result is not None
        sites = e.result.ok_sites().values()
        seen = sum(_num(_get(s, "accuracy", "pages_excluded")) or 0 for s in sites)
        scored = sum(_num(_get(s, "accuracy", "pages_scored")) or 0 for s in sites)
        templates = _num(_get(e.result.summary, "cost", "templates"))
        if scored:
            specs = f" while generating {_plural(templates, 'spec')}" if templates else ""
            lines.append(
                f"- {e.config.label}: the LLM saw {seen:,.0f} of {seen + scored:,.0f} labeled pages "
                f"({_pct(seen / (seen + scored))}){specs}; they are left out of every ditto score."
            )
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
                f"{_fmt(unprompted.get('norm'))}, raw {_fmt(unprompted.get('raw'))}, "
                f"canon {_fmt(unprompted.get('canon'))}"
            )
        precision = e.result.summary.get("schema_precision") or {}
        if precision:
            source = precision.get("file")
            where = f" ({target.link(source)})" if source else ""
            lines.append(
                f"- {e.config.label}, schema precision: {_fmt(precision.get('precision'))} "
                f"({_fmt(precision.get('entity'), 0)} of {_fmt(precision.get('fields'), 0)} "
                f"proposed fields on {_fmt(precision.get('n_sites'), 0)} sites judged entity "
                f"attributes; judged field by field by the build agent, not a human{where})"
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
    return [
        "Per vertical (mean of its sites; each method on its own sites and pages):",
        "",
        *_table(header, rows),
        "",
    ]


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
    table, matched = accuracy_table(entries)
    lines = [f"**{title}**", "", *table, ""]
    if matched and target.detail:
        lines += [
            f"`{SAME_PAGES}` rows: ditto scored on exactly the sites and pages of the baseline "
            "row above them (the same seeded sample for a sampled baseline, less any page the "
            "LLM saw while generating a spec), with ditto's cost on those sites.",
            "",
        ]
    lines += _ditto_notes(entries, target)
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
        "LLM cost per 1,000 pages (log scale; every method's tokens at the same standard-tier "
        "list prices from `config/prices.yaml`) against normalized F1. Black star: ditto on "
        "every site. Each baseline (circle) is joined by a dotted line to ditto (star, same "
        "colour) scored on exactly that baseline's sites and pages. A method without LLM cost "
        "sits at the left edge.",
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


def _cc_domains(datasets_dir: Path) -> tuple[dict[str, Any], Path]:
    path = datasets_dir / "commoncrawl_silver" / "domains.yaml"
    if not path.is_file():
        return {}, path
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (doc if isinstance(doc, dict) else {}), path


def _capture_cells(entry: Mapping[str, Any], detail: bool) -> list[str]:
    captures = entry.get("captures") or {}
    per_crawl = captures.get("index_html_captures") or {}
    index = (
        "; ".join(f"{crawl}: {_fmt(n, 0)}" for crawl, n in sorted(per_crawl.items(), reverse=True))
        if per_crawl
        else ", ".join(entry.get("crawls") or []) or DASH
    )
    distinct = captures.get("distinct_urls", _get(entry, "evidence", "distinct_html_urls"))
    downloaded, planned = captures.get("downloaded"), captures.get("planned")
    if downloaded is None:
        got = "not downloaded yet"
    else:
        got = _fmt(downloaded, 0) + (f" of {_fmt(planned, 0)}" if planned is not None else "")
        parts = captures.get("downloaded_detail"), captures.get("downloaded_other")
        if detail and None not in parts:
            got += f" ({_fmt(parts[0], 0)} detail, {_fmt(parts[1], 0)} other)"
    return [index, _fmt(distinct, 0), got]


def domains_table(datasets_dir: Path, target: Target) -> list[str]:
    doc, path = _cc_domains(datasets_dir)
    if not doc:
        return [f"Common Crawl domains: `{_display(path)}` not found.", ""]
    rows = [
        [
            f"`{d.get('domain')}`",
            d.get("vertical", DASH),
            d.get("target_type", DASH),
            *_capture_cells(d, target.detail),
        ]
        for d in doc.get("selected") or []
    ]
    header = (
        "domain",
        "vertical",
        "JSON-LD type",
        "HTML captures in the index, per crawl",
        "distinct URLs",
        "pages downloaded of planned",
    )
    date = doc.get("selection_date")
    cap = _get(doc, "download", "cap_per_domain")
    head = f"Domains (selected {date}; from {target.link(path)})" if date else "Domains"
    if cap:
        head += (
            f"; download cap {_fmt(cap, 0)} pages per domain, detail pages first, then a sample "
            "of other pages"
        )
    return [head + ":", "", *_table(header, rows), ""]


def _noise_report(datasets_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    """The label-noise check to report: the check on the downloaded pages when it covers every
    domain, else the check on the selection sample when that one does."""
    path = datasets_dir / "commoncrawl_silver" / "noise_check.json"
    report = _read_json(path)
    if report is None:
        return None, path
    sample = report.get("selection_sample")
    if not report.get("complete", True) and isinstance(sample, dict) and sample.get("complete"):
        return {"performed_by": report.get("performed_by"), **sample}, path
    return report, path


def _noise_rows(report: Mapping[str, Any]) -> tuple[str, list[tuple[str, str, Mapping[str, Any]]]]:
    by_domain = report.get("per_domain_field")
    groups = by_domain if isinstance(by_domain, Mapping) else report.get("per_vertical_field")
    kind = "domain" if isinstance(by_domain, Mapping) else "vertical"
    rows = [
        (str(g), str(name), v)
        for g, fields in sorted((groups or {}).items())
        for name, v in (fields or {}).items()
        if isinstance(v, Mapping)
    ]
    return kind, rows


def low_agreement(report: Mapping[str, Any]) -> list[str]:
    """``domain field rate (n pages)`` for every field agreeing on fewer than NOISE_LOW."""
    _, rows = _noise_rows(report)
    return [
        f"`{g}` {name} {_fmt(v.get('rate'), 2)} ({_fmt(v.get('labeled'), 0)} pages)"
        for g, name, v in rows
        if (rate := _num(v.get("rate"))) is not None and rate < NOISE_LOW
    ]


def noise_check_lines(datasets_dir: Path, target: Target) -> list[str]:
    report, path = _noise_report(datasets_dir)
    script = "uv run python eval/datasets/commoncrawl_silver/noise_check.py"
    if report is None:
        return [
            f"Label-noise check (JSON-LD labels against the visible text): {NOT_RUN_YET} "
            f"(`{script}`).",
            "",
        ]
    who = report.get("performed_by") or "the build agent (automated script), not a human"
    checked = report.get("domains_checked") or []
    source = (
        "a sample of the downloaded pages"
        if report.get("source", "labels") == "labels"
        else "the domain-selection sample"
    )
    missing = report.get("domains_missing") or []
    coverage = f", {_plural(len(checked), 'domain')}" if checked else ""
    if missing:
        coverage += f"; not yet checked: {', '.join(f'`{d}`' for d in missing)}"
    head = (
        f"Label-noise check: each JSON-LD label compared with the text the extractor sees, on "
        f"{_fmt(report.get('pages'), 0)} pages from {source}{coverage}; performed by {who} "
        f"({target.link(path)}, `{script}`)."
    )
    low = low_agreement(report)
    summary = (
        f"Fields agreeing on fewer than {NOISE_LOW:.0%} of the checked pages: {'; '.join(low)}."
        if low
        else f"Every field agrees on at least {NOISE_LOW:.0%} of the checked pages."
    )
    if not target.detail:
        pointer = (
            f"{len(low)} fields agree on fewer than {NOISE_LOW:.0%} of the checked pages; they are "
            "listed under [Limitations](#limitations)."
            if low
            else summary
        )
        return [f"{head} {pointer}", ""]
    kind, rows = _noise_rows(report)
    table = [
        [g, name, _fmt(v.get("labeled"), 0), _fmt(v.get("rate")), _fmt(v.get("item_rate"))]
        for g, name, v in rows
    ]
    header = (kind, "field", "labeled pages", "agreement", "item agreement (lists)")
    return [head, summary, "", *_table(header, table), ""]


def section_cc_silver(ctx: Context, target: Target) -> list[str]:
    return [
        *_accuracy_block("Common Crawl silver set", ctx.select("cc_silver"), target),
        *domains_table(ctx.datasets_dir, target),
        *noise_check_lines(ctx.datasets_dir, target),
    ]


def _livewebie_manifest(datasets_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    path = datasets_dir / "livewebie" / "manifest.json"
    return _read_json(path), path


def _manifest_lines(datasets_dir: Path, target: Target) -> list[str]:
    manifest, path = _livewebie_manifest(datasets_dir)
    if manifest is None:
        return [f"LiveWeb-IE snapshot: no manifest at `{_display(path)}`.", ""]
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
        site: str(info.get("excluded_reason"))
        for site, info in sorted((manifest.get("sites") or {}).items())
        if isinstance(info, dict) and info.get("excluded_reason")
    }
    if excluded and target.detail:
        lines += [f"- excluded `{site}`: {_short(reason)}" for site, reason in excluded.items()]
        lines.append("")
    elif excluded:  # README: one line, each reason up to its first qualifier
        reasons = [f"`{s}` ({re.split(r'[;(]', r)[0].strip()})" for s, r in excluded.items()]
        lines += [f"Excluded: {'; '.join(reasons)}.", ""]
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


def published_cost(r: Result) -> float | None:
    """LLM cost behind a run's numbers: its tokens (cached calls included, with their original
    tokens) at standard list prices, i.e. what the run costs from an empty cache."""
    cost = r.summary.get("cost") or {}
    usd, pages = _num(cost.get("usd_per_1k_pages")), _num(cost.get("pages"))
    if usd is None or not pages:
        return None
    return usd * pages / 1000


def section_spend(ctx: Context, target: Target) -> list[str]:
    """LLM cost of the published results per evaluation, and what the runs kept under
    ``eval/results`` were billed according to ditto's ledger. The README table lists only
    evaluations that call an LLM; the rest are named in one line each."""
    rows: list[list[str]] = []
    pending: list[str] = []
    no_llm: list[str] = []
    total_cost = total_billed = 0.0
    n_runs = 0
    for e in sorted(ctx.entries.values(), key=lambda e: e.config.name):
        budget = _usd(e.config.budget_usd) if e.config.uses_llm else "no LLM"
        if e.result is None:
            if target.detail:
                rows.append(
                    [f"`{e.config.name}`", e.config.label, NOT_RUN_YET, DASH, DASH, budget, DASH]
                )
            else:
                pending.append(f"`{e.config.name}`")
            continue
        cost = published_cost(e.result)
        billed = e.result.billed_all_runs
        total_cost += cost or 0.0
        total_billed += billed
        n_runs += e.result.n_runs
        if not target.detail and not e.config.uses_llm and not cost and not billed:
            no_llm.append(target.link(e.result.path, e.config.name))
            continue
        runs = f" ({e.result.n_runs} runs)" if e.result.n_runs > 1 else ""
        rows.append(
            [
                f"`{e.config.name}`",
                e.config.label,
                e.status_text(),
                _usd(cost),
                _usd(billed) + runs,
                budget,
                target.link(e.result.path, e.result.path.name),
            ]
        )
    header = (
        "evaluation",
        "method",
        "status",
        "LLM cost of the published run",
        "billed in kept runs (ledger)",
        "budget cap",
        "latest run",
    )
    lines = [*_table(header, rows), ""] if rows else []
    if no_llm:
        lines += [f"Run without any LLM call: {', '.join(no_llm)}.", ""]
    if pending:
        lines += [f"Not run yet: {', '.join(pending)}.", ""]
    if not any(e.result is not None for e in ctx.entries.values()):
        return [*lines, f"Spend: {NOT_RUN_YET} (no evaluation has results)."]
    return [
        *lines,
        f"Total LLM cost of the published results: {_usd(total_cost)}, every token behind them "
        "(cached calls included) at the standard list prices of `config/prices.yaml`, i.e. what "
        f"rerunning them from an empty cache costs. Billed in the {_plural(n_runs, 'run')} kept "
        f"under `eval/results/`: {_usd(total_billed)} (`meta.json` `spend_usd`, ditto's ledger; "
        "a rerun answered from the cache bills $0, and runs kept for the audit trail count "
        "too). Rewrite with `" + REPORT_CMD + "`.",
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


def _judgment_file(e: Entry) -> str | None:
    if e.config.kind != "ditto" or e.config.mode != "unprompted":
        return None
    return f"eval/judgments/{e.config.dataset}_{e.config.split}_schema_precision.json"


def section_reproduce(ctx: Context, target: Target) -> list[str]:
    """One command per evaluation config (``docs/benchmarks.md``)."""
    entries = sorted(ctx.entries.values(), key=_dataset_key)
    if not entries:
        return ["No evaluation config."]
    rows = []
    for e in entries:
        c = e.config
        what = c.label if c.until == "extract" or c.kind == "baseline" else f"ditto to `{c.until}`"
        sample = f"{c.pages_per_site} pages / site" if c.pages_per_site else "all pages"
        rows.append(
            [
                f"`{c.name}`",
                c.dataset_label,
                what,
                f"{c.n_listed_sites} listed" if c.n_listed_sites else "all",
                sample if c.uses_llm else DASH,
                _usd(c.budget_usd) if c.uses_llm else "no LLM",
                f"`{EVAL_CMD} {c.path}`",
            ]
        )
    header = ("config", "dataset", "method", "sites", "page sample", "budget cap", "command")
    lines = [*_table(header, rows), ""]
    for e in entries:
        judged = _judgment_file(e)
        if judged and Path(judged).is_file():
            lines += [
                f"`{e.config.name}` also regenerates the schema precision from the build agent's "
                f"per-field verdicts in {target.link(judged)} and lists proposed fields without "
                "a verdict (`unjudged`) and verdicts for fields the run did not propose "
                "(`stale`).",
                "",
            ]
    return [*lines, f"Then rewrite every table and chart with `{REPORT_CMD}`."]


# -- limitations (README) ----------------------------------------------------------------------


def _dataset_key(e: Entry) -> tuple[int, int, tuple[int, str, str]]:
    order = ("swde", "cc_silver", "livewebie", "synthetic")
    rank = order.index(e.config.dataset) if e.config.dataset in order else len(order)
    return (
        rank,
        ("dev", "test").index(e.config.split) if e.config.split in ("dev", "test") else 2,
        _method_key(e),
    )


def _sample_limits(ctx: Context) -> list[str]:
    sampled = [
        e
        for e in sorted(ctx.entries.values(), key=_dataset_key)
        if e.config.baseline == "naive_llm" and e.config.pages_per_site
    ]
    if not sampled:
        return []
    parts = ", ".join(f"{e.config.pages_per_site} on the {e.config.dataset_label}" for e in sampled)
    return [
        f"- **The per-page baseline ran on page samples**: seeded samples of {parts} (pages per "
        f"site) instead of the planned {PLANNED_SAMPLE}, to fit the LLM budget. Its cost per "
        "1,000 pages is extrapolated from the sample, and ditto is compared with it on the same "
        "sampled pages."
    ]


def _coverage_limits(ctx: Context) -> list[str]:
    lines: list[str] = []
    for e in sorted(ctx.entries.values(), key=_dataset_key):
        if e.config.kind != "baseline":
            continue
        ditto = partner(e, ctx.entries.values())
        total = ditto.n_sites if ditto else None
        name = f"**{e.config.label}** on the {e.config.dataset_label}"
        if not e.ok:
            if e.result is not None:
                lines.append(f"- {name}: {e.status_text()}.")
            continue
        n = e.n_ok
        if n is None or total is None or n >= total:
            continue
        sample = (
            f", on a seeded {e.config.pages_per_site}-page sample per site"
            if e.config.pages_per_site
            else ""
        )
        reason = COVERAGE_REASONS.get(e.config.baseline or "", "budget")
        lines.append(
            f"- {name} covers {n} of {total} sites{sample} ({reason}); the tables compare it "
            "with ditto on the same sites and pages."
        )
    return lines


def _cc_limits(ctx: Context) -> list[str]:
    doc, _ = _cc_domains(ctx.datasets_dir)
    cap = _num(_get(doc, "download", "cap_per_domain"))
    selected = [d for d in doc.get("selected") or [] if isinstance(d, Mapping)]
    if not cap or not selected:
        return []
    above = [
        f"`{d.get('domain')}` {_fmt(planned, 0)}"
        for d in selected
        if (planned := _num(_get(d, "captures", "planned"))) and planned > cap
    ]
    kept = f" ({', '.join(above)}: pages fetched before the cap was lowered are kept)"
    planned_total = sum(_num(_get(d, "captures", "planned")) or 0 for d in selected)
    downloaded = sum(_num(_get(d, "captures", "downloaded")) or 0 for d in selected)
    progress = (
        f" When this was generated, {_fmt(downloaded, 0)} of the {_fmt(planned_total, 0)} planned "
        "pages had been downloaded."
        if planned_total and downloaded < planned_total
        else ""
    )
    return [
        f"- **Common Crawl capped at {_fmt(cap, 0)} pages per domain**{kept if above else ''} "
        f"instead of the planned {PLANNED_CC_CAP:,}: Common Crawl's CloudFront firewall blocked "
        "this machine's IP when it fetched at 4 concurrent requests, and at the polite rate of "
        f"one request per second {PLANNED_CC_CAP:,} pages per domain did not fit the time "
        f"available.{progress}"
    ]


def _noise_limits(ctx: Context) -> list[str]:
    report, _ = _noise_report(ctx.datasets_dir)
    if report is None:
        return []
    low = low_agreement(report)
    found = (
        f"labels that agree with the visible text on fewer than {NOISE_LOW:.0%} of the checked "
        f"pages: {'; '.join(low)}"
        if low
        else f"every field agreeing on at least {NOISE_LOW:.0%} of the checked pages"
    )
    return [
        "- **Common Crawl labels are silver**: they come from each site's own JSON-LD, not from "
        f"people. The label-noise check (by the build agent, not a human) found {found}; a miss "
        "on such a field is often a label problem, not an extraction error."
    ]


def _livewebie_limits(ctx: Context) -> list[str]:
    manifest, _ = _livewebie_manifest(ctx.datasets_dir)
    if manifest is None:
        return []
    sites = {k: v for k, v in (manifest.get("sites") or {}).items() if isinstance(v, Mapping)}
    total = _num(_get(manifest, "counts", "websites")) or len(sites)
    usable = sum(1 for v in sites.values() if not v.get("excluded_reason"))
    if manifest.get("status") == "skipped":
        return [f"- **LiveWeb-IE skipped**: {manifest.get('reason') or 'no reason recorded'}."]
    return [
        f"- **LiveWeb-IE on {usable} of {total:.0f} sites** (exclusion reasons in the LiveWeb-IE "
        "section), scored on a frozen snapshot with field definitions instead of the live sites "
        "and free-form queries, so its numbers are not comparable with the paper's. The release "
        "is licensed for non-commercial research only; the snapshot stays out of the repository."
    ]


def _synthetic_limits(ctx: Context) -> list[str]:
    e = ctx.entries.get(CURVES_CONFIG)
    if e is None or not e.ok or e.result is None:
        return []
    points = [
        (param, p)
        for param, pts in _curves(e.result).items()
        for p in pts
        if p.get("status") == "ok"
    ]
    aris = [a for _, p in points if (a := _num(p.get("ari"))) is not None]
    if not aris:
        return []
    merged = [
        f"{param} = {_value_label(p.get('value'))}"
        for param, p in points
        if (_num(p.get("over_merge")) or 0) > 0
    ]
    where = f"; templates merge only at {', '.join(merged)}" if merged else ""
    return [
        "- **Synthetic templates are separable by construction** (the generator re-rolls "
        "templates until every pair differs clearly, except the designated near-duplicate "
        f"pairs), so the clustering curves are near-perfect: minimum ARI {_fmt(min(aris))} over "
        f"{len(aris)} points{where}. They show scale and robustness to markup noise, not "
        "accuracy on real sites."
    ]


def _label_problem_limits(ctx: Context) -> list[str]:
    path = ctx.analysis_dir / "swde_dev_misses.json"
    data = _read_json(path)
    ditto = _get(data, "per_method", "ditto") if data else None
    estimate = _get(ditto, "population_estimate")
    if not isinstance(estimate, Mapping):
        return []
    total = sum(_num(v) or 0 for v in estimate.values())
    if not total:
        return []
    share = {k: (_num(v) or 0) / total for k, v in estimate.items()}
    label = share.get("label_problem", 0.0)
    verb = "dominate" if label > 0.5 else "account for part of"
    sampled = _num(_get(ditto, "sample", "classified"))
    return [
        f"- **SWDE label problems {verb} ditto's remaining dev misses**: of its "
        f"{_fmt(_get(ditto, 'norm_misses'), 0)} misses under `norm` on the dev slice, an "
        f"estimated {_pct(label)} are label problems (truncated, over-long or missing labels), "
        f"{_pct(share.get('real_error'))} real errors and {_pct(share.get('formatting'))} "
        f"formatting differences, from a stratified sample of {_fmt(sampled, 0)} misses "
        "classified by the build agent, not a human (`eval/analysis/swde_dev_misses.md`)."
    ]


def _speed_limits(ctx: Context) -> list[str]:
    measured = []
    cpus: set[str] = set()
    for e in sorted(ctx.entries.values(), key=lambda e: e.config.name):
        if e.config.kind != "ditto" or e.config.until != "extract" or not e.ok:
            continue
        assert e.result is not None
        v = _num(_get(e.result.summary, "speed_pages_per_sec_per_core", "extract"))
        if v is not None:
            measured.append((e.config.name, v))
            cpus.add(str(_get(e.result.meta, "machine", "cpu_count") or DASH))
    if not measured:
        return []
    hits = sum(v >= SPEED_TARGET for _, v in measured)
    verdict = "meets" if hits == len(measured) else "partly meets" if hits else "falls short of"
    runs = ", ".join(f"{v:,.0f} on `{name}`" for name, v in measured)
    return [
        f"- **Extraction speed in the evaluation runs** {verdict} the target of "
        f"{SPEED_TARGET:,.0f} pages per second per core: {runs} pages/s/core (parse included, "
        "mean over sites; per-core figures divide by the number of worker processes, on a "
        f"machine with {', '.join(sorted(cpus))} logical CPUs)."
    ]


def section_limitations(ctx: Context, target: Target) -> list[str]:
    lines = [
        *_sample_limits(ctx),
        *_coverage_limits(ctx),
        *_cc_limits(ctx),
        *_noise_limits(ctx),
        *_livewebie_limits(ctx),
        *_synthetic_limits(ctx),
        *_label_problem_limits(ctx),
        *_speed_limits(ctx),
    ]
    return lines or [f"Data-driven limitations: {NOT_RUN_YET} (no evaluation has results)."]


SECTION_RENDERERS = {
    "headline": section_headline,
    "swde": section_swde,
    "cc_silver": section_cc_silver,
    "livewebie": section_livewebie,
    "clustering": section_clustering,
    "union": section_union,
    "spend": section_spend,
    "models": section_models,
    "reproduce": section_reproduce,
    "limitations": section_limitations,
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


# -- LLM spend per build phase (BUILD_REPORT.md) -----------------------------------------------------


def tag_dates(pattern: str = "v*") -> list[tuple[str, datetime]]:
    """Release tags with the commit date of the commit each points at, oldest first."""
    names = subprocess.run(
        ["git", "tag", "-l", pattern], capture_output=True, text=True, check=True
    ).stdout.split()
    dated = []
    for name in names:
        stamp = subprocess.run(
            ["git", "log", "-1", "--format=%cI", name], capture_output=True, text=True, check=True
        ).stdout.strip()
        dated.append((name, datetime.fromisoformat(stamp)))
    return sorted(dated, key=lambda t: t[1])


def spend_by_phase(
    rows: Iterable[Mapping[str, Any]], tags: Sequence[tuple[str, datetime]]
) -> list[dict[str, Any]]:
    """Ledger rows summed per phase: before the first tag, between consecutive tags, after the
    last (``now``). Each phase: name, uncached calls, billed USD (``cost_usd``), list-price USD
    (``list_cost_usd``) and billed USD per role."""
    bounds: list[tuple[str, datetime | None, datetime | None]] = []
    previous: tuple[str, datetime] | None = None
    for name, when in tags:
        bounds.append(
            (
                f"{previous[0]}–{name}" if previous else f"before {name}",
                previous and previous[1],
                when,
            )
        )
        previous = (name, when)
    bounds.append((f"{previous[0]}–now" if previous else "all", previous and previous[1], None))
    phases = [
        {"phase": n, "from": a, "to": b, "calls": 0, "billed": 0.0, "list": 0.0, "roles": {}}
        for n, a, b in bounds
    ]
    for row in rows:
        when = datetime.fromisoformat(str(row.get("ts")))
        for p in phases:
            if (p["from"] is None or when >= p["from"]) and (p["to"] is None or when < p["to"]):
                billed = _num(row.get("cost_usd")) or 0.0
                p["calls"] += 0 if row.get("cached") else 1
                p["billed"] += billed
                p["list"] += _num(row.get("list_cost_usd")) or 0.0
                role = str(row.get("role") or "unknown")
                p["roles"][role] = p["roles"].get(role, 0.0) + billed
                break
    return phases


def spend_by_phase_table(phases: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = [
        [
            p["phase"],
            p["from"].isoformat() if p["from"] else DASH,
            p["to"].isoformat() if p["to"] else DASH,
            _fmt(p["calls"], 0),
            _usd(p["billed"]),
            _usd(p["list"]),
            ", ".join(f"{r} {_usd(v)}" for r, v in sorted(p["roles"].items()) if v) or DASH,
        ]
        for p in phases
    ]
    rows.append(
        [
            "total",
            DASH,
            DASH,
            _fmt(sum(p["calls"] for p in phases), 0),
            _usd(sum(p["billed"] for p in phases)),
            _usd(sum(p["list"] for p in phases)),
            DASH,
        ]
    )
    header = ("phase", "from", "to", "uncached calls", "billed (ledger)", "list price", "by role")
    return [
        *_table(header, rows),
        "",
        "Phases run between the commit dates of the tagged commits. Ledger figures "
        "(`cost_usd`); a billing check found the ledger undercounting the provider's charges by "
        "about 23% (DECISIONS.md).",
    ]


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# -- main ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--configs-dir", type=Path, default=Path("eval/configs"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("eval/datasets"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("eval/analysis"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--benchmarks", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if a table is out of date"
    )
    parser.add_argument(
        "--spend-by-phase",
        action="store_true",
        help="print the ledger's LLM spend per build phase (between release tags) and exit",
    )
    parser.add_argument("--ledger", type=Path, default=Path(".cache/ledger.jsonl"))
    args = parser.parse_args(argv)

    if args.spend_by_phase:
        if not args.ledger.is_file():
            print(f"{args.ledger}: not found")
            return 1
        print(
            "\n".join(spend_by_phase_table(spend_by_phase(_ledger_rows(args.ledger), tag_dates())))
        )
        return 0

    entries = load_entries(args.configs_dir, args.results_dir)
    results = {name: e.result for name, e in entries.items() if e.result is not None}
    ctx = Context(entries=entries, datasets_dir=args.datasets_dir, analysis_dir=args.analysis_dir)
    points = headline_points(entries)
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
