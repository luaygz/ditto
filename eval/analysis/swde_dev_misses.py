"""Offline rescoring and miss analysis on the SWDE development slice (saved predictions, no LLM).

Reads the saved predictions of one ``swde_dev_ditto`` and one ``swde_dev_naive`` evaluation run,
scores them exactly as ``ditto eval run`` does (``ditto.evaluation.metrics.score_site``) and:

1. checks that the rescored ``norm``/``raw`` numbers equal both runs' committed ``results.json``;
2. lists every miss under the ``norm`` column (method, site, field, page, predicted, label) in
   ``data/analysis/swde_dev_misses.jsonl`` (gitignored: it holds page values);
3. draws a stratified sample of misses (per method x vertical, round-robin over site fields) for
   manual classification into ``data/analysis/swde_dev_sample.jsonl``;
4. merges the manual classification ``eval/analysis/swde_dev_misses_classified.jsonl`` into
   ``eval/analysis/swde_dev_misses.json``: counts per category per method, a population estimate
   (each sampled miss weighted by its site field's misses / samples), which misses the ``canon``
   column credits, and short example value pairs;
5. scans the earlier dev-slice runs on disk for the misses ``canon`` credits (supporting
   evidence for the rules, also in ``swde_dev_misses.json``);
6. writes the per-vertical F1 of every match column (``norm`` beside ``canon``) to
   ``eval/analysis/swde_dev_rescore.json`` and prints it as a Markdown table.

Only the dev slice (``eval/datasets/swde/split.json`` "dev") is read: the ``canon`` rules were
derived from these misses and frozen before any test set was rescored.

    uv run python eval/analysis/swde_dev_misses.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ditto.evaluation.datasets import swde
from ditto.evaluation.datasets.base import EvalSite, Selection, is_empty
from ditto.evaluation.metrics import (
    METRIC_VERSION,
    MODES,
    ROW_META_KEYS,
    MatchMode,
    SiteScore,
    aggregate,
    sample_keys,
    score_site,
    score_value,
    to_text,
)
from ditto.evaluation.runner import _page_urls, _seen_key, _url_path, read_output_rows
from ditto.run import RunDir

ROOT = Path(__file__).resolve().parents[2]
DITTO_DIR = ROOT / "data" / "eval" / "swde_dev_ditto"
NAIVE_DIR = ROOT / "data" / "eval" / "swde_dev_naive"
DITTO_RUN = "20261027T102344Z"
NAIVE_RUN = "20261027T085024Z"
SAMPLE_SEED = 1
PER_STRATUM = 12
"""Sampled misses per (method, vertical); every miss of a stratum with fewer."""
SHORT = 80
"""Committed values are cut to this many characters."""

ANALYSIS_DIR = ROOT / "eval" / "analysis"
DATA_DIR = ROOT / "data" / "analysis"
CLASSIFIED = ANALYSIS_DIR / "swde_dev_misses_classified.jsonl"
SUMMARY = ANALYSIS_DIR / "swde_dev_misses.json"
RESCORE = ANALYSIS_DIR / "swde_dev_rescore.json"

CATEGORIES = ("formatting", "real_error", "label_problem")
METHODS = ("ditto", "naive")


@dataclass
class MethodRun:
    """Predictions and scored labels of one method on one site."""

    method: str
    site: EvalSite
    predictions: dict[str, dict[str, Any]]
    labels: dict[str, dict[str, Any]]
    excluded: set[str]


def load_sites() -> list[EvalSite]:
    with tempfile.TemporaryDirectory() as tmp:
        sites = swde.load_sites(Selection(split="dev"), Path(tmp))
    return sorted(sites, key=lambda s: s.name)


def ditto_run(site: EvalSite, run_root: Path) -> tuple[MethodRun, MethodRun]:
    """ditto's predictions as the runner scores them (``_Runner._score_ditto``): every labeled
    page the LLM did not see, and the seeded 50-page sample the baselines ran on."""
    from ditto.specgen.models import Spec

    run = RunDir(run_root)
    urls = _page_urls(run)
    pid_key = {pid: k for pid, url in urls.items() if (k := site.key_for_url(url))}
    path_key = {_url_path(urls[pid]): k for pid, k in pid_key.items()}
    predictions: dict[str, dict[str, Any]] = {}
    for row in read_output_rows(run):
        key = site.key_for_url(str(row.get("url", "")))
        if key is None and row.get("page_id") is not None:
            key = pid_key.get(int(row["page_id"]))
        if key is not None:
            predictions[key] = {k: v for k, v in row.items() if k not in ROW_META_KEYS}
    excluded: set[str] = set()
    for path in sorted(run.specs_dir.glob("*.json")):
        gb = Spec.load(path).generated_by
        excluded |= {k for s in gb.pages_seen if (k := _seen_key(s, site, pid_key, path_key))}
    sample = set(sample_keys({k for k in pid_key.values() if k in site.labels}, 50, 1))
    sampled = {k: v for k, v in site.labels.items() if k in sample}
    return (
        MethodRun("ditto", site, predictions, site.labels, excluded),
        MethodRun("ditto", site, predictions, sampled, excluded),
    )


def naive_run(site: EvalSite, baseline_dir: Path) -> MethodRun:
    """The naive baseline's predictions on its seeded page sample (``_Runner.run_baseline``)."""
    predictions: dict[str, dict[str, Any]] = {}
    for line in (baseline_dir / "naive_llm.predictions.jsonl").read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[str(row["key"])] = {
                k: v for k, v in row.items() if k not in ROW_META_KEYS and k != "key"
            }
    summary = json.loads((baseline_dir / "naive_llm.summary.json").read_text("utf-8"))
    urls = _page_urls(RunDir(baseline_dir.parent / "run"))
    labeled = sorted(k for url in urls.values() if (k := site.key_for_url(url)) in site.labels)
    keys = sample_keys(labeled, int(summary["max_pages"]), int(summary["seed"]))
    scored = set(keys)
    if int(summary["pages_run"]) < len(keys):
        scored &= set(predictions)
    return MethodRun("naive", site, predictions, {k: site.labels[k] for k in scored}, set())


def score(mr: MethodRun) -> SiteScore:
    return score_site(
        mr.site.name,
        mr.site.vertical,
        mr.predictions,
        mr.labels,
        mr.site.field_kinds,
        fields=mr.site.fields(),
        exclude=mr.excluded,
    )


# -- misses --------------------------------------------------------------------------------------


def _value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        items = [to_text(v) for v in value if not is_empty(v)]
        return items if len(items) > 1 else (items[0] if items else None)
    return None if is_empty(value) else to_text(value)


def misses(mr: MethodRun, mode: MatchMode = "norm") -> list[dict[str, Any]]:
    """Every scored (page, field) unit with a false positive or a miss under ``mode``."""
    out: list[dict[str, Any]] = []
    fields = mr.site.fields()
    for key in sorted(mr.labels):
        if key in mr.excluded:
            continue
        label_row = mr.labels[key]
        pred_row = mr.predictions.get(key, {})
        for name in fields:
            if name not in label_row:
                continue
            kind = mr.site.kind(name)
            c = score_value(kind, pred_row.get(name), label_row.get(name), mode)
            if not (c.fp or c.fn):
                continue
            out.append(
                {
                    "method": mr.method,
                    "site": mr.site.name,
                    "vertical": mr.site.vertical,
                    "field": name,
                    "page": key,
                    "kind": kind,
                    "outcome": "wrong" if c.fp and c.fn else ("missing" if c.fn else "extra"),
                    "predicted": _value(pred_row.get(name)),
                    "label": _value(label_row.get(name)),
                }
            )
    return out


def miss_id(m: Mapping[str, Any]) -> str:
    return f"{m['method']}|{m['site']}|{m['field']}|{m['page']}"


def _group(m: Mapping[str, Any]) -> str:
    return f"{m['method']}|{m['site']}|{m['field']}"


def stratified_sample(rows: Sequence[Mapping[str, Any]], per_stratum: int, seed: int) -> list[str]:
    """Miss ids: per (method, vertical) up to ``per_stratum`` misses, round-robin over the
    stratum's site fields, each field's misses in a ``seed``-shuffled order."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for m in rows:
        strata[(m["method"], m["vertical"])][_group(m)].append(miss_id(m))
    chosen: list[str] = []
    for stratum in sorted(strata):
        queues = [sorted(ids) for _, ids in sorted(strata[stratum].items())]
        for ids in queues:
            rng.shuffle(ids)
        picked: list[str] = []
        while len(picked) < per_stratum and any(queues):
            for ids in queues:
                if ids and len(picked) < per_stratum:
                    picked.append(ids.pop())
        chosen.extend(picked)
    return chosen


# -- summary -------------------------------------------------------------------------------------


def short(value: Any) -> Any:
    if value is None:
        return None
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)
    return text if len(text) <= SHORT else text[: SHORT - 1] + "\u2026"


def summarize(
    rows: Sequence[Mapping[str, Any]],
    classified: Sequence[Mapping[str, Any]],
    canon_credited: set[str],
) -> dict[str, Any]:
    """Counts per category per method (sample and population estimate) and example pairs."""
    group_size = Counter(_group(m) for m in rows)
    group_sampled = Counter(_group(c) for c in classified)
    per_method: dict[str, Any] = {}
    for method in METHODS:
        mine_all = [m for m in rows if m["method"] == method]
        mine = [c for c in classified if c["method"] == method]
        estimate: dict[str, float] = defaultdict(float)
        for c in mine:
            estimate[c["category"]] += group_size[_group(c)] / group_sampled[_group(c)]
        credited = [m for m in mine_all if miss_id(m) in canon_credited]
        per_method[method] = {
            "norm_misses": len(mine_all),
            "norm_misses_by_outcome": dict(Counter(m["outcome"] for m in mine_all)),
            "canon_credits": len(credited),
            "canon_credits_by_site_field": dict(
                sorted(Counter(f"{m['site']}.{m['field']}" for m in credited).items())
            ),
            "sample": {
                "classified": len(mine),
                "categories": {k: sum(c["category"] == k for c in mine) for k in CATEGORIES},
                "kinds": {
                    k: dict(Counter(c["kind"] for c in mine if c["category"] == k).most_common())
                    for k in CATEGORIES
                },
                "canon_credits_by_category": {
                    k: sum(c["category"] == k and c["id"] in canon_credited for c in mine)
                    for k in CATEGORIES
                },
            },
            "population_estimate": {k: round(estimate[k]) for k in CATEGORIES},
        }
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for c in classified:
        bucket = f"{c['category']}:{c['kind']}"
        where = (bucket, c["site"], c["field"])
        if len(examples[bucket]) < 3 and where not in seen:
            seen.add(where)
            examples[bucket].append(
                {
                    "method": c["method"],
                    "site": c["site"],
                    "field": c["field"],
                    "predicted": short(c["predicted"]),
                    "label": short(c["label"]),
                    "canon_match": c["id"] in canon_credited,
                }
            )
    return {"per_method": per_method, "examples": dict(sorted(examples.items()))}


def earlier_runs(sites: Sequence[EvalSite], main: Mapping[str, str]) -> dict[str, Any]:
    """Misses ``canon`` credits in the other dev-slice runs on disk, per site field."""
    out: dict[str, Any] = {}
    for method, parent in (("ditto", DITTO_DIR), ("naive", NAIVE_DIR)):
        for run_dir in sorted(p for p in parent.glob("2*") if p.name != main[method]):
            credited: dict[str, Any] = {}
            n_misses = 0
            for site in sites:
                if method == "ditto":
                    if not (run_dir / site.name / "run" / "out").is_dir():
                        continue
                    mr = ditto_run(site, run_dir / site.name / "run")[0]
                else:
                    if not (run_dir / site.name / "baseline" / "naive_llm.summary.json").exists():
                        continue
                    mr = naive_run(site, run_dir / site.name / "baseline")
                norm = misses(mr)
                n_misses += len(norm)
                still = {miss_id(m) for m in misses(mr, "canon")}
                for m in norm:
                    if miss_id(m) in still:
                        continue
                    entry = credited.setdefault(
                        f"{m['site']}.{m['field']}",
                        {"n": 0, "example": [short(m["predicted"]), short(m["label"])]},
                    )
                    entry["n"] += 1
            out[f"{method}/{run_dir.name}"] = {"norm_misses": n_misses, "canon_credits": credited}
    return out


# -- rescoring table -----------------------------------------------------------------------------


def f1_table(aggs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``{run: {vertical | "overall": {mode: f1}}}``."""
    table: dict[str, Any] = {}
    for name, agg in aggs.items():
        rows = {v: {m: e[m]["f1"] for m in MODES} for v, e in agg["per_vertical"].items()}
        rows["overall"] = {m: agg["overall"][m]["f1"] for m in MODES}
        table[name] = rows
    return table


def markdown_table(table: Mapping[str, Mapping[str, Mapping[str, float | None]]]) -> str:
    lines = [
        "| run | vertical | F1 norm | F1 canon | change | F1 raw |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows in table.items():
        for vertical, f1 in rows.items():
            norm, canon, raw = f1["norm"], f1["canon"], f1["raw"]
            cells = [f"{v:.3f}" if v is not None else "" for v in (norm, canon)]
            delta = f"{canon - norm:+.3f}" if norm is not None and canon is not None else ""
            raw_cell = f"{raw:.3f}" if raw is not None else ""
            lines.append(
                f"| {name} | {vertical} | {cells[0]} | {cells[1]} | {delta} | {raw_cell} |"
            )
    return "\n".join(lines)


def check_committed(name: str, results: Path, agg: Mapping[str, Any], key: str) -> list[str]:
    committed = json.loads(results.read_text("utf-8"))["summary"][key]
    problems: list[str] = []
    for mode in ("norm", "raw"):
        for metric in ("precision", "recall", "f1"):
            want, got = committed["overall"][mode][metric], agg["overall"][mode][metric]
            if abs(want - got) > 1e-9:
                problems.append(f"{name} {key} {mode} {metric}: committed {want}, rescored {got}")
    return problems


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rescore and analyze SWDE dev-slice misses offline.")
    ap.add_argument("--ditto-run", default=DITTO_RUN, help="swde_dev_ditto timestamp")
    ap.add_argument("--naive-run", default=NAIVE_RUN, help="swde_dev_naive timestamp")
    args = ap.parse_args(argv)
    main_runs = {"ditto": args.ditto_run, "naive": args.naive_run}

    sites = load_sites()
    runs: dict[str, list[MethodRun]] = {"ditto": [], "ditto_sample": [], "naive": []}
    for site in sites:
        full, sampled = ditto_run(site, DITTO_DIR / args.ditto_run / site.name / "run")
        runs["ditto"].append(full)
        runs["ditto_sample"].append(sampled)
        runs["naive"].append(naive_run(site, NAIVE_DIR / args.naive_run / site.name / "baseline"))
    aggs = {name: aggregate([score(mr) for mr in mrs]) for name, mrs in runs.items()}
    ditto_results = ROOT / "eval/results/swde_dev_ditto" / args.ditto_run / "results.json"
    naive_results = ROOT / "eval/results/swde_dev_naive" / args.naive_run / "results.json"
    problems = check_committed("ditto", ditto_results, aggs["ditto"], "accuracy")
    problems += check_committed("ditto", ditto_results, aggs["ditto_sample"], "accuracy_sample")
    problems += check_committed("naive", naive_results, aggs["naive"], "accuracy")
    for p in problems:
        print(f"MISMATCH {p}", file=sys.stderr)

    rows = [m for method in METHODS for mr in runs[method] for m in misses(mr)]
    canon_rows = [m for method in METHODS for mr in runs[method] for m in misses(mr, "canon")]
    norm_ids, canon_ids = {miss_id(m) for m in rows}, {miss_id(m) for m in canon_rows}
    if canon_ids - norm_ids:
        problems.append(f"{len(canon_ids - norm_ids)} units miss under canon but not under norm")
    by_id = {miss_id(m): m for m in rows}
    sample = stratified_sample(rows, PER_STRATUM, SAMPLE_SEED)
    _write_jsonl(DATA_DIR / "swde_dev_misses.jsonl", rows)
    _write_jsonl(DATA_DIR / "swde_dev_sample.jsonl", [{"id": i, **by_id[i]} for i in sample])

    classified: list[dict[str, Any]] = []
    if CLASSIFIED.exists():
        for line in CLASSIFIED.read_text("utf-8").splitlines():
            if line.strip():
                classified.append({**by_id[json.loads(line)["id"]], **json.loads(line)})
    unclassified = sorted(set(sample) - {c["id"] for c in classified})
    if unclassified:
        problems.append(f"{len(unclassified)} sampled misses are not classified")

    _write_json(
        SUMMARY,
        {
            "about": (
                "Misses under the norm column on the SWDE dev slice (ditto: every labeled page the "
                "LLM did not see; naive: its seeded 50-page sample per site). A stratified sample "
                "was classified by the build agent (not a human) as formatting false negative, "
                "real error or label problem; canon_* fields say which misses the canon column "
                "credits."
            ),
            "metric_version": METRIC_VERSION,
            "runs": main_runs,
            "sample": {"per_stratum": PER_STRATUM, "seed": SAMPLE_SEED, "size": len(sample)},
            **summarize(rows, classified, norm_ids - canon_ids),
            "earlier_dev_runs": earlier_runs(sites, main_runs),
        },
    )
    table = f1_table(aggs)
    _write_json(RESCORE, {"metric_version": METRIC_VERSION, "runs": main_runs, "f1": table})
    print(markdown_table(table))
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
