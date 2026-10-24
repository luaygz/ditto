"""``ditto eval run eval/configs/<name>.yaml``: one evaluation into ``eval/results/<name>/<ts>/``.

The runner loads a dataset through its adapter, runs a method on every selected site (the ditto
pipeline through :mod:`ditto.pipeline`, or a baseline from ``eval/baselines/<name>.py``) inside
a scratch work directory under ``data/eval/`` (run directories hold page content and are never
committed), scores the output against the labels and writes::

    results.json   everything: per-site scores, aggregates, cost, speed, clustering, curves
    results.csv    one row per vertical plus ``overall``
    per_site.csv   one row per site
    meta.json      config, pinned and used model ids, prompt versions, git commit, dataset
                   manifest hashes, machine info, spend
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import urlsplit

import orjson
from lxml import html as lxml_html

from ditto.evaluation.baseline_api import (
    DEFAULT_BASELINES_DIR,
    BaselineOutput,
    NotRun,
    PageInput,
    RunBaseline,
    load_baseline,
)
from ditto.evaluation.clustering import clustering_metrics
from ditto.evaluation.config import DATASET_MODULES, EvalConfig
from ditto.evaluation.datasets.base import DatasetUnavailable, EvalSite, Selection
from ditto.evaluation.metrics import (
    MODES,
    ROW_META_KEYS,
    OptionalRecall,
    SiteScore,
    aggregate,
    align_field_names,
    optional_field_recall,
    rename_fields,
    sample_keys,
    score_site,
)
from ditto.llm.base import ROLES, BudgetExceeded
from ditto.parallel import cpu_count, memory_gb
from ditto.pipeline import RunOptions, continue_run, stage_analyze, stage_cluster, stage_ingest
from ditto.run import RunDir, read_json, write_json
from ditto.schema import FieldsSchema

# -- hooks (the real stages by default; tests substitute fakes) -----------------------------------


def _page_urls(run: RunDir) -> dict[int, str]:
    from ditto.ingest import PageStore

    return {rec.page_id: rec.url for rec in PageStore.open(run).records()}


def _read_html(run: RunDir, page_id: int) -> str:
    from ditto.ingest import read_html

    return read_html(run.root, page_id)


def _settings(models: Mapping[str, str], budget_usd: float | None) -> Any:
    """LLM settings from the environment with the config's pinned models and this evaluation's
    remaining budget as the per-run cap."""
    from ditto.llm import load_settings

    settings = load_settings().with_models(**models)
    if budget_usd is not None:
        settings = dataclasses.replace(settings, run_budget_usd=budget_usd)
    return settings


def _resolve_model(role: str, settings: Any) -> str:
    from ditto.llm import resolve_model
    from ditto.llm.settings import Settings

    if role not in ROLES or not isinstance(settings, Settings):
        raise ValueError(f"cannot resolve the model for role {role!r}")
    return resolve_model(role, settings)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    from ditto.llm import estimate_cost

    return estimate_cost(model, input_tokens, output_tokens)


def _clustering_metrics(true: Sequence[str], pred: Sequence[str | None]) -> dict[str, Any]:
    return dict(clustering_metrics(true, pred))


def _load_sites(dataset: str, selection: Selection, work_dir: Path) -> list[EvalSite]:
    sites: list[EvalSite] = importlib.import_module(DATASET_MODULES[dataset]).load_sites(
        selection, work_dir
    )
    return sites


@dataclass
class Hooks:
    """Everything the runner calls outside this package; replaced by fakes in tests."""

    ingest: Callable[..., RunDir] = stage_ingest
    cluster: Callable[..., Any] = stage_cluster
    analyze: Callable[..., Any] = stage_analyze
    continue_run: Callable[..., Any] = continue_run
    run_options: Callable[..., Any] = RunOptions
    page_urls: Callable[[RunDir], dict[int, str]] = _page_urls
    read_html: Callable[[RunDir, int], str] = _read_html
    settings: Callable[[Mapping[str, str], float | None], Any] = _settings
    resolve_model: Callable[[str, Any], str] = _resolve_model
    estimate_cost: Callable[[str, int, int], tuple[float, bool]] = _estimate_cost
    clustering_metrics: Callable[[Sequence[str], Sequence[str | None]], dict[str, Any]] = (
        _clustering_metrics
    )
    load_sites: Callable[[str, Selection, Path], list[EvalSite]] = _load_sites
    load_baseline: Callable[[str, Path], RunBaseline] = load_baseline


# -- per-site outcome ------------------------------------------------------------------------------


@dataclass
class SiteOutcome:
    site: EvalSite
    status: str = "ok"  # ok | failed | not_run | skipped
    reason: str | None = None
    n_pages: int = 0
    score: SiteScore | None = None
    sample_score: SiteScore | None = None
    optional: OptionalRecall | None = None
    clustering: dict[str, Any] | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    speed: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, dict[str, str]] | None = None
    models: set[str] = field(default_factory=set)
    prompt_versions: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    spend_usd: float = 0.0
    run_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "site": self.site.name,
            "vertical": self.site.vertical,
            "status": self.status,
            "reason": self.reason,
            "n_pages": self.n_pages,
            "n_labeled": len(self.site.labels),
            "accuracy": self.score.to_dict() if self.score else None,
            "accuracy_sample": self.sample_score.to_dict() if self.sample_score else None,
            "optional_field_recall": self.optional.to_dict() if self.optional else None,
            "clustering": self.clustering,
            "cost": self.cost,
            "speed": self.speed,
            "spend_usd": round(self.spend_usd, 6),
            "models": sorted(self.models),
            "prompt_versions": sorted(self.prompt_versions),
            "notes": self.notes,
            "site_meta": self.site.meta,
            "run_dir": str(self.run_dir) if self.run_dir else None,
        }
        if self.score and self.site.field_groups:
            out["groups"] = {
                mode: self.score.group_metrics(self.site.field_groups, mode) for mode in MODES
            }
        if self.alignment is not None:
            out["alignment"] = self.alignment
        return out


# -- reading run outputs ---------------------------------------------------------------------------


def read_usage(run: RunDir) -> list[dict[str, Any]]:
    if not run.usage_jsonl.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in run.usage_jsonl.read_bytes().splitlines():
        if line.strip():
            rows.append(orjson.loads(line))
    return rows


def read_output_rows(run: RunDir) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run.root / "out").glob("*/*.jsonl")):
        for line in path.read_bytes().splitlines():
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def cost_summary(
    usage: Sequence[Mapping[str, Any]],
    *,
    pages: int,
    templates: int,
    estimate: Callable[[str, int, int], tuple[float, bool]],
) -> dict[str, Any]:
    """Tokens per 1,000 pages, calls per template and derived dollars, per model and in total.

    Tokens count cached calls too (a fresh run pays them); ``spend_usd`` is what this run
    actually paid. Dollars per 1,000 pages are derived from the token counts and the price table.
    """
    by_model: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    for u in usage:
        model = str(u.get("model") or "unknown")
        m = by_model.setdefault(
            model, {"calls": 0, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        m["calls"] += 1
        m["cached_calls"] += bool(u.get("cached"))
        m["input_tokens"] += int(u.get("input_tokens") or 0)
        m["output_tokens"] += int(u.get("output_tokens") or 0)
        if not u.get("cached") and u.get("latency_s"):
            latencies.append(float(u["latency_s"]))
    per_1k = 1000.0 / pages if pages else None
    usd_per_1k: float | None = 0.0 if per_1k is not None else None
    estimated = False
    for model, m in by_model.items():
        if per_1k is None:
            continue
        usd, est = estimate(model, m["input_tokens"], m["output_tokens"])
        m["usd_per_1k_pages"] = usd * per_1k
        usd_per_1k = (usd_per_1k or 0.0) + usd * per_1k
        estimated = estimated or est
    calls = sum(m["calls"] for m in by_model.values())
    tokens_in = sum(m["input_tokens"] for m in by_model.values())
    tokens_out = sum(m["output_tokens"] for m in by_model.values())
    return {
        "pages": pages,
        "templates": templates,
        "calls": calls,
        "cached_calls": sum(m["cached_calls"] for m in by_model.values()),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "input_tokens_per_1k_pages": tokens_in * per_1k if per_1k is not None else None,
        "output_tokens_per_1k_pages": tokens_out * per_1k if per_1k is not None else None,
        "calls_per_template": calls / templates if templates else None,
        "usd_per_1k_pages": usd_per_1k,
        "price_estimated": estimated,
        "spend_usd": sum(float(u.get("cost_usd") or 0.0) for u in usage),
        "llm_latency_s_mean": fmean(latencies) if latencies else None,
        "by_model": by_model,
        "models": sorted(by_model),
    }


def speed_summary(run: RunDir) -> dict[str, Any]:
    """Stage -> timing entry from ``timings.json`` (pages/s/core per stage)."""
    if not run.timings_json.exists():
        return {}
    data = read_json(run.timings_json)
    return dict(data) if isinstance(data, dict) else {}


# -- the run ---------------------------------------------------------------------------------------


class _Runner:
    def __init__(
        self,
        cfg: EvalConfig,
        hooks: Hooks,
        work_dir: Path,
        baselines_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.hooks = hooks
        self.work_dir = work_dir
        self.baselines_dir = baselines_dir
        self.spent = 0.0

    # -- shared front stages ---------------------------------------------------------------------
    def _front(self, site: EvalSite, root: Path) -> tuple[RunDir, dict[int, str]]:
        params = self.cfg.method.params
        src: Path | list[Path] = (
            site.input_paths[0] if len(site.input_paths) == 1 else list(site.input_paths)
        )
        run = self.hooks.ingest(
            src, root, manifest=site.manifest, site=site.name, workers=self.cfg.workers
        )
        self.hooks.cluster(run, min_cluster_size=params.min_cluster_size, workers=self.cfg.workers)
        return run, self.hooks.page_urls(run)

    def _clustering(
        self, site: EvalSite, run: RunDir, urls: Mapping[int, str], out: SiteOutcome
    ) -> None:
        from ditto.cluster.models import Clusters

        if not run.clusters_json.exists():
            return
        clusters = Clusters.load(run.clusters_json)
        templates = clusters.templates()
        info: dict[str, Any] = {
            "n_clusters": len(clusters.clusters),
            "n_templates": len(templates),
            "tail_pages": sum(c.size for c in clusters.clusters if c.is_tail),
        }
        if site.truth_templates:
            true: list[str] = []
            pred: list[str | None] = []
            for pid, url in sorted(urls.items()):
                key = site.key_for_url(url)
                if key is None or key not in site.truth_templates:
                    continue
                true.append(site.truth_templates[key])
                pred.append(clusters.template_of(pid))
            info["n_true_templates"] = len(set(site.truth_templates.values()))
            info.update(self.hooks.clustering_metrics(true, pred))
        out.clustering = info

    # -- ditto -----------------------------------------------------------------------------------
    def run_ditto(self, site: EvalSite, out: SiteOutcome, settings: Any) -> None:
        cfg = self.cfg
        params = cfg.method.params
        root = self.work_dir / site.name / "run"
        run, urls = self._front(site, root)
        out.run_dir = run.root
        out.n_pages = len(urls)
        self._clustering(site, run, urls, out)
        if params.until == "extract":
            if cfg.method.mode == "prompted" and site.fields_schema_path is None:
                raise ValueError(f"site {site.name} has no fields.yaml for prompted mode")
            schema = (
                FieldsSchema.load(site.fields_schema_path)
                if cfg.method.mode == "prompted" and site.fields_schema_path
                else None
            )
            self.hooks.analyze(
                run,
                include_tail=params.tail == "spec",
                workers=cfg.workers,
                min_tokens=params.union_min_tokens,
                max_tokens=params.union_max_tokens,
            )
            opts = self.hooks.run_options(
                mode=cfg.method.mode,
                schema=site.fields_schema_path if cfg.method.mode == "prompted" else None,
                yes=True,
                tail=params.tail,
                min_cluster_size=params.min_cluster_size,
                max_repair_calls=params.max_repair_calls,
                workers=cfg.workers,
                concurrency=params.concurrency,
                settings=settings,
            )
            self.hooks.continue_run(run, opts, schema)
            self._score_ditto(site, run, urls, out)
        usage = read_usage(run)
        templates = (out.clustering or {}).get("n_templates", 0)
        out.cost = cost_summary(
            usage, pages=out.n_pages, templates=templates, estimate=self.hooks.estimate_cost
        )
        out.spend_usd = float(out.cost["spend_usd"])
        out.models.update(out.cost["models"])
        out.prompt_versions.update(
            str(u["prompt_version"]) for u in usage if u.get("prompt_version")
        )
        out.speed = speed_summary(run)

    def _score_ditto(
        self, site: EvalSite, run: RunDir, urls: Mapping[int, str], out: SiteOutcome
    ) -> None:
        from ditto.specgen.models import Spec

        cfg = self.cfg
        pid_key = {pid: k for pid, url in urls.items() if (k := site.key_for_url(url))}
        path_key = {_url_path(urls[pid]): k for pid, k in pid_key.items()}
        predictions: dict[str, dict[str, Any]] = {}
        page_template: dict[str, str] = {}
        for row in read_output_rows(run):
            key = site.key_for_url(str(row.get("url", "")))
            if key is None and row.get("page_id") is not None:
                key = pid_key.get(int(row["page_id"]))
            if key is None:
                continue
            predictions[key] = {k: v for k, v in row.items() if k not in ROW_META_KEYS}
            if row.get("template_id"):
                page_template[key] = str(row["template_id"])
        seen: dict[str, set[str]] = {}
        for path in sorted(run.specs_dir.glob("*.json")):
            spec = Spec.load(path)
            gb = spec.generated_by
            out.models.add(gb.model)
            out.prompt_versions.add(gb.prompt_version)
            if gb.repair_prompt_version:
                out.prompt_versions.add(gb.repair_prompt_version)
            keys = {k for s in gb.pages_seen if (k := _seen_key(s, site, pid_key, path_key))}
            seen[spec.template_id] = keys
        excluded = set().union(*seen.values()) if seen else set()
        if cfg.method.mode == "unprompted":
            predictions, out.alignment = self._align(site, predictions, page_template, excluded)
        fields = site.fields()
        out.score = score_site(
            site.name,
            site.vertical,
            predictions,
            site.labels,
            site.field_kinds,
            fields=fields,
            exclude=excluded,
        )
        if cfg.sample.pages_per_site:
            # the same page set and seed as the baselines, hence the same sample
            ingested = {k for k in pid_key.values() if k in site.labels}
            sample = set(sample_keys(ingested, cfg.sample.pages_per_site, cfg.seeds.run))
            out.sample_score = score_site(
                site.name,
                site.vertical,
                predictions,
                {k: v for k, v in site.labels.items() if k in sample},
                site.field_kinds,
                fields=fields,
                exclude=excluded,
            )
        out.optional = optional_field_recall(
            predictions, site.labels, page_template, seen, site.field_kinds
        )

    def _align(
        self,
        site: EvalSite,
        predictions: Mapping[str, Mapping[str, Any]],
        page_template: Mapping[str, str],
        excluded: set[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
        by_template: dict[str, list[str]] = defaultdict(list)
        for key in predictions:
            if key in site.labels and key not in excluded and key in page_template:
                by_template[page_template[key]].append(key)
        alignment: dict[str, dict[str, str]] = {}
        for tid, keys in sorted(by_template.items()):
            sample = sample_keys(keys, self.cfg.sample.align_pages, self.cfg.seeds.run)
            alignment[tid] = align_field_names(
                predictions, site.labels, sample, min_fraction=self.cfg.sample.align_min_fraction
            )
        renamed = {
            key: rename_fields(row, alignment.get(page_template.get(key, ""), {}))
            for key, row in predictions.items()
        }
        return renamed, alignment

    # -- baseline --------------------------------------------------------------------------------
    def run_baseline(self, site: EvalSite, out: SiteOutcome, settings: Any) -> None:
        from ditto.cluster.models import Clusters
        from ditto.fingerprint.dom import clean_html

        cfg = self.cfg
        assert cfg.method.baseline is not None
        fn = self.hooks.load_baseline(cfg.method.baseline, self.baselines_dir)
        if site.fields_schema_path is None:
            raise ValueError(f"site {site.name} has no fields.yaml")
        schema = FieldsSchema.load(site.fields_schema_path)
        site_dir = self.work_dir / site.name
        run, urls = self._front(site, site_dir / "run")
        out.run_dir = run.root
        out.n_pages = len(urls)
        self._clustering(site, run, urls, out)
        clusters = Clusters.load(run.clusters_json) if run.clusters_json.exists() else None
        labeled = {
            k: pid
            for pid, url in sorted(urls.items())
            if (k := site.key_for_url(url)) in site.labels
        }
        keys = sorted(labeled)
        n = cfg.sample.pages_per_site
        if n:
            keys = sample_keys(keys, n, cfg.seeds.run)
        pages = []
        for key in keys:
            pid = labeled[key]
            root = clean_html(self.hooks.read_html(run, pid))
            pages.append(
                PageInput(
                    key=key,
                    url=urls[pid],
                    html=lxml_html.tostring(root, encoding="unicode"),
                    template_id=clusters.template_of(pid) if clusters else None,
                )
            )
        t0 = time.perf_counter()
        result: BaselineOutput = fn(
            pages,
            schema,
            out_dir=site_dir / "baseline",
            settings=settings,
            max_pages=n,
            seed=cfg.seeds.run,
            site=site.name,
        )
        seconds = time.perf_counter() - t0
        pages_run = result.pages_run if result.pages_run is not None else len(pages)
        scored = set(keys)
        if pages_run < len(pages):
            scored = set(result.predictions) & scored
        out.score = score_site(
            site.name,
            site.vertical,
            result.predictions,
            {k: site.labels[k] for k in scored},
            site.field_kinds,
            fields=site.fields(),
        )
        templates = len({p.template_id for p in pages if p.template_id})
        per_1k = 1000.0 / pages_run if pages_run else None
        usd_per_1k: float | None = None
        estimated = False
        if per_1k is not None:
            if result.model:
                usd, estimated = self.hooks.estimate_cost(
                    result.model, result.input_tokens, result.output_tokens
                )
                usd_per_1k = usd * per_1k
            else:
                usd_per_1k = result.cost_usd * per_1k
        spend = result.spend_usd if result.spend_usd is not None else result.cost_usd
        out.cost = {
            "pages": pages_run,
            "templates": templates,
            "calls": result.calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "input_tokens_per_1k_pages": result.input_tokens * per_1k if per_1k else None,
            "output_tokens_per_1k_pages": result.output_tokens * per_1k if per_1k else None,
            "calls_per_template": result.calls / templates if templates else None,
            "calls_per_page": result.calls / pages_run if pages_run else None,
            "usd_per_1k_pages": usd_per_1k,
            "price_estimated": estimated,
            "spend_usd": spend,
            "extrapolated": pages_run < out.n_pages,
            "models": [result.model] if result.model else [],
        }
        if pages_run < out.n_pages:
            out.notes.append(
                f"cost per 1,000 pages extrapolated from {pages_run} of {out.n_pages} pages"
            )
        out.notes.extend(result.notes)
        out.spend_usd = spend
        if result.model:
            out.models.add(result.model)
        out.speed = {
            "baseline": {
                "seconds": round(seconds, 4),
                "pages": pages_run,
                "pages_per_sec": round(pages_run / seconds, 4) if seconds > 0 else None,
            }
        }

    # -- loop ------------------------------------------------------------------------------------
    def run_site(self, site: EvalSite, out: SiteOutcome) -> None:
        settings = None
        if self.cfg.uses_llm:
            remaining = max(0.0, self.cfg.budget_usd - self.spent)
            settings = self.hooks.settings(
                {str(r): m for r, m in self.cfg.models.items()}, remaining
            )
            for role, pinned in self.cfg.models.items():
                effective = self.hooks.resolve_model(role, settings)
                if _qualified(effective) != _qualified(pinned):
                    raise RuntimeError(f"model for role {role} is {effective}, pinned {pinned}")
        if self.cfg.method.kind == "baseline":
            self.run_baseline(site, out, settings)
        else:
            self.run_ditto(site, out, settings)

    def run(self, sites: Sequence[EvalSite]) -> list[SiteOutcome]:
        outcomes: list[SiteOutcome] = []
        stop: str | None = None
        for site in sites:
            if stop is None and self.cfg.uses_llm and self.spent >= self.cfg.budget_usd:
                stop = f"evaluation budget of ${self.cfg.budget_usd:.2f} reached"
            if stop is not None:
                outcomes.append(SiteOutcome(site=site, status="skipped", reason=stop))
                continue
            _log(f"{self.cfg.name}: site {site.name} ({site.vertical})")
            t0 = time.perf_counter()
            out = SiteOutcome(site=site)
            try:
                self.run_site(site, out)
            except NotRun as exc:
                out.status, out.reason = "not_run", str(exc)
            except BudgetExceeded as exc:
                out.status, out.reason = "failed", f"BudgetExceeded: {exc}"
                stop = f"LLM budget exceeded at site {site.name}"
            except Exception as exc:  # one failing site must not end the evaluation
                out.status, out.reason = "failed", f"{type(exc).__name__}: {exc}"
                log = self.work_dir / site.name / "error.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(traceback.format_exc())
                out.notes.append(f"traceback: {log}")
            if out.status != "ok":
                out.score = out.sample_score = out.optional = None
                if out.run_dir is not None and not out.spend_usd:
                    usage = read_usage(RunDir(out.run_dir))
                    out.spend_usd = sum(float(u.get("cost_usd") or 0.0) for u in usage)
            out.cost["wall_seconds"] = round(time.perf_counter() - t0, 3)
            self.spent += out.spend_usd
            if not self.cfg.keep_runs and out.run_dir is not None:
                shutil.rmtree(out.run_dir, ignore_errors=True)
                out.notes.append("run directory deleted after scoring (keep_runs: false)")
            outcomes.append(out)
        return outcomes


def interleave_verticals(sites: Sequence[EvalSite]) -> list[EvalSite]:
    """Sites round-robin across verticals (first site of each vertical, then the second, ...),
    so an evaluation stopped by its budget cap still covers every vertical."""
    by_vertical: dict[str, list[EvalSite]] = defaultdict(list)
    for site in sites:
        by_vertical[site.vertical].append(site)
    queues = [by_vertical[v] for v in sorted(by_vertical)]
    out: list[EvalSite] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


def _qualified(model: str) -> str:
    """``provider/model`` form (a bare id means OpenAI, as in the LLM layer)."""
    return model if "/" in model else f"openai/{model}"


def _seen_key(
    item: str, site: EvalSite, pid_key: Mapping[int, str], path_key: Mapping[str, str]
) -> str | None:
    """Page key of one ``generated_by.pages_seen`` entry: an absolute URL, a URL path (with
    query), ``page:<page_id>``, a bare page id, or a page key."""
    text = str(item).strip()
    pid = text.removeprefix("page:")
    if pid.isdigit() and int(pid) in pid_key:
        return pid_key[int(pid)]
    key = site.key_for_url(text)
    if key is not None:
        return key
    if text in path_key:
        return path_key[text]
    return text if text in site.labels else None


def _url_path(url: str) -> str:
    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "")


def _log(msg: str) -> None:
    print(f"[ditto eval] {msg}", file=sys.stderr, flush=True)


# -- aggregation -----------------------------------------------------------------------------------


def _mean(values: Iterable[Any]) -> float | None:
    kept = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return fmean(kept) if kept else None


def _sum_costs(outcomes: Sequence[SiteOutcome]) -> dict[str, Any]:
    costs = [o.cost for o in outcomes if o.cost.get("pages")]
    pages = sum(int(c["pages"]) for c in costs)
    templates = sum(int(c.get("templates") or 0) for c in costs)
    calls = sum(int(c.get("calls") or 0) for c in costs)
    tin = sum(int(c.get("input_tokens") or 0) for c in costs)
    tout = sum(int(c.get("output_tokens") or 0) for c in costs)
    usd = [
        c["usd_per_1k_pages"] * c["pages"] / 1000
        for c in costs
        if c.get("usd_per_1k_pages") is not None
    ]
    return {
        "pages": pages,
        "templates": templates,
        "calls": calls,
        "input_tokens": tin,
        "output_tokens": tout,
        "input_tokens_per_1k_pages": tin * 1000 / pages if pages else None,
        "output_tokens_per_1k_pages": tout * 1000 / pages if pages else None,
        "calls_per_template": (
            _mean(c.get("calls_per_template") for c in costs) if templates else None
        ),
        "usd_per_1k_pages": sum(usd) * 1000 / pages if pages and usd else None,
        "price_estimated": any(c.get("price_estimated") for c in costs),
        "extrapolated": any(c.get("extrapolated") for c in costs),
        "spend_usd": sum(o.spend_usd for o in outcomes),
        "llm_latency_s_mean": _mean(c.get("llm_latency_s_mean") for c in costs),
    }


def _mean_speed(outcomes: Sequence[SiteOutcome]) -> dict[str, float | None]:
    stages: dict[str, list[Any]] = defaultdict(list)
    for o in outcomes:
        for stage, entry in o.speed.items():
            if isinstance(entry, Mapping):
                stages[stage].append(entry.get("pages_per_sec_per_core"))
    return {stage: _mean(values) for stage, values in stages.items()}


def _mean_clustering(outcomes: Sequence[SiteOutcome]) -> dict[str, Any] | None:
    rows = [o.clustering for o in outcomes if o.clustering]
    if not rows:
        return None
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    return {k: _mean(r.get(k) for r in rows) for k in keys} | {"n_sites": len(rows)}


def _groups(outcomes: Sequence[SiteOutcome]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in MODES:
        per_group: dict[str, list[float | None]] = defaultdict(list)
        for o in outcomes:
            if o.score and o.site.field_groups:
                for g, m in o.score.group_metrics(o.site.field_groups, mode).items():
                    per_group[g].append(m["f1"])
        if per_group:
            out[mode] = {g: {"f1": _mean(v), "n_sites": len(v)} for g, v in per_group.items()}
    return out


def _curves(outcomes: Sequence[SiteOutcome]) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in outcomes:
        param = o.site.meta.get("parameter")
        if param is None:
            continue
        point: dict[str, Any] = {
            "value": o.site.meta.get("value"),
            "site": o.site.name,
            "status": o.status,
            "n_pages": o.n_pages,
        }
        for k, v in (o.clustering or {}).items():
            point[k] = v
        if o.score:
            for mode in MODES:
                for k, v in o.score.metrics(mode).items():
                    point[f"{k}_{mode}"] = v
        if o.optional:
            point["optional_recall_norm"] = o.optional.total("norm").recall
        curves[str(param)].append(point)
    return dict(curves)


def summarize(outcomes: Sequence[SiteOutcome]) -> dict[str, Any]:
    ok = [o for o in outcomes if o.status == "ok"]
    scores = [o.score for o in ok if o.score]
    sample_scores = [o.sample_score for o in ok if o.sample_score]
    optional = OptionalRecall()
    for o in ok:
        if o.optional:
            optional.add(o.optional)
    by_vertical: dict[str, list[SiteOutcome]] = defaultdict(list)
    for o in ok:
        by_vertical[o.site.vertical].append(o)
    unprompted = [o for o in ok if o.alignment is not None]
    return {
        "n_sites": len(outcomes),
        "n_ok": len(ok),
        "status_counts": dict(
            sorted(
                {
                    s: sum(o.status == s for o in outcomes) for s in {o.status for o in outcomes}
                }.items()
            )
        ),
        "accuracy": aggregate(scores) if scores else None,
        "accuracy_sample": aggregate(sample_scores) if sample_scores else None,
        "optional_field_recall": optional.to_dict() if optional.fields else None,
        "groups": _groups(ok),
        "clustering": _mean_clustering(ok),
        "cost": _sum_costs(ok),
        "cost_per_vertical": {v: _sum_costs(os) for v, os in sorted(by_vertical.items())},
        "speed_pages_per_sec_per_core": _mean_speed(ok),
        "curves": _curves(outcomes),
        "unprompted_recall": (
            {
                mode: (
                    aggregate([o.score for o in unprompted if o.score])["overall"][mode]["recall"]
                )
                for mode in MODES
            }
            if unprompted
            else None
        ),
    }


# -- files ---------------------------------------------------------------------------------------


def _git_info(cwd: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def manifest_hashes(datasets_dir: Path) -> dict[str, dict[str, str]]:
    """SHA-256 of every ``<datasets_dir>/*/manifest.json``."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(Path(datasets_dir).glob("*/manifest.json")):
        out[path.parent.name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return out


def machine_info() -> dict[str, Any]:
    return {
        "cpu_count": cpu_count(),
        "memory_gb": round(memory_gb(), 2),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _num(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def _metric_cols(prefix: str, agg: Mapping[str, Any] | None) -> dict[str, Any]:
    cols: dict[str, Any] = {}
    for mode in MODES:
        m = (agg or {}).get(mode) or {}
        for k in ("f1", "precision", "recall"):
            cols[f"{prefix}{k}_{mode}"] = _num(m.get(k))
    return cols


def _cost_cols(cost: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: _num(cost.get(k))
        for k in (
            "input_tokens_per_1k_pages",
            "output_tokens_per_1k_pages",
            "calls_per_template",
            "usd_per_1k_pages",
            "spend_usd",
        )
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns: dict[str, None] = {}
    for row in rows:
        for k in row:
            columns.setdefault(k, None)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})


def _site_row(o: SiteOutcome, stages: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "site": o.site.name,
        "vertical": o.site.vertical,
        "status": o.status,
        "pages": o.n_pages,
        "pages_scored": o.score.pages_scored if o.score else None,
    }
    for mode in MODES:
        m = o.score.metrics(mode) if o.score else {}
        for k in ("f1", "precision", "recall"):
            row[f"{k}_{mode}"] = _num(m.get(k))
    row["optional_recall_norm"] = _num(o.optional.total("norm").recall) if o.optional else None
    for k in ("ari", "purity", "over_merge", "under_merge", "n_templates", "n_true_templates"):
        row[k] = _num((o.clustering or {}).get(k))
    row.update(_cost_cols(o.cost))
    for stage in stages:
        entry = o.speed.get(stage)
        row[f"pps_core_{stage}"] = (
            _num(entry.get("pages_per_sec_per_core")) if isinstance(entry, Mapping) else None
        )
    row["reason"] = o.reason
    return row


def write_results(
    out_dir: Path,
    cfg: EvalConfig,
    outcomes: Sequence[SiteOutcome],
    summary: Mapping[str, Any],
    *,
    status: str,
    reason: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "name": cfg.name,
        "dataset": cfg.dataset,
        "split": cfg.selection.split,
        "method": cfg.method.model_dump(),
        "method_label": cfg.method.label,
        "status": status,
        "reason": reason,
        "summary": summary,
        "sites": [o.to_dict() for o in outcomes],
    }
    write_json(out_dir / "results.json", results)
    stages = sorted({s for o in outcomes for s in o.speed})
    _write_csv(out_dir / "per_site.csv", [_site_row(o, stages) for o in outcomes])
    rows: list[dict[str, Any]] = []
    accuracy = summary.get("accuracy") or {}
    cost_v = summary.get("cost_per_vertical") or {}
    for vertical, agg in (accuracy.get("per_vertical") or {}).items():
        rows.append(
            {
                "scope": vertical,
                "n_sites": agg.get("n_sites"),
                **_metric_cols("", agg),
                **_cost_cols(cost_v.get(vertical) or {}),
            }
        )
    overall = accuracy.get("overall") or {}
    speed = summary.get("speed_pages_per_sec_per_core") or {}
    rows.append(
        {
            "scope": "overall",
            "n_sites": summary.get("n_ok"),
            **_metric_cols("", overall),
            **_cost_cols(summary.get("cost") or {}),
            **{f"pps_core_{k}": _num(v) for k, v in sorted(speed.items())},
            **{
                f"cluster_{k}": _num(v)
                for k, v in (summary.get("clustering") or {}).items()
                if k in ("ari", "purity", "over_merge", "under_merge")
            },
        }
    )
    _write_csv(out_dir / "results.csv", rows)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _unique_dir(parent: Path, stamp: str) -> Path:
    path = parent / stamp
    n = 1
    while path.exists():
        n += 1
        path = parent / f"{stamp}-{n}"
    return path


def run_eval(
    config_path: str | Path,
    *,
    results_dir: str | Path = "eval/results",
    dry_run: bool = False,
    work_root: str | Path = "data/eval",
    datasets_dir: str | Path = "eval/datasets",
    baselines_dir: str | Path = DEFAULT_BASELINES_DIR,
    hooks: Hooks | None = None,
) -> Path:
    """Run one evaluation config; return the results directory it wrote."""
    config_path = Path(config_path)
    cfg = EvalConfig.load(config_path)
    hooks = hooks or Hooks()
    stamp = _timestamp()
    out_dir = _unique_dir(Path(results_dir) / cfg.name, stamp + ("-dryrun" if dry_run else ""))
    work_dir = Path(work_root) / cfg.name / out_dir.name
    started = time.perf_counter()
    meta: dict[str, Any] = {
        "name": cfg.name,
        "description": cfg.description,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": cfg.model_dump(mode="json"),
        "dry_run": dry_run,
        "git": _git_info(Path.cwd()),
        "dataset_manifests": manifest_hashes(Path(datasets_dir)),
        "machine": machine_info(),
        "work_dir": str(work_dir),
        "budget_usd": cfg.budget_usd,
        "cost_basis": (
            "usd_per_1k_pages = tokens x standard-tier list prices (config/prices.yaml, no cache, "
            "batch or flex discounts) for every method; spend_usd = what was actually billed"
        ),
    }
    try:
        from ditto import __version__

        meta["ditto_version"] = __version__
    except ImportError:  # pragma: no cover - the package always exports it
        meta["ditto_version"] = None
    try:
        sites = hooks.load_sites(cfg.dataset, cfg.selection_obj(), work_dir / "_dataset")
    except DatasetUnavailable as exc:
        meta.update(models={"pinned": cfg.models, "used": []}, prompt_versions=[], spend_usd=0.0)
        if not dry_run:
            write_results(out_dir, cfg, [], {}, status="not_run", reason=str(exc))
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "meta.json", meta | {"status": "not_run", "reason": str(exc)})
        _log(f"{cfg.name}: not run ({exc})")
        return out_dir
    notes = sorted({str(n) for s in sites for n in s.meta.get("notes", [])})
    meta["notes"] = notes
    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        plan = {
            "name": cfg.name,
            "method": cfg.method.label,
            "uses_llm": cfg.uses_llm,
            "budget_usd": cfg.budget_usd,
            "models": cfg.models,
            "sites": [
                {
                    "site": s.name,
                    "vertical": s.vertical,
                    "input_kind": s.input_kind,
                    "inputs": [str(p) for p in s.input_paths],
                    "labeled_pages": len(s.labels),
                    "fields": s.fields(),
                    "fields_yaml": str(s.fields_schema_path) if s.fields_schema_path else None,
                }
                for s in sites
            ],
            "projected_usd": None,
            "note": "the template count, hence the cost, is known only after clustering; "
            "the run stops once budget_usd is spent",
        }
        write_json(out_dir / "plan.json", plan)
        write_json(out_dir / "meta.json", meta | {"status": "dry_run"})
        return out_dir
    runner = _Runner(cfg, hooks, work_dir, Path(baselines_dir))
    outcomes = runner.run(interleave_verticals(sites))
    summary = summarize(outcomes)
    statuses = {o.status for o in outcomes}
    status = "ok" if "ok" in statuses else (statuses.pop() if len(statuses) == 1 else "failed")
    reason = None if status == "ok" else "; ".join(sorted({o.reason or "" for o in outcomes}))
    write_results(out_dir, cfg, outcomes, summary, status=status, reason=reason)
    meta.update(
        status=status,
        models={
            "pinned": cfg.models,
            "used": sorted({m for o in outcomes for m in o.models}),
        },
        prompt_versions=sorted({p for o in outcomes for p in o.prompt_versions}),
        spend_usd=round(sum(o.spend_usd for o in outcomes), 6),
        spend_by_site={o.site.name: round(o.spend_usd, 6) for o in outcomes},
        wall_seconds=round(time.perf_counter() - started, 3),
    )
    write_json(out_dir / "meta.json", meta)
    _log(f"{cfg.name}: {status}; results in {out_dir}")
    return out_dir
