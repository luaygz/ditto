"""Synthetic dataset adapter: generated sites with per-page ground-truth sidecars.

``dataset_options`` (config) keys:

- ``site_dir``: use an already generated site directory (``site/``, ``truth/``) as is.
- ``params``: parameter YAML (default ``eval/synth/default.yaml``); ``overrides``: dict applied
  on top; ``seed``: generator seed (the config's ``seeds.synth``, frozen at 1).
- ``pages``: total page count (``total_pages``); ``scale_for_machine``: scale ``pages`` down to
  ``scaled_pages`` (default 20,000) on machines with fewer than 8 cores or less than 16 GB, and
  say so in the results.
- ``sweep``: ``{parameter: [values]}``; one site per value, each changing that single parameter
  from the defaults (curves). ``include_default``: also evaluate the unchanged default site.
  ``sweep_overrides``: ``{parameter: {other: value}}`` applied only while sweeping that
  parameter (e.g. lift ``total_pages`` while sweeping the pages-per-template median).
- ``cache_root``: generated sites are cached under ``<cache_root>/<hash of params+seed>``
  (default ``data/synth``) so reruns do not regenerate them.
- ``truth_only``: read only the true template ids, not the field labels (clustering-only
  evaluations at 100k pages keep memory low).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import orjson
import yaml

from ditto.evaluation.datasets.base import (
    DatasetUnavailable,
    EvalSite,
    LabelKind,
    Selection,
    select_sites,
)
from ditto.parallel import cpu_count, memory_gb
from ditto.schema import FieldRequest, FieldsSchema

DEFAULT_PARAMS = Path("eval/synth/default.yaml")
MIN_CORES = 8
MIN_MEMORY_GB = 16.0
SCALED_PAGES = 20_000
_SAFE = re.compile(r"[^A-Za-z0-9_.=-]+")

_SECTIONS: tuple[tuple[str, LabelKind], ...] = (
    ("fields", "scalar"),
    ("lists", "list"),
    ("records", "record_list"),
    ("kv", "key_value"),
)


def read_site(
    site_dir: Path,
    *,
    name: str,
    work_dir: Path,
    vertical: str = "synthetic",
    truth_only: bool = False,
) -> EvalSite:
    """Build an :class:`EvalSite` from a generated directory (``site/`` + ``truth/``)."""
    site_dir = Path(site_dir)
    truth = site_dir / "truth"
    html_dir = site_dir / "site"
    if not truth.is_dir() or not html_dir.is_dir():
        raise DatasetUnavailable(f"{site_dir} is not a generated synthetic site (site/, truth/)")
    labels: dict[str, dict[str, Any]] = {}
    url_to_key: dict[str, str] = {}
    truth_templates: dict[str, str] = {}
    kinds: dict[str, LabelKind] = {}
    for path in sorted(truth.glob("*.json")):
        if path.name == "templates.json":
            continue
        car = orjson.loads(path.read_bytes())
        key = str(car.get("page") or path.stem)
        row: dict[str, Any] = {}
        if not truth_only:
            for section, kind in _SECTIONS:
                for fname, value in (car.get(section) or {}).items():
                    row[fname] = value
                    kinds.setdefault(fname, kind)
        labels[key] = row
        if car.get("url"):
            url_to_key[str(car["url"])] = key
        if car.get("template_id"):
            truth_templates[key] = str(car["template_id"])
    if not labels:
        raise DatasetUnavailable(f"{truth} holds no page sidecars")
    meta: dict[str, Any] = {}
    synth_json = site_dir / "synth.json"
    if synth_json.exists():
        meta["synth"] = orjson.loads(synth_json.read_bytes())
    site_work = work_dir / _SAFE.sub("_", name)
    return EvalSite(
        name=name,
        vertical=vertical,
        input_paths=[html_dir],
        input_kind="dir",
        fields_schema_path=merge_fields(truth / "fields", site_work / "fields.yaml"),
        labels=labels,
        url_to_key=url_to_key,
        manifest=html_dir / "urls.tsv" if (html_dir / "urls.tsv").exists() else None,
        field_kinds={k: v for k, v in kinds.items() if v != "scalar"},
        truth_templates=truth_templates,
        meta=meta,
    )


def merge_fields(fields_dir: Path, out: Path) -> Path | None:
    """One site-wide ``fields.yaml``: the union of every template's fields (first wins)."""
    merged: dict[str, FieldRequest] = {}
    for path in sorted(Path(fields_dir).glob("*.yaml")):
        for f in FieldsSchema.load(path).fields:
            merged.setdefault(f.name, f)
    if not merged:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fields": [f.model_dump(exclude_defaults=True, mode="json") for f in merged.values()]
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def machine_scaled_pages(pages: int, scaled: int = SCALED_PAGES) -> tuple[int, str | None]:
    """``pages`` capped at ``scaled`` on a small machine, with the note saying so."""
    cores, mem = cpu_count(), memory_gb()
    if pages > scaled and (cores < MIN_CORES or mem < MIN_MEMORY_GB):
        note = (
            f"synthetic benchmark scaled from {pages:,} to {scaled:,} pages: machine has "
            f"{cores} cores and {mem:.1f} GB (needs >= {MIN_CORES} cores and "
            f">= {MIN_MEMORY_GB:.0f} GB for the full size)"
        )
        return scaled, note
    return pages, None


def _generate(
    params_path: Path | None, overrides: dict[str, Any], seed: int, cache_root: Path
) -> Path:
    from ditto.synth import generate_site, load_params

    params = load_params(params_path, overrides)
    digest = hashlib.sha256(
        json.dumps(
            {"params": params.model_dump(mode="json"), "seed": seed}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    out = cache_root / f"seed{seed}-{digest}"
    if not (out / "synth.json").exists():
        generate_site(params, seed, out)
    return out


def _site_name(param: str | None, value: Any) -> str:
    if param is None:
        return "default"
    text = json.dumps(value) if not isinstance(value, str) else value
    return _SAFE.sub("_", f"{param}={text}")


def load_sites(selection: Selection, work_dir: Path) -> list[EvalSite]:
    opts = dict(selection.options)
    truth_only = bool(opts.get("truth_only", False))
    if opts.get("site_dir"):
        site_dir = Path(opts["site_dir"])
        name = str(opts.get("name") or site_dir.name)
        site = read_site(site_dir, name=name, work_dir=work_dir, truth_only=truth_only)
        return select_sites([site], selection)
    params_path = Path(opts["params"]) if opts.get("params") else DEFAULT_PARAMS
    if not params_path.exists():
        raise DatasetUnavailable(f"synthetic parameter file {params_path} not found")
    seed = int(opts.get("seed", 1))
    base: dict[str, Any] = dict(opts.get("overrides") or {})
    notes: list[str] = []
    if opts.get("pages") is not None:
        pages = int(opts["pages"])
        if opts.get("scale_for_machine"):
            pages, note = machine_scaled_pages(pages, int(opts.get("scaled_pages", SCALED_PAGES)))
            if note:
                notes.append(note)
        base["total_pages"] = pages
    cache_root = Path(opts.get("cache_root", "data/synth"))
    points: list[tuple[str | None, Any]] = []
    sweep: dict[str, list[Any]] = dict(opts.get("sweep") or {})
    if not sweep or opts.get("include_default"):
        points.append((None, None))
    for sweep_param, values in sweep.items():
        points.extend((sweep_param, v) for v in values)
    sites: list[EvalSite] = []
    sweep_extra: dict[str, dict[str, Any]] = dict(opts.get("sweep_overrides") or {})
    for index, (param, value) in enumerate(points):
        overrides = dict(base)
        if param is not None:
            overrides.update(sweep_extra.get(param) or {})
            overrides[param] = value
        site_dir = _generate(params_path, overrides, seed, cache_root)
        name = _site_name(param, value)
        site = read_site(site_dir, name=name, work_dir=work_dir, truth_only=truth_only)
        site.meta.update(
            parameter=param,
            value=value,
            index=index,
            seed=seed,
            params=str(params_path),
            overrides=overrides,
            site_dir=str(site_dir),
            notes=list(notes),
        )
        sites.append(site)
    return select_sites(sites, selection)
