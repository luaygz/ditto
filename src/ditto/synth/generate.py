"""``ditto synth generate``: write a synthetic site with ground-truth sidecars.

Layout (``local://contracts.md``, "Synthetic site layout")::

    <out>/site/p000123.html  <out>/site/urls.tsv      what ditto ingests
    <out>/truth/p000123.json                           sidecar per page
    <out>/truth/templates.json                         template definitions
    <out>/truth/fields/<template>.yaml                  prompted-mode fields.yaml
    <out>/synth.json                                   params, seed, counts, measurements, timing

Output bytes are a function of ``(params, seed)`` only: every page draws from its own RNG seeded
with ``(seed, page index)``, so worker count and chunking never change the result.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import yaml

from ditto.parallel import cpu_count, default_workers, pmap
from ditto.synth.blueprint import TemplateDef
from ditto.synth.page import make_page_data, page_url, truth_record
from ditto.synth.params import SynthParams
from ditto.synth.plan import SitePlan, build_plan
from ditto.synth.render import (
    WS_MODES,
    Escaper,
    ShellText,
    Writer,
    WsMode,
    assemble,
    content_tree,
    entity_escaper,
    esc_plain,
    head_text,
    shell_text,
)
from ditto.synth.values import Pools, build_pools

CHUNK_SIZE = 128
_CODECS = {"utf-8": "utf-8", "windows-1252": "cp1252"}
_URL_PATTERNS = {
    "slug-id": "/{segment}/{slug}-{id}",
    "id-slug": "/{segment}/{id}/{slug}",
    "id": "/{segment}/{id}",
    "slug": "/{segment}/{slug}-{id}/",
    "date-slug": "/{segment}/{yyyy}/{mm}/{slug}-{id}",
    "html": "/{segment}/{slug}-{id}.html",
    "query": "/{segment}.php?id={id}",
}


@dataclass
class SynthSummary:
    out: Path
    seed: int
    n_pages: int
    n_templates: int
    template_pages: dict[str, int]
    seconds: float
    pages_per_second: float
    workers: int
    bytes_html: int
    near_duplicates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Worker:
    plan: SitePlan
    pools: Pools
    shells: dict[tuple[int, str], ShellText]
    site_dir: str
    truth_dir: str
    width: int


_STATE: _Worker | None = None

# (index, url, template index, optional mask, content elements, shell elements, bytes)
PageStat = tuple[int, str, int, int, int, int, int]


def _init_worker(
    plan: SitePlan, shells: dict[tuple[int, str], ShellText], site: str, truth: str, width: int
) -> None:
    global _STATE
    _STATE = _Worker(plan, build_pools(plan.seed), shells, site, truth, width)


def page_name(i: int, width: int) -> str:
    return f"p{i:0{width}d}"


def _optional_units(t: TemplateDef) -> list[tuple[str, str | None]]:
    """Optional scalars ``(name, None)`` then optional columns ``(table, column)``."""
    units: list[tuple[str, str | None]] = [(s.name, None) for s in t.scalars if s.optional]
    units += [(tab.name, c.name) for tab in t.tables for c in tab.columns if c.optional]
    return units


def render_page(st: _Worker, i: int) -> tuple[bytes, dict[str, Any], PageStat]:
    """Render page ``i``: (html bytes, sidecar, stats)."""
    plan = st.plan
    params = plan.params
    tidx = plan.page_template[i]
    t = plan.templates[tidx]
    rng = random.Random(f"{plan.seed}|page|{i}")
    r = rng.random()
    variant = 0
    acc = 0.0
    for k, weight in enumerate(plan.variant_weights):
        acc += weight
        if r < acc:
            variant = k
            break
    else:
        variant = len(plan.variant_weights) - 1
    d = make_page_data(t, params, st.pools, rng, host=plan.host)
    uid = 10000 + 7 * i + rng.randrange(7)
    url = page_url(t, plan.host, uid, d.title, rng)

    nrng = random.Random(f"{plan.seed}|noise|{i}")
    noise, rate = params.noise, params.noise_rate
    esc: Escaper = (
        entity_escaper(nrng) if "entities" in noise and nrng.random() < rate else esc_plain
    )
    nbsp = "nbsp" in noise and nrng.random() < rate
    mode: WsMode = "pretty"
    if "whitespace" in noise and nrng.random() < rate:
        mode = nrng.choice(("minified", "loose"))
    charset = "windows-1252" if "charset" in noise and nrng.random() < rate else "utf-8"

    def nb(text: str) -> str:
        if not nbsp:
            return text
        r = nrng.random()
        if r < 0.4:
            return text.replace(" ", "\xa0")
        if r < 0.6:
            return text + "\xa0"
        return text

    shell = st.shells[(variant, mode)]
    w = Writer(mode, esc)
    w.write(content_tree(t, d, nb), shell.depth)
    head = head_text(f"{d.title} | {plan.site_name}", url, charset, mode, esc)
    data = assemble(head, shell, w.text(), mode).encode(
        _CODECS[charset], errors="xmlcharrefreplace"
    )

    page = page_name(i, st.width)
    truth = truth_record(t, d, page, url)
    mask = 0
    for bit, (name, col) in enumerate(_optional_units(t)):
        present = d.scalars[name] is not None if col is None else col in d.tables[name].columns
        mask |= int(present) << bit
    return data, truth, (i, url, tidx, mask, w.count, shell.elements, len(data))


def _render_chunk(chunk: Sequence[int]) -> list[PageStat]:
    st = _STATE
    assert st is not None, "worker not initialized"
    stats: list[PageStat] = []
    for i in chunk:
        data, truth, stat = render_page(st, i)
        page = page_name(i, st.width)
        with open(os.path.join(st.site_dir, page + ".html"), "wb") as fh:
            fh.write(data)
        with open(os.path.join(st.truth_dir, page + ".json"), "wb") as fh:
            fh.write(orjson.dumps(truth, option=orjson.OPT_INDENT_2) + b"\n")
        stats.append(stat)
    return stats


def _prepare_out(out: Path) -> tuple[Path, Path]:
    site, truth = out / "site", out / "truth"
    marker = out / "synth.json"
    busy = [p for p in (site, truth) if p.exists() and any(p.iterdir())]
    if busy and not marker.exists():
        raise FileExistsError(
            f"{busy[0]} exists and is not a synthetic site (no synth.json); refusing to overwrite"
        )
    for p in (site, truth):
        if p.exists():
            shutil.rmtree(p)
    out.mkdir(parents=True, exist_ok=True)
    # marks the directory as ours so an interrupted run can be regenerated in place
    marker.write_bytes(orjson.dumps({"generator": "ditto.synth", "status": "incomplete"}) + b"\n")
    (truth / "fields").mkdir(parents=True)
    site.mkdir(parents=True)
    return site, truth


def generate_site(
    params: SynthParams, seed: int, out: str | Path, *, workers: int | None = None
) -> SynthSummary:
    """Generate a synthetic site under ``out`` (see module docstring for the layout).

    Re-running into a directory that holds a previous synthetic site (``synth.json``) replaces
    its ``site/`` and ``truth/``; any other non-empty ``site/``/``truth/`` raises
    ``FileExistsError``."""
    global _STATE
    t0 = time.perf_counter()
    out = Path(out)
    site, truth = _prepare_out(out)
    plan = build_plan(params, seed)
    t_plan = time.perf_counter() - t0
    n = plan.n_pages
    width = max(6, len(str(max(0, n - 1))))
    shells = {
        (v, m): shell_text(plan.chrome, v, m)
        for v in range(len(plan.variant_weights))
        for m in WS_MODES
    }
    n_chunks = -(-n // CHUNK_SIZE)
    n_workers = default_workers(n_chunks) if workers is None else max(1, min(workers, n_chunks))
    stats: list[PageStat] = []
    try:
        for chunk_stats in pmap(
            _render_chunk,
            list(range(n)),
            workers=n_workers,
            chunk_size=CHUNK_SIZE,
            initializer=_init_worker,
            initargs=(plan, shells, str(site), str(truth), width),
        ):
            stats.extend(chunk_stats)
    finally:
        _STATE = None
    lines = [f"{page_name(i, width)}.html\t{url}\n" for i, url, *_ in stats]
    (site / "urls.tsv").write_text("".join(lines), encoding="utf-8")

    templates_doc = _templates_doc(plan, stats)
    (truth / "templates.json").write_bytes(
        orjson.dumps(templates_doc, option=orjson.OPT_INDENT_2) + b"\n"
    )
    for t in plan.templates:
        (truth / "fields" / f"{t.tid}.yaml").write_text(fields_yaml(t, seed), encoding="utf-8")

    seconds = time.perf_counter() - t0
    shares = [s[5] / (s[4] + s[5]) for s in stats if s[4] + s[5]]
    bytes_html = sum(s[6] for s in stats)
    summary = SynthSummary(
        out=out,
        seed=seed,
        n_pages=n,
        n_templates=len(plan.templates),
        template_pages={t.tid: t.pages for t in plan.templates},
        seconds=round(seconds, 3),
        pages_per_second=round(n / seconds, 1) if seconds else 0.0,
        workers=n_workers if n_chunks > 1 else 1,
        bytes_html=bytes_html,
        near_duplicates=plan.near_duplicates,
    )
    meta = {
        "generator": "ditto.synth",
        "seed": seed,
        "params": params.model_dump(mode="json"),
        "site_name": plan.site_name,
        "host": plan.host,
        "n_pages": n,
        "n_templates": len(plan.templates),
        "n_tail_templates": sum(t.is_tail for t in plan.templates),
        "template_pages": summary.template_pages,
        "near_duplicates": plan.near_duplicates,
        "cross_template_jaccard": plan.cross_jaccard,
        "measured": {
            "boilerplate_share_mean": round(sum(shares) / len(shares), 4) if shares else 0.0,
            "boilerplate_share_median": round(sorted(shares)[len(shares) // 2], 4)
            if shares
            else 0.0,
            "hashed_class_share": round(plan.hashed_class_share, 4),
            "content_elements_median": plan.content_elements_median,
            "shell_elements": [
                shells[(v, "pretty")].elements for v in range(len(plan.variant_weights))
            ],
            "variant_weights": [round(w, 4) for w in plan.variant_weights],
            "bytes_html": bytes_html,
            "mean_page_bytes": round(bytes_html / n, 1) if n else 0.0,
        },
        "generation": {
            "seconds": summary.seconds,
            "plan_seconds": round(t_plan, 3),
            "workers": summary.workers,
            "cpu_count": cpu_count(),
            "pages_per_second": summary.pages_per_second,
            "note": "timing only; every other byte of the output depends on (params, seed) alone",
        },
    }
    (out / "synth.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2) + b"\n")
    return summary


def _templates_doc(plan: SitePlan, stats: list[PageStat]) -> dict[str, Any]:
    counts: dict[int, list[int]] = {}
    pages: dict[int, int] = {}
    examples: dict[int, str] = {}
    for _, url, tidx, mask, *_ in stats:
        units = counts.setdefault(tidx, [0] * len(_optional_units(plan.templates[tidx])))
        for bit in range(len(units)):
            units[bit] += (mask >> bit) & 1
        pages[tidx] = pages.get(tidx, 0) + 1
        examples.setdefault(tidx, url)
    out = []
    for idx, t in enumerate(plan.templates):
        n = pages.get(idx, 0)
        observed = {
            unit: round(c / n, 4) if n else 0.0
            for unit, c in zip(_optional_units(t), counts.get(idx, []), strict=False)
        }
        fields: list[dict[str, Any]] = [
            {
                "name": t.title.name,
                "type": "scalar",
                "value_type": t.title.vtype,
                "optional": False,
                "presence": 1.0,
                "label": None,
                "description": t.title.desc,
            }
        ]
        for s in t.scalars:
            fields.append(
                {
                    "name": s.name,
                    "type": "scalar",
                    "value_type": s.vtype,
                    "optional": s.optional,
                    "presence": s.presence,
                    "observed_presence": observed.get((s.name, None), 1.0),
                    "label": s.label,
                    "description": s.desc,
                }
            )
        for lst in t.lists:
            fields.append(
                {
                    "name": lst.name,
                    "type": "list",
                    "value_type": lst.vtype,
                    "optional": False,
                    "presence": 1.0,
                    "label": lst.heading,
                    "description": lst.desc,
                }
            )
        for tab in t.tables:
            fields.append(
                {
                    "name": tab.name,
                    "type": "record_list",
                    "value_type": None,
                    "optional": False,
                    "presence": 1.0,
                    "label": tab.heading,
                    "description": tab.desc,
                    "columns": [
                        {
                            "name": c.name,
                            "label": c.label,
                            "value_type": c.vtype,
                            "optional": c.optional,
                            "presence": c.presence,
                            "observed_presence": observed.get((tab.name, c.name), 1.0),
                        }
                        for c in tab.columns
                    ],
                }
            )
        for kv in t.kvs:
            fields.append(
                {
                    "name": kv.name,
                    "type": "key_value",
                    "value_type": None,
                    "optional": False,
                    "presence": 1.0,
                    "label": kv.heading,
                    "description": kv.desc,
                    "style": kv.style,
                    "keys": [k for k, _ in kv.keys],
                }
            )
        out.append(
            {
                "template_id": t.tid,
                "kind": t.kind,
                "pages": n,
                "is_tail": t.is_tail,
                "near_duplicate_of": t.near_duplicate_of,
                "url_pattern": _URL_PATTERNS[t.url_style].replace("{segment}", t.segment),
                "url_example": examples.get(idx),
                "markup_style": t.markup,
                "absent_rendering": t.absent,
                "class_prefix": t.prefix,
                "field_scheme": t.field_scheme,
                "field_layout": t.field_layout,
                "fields": fields,
            }
        )
    return {"seed": plan.seed, "site_name": plan.site_name, "host": plan.host, "templates": out}


def fields_yaml(t: TemplateDef, seed: int) -> str:
    """Prompted-mode ``fields.yaml`` (short form ``name: description``) for template ``t``."""
    data: dict[str, str] = {t.title.name: t.title.desc}
    for s in t.scalars:
        data[s.name] = s.desc
    for lst in t.lists:
        data[lst.name] = lst.desc
    for tab in t.tables:
        data[tab.name] = f"{tab.desc} (columns: {', '.join(c.name for c in tab.columns)})"
    for kv in t.kvs:
        data[kv.name] = kv.desc
    header = (
        f"# {t.tid} ({t.kind}): prompted-mode fields, generated by `ditto synth` (seed {seed})\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000)


def synth_main(
    params_path: str | Path | None, seed: int, out: str | Path, *, workers: int | None = None
) -> SynthSummary:
    """CLI entry: ``ditto synth generate --params <yaml> --seed <n> --out <dir>``."""
    from ditto.synth.params import load_params

    summary = generate_site(load_params(params_path), seed, out, workers=workers)
    pairs = ", ".join(f"{p['a']}~{p['b']} J={p['jaccard']}" for p in summary.near_duplicates)
    print(
        f"synth: {summary.n_pages} pages, {summary.n_templates} templates -> {summary.out} "
        f"in {summary.seconds:.1f}s ({summary.pages_per_second:.0f} pages/s, {summary.workers} workers)"
        + (f"; near-duplicates {pairs}" if pairs else "")
    )
    return summary
