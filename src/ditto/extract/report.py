"""``report.json`` and ``report.md``: clusters, per-field stats, LLM tokens and cost, timings."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ditto.cluster.models import Clusters
from ditto.run import RunDir, read_json, write_json
from ditto.specgen.models import Spec
from ditto.verify.models import VerifyReport

STAT_KEYS = (
    "presence_rate",
    "expected_presence",
    "multiplicity",
    "type_check_pass",
    "rows_min",
    "rows_max",
    "keys_seen",
    "failure_rate",
)


def _count_lines(path: Any) -> int:
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def _generation(run: RunDir) -> dict[str, dict[str, Any]]:
    from ditto.verify.stage import load_generation

    return load_generation(run)


def _field_entry(
    spec_field: Any, report: VerifyReport | None, name: str, filled: int | None, pages: int
) -> dict[str, Any]:
    stats = spec_field.stats.model_dump(exclude_none=True)
    entry: dict[str, Any] = {
        "type": spec_field.type,
        "optional": spec_field.optional,
        "degraded": spec_field.degraded,
    }
    entry.update({k: stats[k] for k in STAT_KEYS if k in stats})
    fr = report.fields.get(name) if report else None
    if fr is not None:
        entry["selected"] = fr.selected_xpath
        entry["failures"] = fr.failures
        if fr.path_induced:
            entry["path_induced"] = True
    if filled is not None and pages:
        entry["fill_rate"] = round(filled / pages, 4)
    return entry


def build_report(run: RunDir) -> dict[str, Any]:
    """Collect everything the report shows from the run directory."""
    from ditto.llm import summarize_usage

    meta = run.read_meta()
    site = run.site
    clusters = Clusters.load(run.clusters_json) if run.clusters_json.exists() else None
    generation = _generation(run)
    extract = meta.get("extract") or {}
    extracted: dict[str, Any] = extract.get("templates", {})
    timings = read_json(run.timings_json) if run.timings_json.exists() else {}

    cluster_rows: list[dict[str, Any]] = []
    templates: dict[str, Any] = {}
    for c in clusters.templates(include_tail=True) if clusters else []:
        tid = c.template_id
        spec = Spec.load(run.spec(tid)) if run.spec(tid).exists() else None
        vpath = run.verify_report(tid)
        vrep = VerifyReport.load(vpath) if vpath.exists() else None
        ex = extracted.get(tid) or {}
        gen = generation.get(tid) or {}
        cluster_rows.append(
            {
                "template_id": tid,
                "size": c.size,
                "is_tail": c.is_tail,
                "url_patterns": c.url_patterns[:3],
                "url_examples": c.url_examples[:3],
                "merged_from": c.merged_from,
                "split_from": c.split_from,
                "spec": spec.status if spec else None,
                "generation": gen.get("status"),
                "rows": ex.get("pages"),
            }
        )
        if spec is None:
            continue
        pages = int(ex.get("pages") or 0)
        filled: dict[str, int] = ex.get("filled") or {}
        templates[tid] = {
            "size": c.size,
            "status": spec.status,
            "spec_source": vrep.spec_source if vrep else None,
            "generated_by": {
                "model": spec.generated_by.model,
                "prompt_version": spec.generated_by.prompt_version,
                "mode": spec.generated_by.mode,
                "repair_calls": spec.generated_by.repair_calls,
                "repair_prompt_version": spec.generated_by.repair_prompt_version,
            },
            "rows": pages,
            "fields": {
                name: _field_entry(f, vrep, name, filled.get(name) if ex else None, pages)
                for name, f in spec.fields.items()
            },
            "not_present": dict(spec.not_present),
            "degraded": [n for n, f in spec.fields.items() if f.degraded],
            "merge_split": vrep.merge_split if vrep else [],
            "repairs": [r.model_dump(exclude_none=True) for r in vrep.repairs] if vrep else [],
            "verify_seconds": vrep.seconds if vrep else None,
            "repair_llm_seconds": vrep.llm_seconds if vrep else None,
        }

    n_pages = clusters.n_pages if clusters else int(extract.get("pages") or 0)
    tail = [c for c in clusters.clusters if c.is_tail] if clusters else []
    tail_path = run.out_jsonl(site, None)
    generation_issues = {
        tid: {"status": e.get("status"), "reason": e.get("reason")}
        for tid, e in generation.items()
        if e.get("status") not in (None, "ok")
    }
    return {
        "site": site,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pages": n_pages,
        "clusters": {
            "n_templates": len(cluster_rows) - len(tail),
            "n_tail_clusters": len(tail),
            "tail_pages": sum(c.size for c in tail),
            "min_cluster_size": clusters.min_cluster_size if clusters else None,
            "templates": cluster_rows,
        },
        "templates": templates,
        "generation_issues": generation_issues,
        "tail_direct": (
            {"rows": _count_lines(tail_path), "path": str(tail_path.relative_to(run.root))}
            if tail_path.exists()
            else None
        ),
        "llm": summarize_usage(run),
        "timings": timings,
        "extract": {k: v for k, v in extract.items() if k != "templates"},
    }


# -- markdown ---------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    return "–" if value is None else f"{100 * float(value):.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return out


def _usage_rows(section: dict[str, Any]) -> list[list[str]]:
    return [
        [
            key,
            _fmt(row.get("calls")),
            _fmt(row.get("cached_calls")),
            _fmt(row.get("input_tokens")),
            _fmt(row.get("output_tokens")),
            f"${float(row.get('cost_usd') or 0):.4f}" + (" (est.)" if row.get("estimated") else ""),
            f"${float(row.get('list_cost_usd') or 0):.4f}",
        ]
        for key, row in sorted(section.items())
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# ditto report — {report['site']}", ""]
    cl = report["clusters"]
    llm = report.get("llm") or {}
    totals = llm.get("totals") or {}
    rows = sum(int(t.get("rows") or 0) for t in report["templates"].values())
    lines.append(
        f"{report['pages']} pages · {cl['n_templates']} templates · {cl['n_tail_clusters']} tail "
        f"clusters ({cl['tail_pages']} pages) · {rows} rows extracted · "
        f"{totals.get('calls', 0)} LLM calls, "
        f"{totals.get('input_tokens', 0)} input / {totals.get('output_tokens', 0)} output tokens, "
        f"${float(totals.get('cost_usd') or 0):.4f}"
        + (" (estimated prices)" if totals.get("estimated") else "")
    )
    lines += ["", "## Templates", ""]
    lines += _table(
        ["template", "pages", "tail", "spec", "rows", "URL pattern", "example URL", "notes"],
        [
            [
                c["template_id"],
                str(c["size"]),
                "yes" if c["is_tail"] else "",
                c["spec"] or (c["generation"] or "–"),
                _fmt(c["rows"]),
                (c["url_patterns"] or ["–"])[0],
                (c["url_examples"] or ["–"])[0],
                "; ".join(
                    ([f"merged {', '.join(c['merged_from'])}"] if c["merged_from"] else [])
                    + ([f"split from {c['split_from']}"] if c["split_from"] else [])
                ),
            ]
            for c in cl["templates"]
        ],
    )
    for tid, t in report["templates"].items():
        gb = t["generated_by"]
        lines += [
            "",
            f"## {tid} — {t['size']} pages, {t['rows']} rows",
            "",
            f"Spec {t['status']} from {t['spec_source'] or '?'}; model {gb['model']}, prompt "
            f"{gb['prompt_version']}, mode {gb['mode']}, repair calls {gb['repair_calls']}.",
            "",
        ]
        lines += _table(
            [
                "field",
                "type",
                "presence",
                "expected",
                "multiplicity",
                "type pass",
                "rows / keys",
                "fill",
                "status",
            ],
            [
                [
                    name,
                    f["type"],
                    _pct(f.get("presence_rate")),
                    _pct(f.get("expected_presence")),
                    _fmt(f.get("multiplicity")),
                    _fmt(f.get("type_check_pass")),
                    (
                        f"{f['rows_min']}–{f['rows_max']}"
                        if f.get("rows_max") is not None
                        else _fmt(f.get("keys_seen"))
                    ),
                    _pct(f.get("fill_rate")),
                    ", ".join(
                        s
                        for s, on in (
                            (f"degraded ({_pct(f.get('failure_rate'))} failing)", f["degraded"]),
                            ("optional", f["optional"]),
                            ("path-induced", f.get("path_induced", False)),
                        )
                        if on
                    )
                    or "ok",
                ]
                for name, f in t["fields"].items()
            ],
        )
        if t["not_present"]:
            lines += ["", "Not present:"]
            lines += [f"- `{name}`: {reason}" for name, reason in t["not_present"].items()]
        for ev in t["merge_split"]:
            if ev.get("action") == "merge":
                lines += ["", f"Merged {ev['absorbed']} into {ev['target']}."]
            else:
                lines += [
                    "",
                    f"Split signature groups {ev['signature_groups']} ({ev['pages']} pages) of "
                    f"{ev['template_id']} into {ev['new_template_id']} (fields "
                    f"{', '.join(ev['fields'])} missing there).",
                ]
    if report["generation_issues"]:
        lines += ["", "## Generation issues", ""]
        lines += [
            f"- {tid}: {e['status']}" + (f" — {e['reason']}" if e.get("reason") else "")
            for tid, e in report["generation_issues"].items()
        ]
    if report.get("tail_direct"):
        lines += ["", f"Long tail (direct): {report['tail_direct']['rows']} rows."]
    lines += ["", "## LLM usage", ""]
    usage_headers = ["", "calls", "cached", "input tokens", "output tokens", "cost", "list cost"]
    for title, key in (
        ("By role", "by_role"),
        ("By template", "by_template"),
        ("By model", "by_model"),
    ):
        section = llm.get(key) or {}
        if section:
            lines += [f"{title}:", ""]
            lines += _table(usage_headers, _usage_rows(section))
            lines.append("")
    if totals:
        lines += _table(usage_headers, _usage_rows({"total": totals}))
    lines += ["", "## Stage timings", ""]
    lines += _table(
        ["stage", "seconds", "pages", "workers", "pages/s", "pages/s/core"],
        [
            [
                stage,
                _fmt(t.get("seconds")),
                _fmt(t.get("pages")),
                _fmt(t.get("workers")),
                _fmt(t.get("pages_per_sec"), 1),
                _fmt(t.get("pages_per_sec_per_core"), 1),
            ]
            for stage, t in report["timings"].items()
        ],
    )
    ex = report.get("extract") or {}
    if ex.get("pages"):
        lines += [
            "",
            f"Extraction: {ex['pages']} pages in {_fmt(ex.get('seconds'))} s on "
            f"{ex.get('workers')} worker(s): {_fmt(ex.get('pages_per_sec_per_core'), 1)} "
            f"pages/s/core wall-clock, {_fmt(ex.get('pages_per_cpu_sec'), 1)} pages per "
            "CPU-second (parse included).",
        ]
    lines.append("")
    return "\n".join(lines)


def write_report(run: RunDir) -> dict[str, Any]:
    """Write ``report.json`` and ``report.md``; returns the report dict."""
    report = build_report(run)
    write_json(run.report_json, report)
    run.report_md.write_text(render_markdown(report), encoding="utf-8")
    return report
