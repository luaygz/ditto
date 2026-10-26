"""``cluster_report.md``: what the clustering found, in a form a person can check quickly."""

from __future__ import annotations

from collections.abc import Callable

from ditto.cluster.core import ClusterResult

MAX_TEMPLATE_ROWS = 200
MAX_CHROME_ROWS = 25
MAX_BAND_ROWS = 50


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_cluster_report(
    result: ClusterResult, path_of: Callable[[int], str] | None = None, site: str = ""
) -> str:
    """Markdown report. ``path_of`` maps a path hash to its string (for the chrome list)."""
    c, s = result.clusters, result.stats
    templates = c.templates()
    tails = [x for x in c.clusters if x.is_tail]
    tail_pages = sum(x.size for x in tails)
    group_tid = {g.id: g.template_id for g in c.signature_groups}
    lines = [
        f"# Cluster report{f': {site}' if site else ''}",
        "",
        f"- pages: {c.n_pages}",
        f"- signature groups: {s.n_groups} (distinct paths: {s.n_paths})",
        f"- templates (>= {c.min_cluster_size} pages): {len(templates)}, "
        f"covering {sum(t.size for t in templates)} pages",
        f"- tail clusters (< {c.min_cluster_size} pages): {len(tails)}, covering {tail_pages} pages",
        f"- chrome paths dropped (on > {c.params.get('chrome_df')} of pages): "
        f"{len(s.chrome_paths)}",
        f"- groups judged on full signatures when mostly chrome: {s.n_fallback_groups}",
        f"- LSH: {s.lsh_bands} bands x {s.lsh_rows} rows; {s.n_candidates} candidate pairs, "
        f"{s.n_exact} exact Jaccard checks, {s.strong_edges} confirming edges, "
        f"{len(s.band_merges)} band merges by URL pattern",
        "- parameters: " + ", ".join(f"{k}={v}" for k, v in c.params.items()),
        "- seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in s.seconds.items()),
        "",
        "## Templates",
        "",
        "| template | pages | share | groups | top URL pattern | example | nearest other (J) |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    n = max(c.n_pages, 1)
    for t in templates[:MAX_TEMPLATE_ROWS]:
        near = s.nearest.get(t.template_id)
        near_txt = f"{near[0]} ({near[1]:.2f})" if near else ""
        lines.append(
            f"| {t.template_id} | {t.size} | {t.size / n:.1%} | {len(t.signature_groups)} | "
            f"`{_cell(t.url_patterns[0] if t.url_patterns else '')}` | "
            f"{_cell(t.url_examples[0] if t.url_examples else '')} | {near_txt} |"
        )
    if len(templates) > MAX_TEMPLATE_ROWS:
        lines.append(f"| … {len(templates) - MAX_TEMPLATE_ROWS} more | | | | | | |")
    lines += ["", "## Band merges (Jaccard in the tie-break band, URL patterns agree)", ""]
    if s.band_merges:
        for edge, pattern in s.band_merges[:MAX_BAND_ROWS]:
            lines.append(
                f"- groups {edge.a} + {edge.b} -> {group_tid[edge.a]}: J={edge.jaccard:.3f}"
                f"{' (full signatures)' if edge.full_signature else ''}, pattern `{pattern}`"
            )
    else:
        lines.append("- none")
    if s.band_rejects:
        lines += ["", "## Band pairs kept apart (URL patterns differ or uninformative)", ""]
        for edge, pa, pb in s.band_rejects[:MAX_BAND_ROWS]:
            lines.append(
                f"- {group_tid[edge.a]} / {group_tid[edge.b]}: J={edge.jaccard:.3f}, "
                f"`{pa}` vs `{pb}`"
            )
    lines += ["", "## Chrome paths (sample)", ""]
    if s.chrome_paths:
        chrome = sorted(s.chrome_paths, key=lambda x: -x[1])[:MAX_CHROME_ROWS]
        for h, df in chrome:
            label = path_of(h) if path_of else f"{h:016x}"
            lines.append(f"- {df:.1%} `{_cell(label)}`")
    else:
        lines.append("- none")
    if tails:
        lines += ["", "## Tail clusters (largest first)", ""]
        for t in sorted(tails, key=lambda x: (-x.size, x.template_id))[:MAX_TEMPLATE_ROWS]:
            lines.append(
                f"- {t.template_id}: {t.size} pages, "
                f"`{_cell(t.url_patterns[0] if t.url_patterns else '')}`"
            )
    return "\n".join(lines) + "\n"
