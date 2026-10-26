"""Path-induced fallback selectors (no LLM).

When every LLM candidate of a field fails, or generation could not produce a field at all,
verification builds anchored XPaths from the analysis candidate's own structural path
(``html > body.page > dl.profile > dd > a @href``):

* variant A — the exact child chain from the deepest anchored step (id, canonical class,
  itemprop/data-testid/role) down to the node: ``//dl[...profile...]/dd/a/@href``;
* variant B — the same anchor with a descendant jump to the node:
  ``//dl[...profile...]//a/@href`` (or, when the node itself is the anchor, the next anchored
  ancestor / the parent tag in front of it);
* a label variant (``//*[normalize-space()='Email']/following-sibling::dd[1]``) only when A
  and B do not give two distinct selectors.

Every selector returned passes ``lint_selector``; unanchored paths whose last tag is generic
yield nothing rather than a positional path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ditto.analyze.models import Candidate, CandidateTable, Table
from ditto.fingerprint.dom import Step, split_path
from ditto.specgen.lint import lint_selector
from ditto.specgen.models import (
    KeyValueField,
    ListField,
    RecordListField,
    ScalarField,
    SpecField,
)

MAX_LABEL_CHARS = 80
INDUCED_REASON = "path-induced fallback (built from the analysis path, no LLM)"


def _anchored(step: Step) -> bool:
    return bool(step.id or step.classes or step.attrs)


def _quote(text: str) -> str | None:
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    return None


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def induced_variants(
    steps: Sequence[Step], attr: str | None = None, label: str | None = None
) -> list[str]:
    """Anchored absolute selectors for the node at ``steps`` (lint-clean, deduplicated)."""
    n = len(steps)
    if n == 0:
        return []
    anchors = [i for i in range(1, n) if _anchored(steps[i])]
    variants: list[str] = []
    if anchors:
        k = anchors[-1]
        variants.append("//" + "/".join(s.xpath() for s in steps[k:]))
        if k < n - 1:
            variants.append(f"//{steps[k].xpath()}//{steps[-1].xpath()}")
        elif len(anchors) >= 2:
            variants.append(f"//{steps[anchors[-2]].xpath()}//{steps[k].xpath()}")
        elif n >= 2:
            variants.append(f"//{steps[n - 2].xpath()}/{steps[k].xpath()}")
    else:
        variants.append("//" + "/".join(s.xpath() for s in steps[max(1, n - 3) :]))
    suffix = f"/@{attr}" if attr else ""
    out = _dedupe([v + suffix for v in variants if not lint_selector(v + suffix)])
    if len(out) < 2 and label and len(label) <= MAX_LABEL_CHARS:
        quoted = _quote(label)
        if quoted is not None:
            sel = f"//*[normalize-space()={quoted}]/following-sibling::{steps[-1].xpath()}[1]"
            sel += suffix
            if not lint_selector(sel) and sel not in out:
                out.append(sel)
    return out


def path_induced_selectors(candidate: Candidate) -> list[str]:
    """Anchored XPaths for an analysis candidate (2 variants when possible, at most 3)."""
    try:
        steps, attr = split_path(candidate.path)
    except ValueError:
        return []
    return induced_variants(steps, attr or candidate.attr, candidate.label_hint)[:3]


def combined_selectors(candidates: Sequence[Candidate]) -> list[str]:
    """Induced selectors covering several source candidates (structural variants of one
    field): variant i of every candidate joined with ``|``."""
    per = [s for s in (path_induced_selectors(c) for c in candidates) if s]
    if not per:
        return []
    if len(per) == 1:
        return per[0]
    width = max(len(p) for p in per)
    joined = [" | ".join(_dedupe([p[min(i, len(p) - 1)] for p in per])) for i in range(width)]
    return [s for s in _dedupe(joined) if not lint_selector(s)][:3]


def relative_selector(row: Sequence[Step], target: Sequence[Step], attr: str | None) -> str | None:
    """Selector for ``target`` relative to a row node at ``row``: a child chain when the target
    sits inside the row, ``following-sibling::x[1]`` when it is the row's sibling (dt/dd)."""
    r = len(row)
    if r == 0 or len(target) < r:
        return None
    if list(target[:r]) == list(row):
        rest = target[r:]
        expr = "." if not rest else "./" + "/".join(s.xpath() for s in rest)
    elif list(target[: r - 1]) == list(row[: r - 1]):
        expr = f"./following-sibling::{target[r - 1].xpath()}[1]"
        expr += "".join("/" + s.xpath() for s in target[r:])
    else:
        return None
    return f"{expr}/@{attr}" if attr else expr


def _header_column(name: str, cell: Step, rest: Sequence[Step], attr: str | None) -> str | None:
    quoted = _quote(name)
    if quoted is None or cell.tag not in ("td", "th"):
        return None
    expr = (
        f"./{cell.tag}[count(ancestor::table[1]//th[normalize-space()={quoted}]"
        f"/preceding-sibling::th)+1]"
    )
    expr += "".join("/" + s.xpath() for s in rest)
    return f"{expr}/@{attr}" if attr else expr


def _common(
    description: str | None, source: list[str], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "description": description,
        "source_candidates": source,
        "selection_reason": INDUCED_REASON,
    }
    out.update(extra or {})
    return out


def induced_record_list(
    table: Table, cands: CandidateTable, description: str | None, source: list[str]
) -> RecordListField | None:
    try:
        row_steps, _ = split_path(table.row_path)
    except ValueError:
        return None
    containers = induced_variants(row_steps)
    if not containers:
        return None
    by_header: dict[str, str] = {}
    by_path: dict[str, str] = {}
    for col in table.columns:
        try:
            cand = cands.by_id(col.candidate_id)
            steps, attr = split_path(cand.path)
        except (KeyError, ValueError):
            continue
        attr = attr or cand.attr
        rel = relative_selector(row_steps, steps, attr)
        if rel is None:
            continue
        by_path[col.name] = rel
        inside = list(steps[: len(row_steps)]) == list(row_steps) and len(steps) > len(row_steps)
        if col.header_source == "th" and inside:
            header = _header_column(
                col.name, steps[len(row_steps)], steps[len(row_steps) + 1 :], attr
            )
            if header is not None and not lint_selector(header, relative=True):
                by_header[col.name] = header
    if not by_path:
        return None
    use_header = len(by_header) == len(by_path)
    columns = by_header if use_header else by_path
    try:
        return RecordListField(
            container=containers[0],
            container_alternates=containers[1:3],
            columns=columns,
            column_source="header_row" if use_header else "llm",
            **_common(description, source),
        )
    except ValueError:
        return None


def induced_key_value(
    table: Table, cands: CandidateTable, description: str | None, source: list[str]
) -> KeyValueField | None:
    if not table.key_candidate or not table.value_candidate:
        return None
    try:
        row_steps, _ = split_path(table.row_path)
        key_c, value_c = cands.by_id(table.key_candidate), cands.by_id(table.value_candidate)
        key_steps, key_attr = split_path(key_c.path)
        value_steps, value_attr = split_path(value_c.path)
    except (KeyError, ValueError):
        return None
    pairs = induced_variants(row_steps)
    key = relative_selector(row_steps, key_steps, key_attr or key_c.attr)
    value = relative_selector(row_steps, value_steps, value_attr or value_c.attr)
    if not pairs or key is None or value is None:
        return None
    try:
        return KeyValueField(
            pairs=pairs[0],
            pairs_alternates=pairs[1:3],
            key=key,
            value=value,
            **_common(description, source),
        )
    except ValueError:
        return None


def induced_field(
    source_ids: Sequence[str],
    cands: CandidateTable,
    *,
    field_type: str | None = None,
    description: str | None = None,
    type_check: str | None = None,
) -> SpecField | None:
    """Build a whole field from analysis candidates (or a table id) with no LLM."""
    picked: list[Candidate] = []
    tables: list[Table] = []
    by_table = {t.id: t for t in cands.tables}
    for sid in source_ids:
        if sid in by_table:
            tables.append(by_table[sid])
            continue
        try:
            c = cands.by_id(sid)
        except KeyError:
            continue
        picked.append(c)
        if c.table_id and c.table_id in by_table:
            tables.append(by_table[c.table_id])
    source = list(source_ids)
    if field_type in ("record_list", "key_value") or (field_type is None and tables):
        wanted = field_type or tables[0].kind
        for t in tables:
            if t.kind == wanted == "record_list":
                return induced_record_list(t, cands, description, source)
            if t.kind == wanted == "key_value":
                return induced_key_value(t, cands, description, source)
        return None
    selectors = combined_selectors(picked)
    if len(selectors) < 2:
        return None
    plural = field_type == "list" or (
        field_type is None and any(c.cls == "list" or c.repeat_max > 1 for c in picked)
    )
    extra: dict[str, Any] = {"candidates": selectors[:3]}
    if type_check in ("email", "url", "number", "date", "price", "text"):
        extra["type_check"] = type_check
    try:
        if plural:
            return ListField(**_common(description, source, extra))
        return ScalarField(**_common(description, source, extra))
    except ValueError:
        return None
