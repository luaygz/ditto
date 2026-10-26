"""Stage 7 entry points: ``verify_run`` (score, fall back, repair, merge/split, finalize) and
``repair_rejected`` (send fields a reviewer rejected back through repair)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ditto.analyze.models import CandidateTable
from ditto.cluster.merge_split import (
    MERGE_SAMPLE,
    MergePlan,
    apply_merges,
    apply_splits,
    merge_samples,
    merge_spec_fields,
    plan_merges,
    plan_splits,
    spec_presence,
)
from ditto.cluster.models import Cluster, Clusters
from ditto.llm.base import LLM, BudgetExceeded, CacheMiss, LLMError
from ditto.parallel import default_workers
from ditto.run import RunDir, read_json
from ditto.specgen.models import GeneratedBy, Spec, SpecField
from ditto.verify.fallback import induced_field
from ditto.verify.models import RepairRecord, VerifyReport
from ditto.verify.template import TemplateOutcome, TemplateVerifier

FALLBACK_MODEL = "none (path-induced)"


# -- inputs -----------------------------------------------------------------------------------


def load_generation(run: RunDir) -> dict[str, dict[str, Any]]:
    """Per-template entries of ``generation.json`` (empty when generation has not run)."""
    if not run.generation_json.exists():
        return {}
    data = read_json(run.generation_json)
    if not isinstance(data, dict):
        return {}
    templates = data.get("templates", data)
    if not isinstance(templates, dict):
        return {}
    return {tid: e for tid, e in templates.items() if isinstance(e, dict)}


def failed_fields(entry: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``failed_fields`` of a generation entry as ``name -> info`` (also accepts a list of
    names or of ``{"name": ...}`` objects)."""
    raw = (entry or {}).get("failed_fields") or {}
    if isinstance(raw, dict):
        return {str(k): v if isinstance(v, dict) else {} for k, v in raw.items()}
    out: dict[str, dict[str, Any]] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            out[item] = {}
        elif isinstance(item, dict) and item.get("name"):
            out[str(item["name"])] = item
    return out


def load_table(run: RunDir, cluster: Cluster) -> CandidateTable | None:
    """The analysis table of a cluster (a split cluster uses its parent's)."""
    for tid in (cluster.template_id, cluster.split_from):
        if tid and run.candidates(tid).exists():
            return CandidateTable.load(run.candidates(tid))
    return None


def newest_spec(run: RunDir, template_id: str) -> tuple[Spec | None, str]:
    """``specs/<tid>.json`` when it is at least as new as the draft (a verified or user-edited
    spec is re-verified as is), else the draft."""
    draft, final = run.draft_spec(template_id), run.spec(template_id)
    if final.exists() and (
        not draft.exists() or final.stat().st_mtime_ns >= draft.stat().st_mtime_ns
    ):
        return Spec.load(final), "final"
    if draft.exists():
        return Spec.load(draft), "draft"
    return None, ""


@dataclass
class StartingSpec:
    spec: Spec
    source: str
    induced: frozenset[str] = frozenset()
    notes: list[str] = field(default_factory=list)


def starting_spec(
    run: RunDir,
    cluster: Cluster,
    entry: Mapping[str, Any] | None,
    table: CandidateTable | None,
) -> StartingSpec | None:
    """The spec to verify, with fields generation failed on built from path-induced
    selectors. None when the template has nothing to verify."""
    tid = cluster.template_id
    spec, source = newest_spec(run, tid)
    if spec is None and cluster.split_from:
        parent, _ = newest_spec(run, cluster.split_from)
        if parent is not None:
            spec, source = parent.model_copy(update={"template_id": tid}), "split_parent"
    built: dict[str, SpecField] = {}
    not_present: dict[str, str] = {}
    notes: list[str] = []
    for name, info in failed_fields(entry).items():
        if spec is not None and name in spec.fields:
            continue
        new = None
        if table is not None:
            new = induced_field(
                [str(x) for x in info.get("source_candidates") or []],
                table,
                field_type=info.get("type"),
                description=info.get("description"),
                type_check=info.get("type_check"),
            )
        if new is None:
            errors = info.get("errors") or []
            detail = f" ({errors[0]})" if errors else ""
            not_present[name] = (
                f"generation failed{detail} and no path-induced selector could be built"
            )
        else:
            built[name] = new
    if built:
        notes.append(f"path-induced fields for failed generation: {', '.join(sorted(built))}")
    if spec is None:
        if not built:
            return None
        spec = Spec(
            template_id=tid,
            site=run.site,
            generated_by=GeneratedBy(model=FALLBACK_MODEL, prompt_version="none", mode="fallback"),
            fields=built,
            not_present=not_present,
        )
        return StartingSpec(spec, "fallback", frozenset(built), notes)
    if built or not_present:
        spec = spec.model_copy(
            update={
                "fields": {**spec.fields, **built},
                "not_present": {**spec.not_present, **not_present},
            }
        )
    return StartingSpec(spec, source, frozenset(built), notes)


def group_index(clusters: Clusters) -> dict[int, int]:
    return {pid: g.id for g in clusters.signature_groups for pid in g.page_ids}


def _url_lookup(run: RunDir) -> Callable[[int], str] | None:
    if not run.pages_db.exists():
        return None
    from ditto.ingest import PageStore

    return PageStore.open(run).url


# -- verify -----------------------------------------------------------------------------------


def _reverify(
    run: RunDir,
    clusters: Clusters,
    tid: str,
    spec: Spec,
    table: CandidateTable | None,
    report: VerifyReport,
    workers: int | None,
) -> TemplateOutcome:
    """Re-score a template on its (changed) pages without repair calls."""
    verifier = TemplateVerifier(
        run.root, tid, clusters.get(tid).page_ids, table, group_index(clusters), workers=workers
    )
    start = time.perf_counter()
    outcome = verifier.verify(spec, report)
    report.seconds = round(report.seconds + time.perf_counter() - start, 3)
    return outcome


def _apply_splits(
    run: RunDir,
    clusters: Clusters,
    outcomes: dict[str, TemplateOutcome],
    workers: int | None,
) -> tuple[Clusters, bool]:
    group_of = group_index(clusters)
    sig_hash = {g.id: g.sig_hash for g in clusters.signature_groups}
    plans = []
    for tid, out in outcomes.items():
        if out.report.status != "final":
            continue
        groups = [group_of.get(int(p), -1) for p in out.page_ids]
        plans += plan_splits(tid, out.page_ids.tolist(), groups, out.presence, sig_hash)
    if not plans:
        return clusters, False
    clusters = apply_splits(clusters, plans, url_of=_url_lookup(run))
    for plan in plans:
        parent = outcomes[plan.template_id]
        event = plan.event()
        child_spec = parent.spec.model_copy(update={"template_id": plan.new_template_id})
        child_report = VerifyReport(template_id=plan.new_template_id, spec_source="split_parent")
        for tid, spec, report in (
            (plan.template_id, parent.spec, parent.report),
            (plan.new_template_id, child_spec, child_report),
        ):
            report.merge_split.append(event)
            outcomes[tid] = _reverify(run, clusters, tid, spec, parent.table, report, workers)
    return clusters, True


def _apply_merges(
    run: RunDir,
    clusters: Clusters,
    outcomes: dict[str, TemplateOutcome],
    pool: set[str] | None,
    workers: int | None,
) -> tuple[Clusters, bool]:
    specs = {tid: o.spec for tid, o in outcomes.items() if o.report.status == "final"}
    if not specs:
        return clusters, False
    samples = merge_samples(clusters, templates=pool)
    for tid in specs:
        samples.setdefault(tid, clusters.sample(tid, MERGE_SAMPLE, seed=0))
    if len(samples) < 2:
        return clusters, False
    rates = spec_presence(run.root, specs, samples, workers=workers)
    plans = plan_merges(clusters, specs, rates, samples)
    if not plans:
        return clusters, False
    clusters = apply_merges(clusters, plans)
    by_target: dict[str, list[MergePlan]] = {}
    for plan in plans:
        by_target.setdefault(plan.target, []).append(plan)
    for a, merged in by_target.items():
        target = outcomes[a]
        spec = target.spec
        for plan in merged:
            target.report.merge_split.append(plan.event())
            absorbed = outcomes.get(plan.absorbed)
            if absorbed is None:
                continue
            spec, added = merge_spec_fields(spec, absorbed.spec, target.table, absorbed.table)
            if added:
                target.report.notes.append(
                    f"fields carried over from {plan.absorbed}: {', '.join(added)}"
                )
            absorbed.report.status = "merged"
            absorbed.report.merge_split.append(plan.event())
        outcomes[a] = _reverify(run, clusters, a, spec, target.table, target.report, workers)
    return clusters, True


def write_outcomes(run: RunDir, outcomes: Mapping[str, TemplateOutcome]) -> None:
    for tid, out in outcomes.items():
        if out.report.status == "merged":
            run.spec(tid).unlink(missing_ok=True)
        else:
            out.spec.save(run.spec(tid))
        out.report.save(run.verify_report(tid))


def verify_run(
    run: RunDir,
    *,
    max_repair_calls: int = 3,
    llm: LLM | None = None,
    workers: int | None = None,
    template_ids: Iterable[str] | None = None,
    merge_split: bool = True,
) -> dict[str, VerifyReport]:
    """Verify every template that has a draft (or final, or fallback-buildable) spec.

    Writes ``specs/<tid>.json`` (status final) and ``verify/<tid>.json``; updates
    ``clusters.json`` when the merge/split rules fire. ``llm=None`` disables repair calls
    (fields still failing are marked degraded)."""
    t0 = time.perf_counter()
    clusters = Clusters.load(run.clusters_json)
    generation = load_generation(run)
    group_of = group_index(clusters)
    wanted = set(template_ids) if template_ids is not None else None
    outcomes: dict[str, TemplateOutcome] = {}
    pages = 0
    active = llm
    for cluster in clusters.templates(include_tail=True):
        tid = cluster.template_id
        if wanted is not None and tid not in wanted:
            continue
        table = load_table(run, cluster)
        start = starting_spec(run, cluster, generation.get(tid), table)
        if start is None:
            continue
        report = VerifyReport(template_id=tid, spec_source=start.source, notes=start.notes)
        verifier = TemplateVerifier(
            run.root, tid, cluster.page_ids, table, group_of, workers=workers
        )
        t = time.perf_counter()
        outcomes[tid] = verifier.verify(
            start.spec,
            report,
            llm=active,
            max_repair_calls=max_repair_calls,
            induced=start.induced,
        )
        report.seconds = round(time.perf_counter() - t, 3)
        pages += verifier.n_pages
        if verifier.budget_exhausted:
            active = None
    changed = False
    if merge_split and outcomes:
        clusters, split = _apply_splits(run, clusters, outcomes, workers)
        pool = None if wanted is None else set(outcomes)
        clusters, merged = _apply_merges(run, clusters, outcomes, pool, workers)
        changed = split or merged
    if changed:
        clusters.save(run.clusters_json)
    write_outcomes(run, outcomes)
    llm_seconds = sum(o.report.llm_seconds for o in outcomes.values())
    seconds = max(time.perf_counter() - t0 - llm_seconds, 1e-9)
    run.record_timing("verify", seconds, pages, workers or default_workers())
    return {tid: o.report for tid, o in outcomes.items()}


# -- review rejections ------------------------------------------------------------------------


def repair_rejected(
    run: RunDir,
    rejections: Mapping[str, Mapping[str, str]],
    *,
    llm: LLM | None = None,
    workers: int | None = None,
) -> dict[str, VerifyReport]:
    """Send fields a reviewer rejected (``template -> field -> note``) back through repair:
    one call per template with every rejected field and its note, then re-verify and rewrite
    ``specs/<tid>.json`` and ``verify/<tid>.json``. Fields that cannot be fixed stay as they
    were; every note is recorded under ``rejections`` in the verify report."""
    clusters = Clusters.load(run.clusters_json)
    group_of = group_index(clusters)
    out: dict[str, VerifyReport] = {}
    for tid, notes in rejections.items():
        if not run.spec(tid).exists() or tid not in {c.template_id for c in clusters.clusters}:
            continue
        spec = Spec.load(run.spec(tid))
        cluster = clusters.get(tid)
        table = load_table(run, cluster)
        path = run.verify_report(tid)
        report = (
            VerifyReport.load(path)
            if path.exists()
            else VerifyReport(template_id=tid, spec_source="final")
        )
        verifier = TemplateVerifier(
            run.root, tid, cluster.page_ids, table, group_of, workers=workers
        )
        plans = verifier.score_spec(spec)
        names = [n for n in notes if n in plans]
        accepted: list[str] = []
        error: str | None = None if llm is not None else "no LLM available for repair"
        calls = 0
        if names and llm is not None:
            present = np.zeros(verifier.n_pages, np.int32)
            for n in names:
                present += plans[n].presence
            idx = int(present.argmax()) if verifier.n_pages else 0
            page_id = int(verifier.page_ids[idx])
            record = RepairRecord(
                call=report.repair_calls + 1,
                signature_group=int(verifier.groups[idx]) if verifier.groups[idx] >= 0 else None,
                page_id=page_id,
                fields=names,
                failures_before={n: plans[n].failures for n in names},
                note="; ".join(f"{n}: {notes[n]}" for n in names),
            )
            report.repairs.append(record)
            situation = (
                "A reviewer rejected the values extracted for the fields below; their notes "
                "say what is wrong. Write selectors that extract the right values."
            )
            try:
                repaired = verifier.ask(llm, spec, plans, names, page_id, situation, notes)
            except (BudgetExceeded, CacheMiss) as exc:
                error = record.error = f"{type(exc).__name__}: {exc}"
            except LLMError as exc:
                calls = 1
                error = record.error = f"{type(exc).__name__}: {exc}"
            else:
                calls = 1
                accepted = verifier.accept_repairs(
                    plans, repaired, combine=False, require_better=False
                )
                record.accepted = accepted
                record.failures_after = {n: plans[n].failures for n in names}
        for n, note in notes.items():
            if n not in plans:
                reason = "field is not in the spec"
            elif n in accepted:
                reason = "repaired selectors accepted"
            else:
                reason = error or "the repair answer did not produce a matching selector"
            report.rejections[n] = {"note": note, "repaired": n in accepted, "reason": reason}
        outcome = verifier.finalize(spec, plans, report, repair_calls=calls)
        write_outcomes(run, {tid: outcome})
        out[tid] = outcome.report
    return out
