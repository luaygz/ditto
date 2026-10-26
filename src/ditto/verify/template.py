"""Verify one template: score every candidate on every page, pick winners, fall back to
path-induced selectors, repair failing signature groups within the call cap, finalize."""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree

from ditto.analyze.models import Candidate, CandidateTable
from ditto.extract.runner import evaluate
from ditto.fingerprint.dom import clean_html
from ditto.llm.base import LLM, BudgetExceeded, CacheMiss, LLMError
from ditto.specgen.models import (
    FieldStats,
    KeyValueField,
    ListField,
    RecordListField,
    ScalarField,
    Spec,
    SpecField,
)
from ditto.specgen.typecheck import infer_value_type
from ditto.verify.fallback import combined_selectors, induced_field
from ditto.verify.models import (
    CandidateOrigin,
    CandidateScore,
    FailureKind,
    FieldReport,
    PageFailure,
    RepairRecord,
    VerifyReport,
)
from ditto.verify.pages import (
    PathProbe,
    Probe,
    RowsProbe,
    SelectorProbe,
    evaluate_probes,
    path_texts,
)
from ditto.verify.repair import (
    compress_page,
    describe_nodes,
    repair_prompt_id,
    repair_variants,
    request_repair,
)
from ditto.verify.scoring import (
    FAILURE_NAMES,
    Expectation,
    Kind,
    Obs,
    choose_winner,
    effective,
    fallback_indices,
    fill_agreement,
    page_failures,
    score_candidate,
    selection_reason,
    winner_key,
)

FAIL_SCORE = 0.05  # every candidate below this -> path-induced fallback
FAILURE_TOLERANCE = 0.01  # failure rate above this -> repair target, then degraded
OPTIONAL_BELOW = 0.95
ROW_SAMPLE = 200
MAX_FAILURE_EXAMPLES = 10

FieldItem = tuple[str, SpecField, Sequence[CandidateOrigin], bool]


def kind_of(field: SpecField) -> Kind:
    if field.type == "scalar":
        return "scalar"
    if field.type == "list":
        return "list"
    return "rows"


def selectors_of(field: SpecField) -> list[tuple[str, str]]:
    """``(role, xpath)`` of the field's competing selectors, in index order."""
    if isinstance(field, (ScalarField, ListField)):
        return [(f"candidates[{i}]", x) for i, x in enumerate(field.candidates)]
    if isinstance(field, RecordListField):
        alts = [(f"container_alternates[{i}]", x) for i, x in enumerate(field.container_alternates)]
        return [("container", field.container), *alts]
    alts = [(f"pairs_alternates[{i}]", x) for i, x in enumerate(field.pairs_alternates)]
    return [("pairs", field.pairs), *alts]


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _failure_kind(code: int) -> FailureKind:
    name = FAILURE_NAMES[code]
    if name == "missing":
        return "missing"
    return "multiple" if name == "multiple" else "type"


@dataclass
class FieldPlan:
    """A field variant scored on every page of the template."""

    name: str
    field: SpecField
    kind: Kind
    value_type: str | None
    exp: Expectation
    obs: list[Obs]
    scores: list[CandidateScore]
    winner: int
    order: list[int]
    eff: Obs
    codes: np.ndarray[Any, np.dtype[np.uint8]]
    path_induced: bool = False

    @property
    def failures(self) -> int:
        return int((self.codes != 0).sum())

    @property
    def denominator(self) -> int:
        if self.exp.mask is not None and self.exp.known:
            return int(self.exp.mask.sum())
        return int(self.codes.shape[0])

    @property
    def failure_rate(self) -> float:
        return self.failures / self.denominator if self.denominator else 0.0

    @property
    def winner_score(self) -> float:
        return self.scores[self.winner].score

    @property
    def presence(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return self.eff.values > 0


def better(new: FieldPlan, old: FieldPlan) -> bool:
    """A variant replaces the current field only if it fails on fewer pages (or equally many
    with a higher winning score)."""
    if new.failures != old.failures:
        return new.failures < old.failures
    return new.winner_score > old.winner_score + 1e-9


@dataclass
class TemplateOutcome:
    """Final state of one verified template (what merge/split and the writers need)."""

    spec: Spec
    report: VerifyReport
    page_ids: np.ndarray[Any, np.dtype[np.int64]]
    presence: dict[str, np.ndarray[Any, np.dtype[np.bool_]]]
    table: CandidateTable | None


class TemplateVerifier:
    """Verification state for one template over a fixed set of pages."""

    def __init__(
        self,
        run_root: Path,
        template_id: str,
        page_ids: Sequence[int],
        table: CandidateTable | None,
        group_of: Mapping[int, int],
        *,
        workers: int | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.template_id = template_id
        ids = sorted(int(p) for p in page_ids)
        self.page_ids = np.asarray(ids, np.int64)
        self.table = table
        self.groups = np.asarray([group_of.get(p, -1) for p in ids], np.int64)
        self.workers = workers
        rng = random.Random(f"rows:{template_id}")
        self.sampled = frozenset(ids if len(ids) <= ROW_SAMPLE else rng.sample(ids, ROW_SAMPLE))
        self.path_counts: dict[str, np.ndarray[Any, np.dtype[np.int32]]] = {}
        self.llm_seconds = 0.0
        self.budget_exhausted = False
        self._by_id = {c.id: c for c in table.candidates} if table else {}
        self._tables = {t.id: t for t in table.tables} if table else {}

    @property
    def n_pages(self) -> int:
        return int(self.page_ids.shape[0])

    # -- analysis expectations ----------------------------------------------------------------

    def sources(self, field: SpecField) -> list[Candidate]:
        """Analysis candidates the field maps to (a table id stands for all its cells)."""
        out: list[Candidate] = []
        for cid in field.source_candidates:
            if cid in self._by_id:
                out.append(self._by_id[cid])
            elif cid in self._tables and self.table is not None:
                out.extend(c for c in self.table.candidates if c.table_id == cid)
        return out

    def expectation(self, field: SpecField) -> Expectation:
        cands = self.sources(field)
        if not cands:
            return Expectation()
        p = max(c.p for c in cands)
        counts = np.zeros(self.n_pages, np.int32)
        for c in cands:
            if c.path in self.path_counts:
                counts += self.path_counts[c.path]
        mask = counts > 0
        if not mask.any():
            return Expectation(analysis_p=p)
        return Expectation(mask, counts, p)

    def value_type(self, field: SpecField) -> str | None:
        """The type to check: the field's ``type_check``, else the analysis value type of its
        source candidates, else the type their examples suggest. ``text`` means no check."""
        if isinstance(field, (RecordListField, KeyValueField)):
            return None
        if field.type_check:
            return None if field.type_check == "text" else field.type_check
        cands = self.sources(field)
        typed = [c.value_type for c in cands if c.value_type]
        if typed:
            specific = [t for t in typed if t != "text"]
            return Counter(specific).most_common(1)[0][0] if specific else None
        inferred = infer_value_type([e for c in cands for e in c.examples])
        return None if inferred in (None, "text") else inferred

    def column_types(self, field: RecordListField) -> tuple[tuple[str, str], ...]:
        by_col: dict[str, str] = {}
        for c in self.sources(field):
            if c.column and c.value_type and c.value_type != "text":
                by_col.setdefault(_norm_name(c.column), c.value_type)
            table = self._tables.get(c.table_id or "")
            for col in table.columns if table else []:
                cand = self._by_id.get(col.candidate_id)
                if cand and cand.value_type and cand.value_type != "text":
                    by_col.setdefault(_norm_name(col.name), cand.value_type)
        return tuple(
            (name, by_col[_norm_name(name)]) for name in field.columns if _norm_name(name) in by_col
        )

    def probes(self, field: SpecField, value_type: str | None) -> list[Probe]:
        if isinstance(field, (ScalarField, ListField)):
            plural = isinstance(field, ListField)
            transforms = tuple(field.transforms)
            return [SelectorProbe(x, transforms, value_type, plural) for x in field.candidates]
        if isinstance(field, RecordListField):
            col_types = self.column_types(field)
            return [
                RowsProbe(field.model_copy(update={"container": x}), col_types)
                for _, x in selectors_of(field)
            ]
        return [RowsProbe(field.model_copy(update={"pairs": x})) for _, x in selectors_of(field)]

    # -- scoring ------------------------------------------------------------------------------

    def score_fields(self, items: Sequence[FieldItem]) -> list[FieldPlan]:
        """Score field variants ``(name, field, origins, path_induced)`` in one page pass."""
        needed = {c.path for _, f, _, _ in items for c in self.sources(f)}
        paths = sorted(needed - set(self.path_counts))
        probes: list[Probe] = [PathProbe(p) for p in paths]
        spans: list[tuple[int, int]] = []
        vtypes: list[str | None] = []
        for _, f, _, _ in items:
            vt = self.value_type(f)
            ps = self.probes(f, vt)
            vtypes.append(vt)
            spans.append((len(probes), len(probes) + len(ps)))
            probes.extend(ps)
        m = evaluate_probes(
            self.run_root,
            self.page_ids.tolist(),
            probes,
            sampled=self.sampled,
            workers=self.workers,
        )
        for j, path in enumerate(paths):
            self.path_counts[path] = m.nodes[:, j].copy()
        plans = []
        for (name, f, origins, induced), (a, b), vt in zip(items, spans, vtypes, strict=True):
            obs = [Obs.column(m, j) for j in range(a, b)]
            plans.append(self.plan(name, f, origins, obs, vt, induced))
        return plans

    def plan(
        self,
        name: str,
        field: SpecField,
        origins: Sequence[CandidateOrigin],
        obs: list[Obs],
        value_type: str | None,
        path_induced: bool = False,
    ) -> FieldPlan:
        kind = kind_of(field)
        exp = self.expectation(field)
        scores = [
            score_candidate(
                kind,
                o,
                exp,
                index=i,
                role=role,
                xpath=x,
                origin=origins[i] if i < len(origins) else "llm",
            )
            for i, ((role, x), o) in enumerate(zip(selectors_of(field), obs, strict=True))
        ]
        fill_agreement(scores, obs)
        w = choose_winner(scores)
        order = [w]
        eff = effective(obs, order)
        codes = page_failures(kind, eff, exp)
        for i in fallback_indices(scores, w) if kind != "rows" else []:
            # a fallback must not add failures, nor fill pages where the analysis says the
            # field is absent
            trial = effective(obs, [*order, i])
            trial_codes = page_failures(kind, trial, exp)
            added = (trial.values > 0) & (eff.values == 0)
            spurious = exp.mask is not None and exp.known and bool((added & ~exp.mask).any())
            if not spurious and (trial_codes != 0).sum() <= (codes != 0).sum():
                order.append(i)
                eff, codes = trial, trial_codes
        return FieldPlan(
            name, field, kind, value_type, exp, obs, scores, w, order, eff, codes, path_induced
        )

    def score_spec(self, spec: Spec, induced: frozenset[str] = frozenset()) -> dict[str, FieldPlan]:
        items: list[FieldItem] = []
        for name, f in spec.fields.items():
            origin: CandidateOrigin = "path_induced" if name in induced else "llm"
            items.append((name, f, [origin] * len(selectors_of(f)), name in induced))
        return {p.name: p for p in self.score_fields(items)}

    # -- path-induced fallback ------------------------------------------------------------------

    def induced_variant(self, plan: FieldPlan) -> tuple[SpecField, list[CandidateOrigin]] | None:
        f = plan.field
        if isinstance(f, (ScalarField, ListField)):
            induced = combined_selectors(self.sources(f))
            if not induced:
                return None
            best = plan.scores[plan.winner]
            keep = [best.xpath] if best.score > 0 else []
            sels = list(dict.fromkeys([*keep, *induced]))[:3]
            if len(sels) < 2:
                return None
            origins: list[CandidateOrigin] = ["llm" if x in keep else "path_induced" for x in sels]
            update = {"candidates": sels, "selected": 0, "fallbacks": []}
            return f.model_copy(update=update), origins
        if self.table is None:
            return None
        new = induced_field(
            f.source_candidates, self.table, field_type=f.type, description=f.description
        )
        if new is None:
            return None
        return new, ["path_induced"] * len(selectors_of(new))

    def induce(self, plans: dict[str, FieldPlan]) -> list[str]:
        """Add path-induced selectors to fields whose every candidate fails; returns the
        fields that improved."""
        items: list[FieldItem] = []
        for name, plan in plans.items():
            if max(s.score for s in plan.scores) >= FAIL_SCORE:
                continue
            variant = self.induced_variant(plan)
            if variant is not None:
                items.append((name, variant[0], variant[1], True))
        improved = []
        for new in self.score_fields(items) if items else []:
            if better(new, plans[new.name]):
                plans[new.name] = new
                improved.append(new.name)
        return improved

    # -- repair -------------------------------------------------------------------------------

    def _failure_text(
        self, plan: FieldPlan, idx: int, tree: etree._Element, note: str | None
    ) -> str:
        f = plan.field
        win = plan.scores[plan.winner]
        lines = [f"- {plan.name} ({f.type}; value type {plan.value_type or 'any'})"]
        lines.append(f"  selected {win.role}: {win.xpath}")
        code = int(plan.codes[idx])
        if isinstance(f, (ScalarField, ListField)):
            got = describe_nodes(evaluate(win.xpath, tree))
        else:
            got = f"{len(evaluate(win.xpath, tree))} row node(s)"
        if code:
            lines.append(f"  on this page: {FAILURE_NAMES[code]} — returned {got}")
        else:
            lines.append(f"  on this page it returned {got}")
        kinds = Counter(FAILURE_NAMES[int(c)] for c in plan.codes[plan.codes != 0])
        detail = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none"
        lines.append(
            f"  across the template: {plan.failures} failing pages ({detail}) of "
            f"{plan.denominator} where the field is expected"
        )
        if note:
            lines.append(f"  reviewer rejected the extracted values: {note}")
        return "\n".join(lines)

    def _expected_text(self, plan: FieldPlan, tree: etree._Element) -> str:
        cands = self.sources(plan.field)
        if not cands:
            return f"- {plan.name}: no analysis candidate is mapped to this field"
        parts = []
        for c in cands[:6]:
            here = path_texts(c.path, tree)
            shown = ", ".join(repr(v[:120]) for v in here) or "(absent on this page)"
            examples = ", ".join(repr(e[:80]) for e in c.examples) or "none"
            parts.append(
                f"- {plan.name} via {c.id} `{c.path}`: on this page {shown}; "
                f"examples from other pages: {examples}"
            )
        return "\n".join(parts)

    def ask(
        self,
        llm: LLM,
        spec: Spec,
        plans: Mapping[str, FieldPlan],
        names: Sequence[str],
        page_id: int,
        situation: str,
        notes: Mapping[str, str] | None = None,
    ) -> dict[str, SpecField]:
        """Build and send one repair request for ``names`` using page ``page_id``."""
        from ditto.ingest import read_html

        idx = int(np.searchsorted(self.page_ids, page_id))
        page_html = read_html(self.run_root, page_id)
        tree = clean_html(page_html)
        failures = "\n".join(
            self._failure_text(plans[n], idx, tree, (notes or {}).get(n)) for n in names
        )
        expected = "\n".join(self._expected_text(plans[n], tree) for n in names)
        start = time.perf_counter()
        try:
            page = compress_page(page_html, page_id, self.table)
            return request_repair(
                llm,
                spec,
                names,
                situation=situation,
                failures=failures,
                expected=expected,
                page=page,
                page_id=page_id,
            )
        finally:
            self.llm_seconds += time.perf_counter() - start

    def accept_repairs(
        self,
        plans: dict[str, FieldPlan],
        repaired: Mapping[str, SpecField],
        *,
        combine: bool = True,
        require_better: bool = True,
    ) -> list[str]:
        """Score the repaired fields (and their combinations with the old winners) and keep
        the ones that beat the current field. With ``require_better=False`` (reviewer
        rejections) a repaired field that matches anything replaces the rejected one.
        Returns the accepted field names."""
        items: list[FieldItem] = []
        for name, new in repaired.items():
            old = plans[name]
            old_origins = {s.xpath: s.origin for s in old.scores}
            winner_xpath = old.scores[old.winner].xpath
            variants = repair_variants(old.field, new, winner_xpath) if combine else [new]
            for v in variants:
                origins: list[CandidateOrigin] = [
                    old_origins.get(x, "repair") for _, x in selectors_of(v)
                ]
                items.append((name, v, origins, old.path_induced))
        accepted: list[str] = []
        for new_plan in self.score_fields(items) if items else []:
            current = plans[new_plan.name]
            first = new_plan.name not in accepted
            if require_better or not first:
                ok = better(new_plan, current)
            else:
                ok = new_plan.winner_score > 0
            if ok:
                plans[new_plan.name] = new_plan
                if first:
                    accepted.append(new_plan.name)
        return accepted

    def repair(
        self,
        spec: Spec,
        plans: dict[str, FieldPlan],
        llm: LLM,
        max_calls: int,
        report: VerifyReport,
    ) -> int:
        """Group failing pages by signature group; one repair call per group (most failing
        pages first) up to ``max_calls``. A budget refusal stops repairs and sets
        ``budget_exhausted`` so the run stops calling the LLM."""
        calls = 0
        tried: set[int] = set()
        while calls < max_calls:
            failing = {n: p for n, p in plans.items() if p.failure_rate > FAILURE_TOLERANCE}
            if not failing:
                break
            per_group: Counter[int] = Counter()
            for p in failing.values():
                per_group.update(self.groups[p.codes != 0].tolist())
            choices = [(cnt, g) for g, cnt in per_group.items() if g not in tried]
            if not choices:
                break
            _, group = max(choices, key=lambda t: (t[0], -t[1]))
            tried.add(group)
            in_group = self.groups == group
            names = [n for n, p in failing.items() if ((p.codes != 0) & in_group).any()]
            fail_counts = np.zeros(self.n_pages, np.int32)
            for n in names:
                fail_counts += (plans[n].codes != 0).astype(np.int32)
            idx = int(np.argmax(np.where(in_group, fail_counts, -1)))
            page_id = int(self.page_ids[idx])
            failing_here = int((in_group & (fail_counts > 0)).sum())
            record = RepairRecord(
                call=calls + 1,
                signature_group=group if group >= 0 else None,
                page_id=page_id,
                fields=names,
                failures_before={n: plans[n].failures for n in names},
            )
            report.repairs.append(record)
            situation = (
                f"Signature group {group}: {failing_here} failing pages of "
                f"{int(in_group.sum())} in the group; {self.n_pages} pages in the template."
            )
            try:
                repaired = self.ask(
                    llm, self.current_spec(spec, plans), plans, names, page_id, situation
                )
            except BudgetExceeded as exc:
                record.error = f"budget exceeded: {exc}"
                report.notes.append("repairs stopped: LLM budget exceeded")
                self.budget_exhausted = True
                break
            except CacheMiss as exc:
                record.error = f"no recorded response (offline): {exc}"
                break
            except LLMError as exc:
                calls += 1
                record.error = f"{type(exc).__name__}: {exc}"
                continue
            calls += 1
            record.accepted = self.accept_repairs(plans, repaired)
            record.failures_after = {n: plans[n].failures for n in names}
        return calls

    # -- finalize -----------------------------------------------------------------------------

    @staticmethod
    def current_spec(spec: Spec, plans: Mapping[str, FieldPlan]) -> Spec:
        fields = {n: plans[n].field if n in plans else f for n, f in spec.fields.items()}
        return spec.model_copy(update={"fields": fields})

    def final_field(self, plan: FieldPlan, degraded: bool) -> SpecField:
        f = plan.field
        eff = score_candidate(plan.kind, plan.eff, plan.exp)
        expected = plan.exp.expected_presence
        stats = FieldStats.model_validate(
            {
                "presence_rate": eff.presence_rate,
                "expected_presence": None if expected is None else round(expected, 4),
                "multiplicity": eff.multiplicity,
                "type_check_pass": eff.type_check_pass,
                "rows_min": eff.rows_min,
                "rows_max": eff.rows_max,
                "keys_seen": eff.keys_seen,
                "pages_scored": self.n_pages,
                "failure_rate": round(plan.failure_rate, 4),
            }
        )
        reason = selection_reason(plan.scores, plan.winner, plan.value_type)
        if plan.path_induced:
            reason += "; path-induced fallback selectors (every LLM candidate failed)"
        if degraded:
            reason += (
                f"; degraded: fails on {plan.failure_rate:.1%} of the {plan.denominator} pages "
                "where the field is expected"
            )
        update: dict[str, Any] = {
            "stats": stats,
            "optional": expected < OPTIONAL_BELOW if expected is not None else f.optional,
            "degraded": degraded,
            "selection_reason": reason,
        }
        if isinstance(f, (ScalarField, ListField)):
            update["selected"] = plan.winner
            update["fallbacks"] = plan.order[1:]
        else:
            sels = [x for _, x in selectors_of(f)]
            ranked = sorted(range(len(sels)), key=lambda i: winner_key(plan.scores[i]))
            alternates = [sels[i] for i in ranked if i != plan.winner][:2]
            if isinstance(f, RecordListField):
                update["container"] = sels[plan.winner]
                update["container_alternates"] = alternates
            else:
                update["pairs"] = sels[plan.winner]
                update["pairs_alternates"] = alternates
        return f.model_copy(update=update)

    def field_report(self, plan: FieldPlan, final: SpecField) -> FieldReport:
        failing = plan.codes != 0
        kinds = Counter(FAILURE_NAMES[int(c)] for c in plan.codes[failing])
        groups = Counter(int(g) for g in self.groups[failing])
        examples = [
            PageFailure(
                page_id=int(self.page_ids[i]),
                signature_group=int(self.groups[i]) if self.groups[i] >= 0 else None,
                kind=_failure_kind(int(plan.codes[i])),
            )
            for i in np.flatnonzero(failing)[:MAX_FAILURE_EXAMPLES]
        ]
        expected = plan.exp.expected_presence
        return FieldReport(
            name=plan.name,
            type=final.type,
            selected=plan.winner,
            selected_xpath=plan.scores[plan.winner].xpath,
            selection_reason=final.selection_reason,
            value_type=plan.value_type,
            expected_presence=None if expected is None else round(expected, 4),
            analysis_p=plan.exp.analysis_p,
            optional=final.optional,
            degraded=final.degraded,
            path_induced=plan.path_induced,
            candidates=plan.scores,
            failures=plan.failures,
            failure_rate=round(plan.failure_rate, 4),
            failures_by_kind=dict(kinds),
            failures_by_group={str(g): n for g, n in groups.most_common()},
            failure_examples=examples,
        )

    def finalize(
        self,
        spec: Spec,
        plans: Mapping[str, FieldPlan],
        report: VerifyReport,
        *,
        repair_calls: int = 0,
    ) -> TemplateOutcome:
        """Write winners, fallbacks, stats, optional/degraded into a final spec and fill the
        report's field section."""
        fields: dict[str, SpecField] = {}
        report.fields = {}
        for name, plan in plans.items():
            degraded = plan.failure_rate > FAILURE_TOLERANCE
            fields[name] = self.final_field(plan, degraded)
            report.fields[name] = self.field_report(plan, fields[name])
        gen = spec.generated_by.model_copy(
            update={"repair_calls": spec.generated_by.repair_calls + repair_calls}
        )
        if repair_calls:
            gen.repair_prompt_version = repair_prompt_id()
        draft = spec.model_copy(update={"fields": fields, "status": "final", "generated_by": gen})
        final = Spec.model_validate(draft.model_dump())
        report.n_pages = self.n_pages
        report.not_present = dict(final.not_present)
        report.repair_calls += repair_calls
        report.degraded = [n for n, f in final.fields.items() if f.degraded]
        report.llm_seconds = round(report.llm_seconds + self.llm_seconds, 3)
        self.llm_seconds = 0.0
        return TemplateOutcome(
            spec=final,
            report=report,
            page_ids=self.page_ids,
            presence={n: p.presence for n, p in plans.items()},
            table=self.table,
        )

    def verify(
        self,
        spec: Spec,
        report: VerifyReport,
        *,
        llm: LLM | None = None,
        max_repair_calls: int = 0,
        induced: frozenset[str] = frozenset(),
    ) -> TemplateOutcome:
        """Score, fall back, repair (when ``llm`` is given), finalize."""
        plans = self.score_spec(spec, induced)
        improved = self.induce(plans)
        if improved:
            report.notes.append(f"path-induced fallback used for: {', '.join(improved)}")
        calls = 0
        if llm is not None and max_repair_calls > 0:
            calls = self.repair(spec, plans, llm, max_repair_calls, report)
        return self.finalize(spec, plans, report, repair_calls=calls)
