"""Stage 9a: probe. Which template holds which requested field, before any spec is written.

``probe_run`` scores every candidate of every non-tail template against every requested field
(see :mod:`ditto.checkpoints.match`; no LLM), optionally resolves ambiguous fields with one
small-model call per template over the candidate table alone, and writes ``probe.json`` (full
detail) and ``probe.yaml`` (the editable proposal that ``confirm`` reads).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ditto.analyze.models import Candidate, CandidateTable, Table
from ditto.checkpoints._common import short
from ditto.checkpoints.match import (
    FieldContext,
    FieldMatch,
    FieldStatus,
    Prepared,
    Scored,
    match_template,
    prepare,
    score_by_id,
    score_candidate,
)
from ditto.cluster.models import Cluster, Clusters
from ditto.run import RunDir, read_json, write_json
from ditto.schema import ABSENT, FieldRequest, FieldsSchema

if TYPE_CHECKING:
    from ditto.llm.base import LLM
    from ditto.specgen import Prompt

PROMPT_NAME = "probe"
PROMPT_VERSION = 1
MAX_LLM_ROWS = 400  # candidate rows sent per probe call (most relevant first when trimmed)
NO_FIELD = "holds no requested field"

# field -> (mapping value, chosen id or "absent", reason)
Answers = dict[str, tuple[str | list[str], str, str]]


class CandidateInfo(BaseModel):
    """One ranked candidate (or whole table) for a field, as shown in probe.json/yaml."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    score: float
    use: str | list[str] = Field(description="What mapping.json gets when this is chosen.")
    cls: str | None = Field(None, alias="class")
    p: float | None = None
    label_hint: str | None = None
    value_type: str | None = None
    path: str | None = None
    key: str | None = Field(None, description="Key of a key/value table the field matched.")
    examples: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @classmethod
    def from_scored(cls, s: Scored) -> CandidateInfo:
        c: Candidate | None = s.candidate
        t: Table | None = s.table
        return cls(
            id=s.id,
            score=round(s.score, 3),
            use=s.use,
            cls=c.cls if c is not None else (f"{t.kind} table" if t is not None else None),
            p=round(c.p if c is not None else t.p if t is not None else 0.0, 3),
            label_hint=c.label_hint if c is not None else None,
            value_type=c.value_type if c is not None else None,
            path=c.path if c is not None else t.container_path if t is not None else None,
            key=s.key,
            examples=s.examples[:3],
            reasons=s.reasons,
        )


class FieldProbe(BaseModel):
    status: FieldStatus
    use: str | list[str] = ABSENT
    resolved_by: Literal["rules", "llm"] = "rules"
    key: str | None = None
    examples: list[str] = Field(default_factory=list, description="Three values of `use`.")
    candidates: list[CandidateInfo] = Field(default_factory=list, description="Best first.")
    note: str | None = None

    @classmethod
    def from_match(cls, m: FieldMatch) -> FieldProbe:
        top = m.top
        return cls(
            status=m.status,
            use=m.use,
            key=top.key if top is not None and m.status != "missing" else None,
            examples=top.examples[:3] if top is not None and m.status != "missing" else [],
            candidates=[CandidateInfo.from_scored(s) for s in m.ranked],
        )


class TemplateProbe(BaseModel):
    size: int
    url_patterns: list[str] = Field(default_factory=list)
    url_examples: list[str] = Field(default_factory=list)
    include: bool = True
    exclude_reason: str | None = None
    fields_held: list[str] = Field(default_factory=list)
    fields: dict[str, FieldProbe] = Field(default_factory=dict)

    def refresh(self) -> None:
        """Recompute ``fields_held`` and the include proposal from the field statuses."""
        self.fields_held = [f for f, fp in self.fields.items() if fp.status != "missing"]
        self.include = bool(self.fields_held)
        self.exclude_reason = None if self.include else NO_FIELD


class ProbeLLM(BaseModel):
    used: bool = False
    model: str | None = None
    prompt_version: str | None = None
    calls: int = 0
    resolved: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class ProbeResult(BaseModel):
    """``probe.json``: per template size, URL examples, fields held, per-field status."""

    version: Literal[1] = 1
    fields: list[FieldRequest]
    llm: ProbeLLM = Field(default_factory=ProbeLLM)
    templates: dict[str, TemplateProbe] = Field(default_factory=dict)
    skipped: dict[str, str] = Field(
        default_factory=dict, description="Non-tail templates without a candidate table."
    )
    seconds: float = 0.0

    @property
    def schema_fields(self) -> list[str]:
        return [f.name for f in self.fields]

    def counts(self) -> dict[str, int]:
        out = {"found": 0, "ambiguous": 0, "missing": 0, "templates": len(self.templates)}
        out["excluded"] = sum(not t.include for t in self.templates.values())
        for t in self.templates.values():
            for fp in t.fields.values():
                out[fp.status] += 1
        return out

    @classmethod
    def load(cls, path: str | Path) -> ProbeResult:
        return cls.model_validate(read_json(Path(path)))

    def save(self, path: str | Path) -> None:
        write_json(Path(path), self.model_dump(mode="json", by_alias=True))


# -- probe.yaml --------------------------------------------------------------------------------

YAML_HEADER = """\
# ditto probe: proposed mapping of requested fields to each template's candidates.
# Edit and save, then run `ditto confirm` (or close the editor it opened):
#   include: true | false   generate a spec for this template or skip it
#   use: c17                the candidate holding the field; a list [c4, c5] for record or
#                           key/value groups (a table id like tb1 also works); or `absent`
# Everything else (status, candidates with scores and examples) is informational.
"""


def candidate_line(ci: CandidateInfo) -> str:
    parts = [f"{ci.id} ({ci.score:.2f})"]
    if ci.cls:
        parts.append(ci.cls)
    if ci.label_hint:
        parts.append(f'label "{short(ci.label_hint, 40)}"')
    if ci.key:
        parts.append(f'key "{short(ci.key, 40)}"')
    if ci.value_type and ci.value_type != "text":
        parts.append(ci.value_type)
    if isinstance(ci.use, list):
        parts.append("use [" + ", ".join(ci.use) + "]")
    if ci.examples:
        parts.append("e.g. " + " | ".join(short(e, 40) for e in ci.examples[:3]))
    return " · ".join(parts)


def probe_yaml_data(result: ProbeResult) -> dict[str, Any]:
    templates: dict[str, Any] = {}
    for tid, tp in result.templates.items():
        entry: dict[str, Any] = {"include": tp.include, "size": tp.size}
        if tp.exclude_reason:
            entry["excluded"] = tp.exclude_reason
        entry["url_examples"] = tp.url_examples[:3]
        fields: dict[str, Any] = {}
        for name, fp in tp.fields.items():
            fentry: dict[str, Any] = {"use": fp.use, "status": fp.status}
            if fp.resolved_by != "rules":
                fentry["resolved_by"] = fp.resolved_by
            if fp.key:
                fentry["key"] = fp.key
            if fp.note:
                fentry["note"] = fp.note
            fentry["candidates"] = [candidate_line(ci) for ci in fp.candidates]
            fields[name] = fentry
        entry["fields"] = fields
        templates[tid] = entry
    return {"fields": result.schema_fields, "templates": templates}


def dump_probe_yaml(data: dict[str, Any]) -> str:
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    return YAML_HEADER + body


def write_probe_yaml(result: ProbeResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_probe_yaml(probe_yaml_data(result)), encoding="utf-8")


# -- probe ---------------------------------------------------------------------------------------


def probe_template(table: CandidateTable, cluster: Cluster, schema: FieldsSchema) -> TemplateProbe:
    """Rule-based probe of one template (no I/O, no LLM)."""
    matches = match_template(table, schema.fields)
    tp = TemplateProbe(
        size=cluster.size,
        url_patterns=cluster.url_patterns[:3],
        url_examples=cluster.url_examples[:3],
        fields={name: FieldProbe.from_match(m) for name, m in matches.items()},
    )
    tp.refresh()
    return tp


def probe_run(
    run: RunDir,
    schema: FieldsSchema,
    *,
    use_llm: bool = False,
    llm: LLM | None = None,
) -> ProbeResult:
    """Score every candidate of every non-tail template against every requested field.

    Writes ``probe.json`` and ``probe.yaml``. With ``use_llm`` (``--probe-llm``), templates
    with ambiguous fields get one small-model call (role ``probe``) over the candidate table.
    """
    t0 = time.perf_counter()
    clusters = Clusters.load(run.clusters_json)
    result = ProbeResult(fields=list(schema.fields))
    tables: dict[str, CandidateTable] = {}
    pages = 0
    for cluster in clusters.templates(include_tail=False):
        tid = cluster.template_id
        path = run.candidates(tid)
        if not path.exists():
            result.skipped[tid] = "no candidate table (run analysis first)"
            continue
        table = CandidateTable.load(path)
        tables[tid] = table
        result.templates[tid] = probe_template(table, cluster, schema)
        pages += cluster.size
    if use_llm:
        if llm is None:
            from ditto.llm import get_llm

            llm = get_llm("probe", run=run)
        resolve_ambiguous(result, tables, schema, llm)
    result.seconds = round(time.perf_counter() - t0, 4)
    result.save(run.probe_json)
    write_probe_yaml(result, run.probe_yaml)
    run.record_timing("probe", time.perf_counter() - t0, pages, 1)
    return result


# -- optional LLM resolution ---------------------------------------------------------------------


def _row(c: Candidate) -> str:
    reps = str(c.repeat_min) if c.repeat_min == c.repeat_max else f"{c.repeat_min}-{c.repeat_max}"
    examples = " · ".join(short(e, 60) for e in c.examples) or "-"
    tail = " > ".join(c.path.split(" > ")[-2:])
    hint = short(c.label_hint, 40) if c.label_hint else "-"
    return (
        f"{c.id} | {c.cls} | p={c.p:.2f} | x{reps} | {hint} | {c.value_type or '-'} | "
        f"{examples} | {tail}"
    )


def _table_row(t: Table, by_id: dict[str, Candidate]) -> str:
    names = [c.name for c in t.columns] if t.kind == "record_list" else t.keys_seen
    ids = (
        [c.candidate_id for c in t.columns]
        if t.kind == "record_list"
        else [i for i in (t.key_candidate, t.value_candidate) if i]
    )
    sample = ", ".join(short(n, 30) for n in names[:8]) or "-"
    tail = " > ".join(t.container_path.split(" > ")[-2:])
    first = by_id.get(ids[0]) if ids else None
    hint = short(first.label_hint, 40) if first is not None and first.label_hint else "-"
    return (
        f"{t.id} | {t.kind} table | p={t.p:.2f} | rows {t.rows_min}-{t.rows_max} | {hint} | "
        f"{'columns' if t.kind == 'record_list' else 'keys'}: {sample} | "
        f"cells {', '.join(ids)} | {tail}"
    )


def _field_line(req: FieldRequest, fp: FieldProbe) -> str:
    parts = [f"- {req.name}: {req.description or '(no description)'}"]
    if req.type:
        parts.append(f"type {req.type}")
    if req.type_check:
        parts.append(f"check {req.type_check}")
    if req.examples:
        parts.append("examples " + " | ".join(short(e, 40) for e in req.examples[:3]))
    guesses = ", ".join(ci.id for ci in fp.candidates) or "none"
    parts.append(f"guesses {guesses}")
    return "; ".join(parts)


def llm_rows(prep: Prepared, reqs: list[FieldRequest]) -> list[str]:
    """Candidate-table rows for the probe call: every usable candidate (most relevant first
    when trimmed to ``MAX_LLM_ROWS``, then in id order) plus tables for group fields."""
    usable = prep.usable
    if len(usable) > MAX_LLM_ROWS:
        ctxs = [FieldContext.of(r) for r in reqs]
        best = {c.id: max(score_candidate(x, c, prep).score for x in ctxs) for c in usable}
        keep = set(sorted(best, key=lambda i: -best[i])[:MAX_LLM_ROWS])
        usable = [c for c in usable if c.id in keep]
    rows = [_row(c) for c in usable]
    if any(r.type in ("record_list", "key_value") for r in reqs):
        rows += [_table_row(t, prep.by_id) for t in prep.tables.values()]
    return rows


def _make_validator(prep: Prepared, ambiguous: set[str]) -> Callable[[Any], Answers]:
    usable = {c.id for c in prep.usable}
    by_id = prep.by_id
    tables = prep.tables

    def resolve_id(cid: str, errors: list[str], field: str) -> str | list[str] | None:
        if cid in usable:
            return cid
        if cid in tables:
            t = tables[cid]
            if t.kind == "key_value":
                return [i for i in (t.key_candidate, t.value_candidate) if i]
            return [c.candidate_id for c in t.columns]
        if cid in by_id:
            errors.append(f"{field}: {cid} is a {by_id[cid].cls}, never a field value")
        else:
            errors.append(f"{field}: unknown candidate id {cid!r}")
        return None

    def validate(obj: Any) -> Answers:
        if not isinstance(obj, dict) or not isinstance(obj.get("fields"), dict):
            raise ValueError('expected an object {"fields": {"<field>": {"candidate": ...}}}')
        errors: list[str] = []
        out: Answers = {}
        for name, ans in obj["fields"].items():
            if name not in ambiguous:
                continue  # extra fields are harmless; only the asked ones are used
            if isinstance(ans, dict):
                choice, reason = ans.get("candidate"), str(ans.get("reason") or "")
            else:
                choice, reason = ans, ""
            if choice is None or (isinstance(choice, str) and choice.strip().lower() == ABSENT):
                out[name] = (ABSENT, ABSENT, reason)
            elif isinstance(choice, str):
                use = resolve_id(choice.strip(), errors, name)
                if use is not None:
                    out[name] = (use, choice.strip(), reason)
            else:
                errors.append(f"{name}: candidate must be an id string or 'absent'")
        if errors:
            raise ValueError("; ".join(errors))
        return out

    return validate


def answer_schema(fields: list[str]) -> dict[str, Any]:
    """Strict JSON Schema of the probe answer for the given field names."""
    answer = {
        "type": "object",
        "properties": {"candidate": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["candidate", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": {f: answer for f in fields},
                "required": list(fields),
                "additionalProperties": False,
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


def resolve_template(
    tid: str,
    tp: TemplateProbe,
    table: CandidateTable,
    schema: FieldsSchema,
    llm: LLM,
    prompt: Prompt,
) -> int:
    """One probe call for one template; returns how many ambiguous fields it resolved."""
    from ditto.llm import complete_json

    ambiguous = [f for f, fp in tp.fields.items() if fp.status == "ambiguous"]
    if not ambiguous:
        return 0
    reqs = [r for r in schema.fields if r.name in ambiguous]
    prep = prepare(table)
    messages = prompt.messages(
        template_id=tid,
        size=str(tp.size),
        url_examples="\n".join(tp.url_examples[:3]) or "(none)",
        fields="\n".join(_field_line(r, tp.fields[r.name]) for r in reqs),
        candidates="\n".join(llm_rows(prep, reqs)),
    )
    answers = complete_json(
        llm,
        messages,
        validate=_make_validator(prep, set(ambiguous)),
        prompt_version=prompt.id,
        template_id=tid,
        max_tokens=2000,
        json_schema=answer_schema(ambiguous),
    )
    resolved = 0
    for name, (use, chosen, reason) in answers.items():
        req = schema.get(name)
        assert req is not None
        fp = tp.fields[name]
        if use == ABSENT:
            tp.fields[name] = fp.model_copy(
                update={
                    "status": "missing",
                    "use": ABSENT,
                    "resolved_by": "llm",
                    "examples": [],
                    "key": None,
                    "note": reason or None,
                }
            )
        else:
            scored = score_by_id(req, prep, chosen)
            info = CandidateInfo.from_scored(scored) if scored is not None else None
            others = [ci for ci in fp.candidates if ci.id != chosen]
            tp.fields[name] = fp.model_copy(
                update={
                    "status": "found",
                    "use": use,
                    "resolved_by": "llm",
                    "examples": info.examples if info else [],
                    "key": info.key if info else None,
                    "candidates": ([info] if info else []) + others,
                    "note": reason or None,
                }
            )
        resolved += 1
    tp.refresh()
    return resolved


def resolve_ambiguous(
    result: ProbeResult, tables: dict[str, CandidateTable], schema: FieldsSchema, llm: LLM
) -> None:
    """``--probe-llm``: one small-model call per template that still has ambiguous fields.

    A template whose answer stays invalid after the layer's retry keeps its ambiguous fields
    (the error is recorded); budget and provider errors propagate.
    """
    from ditto.llm.base import InvalidResponse
    from ditto.specgen import load_prompt

    prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    result.llm = ProbeLLM(used=True, model=llm.model, prompt_version=prompt.id)
    for tid, tp in result.templates.items():
        if not any(fp.status == "ambiguous" for fp in tp.fields.values()):
            continue
        result.llm.calls += 1
        try:
            result.llm.resolved += resolve_template(tid, tp, tables[tid], schema, llm, prompt)
        except InvalidResponse as exc:
            result.llm.errors[tid] = str(exc)[:500]


__all__ = [
    "CandidateInfo",
    "FieldProbe",
    "ProbeLLM",
    "ProbeResult",
    "TemplateProbe",
    "probe_run",
    "probe_template",
    "write_probe_yaml",
]
