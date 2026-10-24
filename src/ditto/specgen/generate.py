"""Stage 6: spec generation, one main-model call per template.

Input per template: the union page (``union/<tid>.html``), the compact candidate table
(``candidates/<tid>.json``), the spec JSON Schema, the mode instructions and, in prompted mode,
the requested fields plus the confirmed ``mapping.json`` when present. Output: a validated draft
spec (``specs/draft/<tid>.json``), the per-template outcome (``generation.json``) and, in
unprompted mode, the proposed schema for the checkpoint (``schema_proposal.json``).

Validation compiles and lints every selector through the Pydantic spec model. One retry
carries the exact errors back (``ditto.llm.complete_json``). After a second failure the fields
that validate on their own are kept and the failing ones are recorded with their candidate ids
in ``generation.json`` so verification's path-induced fallback can build them.

Prompted runs with ``mapping.json``: excluded and unmapped templates are skipped, fields the
mapping marks ``absent`` are not asked for (they go to ``not_present``), the mapping is stated
in the prompt, and the union page is rebuilt trimmed to the mapped candidates' neighborhoods via
``ditto.union.build_union_run(..., focus=...)``; that overwrites ``union/<tid>.html`` with the
page the model actually saw. Unprompted runs rebuild a union page left focused by an earlier
prompted run.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, get_args

from pydantic import TypeAdapter, ValidationError

from ditto.analyze.models import Candidate, CandidateTable
from ditto.cluster.models import Cluster, Clusters
from ditto.ingest import PageStore
from ditto.llm import (
    LLM,
    BudgetExceeded,
    Completion,
    InvalidResponse,
    LLMError,
    Message,
    check_budget,
    complete_json,
    estimate_cost,
    get_llm,
    parse_json,
    run_concurrently,
)
from ditto.run import RunDir, read_json, write_json
from ditto.schema import ABSENT, FieldRequest, FieldsSchema, Mapping, TemplateMapping
from ditto.specgen.models import (
    RESERVED_FIELD_NAMES,
    FieldType,
    GeneratedBy,
    Spec,
    SpecField,
    TypeCheck,
    spec_json_schema,
)
from ditto.specgen.prompts import Prompt, load_prompt
from ditto.union import build_union_run, count_tokens

Mode = Literal["prompted", "unprompted"]
TailMode = Literal["skip", "direct", "spec"]
Status = Literal["ok", "partial", "failed", "skipped", "unprocessed"]

PROMPT_NAME = "specgen"
OPTIONAL_BELOW = 0.95  # brief: optional when the field's path is on < 95% of the cluster's pages
DEFAULT_MAX_TOKENS = 16384  # room for reasoning tokens plus a large unprompted spec
PROPOSAL_PAGES = 5
EXAMPLE_CHARS = 80
LABEL_CHARS = 60
KEYS_SHOWN = 30

# Keys verification owns; dropped from the model's answer instead of failing validation.
_VERIFIER_KEYS = ("stats", "selected", "fallbacks", "degraded", "selection_reason")
_FIELD_DEFS = ("ScalarField", "ListField", "RecordListField", "KeyValueField")
_FIELD_TYPES: frozenset[str] = frozenset(get_args(FieldType))
_TYPE_CHECKS: frozenset[str] = frozenset(get_args(TypeCheck))
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CID = re.compile(r"^c\d+$")

_FIELD_ADAPTER: TypeAdapter[SpecField] = TypeAdapter(SpecField)


# --------------------------------------------------------------------------------------------
# Output schema shown to the model
# --------------------------------------------------------------------------------------------


def _simplify_schema(node: Any) -> Any:
    """Drop titles, null defaults and ``anyOf [X, null]`` wrappers: same constraints, fewer
    tokens."""
    if isinstance(node, list):
        return [_simplify_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("title", "discriminator"):
            continue
        if key == "default" and value in (None, [], {}, False):
            continue
        out[key] = _simplify_schema(value)
    any_of = out.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2 and {"type": "null"} in any_of:
        other = next(x for x in any_of if x != {"type": "null"})
        del out["anyOf"]
        out = {**other, **out}
    return out


def llm_output_schema() -> dict[str, Any]:
    """JSON Schema of what the model returns: ``{fields: {name: field}, not_present}``.

    Derived from the Pydantic spec model (the single source of truth), minus everything the
    pipeline fills itself: the spec header (template, site, generated_by, status) and the keys
    verification owns (stats, selected, fallbacks, degraded, selection_reason).
    """
    full = spec_json_schema()
    defs: dict[str, Any] = {}
    for name in _FIELD_DEFS:
        d = copy.deepcopy(full["$defs"][name])
        for key in _VERIFIER_KEYS:
            d["properties"].pop(key, None)
        d["properties"]["type"].pop("default", None)
        d["required"] = sorted({*d.get("required", []), "type"})
        defs[name] = _simplify_schema(d)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "fields": _simplify_schema(full["properties"]["fields"]),
            "not_present": _simplify_schema(full["properties"]["not_present"]),
        },
        "required": ["fields", "not_present"],
        "additionalProperties": False,
        "$defs": defs,
    }
    return schema


# --------------------------------------------------------------------------------------------
# Inputs and outcomes
# --------------------------------------------------------------------------------------------


@dataclass
class TemplateInput:
    """Everything one generation call needs; built from the run directory."""

    template_id: str
    site: str
    mode: Mode
    table: CandidateTable
    union_html: str
    size: int = 0
    url_patterns: list[str] = field(default_factory=list)
    url_examples: list[str] = field(default_factory=list)
    pages_seen: list[str] = field(default_factory=list)
    shown: set[str] | None = None  # candidate ids annotated on the union page (None: unknown)
    fields: list[FieldRequest] = field(default_factory=list)  # prompted: fields to ask for
    mapped: dict[str, list[str]] = field(default_factory=dict)  # field -> confirmed cand. ids
    absent: list[str] = field(default_factory=list)  # fields mapping.json marks absent
    focused: bool = False


@dataclass
class FailedField:
    """A field the model did not deliver valid selectors for (built by path-induced fallback)."""

    type: FieldType | None
    source_candidates: list[str]
    description: str | None
    type_check: TypeCheck | None
    errors: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source_candidates": self.source_candidates,
            "description": self.description,
            "type_check": self.type_check,
            "errors": self.errors,
        }


@dataclass
class TemplateGeneration:
    template_id: str
    status: Status
    spec: Spec | None = None
    reason: str | None = None
    size: int = 0
    not_present: dict[str, str] = field(default_factory=dict)
    failed_fields: dict[str, FailedField] = field(default_factory=dict)
    dropped_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    focused: bool = False
    prompt_tokens: int = 0
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def to_json(self, run: RunDir | None = None) -> dict[str, Any]:
        draft: str | None = None
        if self.spec is not None:
            draft = (
                str(run.draft_spec(self.template_id).relative_to(run.root))
                if run is not None
                else f"specs/draft/{self.template_id}.json"
            )
        return {
            "status": self.status,
            "reason": self.reason,
            "size": self.size,
            "draft": draft,
            "fields": sorted(self.spec.fields) if self.spec is not None else [],
            "not_present": self.not_present,
            "failed_fields": {k: v.to_json() for k, v in self.failed_fields.items()},
            "dropped_fields": self.dropped_fields,
            "warnings": self.warnings,
            "focused": self.focused,
            "prompt_tokens": self.prompt_tokens,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 3),
        }


@dataclass
class GenerationResult:
    mode: Mode
    model: str
    prompt_version: str
    templates: dict[str, TemplateGeneration] = field(default_factory=dict)
    unprocessed: list[str] = field(default_factory=list)
    aborted: str | None = None
    projected_cost_usd: float = 0.0
    projected_estimated: bool = False
    proposal: dict[str, Any] | None = None
    tail: Any = None  # ditto.specgen.tail.TailResult when --tail direct ran
    tail_error: str | None = None

    @property
    def specs(self) -> dict[str, Spec]:
        return {tid: t.spec for tid, t in self.templates.items() if t.spec is not None}

    def by_status(self, status: Status) -> list[str]:
        return [tid for tid, t in self.templates.items() if t.status == status]


# --------------------------------------------------------------------------------------------
# Prompt sections
# --------------------------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _candidate_row(c: Candidate, shown: set[str] | None) -> str:
    cid = f"{c.id} @{c.attr}" if c.attr else c.id
    rep = str(c.repeat_min) if c.repeat_min == c.repeat_max else f"{c.repeat_min}-{c.repeat_max}"
    table = f"{c.table_id}:{c.table_role}" if c.table_id and c.table_role else "-"
    if c.column:
        table += f" ({_clip(c.column, LABEL_CHARS)})"
    label = (
        json.dumps(_clip(c.label_hint, LABEL_CHARS), ensure_ascii=False) if c.label_hint else "-"
    )
    examples = " · ".join(
        json.dumps(_clip(e, EXAMPLE_CHARS), ensure_ascii=False) for e in c.examples[:3]
    )
    row = (
        f"{cid} | {c.cls} | {c.p:.2f} | {rep} | {label} | {c.value_type or '-'} | {table} | "
        f"{examples or '-'}"
    )
    if shown is not None and c.id not in shown:
        row += f" | not on union page; path: {c.path}"
    return row


def format_candidate_table(table: CandidateTable, shown: set[str] | None = None) -> str:
    """Compact rows for field candidates and every candidate that belongs to a table."""
    rows = [
        _candidate_row(c, shown) for c in table.candidates if c.is_field or c.table_id is not None
    ]
    if not rows:
        return "(no varying content was found on this template)"
    header = "id | class | p | rep | label | type | table | examples"
    return "\n".join([header, *rows])


def format_tables(table: CandidateTable) -> str:
    if not table.tables:
        return "(none)"
    lines: list[str] = []
    for t in table.tables:
        head = (
            f"{t.id} {t.kind}: {t.rows_min}-{t.rows_max} rows per page, on {t.p:.0%} of pages; "
            f"row path {t.row_path}"
        )
        if t.kind == "key_value":
            keys = ", ".join(_clip(k, 40) for k in t.keys_seen[:KEYS_SHOWN])
            more = (
                f" (+{len(t.keys_seen) - KEYS_SHOWN} more)" if len(t.keys_seen) > KEYS_SHOWN else ""
            )
            head += f"; key {t.key_candidate or '-'}, value {t.value_candidate or '-'}"
            head += f"; keys seen: {keys}{more}" if keys else ""
        else:
            cols = ", ".join(
                f"{_clip(col.name, 40)} ({col.candidate_id}"
                + (", header cell" if col.header_source == "th" else ", no header")
                + ")"
                for col in t.columns
            )
            head += f"; columns: {cols}" if cols else ""
        lines.append(head)
    return "\n".join(lines)


def format_template_context(inp: TemplateInput) -> str:
    parts = [f"Site: {inp.site}. Template {inp.template_id}: {inp.size:,} pages."]
    if inp.url_patterns:
        parts.append("URL patterns: " + ", ".join(inp.url_patterns[:3]) + ".")
    if inp.url_examples:
        parts.append("URL examples: " + ", ".join(inp.url_examples[:3]) + ".")
    return " ".join(parts)


def format_fields_section(inp: TemplateInput) -> str:
    if inp.mode == "unprompted":
        return (
            "## Fields\n\nNone requested: choose the entity fields yourself from the candidate "
            "table and the union page."
        )
    lines = ["## Requested fields", ""]
    if inp.mapped:
        lines += [
            "The user confirmed which candidates hold the fields (mapping.json); write the "
            "selectors for exactly those candidates.",
            "",
        ]
    for f in inp.fields:
        attrs = [a for a in (f.type, f.type_check and f"type_check {f.type_check}") if a]
        line = f"- `{f.name}`"
        if attrs:
            line += f" ({', '.join(attrs)})"
        line += f": {f.description}" if f.description else ""
        if f.examples:
            line += ". Examples: " + " · ".join(
                json.dumps(_clip(e, EXAMPLE_CHARS), ensure_ascii=False) for e in f.examples[:3]
            )
        if f.name in inp.mapped:
            line += ". Confirmed mapping: " + ", ".join(inp.mapped[f.name])
        lines.append(line)
    return "\n".join(lines)


def build_messages(prompt: Prompt, inp: TemplateInput, schema_text: str) -> list[Message]:
    return prompt.messages(
        mode=inp.mode,
        output_schema=schema_text,
        template_context=format_template_context(inp),
        fields_section=format_fields_section(inp),
        candidate_table=format_candidate_table(inp.table, inp.shown),
        tables_section=format_tables(inp.table),
        union_page=f"```html\n{inp.union_html.strip()}\n```",
    )


def schema_text() -> str:
    return json.dumps(llm_output_schema(), separators=(",", ":"))


# --------------------------------------------------------------------------------------------
# Validation of the model's answer
# --------------------------------------------------------------------------------------------


@dataclass
class _Checked:
    fields: dict[str, SpecField] = field(default_factory=dict)
    not_present: dict[str, str] = field(default_factory=dict)
    raw_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    field_errors: dict[str, list[str]] = field(default_factory=dict)
    global_errors: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        out = list(self.global_errors)
        for name, errs in self.field_errors.items():
            out.extend(f"fields.{name}: {e}" for e in errs)
        return out


def _validation_messages(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(x) for x in err["loc"])
        msg = str(err["msg"]).removeprefix("Value error, ")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def _parse_or_none(text: str) -> Any:
    """The model's last answer parsed for salvage (same lenient parser as ``complete_json``)."""
    try:
        return parse_json(text)
    except ValueError:
        return None


class _Checker:
    """Validates one answer against the spec model and the template's request."""

    def __init__(self, inp: TemplateInput) -> None:
        self.inp = inp
        self.requested = {f.name: f for f in inp.fields}
        self.requested_ci = {f.name.lower(): f.name for f in inp.fields}
        self.known = {c.id: c for c in inp.table.candidates}

    def validate(self, data: Any) -> _Checked:
        checked = self.check(data)
        if checked.errors:
            raise ValueError("\n".join(checked.errors))
        return checked

    def check(self, data: Any) -> _Checked:
        out = _Checked()
        if not isinstance(data, dict):
            out.global_errors.append(
                "the answer must be one JSON object with keys 'fields' and 'not_present'"
            )
            return out
        raw_fields = data.get("fields")
        if not isinstance(raw_fields, dict):
            out.global_errors.append("'fields' must be an object mapping field name -> field")
            raw_fields = {}
        raw_np = data.get("not_present", {})
        if raw_np is None:
            raw_np = {}
        if not isinstance(raw_np, dict):
            out.global_errors.append("'not_present' must be an object mapping field -> reason")
            raw_np = {}
        prompted = self.inp.mode == "prompted"

        for raw_name, raw in raw_fields.items():
            name = self._field_name(str(raw_name), out)
            if name is None:
                continue
            if not isinstance(raw, dict):
                out.field_errors[name] = ["must be an object with a 'type' key"]
                continue
            prepared = self._prepare(name, raw, out)
            out.raw_fields[name] = prepared
            try:
                spec_field = _FIELD_ADAPTER.validate_python(prepared)
            except ValidationError as exc:
                out.field_errors[name] = _validation_messages(exc)
                continue
            request = self.requested.get(name)
            if request is not None and request.type and spec_field.type != request.type:
                out.field_errors[name] = [
                    f"type must be {request.type!r} as requested, got {spec_field.type!r}"
                ]
                continue
            out.fields[name] = spec_field

        if prompted:
            for raw_name, reason in raw_np.items():
                name = self.requested_ci.get(str(raw_name).lower())
                if name is None or name in self.inp.absent:
                    out.dropped.append(str(raw_name))
                    continue
                if name in out.raw_fields or name in out.field_errors:
                    continue  # a delivered field wins over a not_present entry
                out.not_present[name] = str(reason) if reason else "not found on this template"
            for name in self.requested:
                delivered = name in out.raw_fields or name in out.field_errors
                if not delivered and name not in out.not_present:
                    out.global_errors.append(
                        f"requested field {name!r} is missing: return a spec for it under "
                        "'fields' or list it under 'not_present' with a reason"
                    )
        else:
            if raw_np:
                out.warnings.append("unprompted mode: ignored not_present entries")
            if not raw_fields:
                out.global_errors.append(
                    "no fields returned: choose the entity fields from the candidate table"
                )
        return out

    def _field_name(self, raw_name: str, out: _Checked) -> str | None:
        if self.inp.mode == "prompted":
            name = self.requested_ci.get(raw_name.lower())
            if name is None or name in self.inp.absent:
                out.dropped.append(raw_name)
                return None
            return name
        if not _SNAKE.match(raw_name):
            out.field_errors[raw_name] = [
                "field name must be snake_case (lowercase letters, digits, underscores)"
            ]
            return None
        if raw_name in RESERVED_FIELD_NAMES:
            out.field_errors[raw_name] = [
                f"field name {raw_name!r} is reserved (every output row already has "
                f"{', '.join(RESERVED_FIELD_NAMES)}); choose a more specific name"
            ]
            return None
        return raw_name

    def _prepare(self, name: str, raw: dict[str, Any], out: _Checked) -> dict[str, Any]:
        """Fill what the pipeline knows and drop what verification owns, before validation."""
        prepared = {k: v for k, v in raw.items() if k not in _VERIFIER_KEYS}
        request = self.requested.get(name)
        if request is not None:
            if request.description and not prepared.get("description"):
                prepared["description"] = request.description
            if request.type_check and not prepared.get("type_check"):
                prepared["type_check"] = request.type_check
        if prepared.get("type") in ("record_list", "key_value"):
            prepared.pop("type_check", None)  # type checks apply to scalar/list values only
        sources = prepared.get("source_candidates")
        ids = [str(s) for s in sources] if isinstance(sources, list) else []
        unknown = [s for s in ids if s not in self.known]
        if unknown:
            out.warnings.append(f"{name}: dropped unknown source candidate ids {unknown}")
        ids = [s for s in ids if s in self.known]
        if not ids and name in self.inp.mapped:
            ids = [s for s in self.inp.mapped[name] if s in self.known]
        prepared["source_candidates"] = ids
        if ids:
            prepared["optional"] = max(self.known[s].p for s in ids) < OPTIONAL_BELOW
        return prepared

    def failed_field(self, name: str, errors: list[str], raw: dict[str, Any] | None) -> FailedField:
        request = self.requested.get(name)
        raw = raw or {}
        ftype = raw.get("type") if raw.get("type") in _FIELD_TYPES else None
        if ftype is None and request is not None:
            ftype = request.type
        tcheck = raw.get("type_check") if raw.get("type_check") in _TYPE_CHECKS else None
        if tcheck is None and request is not None:
            tcheck = request.type_check
        sources = raw.get("source_candidates")
        ids = [s for s in sources if s in self.known] if isinstance(sources, list) else []
        if not ids:
            ids = [s for s in self.inp.mapped.get(name, []) if s in self.known]
        description = raw.get("description") if isinstance(raw.get("description"), str) else None
        if description is None and request is not None and request.description:
            description = request.description
        return FailedField(
            type=ftype,
            source_candidates=ids,
            description=description,
            type_check=tcheck,
            errors=errors,
        )


# --------------------------------------------------------------------------------------------
# One template
# --------------------------------------------------------------------------------------------


class RecordingLLM:
    """Pass-through LLM that keeps the completions made through it (list.append is atomic, so
    one instance may be shared by worker threads)."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.model = inner.model
        self.role = inner.role
        self.completions: list[Completion] = []

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        result = self.inner.complete(
            messages,
            json_schema=json_schema,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
            template_id=template_id,
        )
        self.completions.append(result)
        return result


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _make_spec(inp: TemplateInput, llm: LLM, prompt: Prompt, checked: _Checked) -> Spec:
    not_present = dict(checked.not_present)
    for name in inp.absent:
        not_present[name] = "marked absent for this template in mapping.json"
    return Spec(
        template_id=inp.template_id,
        site=inp.site,
        generated_by=GeneratedBy(
            model=llm.model,
            prompt_version=prompt.id,
            pages_seen=inp.pages_seen,
            mode=inp.mode,
            created_at=_now(),
        ),
        fields=checked.fields,
        not_present=not_present,
    )


def generate_spec(
    inp: TemplateInput,
    *,
    llm: LLM,
    prompt: Prompt | None = None,
    messages: list[Message] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> TemplateGeneration:
    """One generation call (plus at most one retry) for one template. Writes nothing.

    ``BudgetExceeded`` propagates (the caller stops scheduling); other LLM errors and a second
    invalid answer become a ``failed``/``partial`` outcome.
    """
    t0 = time.perf_counter()
    prompt = prompt or load_prompt(PROMPT_NAME)
    if messages is None:
        messages = build_messages(prompt, inp, schema_text())
    checker = _Checker(inp)
    recorder = RecordingLLM(llm)
    gen = TemplateGeneration(
        template_id=inp.template_id, status="ok", size=inp.size, focused=inp.focused
    )
    checked: _Checked | None = None
    try:
        checked = complete_json(
            recorder,
            messages,
            validate=checker.validate,
            prompt_version=prompt.id,
            template_id=inp.template_id,
            max_tokens=max_tokens,
        )
    except BudgetExceeded:
        raise
    except InvalidResponse as exc:
        gen.reason = "invalid after retry: " + "; ".join(exc.errors)[:2000]
        data = _parse_or_none(exc.raw)
        checked = checker.check(data) if isinstance(data, dict) else None
        _record_failures(gen, checker, checked)
    except LLMError as exc:
        gen.status = "failed"
        gen.reason = f"{type(exc).__name__}: {exc}"
        _record_failures(gen, checker, None)
    if checked is not None:
        gen.dropped_fields = checked.dropped
        gen.warnings = checked.warnings
        if checked.fields or checked.not_present or (gen.reason is None and inp.absent):
            gen.spec = _make_spec(inp, llm, prompt, checked)
            gen.not_present = gen.spec.not_present
    if gen.reason is not None:
        gen.status = "partial" if gen.spec is not None else "failed"
    elif gen.spec is None:  # pragma: no cover - validate() guarantees fields or not_present
        gen.status = "failed"
    _account(gen, recorder.completions)
    gen.seconds = time.perf_counter() - t0
    return gen


def _record_failures(gen: TemplateGeneration, checker: _Checker, checked: _Checked | None) -> None:
    """Failed fields after a second invalid answer (or an LLM error), for the fallback."""
    if checked is not None:
        for name, errs in checked.field_errors.items():
            gen.failed_fields[name] = checker.failed_field(name, errs, checked.raw_fields.get(name))
    delivered = set(checked.fields) | set(checked.not_present) if checked is not None else set()
    for name in checker.requested:
        if name not in delivered and name not in gen.failed_fields:
            gen.failed_fields[name] = checker.failed_field(
                name, ["no valid answer for this field"], None
            )


def _account(gen: TemplateGeneration, completions: Sequence[Completion]) -> None:
    gen.calls = len(completions)
    gen.cached_calls = sum(1 for c in completions if c.cached)
    gen.input_tokens = sum(c.usage.input_tokens for c in completions)
    gen.output_tokens = sum(c.usage.output_tokens for c in completions)
    gen.cost_usd = sum(c.cost_usd for c in completions)


def expected_output_tokens(inp: TemplateInput) -> int:
    """Rough answer size for the cost projection (selectors ~250 tokens per field, plus room for
    reasoning tokens)."""
    if inp.mode == "prompted":
        n = len(inp.fields)
    else:
        n = min(25, sum(1 for c in inp.table.candidates if c.is_field))
    return 600 + 250 * max(n, 1)


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


class _Skip(Exception):
    pass


@dataclass
class _Plan:
    cluster: Cluster
    table: CandidateTable
    fields: list[FieldRequest]
    mapped: dict[str, list[str]]
    absent: list[str]
    focus: set[str] | None


def _select_templates(
    clusters: Clusters, min_cluster_size: int, tail: TailMode, template_ids: Sequence[str] | None
) -> list[Cluster]:
    pool = clusters.templates(include_tail=True)
    if tail != "spec":
        pool = [c for c in pool if c.size >= min_cluster_size]
    if template_ids is not None:
        wanted = set(template_ids)
        unknown = wanted - {c.template_id for c in clusters.clusters}
        if unknown:
            raise KeyError(f"unknown template id(s): {sorted(unknown)}")
        pool = [c for c in pool if c.template_id in wanted]
    return pool


def _plan(
    run: RunDir,
    cluster: Cluster,
    mode: Mode,
    schema: FieldsSchema | None,
    mapping: Mapping | None,
) -> _Plan:
    tid = cluster.template_id
    tm: TemplateMapping | None = None
    if mapping is not None:
        tm = mapping.templates.get(tid)
        if tm is None:
            raise _Skip("not in mapping.json")
        if not tm.included:
            raise _Skip("excluded in mapping.json" + (f": {tm.note}" if tm.note else ""))
    path = run.candidates(tid)
    if not path.exists():
        raise _Skip(f"no candidate table at {path.relative_to(run.root)}; run analyze first")
    table = CandidateTable.load(path)
    fields: list[FieldRequest] = []
    mapped: dict[str, list[str]] = {}
    absent: list[str] = []
    focus: set[str] | None = None
    if mode == "prompted":
        assert schema is not None
        known = {c.id for c in table.candidates}
        for f in schema.fields:
            value = tm.fields.get(f.name) if tm is not None else None
            if value == ABSENT:
                absent.append(f.name)
                continue
            fields.append(f)
            if value is not None:
                ids = [value] if isinstance(value, str) else list(value)
                ids = [i for i in ids if i in known and _CID.match(i)]
                if ids:
                    mapped[f.name] = ids
        if not fields:
            raise _Skip("every requested field is marked absent in mapping.json")
        if mapped:
            focus = {i for ids in mapped.values() for i in ids}
    return _Plan(cluster, table, fields, mapped, absent, focus)


def _union_pages(run: RunDir, plans: list[_Plan]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Union html + meta per template, (re)building only what is missing or mismatched."""
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    rebuild: list[str] = []
    focus: dict[str, set[str]] = {}
    for p in plans:
        tid = p.cluster.template_id
        html_path, meta_path = run.union_html(tid), run.union_meta(tid)
        meta: dict[str, Any] | None = None
        if html_path.exists() and meta_path.exists():
            meta = read_json(meta_path)
        want = sorted(p.focus) if p.focus else None
        if meta is not None and (meta.get("focus") or None) == want:
            out[tid] = (html_path.read_text(encoding="utf-8"), meta)
            continue
        rebuild.append(tid)
        if p.focus:
            focus[tid] = p.focus
    if rebuild:
        pages = build_union_run(run, template_ids=rebuild, focus=focus or None)
        for tid in rebuild:
            page = pages.get(tid)
            if page is not None:
                out[tid] = (page.html, page.meta())
    return out


def _template_input(
    plan: _Plan, mode: Mode, site: str, html: str, meta: dict[str, Any]
) -> TemplateInput:
    pages_seen = [u for u in [meta.get("base_url"), *(meta.get("donor_urls") or [])] if u]
    if not pages_seen and meta.get("base_page_id") is not None:
        pages_seen = [f"page:{meta['base_page_id']}"]
    shown = meta.get("shown")
    return TemplateInput(
        template_id=plan.cluster.template_id,
        site=site,
        mode=mode,
        table=plan.table,
        union_html=html,
        size=plan.cluster.size,
        url_patterns=list(plan.cluster.url_patterns),
        url_examples=list(plan.cluster.url_examples),
        pages_seen=[str(u) for u in pages_seen],
        shown=set(shown) if isinstance(shown, list) else None,
        fields=plan.fields,
        mapped=plan.mapped,
        absent=plan.absent,
        focused=bool(plan.focus),
    )


def generate_run(
    run: RunDir,
    *,
    mode: Mode,
    schema: FieldsSchema | None = None,
    mapping: Mapping | None = None,
    llm: LLM | None = None,
    min_cluster_size: int = 20,
    tail: TailMode = "skip",
    template_ids: Sequence[str] | None = None,
    concurrency: int = 4,
    tail_llm: LLM | None = None,
) -> GenerationResult:
    """Generate draft specs for every template at or above ``min_cluster_size``.

    ``tail="spec"`` forces generation for tail clusters too; ``tail="direct"`` additionally runs
    one small-model call per tail page (``ditto.specgen.tail.tail_direct_run``, needs a schema).
    Refuses to start (raises ``BudgetExceeded`` after recording the refusal in
    ``generation.json``) when the projected cost exceeds the remaining budget; a budget abort
    mid-run returns normally with the unprocessed templates listed.
    """
    t0 = time.perf_counter()
    if mode not in ("prompted", "unprompted"):
        raise ValueError(f"mode must be 'prompted' or 'unprompted', got {mode!r}")
    if tail not in ("skip", "direct", "spec"):
        raise ValueError(f"tail must be 'skip', 'direct' or 'spec', got {tail!r}")
    if mode == "prompted" and schema is None:
        raise ValueError("prompted mode needs a schema (fields.yaml)")
    names = schema.names() if schema is not None else []
    bad = [n for n in names if n in RESERVED_FIELD_NAMES or n.startswith("_")]
    if bad:
        raise ValueError(
            f"fields.yaml uses reserved field names {bad}: "
            f"{', '.join(RESERVED_FIELD_NAMES)} and names starting with '_' are output-row keys"
        )
    if tail == "direct" and schema is None:
        raise ValueError("--tail direct extracts the prompted schema: it needs a schema")
    if mode == "prompted" and mapping is None and run.mapping_json.exists():
        mapping = Mapping.load(run.mapping_json)
    if mode == "unprompted":
        mapping = None

    clusters = Clusters.load(run.clusters_json)
    site = run.site
    prompt = load_prompt(PROMPT_NAME)
    llm = llm or get_llm("generation", run=run)
    result = GenerationResult(mode=mode, model=llm.model, prompt_version=prompt.id)

    selected = _select_templates(clusters, min_cluster_size, tail, template_ids)
    plans: list[_Plan] = []
    for cluster in selected:
        try:
            plans.append(_plan(run, cluster, mode, schema, mapping))
        except _Skip as skip:
            result.templates[cluster.template_id] = TemplateGeneration(
                template_id=cluster.template_id,
                status="skipped",
                reason=str(skip),
                size=cluster.size,
            )
    unions = _union_pages(run, plans)
    inputs: list[TemplateInput] = []
    for plan in plans:
        tid = plan.cluster.template_id
        if tid not in unions:
            result.templates[tid] = TemplateGeneration(
                template_id=tid, status="skipped", reason="no union page", size=plan.cluster.size
            )
            continue
        html, meta = unions[tid]
        inputs.append(_template_input(plan, mode, site, html, meta))

    # Projected cost: exact prompt tokens, estimated answer tokens.
    schema_json = schema_text()
    messages = {inp.template_id: build_messages(prompt, inp, schema_json) for inp in inputs}
    prompt_tokens = {
        tid: sum(count_tokens(m["content"]) for m in msgs) for tid, msgs in messages.items()
    }
    out_tokens = sum(expected_output_tokens(inp) for inp in inputs)
    projected, estimated = estimate_cost(llm.model, sum(prompt_tokens.values()), out_tokens)
    result.projected_cost_usd, result.projected_estimated = projected, estimated
    try:
        if inputs:
            check_budget(projected, run=run)
    except BudgetExceeded as exc:
        result.aborted = f"refused: {exc}"
        for inp in inputs:
            result.templates[inp.template_id] = TemplateGeneration(
                template_id=inp.template_id,
                status="unprocessed",
                reason="projected cost exceeds the remaining budget",
                size=inp.size,
                prompt_tokens=prompt_tokens[inp.template_id],
            )
        result.unprocessed = [inp.template_id for inp in inputs]
        _write_generation(run, result, schema, min_cluster_size, tail, template_ids)
        raise

    def work(inp: TemplateInput) -> TemplateGeneration:
        gen = generate_spec(inp, llm=llm, prompt=prompt, messages=messages[inp.template_id])
        gen.prompt_tokens = prompt_tokens[inp.template_id]
        return gen

    done, unprocessed, budget_error = _run_bounded(work, inputs, concurrency)
    for gen in done:
        result.templates[gen.template_id] = gen
    for inp in unprocessed:
        result.templates[inp.template_id] = TemplateGeneration(
            template_id=inp.template_id,
            status="unprocessed",
            reason="budget exhausted before this template",
            size=inp.size,
            prompt_tokens=prompt_tokens[inp.template_id],
        )
    result.unprocessed = [inp.template_id for inp in unprocessed]
    if budget_error is not None:
        result.aborted = f"budget exhausted: {budget_error}"

    _write_drafts(run, result, selected, full=template_ids is None)
    if mode == "unprompted":
        result.proposal = write_schema_proposal(
            run, result, clusters, merge=template_ids is not None
        )
    _write_generation(run, result, schema, min_cluster_size, tail, template_ids)

    if tail == "direct":
        assert schema is not None
        from ditto.specgen.tail import tail_direct_run

        try:
            result.tail = tail_direct_run(run, schema, llm=tail_llm, concurrency=concurrency)
        except BudgetExceeded as exc:
            result.tail_error = f"refused: {exc}"
        _write_generation(run, result, schema, min_cluster_size, tail, template_ids)

    run.record_timing(
        "generate",
        time.perf_counter() - t0,
        pages=sum(inp.size for inp in inputs),
        workers=max(1, min(concurrency, len(inputs))),
    )
    return result


def _run_bounded(
    fn: Callable[[TemplateInput], TemplateGeneration],
    items: list[TemplateInput],
    concurrency: int,
) -> tuple[list[TemplateGeneration], list[TemplateInput], str | None]:
    """``ditto.llm.run_concurrently`` adapter: (outcomes, unprocessed items, budget message).

    An unexpected exception in one template becomes that template's ``failed`` outcome."""
    if not items:
        return [], [], None
    pool = run_concurrently(fn, items, concurrency=concurrency)
    done = [gen for _, gen in pool.results]
    for inp, exc in pool.errors:
        done.append(
            TemplateGeneration(
                template_id=inp.template_id,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                size=inp.size,
                focused=inp.focused,
            )
        )
    budget = str(pool.budget_error) if pool.budget_error is not None else None
    return done, list(pool.not_processed), budget


def _write_drafts(
    run: RunDir, result: GenerationResult, selected: list[Cluster], *, full: bool
) -> None:
    written: set[str] = set()
    for tid, gen in result.templates.items():
        if gen.spec is not None:
            gen.spec.save(run.draft_spec(tid))
            written.add(tid)
    # A draft left by an earlier run must not outlive this run's outcome for its template.
    stale = {c.template_id for c in selected} - written
    if full and run.drafts_dir.exists():
        stale |= {p.stem for p in run.drafts_dir.glob("*.json")} - written
    for tid in stale:
        run.draft_spec(tid).unlink(missing_ok=True)


def _write_generation(
    run: RunDir,
    result: GenerationResult,
    schema: FieldsSchema | None,
    min_cluster_size: int,
    tail: TailMode,
    template_ids: Sequence[str] | None,
) -> None:
    templates: dict[str, Any] = {}
    if template_ids is not None and run.generation_json.exists():
        previous = read_json(run.generation_json)
        if isinstance(previous, dict) and isinstance(previous.get("templates"), dict):
            templates.update(previous["templates"])
    templates.update({tid: gen.to_json(run) for tid, gen in result.templates.items()})
    tail_info: dict[str, Any] | None = None
    if result.tail is not None:
        tail_info = result.tail.to_json()
    elif result.tail_error is not None:
        tail_info = {"status": "refused", "reason": result.tail_error}
    data = {
        "version": 1,
        "mode": result.mode,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "created_at": _now(),
        "min_cluster_size": min_cluster_size,
        "tail": tail,
        "schema_fields": schema.names() if schema is not None else [],
        "projected_cost_usd": round(result.projected_cost_usd, 6),
        "projected_cost_estimated": result.projected_estimated,
        "cost_usd": round(sum(g.cost_usd for g in result.templates.values()), 6),
        "calls": sum(g.calls for g in result.templates.values()),
        "aborted": result.aborted,
        "unprocessed": sorted(tid for tid, t in templates.items() if t["status"] == "unprocessed"),
        "templates": templates,
        "tail_direct": tail_info,
    }
    write_json(run.generation_json, data)


# --------------------------------------------------------------------------------------------
# Unprompted: schema proposal for the checkpoint
# --------------------------------------------------------------------------------------------


def _example(value: Any) -> Any:
    """Bound an extracted value for display: long strings clipped, long lists cut."""
    if isinstance(value, str):
        return _clip(value, 200)
    if isinstance(value, list):
        return [_example(v) for v in value[:3]]
    if isinstance(value, dict):
        return {str(k): _example(v) for k, v in list(value.items())[:8]}
    return value


def write_schema_proposal(
    run: RunDir,
    result: GenerationResult,
    clusters: Clusters,
    *,
    merge: bool = False,
    pages_per_template: int = PROPOSAL_PAGES,
) -> dict[str, Any]:
    """``schema_proposal.json``: per template, the proposed fields with types, descriptions and
    up to three example values from applying the draft spec to a few sampled pages."""
    from ditto.extract.runner import apply_spec_html, is_missing

    store: PageStore | None = PageStore.open(run) if run.pages_db.exists() else None
    templates: dict[str, Any] = {}
    if merge and run.schema_proposal.exists():
        previous = read_json(run.schema_proposal)
        if isinstance(previous, dict) and isinstance(previous.get("templates"), dict):
            templates.update(previous["templates"])
    for tid, gen in result.templates.items():
        templates.pop(tid, None)
        spec = gen.spec
        if spec is None:
            continue
        cluster = clusters.get(tid)
        sample = clusters.sample(tid, n=pages_per_template, seed=0) if store is not None else []
        examples: dict[str, list[Any]] = {name: [] for name in spec.fields}
        for pid in sample:
            assert store is not None
            row = apply_spec_html(spec, store.read_html(pid))
            for name in spec.fields:
                value = row.get(name)
                shown = _example(value)
                if (
                    not is_missing(value)
                    and shown not in examples[name]
                    and len(examples[name]) < 3
                ):
                    examples[name].append(shown)
        fields: dict[str, Any] = {}
        for name, f in spec.fields.items():
            entry: dict[str, Any] = {
                "type": f.type,
                "type_check": getattr(f, "type_check", None),
                "description": f.description,
                "optional": f.optional,
                "source_candidates": f.source_candidates,
            }
            if f.type == "record_list":
                entry["columns"] = list(f.columns)
            entry["examples"] = examples[name]
            fields[name] = entry
        templates[tid] = {
            "size": cluster.size,
            "url_examples": cluster.url_examples[:3],
            "pages_seen": spec.generated_by.pages_seen,
            "sample_page_ids": sample,
            "fields": fields,
        }
    proposal = {
        "version": 1,
        "site": run.site,
        "mode": "unprompted",
        "model": result.model,
        "prompt_version": result.prompt_version,
        "created_at": _now(),
        "templates": templates,
    }
    write_json(run.schema_proposal, proposal)
    return proposal
