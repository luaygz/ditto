"""Stage 10, ``--tail direct``: one small-model call per long-tail page.

Pages of clusters below the minimum size (``is_tail``) get no spec unless ``--tail spec`` made
one. With ``--tail direct`` each page of a tail cluster without a final spec is sent, cleaned and
compacted, to the small model, which extracts the prompted schema directly. Rows have the
extraction JSONL shape with ``template_id: null`` and are written to ``out/<site>/_tail.jsonl``
(``RunDir.out_jsonl(site, None)``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
from lxml import html

from ditto.cluster.models import Clusters
from ditto.fingerprint.dom import canonical_classes, clean_html
from ditto.ingest import PageStore
from ditto.llm import check_budget, complete_json, estimate_cost, get_llm, run_concurrently
from ditto.llm.base import LLM, BudgetExceeded, LLMError, Message
from ditto.run import RunDir
from ditto.schema import FieldRequest, FieldsSchema
from ditto.specgen.generate import RecordingLLM
from ditto.specgen.models import Spec
from ditto.specgen.prompts import Prompt, load_prompt
from ditto.union import count_tokens

PROMPT_NAME = "tail_direct"
PAGE_TOKENS = 12000  # cap on the compacted page sent per call
MAX_TOKENS = 4096
URL_CHARS = 120
_KEEP_ATTRS = frozenset(
    {
        "id",
        "class",
        "href",
        "src",
        "alt",
        "title",
        "itemprop",
        "datetime",
        "content",
        "aria-label",
        "data-testid",
        "role",
    }
)
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------------------------
# Page compaction
# --------------------------------------------------------------------------------------------


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = count_tokens(text)
    for _ in range(4):
        if tokens <= max_tokens:
            return text
        text = text[: max(1, int(len(text) * max_tokens / tokens * 0.97))]
        tokens = count_tokens(text)
    return text


def compact_page(page: str | bytes, max_tokens: int = PAGE_TOKENS) -> str:
    """Cleaned page HTML with attributes pruned, whitespace collapsed, capped at ``max_tokens``.

    Uses the same cleaning as extraction (scripts, styles, head except title removed).
    """
    root = clean_html(page)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for name in list(el.attrib):
            if name not in _KEEP_ATTRS:
                del el.attrib[name]
        cls = el.get("class")
        if cls is not None:
            kept = canonical_classes(cls)
            if kept:
                el.set("class", " ".join(kept))
            else:
                del el.attrib["class"]
        for name in ("href", "src"):
            value = el.get(name)
            if value is not None and len(value) > URL_CHARS:
                el.set(name, value[:URL_CHARS])
    text = html.tostring(root, encoding="unicode", method="html")
    return _truncate_to_tokens(_WS.sub(" ", text).strip(), max_tokens)


# --------------------------------------------------------------------------------------------
# Answer validation
# --------------------------------------------------------------------------------------------


def _empty(ftype: str | None) -> Any:
    if ftype in ("list", "record_list"):
        return []
    if ftype == "key_value":
        return {}
    return None


def _scalar(value: Any, where: str) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = _WS.sub(" ", value).strip()
        return text or None
    raise ValueError(f"{where}: expected a string or null, got {type(value).__name__}")


def _normalize(request: FieldRequest, value: Any) -> Any:
    name, ftype = request.name, request.type
    if ftype == "scalar" or ftype is None:
        if isinstance(value, list):
            if ftype is None:
                items = [_scalar(v, f"{name}[{i}]") for i, v in enumerate(value)]
                return [v for v in items if v is not None]
            if len(value) != 1:
                raise ValueError(f"{name}: scalar field expects one string or null, got a list")
            value = value[0]
        return _scalar(value, name)
    if ftype == "list":
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        items = [_scalar(v, f"{name}[{i}]") for i, v in enumerate(values)]
        return [v for v in items if v is not None]
    if ftype == "record_list":
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"{name}: record_list expects an array of objects")
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(value):
            if not isinstance(row, dict):
                raise ValueError(f"{name}[{i}]: each row must be an object")
            clean = {str(k): _scalar(v, f"{name}[{i}].{k}") for k, v in row.items()}
            if any(v is not None for v in clean.values()):
                rows.append(clean)
        return rows
    # key_value
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name}: key_value expects an object of label -> value")
    out: dict[str, Any] = {}
    for k, v in value.items():
        key = _WS.sub(" ", str(k)).strip().rstrip(":").strip()
        if key:
            out[key] = _scalar(v, f"{name}.{key}")
    return out


class _RowValidator:
    def __init__(self, schema: FieldsSchema) -> None:
        self.schema = schema

    def __call__(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("the answer must be one JSON object keyed by the requested fields")
        lower = {str(k).lower(): v for k, v in data.items()}
        errors: list[str] = []
        out: dict[str, Any] = {}
        for request in self.schema.fields:
            raw = data[request.name] if request.name in data else lower.get(request.name.lower())
            try:
                out[request.name] = _normalize(request, raw)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("\n".join(errors))
        return out


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


def format_fields(schema: FieldsSchema) -> str:
    lines: list[str] = []
    for f in schema.fields:
        line = f"- `{f.name}`" + (f" ({f.type})" if f.type else "")
        line += f": {f.description}" if f.description else ""
        if f.examples:
            line += ". Examples: " + " · ".join(repr(e) for e in f.examples[:3])
        lines.append(line)
    return "\n".join(lines)


@dataclass
class TailPage:
    page_id: int
    url: str
    cluster_id: str


@dataclass
class TailResult:
    pages: int
    rows: int = 0
    path: Path | None = None
    failed: list[int] = field(default_factory=list)
    unprocessed: list[int] = field(default_factory=list)
    aborted: str | None = None
    prompt_version: str = ""
    model: str = ""
    projected_cost_usd: float = 0.0
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "aborted" if self.aborted else "ok",
            "pages": self.pages,
            "rows": self.rows,
            "path": str(self.path) if self.path is not None else None,
            "failed": self.failed,
            "unprocessed": self.unprocessed,
            "aborted": self.aborted,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "projected_cost_usd": round(self.projected_cost_usd, 6),
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 3),
        }


def tail_row(
    page: TailPage, schema: FieldsSchema, values: dict[str, Any] | None, prompt_id: str, error: str
) -> dict[str, Any]:
    """One JSONL row in the extraction shape, ``template_id`` null."""
    row: dict[str, Any] = {"url": page.url, "page_id": page.page_id, "template_id": None}
    missing: list[str] = []
    for f in schema.fields:
        value = values.get(f.name) if values is not None else None
        if value is None or value == [] or value == {}:
            value = _empty(f.type)
            missing.append(f.name)
        row[f.name] = value
    meta: dict[str, Any] = {
        "missing": missing,
        "spec_version": None,
        "method": "tail_direct",
        "prompt_version": prompt_id,
        "cluster_id": page.cluster_id,
    }
    if error:
        meta["error"] = error
    row["_meta"] = meta
    return row


def _has_final_spec(run: RunDir, template_id: str) -> bool:
    path = run.spec(template_id)
    return path.exists() and Spec.load(path).status == "final"


def tail_pages(run: RunDir, store: PageStore) -> list[TailPage]:
    """Pages of tail clusters without a final spec (extraction covers the others)."""
    clusters = Clusters.load(run.clusters_json)
    pages = [
        TailPage(pid, store.url(pid), c.template_id)
        for c in clusters.clusters
        if c.is_tail and not _has_final_spec(run, c.template_id)
        for pid in c.page_ids
    ]
    return sorted(pages, key=lambda p: p.page_id)


def _messages(prompt: Prompt, fields: str, page: TailPage, body: str) -> list[Message]:
    return prompt.messages(fields=fields, url=page.url, page=body)


def tail_direct_run(
    run: RunDir,
    schema: FieldsSchema,
    *,
    llm: LLM | None = None,
    concurrency: int = 4,
    max_page_tokens: int = PAGE_TOKENS,
) -> TailResult:
    """Extract ``schema`` from every page of a tail cluster without a final spec with one
    small-model call each.

    Refuses (raises ``BudgetExceeded``) when the projected cost exceeds the remaining budget;
    a budget abort mid-run writes the rows done so far and lists the pages not processed.
    """
    t0 = time.perf_counter()
    store = PageStore.open(run)
    pages = tail_pages(run, store)
    prompt = load_prompt(PROMPT_NAME)
    llm = llm or get_llm("tail_direct", run=run)
    result = TailResult(pages=len(pages), prompt_version=prompt.id, model=llm.model)
    fields_text = format_fields(schema)
    validator = _RowValidator(schema)

    # Projection: compact every page once (also reused for the calls).
    bodies: dict[int, str] = {}
    in_tokens = 0
    for page in pages:
        body = compact_page(store.read_html(page.page_id), max_page_tokens)
        bodies[page.page_id] = body
        in_tokens += sum(
            count_tokens(m["content"]) for m in _messages(prompt, fields_text, page, body)
        )
    out_tokens = len(pages) * (300 + 60 * len(schema.fields))
    result.projected_cost_usd, _ = estimate_cost(llm.model, in_tokens, out_tokens)
    if pages:
        check_budget(result.projected_cost_usd, run=run)

    counter = RecordingLLM(llm)

    def work(page: TailPage) -> dict[str, Any]:
        messages = _messages(prompt, fields_text, page, bodies[page.page_id])
        try:
            values = complete_json(
                counter,
                messages,
                validate=validator,
                prompt_version=prompt.id,
                template_id=page.cluster_id,
                max_tokens=MAX_TOKENS,
            )
        except BudgetExceeded:
            raise
        except LLMError as exc:
            return tail_row(page, schema, None, prompt.id, f"{type(exc).__name__}: {exc}")
        return tail_row(page, schema, values, prompt.id, "")

    rows: list[dict[str, Any]] = []
    if pages:
        pool = run_concurrently(work, pages, concurrency=concurrency)
        rows = [row for _, row in pool.results]
        rows += [
            tail_row(page, schema, None, prompt.id, f"{type(exc).__name__}: {exc}")
            for page, exc in pool.errors
        ]
        result.unprocessed = sorted(p.page_id for p in pool.not_processed)
        if pool.budget_error is not None:
            result.aborted = f"budget exhausted: {pool.budget_error}"
    rows.sort(key=lambda r: int(r["page_id"]))
    result.failed = [int(r["page_id"]) for r in rows if "error" in r["_meta"]]

    path = run.out_jsonl(run.site, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row))
            fh.write(b"\n")
    tmp.replace(path)
    result.path = path
    result.rows = len(rows)
    completions = counter.completions
    result.calls = len(completions)
    result.cached_calls = sum(1 for c in completions if c.cached)
    result.input_tokens = sum(c.usage.input_tokens for c in completions)
    result.output_tokens = sum(c.usage.output_tokens for c in completions)
    result.cost_usd = sum(c.cost_usd for c in completions)
    result.seconds = time.perf_counter() - t0
    run.record_timing(
        "tail_direct",
        result.seconds,
        pages=len(pages),
        workers=max(1, min(concurrency, len(pages))),
    )
    return result


__all__ = [
    "TailPage",
    "TailResult",
    "compact_page",
    "format_fields",
    "tail_direct_run",
    "tail_pages",
    "tail_row",
]
