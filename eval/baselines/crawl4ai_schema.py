"""Crawl4AI baseline: ``JsonCssExtractionStrategy.generate_schema`` on one sample page per
template, then the generated CSS schema applied with ``JsonCssExtractionStrategy`` to every page
of the template. It isolates what ditto's union page and verification add over "one LLM call on
one page writes the selectors".

How it is run (crawl4ai 0.9.5, read from its source):

* ``generate_schema(html=<sample page>, schema_type="CSS", query=<field list>,
  target_json_example=<object with exactly the requested field names>, llm_config=LLMConfig(
  provider=<main model "openai/gpt-6.1-sol">, api_token=<OPENAI_API_KEY>, base_url=...),
  validate=True, max_refinements=3, usage=TokenUsage())`` — crawl4ai's defaults for validation
  and refinement. Crawl4AI preprocesses the HTML itself (``preprocess_html_for_schema``) and
  calls the model through ``litellm``. Because a target example is given, crawl4ai does not
  make its extra target-inference call; with validation on, it re-prompts (up to 3 times) when
  the schema misses a requested field on the sample page.
* The sample page is a seeded random page of the template; the page HTML is the same input
  ditto sees. Pages without a template (the long tail) get no schema and an empty prediction,
  matching ditto's default ``--tail skip``.
* Extraction: ``JsonCssExtractionStrategy(schema).extract(url, html)`` returns one item per
  ``baseSelector`` match; the item with the most requested fields filled is the page's record
  (first on ties). Output names map to requested names exactly, then case/punctuation-
  insensitively, then one level inside nested objects; values are coerced to ditto's shapes.
* Accounting: every model call goes through ``crawl4ai.utils.aperform_completion_with_backoff``;
  a scoped wrapper counts the calls and reads ``response.usage``. Generated schemas and their
  usage are cached under ``.cache/llm/crawl4ai/`` keyed by SHA-256 of (crawl4ai version, model,
  sample HTML, query, example), so reruns spend nothing; ``cost_usd`` still prices all tokens.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ditto.evaluation.baseline_api import BaselineOutput, NotRun, PageInput

if TYPE_CHECKING:
    from ditto.schema import FieldsSchema


def _load_common() -> ModuleType:
    name = "ditto_eval_baseline_common"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("common.py"))
        if spec is None or spec.loader is None:  # pragma: no cover - file ships with this one
            raise ImportError("eval/baselines/common.py not found")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


common = _load_common()

NAME = "crawl4ai_schema"
SCHEMA_TYPE = "CSS"
VALIDATE = True  # crawl4ai's default
MAX_REFINEMENTS = 3  # crawl4ai's default
CACHE_DIR = common.REPO_ROOT / ".cache" / "llm" / "crawl4ai"
PROMPT_ID = "crawl4ai.generate_schema"
TEMPERATURE = 1.0


# --------------------------------------------------------------------------- prompt inputs


def build_query(schema: FieldsSchema) -> str:
    """Natural-language description of the record crawl4ai should write a schema for."""
    lines = [
        "This is a detail page about one entity. Extract one record for the page's main entity "
        "(not related items, navigation, or ads) with exactly these fields:"
    ]
    for f in schema.fields:
        kind = common.field_kind(f)
        line = f"- {f.name} ({kind})"
        if f.description:
            line += f": {f.description}"
        if f.examples:
            line += f" (e.g. {', '.join(f.examples[:2])})"
        lines.append(line)
    return "\n".join(lines)


def target_example(schema: FieldsSchema) -> dict[str, Any]:
    """An example record whose keys are exactly the requested field names."""
    example: dict[str, Any] = {}
    for f in schema.fields:
        kind = common.field_kind(f)
        sample = f.examples[0] if f.examples else f"<{f.description or f.name}>"
        if kind == "scalar":
            example[f.name] = sample
        elif kind == "list":
            second = f.examples[1] if len(f.examples) > 1 else sample
            example[f.name] = [{"value": sample}, {"value": second}]
        elif kind == "record_list":
            example[f.name] = [{"column_1": "<cell text>", "column_2": "<cell text>"}]
        else:
            example[f.name] = [{"key": "<label text>", "value": "<value text>"}]
    return example


# --------------------------------------------------------------------------- model target


@dataclass(frozen=True)
class Target:
    """The main (baseline-role) model as litellm needs it, plus the ditto settings."""

    settings: Any
    model: str  # ditto's "provider/model" is already litellm's format
    api_key: str | None
    base_url: str | None


def model_target(settings: Any = None) -> Target:
    from ditto.llm import load_settings, resolve_model

    s = settings if settings is not None else load_settings()
    model = resolve_model("baseline", s)
    if not model.startswith("openai/"):
        raise NotRun(f"crawl4ai baseline is wired for the OpenAI adapter only, got {model}")
    return Target(s, model, s.openai_api_key, s.openai_base_url)


def projected_usd(html: str, model: str) -> float:
    """Upper-bound cost of one schema generation, for the budget guard: the page plus
    crawl4ai's ~4k-token instructions, re-sent on every refinement, 4k output tokens each."""
    import tiktoken

    page_tokens = len(tiktoken.get_encoding("o200k_base").encode(html, disallowed_special=()))
    attempts = 1 + (MAX_REFINEMENTS if VALIDATE else 0)
    usd, _ = common.price(model, (page_tokens + 4000) * attempts, 4000 * attempts)
    return float(usd)


# --------------------------------------------------------------------------- generation


@contextlib.contextmanager
def _count_calls(log: list[dict[str, Any]]) -> Iterator[None]:
    """Count crawl4ai's model calls and capture each response's usage (scoped patch)."""
    import crawl4ai.utils as c4u

    original = c4u.aperform_completion_with_backoff

    async def counted(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        response = await original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        log.append(
            {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "latency_s": round(time.perf_counter() - started, 3),
            }
        )
        return response

    c4u.aperform_completion_with_backoff = counted
    try:
        yield
    finally:
        c4u.aperform_completion_with_backoff = original


def _cache_key(model: str, html: str, query: str, example: dict[str, Any]) -> str:
    import crawl4ai

    payload = json.dumps(
        {
            "crawl4ai": str(crawl4ai.__version__.__version__),
            "model": model,
            "schema_type": SCHEMA_TYPE,
            "validate": VALIDATE,
            "max_refinements": MAX_REFINEMENTS,
            "temperature": TEMPERATURE,
            "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "query": query,
            "example": example,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_schema(
    html: str,
    schema: FieldsSchema,
    *,
    target: Target,
    template_id: str | None = None,
    cache_dir: Path | None = CACHE_DIR,
) -> dict[str, Any]:
    """Run crawl4ai's schema generation for one sample page.

    Returns ``{"schema", "calls", "input_tokens", "output_tokens", "call_log", "cached",
    "error"}``. On a cache miss the projected cost is checked against ditto's budget first and
    every model call is appended to ditto's global spend ledger afterwards.
    """
    from ditto.llm import check_budget, record_external_usage

    model = target.model
    query = build_query(schema)
    example = target_example(schema)
    key = _cache_key(model, html, query, example)
    cache_file = cache_dir / f"{key}.json" if cache_dir is not None else None
    if cache_file is not None and cache_file.is_file():
        hit = json.loads(cache_file.read_text(encoding="utf-8"))
        hit["cached"] = True
        return dict(hit)

    check_budget(projected_usd(html, model), settings=target.settings)

    from crawl4ai import JsonCssExtractionStrategy, LLMConfig
    from crawl4ai.models import TokenUsage

    usage = TokenUsage()
    calls: list[dict[str, Any]] = []
    result: dict[str, Any] = {"schema": None, "error": None}
    with _count_calls(calls):
        try:
            result["schema"] = JsonCssExtractionStrategy.generate_schema(
                html=html,
                schema_type=SCHEMA_TYPE,
                query=query,
                target_json_example=json.dumps(example, ensure_ascii=False, indent=2),
                llm_config=LLMConfig(
                    provider=model, api_token=target.api_key, base_url=target.base_url
                ),
                validate=VALIDATE,
                max_refinements=MAX_REFINEMENTS,
                usage=usage,
                # crawl4ai hard-codes temperature=0.01; the main model only accepts its
                # default (1) and litellm cannot drop the parameter for an unknown model id.
                temperature=TEMPERATURE,
            )
        except Exception as exc:  # crawl4ai wraps every failure in a bare Exception
            result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    for call in calls:
        record_external_usage(
            model,
            call["input_tokens"],
            call["output_tokens"],
            role="baseline",
            prompt_version=PROMPT_ID,
            template_id=template_id,
            settings=target.settings,
        )
    result.update(
        calls=len(calls),
        input_tokens=int(usage.prompt_tokens),
        output_tokens=int(usage.completion_tokens),
        call_log=calls,
        model=model,
        cached=False,
    )
    if cache_file is not None and result["schema"] is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


# --------------------------------------------------------------------------- extraction

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(name: str) -> str:
    return _NORM.sub("", name.lower())


def map_item(item: dict[str, Any], schema: FieldsSchema) -> dict[str, Any]:
    """Map one crawl4ai item onto the requested fields (raw values, not yet coerced)."""
    by_norm: dict[str, Any] = {}
    for k, v in item.items():
        by_norm.setdefault(_norm(str(k)), v)
    nested = [v for v in item.values() if isinstance(v, dict)]
    out: dict[str, Any] = {}
    for f in schema.fields:
        if f.name in item:
            out[f.name] = item[f.name]
        elif _norm(f.name) in by_norm:
            out[f.name] = by_norm[_norm(f.name)]
        else:
            out[f.name] = next((d[f.name] for d in nested if f.name in d), None)
    return out


def _filled(prediction: dict[str, Any]) -> int:
    return sum(1 for v in prediction.values() if v not in (None, [], {}))


def extract_page(
    css_schema: dict[str, Any], page_url: str, html: str, schema: FieldsSchema
) -> dict[str, Any]:
    """Apply a generated CSS schema to one page and return the page's prediction."""
    from crawl4ai import JsonCssExtractionStrategy

    try:
        items = JsonCssExtractionStrategy(css_schema).extract(page_url, html)
    except Exception:  # a malformed generated selector fails the page, not the run
        items = []
    best: dict[str, Any] | None = None
    for item in items:
        if not isinstance(item, dict):
            continue
        prediction = common.coerce_prediction(map_item(item, schema), schema)
        if best is None or _filled(prediction) > _filled(best):
            best = prediction
    if best is None:
        best = {f.name: common.empty_value(common.field_kind(f)) for f in schema.fields}
    return best


def _extract_chunk(
    chunk: Sequence[tuple[str, str, str, str, str]],
) -> list[tuple[str, dict[str, Any]]]:
    """pmap worker: ``(key, url, html, css_schema_json, fields_schema_json)`` per item."""
    from ditto.schema import FieldsSchema

    schemas: dict[str, FieldsSchema] = {}
    out: list[tuple[str, dict[str, Any]]] = []
    for key, url, html, css_json, fields_json in chunk:
        fields = schemas.get(fields_json)
        if fields is None:
            fields = schemas[fields_json] = FieldsSchema.model_validate_json(fields_json)
        out.append((key, extract_page(json.loads(css_json), url, html, fields)))
    return out


# --------------------------------------------------------------------------- entry point


def run_baseline(
    pages: list[PageInput],
    schema: FieldsSchema,
    *,
    out_dir: Path,
    settings: Any = None,
    max_pages: int | None = None,
    seed: int = 0,
    site: str | None = None,
    workers: int | None = None,
    cache_dir: Path | None = CACHE_DIR,
) -> BaselineOutput:
    """Generate one CSS schema per template from a sample page and apply it to the template."""
    from ditto.parallel import pmap_flat

    if importlib.util.find_spec("crawl4ai") is None:
        raise NotRun("crawl4ai is not installed (install the `eval` extra)")
    out_dir = Path(out_dir)
    selected = common.sample_pages(pages, max_pages, seed)
    groups, tail = common.group_by_template(selected)
    target = model_target(settings)
    model = target.model
    rng = random.Random(seed)
    fields_json = schema.model_dump_json()

    tally = common.UsageTally()
    template_rows: list[dict[str, Any]] = []
    work: list[tuple[str, str, str, str, str]] = []
    predictions: dict[str, dict[str, Any]] = {}
    empty = {f.name: common.empty_value(common.field_kind(f)) for f in schema.fields}
    for tid in sorted(groups):
        group = groups[tid]
        sample = rng.choice(group)
        gen = generate_schema(
            sample.html, schema, target=target, template_id=tid, cache_dir=cache_dir
        )
        for call in gen["call_log"]:
            nominal, estimated = common.price(
                model, int(call.get("input_tokens", 0)), int(call.get("output_tokens", 0))
            )
            tally.add(
                int(call.get("input_tokens", 0)),
                int(call.get("output_tokens", 0)),
                nominal_usd=nominal,
                spent_usd=0.0 if gen["cached"] else nominal,
                cached=bool(gen["cached"]),
                estimated=estimated,
            )
        css_schema = gen.get("schema")
        template_rows.append(
            {
                "template_id": tid,
                "pages": len(group),
                "sample_key": sample.key,
                "sample_url": sample.url,
                "calls": gen.get("calls", 0),
                "input_tokens": gen.get("input_tokens", 0),
                "output_tokens": gen.get("output_tokens", 0),
                "cached": gen.get("cached", False),
                "error": gen.get("error"),
                "schema": css_schema,
            }
        )
        if not isinstance(css_schema, dict) or "baseSelector" not in css_schema:
            for page in group:
                predictions[page.key] = dict(empty)
            continue
        css_json = json.dumps(css_schema)
        work.extend((p.key, p.url, p.html, css_json, fields_json) for p in group)

    started = time.perf_counter()
    for key, prediction in pmap_flat(_extract_chunk, work, workers=workers, chunk_size=32):
        predictions[key] = prediction
    extract_s = time.perf_counter() - started
    for page in tail:
        predictions[page.key] = dict(empty)

    failed = [r["template_id"] for r in template_rows if r["schema"] is None]
    notes = [
        f"crawl4ai JsonCssExtractionStrategy.generate_schema ({SCHEMA_TYPE}, validate={VALIDATE},"
        f" max_refinements={MAX_REFINEMENTS}) with {model} on one seeded random sample page per "
        f"template ({len(groups)} templates), schema applied to {len(work)} pages",
        f"cost_usd prices all tokens from config/prices.yaml; actually spent this run: "
        f"${tally.spent_usd:.4f} ({tally.cached_calls}/{tally.calls} calls from the schema cache)",
        f"extraction: {len(work)} pages in {extract_s:.2f}s",
    ]
    if tail:
        notes.append(f"{len(tail)} tail page(s) without a template got empty predictions")
    if failed:
        notes.append(f"schema generation failed for {len(failed)} template(s): {failed}")
    output = BaselineOutput(
        predictions=predictions,
        calls=tally.calls,
        input_tokens=tally.input_tokens,
        output_tokens=tally.output_tokens,
        cost_usd=tally.nominal_usd,
        notes=notes,
        model=model,
        pages_run=len(selected),
        spend_usd=tally.spent_usd,
        prompt_version=PROMPT_ID,
    )
    common.write_jsonl(out_dir / f"{NAME}.templates.jsonl", template_rows)
    common.write_jsonl(
        out_dir / f"{NAME}.predictions.jsonl", common.predictions_rows(selected, predictions)
    )
    common.write_summary(
        out_dir,
        NAME,
        output,
        {"seed": seed, "max_pages": max_pages, "usage": tally.as_dict(), "extract_s": extract_s},
    )
    return output
