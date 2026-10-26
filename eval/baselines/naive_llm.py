"""Naive per-page LLM extraction: the accuracy ceiling and cost floor ditto has to beat.

Every sampled page goes to the main model (role ``baseline``, the same model that writes ditto's
specs) with ``prompts/naive_baseline.v1.md``. The model sees exactly
:func:`common.page_for_llm` of the page: ditto's cleaned tree (no scripts, styles, comments or
head except the title) with attributes pruned and whitespace collapsed. It answers with one JSON
object holding every requested field; the answer is parsed and validated by
``ditto.llm.complete_json`` (one retry carrying the errors) and coerced onto ditto's value shapes.

Per the brief this runs on a seeded random sample of at most ``max_pages`` pages per site (the
harness passes 200); the harness extrapolates cost per 1,000 pages from ``cost_usd /
pages_run``. ``cost_usd`` prices every token from ``config/prices.yaml`` whether or not the
response came from the on-disk LLM cache, so a cached rerun reports the same cost; the money a
run actually spent is in the notes and the summary file. Per-page token counts are written to
``<out_dir>/naive_llm.pages.jsonl``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ditto.evaluation.baseline_api import BaselineOutput, PageInput

if TYPE_CHECKING:
    from ditto.llm.base import LLM, Message
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

NAME = "naive_llm"
PROMPT_NAME = "naive_baseline"
PROMPT_VERSION = 1
MAX_OUTPUT_TOKENS = 8192
DEFAULT_CONCURRENCY = 4
# OpenAI Flex processing: same model and answers, ~50% price, slower; None = default tier.
SERVICE_TIER: str | None = "flex"


def build_messages(page: PageInput, schema: FieldsSchema) -> list[Message]:
    """The chat messages for one page (system rules + fields + serialized page)."""
    from ditto.specgen.prompts import load_prompt

    prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
    return prompt.messages(
        url=page.url or "(unknown)",
        fields=json.dumps(common.fields_payload(schema), ensure_ascii=False, indent=1),
        page=common.page_for_llm(page.html),
    )


def make_validator(schema: FieldsSchema) -> Any:
    """``complete_json`` validator: the answer must be a JSON object; values are then coerced.

    Missing keys count as absent (``null``); extra keys are ignored. Only a non-object answer
    is an error worth a retry.
    """

    def validate(obj: Any) -> dict[str, Any]:
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValueError(
                f"expected one JSON object with keys {schema.names()}, got {type(obj).__name__}"
            )
        prediction: dict[str, Any] = common.coerce_prediction(obj, schema)
        return prediction

    return validate


def _empty_prediction(schema: FieldsSchema) -> dict[str, Any]:
    return {f.name: common.empty_value(common.field_kind(f)) for f in schema.fields}


def run_baseline(
    pages: list[PageInput],
    schema: FieldsSchema,
    *,
    out_dir: Path,
    settings: Any = None,
    max_pages: int | None = None,
    seed: int = 0,
    site: str | None = None,
    llm: LLM | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    service_tier: str | None = SERVICE_TIER,
) -> BaselineOutput:
    """Extract every field of ``schema`` from each sampled page with one main-model call.

    ``llm`` overrides :func:`common.baseline_llm` (role ``baseline`` on ``service_tier``);
    tests pass a fake.
    """
    from ditto.llm import complete_json
    from ditto.llm.base import BudgetExceeded, InvalidResponse, LLMError

    out_dir = Path(out_dir)
    sample = common.sample_pages(pages, max_pages, seed)
    if llm is None:
        base_llm = common.baseline_llm(settings, service_tier=service_tier)
        tier = service_tier or "default"
    else:
        base_llm, tier = llm, "caller-supplied LLM"
    recorder = common.RecordingLLM(base_llm)
    validate = make_validator(schema)
    prompt_id = f"{PROMPT_NAME}.v{PROMPT_VERSION}"

    def one(page: PageInput) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = build_messages(page, schema)
        local = common.RecordingLLM(recorder)
        started = time.perf_counter()
        error: str | None = None
        try:
            prediction = complete_json(
                local,
                messages,
                validate=validate,
                prompt_version=prompt_id,
                template_id=page.template_id,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except BudgetExceeded:
            raise
        except InvalidResponse as exc:
            prediction, error = _empty_prediction(schema), f"invalid response: {exc.errors[:3]}"
        except LLMError as exc:
            prediction, error = _empty_prediction(schema), f"llm error: {exc}"
        calls = local.drain()
        row = {
            "key": page.key,
            "url": page.url,
            "template_id": page.template_id,
            "calls": len(calls),
            "input_tokens": sum(c.usage.input_tokens for c in calls),
            "output_tokens": sum(c.usage.output_tokens for c in calls),
            "cached": all(c.cached for c in calls) if calls else False,
            "spent_usd": sum(0.0 if c.cached else c.cost_usd for c in calls),
            "latency_s": round(time.perf_counter() - started, 3),
            "error": error,
        }
        return prediction, row

    predictions: dict[str, dict[str, Any]] = {}
    page_rows: list[dict[str, Any]] = []
    workers = max(1, min(concurrency, len(sample) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for page, (prediction, row) in zip(sample, pool.map(one, sample), strict=True):
            predictions[page.key] = prediction
            page_rows.append(row)

    tally = common.UsageTally()
    for completion in recorder.drain():
        common.tally_completion(tally, completion)

    failed = [r["key"] for r in page_rows if r["error"]]
    notes = [
        f"naive per-page extraction with {base_llm.model} ({prompt_id}) on {len(sample)} of "
        f"{len(pages)} pages (seeded random sample, seed={seed}, max_pages={max_pages})",
        f"cost_usd prices all {tally.input_tokens} input + {tally.output_tokens} output tokens "
        f"from config/prices.yaml (cached calls included); actually spent this run: "
        f"${tally.spent_usd:.4f} ({tally.cached_calls}/{tally.calls} calls served from cache)",
    ]
    if sample:
        per_page_in = tally.input_tokens / len(sample)
        per_page_out = tally.output_tokens / len(sample)
        notes.append(
            f"per page: {per_page_in:.0f} input + {per_page_out:.0f} output tokens; "
            f"extrapolated ${tally.nominal_usd / len(sample) * 1000:.2f} per 1,000 pages"
        )
    if tally.estimated_price:
        notes.append("price entry for the model is marked estimated")
    notes.append(
        f"service tier: {tier} (falls back to the default tier when Flex has no capacity); "
        "spend is billed per call at the tier the provider reports, cost tables use standard prices"
    )
    if failed:
        notes.append(f"{len(failed)} page(s) got no valid answer and score as empty: {failed[:10]}")

    output = BaselineOutput(
        predictions=predictions,
        calls=tally.calls,
        input_tokens=tally.input_tokens,
        output_tokens=tally.output_tokens,
        cost_usd=tally.nominal_usd,
        notes=notes,
        model=base_llm.model,
        pages_run=len(sample),
        spend_usd=tally.spent_usd,
        pages_failed=len(failed),
    )
    common.write_jsonl(out_dir / f"{NAME}.pages.jsonl", page_rows)
    common.write_jsonl(
        out_dir / f"{NAME}.predictions.jsonl", common.predictions_rows(sample, predictions)
    )
    common.write_summary(
        out_dir,
        NAME,
        output,
        {"prompt": prompt_id, "seed": seed, "max_pages": max_pages, "usage": tally.as_dict()},
    )
    return output
