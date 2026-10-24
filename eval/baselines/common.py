"""Shared helpers for the evaluation baselines in ``eval/baselines/``.

The harness loads every baseline module by file path (``ditto.evaluation.baseline_api``), so
siblings load this file with :func:`load_common`-style code instead of a package import.

What lives here:

* the interface types, re-exported from :mod:`ditto.evaluation.baseline_api`;
* seeded page sampling (``max_pages`` per site);
* :func:`page_for_llm`, the exact page serialization the naive LLM baseline sees;
* :func:`coerce_value`, which maps loosely shaped baseline output onto ditto's value shapes
  (``scalar`` -> ``str | None``, ``list`` -> ``list[str]``, ``record_list`` ->
  ``list[dict[str, str | None]]``, ``key_value`` -> ``dict[str, str | None]``);
* :class:`UsageTally` / :class:`RecordingLLM` for token and cost accounting;
* writers for the per-baseline artifacts in ``out_dir``.
"""

from __future__ import annotations

import json
import random
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lxml import etree

from ditto.evaluation.baseline_api import BaselineOutput, NotRun, PageInput
from ditto.fingerprint.dom import canonical_classes, clean_html
from ditto.specgen.models import FieldType
from ditto.textutil import normalize_ws

if TYPE_CHECKING:
    from ditto.llm.base import LLM, Completion, Message, Role
    from ditto.schema import FieldRequest, FieldsSchema

__all__ = [
    "REPO_ROOT",
    "BaselineOutput",
    "NotRun",
    "PageInput",
    "RecordingLLM",
    "UsageTally",
    "baseline_llm",
    "coerce_prediction",
    "coerce_value",
    "empty_value",
    "field_kind",
    "fields_payload",
    "group_by_template",
    "page_for_llm",
    "sample_pages",
    "write_jsonl",
    "write_summary",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Attributes the naive baseline's model sees (the union page's list plus value-bearing ones).
LLM_KEPT_ATTRS = (
    "id",
    "class",
    "href",
    "src",
    "itemprop",
    "itemtype",
    "content",
    "datetime",
    "data-testid",
    "role",
    "aria-label",
    "type",
    "alt",
    "title",
)
LINK_ATTR_MAX = 80
OTHER_ATTR_MAX = 300


# --------------------------------------------------------------------------- sampling


def sample_pages(pages: Sequence[PageInput], max_pages: int | None, seed: int) -> list[PageInput]:
    """A seeded random sample of at most ``max_pages`` pages, kept in input order."""
    if max_pages is None or max_pages >= len(pages):
        return list(pages)
    if max_pages <= 0:
        return []
    chosen = set(random.Random(seed).sample(range(len(pages)), max_pages))
    return [p for i, p in enumerate(pages) if i in chosen]


def group_by_template(
    pages: Iterable[PageInput],
) -> tuple[dict[str, list[PageInput]], list[PageInput]]:
    """``(template_id -> pages in input order, tail pages without a template)``."""
    groups: dict[str, list[PageInput]] = {}
    tail: list[PageInput] = []
    for page in pages:
        if page.template_id is None:
            tail.append(page)
        else:
            groups.setdefault(page.template_id, []).append(page)
    return groups, tail


# --------------------------------------------------------------------------- page serialization


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit]


def page_for_llm(html: str | bytes) -> str:
    """Serialize a page the way the naive LLM baseline sees it.

    1. ``ditto.fingerprint.dom.clean_html`` (scripts, styles, noscript, svg, iframe, template,
       comments, processing instructions and ``<head>`` except ``<title>`` removed): the same
       tree ditto's extractor runs on.
    2. Attributes pruned to :data:`LLM_KEPT_ATTRS`; ``class`` reduced to its canonical classes
       (utility, state and generated classes dropped, as in ditto's structural paths);
       ``href``/``src`` cut to 80 characters, other kept values to 300.
    3. Runs of whitespace in text collapsed to one space (the visible text is unchanged).
    4. Serialized with lxml's HTML serializer.
    """
    root = clean_html(html)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        attrib = el.attrib
        for name in list(attrib):
            if name not in LLM_KEPT_ATTRS:
                del attrib[name]
                continue
            value = attrib[name]
            if name == "class":
                classes = canonical_classes(value)
                if classes:
                    attrib[name] = " ".join(classes)
                else:
                    del attrib[name]
            elif name in ("href", "src"):
                attrib[name] = _truncate(value.strip(), LINK_ATTR_MAX)
            else:
                attrib[name] = _truncate(normalize_ws(value), OTHER_ATTR_MAX)
        if el.text:
            el.text = _collapse(el.text)
        if el.tail:
            el.tail = _collapse(el.tail)
    return etree.tostring(root, encoding="unicode", method="html")


def _collapse(text: str) -> str:
    return " ".join(text.split()) if text.strip() else (" " if text else "")


# --------------------------------------------------------------------------- value shapes


def field_kind(request: FieldRequest) -> FieldType:
    return request.type or "scalar"


def fields_payload(schema: FieldsSchema) -> list[dict[str, Any]]:
    """The requested fields as plain JSON objects (name, type, description, examples)."""
    out: list[dict[str, Any]] = []
    for f in schema.fields:
        item: dict[str, Any] = {"name": f.name, "type": field_kind(f)}
        if f.description:
            item["description"] = f.description
        if f.examples:
            item["examples"] = list(f.examples)
        out.append(item)
    return out


def empty_value(kind: FieldType) -> Any:
    if kind == "list" or kind == "record_list":
        return []
    if kind == "key_value":
        return {}
    return None


def _scalar(value: Any) -> str | None:
    """One value as a whitespace-normalized string; containers and blanks become ``None``."""
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    text = normalize_ws(text)
    return text or None


def _first_scalar(values: Iterable[Any]) -> str | None:
    for v in values:
        s = _scalar(v) if not isinstance(v, dict) else _first_scalar(v.values())
        if s is not None:
            return s
    return None


def coerce_value(value: Any, kind: FieldType) -> Any:
    """Map a loosely shaped value onto ditto's output shape for ``kind``."""
    if kind == "scalar":
        if isinstance(value, list):
            return _first_scalar(value)
        if isinstance(value, dict):
            return _first_scalar(value.values()) if len(value) == 1 else None
        return _scalar(value)
    if kind == "list":
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        out: list[str] = []
        for item in items:
            s = _first_scalar(item.values()) if isinstance(item, dict) else _scalar(item)
            if s is not None:
                out.append(s)
        return out
    if kind == "record_list":
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return []
        rows: list[dict[str, str | None]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            record = {str(k): _scalar(v) for k, v in row.items()}
            if any(v is not None for v in record.values()):
                rows.append(record)
        return rows
    # key_value
    pairs: dict[str, str | None] = {}
    if isinstance(value, dict):
        items_kv: Iterable[tuple[Any, Any]] = value.items()
    elif isinstance(value, list):  # [{"key": .., "value": ..}] or [[k, v]]
        items_kv = []
        for entry in value:
            if isinstance(entry, dict) and "key" in entry:
                items_kv.append((entry.get("key"), entry.get("value")))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                items_kv.append((entry[0], entry[1]))
    else:
        return {}
    for k, v in items_kv:
        key = _scalar(k)
        if key is None:
            continue
        key = key.rstrip(":").strip()
        if key and key not in pairs:
            pairs[key] = _scalar(v)
    return pairs


def coerce_prediction(raw: dict[str, Any], schema: FieldsSchema) -> dict[str, Any]:
    """One page's prediction with exactly the schema's fields, each in its ditto shape."""
    return {f.name: coerce_value(raw.get(f.name), field_kind(f)) for f in schema.fields}


# --------------------------------------------------------------------------- accounting


@dataclass
class UsageTally:
    """Token and cost totals for one baseline run.

    ``nominal_usd`` prices every token (cached or not) from ``config/prices.yaml``; it is what the
    harness divides by ``pages_run`` to extrapolate cost per 1,000 pages. ``spent_usd`` is the
    money this run actually spent (cache hits cost nothing).
    """

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    nominal_usd: float = 0.0
    spent_usd: float = 0.0
    estimated_price: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        nominal_usd: float,
        spent_usd: float,
        cached: bool = False,
        estimated: bool = False,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.cached_calls += int(cached)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.nominal_usd += nominal_usd
            self.spent_usd += spent_usd
            self.estimated_price = self.estimated_price or estimated

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}


class RecordingLLM:
    """Wraps a ditto ``LLM`` and records every ``Completion`` it returns (thread-safe)."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.model: str = inner.model
        self.role: Role = inner.role
        self.completions: list[Completion] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        completion = self.inner.complete(
            messages,
            json_schema=json_schema,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
            template_id=template_id,
        )
        with self._lock:
            self.completions.append(completion)
        return completion

    def drain(self) -> list[Completion]:
        """Return and forget the completions recorded so far."""
        with self._lock:
            out, self.completions = self.completions, []
        return out


def price(model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    """``(usd, estimated)`` for a call, from ``config/prices.yaml`` via the LLM layer."""
    from ditto.llm import estimate_cost

    usd, estimated = estimate_cost(model, input_tokens, output_tokens)
    return float(usd), bool(estimated)


def tally_completion(tally: UsageTally, completion: Completion) -> None:
    nominal, estimated = price(
        completion.model, completion.usage.input_tokens, completion.usage.output_tokens
    )
    tally.add(
        completion.usage.input_tokens,
        completion.usage.output_tokens,
        nominal_usd=nominal,
        spent_usd=0.0 if completion.cached else float(completion.cost_usd),
        cached=completion.cached,
        estimated=estimated or completion.cost_estimated,
    )


# --------------------------------------------------------------------------- model access

FLEX_TIMEOUT_S = 900.0  # OpenAI recommends a 15-minute timeout for Flex processing


class _TieredCompletions:
    def __init__(self, inner: Any, service_tier: str) -> None:
        self._inner = inner
        self._tier = service_tier

    def create(self, **kwargs: Any) -> Any:
        return self._inner.create(service_tier=self._tier, **kwargs)


class TieredClient:
    """An ``openai.OpenAI`` stand-in that adds ``service_tier`` to every chat completion."""

    def __init__(self, client: Any, service_tier: str) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _TieredCompletions(client.chat.completions, service_tier)
        self.models = client.models


def baseline_llm(settings: Any = None, *, service_tier: str | None = None) -> LLM:
    """``ditto.llm.get_llm("baseline")``, optionally on an OpenAI service tier (``"flex"``).

    The tier only changes billing and latency, not the model or its answers, so cache keys
    are shared with the default tier. Spend is still recorded at the standard price from
    ``config/prices.yaml`` (Flex bills about half of that), which keeps the budget guard
    conservative and ``cost_usd`` comparable with ditto's own calls.
    """
    from ditto.llm import get_llm, load_settings, resolve_model

    s = settings if settings is not None else load_settings()
    if service_tier is None or s.offline or not s.openai_api_key:
        return get_llm("baseline", settings=s)
    if not resolve_model("baseline", s).startswith("openai/"):
        return get_llm("baseline", settings=s)
    import openai

    from ditto.llm.openai_compat import OpenAICompatAdapter

    timeout = max(float(s.request_timeout_s), FLEX_TIMEOUT_S)
    client = openai.OpenAI(
        api_key=s.openai_api_key, base_url=s.openai_base_url, timeout=timeout, max_retries=0
    )
    adapter = OpenAICompatAdapter(
        api_key=s.openai_api_key,
        base_url=s.openai_base_url,
        timeout_s=timeout,
        client=cast(Any, TieredClient(client, service_tier)),
    )
    return get_llm("baseline", settings=s, adapter=adapter)


# --------------------------------------------------------------------------- artifacts


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_summary(out_dir: Path, name: str, output: BaselineOutput, extra: dict[str, Any]) -> Path:
    """Write ``<out_dir>/<name>.summary.json`` with the usage totals, notes and ``extra``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.summary.json"
    data = {
        "baseline": name,
        "model": output.model,
        "pages_run": output.pages_run,
        "calls": output.calls,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "cost_usd": output.cost_usd,
        "spend_usd": output.spend_usd,
        "notes": output.notes,
        **extra,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


def predictions_rows(
    pages: Sequence[PageInput], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {"key": p.key, "url": p.url, "template_id": p.template_id, **predictions[p.key]}
        for p in pages
        if p.key in predictions
    ]
