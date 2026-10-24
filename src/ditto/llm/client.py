"""The shared LLM layer above the adapters: response cache, retries, accounting, budget.

- Cache: ``<cache>/llm/<sha256>.json`` keyed by SHA-256 of (prompt version, ``provider/model``,
  full request: messages after rendering, ``max_tokens``, ``json_schema``). ``no_cache`` skips
  the lookup but still stores the fresh answer; offline mode answers from the cache only and
  raises :class:`CacheMiss` otherwise. Cached calls cost 0 and keep their original tokens.
- Retries: 429 and 5xx (and connection errors) with exponential backoff and jitter, honouring
  ``Retry-After``; other errors raise at once.
- Accounting: every call appends a line to ``run.usage_jsonl`` (when a run is given); real
  (non-cached) spend is appended to the global ledger ``<cache>/ledger.jsonl``.
- Budget: before each real call its worst-case cost (input estimated from characters, output at
  ``max_tokens``) is reserved against the global budget (``LLM_BUDGET_USD`` minus the ledger)
  and the optional per-run budget; :class:`BudgetExceeded` is raised when it does not fit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from ditto.llm.base import (
    ROLES,
    BudgetExceeded,
    CacheMiss,
    ChatAdapter,
    Completion,
    LLMError,
    Message,
    ProviderError,
    Role,
    Usage,
)
from ditto.llm.models import resolve_model
from ditto.llm.prices import PriceTable, estimate_cost, load_prices
from ditto.llm.settings import Settings, load_settings, split_model_ref
from ditto.run import RunDir

log = logging.getLogger("ditto.llm")

CACHE_FORMAT = 1
DEFAULT_MAX_ATTEMPTS = 6
CHARS_PER_TOKEN_ESTIMATE = 3.0  # conservative (HTML ~3.5, English ~4): the guard errs safe
SCHEMA_INSTRUCTION = "The JSON must validate against this JSON Schema:\n"


# -- adapters ------------------------------------------------------------------------------------
def make_adapter(provider: str, settings: Settings) -> ChatAdapter:
    """The adapter for ``provider`` built from ``settings`` (no network until a request)."""
    if provider == "openai":
        from ditto.llm.openai_compat import OpenAICompatAdapter

        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set (put it in .env or the environment)")
        return OpenAICompatAdapter(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_s=settings.request_timeout_s,
        )
    if provider == "anthropic":
        from ditto.llm.anthropic import AnthropicAdapter

        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set (needed for anthropic/ models)")
        return AnthropicAdapter(
            api_key=settings.anthropic_api_key, timeout_s=settings.request_timeout_s
        )
    raise LLMError(f"unknown provider {provider!r}")


# -- cache ---------------------------------------------------------------------------------------
def render_schema(messages: Sequence[Message], json_schema: dict[str, Any] | None) -> list[Message]:
    """Copy of ``messages`` with ``json_schema`` appended to the system message (structured
    output is never delegated to the provider, so the schema travels in the prompt)."""
    out: list[Message] = [{"role": m["role"], "content": m["content"]} for m in messages]
    if json_schema is None:
        return out
    text = SCHEMA_INSTRUCTION + json.dumps(json_schema, ensure_ascii=False, separators=(",", ":"))
    if out and out[0]["role"] == "system":
        out[0]["content"] = f"{out[0]['content'].rstrip()}\n\n{text}"
    else:
        out.insert(0, {"role": "system", "content": text})
    return out


def request_body(
    messages: Sequence[Message], *, max_tokens: int, json_schema: dict[str, Any] | None
) -> dict[str, Any]:
    """The provider-independent request that is cached and hashed."""
    return {
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "max_tokens": max_tokens,
        "json_schema": json_schema,
    }


def cache_key(prompt_version: str, model: str, body: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of (prompt version, ``provider/model``, request body)."""
    payload = {"v": CACHE_FORMAT, "prompt_version": prompt_version, "model": model, "request": body}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per response: ``<dir>/<key>.json``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = orjson.loads(self.path(key).read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, orjson.JSONDecodeError) as exc:
            log.warning("ignoring unreadable LLM cache entry %s: %s", key, exc)
            return None
        return data if isinstance(data.get("response"), dict) else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
        tmp.replace(self.path(key))


# -- spend logs, ledger, budget --------------------------------------------------------------------
class SpendLog:
    """An append-only JSONL file whose ``cost_usd`` values are summed incrementally.

    Used for the global ledger and for each run's ``usage.jsonl``. Other processes may append
    too: only complete lines are counted, and new lines are picked up on the next read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reserved = 0.0  # in-flight worst-case cost, guarded by _BUDGET_LOCK
        self._lock = threading.Lock()
        self._offset = 0
        self._spent = 0.0

    def spent(self) -> float:
        with self._lock:
            self._refresh()
            return self._spent

    def append(self, record: dict[str, Any]) -> None:
        line = orjson.dumps(record) + b"\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as fh:
                fh.write(line)

    def _refresh(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            self._offset, self._spent = 0, 0.0
            return
        if size < self._offset:  # truncated or replaced: recount
            self._offset, self._spent = 0, 0.0
        if size == self._offset:
            return
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read(size - self._offset)
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        for line in chunk[:end].splitlines():
            try:
                self._spent += float(orjson.loads(line).get("cost_usd") or 0.0)
            except (orjson.JSONDecodeError, AttributeError, TypeError, ValueError):
                continue
        self._offset += end + 1


_LOGS: dict[Path, SpendLog] = {}
_LOGS_LOCK = threading.Lock()
_BUDGET_LOCK = threading.Lock()


def spend_log(path: Path) -> SpendLog:
    """The process-wide :class:`SpendLog` for ``path`` (shared so reservations add up)."""
    key = path.resolve()
    with _LOGS_LOCK:
        if key not in _LOGS:
            _LOGS[key] = SpendLog(key)
        return _LOGS[key]


@dataclass(frozen=True)
class _Scope:
    name: str  # "global" | "run"
    log: SpendLog
    budget_usd: float
    hint: str


def _scopes(settings: Settings, run: RunDir | None) -> list[_Scope]:
    scopes = [
        _Scope("global", spend_log(settings.ledger_path), settings.budget_usd, "LLM_BUDGET_USD")
    ]
    if run is not None and settings.run_budget_usd is not None:
        scopes.append(
            _Scope("run", spend_log(run.usage_jsonl), settings.run_budget_usd, "the run budget")
        )
    return scopes


def _check(scope: _Scope, projected_usd: float) -> None:
    """Raise :class:`BudgetExceeded` unless ``projected_usd`` fits. Call under _BUDGET_LOCK."""
    spent = scope.log.spent()
    committed = spent + scope.log.reserved
    remaining = scope.budget_usd - committed
    if projected_usd <= remaining + 1e-12:
        return
    in_flight = f", ${scope.log.reserved:.4f} in flight" if scope.log.reserved > 0 else ""
    what = "LLM budget" if scope.name == "global" else "per-run LLM budget"
    raise BudgetExceeded(
        f"{what} exceeded: projected ${projected_usd:.4f}, but only ${max(remaining, 0.0):.4f} "
        f"of ${scope.budget_usd:.2f} remains (spent ${spent:.4f}{in_flight}; "
        f"{scope.log.path}). Raise {scope.hint} or reduce the work.",
        scope=scope.name,
        budget_usd=scope.budget_usd,
        spent_usd=spent,
        remaining_usd=max(remaining, 0.0),
        projected_usd=projected_usd,
    )


def check_budget(
    projected_usd: float, *, settings: Settings | None = None, run: RunDir | None = None
) -> None:
    """Raise :class:`BudgetExceeded` (with spent/remaining/projected totals) when
    ``projected_usd`` exceeds the remaining global budget or the run's budget."""
    settings = settings or load_settings()
    with _BUDGET_LOCK:
        for scope in _scopes(settings, run):
            _check(scope, projected_usd)


@contextmanager
def _reserve(scopes: list[_Scope], projected_usd: float) -> Iterator[None]:
    with _BUDGET_LOCK:
        for scope in scopes:
            _check(scope, projected_usd)
        for scope in scopes:
            scope.log.reserved += projected_usd
    try:
        yield
    finally:
        with _BUDGET_LOCK:
            for scope in scopes:
                scope.log.reserved -= projected_usd


@dataclass(frozen=True)
class BudgetStatus:
    budget_usd: float
    spent_usd: float
    remaining_usd: float
    ledger_path: Path
    run_budget_usd: float | None = None
    run_spent_usd: float | None = None
    run_remaining_usd: float | None = None


def budget_status(settings: Settings | None = None, run: RunDir | None = None) -> BudgetStatus:
    """Global (and per-run, when a run is given) spend and what remains."""
    settings = settings or load_settings()
    spent = spend_log(settings.ledger_path).spent()
    run_spent = spend_log(run.usage_jsonl).spent() if run is not None else None
    run_budget = settings.run_budget_usd
    return BudgetStatus(
        budget_usd=settings.budget_usd,
        spent_usd=spent,
        remaining_usd=max(settings.budget_usd - spent, 0.0),
        ledger_path=settings.ledger_path,
        run_budget_usd=run_budget,
        run_spent_usd=run_spent,
        run_remaining_usd=(
            max(run_budget - run_spent, 0.0)
            if run_budget is not None and run_spent is not None
            else None
        ),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def record_external_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    role: str = "baseline",
    prompt_version: str = "",
    template_id: str | None = None,
    settings: Settings | None = None,
    run: RunDir | None = None,
) -> float:
    """Account a real call made outside this layer (e.g. crawl4ai via litellm, or the model
    selection test call): appends to the global ledger, and to ``run.usage_jsonl`` when a run
    is given. Returns its cost in USD from ``config/prices.yaml``."""
    settings = settings or load_settings()
    ref = model if "/" in model else f"openai/{model}"
    cost, estimated = estimate_cost(
        ref, input_tokens, output_tokens, cached_input_tokens=cached_input_tokens
    )
    line: dict[str, Any] = {
        "ts": _now(),
        "role": role,
        "template_id": template_id,
        "model": ref,
        "prompt_version": prompt_version,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cost_usd": cost,
        "list_cost_usd": cost,
        "estimated": estimated,
        "cached": False,
        "latency_s": None,
        "key": None,
    }
    spend_log(settings.ledger_path).append(
        {**line, "run": str(run.root) if run is not None else None}
    )
    if run is not None:
        spend_log(run.usage_jsonl).append(line)
    return cost


def record_selection_spend(settings: Settings, completion: Completion) -> None:
    """Ledger line for the JSON-only confirmation call of automatic model selection."""
    u = completion.usage
    record_external_usage(
        completion.model,
        u.input_tokens,
        u.output_tokens,
        cached_input_tokens=u.cached_input_tokens,
        role="model_selection",
        prompt_version="model_selection",
        settings=settings,
    )


# -- retries -------------------------------------------------------------------------------------
def backoff_delay(
    attempt: int,
    *,
    retry_after: float | None = None,
    base_s: float = 1.0,
    cap_s: float = 60.0,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait after failed ``attempt`` (1-based): exponential with equal jitter,
    ``[d/2, d]`` for ``d = min(cap, base * 2**(attempt-1))``, never below ``Retry-After``."""
    d = min(cap_s, base_s * 2.0 ** max(attempt - 1, 0))
    jittered = d / 2 + (rng or random).uniform(0, d / 2)
    return max(jittered, min(retry_after or 0.0, cap_s))


# -- the LLM -------------------------------------------------------------------------------------
class CachedLLM:
    """:class:`ditto.llm.base.LLM` implementation shared by every provider."""

    def __init__(
        self,
        *,
        model: str,
        role: Role,
        adapter: ChatAdapter | None,
        settings: Settings,
        run: RunDir | None = None,
        no_cache: bool = False,
        prices: PriceTable | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.provider, self.model_id = split_model_ref(model)
        self.model = f"{self.provider}/{self.model_id}"
        self.role: Role = role
        self.adapter = adapter
        self.settings = settings
        self.run = run
        self.no_cache = no_cache
        self.prices = prices or load_prices()
        self.max_attempts = max(1, max_attempts)
        self.cache = ResponseCache(settings.llm_cache_dir)
        self._sleep = sleep
        self._rng = rng or random.Random()

    def __repr__(self) -> str:
        return f"CachedLLM(model={self.model!r}, role={self.role!r})"

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        msgs = render_schema(messages, json_schema)
        body = request_body(msgs, max_tokens=max_tokens, json_schema=json_schema)
        key = cache_key(prompt_version, self.model, body)

        if not self.no_cache:
            start = time.perf_counter()
            entry = self.cache.get(key)
            if entry is not None:
                completion = self._from_cache(entry, key, time.perf_counter() - start)
                self._record_usage(completion, prompt_version, template_id)
                return completion
        if self.settings.offline or self.adapter is None:
            raise CacheMiss(
                f"offline mode: no recorded response for {self.model} (role {self.role}, "
                f"prompt {prompt_version or '?'}, template {template_id or '-'}) under "
                f"{self.cache.path(key)}"
            )

        projected = self.projected_cost(msgs, max_tokens)
        with _reserve(_scopes(self.settings, self.run), projected):
            start = time.perf_counter()
            raw, attempts = self._send(msgs, json_schema, max_tokens)
            latency = time.perf_counter() - start
            u = raw.usage
            cost, estimated = estimate_cost(
                self.model,
                u.input_tokens,
                u.output_tokens,
                cached_input_tokens=u.cached_input_tokens,
                prices=self.prices,
            )
            completion = replace(
                raw,
                model=self.model,
                cached=False,
                cost_usd=cost,
                cost_estimated=estimated,
                latency_s=latency,
                meta={**raw.meta, "cache_key": key, "attempts": attempts},
            )
            spend_log(self.settings.ledger_path).append(
                {
                    **self._line(completion, prompt_version, template_id, key),
                    "run": str(self.run.root) if self.run is not None else None,
                }
            )
            self._record_usage(completion, prompt_version, template_id)
        try:
            self.cache.put(key, self._cache_record(completion, key, prompt_version, body))
        except OSError as exc:
            log.warning("could not write LLM cache entry %s: %s", key, exc)
        return completion

    def projected_cost(self, messages: Sequence[Message], max_tokens: int) -> float:
        """Worst-case cost of one call: input from characters, output at ``max_tokens``."""
        chars = sum(len(m["content"]) for m in messages)
        input_tokens = int(chars / CHARS_PER_TOKEN_ESTIMATE) + 8 * len(messages)
        cost, _ = estimate_cost(self.model, input_tokens, max_tokens, prices=self.prices)
        return cost

    def _send(
        self, messages: list[Message], json_schema: dict[str, Any] | None, max_tokens: int
    ) -> tuple[Completion, int]:
        assert self.adapter is not None
        for attempt in range(1, self.max_attempts + 1):
            try:
                completion = self.adapter.complete(
                    messages, model=self.model_id, json_schema=json_schema, max_tokens=max_tokens
                )
                return completion, attempt
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_attempts:
                    raise
                delay = backoff_delay(attempt, retry_after=exc.retry_after, rng=self._rng)
                log.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    self.model,
                    attempt,
                    self.max_attempts,
                    exc,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _from_cache(self, entry: dict[str, Any], key: str, latency: float) -> Completion:
        response = entry["response"]
        u = response.get("usage") or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens", 0)),
            output_tokens=int(u.get("output_tokens", 0)),
            cached_input_tokens=int(u.get("cached_input_tokens", 0)),
        )
        _, estimated = estimate_cost(self.model, 0, 0, prices=self.prices)
        return Completion(
            text=str(response.get("text", "")),
            usage=usage,
            model=self.model,
            cached=True,
            cost_usd=0.0,
            cost_estimated=estimated,
            latency_s=latency,
            meta={
                **(response.get("meta") or {}),
                "cache_key": key,
                "original_latency_s": response.get("latency_s"),
                "original_cost_usd": response.get("cost_usd"),
            },
        )

    def _cache_record(
        self, completion: Completion, key: str, prompt_version: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        meta = {k: v for k, v in completion.meta.items() if k not in {"cache_key", "attempts"}}
        u = completion.usage
        return {
            "key": key,
            "format": CACHE_FORMAT,
            "created": _now(),
            "prompt_version": prompt_version,
            "model": self.model,
            "role": self.role,
            "request": body,
            "response": {
                "text": completion.text,
                "usage": {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cached_input_tokens": u.cached_input_tokens,
                },
                "meta": meta,
                "latency_s": round(completion.latency_s, 3),
                "cost_usd": completion.cost_usd,
                "cost_estimated": completion.cost_estimated,
            },
        }

    def _line(
        self, completion: Completion, prompt_version: str, template_id: str | None, key: str
    ) -> dict[str, Any]:
        u = completion.usage
        list_cost, estimated = estimate_cost(
            self.model,
            u.input_tokens,
            u.output_tokens,
            cached_input_tokens=u.cached_input_tokens,
            prices=self.prices,
        )
        return {
            "ts": _now(),
            "role": self.role,
            "template_id": template_id,
            "model": self.model,
            "prompt_version": prompt_version,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cached_input_tokens": u.cached_input_tokens,
            "cost_usd": completion.cost_usd,
            "list_cost_usd": list_cost,
            "estimated": estimated,
            "cached": completion.cached,
            "latency_s": round(completion.latency_s, 4),
            "key": key,
        }

    def _record_usage(
        self, completion: Completion, prompt_version: str, template_id: str | None
    ) -> None:
        if self.run is None:
            return
        key = str(completion.meta.get("cache_key", ""))
        spend_log(self.run.usage_jsonl).append(
            self._line(completion, prompt_version, template_id, key)
        )


def get_llm(
    role: Role,
    *,
    settings: Settings | None = None,
    run: RunDir | None = None,
    no_cache: bool = False,
    adapter: ChatAdapter | None = None,
) -> CachedLLM:
    """The configured LLM for ``role`` (``generation``, ``repair``, ``baseline`` use the main
    model; ``probe``, ``tail_direct`` the small one). ``adapter`` overrides the provider
    adapter (tests); in offline mode no adapter is built and misses raise :class:`CacheMiss`."""
    if role not in ROLES:
        raise ValueError(f"unknown LLM role {role!r} (known: {', '.join(ROLES)})")
    settings = settings or load_settings()
    if no_cache and settings.offline:
        raise LLMError("no_cache (--no-cache) cannot be combined with DITTO_LLM_OFFLINE")
    model = resolve_model(role, settings)
    provider, _ = split_model_ref(model)
    if adapter is None and not settings.offline:
        adapter = make_adapter(provider, settings)
    return CachedLLM(
        model=model, role=role, adapter=adapter, settings=settings, run=run, no_cache=no_cache
    )


# -- usage summary ---------------------------------------------------------------------------------
def _empty_row() -> dict[str, Any]:
    return {
        "calls": 0,
        "cached_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cost_usd": 0.0,
        "list_cost_usd": 0.0,
        "estimated": False,
    }


def _add(row: dict[str, Any], rec: dict[str, Any]) -> None:
    row["calls"] += 1
    row["cached_calls"] += 1 if rec.get("cached") else 0
    for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
        row[name] += int(rec.get(name) or 0)
    row["cost_usd"] += float(rec.get("cost_usd") or 0.0)
    row["list_cost_usd"] += float(rec.get("list_cost_usd") or rec.get("cost_usd") or 0.0)
    row["estimated"] = bool(row["estimated"] or rec.get("estimated"))


def read_usage(run: RunDir) -> list[dict[str, Any]]:
    """Every complete line of ``run.usage_jsonl`` (missing file: none)."""
    if not run.usage_jsonl.is_file():
        return []
    records = []
    for line in run.usage_jsonl.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            rec = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def summarize_usage(run: RunDir) -> dict[str, Any]:
    """Usage of a run aggregated per role, per template, per model and in total.

    Rows: ``calls, cached_calls, input_tokens, output_tokens, cached_input_tokens, cost_usd``
    (real spend; 0 for cached calls), ``list_cost_usd`` (what the same calls cost uncached) and
    ``estimated`` (any price entry estimated). Calls without a template are under ``"_none"``.
    """
    totals = _empty_row()
    by_role: dict[str, dict[str, Any]] = {}
    by_template: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    for rec in read_usage(run):
        _add(totals, rec)
        _add(by_role.setdefault(str(rec.get("role")), _empty_row()), rec)
        _add(by_template.setdefault(str(rec.get("template_id") or "_none"), _empty_row()), rec)
        _add(by_model.setdefault(str(rec.get("model")), _empty_row()), rec)
    for row in (totals, *by_role.values(), *by_template.values(), *by_model.values()):
        row["cost_usd"] = round(row["cost_usd"], 6)
        row["list_cost_usd"] = round(row["list_cost_usd"], 6)
    return {
        "site": run.site,
        "totals": totals,
        "by_role": dict(sorted(by_role.items())),
        "by_template": dict(sorted(by_template.items())),
        "by_model": dict(sorted(by_model.items())),
    }
