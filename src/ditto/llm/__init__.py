"""The LLM layer: one small interface, two thin adapters, and the shared cache, retries,
accounting and budget above them, so the provider is a config string (``provider/model``)."""

from __future__ import annotations

from ditto.llm.base import (
    LLM,
    ROLES,
    BudgetExceeded,
    CacheMiss,
    ChatAdapter,
    Completion,
    InvalidResponse,
    LLMError,
    Message,
    ProviderError,
    Role,
    Usage,
)
from ditto.llm.client import (
    BudgetStatus,
    CachedLLM,
    budget_status,
    cache_key,
    check_budget,
    get_llm,
    make_adapter,
    read_usage,
    record_external_usage,
    summarize_usage,
)
from ditto.llm.models import rank_models, resolve_model
from ditto.llm.pool import PoolResult, run_concurrently
from ditto.llm.prices import PriceTable, estimate_cost, load_prices
from ditto.llm.settings import Settings, load_settings
from ditto.llm.structured import complete_json, parse_json, strip_code_fences

__all__ = [
    "LLM",
    "ROLES",
    "BudgetExceeded",
    "BudgetStatus",
    "CacheMiss",
    "CachedLLM",
    "ChatAdapter",
    "Completion",
    "InvalidResponse",
    "LLMError",
    "Message",
    "PoolResult",
    "PriceTable",
    "ProviderError",
    "Role",
    "Settings",
    "Usage",
    "budget_status",
    "cache_key",
    "check_budget",
    "complete_json",
    "estimate_cost",
    "get_llm",
    "load_prices",
    "load_settings",
    "make_adapter",
    "parse_json",
    "rank_models",
    "read_usage",
    "record_external_usage",
    "resolve_model",
    "run_concurrently",
    "strip_code_fences",
    "summarize_usage",
]
