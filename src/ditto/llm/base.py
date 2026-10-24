"""The small LLM interface every adapter implements and every stage calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

Role = Literal["generation", "repair", "probe", "tail_direct", "baseline"]
ROLES: tuple[Role, ...] = ("generation", "repair", "probe", "tail_direct", "baseline")


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    """Token counts of one call. ``input_tokens`` is the total prompt size and *includes*
    ``cached_input_tokens`` (prompt tokens the provider served from its prompt cache)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    model: str  # "provider/model" as configured, e.g. "openai/gpt-6.1-sol"
    cached: bool = False  # served from the on-disk response cache (cost 0, tokens kept)
    cost_usd: float = 0.0  # actual spend of this call (0 when cached)
    cost_estimated: bool = False  # price entry marked estimated
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Base class for typed LLM failures the pipeline handles."""


class BudgetExceeded(LLMError):
    """Raised before a call that would exceed the remaining LLM budget.

    The keyword attributes carry the totals the CLI prints; all are optional so
    ``BudgetExceeded("message")`` keeps working.
    """

    def __init__(
        self,
        message: str,
        *,
        scope: str = "global",
        budget_usd: float | None = None,
        spent_usd: float | None = None,
        remaining_usd: float | None = None,
        projected_usd: float | None = None,
    ) -> None:
        super().__init__(message)
        self.scope = scope  # "global" (LLM_BUDGET_USD ledger) or "run" (per-run budget)
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.remaining_usd = remaining_usd
        self.projected_usd = projected_usd


class ProviderError(LLMError):
    """A provider request failed. ``retryable`` is set for 429 (rate limit), 5xx and
    connection errors; the shared layer retries those with backoff and raises the rest."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after


class InvalidResponse(LLMError):
    """The model's answer failed JSON parsing / validation twice (after one retry)."""

    def __init__(self, message: str, errors: list[str], raw: str) -> None:
        super().__init__(message)
        self.errors = errors
        self.raw = raw


class CacheMiss(LLMError):
    """Offline mode (recorded responses only) and the request is not in the cache."""


class ChatAdapter(Protocol):
    """Provider adapter: one raw request, no caching, no retries, no accounting."""

    provider: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> Completion: ...


class LLM(Protocol):
    """What pipeline stages depend on: ``complete(messages, *, json_schema, max_tokens)``."""

    model: str  # "provider/model"
    role: Role

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion: ...
