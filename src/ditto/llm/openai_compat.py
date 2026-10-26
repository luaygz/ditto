"""OpenAI-compatible chat adapter (OpenAI, OpenRouter, vLLM, Ollama, Together via ``base_url``).

Thin by design: one ``chat.completions`` request -> :class:`Completion` with usage. No caching,
no retries (the SDK's own retries are disabled), no accounting; :mod:`ditto.llm.client` does
those for every provider.
"""

from __future__ import annotations

import time
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam

from ditto.llm.base import Completion, Message, ProviderError, Usage


class OpenAICompatAdapter:
    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_s: float = 600.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0
        )

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> Completion:
        # ``json_schema`` is already rendered into the prompt by the shared layer: structured
        # output is never delegated to the provider, so it is not sent as ``response_format``.
        del json_schema
        params = cast(list[ChatCompletionMessageParam], [dict(m) for m in messages])
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=model, messages=params, max_completion_tokens=max_tokens
            )
        except openai.APIStatusError as exc:
            raise _status_error(exc) from exc
        except openai.APIConnectionError as exc:  # includes timeouts
            raise ProviderError(f"openai: {exc}", retryable=True) from exc
        latency = time.perf_counter() - start
        if not resp.choices:
            raise ProviderError(f"openai: response for {model} has no choices", retryable=True)
        choice = resp.choices[0]
        usage = resp.usage
        details = usage.prompt_tokens_details if usage else None
        return Completion(
            text=choice.message.content or "",
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                cached_input_tokens=(details.cached_tokens or 0) if details else 0,
            ),
            model=f"{self.provider}/{model}",
            latency_s=latency,
            meta={
                "finish_reason": choice.finish_reason,
                "response_model": resp.model,
                "response_id": resp.id,
                "refusal": choice.message.refusal,
                "service_tier": getattr(resp, "service_tier", None),
            },
        )

    def list_models(self) -> list[tuple[str, int]]:
        """``(id, created)`` of every model at ``{base_url}/models``."""
        try:
            return [(m.id, int(m.created or 0)) for m in self._client.models.list()]
        except openai.APIStatusError as exc:
            raise _status_error(exc) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"openai: {exc}", retryable=True) from exc


def _status_error(exc: openai.APIStatusError) -> ProviderError:
    status = exc.status_code
    code = exc.code
    # 429 "insufficient_quota" is a billing state, not a transient rate limit.
    retryable = (status == 429 and code != "insufficient_quota") or status >= 500
    return ProviderError(
        f"openai: HTTP {status}: {exc.message}",
        status=status,
        code=code,
        retryable=retryable,
        retry_after=retry_after_seconds(exc.response.headers),
    )


def retry_after_seconds(headers: Any) -> float | None:
    """Seconds from ``retry-after-ms`` / ``retry-after`` headers (numeric forms only)."""
    for name, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name) if headers is not None else None
        if value is None:
            continue
        try:
            seconds = float(value) / divisor
        except (TypeError, ValueError):
            continue
        if 0 <= seconds <= 300:
            return seconds
    return None
