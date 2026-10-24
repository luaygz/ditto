"""Native Anthropic adapter (``anthropic`` SDK, Messages API).

Thin by design: one ``messages.create`` request -> :class:`Completion` with usage. System
messages are joined into the ``system`` parameter; caching, retries and accounting live in
:mod:`ditto.llm.client`.
"""

from __future__ import annotations

import time
from typing import Any, cast

import anthropic
from anthropic.types import MessageParam

from ditto.llm.base import Completion, Message, ProviderError, Usage
from ditto.llm.openai_compat import retry_after_seconds


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic(
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
        del json_schema  # rendered into the prompt by the shared layer; never delegated
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = cast(
            list[MessageParam],
            [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m["role"] != "system"
            ],
        )
        start = time.perf_counter()
        try:
            if system:
                resp = self._client.messages.create(
                    model=model, max_tokens=max_tokens, messages=turns, system=system
                )
            else:
                resp = self._client.messages.create(
                    model=model, max_tokens=max_tokens, messages=turns
                )
        except anthropic.APIStatusError as exc:
            status = exc.status_code
            raise ProviderError(
                f"anthropic: HTTP {status}: {exc.message}",
                status=status,
                code=exc.type,
                retryable=status == 429 or status >= 500,  # 529 = overloaded
                retry_after=retry_after_seconds(exc.response.headers),
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"anthropic: {exc}", retryable=True) from exc
        latency = time.perf_counter() - start
        text = "".join(block.text for block in resp.content if block.type == "text")
        u = resp.usage
        cache_read = u.cache_read_input_tokens or 0
        cache_write = u.cache_creation_input_tokens or 0
        return Completion(
            text=text,
            # Anthropic reports cache reads/writes outside input_tokens; Usage.input_tokens
            # is the total prompt size, so add them back.
            usage=Usage(
                input_tokens=u.input_tokens + cache_read + cache_write,
                output_tokens=u.output_tokens,
                cached_input_tokens=cache_read,
            ),
            model=f"{self.provider}/{model}",
            latency_s=latency,
            meta={
                "finish_reason": "length" if resp.stop_reason == "max_tokens" else resp.stop_reason,
                "response_model": resp.model,
                "response_id": resp.id,
            },
        )
