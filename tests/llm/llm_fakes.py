"""Fake provider adapter for LLM-layer tests (no network)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ditto.llm.base import Completion, Message, Usage
from ditto.llm.prices import Price, PriceTable
from ditto.llm.settings import Settings

Scripted = str | Completion | BaseException | Callable[[list[Message]], Any]

PRICES = PriceTable(
    models={
        "openai/gpt-test": Price(input=2.0, output=10.0, cached_input=0.5),
        "openai/gpt-test-mini": Price(input=0.5, output=2.0),
        "openai/gpt-guess": Price(input=1.0, output=1.0, estimated=True),
    },
    default=Price(input=10.0, output=50.0, estimated=True),
)


DEFAULT_USAGE = Usage(1000, 500, 0)


class FakeAdapter:
    """Answers from a script (strings, Completions, exceptions or callables), then a default."""

    provider = "openai"

    def __init__(
        self,
        script: list[Scripted] | None = None,
        *,
        default: str = '{"ok": true}',
        usage: Usage = DEFAULT_USAGE,
        models: list[tuple[str, int]] | None = None,
    ) -> None:
        self.script = list(script or [])
        self.default = default
        self.usage = usage
        self.models = models or []
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> Completion:
        with self._lock:
            self.calls.append(
                {
                    "messages": [dict(m) for m in messages],
                    "model": model,
                    "json_schema": json_schema,
                    "max_tokens": max_tokens,
                }
            )
            item: Any = self.script.pop(0) if self.script else self.default
        if callable(item) and not isinstance(item, BaseException):
            item = item(messages)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Completion):
            return item
        return Completion(
            text=str(item),
            usage=self.usage,
            model=f"openai/{model}",
            latency_s=0.01,
            meta={"finish_reason": "stop"},
        )

    def list_models(self) -> list[tuple[str, int]]:
        return list(self.models)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "openai_api_key": "sk-test",
        "openai_model": "gpt-test",
        "openai_model_small": "gpt-test-mini",
        "cache_dir": tmp_path / "cache",
        "budget_usd": 10.0,
    }
    base.update(overrides)
    return Settings(**base)
