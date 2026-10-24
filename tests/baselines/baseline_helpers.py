"""Helpers for the baseline tests: load ``eval/baselines/*.py`` by file path (exactly like the
evaluation harness) and a fake ``LLM``."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from ditto.llm.base import Completion, Message, Usage

BASELINES_DIR = Path(__file__).resolve().parents[2] / "eval" / "baselines"


def load_module(name: str) -> ModuleType:
    module_name = f"ditto_eval_baseline_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, BASELINES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeLLM:
    """An ``LLM`` that answers from a function of the messages and counts tokens crudely."""

    def __init__(
        self,
        answer: Callable[[list[Message]], str],
        *,
        model: str = "openai/gpt-6.1-sol",
        cached: bool = False,
    ) -> None:
        self.model = model
        self.role = "baseline"
        self.answer = answer
        self.cached = cached
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        self.calls.append(
            {"messages": messages, "prompt_version": prompt_version, "template_id": template_id}
        )
        text = self.answer(messages)
        usage = Usage(
            input_tokens=sum(len(m["content"]) for m in messages) // 4,
            output_tokens=len(text) // 4,
        )
        return Completion(
            text=text,
            usage=usage,
            model=self.model,
            cached=self.cached,
            cost_usd=0.0 if self.cached else 0.001,
            meta={"finish_reason": "stop"},
        )
