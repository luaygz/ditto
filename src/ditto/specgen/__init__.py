"""Spec format, prompts, and stage 6 (spec generation) plus stage 10 (long tail).

``generate_run`` / ``tail_direct_run`` load lazily: importing ``ditto`` (which imports the spec
model) must not pull in the LLM SDKs or tokenizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ditto.specgen.prompts import Prompt, load_prompt

if TYPE_CHECKING:
    from ditto.specgen.generate import (
        GenerationResult,
        TemplateGeneration,
        generate_run,
        generate_spec,
    )
    from ditto.specgen.tail import TailResult, tail_direct_run

_LAZY = {
    "GenerationResult": "ditto.specgen.generate",
    "TemplateGeneration": "ditto.specgen.generate",
    "generate_run": "ditto.specgen.generate",
    "generate_spec": "ditto.specgen.generate",
    "TailResult": "ditto.specgen.tail",
    "tail_direct_run": "ditto.specgen.tail",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'ditto.specgen' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


__all__ = [
    "GenerationResult",
    "Prompt",
    "TailResult",
    "TemplateGeneration",
    "generate_run",
    "generate_spec",
    "load_prompt",
    "tail_direct_run",
]
