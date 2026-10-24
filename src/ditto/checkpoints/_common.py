"""Small helpers shared by the checkpoint modules (console, value display, input parsing)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import orjson
from rich.console import Console

from ditto.textutil import normalize_ws

InputFn = Callable[[str], str]


def make_console(console: Console | None) -> Console:
    """The console checkpoints print to: the given one, else stderr (stdout stays clean)."""
    return console if console is not None else Console(file=sys.stderr, highlight=False)


def short(value: Any, limit: int = 80) -> str:
    """One-line display form of an extracted value, truncated to ``limit`` characters."""
    if value is None:
        return "—"
    text = value if isinstance(value, str) else orjson.dumps(value).decode()
    text = normalize_ws(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def ask(input_fn: InputFn, prompt: str) -> str:
    """Read one answer; end of input counts as an empty answer (accept the default)."""
    try:
        return input_fn(prompt).strip()
    except EOFError:
        return ""


def split_names(text: str) -> list[str]:
    """``"price, company title"`` -> ``["price", "company", "title"]``."""
    return [p for p in text.replace(",", " ").split() if p]
