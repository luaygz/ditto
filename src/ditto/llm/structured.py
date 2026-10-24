"""JSON answers without provider structured output: ask JSON only, strip code fences, parse,
validate, one retry with the exact errors, then :class:`InvalidResponse`."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import ValidationError

from ditto.llm.base import LLM, InvalidResponse, Message

T = TypeVar("T")

JSON_ONLY_INSTRUCTION = (
    "Answer with a single JSON value only: no prose before or after it and no Markdown code fences."
)
RETRY_INSTRUCTION = (
    "Your previous answer could not be used. Errors:\n{errors}\n\n"
    "Return the complete corrected JSON only: no prose, no code fences."
)
_FENCE = re.compile(r"```[ \t]*(?:json|JSON|json5)?[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """The body of the first Markdown code fence in ``text``, else ``text`` stripped.

    Handles ```` ```json ... ``` ````, bare ```` ``` ```` fences, prose around the fence and a
    missing closing fence (truncated answers).
    """
    stripped = text.strip().lstrip("\ufeff")
    match = _FENCE.search(stripped)
    if match:
        return match.group(1).strip()
    if stripped.startswith("```"):  # opening fence without a closing one
        first_newline = stripped.find("\n")
        return stripped[first_newline + 1 :].strip() if first_newline >= 0 else ""
    return stripped


def parse_json(text: str) -> Any:
    """Parse a model answer as JSON, defensively.

    Tries the fence-stripped text, then the outermost ``{...}`` / ``[...]`` span (prose around
    the JSON). Raises ``ValueError`` with the parser's message for the first attempt.
    """
    body = strip_code_fences(text)
    if not body:
        raise ValueError("the answer is empty")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        first_error = exc
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = body.find(opener), body.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(body[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(
        f"not valid JSON: {first_error.msg} at line {first_error.lineno} column {first_error.colno}"
    )


def validation_errors(exc: Exception) -> list[str]:
    """Human/LLM-readable messages for an exception raised by a ``validate`` callable."""
    if isinstance(exc, ValidationError):
        out = []
        for err in exc.errors(include_url=False):
            loc = ".".join(str(part) for part in err.get("loc", ())) or "(root)"
            out.append(f"{loc}: {err.get('msg', 'invalid')}")
        return out or [str(exc)]
    if isinstance(exc, KeyError):
        return [f"missing key: {exc.args[0]!r}" if exc.args else "missing key"]
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines or [type(exc).__name__]


def with_json_instruction(messages: list[Message]) -> list[Message]:
    """Copy of ``messages`` whose system message asks for JSON only."""
    out: list[Message] = [{"role": m["role"], "content": m["content"]} for m in messages]
    if out and out[0]["role"] == "system":
        if JSON_ONLY_INSTRUCTION not in out[0]["content"]:
            out[0]["content"] = f"{out[0]['content'].rstrip()}\n\n{JSON_ONLY_INSTRUCTION}"
    else:
        out.insert(0, {"role": "system", "content": JSON_ONLY_INSTRUCTION})
    return out


class _Failed:
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors


def _attempt(text: str, validate: Callable[[Any], T], finish_reason: Any) -> T | _Failed:
    notes: list[str] = []
    if finish_reason == "length":
        notes.append("the answer was cut off at the output token limit; answer more compactly")
    try:
        data = parse_json(text)
    except ValueError as exc:
        return _Failed([*notes, str(exc)])
    try:
        return validate(data)
    except (ValueError, TypeError, KeyError) as exc:  # pydantic ValidationError is a ValueError
        return _Failed([*notes, *validation_errors(exc)])


def complete_json(
    llm: LLM,
    messages: list[Message],
    *,
    validate: Callable[[Any], T],
    prompt_version: str,
    template_id: str | None = None,
    max_tokens: int = 8192,
    json_schema: dict[str, Any] | None = None,
) -> T:
    """Ask ``llm`` for JSON, parse and ``validate`` it; one retry with the errors appended.

    ``validate`` receives the parsed JSON and returns the typed value, raising ``ValueError``
    (pydantic ``ValidationError`` included), ``TypeError`` or ``KeyError`` with messages on
    bad input. After a failed retry raises :class:`InvalidResponse` carrying the last errors
    and the last raw answer. LLM errors (budget, cache miss, provider) propagate unchanged.
    """
    msgs = with_json_instruction(messages)
    kwargs: dict[str, Any] = {
        "json_schema": json_schema,
        "max_tokens": max_tokens,
        "prompt_version": prompt_version,
        "template_id": template_id,
    }
    first = llm.complete(msgs, **kwargs)
    result = _attempt(first.text, validate, first.meta.get("finish_reason"))
    if not isinstance(result, _Failed):
        return result
    bullets = "\n".join(f"- {e}" for e in result.errors)
    retry: list[Message] = [
        *msgs,
        {"role": "assistant", "content": first.text},
        {"role": "user", "content": RETRY_INSTRUCTION.format(errors=bullets)},
    ]
    second = llm.complete(retry, **kwargs)
    result = _attempt(second.text, validate, second.meta.get("finish_reason"))
    if not isinstance(result, _Failed):
        return result
    errors = result.errors
    shown = "; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else "")
    raise InvalidResponse(
        f"{llm.model} returned invalid JSON twice (prompt {prompt_version or '?'}"
        f"{', template ' + template_id if template_id else ''}): {shown}",
        errors=errors,
        raw=second.text,
    )
