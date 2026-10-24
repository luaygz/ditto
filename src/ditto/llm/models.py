"""Model resolution per role and automatic model selection from ``GET {base_url}/models``.

Ranking rules (main model):

1. Keep ids of the form ``gpt-<version>...`` (``gpt-live-1``/``gpt-audio`` have no version
   right after ``gpt-`` and are dropped). Drop ids containing ``mini``, ``nano``, ``realtime``,
   ``audio``, ``image``, ``embedding``, ``tts``, ``transcribe``, ``search``, ``preview``.
2. General-purpose ids rank above special-purpose variants: ``-pro`` (Responses-API-only on
   OpenAI and ~10x the price), ``-chat-latest`` (a moving ChatGPT alias, not meant for pinned
   API use), ``-codex`` (code-tuned, Responses-only), ``-deep-research`` and ``-instruct``
   (legacy completions endpoint). Those are used only when nothing general is left.
3. Then the highest version, compared numerically (``5.10`` > ``5.9``).
4. Then the undated alias over its dated snapshots (``gpt-5.4`` over ``gpt-5.4-2026-03-05``:
   same weights, and the alias is what the price table lists); among snapshots the newest date.
5. Then the newest ``created`` timestamp from ``/models`` (same-version variants such as
   ``gpt-6-sol`` / ``gpt-6-astra``), then the id for determinism.

The small model follows the same rules except ``mini`` is not dropped and ids containing
``mini`` rank first. The top candidate must answer a JSON-only prompt (next candidates are
tried on failure, at most three); the choice is cached per base URL in ``<cache>/models.json``.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from ditto.llm.base import CacheMiss, Completion, LLMError, Message, Role
from ditto.llm.settings import Settings, load_settings, model_kind, normalize_model_ref
from ditto.llm.structured import parse_json
from ditto.run import write_json

Kind = Literal["main", "small"]

DROP_WORDS = (
    "nano",
    "realtime",
    "audio",
    "image",
    "embedding",
    "tts",
    "transcribe",
    "search",
    "preview",
)
_VERSIONED = re.compile(r"^gpt-(\d+(?:\.\d+)*)(.*)$")
_DATED = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{4})$")
_SPECIAL_TOKENS = frozenset({"pro", "codex", "instruct"})
_SPECIAL_PHRASES = ("chat-latest", "deep-research")
MAX_CONFIRM_ATTEMPTS = 3

CONFIRM_MESSAGES: list[Message] = [
    {
        "role": "user",
        "content": 'Reply with JSON only, no prose and no code fences: {"ok": true}',
    }
]
CONFIRM_MAX_TOKENS = 512  # reasoning models spend part of this before answering


class ModelLister(Protocol):
    """An adapter that can list models and answer a chat request."""

    def list_models(self) -> list[tuple[str, int]]: ...

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> Completion: ...


def _rank_key(model_id: str, created: int, *, small: bool) -> tuple[Any, ...] | None:
    lowered = model_id.lower()
    m = _VERSIONED.match(lowered)
    if not m:
        return None
    if any(word in lowered for word in DROP_WORDS):
        return None
    is_mini = "mini" in lowered
    if is_mini and not small:
        return None
    version = tuple(int(part) for part in m.group(1).split("."))
    rest = m.group(2)
    dated = _DATED.search(rest)
    base_rest = rest[: dated.start()] if dated else rest
    tokens = set(base_rest.strip("-").split("-"))
    special = bool(tokens & _SPECIAL_TOKENS) or any(p in base_rest for p in _SPECIAL_PHRASES)
    date = dated.group(1) if dated else ""
    # Larger sorts first (we sort descending): mini first for the small model, general before
    # special, higher version, undated alias before snapshots, newer snapshot, newer release.
    return (is_mini if small else False, not special, version, not dated, date, created, model_id)


def rank_models(
    ids: Iterable[str], *, small: bool = False, created: Mapping[str, int] | None = None
) -> list[str]:
    """Eligible model ids, best first (see the module docstring for the rules)."""
    created = created or {}
    keyed = []
    for model_id in set(ids):
        key = _rank_key(model_id, created.get(model_id, 0), small=small)
        if key is not None:
            keyed.append((key, model_id))
    keyed.sort(reverse=True)
    return [model_id for _, model_id in keyed]


def confirm_json(adapter: ModelLister, model: str) -> Completion:
    """One JSON-only test call; raises :class:`LLMError` unless the answer is ``{"ok": true}``."""
    completion = adapter.complete(
        CONFIRM_MESSAGES, model=model, json_schema=None, max_tokens=CONFIRM_MAX_TOKENS
    )
    try:
        value = parse_json(completion.text)
    except ValueError as exc:
        raise LLMError(f"{model} did not answer the JSON-only test prompt: {exc}") from exc
    if not (isinstance(value, dict) and value.get("ok") is True):
        raise LLMError(f"{model} answered the JSON-only test prompt with {completion.text!r}")
    return completion


def select_model(
    adapter: ModelLister,
    kind: Kind,
    *,
    on_confirm: Callable[[Completion], None] | None = None,
) -> dict[str, Any]:
    """Rank the endpoint's models and confirm the best one. Returns the record cached in
    ``models.json``: ``{"id", "selected_at", "candidates", "rejected"}``."""
    listed = adapter.list_models()
    created = dict(listed)
    ranked = rank_models(created, small=kind == "small", created=created)
    if not ranked:
        raise LLMError(
            f"no eligible gpt- model for the {kind} role among {len(listed)} listed models; "
            f"set {'OPENAI_MODEL' if kind == 'main' else 'OPENAI_MODEL_SMALL'}"
        )
    rejected: dict[str, str] = {}
    for model in ranked[:MAX_CONFIRM_ATTEMPTS]:
        try:
            completion = confirm_json(adapter, model)
        except LLMError as exc:
            rejected[model] = str(exc)
            continue
        if on_confirm is not None:
            on_confirm(completion)
        return {
            "id": model,
            "selected_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "candidates": ranked[:5],
            "rejected": rejected,
        }
    raise LLMError(f"no {kind} model passed the JSON-only test call: {rejected}")


_SELECT_LOCK = threading.Lock()


def cached_selection(settings: Settings, kind: Kind) -> str | None:
    """The cached automatic choice for ``kind`` at the configured base URL, if any."""
    path = settings.models_json
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = (data.get(settings.openai_base_url) or {}).get(kind) or {}
    model_id = entry.get("id")
    return str(model_id) if model_id else None


def store_selection(settings: Settings, kind: Kind, record: dict[str, Any]) -> None:
    path = settings.models_json
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    data.setdefault(settings.openai_base_url, {})[kind] = record
    write_json(path, data)


def resolve_model(
    role: Role,
    settings: Settings | None = None,
    *,
    adapter: ModelLister | None = None,
) -> str:
    """Effective ``"provider/model"`` for ``role``.

    Precedence: ``DITTO_MODEL_<ROLE>`` override, ``OPENAI_MODEL`` / ``OPENAI_MODEL_SMALL``, the
    cached automatic choice, then automatic selection (network; needs ``OPENAI_API_KEY``).
    """
    settings = settings or load_settings()
    override = settings.models.get(role)
    if override:
        return normalize_model_ref(override)
    kind: Kind = "main" if model_kind(role) == "main" else "small"
    explicit = settings.openai_model if kind == "main" else settings.openai_model_small
    if explicit:
        return f"openai/{explicit}"
    with _SELECT_LOCK:
        chosen = cached_selection(settings, kind)
        if chosen:
            return f"openai/{chosen}"
        if settings.offline:
            raise CacheMiss(
                f"offline mode: no {kind} model configured for role {role!r}; set "
                f"{'OPENAI_MODEL' if kind == 'main' else 'OPENAI_MODEL_SMALL'} or provide "
                f"{settings.models_json}"
            )
        if adapter is None:
            if not settings.openai_api_key:
                raise LLMError(
                    f"OPENAI_API_KEY is not set: cannot auto-select the {kind} model "
                    f"(set {'OPENAI_MODEL' if kind == 'main' else 'OPENAI_MODEL_SMALL'} "
                    "and a key, or use DITTO_LLM_OFFLINE=1 with recorded responses)"
                )
            from ditto.llm.openai_compat import OpenAICompatAdapter  # heavy SDK import

            adapter = OpenAICompatAdapter(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout_s=settings.request_timeout_s,
            )
        from ditto.llm.client import record_selection_spend  # client imports this module

        record = select_model(
            adapter, kind, on_confirm=lambda c: record_selection_spend(settings, c)
        )
        store_selection(settings, kind, record)
        return f"openai/{record['id']}"
