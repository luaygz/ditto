"""LLM settings from ``.env`` and the environment.

Variables (environment wins over ``.env``; ``.env`` is searched from the cwd upward):

- ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``)
- ``OPENAI_MODEL`` / ``OPENAI_MODEL_SMALL``: bare model ids at the OpenAI-compatible endpoint;
  auto-selected from ``GET {base_url}/models`` when unset (see :mod:`ditto.llm.models`)
- ``LLM_BUDGET_USD`` (default 40): global budget over the spend ledger in the cache dir
- ``DITTO_RUN_BUDGET_USD``: optional per-run budget over the run's ``usage.jsonl``
- ``ANTHROPIC_API_KEY``: for ``anthropic/...`` models
- ``DITTO_CACHE_DIR`` (default ``.cache``), ``DITTO_LLM_OFFLINE`` (recorded responses only)
- ``DITTO_MODEL_<ROLE>``: per-role override as ``provider/model`` (a bare id means ``openai/``)
- ``DITTO_LLM_TIMEOUT``: request timeout in seconds (default 600)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import dotenv_values, find_dotenv

from ditto.llm.base import ROLES, Role

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_BUDGET_USD = 40.0
DEFAULT_TIMEOUT_S = 600.0
PROVIDERS = ("openai", "anthropic")

# Roles served by the main model; the rest use the small model.
MAIN_ROLES: frozenset[Role] = frozenset({"generation", "repair", "baseline"})
SMALL_ROLES: frozenset[Role] = frozenset({"probe", "tail_direct"})

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Resolved LLM configuration. Build with :func:`load_settings`, or directly in tests."""

    openai_api_key: str | None = field(default=None, repr=False)
    openai_base_url: str = DEFAULT_BASE_URL
    openai_model: str | None = None  # bare id, e.g. "gpt-6.1-sol"
    openai_model_small: str | None = None
    anthropic_api_key: str | None = field(default=None, repr=False)
    budget_usd: float = DEFAULT_BUDGET_USD
    run_budget_usd: float | None = None
    cache_dir: Path = Path(".cache")
    offline: bool = False
    models: Mapping[Role, str] = field(default_factory=dict)  # per-role "provider/model"
    request_timeout_s: float = DEFAULT_TIMEOUT_S
    dotenv_path: Path | None = None  # the .env file that was read, if any

    @property
    def llm_cache_dir(self) -> Path:
        """Response cache: ``<cache>/llm/<sha256>.json``."""
        return self.cache_dir / "llm"

    @property
    def ledger_path(self) -> Path:
        """Global ledger of real (non-cached) spend, one JSON line per call."""
        return self.cache_dir / "ledger.jsonl"

    @property
    def models_json(self) -> Path:
        """Cached automatic model choice per base URL."""
        return self.cache_dir / "models.json"

    def with_models(self, **models: str) -> Settings:
        """Copy with per-role model overrides, e.g. ``with_models(generation="openai/gpt-5.5")``."""
        merged: dict[Role, str] = dict(self.models)
        for role, ref in models.items():
            merged[_check_role(role)] = normalize_model_ref(ref)
        return replace(self, models=merged)


def load_settings(
    env: Mapping[str, str] | None = None, *, dotenv: bool | str | Path = True
) -> Settings:
    """Read settings from ``env`` (default ``os.environ``) plus a ``.env`` file.

    ``dotenv=True`` searches ``.env`` from the current directory upward; a path reads that
    file; ``False`` reads none. Values already in ``env`` win over the file.
    """
    source: dict[str, str] = {}
    dotenv_path: Path | None = None
    if dotenv is not False:
        found = find_dotenv(usecwd=True) if dotenv is True else str(dotenv)
        if found and Path(found).is_file():
            dotenv_path = Path(found).resolve()
            source.update({k: v for k, v in dotenv_values(dotenv_path).items() if v is not None})
    source.update(os.environ if env is None else env)

    def get(name: str) -> str | None:
        value = source.get(name, "").strip()
        return value or None

    models: dict[Role, str] = {}
    for role in ROLES:
        ref = get(f"DITTO_MODEL_{role.upper()}")
        if ref:
            models[role] = normalize_model_ref(ref)

    run_budget = get("DITTO_RUN_BUDGET_USD")
    return Settings(
        openai_api_key=get("OPENAI_API_KEY"),
        openai_base_url=(get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        openai_model=get("OPENAI_MODEL"),
        openai_model_small=get("OPENAI_MODEL_SMALL"),
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        budget_usd=_float(get("LLM_BUDGET_USD"), "LLM_BUDGET_USD", DEFAULT_BUDGET_USD),
        run_budget_usd=_float(run_budget, "DITTO_RUN_BUDGET_USD", 0.0) if run_budget else None,
        cache_dir=Path(get("DITTO_CACHE_DIR") or ".cache").expanduser().resolve(),
        offline=(get("DITTO_LLM_OFFLINE") or "").lower() in _TRUE,
        models=models,
        request_timeout_s=_float(get("DITTO_LLM_TIMEOUT"), "DITTO_LLM_TIMEOUT", DEFAULT_TIMEOUT_S),
        dotenv_path=dotenv_path,
    )


def normalize_model_ref(ref: str) -> str:
    """``"gpt-5.5"`` -> ``"openai/gpt-5.5"``; ``"anthropic/claude-x"`` unchanged.

    Everything after the first ``/`` is the provider's model id, so OpenRouter ids such as
    ``openai/anthropic/claude-sonnet`` keep their own slash.
    """
    ref = ref.strip()
    if "/" not in ref:
        return f"openai/{ref}"
    provider, model = split_model_ref(ref)
    return f"{provider}/{model}"


def split_model_ref(ref: str) -> tuple[str, str]:
    """``"provider/model"`` -> ``(provider, model)``; raises ValueError for unknown providers."""
    provider, sep, model = ref.partition("/")
    if not sep or not model:
        raise ValueError(f"model {ref!r} is not of the form provider/model")
    if provider not in PROVIDERS:
        raise ValueError(
            f"model {ref!r}: unknown provider {provider!r} (known: {', '.join(PROVIDERS)}); "
            "OpenAI-compatible servers (OpenRouter, vLLM, Ollama, Together) use openai/ "
            "with OPENAI_BASE_URL"
        )
    return provider, model


def model_kind(role: Role) -> str:
    """``"main"`` for generation/repair/baseline, ``"small"`` for probe/tail_direct."""
    return "main" if role in MAIN_ROLES else "small"


def _check_role(role: str) -> Role:
    for known in ROLES:
        if known == role:
            return known
    raise ValueError(f"unknown LLM role {role!r} (known: {', '.join(ROLES)})")


def _float(value: str | None, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a number") from None
