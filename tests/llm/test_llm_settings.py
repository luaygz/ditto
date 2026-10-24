from __future__ import annotations

from pathlib import Path

import pytest

from ditto.llm.base import ROLES
from ditto.llm.settings import (
    DEFAULT_BASE_URL,
    Settings,
    load_settings,
    model_kind,
    normalize_model_ref,
    split_model_ref,
)


def test_defaults_without_env_or_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={}, dotenv=False)
    assert s.openai_api_key is None
    assert s.openai_base_url == DEFAULT_BASE_URL
    assert s.openai_model is None and s.openai_model_small is None
    assert s.budget_usd == 40.0
    assert s.run_budget_usd is None
    assert s.cache_dir == (tmp_path / ".cache").resolve()
    assert s.llm_cache_dir == s.cache_dir / "llm"
    assert s.ledger_path == s.cache_dir / "ledger.jsonl"
    assert s.models_json == s.cache_dir / "models.json"
    assert s.offline is False
    assert dict(s.models) == {}
    assert s.dotenv_path is None


def test_all_variables(tmp_path: Path) -> None:
    s = load_settings(
        env={
            "OPENAI_API_KEY": "sk-abc",
            "OPENAI_BASE_URL": "http://localhost:11434/v1/",
            "OPENAI_MODEL": "gpt-6.1-sol",
            "OPENAI_MODEL_SMALL": "gpt-5.4-mini",
            "LLM_BUDGET_USD": "12.5",
            "DITTO_RUN_BUDGET_USD": "1.25",
            "ANTHROPIC_API_KEY": "sk-ant",
            "DITTO_CACHE_DIR": str(tmp_path / "c"),
            "DITTO_LLM_OFFLINE": "yes",
            "DITTO_MODEL_GENERATION": "anthropic/claude-x",
            "DITTO_MODEL_PROBE": "gpt-5.4-nano",
            "DITTO_MODEL_BASELINE": "openai/anthropic/claude-via-openrouter",
            "DITTO_LLM_TIMEOUT": "30",
        },
        dotenv=False,
    )
    assert s.openai_api_key == "sk-abc"
    assert s.openai_base_url == "http://localhost:11434/v1"
    assert s.openai_model == "gpt-6.1-sol"
    assert s.openai_model_small == "gpt-5.4-mini"
    assert s.budget_usd == 12.5
    assert s.run_budget_usd == 1.25
    assert s.anthropic_api_key == "sk-ant"
    assert s.cache_dir == (tmp_path / "c").resolve()
    assert s.offline is True
    assert s.request_timeout_s == 30.0
    assert dict(s.models) == {
        "generation": "anthropic/claude-x",
        "probe": "openai/gpt-5.4-nano",
        "baseline": "openai/anthropic/claude-via-openrouter",
    }


def test_dotenv_searched_upward_and_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-from-file\nLLM_BUDGET_USD=5\nOPENAI_MODEL=gpt-file\n"
    )
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    s = load_settings(env={"LLM_BUDGET_USD": "3"})
    assert s.dotenv_path == (tmp_path / ".env").resolve()
    assert s.openai_api_key == "sk-from-file"
    assert s.openai_model == "gpt-file"
    assert s.budget_usd == 3.0  # the environment wins over .env


def test_explicit_dotenv_path(tmp_path: Path) -> None:
    f = tmp_path / "custom.env"
    f.write_text("DITTO_LLM_OFFLINE=1\n")
    assert load_settings(env={}, dotenv=f).offline is True
    assert load_settings(env={}, dotenv=False).offline is False


def test_invalid_values_raise() -> None:
    with pytest.raises(ValueError, match="LLM_BUDGET_USD"):
        load_settings(env={"LLM_BUDGET_USD": "lots"}, dotenv=False)
    with pytest.raises(ValueError, match="unknown provider"):
        load_settings(env={"DITTO_MODEL_REPAIR": "gemini/pro"}, dotenv=False)


def test_repr_hides_keys() -> None:
    s = Settings(openai_api_key="sk-secret", anthropic_api_key="sk-ant-secret")
    assert "secret" not in repr(s)


def test_with_models_and_refs() -> None:
    s = Settings().with_models(generation="gpt-5.5", probe="anthropic/claude-haiku")
    assert dict(s.models) == {"generation": "openai/gpt-5.5", "probe": "anthropic/claude-haiku"}
    with pytest.raises(ValueError, match="unknown LLM role"):
        Settings().with_models(writer="gpt-5.5")
    assert normalize_model_ref(" gpt-4.1 ") == "openai/gpt-4.1"
    assert split_model_ref("openai/vendor/model") == ("openai", "vendor/model")
    with pytest.raises(ValueError, match="provider/model"):
        split_model_ref("openai/")


def test_role_to_model_kind() -> None:
    kinds = {role: model_kind(role) for role in ROLES}
    assert kinds == {
        "generation": "main",
        "repair": "main",
        "baseline": "main",
        "probe": "small",
        "tail_direct": "small",
    }
