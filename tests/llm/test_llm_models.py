from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_fakes import FakeAdapter, make_settings

from ditto.llm.base import CacheMiss, LLMError
from ditto.llm.models import rank_models, resolve_model, select_model

# A subset of what GET https://api.openai.com/v1/models returned on 2026-10-24.
LISTED = [
    "babbage-002",
    "chat-latest",
    "chatgpt-image-latest",
    "computer-use-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-instruct",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-search-preview",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-tts",
    "gpt-4o-search-preview",
    "gpt-4o-transcribe",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-codex",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5-search-api",
    "gpt-5.1-chat-latest",
    "gpt-5.1-codex-mini",
    "gpt-5.2-pro",
    "gpt-5.3-chat-latest",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-2026-04-23",
    "gpt-5.5-pro",
    "gpt-5.5-pro-2026-04-23",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-sol",
    "gpt-audio",
    "gpt-audio-mini",
    "gpt-image-1",
    "gpt-image-2.5-flare",
    "gpt-live-1",
    "gpt-live-transcribe",
    "gpt-realtime",
    "gpt-realtime-2.1",
    "gpt-realtime-2.1-mini",
    "gpt-realtime-translate",
    "gpt-transcribe",
    "o3",
    "o4-mini",
    "omni-moderation-latest",
    "text-embedding-3-large",
    "tts-1",
    "whisper-1",
]


def test_main_ranking_on_real_listing() -> None:
    ranked = rank_models(LISTED)
    assert ranked[0] == "gpt-6.1-sol"
    excluded = {
        "gpt-5.4-mini",
        "gpt-realtime-2.1-mini",
        "gpt-realtime-2.1",
        "gpt-4.1-nano",
        "gpt-4o-search-preview",
        "gpt-5-search-api",
        "gpt-image-2.5-flare",
        "gpt-audio",
        "gpt-4o-transcribe",
        "gpt-4o-mini-tts",
        "gpt-live-1",  # no version right after "gpt-"
        "gpt-transcribe",
        "chat-latest",
        "text-embedding-3-large",
        "o3",
    }
    assert not excluded & set(ranked)
    # every general-purpose id ranks above every special-purpose one
    special = [m for m in ranked if any(s in m for s in ("-pro", "chat-latest", "codex", "instr"))]
    assert ranked[-len(special) :] == special
    assert ranked.index("gpt-5.5") < ranked.index("gpt-5.5-2026-04-23") < ranked.index("gpt-5.4")


def test_small_ranking_prefers_mini() -> None:
    ranked = rank_models(LISTED, small=True)
    assert ranked[0] == "gpt-5.4-mini"
    assert ranked[:5] == [
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4o-mini",
    ]
    assert "gpt-realtime-2.1-mini" not in ranked
    assert "gpt-audio-mini" not in ranked
    assert "gpt-5.4-nano" not in ranked
    # the codex mini is special-purpose: after the general minis, before the non-minis
    assert ranked.index("gpt-5.1-codex-mini") == 5
    assert ranked[6] == "gpt-6.1-sol"


def test_small_falls_back_to_general_models_without_minis() -> None:
    assert rank_models(["gpt-5.5", "gpt-6-sol"], small=True) == ["gpt-6-sol", "gpt-5.5"]


def test_pro_chat_latest_and_snapshots_rank_after_general_ids() -> None:
    ids = [
        "gpt-7-pro",
        "gpt-7-chat-latest",
        "gpt-6.1-sol-2026-09-01",
        "gpt-6.1-sol",
        "gpt-6.1-sol-2026-10-01",
        "gpt-6",
    ]
    assert rank_models(ids) == [
        "gpt-6.1-sol",  # undated alias first
        "gpt-6.1-sol-2026-10-01",  # then the newest snapshot
        "gpt-6.1-sol-2026-09-01",
        "gpt-6",
        "gpt-7-pro",  # special-purpose only as a last resort, even at a higher version
        "gpt-7-chat-latest",
    ]
    assert rank_models(["gpt-7-pro"]) == ["gpt-7-pro"]


def test_versions_compare_numerically_and_created_breaks_ties() -> None:
    assert rank_models(["gpt-5.9", "gpt-5.10", "gpt-5"])[0] == "gpt-5.10"
    created = {"gpt-6-astra": 300, "gpt-6-sol": 200, "gpt-6-luna": 100}
    assert rank_models(created, created=created) == ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]
    # without timestamps the id decides, deterministically
    assert rank_models(["gpt-6-sol", "gpt-6-astra"]) == ["gpt-6-sol", "gpt-6-astra"]


def test_select_model_confirms_and_skips_failures() -> None:
    adapter = FakeAdapter(
        ["Sure! Here is JSON: maybe", '```json\n{"ok": true}\n```'],
        models=[("gpt-6.1-sol", 2), ("gpt-6-sol", 1), ("gpt-5.4-mini", 1)],
    )
    confirmed = []
    record = select_model(adapter, "main", on_confirm=confirmed.append)
    assert record["id"] == "gpt-6-sol"
    assert list(record["rejected"]) == ["gpt-6.1-sol"]
    assert record["candidates"][:2] == ["gpt-6.1-sol", "gpt-6-sol"]
    assert [c["model"] for c in adapter.calls] == ["gpt-6.1-sol", "gpt-6-sol"]
    assert "JSON only" in adapter.calls[0]["messages"][0]["content"]
    assert len(confirmed) == 1


def test_select_model_errors() -> None:
    with pytest.raises(LLMError, match="no eligible"):
        select_model(FakeAdapter(models=[("o3", 1), ("gpt-4o-mini", 1)]), "main")
    always_prose = FakeAdapter(default="no", models=[("gpt-5", 1)])
    with pytest.raises(LLMError, match="passed the JSON-only test"):
        select_model(always_prose, "main")
    wrong_json = FakeAdapter(default='{"ok": false}', models=[("gpt-5", 1)])
    with pytest.raises(LLMError):
        select_model(wrong_json, "main")


def test_resolve_model_precedence(tmp_path: Path) -> None:
    s = make_settings(tmp_path).with_models(repair="anthropic/claude-x")
    assert resolve_model("repair", s) == "anthropic/claude-x"
    assert resolve_model("generation", s) == "openai/gpt-test"
    assert resolve_model("baseline", s) == "openai/gpt-test"
    assert resolve_model("probe", s) == "openai/gpt-test-mini"
    assert resolve_model("tail_direct", s) == "openai/gpt-test-mini"


def test_auto_selection_is_cached_and_accounted(tmp_path: Path) -> None:
    s = make_settings(tmp_path, openai_model=None, openai_model_small=None)
    adapter = FakeAdapter(models=[(m, 0) for m in LISTED])
    assert resolve_model("generation", s, adapter=adapter) == "openai/gpt-6.1-sol"
    assert resolve_model("probe", s, adapter=adapter) == "openai/gpt-5.4-mini"
    assert len(adapter.calls) == 2  # one JSON-only confirmation per kind
    cached = json.loads(s.models_json.read_text())
    assert cached[s.openai_base_url]["main"]["id"] == "gpt-6.1-sol"
    assert cached[s.openai_base_url]["small"]["id"] == "gpt-5.4-mini"
    # cached now: no more calls, even without an adapter or a key
    no_key = make_settings(
        tmp_path, openai_model=None, openai_model_small=None, openai_api_key=None
    )
    assert resolve_model("repair", no_key) == "openai/gpt-6.1-sol"
    assert len(adapter.calls) == 2
    ledger = [json.loads(line) for line in s.ledger_path.read_text().splitlines()]
    assert [r["role"] for r in ledger] == ["model_selection", "model_selection"]
    assert {r["model"] for r in ledger} == {"openai/gpt-6.1-sol", "openai/gpt-5.4-mini"}
    assert all(r["cost_usd"] > 0 for r in ledger)


def test_selection_cache_is_per_base_url(tmp_path: Path) -> None:
    s = make_settings(tmp_path, openai_model=None)
    adapter = FakeAdapter(models=[("gpt-5", 0)])
    assert resolve_model("generation", s, adapter=adapter) == "openai/gpt-5"
    other = make_settings(tmp_path, openai_model=None, openai_base_url="http://local/v1")
    adapter2 = FakeAdapter(models=[("gpt-4.1", 0)])
    assert resolve_model("generation", other, adapter=adapter2) == "openai/gpt-4.1"


def test_offline_or_keyless_without_cached_choice(tmp_path: Path) -> None:
    offline = make_settings(tmp_path, openai_model=None, offline=True)
    with pytest.raises(CacheMiss, match="OPENAI_MODEL"):
        resolve_model("generation", offline)
    keyless = make_settings(tmp_path, openai_model_small=None, openai_api_key=None)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        resolve_model("probe", keyless)
