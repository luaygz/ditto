from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Any

import pytest
from llm_fakes import PRICES, FakeAdapter, make_settings

from ditto.llm.anthropic import AnthropicAdapter
from ditto.llm.base import (
    BudgetExceeded,
    CacheMiss,
    Completion,
    LLMError,
    Message,
    ProviderError,
    Usage,
)
from ditto.llm.client import (
    CachedLLM,
    _reserve,
    _scopes,
    backoff_delay,
    budget_status,
    cache_key,
    check_budget,
    get_llm,
    make_adapter,
    read_usage,
    record_external_usage,
    render_schema,
    request_body,
    spend_log,
    summarize_usage,
)
from ditto.llm.openai_compat import OpenAICompatAdapter
from ditto.llm.settings import Settings
from ditto.run import RunDir

MSGS: list[Message] = [
    {"role": "system", "content": "You write specs."},
    {"role": "user", "content": "Union page goes here."},
]


def _llm(
    tmp_path: Path,
    adapter: FakeAdapter | None,
    *,
    settings: Settings | None = None,
    run: RunDir | None = None,
    no_cache: bool = False,
    sleeps: list[float] | None = None,
    max_attempts: int = 6,
) -> CachedLLM:
    return CachedLLM(
        model="openai/gpt-test",
        role="generation",
        adapter=adapter,
        settings=settings or make_settings(tmp_path),
        run=run,
        no_cache=no_cache,
        prices=PRICES,
        max_attempts=max_attempts,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        rng=random.Random(0),
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- cache -----------------------------------------------------------------------------------------
def test_cache_miss_then_hit(tmp_path: Path) -> None:
    adapter = FakeAdapter(['{"a": 1}'])
    llm = _llm(tmp_path, adapter)
    first = llm.complete(MSGS, prompt_version="specgen.v1", template_id="t_1")
    assert first.text == '{"a": 1}'
    assert first.cached is False
    assert first.model == "openai/gpt-test"
    assert first.cost_usd == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)
    key = first.meta["cache_key"]
    assert (llm.settings.llm_cache_dir / f"{key}.json").is_file()

    second = llm.complete(MSGS, prompt_version="specgen.v1", template_id="t_1")
    assert len(adapter.calls) == 1
    assert second.cached is True
    assert second.text == first.text
    assert second.cost_usd == 0.0
    assert second.usage == first.usage  # original token counts kept
    assert second.meta["cache_key"] == key
    assert second.meta["original_cost_usd"] == pytest.approx(first.cost_usd)
    assert second.meta["finish_reason"] == "stop"

    # a new client over the same cache dir replays it too
    replay = _llm(tmp_path, FakeAdapter()).complete(
        MSGS, prompt_version="specgen.v1", template_id="other"
    )
    assert replay.cached and replay.text == '{"a": 1}'


def test_cache_key_is_stable_and_sensitive() -> None:
    body = request_body(MSGS, max_tokens=100, json_schema=None)
    key = cache_key("specgen.v1", "openai/gpt-test", body)
    # pinned: changing the key format invalidates every recorded fixture under tests/fixtures/llm
    assert key == "766943129ccf2e41278acb9ddd06ba57da59c8bd827c793d49c5e51e9168134e"
    assert key == cache_key("specgen.v1", "openai/gpt-test", json.loads(json.dumps(body)))
    variants = {
        cache_key("specgen.v2", "openai/gpt-test", body),
        cache_key("specgen.v1", "openai/gpt-other", body),
        cache_key(
            "specgen.v1", "openai/gpt-test", request_body(MSGS, max_tokens=101, json_schema=None)
        ),
        cache_key(
            "specgen.v1",
            "openai/gpt-test",
            request_body(MSGS[:1], max_tokens=100, json_schema=None),
        ),
        cache_key(
            "specgen.v1", "openai/gpt-test", request_body(MSGS, max_tokens=100, json_schema={})
        ),
    }
    assert key not in variants and len(variants) == 5


def test_template_id_and_role_do_not_change_the_key(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    llm = _llm(tmp_path, adapter)
    a = llm.complete(MSGS, prompt_version="p", template_id="t_a")
    b = llm.complete(MSGS, prompt_version="p", template_id="t_b")
    assert a.meta["cache_key"] == b.meta["cache_key"] and b.cached
    assert len(adapter.calls) == 1


def test_no_cache_skips_lookup_but_refreshes_entry(tmp_path: Path) -> None:
    _llm(tmp_path, FakeAdapter(["old"])).complete(MSGS, prompt_version="p")
    fresh = FakeAdapter(["new"])
    result = _llm(tmp_path, fresh, no_cache=True).complete(MSGS, prompt_version="p")
    assert result.text == "new" and not result.cached
    assert len(fresh.calls) == 1
    assert _llm(tmp_path, FakeAdapter()).complete(MSGS, prompt_version="p").text == "new"


def test_offline_mode_serves_cache_and_raises_cache_miss(tmp_path: Path) -> None:
    _llm(tmp_path, FakeAdapter(["recorded"])).complete(MSGS, prompt_version="p")
    offline = make_settings(tmp_path, offline=True)
    llm = get_llm("generation", settings=offline)
    assert llm.adapter is None
    llm.prices = PRICES
    hit = llm.complete(MSGS, prompt_version="p")
    assert hit.cached and hit.text == "recorded"
    with pytest.raises(CacheMiss, match="no recorded response"):
        llm.complete(MSGS, prompt_version="p2")
    with pytest.raises(LLMError, match="no_cache"):
        get_llm("generation", settings=offline, no_cache=True)


def test_unreadable_cache_entry_is_ignored(tmp_path: Path) -> None:
    adapter = FakeAdapter(["one", "two"])
    llm = _llm(tmp_path, adapter)
    first = llm.complete(MSGS, prompt_version="p")
    (llm.settings.llm_cache_dir / f"{first.meta['cache_key']}.json").write_text("{broken")
    assert llm.complete(MSGS, prompt_version="p").text == "two"


def test_json_schema_is_rendered_into_the_prompt(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    _llm(tmp_path, adapter).complete(MSGS, json_schema=schema, prompt_version="p")
    sent = adapter.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith("You write specs.")
    assert '{"type":"object","properties":{"a":{"type":"integer"}}}' in sent[0]["content"]
    assert sent[1] == MSGS[1]
    no_system = render_schema([MSGS[1]], schema)
    assert no_system[0]["role"] == "system" and no_system[1] == MSGS[1]
    assert render_schema(MSGS, None) == MSGS


# -- retries ---------------------------------------------------------------------------------------
def test_retry_on_429_and_5xx_then_success(tmp_path: Path) -> None:
    adapter = FakeAdapter(
        [
            ProviderError("rate limited", status=429, retryable=True),
            ProviderError("server error", status=503, retryable=True),
            '{"ok": true}',
        ]
    )
    sleeps: list[float] = []
    result = _llm(tmp_path, adapter, sleeps=sleeps).complete(MSGS, prompt_version="p")
    assert result.text == '{"ok": true}'
    assert result.meta["attempts"] == 3
    assert len(adapter.calls) == 3
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0


def test_retry_after_is_honoured(tmp_path: Path) -> None:
    adapter = FakeAdapter([ProviderError("slow down", status=429, retryable=True, retry_after=7)])
    sleeps: list[float] = []
    _llm(tmp_path, adapter, sleeps=sleeps).complete(MSGS, prompt_version="p")
    assert sleeps == [7.0]


def test_non_retryable_error_raises_at_once_and_records_nothing(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run")
    adapter = FakeAdapter([ProviderError("bad request", status=400)])
    sleeps: list[float] = []
    llm = _llm(tmp_path, adapter, sleeps=sleeps, run=run)
    with pytest.raises(ProviderError, match="bad request"):
        llm.complete(MSGS, prompt_version="p")
    assert len(adapter.calls) == 1 and sleeps == []
    assert not llm.settings.ledger_path.exists()
    assert not run.usage_jsonl.exists()
    assert spend_log(llm.settings.ledger_path).reserved == 0.0  # reservation released


def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    adapter = FakeAdapter([ProviderError("429", status=429, retryable=True)] * 5)
    sleeps: list[float] = []
    with pytest.raises(ProviderError):
        _llm(tmp_path, adapter, sleeps=sleeps, max_attempts=3).complete(MSGS, prompt_version="p")
    assert len(adapter.calls) == 3 and len(sleeps) == 2


def test_backoff_delay_grows_with_jitter() -> None:
    rng = random.Random(1)
    for attempt in range(1, 10):
        d = min(60.0, 2.0 ** (attempt - 1))
        delay = backoff_delay(attempt, rng=rng)
        assert d / 2 <= delay <= d
    assert backoff_delay(1, retry_after=5.0, rng=rng) == 5.0
    assert backoff_delay(1, retry_after=500.0, rng=rng) == 60.0


# -- budget ----------------------------------------------------------------------------------------
def test_budget_refusal_message_and_totals(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, budget_usd=0.0001)
    adapter = FakeAdapter()
    with pytest.raises(BudgetExceeded) as err:
        _llm(tmp_path, adapter, settings=settings).complete(MSGS, prompt_version="p")
    assert adapter.calls == []
    exc = err.value
    assert exc.scope == "global"
    assert exc.budget_usd == 0.0001 and exc.spent_usd == 0.0
    assert exc.projected_usd is not None and exc.projected_usd > exc.budget_usd
    message = str(exc)
    assert "LLM budget exceeded" in message
    assert "projected $" in message and "remains" in message and "spent $0.0000" in message
    assert "LLM_BUDGET_USD" in message


def test_budget_counts_the_ledger(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, budget_usd=1.0)
    ledger = spend_log(settings.ledger_path)
    ledger.append({"cost_usd": 0.95})
    with pytest.raises(BudgetExceeded, match=r"spent \$0\.9500"):
        _llm(tmp_path, FakeAdapter(), settings=settings).complete(
            MSGS, prompt_version="p", max_tokens=8192
        )
    # a small call still fits: 8192 output tokens at $10/M is $0.08, 100 is $0.001
    ok = _llm(tmp_path, FakeAdapter(), settings=settings).complete(
        MSGS, prompt_version="p", max_tokens=100
    )
    assert not ok.cached


def test_check_budget(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, budget_usd=40.0)
    for _ in range(3):
        spend_log(settings.ledger_path).append({"cost_usd": 12.0})
    status = budget_status(settings)
    assert status.spent_usd == pytest.approx(3 * 12.0)
    assert status.remaining_usd == pytest.approx(4.0)
    check_budget(3.9, settings=settings)
    with pytest.raises(BudgetExceeded) as err:
        check_budget(5.0, settings=settings)
    assert err.value.remaining_usd == pytest.approx(4.0)
    assert "projected $5.0000" in str(err.value) and "of $40.00" in str(err.value)


def test_per_run_budget_aborts(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run")
    settings = make_settings(tmp_path, budget_usd=100.0, run_budget_usd=0.005)
    # usage matching the projection (10 input tokens, max_tokens output): $0.00102 per call
    llm = _llm(tmp_path, FakeAdapter(usage=Usage(10, 100)), settings=settings, run=run)
    spent = 0.0
    calls = 0
    with pytest.raises(BudgetExceeded) as err:
        for i in range(10):
            spent += llm.complete(
                [{"role": "user", "content": f"page {i}"}], prompt_version="p", max_tokens=100
            ).cost_usd
            calls += 1
    assert err.value.scope == "run"
    assert "per-run LLM budget exceeded" in str(err.value)
    assert calls == 4
    assert 0 < spent <= 0.005
    status = budget_status(settings, run)
    assert status.run_spent_usd == pytest.approx(spent)
    assert status.run_remaining_usd == pytest.approx(0.005 - spent)
    # another run has its own budget
    check_budget(0.004, settings=settings, run=RunDir(tmp_path / "other"))


def test_in_flight_reservations_count(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, budget_usd=1.0)
    scopes = _scopes(settings, None)
    with _reserve(scopes, 0.6), pytest.raises(BudgetExceeded, match="in flight"):
        check_budget(0.6, settings=settings)
    check_budget(0.6, settings=settings)


def test_concurrent_calls_never_overspend(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, budget_usd=0.05)
    release = threading.Event()

    def slow(_messages: list[Message]) -> str:
        release.wait(5)
        return "{}"

    adapter = FakeAdapter([slow] * 20)
    llm = _llm(tmp_path, adapter, settings=settings)
    outcomes: list[str] = []

    def worker(i: int) -> None:
        try:
            llm.complete(
                [{"role": "user", "content": f"x{i}"}], prompt_version="p", max_tokens=1000
            )
            outcomes.append("ok")
        except BudgetExceeded:
            outcomes.append("refused")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    while len(adapter.calls) + outcomes.count("refused") < 8:
        threading.Event().wait(0.01)
    release.set()
    for t in threads:
        t.join()
    # each call reserves ~$0.01 (1000 output tokens at $10/M): only 4 fit in $0.05 at once
    assert outcomes.count("ok") == 4 and outcomes.count("refused") == 4
    assert budget_status(settings).spent_usd <= 0.05


# -- accounting ------------------------------------------------------------------------------------
def test_usage_jsonl_and_ledger_contents(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run")
    llm = _llm(tmp_path, FakeAdapter(usage=Usage(1000, 500, 200)), run=run)
    real = llm.complete(MSGS, prompt_version="specgen.v1", template_id="t_1")
    llm.complete(MSGS, prompt_version="specgen.v1", template_id="t_1")  # cached
    lines = _lines(run.usage_jsonl)
    assert len(lines) == 2
    first, second = lines
    assert set(first) == {
        "ts",
        "role",
        "template_id",
        "model",
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cost_usd",
        "list_cost_usd",
        "estimated",
        "cached",
        "latency_s",
        "key",
    }
    expected_cost = (800 * 2.0 + 200 * 0.5 + 500 * 10.0) / 1e6
    assert first["role"] == "generation" and first["template_id"] == "t_1"
    assert first["model"] == "openai/gpt-test" and first["prompt_version"] == "specgen.v1"
    assert (first["input_tokens"], first["output_tokens"], first["cached_input_tokens"]) == (
        1000,
        500,
        200,
    )
    assert first["cost_usd"] == pytest.approx(expected_cost) == pytest.approx(real.cost_usd)
    assert first["cached"] is False and first["estimated"] is False
    assert second["cached"] is True and second["cost_usd"] == 0.0
    assert second["list_cost_usd"] == pytest.approx(expected_cost)
    assert second["input_tokens"] == 1000 and second["key"] == first["key"]
    ledger = _lines(llm.settings.ledger_path)
    assert len(ledger) == 1  # cached calls are not real spend
    assert ledger[0]["run"] == str(run.root)
    assert ledger[0]["cost_usd"] == pytest.approx(expected_cost)


def test_flex_tier_bills_half_but_lists_standard(tmp_path: Path) -> None:
    from dataclasses import replace

    run = RunDir(tmp_path / "run")
    flex = Completion(
        text='{"ok": true}',
        usage=Usage(1000, 500, 0),
        model="openai/gpt-test",
        meta={"finish_reason": "stop", "service_tier": "flex"},
    )
    llm = _llm(tmp_path, FakeAdapter([flex]), run=run)
    llm.prices = replace(PRICES, tiers={"flex": 0.5})
    real = llm.complete(MSGS, prompt_version="p.v1")
    standard = (1000 * 2.0 + 500 * 10.0) / 1e6
    assert real.cost_usd == pytest.approx(standard / 2)
    (line,) = _lines(run.usage_jsonl)
    assert line["cost_usd"] == pytest.approx(standard / 2)
    assert line["list_cost_usd"] == pytest.approx(standard)
    assert _lines(llm.settings.ledger_path)[0]["cost_usd"] == pytest.approx(standard / 2)


def test_is_cached_matches_complete(tmp_path: Path) -> None:
    llm = _llm(tmp_path, FakeAdapter())
    assert not llm.is_cached(MSGS, max_tokens=100, prompt_version="p.v1")
    llm.complete(MSGS, max_tokens=100, prompt_version="p.v1")
    assert llm.is_cached(MSGS, max_tokens=100, prompt_version="p.v1")
    assert not llm.is_cached(MSGS, max_tokens=200, prompt_version="p.v1")
    assert not llm.is_cached(MSGS, max_tokens=100, prompt_version="p.v2")
    assert not _llm(tmp_path, FakeAdapter(), no_cache=True).is_cached(
        MSGS, max_tokens=100, prompt_version="p.v1"
    )


def test_summarize_usage(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "run")
    run.ensure().write_meta(site="example.org")
    gen = _llm(tmp_path, FakeAdapter(), run=run)
    gen.complete(MSGS, prompt_version="p", template_id="t_a")
    gen.complete(MSGS, prompt_version="p", template_id="t_a")  # cached
    gen.complete([MSGS[1]], prompt_version="p", template_id="t_b")
    small = CachedLLM(
        model="openai/gpt-guess",
        role="probe",
        adapter=FakeAdapter(),
        settings=gen.settings,
        run=run,
        prices=PRICES,
    )
    small.complete(MSGS, prompt_version="probe.v1")
    summary = summarize_usage(run)
    assert summary["site"] == "example.org"
    per_call = (1000 * 2.0 + 500 * 10.0) / 1e6
    guess_call = (1000 * 1.0 + 500 * 1.0) / 1e6
    totals = summary["totals"]
    assert totals["calls"] == 4 and totals["cached_calls"] == 1
    assert totals["input_tokens"] == 4000 and totals["output_tokens"] == 2000
    assert totals["cost_usd"] == pytest.approx(2 * per_call + guess_call)
    assert totals["list_cost_usd"] == pytest.approx(3 * per_call + guess_call)
    assert totals["estimated"] is True
    assert set(summary["by_role"]) == {"generation", "probe"}
    assert summary["by_role"]["generation"]["estimated"] is False
    assert summary["by_template"]["t_a"]["calls"] == 2
    assert summary["by_template"]["t_a"]["cached_calls"] == 1
    assert summary["by_template"]["_none"]["calls"] == 1
    assert summary["by_model"]["openai/gpt-guess"]["cost_usd"] == pytest.approx(guess_call)
    assert len(read_usage(run)) == 4


def test_summarize_usage_of_empty_run(tmp_path: Path) -> None:
    summary = summarize_usage(RunDir(tmp_path / "nothing"))
    assert summary["totals"]["calls"] == 0 and summary["totals"]["cost_usd"] == 0.0
    assert summary["by_role"] == {} and summary["site"] == "nothing"


def test_record_external_usage(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run = RunDir(tmp_path / "run")
    cost = record_external_usage(
        "gpt-6.1-sol",
        1000,
        100,
        role="baseline",
        prompt_version="crawl4ai",
        settings=settings,
        run=run,
    )
    assert cost > 0
    (ledger,) = _lines(settings.ledger_path)
    (usage,) = _lines(run.usage_jsonl)
    assert ledger["model"] == usage["model"] == "openai/gpt-6.1-sol"
    assert ledger["cost_usd"] == usage["cost_usd"] == cost
    assert usage["role"] == "baseline" and usage["cached"] is False


def test_spend_log_reads_incrementally_and_skips_partial_lines(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    log = spend_log(path)
    assert log.spent() == 0.0
    log.append({"cost_usd": 1.5})
    with open(path, "ab") as fh:  # another process mid-write
        fh.write(b'{"cost_usd": 2.0}\nnot json\n{"cost_usd": 4')
    assert log.spent() == pytest.approx(3.5)
    with open(path, "ab") as fh:
        fh.write(b".0}\n")
    assert log.spent() == pytest.approx(7.5)
    path.write_text('{"cost_usd": 0.25}\n')  # replaced by a shorter file: recount
    assert log.spent() == pytest.approx(0.25)
    path.unlink()
    assert log.spent() == 0.0


# -- get_llm / adapters ------------------------------------------------------------------------------
def test_get_llm_roles_and_injected_adapter(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    adapter = FakeAdapter()
    assert get_llm("generation", settings=settings, adapter=adapter).model == "openai/gpt-test"
    probe = get_llm("probe", settings=settings, adapter=adapter)
    assert (probe.model, probe.role) == ("openai/gpt-test-mini", "probe")
    assert "gpt-test-mini" in repr(probe)
    with pytest.raises(ValueError, match="unknown LLM role"):
        get_llm("writer", settings=settings)  # type: ignore[arg-type]


def test_make_adapter(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, anthropic_api_key="sk-ant")
    assert isinstance(make_adapter("openai", settings), OpenAICompatAdapter)
    assert isinstance(make_adapter("anthropic", settings), AnthropicAdapter)
    keyless = make_settings(tmp_path, openai_api_key=None)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        make_adapter("openai", keyless)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_adapter("anthropic", keyless)
    with pytest.raises(LLMError, match="unknown provider"):
        make_adapter("gemini", settings)
    real = get_llm("generation", settings=settings)
    assert isinstance(real.adapter, OpenAICompatAdapter)
    anthropic_llm = get_llm("repair", settings=settings.with_models(repair="anthropic/claude-x"))
    assert isinstance(anthropic_llm.adapter, AnthropicAdapter)


def test_completion_from_adapter_keeps_meta(tmp_path: Path) -> None:
    comp = Completion(
        text="x", usage=Usage(10, 5), model="openai/gpt-test", meta={"finish_reason": "length"}
    )
    result = _llm(tmp_path, FakeAdapter([comp])).complete(MSGS, prompt_version="p")
    assert result.meta["finish_reason"] == "length" and result.meta["attempts"] == 1
    assert result.latency_s >= 0.0
