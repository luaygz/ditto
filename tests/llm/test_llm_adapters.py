"""Adapters through the real SDKs with an in-process HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from ditto.llm.anthropic import AnthropicAdapter
from ditto.llm.base import Message, ProviderError
from ditto.llm.openai_compat import OpenAICompatAdapter, retry_after_seconds

Handler = Callable[[httpx2.Request], httpx2.Response]

MSGS: list[Message] = [
    {"role": "system", "content": "Be terse."},
    {"role": "system", "content": "JSON only."},
    {"role": "user", "content": "Say ok."},
]

CHAT_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-6.1-sol-2026-10-01",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"ok": true}', "refusal": None},
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 64},
    },
}


def _openai(handler: Handler) -> OpenAICompatAdapter:
    client = openai.OpenAI(
        api_key="sk-test",
        base_url="http://llm.test/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAICompatAdapter(api_key="sk-test", base_url="http://llm.test/v1", client=client)


def _anthropic(handler: Handler) -> AnthropicAdapter:
    client = anthropic.Anthropic(
        api_key="sk-ant",
        base_url="http://anthropic.test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return AnthropicAdapter(api_key="sk-ant", client=client)


def test_openai_request_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=CHAT_OK)

    comp = _openai(handler).complete(
        MSGS, model="gpt-6.1-sol", json_schema={"type": "object"}, max_tokens=777
    )
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    body = seen["body"]
    assert body["model"] == "gpt-6.1-sol"
    assert body["max_completion_tokens"] == 777
    assert "max_tokens" not in body and "response_format" not in body
    assert body["messages"] == [dict(m) for m in MSGS]
    assert comp.text == '{"ok": true}'
    assert comp.model == "openai/gpt-6.1-sol"
    assert (comp.usage.input_tokens, comp.usage.output_tokens) == (120, 30)
    assert comp.usage.cached_input_tokens == 64
    assert comp.meta["finish_reason"] == "stop"
    assert comp.meta["response_model"] == "gpt-6.1-sol-2026-10-01"
    assert comp.latency_s >= 0


def test_openai_missing_usage_and_content() -> None:
    reply = json.loads(json.dumps(CHAT_OK))
    reply["choices"][0]["message"]["content"] = None
    reply["choices"][0]["finish_reason"] = "length"
    del reply["usage"]
    comp = _openai(lambda _r: httpx2.Response(200, json=reply)).complete(
        MSGS, model="m", json_schema=None, max_tokens=5
    )
    assert comp.text == "" and comp.usage.input_tokens == 0
    assert comp.meta["finish_reason"] == "length"
    empty = {**CHAT_OK, "choices": []}
    with pytest.raises(ProviderError, match="no choices"):
        _openai(lambda _r: httpx2.Response(200, json=empty)).complete(
            MSGS, model="m", json_schema=None, max_tokens=5
        )


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (429, "rate_limit_exceeded", True),
        (429, "insufficient_quota", False),
        (500, None, True),
        (503, None, True),
        (400, "invalid_request_error", False),
        (401, "invalid_api_key", False),
    ],
)
def test_openai_errors_are_classified(status: int, code: str | None, retryable: bool) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"error": {"message": "nope", "type": "x", "code": code}},
            headers={"retry-after": "2"},
        )

    with pytest.raises(ProviderError) as err:
        _openai(handler).complete(MSGS, model="m", json_schema=None, max_tokens=5)
    assert err.value.status == status
    assert err.value.retryable is retryable
    assert err.value.code == code
    assert err.value.retry_after == 2.0


def test_openai_connection_error_is_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    adapter = _openai(handler)
    with pytest.raises(ProviderError) as err:
        adapter.complete(MSGS, model="m", json_schema=None, max_tokens=5)
    assert err.value.retryable and err.value.status is None
    with pytest.raises(ProviderError):
        adapter.list_models()


def test_openai_list_models() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/models"
        data = [
            {"id": "gpt-6.1-sol", "object": "model", "created": 10, "owned_by": "system"},
            {"id": "gpt-5.4-mini", "object": "model", "created": 5, "owned_by": "system"},
        ]
        return httpx2.Response(200, json={"object": "list", "data": data})

    assert _openai(handler).list_models() == [("gpt-6.1-sol", 10), ("gpt-5.4-mini", 5)]
    denied = _openai(lambda _r: httpx2.Response(403, json={"error": {"message": "no"}}))
    with pytest.raises(ProviderError) as err:
        denied.list_models()
    assert err.value.status == 403 and not err.value.retryable


def test_retry_after_parsing() -> None:
    assert retry_after_seconds({"retry-after-ms": "1500"}) == 1.5
    assert retry_after_seconds({"retry-after": "3"}) == 3.0
    assert retry_after_seconds({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert retry_after_seconds({"retry-after": "100000"}) is None
    assert retry_after_seconds(None) is None


def test_anthropic_request_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-x-2026",
                "content": [
                    {"type": "text", "text": '{"ok": '},
                    {"type": "text", "text": "true}"},
                ],
                "stop_reason": "max_tokens",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                },
            },
        )

    comp = _anthropic(handler).complete(MSGS, model="claude-x", json_schema=None, max_tokens=64)
    assert seen["path"] == "/v1/messages"
    body = seen["body"]
    assert body["system"] == "Be terse.\n\nJSON only."
    assert body["messages"] == [{"role": "user", "content": "Say ok."}]
    assert body["max_tokens"] == 64 and body["model"] == "claude-x"
    assert comp.text == '{"ok": true}'
    assert comp.model == "anthropic/claude-x"
    assert comp.usage.input_tokens == 75  # reads and writes of the prompt cache included
    assert comp.usage.cached_input_tokens == 20
    assert comp.usage.output_tokens == 10
    assert comp.meta["finish_reason"] == "length"


def test_anthropic_without_system_and_errors() -> None:
    seen: dict[str, Any] = {}

    def ok(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "model": "claude-x",
                "content": [{"type": "text", "text": "{}"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    comp = _anthropic(ok).complete([MSGS[2]], model="claude-x", json_schema=None, max_tokens=8)
    assert "system" not in seen["body"]
    assert comp.usage.input_tokens == 5 and comp.meta["finish_reason"] == "end_turn"

    for status, retryable in ((529, True), (429, True), (400, False)):

        def failing(_request: httpx2.Request, status: int = status) -> httpx2.Response:
            return httpx2.Response(
                status,
                json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            )

        with pytest.raises(ProviderError) as err:
            _anthropic(failing).complete(MSGS, model="claude-x", json_schema=None, max_tokens=8)
        assert err.value.status == status and err.value.retryable is retryable

    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(ProviderError) as err:
        _anthropic(down).complete(MSGS, model="claude-x", json_schema=None, max_tokens=8)
    assert err.value.retryable
