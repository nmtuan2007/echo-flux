"""Pins the OpenAI-compatible text transport and its Translation/Assistant adapters.

Every test uses ``httpx.MockTransport`` or an in-memory byte stream; nothing reaches a
network. SECRET is a sentinel that must never appear in errors, reprs, or logs.
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pytest

from engine.providers.assistant.openai_compatible import OpenAICompatibleAssistantProvider
from engine.providers.contracts import AssistantProvider, TranslationProvider
from engine.providers.text.openai_compatible import (
    AssemblyAIGatewayStrategy,
    AzureOpenAIStrategy,
    CustomOpenAICompatibleStrategy,
    LocalOpenAICompatibleStrategy,
    OpenAICompatibleTextClient,
    OpenAIStrategy,
    TextCompletion,
    TextConfigError,
    TextGenerationOptions,
    TextMessage,
    TextTransportOptions,
)
from engine.providers.translation.openai_compatible import OpenAICompatibleTranslationProvider
from engine.providers.types import (
    AssistRequest,
    AssistResult,
    AssistSuggestion,
    ConversationMessage,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    SummaryRequest,
    TranslationRequest,
)

SECRET = "sk-test-not-real-EF017-0001"
HEADER_SECRET = "hdr-test-not-real-EF017-0002"
TRANSLATION = ProviderCapability.TRANSLATION
ASSISTANT = ProviderCapability.ASSISTANT
MESSAGES = (TextMessage("system", "sys"), TextMessage("user", "hello"))
FAST = TextTransportOptions(retry_backoff_s=0.0, max_retry_after_s=0.0)


def _completion_payload(text: str = "xin chao", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    payload["choices"][0]["finish_reason"] = "stop"
    payload.update(extra)
    return payload


class Recorder:
    """MockTransport handler that records requests and replays scripted responses."""

    def __init__(self, *responses: Any) -> None:
        self.responses: List[Any] = list(responses)
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self, index: int = -1) -> Dict[str, Any]:
        return json.loads(self.requests[index].content)


def _ok(payload: Optional[Dict[str, Any]] = None) -> httpx.Response:
    return httpx.Response(200, json=payload or _completion_payload())


def _client(
    strategy: Any,
    recorder: Recorder,
    capability: ProviderCapability = TRANSLATION,
    options: TextTransportOptions = FAST,
) -> OpenAICompatibleTextClient:
    async def _no_sleep(_: float) -> None:
        return None

    return OpenAICompatibleTextClient(
        strategy,
        capability=capability,
        options=options,
        http_transport=recorder.transport(),
        sleep=_no_sleep,
    )


@pytest.fixture
def captured_logs():
    records: List[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=logging.DEBUG)
    root = logging.getLogger("echoflux")
    previous = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)


def _log_text(records: Sequence[logging.LogRecord]) -> str:
    return "\n".join(record.getMessage() for record in records)


def _assert_no_secret(*values: Any) -> None:
    for value in values:
        text = value if isinstance(value, str) else repr(value)
        assert SECRET not in text
        assert HEADER_SECRET not in text


# ── Provider-specific request and auth shapes ───────────────────────────────


async def test_openai_request_and_auth_shape():
    recorder = Recorder(_ok())
    client = _client(OpenAIStrategy("https://api.openai.com/v1/", SECRET), recorder)
    completion = await client.complete(
        "gpt-test", MESSAGES, TextGenerationOptions(temperature=0.2, max_output_tokens=50)
    )
    await client.aclose()

    request = recorder.last
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert "api-key" not in request.headers
    assert recorder.body() == {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 50,
    }
    assert completion.text == "xin chao"
    assert (completion.input_tokens, completion.output_tokens) == (11, 3)
    assert completion.response_id == "chatcmpl-1"
    assert completion.finish_reason == "stop"


async def test_azure_v1_surface_uses_api_key_header_and_deployment_as_model():
    recorder = Recorder(_ok())
    strategy = AzureOpenAIStrategy("https://my-res.openai.azure.com", "prod-deploy", "v1", SECRET)
    client = _client(strategy, recorder)
    await client.complete("ignored-model", MESSAGES, TextGenerationOptions(max_output_tokens=9))
    await client.aclose()

    request = recorder.last
    assert str(request.url) == "https://my-res.openai.azure.com/openai/v1/chat/completions"
    assert request.headers["api-key"] == SECRET
    assert "authorization" not in request.headers
    body = recorder.body()
    assert body["model"] == "prod-deploy"
    assert body["max_completion_tokens"] == 9


async def test_azure_dated_api_version_uses_deployment_path_without_model():
    recorder = Recorder(_ok())
    strategy = AzureOpenAIStrategy(
        "https://my-res.openai.azure.com/", "prod-deploy", "2024-10-21", SECRET
    )
    client = _client(strategy, recorder)
    await client.complete("ignored-model", MESSAGES)
    await client.aclose()

    request = recorder.last
    assert request.url.path == "/openai/deployments/prod-deploy/chat/completions"
    assert request.url.params["api-version"] == "2024-10-21"
    assert request.headers["api-key"] == SECRET
    assert "model" not in recorder.body()


@pytest.mark.parametrize(
    "kwargs, field_name",
    [
        ({"resource_endpoint": "https://r.openai.azure.com/openai/v1"}, "endpoint"),
        ({"resource_endpoint": "http://r.openai.azure.com"}, "endpoint"),
        ({"deployment": ""}, "deployment"),
        ({"deployment": "a/b"}, "deployment"),
        ({"api_version": ""}, "api_version"),
        ({"api_key": ""}, "api_key"),
    ],
)
def test_azure_fields_are_explicit_and_validated(kwargs, field_name):
    values = {
        "resource_endpoint": "https://r.openai.azure.com",
        "deployment": "d1",
        "api_version": "v1",
        "api_key": SECRET,
    }
    values.update(kwargs)
    with pytest.raises(TextConfigError) as excinfo:
        AzureOpenAIStrategy(**values)
    assert excinfo.value.field_name == field_name
    _assert_no_secret(str(excinfo.value))


async def test_assemblyai_gateway_uses_raw_authorization_and_max_tokens():
    payload = {
        "request_id": "7a1c0b7e-0000-4000-8000-000000000001",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
    }
    recorder = Recorder(_ok(payload))
    client = _client(
        AssemblyAIGatewayStrategy("https://llm-gateway.assemblyai.com/v1", SECRET), recorder
    )
    completion = await client.complete(
        "claude-test", MESSAGES, TextGenerationOptions(max_output_tokens=100)
    )
    await client.aclose()

    request = recorder.last
    assert str(request.url) == "https://llm-gateway.assemblyai.com/v1/chat/completions"
    assert request.headers["authorization"] == SECRET
    body = recorder.body()
    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body
    assert (completion.input_tokens, completion.output_tokens) == (20, 4)
    assert completion.response_id == payload["request_id"]


async def test_local_mode_works_without_api_key():
    recorder = Recorder(_ok())
    client = _client(LocalOpenAICompatibleStrategy("http://127.0.0.1:11434/v1"), recorder)
    completion = await client.complete(
        "llama-test", MESSAGES, TextGenerationOptions(max_output_tokens=8)
    )
    await client.aclose()

    request = recorder.last
    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    assert "authorization" not in request.headers
    assert recorder.body()["max_tokens"] == 8
    assert completion.text == "xin chao"


def test_local_mode_optional_key_is_bearer_and_never_sent_over_remote_http():
    strategy = LocalOpenAICompatibleStrategy("http://localhost:1234/v1", SECRET)
    assert strategy.headers() == {"Authorization": f"Bearer {SECRET}"}
    assert LocalOpenAICompatibleStrategy("http://192.168.1.10:8000/v1").headers() == {}
    with pytest.raises(TextConfigError):
        LocalOpenAICompatibleStrategy("http://192.168.1.10:8000/v1", SECRET)


async def test_custom_mode_sends_bearer_secret_headers_and_plain_headers():
    recorder = Recorder(_ok())
    strategy = CustomOpenAICompatibleStrategy(
        "https://llm.example.invalid/v1",
        SECRET,
        headers={"X-Org": "team-a"},
        secret_headers={"X-Gateway-Token": HEADER_SECRET},
        max_tokens_field="max_completion_tokens",
    )
    client = _client(strategy, recorder)
    await client.complete("m", MESSAGES, TextGenerationOptions(max_output_tokens=5))
    await client.aclose()

    headers = recorder.last.headers
    assert headers["authorization"] == f"Bearer {SECRET}"
    assert headers["x-gateway-token"] == HEADER_SECRET
    assert headers["x-org"] == "team-a"
    assert recorder.body()["max_completion_tokens"] == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"headers": {"Authorization": SECRET}},
        {"headers": {"X-Api-Key": SECRET}},
        {"headers": {"api_key": SECRET}},
        {"headers": {"Cookie": SECRET}},
        {"headers": {"X-Ok": "line\r\nInjected: " + SECRET}},
        {"headers": {"Bad Name": "v"}},
        {"headers": {"Content-Type": "text/plain"}},
        {"secret_headers": {"Authorization": HEADER_SECRET}, "api_key": SECRET},
        {"secret_headers": {"X-Token": ""}},
        {"max_tokens_field": "tokens"},
    ],
)
def test_custom_mode_rejects_unapproved_secret_and_unsafe_headers(kwargs):
    values: Dict[str, Any] = {"base_url": "https://llm.example.invalid/v1"}
    values.update(kwargs)
    with pytest.raises(TextConfigError) as excinfo:
        CustomOpenAICompatibleStrategy(**values)
    _assert_no_secret(str(excinfo.value), repr(excinfo.value))


@pytest.mark.parametrize(
    "url",
    [
        f"https://llm.example.invalid/v1?api_key={SECRET}",
        f"https://user:{SECRET}@llm.example.invalid/v1",
        "ftp://llm.example.invalid/v1",
        "https:///v1",
        "",
    ],
)
def test_endpoints_with_embedded_credentials_or_bad_schemes_are_rejected(url):
    with pytest.raises(TextConfigError) as excinfo:
        CustomOpenAICompatibleStrategy(url)
    _assert_no_secret(str(excinfo.value))


def test_cloud_modes_require_credentials_and_https():
    with pytest.raises(TextConfigError):
        OpenAIStrategy("https://api.openai.com/v1", "")
    with pytest.raises(TextConfigError):
        OpenAIStrategy("http://api.openai.com/v1", SECRET)
    with pytest.raises(TextConfigError):
        AssemblyAIGatewayStrategy("https://llm-gateway.assemblyai.com/v1", None)  # type: ignore[arg-type]
    with pytest.raises(TextConfigError):
        CustomOpenAICompatibleStrategy("http://llm.example.invalid/v1", SECRET)


def test_reprs_never_contain_secrets():
    strategies = [
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        AzureOpenAIStrategy("https://r.openai.azure.com", "d", "v1", SECRET),
        AssemblyAIGatewayStrategy("https://llm-gateway.assemblyai.com/v1", SECRET),
        LocalOpenAICompatibleStrategy("http://127.0.0.1:1/v1", SECRET),
        CustomOpenAICompatibleStrategy(
            "https://x.example.invalid/v1", SECRET, secret_headers={"X-T": HEADER_SECRET}
        ),
    ]
    for strategy in strategies:
        client = OpenAICompatibleTextClient(strategy, capability=TRANSLATION)
        translation = OpenAICompatibleTranslationProvider(
            strategy, "m", message_builder=lambda r: MESSAGES
        )
        _assert_no_secret(repr(strategy), repr(client), repr(translation), vars(translation))


# ── Errors and redaction ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status, body, code, retryable",
    [
        (401, {"error": {"code": "invalid_api_key"}}, ProviderErrorCode.AUTH_FAILED, False),
        (429, {"error": {"code": "rate_limit_exceeded"}}, ProviderErrorCode.RATE_LIMITED, True),
        (429, {"error": {"code": "insufficient_quota"}}, ProviderErrorCode.QUOTA_EXCEEDED, False),
        (
            400,
            {"error": {"code": "context_length_exceeded"}},
            ProviderErrorCode.CONTEXT_OVERFLOW,
            False,
        ),
        (
            404,
            {"error": {"code": "DeploymentNotFound"}},
            ProviderErrorCode.MODEL_UNAVAILABLE,
            False,
        ),
        (500, {"error": {"type": "server_error"}}, ProviderErrorCode.INTERNAL_ERROR, True),
        (503, "not json", ProviderErrorCode.NETWORK_UNAVAILABLE, True),
    ],
)
async def test_http_errors_map_to_stable_codes(status, body, code, retryable):
    response = (
        httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status)
    )
    recorder = Recorder(response)
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder, ASSISTANT)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    await client.aclose()
    error = excinfo.value.error
    assert error.code is code
    assert error.retryable is retryable
    assert error.capability is ASSISTANT
    assert error.details["http_status"] == status
    assert len(recorder.requests) == 1


async def test_custom_mode_secrets_never_reach_errors_or_logs(captured_logs):
    echo = {
        "error": {
            "message": f"bad key {SECRET} / {HEADER_SECRET}",
            "code": HEADER_SECRET,
            "type": SECRET,
        }
    }
    recorder = Recorder(httpx.Response(401, json=echo))
    strategy = CustomOpenAICompatibleStrategy(
        "https://llm.example.invalid/v1?tenant=abc",
        SECRET,
        secret_headers={"X-Gateway-Token": HEADER_SECRET},
    )
    provider = OpenAICompatibleTranslationProvider(
        strategy,
        "m",
        message_builder=lambda r: MESSAGES,
        provider_id="custom-openai-compatible",
        http_transport=recorder.transport(),
        transport_options=FAST,
    )
    await provider.start()
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await provider.translate(TranslationRequest("hi", "en", "vi"))
    health = await provider.health_check()
    await provider.stop()

    error = excinfo.value.error
    assert error.code is ProviderErrorCode.AUTH_FAILED
    assert "provider_code" not in (error.details or {})
    _assert_no_secret(
        str(excinfo.value),
        repr(error),
        error.message,
        error.details,
        health,
        _log_text(captured_logs),
    )
    assert "tenant=abc" not in _log_text(captured_logs)


async def test_transport_exceptions_are_redacted(captured_logs):
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError(f"failure talking to {request.url} with {SECRET}")

    recorder = Recorder(_boom)
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    await client.aclose()
    assert excinfo.value.error.code is ProviderErrorCode.NETWORK_UNAVAILABLE
    assert excinfo.value.__cause__ is None
    _assert_no_secret(str(excinfo.value), _log_text(captured_logs))


# ── Timeouts, bounds, and retries ────────────────────────────────────────────


async def test_completion_post_is_not_retried_after_the_request_was_sent():
    recorder = Recorder(httpx.Response(500), _ok())
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder)
    with pytest.raises(ProviderRuntimeError):
        await client.complete("m", MESSAGES)
    assert len(recorder.requests) == 1

    timeout_recorder = Recorder(httpx.ReadTimeout("slow"), _ok())
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), timeout_recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.TIMEOUT
    assert len(timeout_recorder.requests) == 1


async def test_completion_post_is_retried_when_connection_never_opened():
    recorder = Recorder(httpx.ConnectError("refused"), _ok())
    client = _client(LocalOpenAICompatibleStrategy("http://127.0.0.1:9/v1"), recorder)
    completion = await client.complete("m", MESSAGES)
    await client.aclose()
    assert completion.text == "xin chao"
    assert len(recorder.requests) == 2

    always = Recorder(httpx.ConnectError("refused"))
    client = _client(LocalOpenAICompatibleStrategy("http://127.0.0.1:9/v1"), always)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.NETWORK_UNAVAILABLE
    assert len(always.requests) == 1 + FAST.max_retries


async def test_idempotent_probe_retries_transient_statuses():
    recorder = Recorder(
        httpx.Response(503, headers={"retry-after": "1"}), httpx.Response(200, json={"data": []})
    )
    provider = OpenAICompatibleAssistantProvider(
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        "m",
        prompt_policy=_Policy(),
        http_transport=recorder.transport(),
        transport_options=FAST,
    )
    health = await provider.health_check()
    await provider.stop()
    assert health.status is ProviderHealthStatus.HEALTHY
    assert [r.method for r in recorder.requests] == ["GET", "GET"]
    assert str(recorder.last.url) == "https://api.openai.com/v1/models"


async def test_health_check_without_documented_probe_makes_no_request():
    recorder = Recorder(_ok())
    provider = OpenAICompatibleTranslationProvider(
        AssemblyAIGatewayStrategy("https://llm-gateway.assemblyai.com/v1", SECRET),
        "m",
        message_builder=lambda r: MESSAGES,
        http_transport=recorder.transport(),
    )
    health = await provider.health_check()
    await provider.stop()
    assert health.status is ProviderHealthStatus.DEGRADED
    assert recorder.requests == []


async def test_unauthorized_probe_reports_unhealthy():
    recorder = Recorder(httpx.Response(401))
    provider = OpenAICompatibleTranslationProvider(
        AzureOpenAIStrategy("https://r.openai.azure.com", "d", "v1", SECRET),
        "m",
        message_builder=lambda r: MESSAGES,
        http_transport=recorder.transport(),
        transport_options=FAST,
    )
    health = await provider.health_check()
    await provider.stop()
    assert health.status is ProviderHealthStatus.UNHEALTHY
    assert health.details["code"] == "auth_failed"
    assert str(recorder.last.url) == "https://r.openai.azure.com/openai/v1/models"
    assert len(recorder.requests) == 1


async def test_oversized_response_is_rejected():
    big = {"choices": [{"message": {"content": "x" * 5000}}]}
    recorder = Recorder(_ok(big))
    client = _client(
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        recorder,
        options=TextTransportOptions(max_response_bytes=1024),
    )
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize("payload", [{"choices": []}, {"nope": 1}, [1, 2]])
async def test_malformed_completion_is_invalid_response(payload):
    recorder = Recorder(httpx.Response(200, json=payload))
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE


async def test_total_timeout_bounds_a_slow_completion():
    async def _slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return _ok()

    client = OpenAICompatibleTextClient(
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        capability=TRANSLATION,
        options=TextTransportOptions(total_timeout_s=0.05),
        http_transport=httpx.MockTransport(_slow),
    )
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", MESSAGES)
    await client.aclose()
    assert excinfo.value.error.code is ProviderErrorCode.TIMEOUT


@pytest.mark.parametrize(
    "messages, generation, field_name",
    [
        ((), TextGenerationOptions(), "messages"),
        ((TextMessage("tool", "x"),), TextGenerationOptions(), "messages"),
        (MESSAGES, TextGenerationOptions(temperature=3.0), "temperature"),
        (MESSAGES, TextGenerationOptions(max_output_tokens=0), "max_output_tokens"),
    ],
)
async def test_invalid_requests_fail_before_any_network_call(messages, generation, field_name):
    recorder = Recorder(_ok())
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder, ASSISTANT)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("m", messages, generation)
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_REQUEST
    assert excinfo.value.error.capability is ASSISTANT
    assert excinfo.value.error.details["field"] == field_name
    assert recorder.requests == []


# ── Streaming ────────────────────────────────────────────────────────────────


def _sse(*events: Any) -> bytes:
    lines = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


def _delta(text: str, finish: Optional[str] = None) -> Dict[str, Any]:
    return {"id": "chatcmpl-s", "choices": [{"delta": {"content": text}, "finish_reason": finish}]}


async def test_stream_yields_deltas_usage_and_final_chunk():
    body = _sse(
        _delta("Xin "),
        _delta("chao", "stop"),
        {"id": "chatcmpl-s", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        "[DONE]",
    )
    recorder = Recorder(httpx.Response(200, content=b": keepalive\n\n" + body))
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder, ASSISTANT)
    chunks = [chunk async for chunk in client.stream("m", MESSAGES)]
    await client.aclose()

    assert [c.delta_text for c in chunks] == ["Xin ", "chao", ""]
    final = chunks[-1]
    assert final.is_done and final.completion is not None
    assert final.completion.text == "Xin chao"
    assert (final.completion.input_tokens, final.completion.output_tokens) == (7, 2)
    assert final.completion.finish_reason == "stop"
    body_sent = recorder.body()
    assert body_sent["stream"] is True
    assert body_sent["stream_options"] == {"include_usage": True}
    assert recorder.last.headers["accept"] == "text/event-stream"


async def test_stream_options_only_sent_where_supported():
    recorder = Recorder(httpx.Response(200, content=_sse(_delta("a", "stop"), "[DONE]")))
    client = _client(
        AssemblyAIGatewayStrategy("https://llm-gateway.assemblyai.com/v1", SECRET), recorder
    )
    chunks = [chunk async for chunk in client.stream("m", MESSAGES)]
    assert chunks[-1].completion.text == "a"
    assert "stream_options" not in recorder.body()


async def test_stream_ending_without_done_or_finish_is_connection_closed():
    recorder = Recorder(httpx.Response(200, content=_sse(_delta("partial"))))
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for _ in client.stream("m", MESSAGES):
            pass
    assert excinfo.value.error.code is ProviderErrorCode.CONNECTION_CLOSED


async def test_stream_error_event_is_mapped_and_redacted():
    event = {"error": {"code": "rate_limit_exceeded", "message": f"slow down {SECRET}"}}
    recorder = Recorder(httpx.Response(200, content=_sse(_delta("a"), event)))
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for _ in client.stream("m", MESSAGES):
            pass
    assert excinfo.value.error.code is ProviderErrorCode.RATE_LIMITED
    _assert_no_secret(str(excinfo.value), excinfo.value.error.details)


async def test_stream_http_error_is_raised_before_any_chunk():
    recorder = Recorder(httpx.Response(403, json={"error": {"code": "unauthorized"}}))
    client = _client(OpenAIStrategy("https://api.openai.com/v1", SECRET), recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for _ in client.stream("m", MESSAGES):
            pass
    assert excinfo.value.error.code is ProviderErrorCode.AUTH_FAILED


class TrackingStream(httpx.AsyncByteStream):
    """Yields scripted chunks, then blocks until closed; records ``aclose``."""

    def __init__(self, *chunks: bytes, hang: bool = True) -> None:
        self.chunks = chunks
        self.hang = hang
        self.closed = asyncio.Event()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.hang:
            await self.closed.wait()

    async def aclose(self) -> None:
        self.closed.set()


def _streaming_client(stream: TrackingStream, **options: Any) -> OpenAICompatibleTextClient:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    return OpenAICompatibleTextClient(
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        capability=ASSISTANT,
        options=TextTransportOptions(**options),
        http_transport=httpx.MockTransport(_handler),
    )


async def test_closing_a_stream_early_closes_the_response_promptly():
    stream = TrackingStream(_sse(_delta("first")))
    client = _streaming_client(stream)
    iterator = client.stream("m", MESSAGES)
    first = await iterator.__anext__()
    assert first.delta_text == "first"
    assert not stream.closed.is_set()
    await asyncio.wait_for(iterator.aclose(), timeout=1.0)
    assert stream.closed.is_set()
    await client.aclose()


async def test_cancelling_a_stream_consumer_closes_the_response_promptly():
    stream = TrackingStream(_sse(_delta("first")))
    client = _streaming_client(stream)
    received = asyncio.Event()

    async def _consume() -> None:
        async for _ in client.stream("m", MESSAGES):
            received.set()

    task = asyncio.ensure_future(_consume())
    await asyncio.wait_for(received.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(stream.closed.wait(), timeout=1.0)
    await client.aclose()


async def test_stalled_stream_hits_idle_timeout_and_closes():
    stream = TrackingStream(_sse(_delta("first")))
    client = _streaming_client(stream, stream_idle_timeout_s=0.05)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for _ in client.stream("m", MESSAGES):
            pass
    assert excinfo.value.error.code is ProviderErrorCode.TIMEOUT
    assert stream.closed.is_set()


async def test_stream_byte_limit_is_enforced():
    stream = TrackingStream(_sse(*[_delta("x" * 200)] * 20), hang=False)
    client = _streaming_client(stream, max_stream_bytes=1024)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for _ in client.stream("m", MESSAGES):
            pass
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert stream.closed.is_set()


async def test_unterminated_sse_line_is_bounded():
    stream = TrackingStream(b"data: " + b"x" * 4096, hang=False)
    client = _streaming_client(stream, max_sse_line_bytes=1024)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for _ in client.stream("m", MESSAGES):
            pass
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE


# ── Translation adapter ──────────────────────────────────────────────────────


async def test_translation_adapter_returns_canonical_result():
    recorder = Recorder(_ok(_completion_payload("  Xin chao  ")))
    seen: List[TranslationRequest] = []

    def _builder(request: TranslationRequest) -> Sequence[TextMessage]:
        seen.append(request)
        return (TextMessage("user", request.text),)

    provider = OpenAICompatibleTranslationProvider(
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        "gpt-test",
        message_builder=_builder,
        provider_id="openai",
        generation=TextGenerationOptions(temperature=0.3, max_output_tokens=200),
        http_transport=recorder.transport(),
    )
    assert isinstance(provider, TranslationProvider)
    await provider.start()
    request = TranslationRequest("Hello", "en", "vi", stream_id="mic", sequence=4, revision=2)
    result = await provider.translate(request)
    await provider.stop()

    assert seen == [request]
    assert result.translated_text == "Xin chao"
    assert (result.source_text, result.source_lang, result.target_lang) == ("Hello", "en", "vi")
    assert (result.stream_id, result.sequence, result.revision) == ("mic", 4, 2)
    assert (result.input_tokens, result.output_tokens) == (11, 3)
    assert result.provider_event_id == "chatcmpl-1"
    assert result.latency_ms is not None
    assert provider.capability is TRANSLATION
    assert recorder.body()["temperature"] == 0.3


async def test_translation_adapter_skips_blank_text_and_rejects_empty_output():
    recorder = Recorder(_ok(_completion_payload("   ")))
    provider = OpenAICompatibleTranslationProvider(
        LocalOpenAICompatibleStrategy("http://127.0.0.1:1234/v1"),
        "local-model",
        message_builder=lambda r: MESSAGES,
        http_transport=recorder.transport(),
    )
    blank = await provider.translate(TranslationRequest("  ", "en", "vi"))
    assert blank.translated_text == "" and recorder.requests == []
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await provider.translate(TranslationRequest("hi", "en", "vi"))
    await provider.stop()
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert excinfo.value.error.capability is TRANSLATION


def test_translation_adapter_requires_injected_message_builder():
    with pytest.raises(ProviderRuntimeError) as excinfo:
        OpenAICompatibleTranslationProvider(
            OpenAIStrategy("https://api.openai.com/v1", SECRET),
            "m",
            message_builder=None,  # type: ignore[arg-type]
        )
    assert excinfo.value.error.details["field"] == "message_builder"


# ── Assistant adapter ────────────────────────────────────────────────────────


class _Policy:
    def __init__(self, fail_parse: bool = False) -> None:
        self.fail_parse = fail_parse

    def build_assist_messages(self, request: AssistRequest) -> Sequence[TextMessage]:
        return (TextMessage("user", request.target_message.text),)

    def parse_assist_result(
        self, request: AssistRequest, completion: TextCompletion
    ) -> AssistResult:
        if self.fail_parse:
            raise ValueError("not json")
        return AssistResult(
            target_message_id=request.target_message.message_id,
            suggested_replies=(AssistSuggestion("direct", completion.text),),
            clarifying_questions=(),
            input_tokens=completion.input_tokens,
        )

    def build_summary_messages(self, request: SummaryRequest) -> Sequence[TextMessage]:
        return tuple(TextMessage("user", m.text) for m in request.messages)


def _message(text: str = "Can we ship Friday?") -> ConversationMessage:
    return ConversationMessage(message_id="e-1", speaker="them", text=text, timestamp_ms=1.0)


def _assistant(recorder: Recorder, policy: Any = None, **kwargs: Any) -> Any:
    return OpenAICompatibleAssistantProvider(
        kwargs.pop("strategy", OpenAIStrategy("https://api.openai.com/v1", SECRET)),
        "gpt-test",
        prompt_policy=policy or _Policy(),
        provider_id="openai",
        http_transport=recorder.transport(),
        transport_options=FAST,
        **kwargs,
    )


async def test_assistant_assist_delegates_prompt_and_parsing_to_policy():
    recorder = Recorder(_ok(_completion_payload("Yes, Friday works.")))
    provider = _assistant(recorder)
    assert isinstance(provider, AssistantProvider)
    result = await provider.assist(AssistRequest(target_message=_message()))
    await provider.stop()
    assert result.target_message_id == "e-1"
    assert result.suggested_replies[0].text == "Yes, Friday works."
    assert recorder.body()["messages"] == [{"role": "user", "content": "Can we ship Friday?"}]


async def test_assistant_unparseable_reply_is_invalid_response():
    recorder = Recorder(_ok(_completion_payload(f"garbage {SECRET}")))
    provider = _assistant(recorder, _Policy(fail_parse=True))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await provider.assist(AssistRequest(target_message=_message()))
    await provider.stop()
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert excinfo.value.error.capability is ASSISTANT
    assert excinfo.value.__cause__ is None


async def test_assistant_summarize_returns_markdown_and_uses_reserved_tokens():
    recorder = Recorder(_ok(_completion_payload("- point")))
    provider = _assistant(recorder, generation=TextGenerationOptions(temperature=0.4))
    result = await provider.summarize(
        SummaryRequest(messages=(_message("a"), _message("b")), reserved_output_tokens=321)
    )
    await provider.stop()
    assert result.summary_markdown == "- point"
    assert (result.input_tokens, result.output_tokens) == (11, 3)
    body = recorder.body()
    assert body["max_completion_tokens"] == 321
    assert body["temperature"] == 0.4


async def test_assistant_stream_summary_yields_canonical_chunks():
    recorder = Recorder(
        httpx.Response(200, content=_sse(_delta("Sum"), _delta("mary", "stop"), "[DONE]"))
    )
    provider = _assistant(
        recorder, strategy=LocalOpenAICompatibleStrategy("http://127.0.0.1:1234/v1")
    )
    chunks = [c async for c in provider.stream_summary(SummaryRequest(messages=(_message(),)))]
    await provider.stop()
    assert [(c.delta_text, c.is_done) for c in chunks] == [
        ("Sum", False),
        ("mary", False),
        ("", True),
    ]
    assert "authorization" not in recorder.last.headers
    assert recorder.body()["max_tokens"] == 1000


async def test_assistant_stream_summary_close_releases_response():
    stream = TrackingStream(_sse(_delta("first")))

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    provider = OpenAICompatibleAssistantProvider(
        OpenAIStrategy("https://api.openai.com/v1", SECRET),
        "m",
        prompt_policy=_Policy(),
        http_transport=httpx.MockTransport(_handler),
    )
    iterator = provider.stream_summary(SummaryRequest(messages=(_message(),)))
    assert (await iterator.__anext__()).delta_text == "first"
    await asyncio.wait_for(iterator.aclose(), timeout=1.0)
    assert stream.closed.is_set()
    await provider.stop()


def test_assistant_requires_complete_prompt_policy():
    class _Partial:
        def build_assist_messages(self, request: AssistRequest) -> Sequence[TextMessage]:
            return ()

    with pytest.raises(ProviderRuntimeError) as excinfo:
        OpenAICompatibleAssistantProvider(
            OpenAIStrategy("https://api.openai.com/v1", SECRET), "m", prompt_policy=_Partial()
        )
    assert excinfo.value.error.details["field"] == "prompt_policy"
    assert excinfo.value.error.capability is ASSISTANT


def test_adapter_config_errors_carry_adapter_capability():
    with pytest.raises(ProviderRuntimeError) as excinfo:
        OpenAICompatibleAssistantProvider(
            object(),  # type: ignore[arg-type]
            "m",
            prompt_policy=_Policy(),
        )
    assert excinfo.value.error.capability is ASSISTANT
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_REQUEST


# ── Import hygiene ───────────────────────────────────────────────────────────


def test_modules_import_without_loading_http_or_sdk_clients():
    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "import json, sys\n"
        "import engine.providers.assistant\n"
        "import engine.providers.text.openai_compatible\n"
        "import engine.providers.translation.openai_compatible\n"
        "import engine.providers.assistant.openai_compatible\n"
        "print(json.dumps(sorted(m for m in ('httpx', 'openai', 'aiohttp', 'requests')"
        " if m in sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_assistant_package_exports_resolve_lazily():
    import engine.providers.assistant as package

    assert package.OpenAICompatibleAssistantProvider is OpenAICompatibleAssistantProvider
    with pytest.raises(AttributeError):
        _ = package.DoesNotExist
