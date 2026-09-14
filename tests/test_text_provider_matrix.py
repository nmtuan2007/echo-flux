"""Pins the Gemini text transport/adapters and the approved Translation/Assistant matrix.

Every Gemini test uses ``httpx.MockTransport`` or an in-memory byte stream; nothing reaches
a network and no SDK is imported. SECRET is a sentinel that must never appear in errors,
reprs, logs, or registry output.
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from engine.core.config import CredentialRef, ServiceConfig
from engine.providers.assistant.gemini import GeminiAssistantProvider
from engine.providers.contracts import AssistantProvider, TranslationProvider
from engine.providers.manifests import BUILTIN_MANIFESTS, FieldType, ProtocolFamily
from engine.providers.registry import (
    ProviderRegistryError,
    RegistryErrorCode,
    build_default_registry,
)
from engine.providers.text.gemini import (
    GeminiResponseFormat,
    GeminiStrategy,
    GeminiTextClient,
)
from engine.providers.text.openai_compatible import (
    TextCompletion,
    TextConfigError,
    TextGenerationOptions,
    TextMessage,
    TextTransportOptions,
)
from engine.providers.translation.gemini import GeminiTranslationProvider
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

SECRET = "gm-test-not-real-EF018-0001"
STT = ProviderCapability.STT
TRANSLATION = ProviderCapability.TRANSLATION
ASSISTANT = ProviderCapability.ASSISTANT
ENDPOINT = "https://gemini.example.invalid/v1beta"
MESSAGES = (TextMessage("system", "sys"), TextMessage("user", "hello"))
FAST = TextTransportOptions(retry_backoff_s=0.0, max_retry_after_s=0.0)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _response(text: str = "xin chao", finish: Optional[str] = "STOP", **extra: Any) -> Dict:
    candidate: Dict[str, Any] = {
        "content": {"role": "model", "parts": [{"text": text}]},
        "index": 0,
    }
    if finish is not None:
        candidate["finishReason"] = finish
    payload: Dict[str, Any] = {
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3},
        "responseId": "resp-1",
        "modelVersion": "gemini-test",
    }
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
        return item

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self, index: int = -1) -> Dict[str, Any]:
        return json.loads(self.requests[index].content)


def _ok(payload: Optional[Dict[str, Any]] = None) -> httpx.Response:
    return httpx.Response(200, json=payload or _response())


def _strategy() -> GeminiStrategy:
    return GeminiStrategy(ENDPOINT, SECRET)


def _client(recorder: Recorder, capability: ProviderCapability = TRANSLATION) -> GeminiTextClient:
    async def _no_sleep(_: float) -> None:
        return None

    return GeminiTextClient(
        _strategy(),
        capability=capability,
        options=FAST,
        http_transport=recorder.transport(),
        sleep=_no_sleep,
    )


def _sse(*events: Any) -> bytes:
    return "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event)}\r\n\r\n"
        for event in events
    ).encode()


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


def _assert_no_secret(*values: Any) -> None:
    for value in values:
        text = value if isinstance(value, str) else repr(value)
        assert SECRET not in text


async def _drain(iterator: Any) -> List[Any]:
    return [chunk async for chunk in iterator]


# ── Gemini request and auth shape ────────────────────────────────────────────


async def test_completion_request_uses_header_auth_and_gemini_body_shape():
    payload = _response()
    payload["candidates"][0]["content"]["parts"] = [
        {"text": "thinking...", "thought": True},
        {"text": "Xin "},
        {"text": "chao"},
    ]
    payload["usageMetadata"]["thoughtsTokenCount"] = 2
    recorder = Recorder(_ok(payload))
    client = _client(recorder)
    messages = (
        TextMessage("system", "rules"),
        TextMessage("developer", "more rules"),
        TextMessage("user", "Hello"),
        TextMessage("assistant", "Hi"),
        TextMessage("user", "Translate"),
    )
    completion = await client.complete(
        "gemini-test", messages, TextGenerationOptions(temperature=0.3, max_output_tokens=200)
    )
    await client.aclose()

    request = recorder.last
    assert request.method == "POST"
    assert str(request.url) == f"{ENDPOINT}/models/gemini-test:generateContent"
    assert request.url.query == b""
    assert request.headers["x-goog-api-key"] == SECRET
    assert "authorization" not in request.headers
    assert SECRET not in str(request.url)
    assert recorder.body() == {
        "systemInstruction": {"parts": [{"text": "rules"}, {"text": "more rules"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi"}]},
            {"role": "user", "parts": [{"text": "Translate"}]},
        ],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200},
    }
    assert completion.text == "Xin chao"
    assert completion.finish_reason == "STOP"
    assert completion.response_id == "resp-1"
    assert (completion.input_tokens, completion.output_tokens) == (11, 5)
    assert completion.latency_ms is not None


async def test_model_prefix_is_normalized_and_generation_config_omitted_when_unset():
    recorder = Recorder(_ok())
    client = _client(recorder)
    await client.complete("models/gemini-test", (TextMessage("user", "hi"),))
    assert recorder.last.url.path == "/v1beta/models/gemini-test:generateContent"
    assert "generationConfig" not in recorder.body()
    assert "systemInstruction" not in recorder.body()


async def test_structured_output_maps_to_response_mime_type_and_json_schema():
    schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    recorder = Recorder(_ok(_response('{"text": "ok"}')))
    client = _client(recorder, ASSISTANT)
    completion = await client.complete(
        "gemini-test", MESSAGES, response_format=GeminiResponseFormat(json_schema=schema)
    )
    assert json.loads(completion.text) == {"text": "ok"}
    assert recorder.body()["generationConfig"] == {
        "responseMimeType": "application/json",
        "responseJsonSchema": schema,
    }


@pytest.mark.parametrize(
    ("endpoint", "api_key", "field_name"),
    [
        ("", SECRET, "endpoint"),
        ("http://gemini.example.invalid/v1beta", SECRET, "endpoint"),
        (f"https://gemini.example.invalid/v1beta?key={SECRET}", SECRET, "endpoint"),
        ("https://gemini.example.invalid/v1beta?alt=json", SECRET, "endpoint"),
        (f"https://user:{SECRET}@gemini.example.invalid/v1beta", SECRET, "endpoint"),
        ("https://gemini.example.invalid/v1beta/models", SECRET, "endpoint"),
        (ENDPOINT, "", "api_key"),
        (ENDPOINT, None, "api_key"),
        (ENDPOINT, "bad\nkey", "api_key"),
    ],
)
def test_strategy_requires_resolved_endpoint_and_key_without_defaults(
    endpoint, api_key, field_name
):
    with pytest.raises(TextConfigError) as excinfo:
        GeminiStrategy(endpoint, api_key)
    assert excinfo.value.field_name == field_name
    _assert_no_secret(str(excinfo.value))


@pytest.mark.parametrize(
    ("model", "messages", "generation", "field_name"),
    [
        ("", MESSAGES, None, "model"),
        ("../evil", MESSAGES, None, "model"),
        ("gemini-test:generateContent", MESSAGES, None, "model"),
        ("gemini-test", (), None, "messages"),
        ("gemini-test", (TextMessage("system", "only system"),), None, "messages"),
        ("gemini-test", (TextMessage("tool", "x"),), None, "messages"),
        ("gemini-test", MESSAGES, TextGenerationOptions(temperature=3.0), "temperature"),
        ("gemini-test", MESSAGES, TextGenerationOptions(max_output_tokens=0), "max_output_tokens"),
    ],
)
async def test_invalid_requests_fail_before_any_network_call(
    model, messages, generation, field_name
):
    recorder = Recorder(_ok())
    client = _client(recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete(model, messages, generation)
    assert recorder.requests == []
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_REQUEST
    assert excinfo.value.error.details["field"] == field_name
    assert excinfo.value.error.details["mode"] == "gemini"


def test_client_rejects_non_gemini_strategy():
    from engine.providers.text.openai_compatible import OpenAIStrategy

    with pytest.raises(TextConfigError):
        GeminiTextClient(
            OpenAIStrategy("https://api.openai.com/v1", SECRET),  # type: ignore[arg-type]
            capability=TRANSLATION,
        )


# ── Errors, safety, and redaction ────────────────────────────────────────────


def _google_error(status: int, status_text: str, reason: Optional[str] = None) -> Dict:
    error: Dict[str, Any] = {
        "code": status,
        "message": f"API key {SECRET} rejected",
        "status": status_text,
    }
    if reason:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    return {"error": error}


@pytest.mark.parametrize(
    ("status", "body", "code", "retryable"),
    [
        (
            400,
            _google_error(400, "INVALID_ARGUMENT", "API_KEY_INVALID"),
            ProviderErrorCode.AUTH_FAILED,
            False,
        ),
        (400, _google_error(400, "INVALID_ARGUMENT"), ProviderErrorCode.INVALID_REQUEST, False),
        (400, _google_error(400, "FAILED_PRECONDITION"), ProviderErrorCode.INVALID_REQUEST, False),
        (403, _google_error(403, "PERMISSION_DENIED"), ProviderErrorCode.AUTH_FAILED, False),
        (404, _google_error(404, "NOT_FOUND"), ProviderErrorCode.MODEL_UNAVAILABLE, False),
        (429, _google_error(429, "RESOURCE_EXHAUSTED"), ProviderErrorCode.RATE_LIMITED, True),
        (
            429,
            {"error": {"code": "quota_exceeded", "message": SECRET}},
            ProviderErrorCode.QUOTA_EXCEEDED,
            False,
        ),
        (500, _google_error(500, "INTERNAL"), ProviderErrorCode.INTERNAL_ERROR, True),
        (503, _google_error(503, "UNAVAILABLE"), ProviderErrorCode.NETWORK_UNAVAILABLE, True),
        (504, _google_error(504, "DEADLINE_EXCEEDED"), ProviderErrorCode.NETWORK_UNAVAILABLE, True),
        (502, None, ProviderErrorCode.NETWORK_UNAVAILABLE, True),
    ],
)
async def test_http_errors_map_to_stable_codes_without_leaking(
    status, body, code, retryable, captured_logs
):
    response = (
        httpx.Response(status, json=body)
        if body is not None
        else httpx.Response(status, content=f"<html>{SECRET}</html>".encode())
    )
    recorder = Recorder(response)
    client = _client(recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    error = excinfo.value.error
    assert error.code is code
    assert error.retryable is retryable
    assert error.details["http_status"] == status
    assert len(recorder.requests) == 1
    _assert_no_secret(str(excinfo.value), error.details, error.message)
    _assert_no_secret("\n".join(r.getMessage() for r in captured_logs))


async def test_prompt_block_is_normalized_as_blocked_request():
    payload = {
        "promptFeedback": {"blockReason": "PROHIBITED_CONTENT", "safetyRatings": []},
        "usageMetadata": {"promptTokenCount": 4},
    }
    client = _client(Recorder(_ok(payload)))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    error = excinfo.value.error
    assert error.code is ProviderErrorCode.INVALID_REQUEST
    assert error.retryable is False
    assert error.details == {
        "mode": "gemini",
        "provider_code": "PROHIBITED_CONTENT",
        "blocked": "prompt",
    }


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "BLOCKLIST", "SPII"])
async def test_withheld_candidate_is_normalized_as_blocked_response(reason):
    payload = _response(finish=reason)
    del payload["candidates"][0]["content"]
    payload["candidates"][0]["safetyRatings"] = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH", "blocked": True}
    ]
    client = _client(Recorder(_ok(payload)))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    error = excinfo.value.error
    assert error.code is ProviderErrorCode.INVALID_RESPONSE
    assert error.details["blocked"] == "response"
    assert error.details["provider_code"] == reason
    assert error.retryable is False


async def test_unsupported_language_finish_maps_to_stable_code():
    client = _client(Recorder(_ok(_response(text="", finish="LANGUAGE"))))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.UNSUPPORTED_LANGUAGE


async def test_token_limited_completion_keeps_text_and_finish_reason():
    client = _client(Recorder(_ok(_response("partial", finish="MAX_TOKENS"))))
    completion = await client.complete("gemini-test", MESSAGES)
    assert (completion.text, completion.finish_reason) == ("partial", "MAX_TOKENS")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"candidates": []},
        {"candidates": "nope"},
        {"candidates": ["nope"]},
        {"candidates": [{"content": {"parts": "nope"}}]},
        ["not", "an", "object"],
    ],
)
async def test_malformed_responses_are_invalid_response(payload):
    client = _client(Recorder(httpx.Response(200, json=payload)))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE


async def test_non_json_body_is_invalid_response():
    client = _client(Recorder(httpx.Response(200, content=b"<html>")))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE


async def test_generation_post_is_not_retried_after_the_request_was_sent(captured_logs):
    recorder = Recorder(httpx.ReadTimeout(f"timed out {SECRET}"), _ok())
    client = _client(recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await client.complete("gemini-test", MESSAGES)
    assert excinfo.value.error.code is ProviderErrorCode.TIMEOUT
    assert len(recorder.requests) == 1
    _assert_no_secret("\n".join(r.getMessage() for r in captured_logs))


async def test_generation_post_is_retried_when_connection_never_opened():
    recorder = Recorder(httpx.ConnectError("refused"), _ok())
    client = _client(recorder)
    completion = await client.complete("gemini-test", MESSAGES)
    assert completion.text == "xin chao"
    assert len(recorder.requests) == 2


def test_reprs_never_contain_secrets():
    strategy = _strategy()
    client = GeminiTextClient(strategy, capability=TRANSLATION)
    translation = GeminiTranslationProvider(strategy, "m", message_builder=lambda r: MESSAGES)
    assistant = GeminiAssistantProvider(strategy, "m", prompt_policy=_Policy())
    _assert_no_secret(repr(strategy), repr(client), repr(translation), repr(assistant))
    assert client.redacted_endpoint == f"{ENDPOINT}/models"


# ── Streaming ────────────────────────────────────────────────────────────────


def _chunk(text: str, finish: Optional[str] = None, usage: bool = False) -> Dict[str, Any]:
    payload = _response(text, finish=finish)
    if not usage:
        del payload["usageMetadata"]
    return payload


async def test_stream_yields_deltas_usage_and_final_chunk():
    body = b": keepalive\r\n\r\n" + _sse(
        _chunk("Xin "), "not-json", _chunk("chao", "STOP", usage=True)
    )
    recorder = Recorder(httpx.Response(200, content=body))
    client = _client(recorder, ASSISTANT)
    chunks = await _drain(client.stream("gemini-test", MESSAGES))
    await client.aclose()

    assert [c.delta_text for c in chunks] == ["Xin ", "chao", ""]
    final = chunks[-1]
    assert final.is_done and final.completion is not None
    assert final.completion.text == "Xin chao"
    assert final.completion.finish_reason == "STOP"
    assert (final.completion.input_tokens, final.completion.output_tokens) == (11, 3)
    assert final.completion.response_id == "resp-1"
    request = recorder.last
    assert request.url.path == "/v1beta/models/gemini-test:streamGenerateContent"
    assert parse_qs(urlsplit(str(request.url)).query) == {"alt": ["sse"]}
    assert request.headers["x-goog-api-key"] == SECRET
    assert "stream" not in recorder.body()


async def test_stream_ending_without_finish_reason_is_connection_closed():
    client = _client(Recorder(httpx.Response(200, content=_sse(_chunk("partial")))))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await _drain(client.stream("gemini-test", MESSAGES))
    assert excinfo.value.error.code is ProviderErrorCode.CONNECTION_CLOSED


async def test_stream_error_event_is_mapped_and_redacted():
    event = _google_error(429, "RESOURCE_EXHAUSTED")
    client = _client(Recorder(httpx.Response(200, content=_sse(_chunk("a"), event))))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await _drain(client.stream("gemini-test", MESSAGES))
    assert excinfo.value.error.code is ProviderErrorCode.RATE_LIMITED
    _assert_no_secret(str(excinfo.value), excinfo.value.error.details)


async def test_stream_safety_block_after_partial_output_is_raised():
    body = _sse(_chunk("first "), {"candidates": [{"finishReason": "SAFETY", "index": 0}]})
    client = _client(Recorder(httpx.Response(200, content=body)), ASSISTANT)
    received: List[str] = []
    with pytest.raises(ProviderRuntimeError) as excinfo:
        async for chunk in client.stream("gemini-test", MESSAGES):
            received.append(chunk.delta_text)
    assert received == ["first "]
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert excinfo.value.error.details["blocked"] == "response"


async def test_stream_prompt_block_is_raised_before_any_delta():
    body = _sse({"promptFeedback": {"blockReason": "SAFETY"}})
    client = _client(Recorder(httpx.Response(200, content=body)))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await _drain(client.stream("gemini-test", MESSAGES))
    assert excinfo.value.error.details["blocked"] == "prompt"


async def test_stream_http_error_is_raised_before_any_chunk():
    recorder = Recorder(httpx.Response(401, json=_google_error(401, "UNAUTHENTICATED")))
    client = _client(recorder)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await _drain(client.stream("gemini-test", MESSAGES))
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


def _streaming_client(stream: TrackingStream, **options: Any) -> GeminiTextClient:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    return GeminiTextClient(
        _strategy(),
        capability=ASSISTANT,
        options=TextTransportOptions(**options),
        http_transport=httpx.MockTransport(_handler),
    )


async def test_closing_a_stream_early_closes_the_response_promptly():
    stream = TrackingStream(_sse(_chunk("first")))
    client = _streaming_client(stream)
    iterator = client.stream("gemini-test", MESSAGES)
    assert (await iterator.__anext__()).delta_text == "first"
    assert not stream.closed.is_set()
    await asyncio.wait_for(iterator.aclose(), timeout=1.0)
    assert stream.closed.is_set()
    await client.aclose()


async def test_cancelling_a_stream_consumer_closes_the_response_promptly():
    stream = TrackingStream(_sse(_chunk("first")))
    client = _streaming_client(stream)
    received = asyncio.Event()

    async def _consume() -> None:
        async for _ in client.stream("gemini-test", MESSAGES):
            received.set()

    task = asyncio.ensure_future(_consume())
    await asyncio.wait_for(received.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(stream.closed.wait(), timeout=1.0)
    await client.aclose()


async def test_stalled_stream_hits_idle_timeout_and_closes():
    stream = TrackingStream(_sse(_chunk("first")))
    client = _streaming_client(stream, stream_idle_timeout_s=0.05)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await _drain(client.stream("gemini-test", MESSAGES))
    assert excinfo.value.error.code is ProviderErrorCode.TIMEOUT
    assert stream.closed.is_set()


async def test_stream_byte_limit_is_enforced():
    stream = TrackingStream(_sse(*[_chunk("x" * 200)] * 20), hang=False)
    client = _streaming_client(stream, max_stream_bytes=1024)
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await _drain(client.stream("gemini-test", MESSAGES))
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert stream.closed.is_set()


# ── Translation adapter ──────────────────────────────────────────────────────


async def test_translation_adapter_returns_canonical_result():
    recorder = Recorder(_ok(_response("  Xin chao  ")))
    seen: List[TranslationRequest] = []

    def _builder(request: TranslationRequest) -> Sequence[TextMessage]:
        seen.append(request)
        return (TextMessage("user", request.text),)

    provider = GeminiTranslationProvider(
        _strategy(),
        "gemini-test",
        message_builder=_builder,
        generation=TextGenerationOptions(temperature=0.3, max_output_tokens=200),
        transport_options=FAST,
        http_transport=recorder.transport(),
    )
    assert isinstance(provider, TranslationProvider)
    assert provider.provider_id == "gemini"
    assert provider.capability is TRANSLATION
    await provider.start()
    request = TranslationRequest("Hello", "auto", "vi", stream_id="mic", sequence=4, revision=2)
    result = await provider.translate(request)
    await provider.stop()

    assert seen == [request]
    assert result.translated_text == "Xin chao"
    assert (result.source_text, result.source_lang, result.target_lang) == ("Hello", "auto", "vi")
    assert (result.stream_id, result.sequence, result.revision) == ("mic", 4, 2)
    assert (result.input_tokens, result.output_tokens) == (11, 3)
    assert result.provider_event_id == "resp-1"
    assert recorder.body()["generationConfig"] == {"temperature": 0.3, "maxOutputTokens": 200}


async def test_translation_adapter_skips_blank_text_and_rejects_empty_output():
    recorder = Recorder(_ok(_response("   ")))
    provider = GeminiTranslationProvider(
        _strategy(),
        "gemini-test",
        message_builder=lambda r: MESSAGES,
        http_transport=recorder.transport(),
    )
    blank = await provider.translate(TranslationRequest("  ", "auto", "vi"))
    assert blank.translated_text == "" and recorder.requests == []
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await provider.translate(TranslationRequest("hi", "auto", "vi"))
    await provider.stop()
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert excinfo.value.error.capability is TRANSLATION


async def test_translation_adapter_blocked_content_keeps_translation_capability():
    recorder = Recorder(_ok({"promptFeedback": {"blockReason": "SAFETY"}}))
    provider = GeminiTranslationProvider(
        _strategy(), "m", message_builder=lambda r: MESSAGES, http_transport=recorder.transport()
    )
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await provider.translate(TranslationRequest("hi", "auto", "vi"))
    assert excinfo.value.error.capability is TRANSLATION
    assert excinfo.value.error.details["blocked"] == "prompt"


def test_translation_adapter_requires_builder_and_gemini_strategy():
    with pytest.raises(ProviderRuntimeError) as excinfo:
        GeminiTranslationProvider(_strategy(), "m", message_builder=None)  # type: ignore[arg-type]
    assert excinfo.value.error.details["field"] == "message_builder"
    with pytest.raises(ProviderRuntimeError) as excinfo:
        GeminiTranslationProvider(object(), "m", message_builder=lambda r: MESSAGES)  # type: ignore[arg-type]
    assert excinfo.value.error.capability is TRANSLATION
    assert excinfo.value.error.details["field"] == "strategy"


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
        )

    def build_summary_messages(self, request: SummaryRequest) -> Sequence[TextMessage]:
        return tuple(TextMessage("user", m.text) for m in request.messages)


def _message(text: str = "Can we ship Friday?") -> ConversationMessage:
    return ConversationMessage(message_id="e-1", speaker="them", text=text, timestamp_ms=1.0)


def _assistant(recorder: Recorder, policy: Any = None, **kwargs: Any) -> GeminiAssistantProvider:
    return GeminiAssistantProvider(
        _strategy(),
        "gemini-test",
        prompt_policy=policy or _Policy(),
        http_transport=recorder.transport(),
        transport_options=FAST,
        **kwargs,
    )


async def test_assistant_assist_delegates_prompt_and_parsing_to_policy():
    recorder = Recorder(_ok(_response("Yes, Friday works.")))
    provider = _assistant(recorder)
    assert isinstance(provider, AssistantProvider)
    assert provider.capability is ASSISTANT
    result = await provider.assist(AssistRequest(target_message=_message()))
    await provider.stop()
    assert result.suggested_replies[0].text == "Yes, Friday works."
    assert recorder.body()["contents"] == [
        {"role": "user", "parts": [{"text": "Can we ship Friday?"}]}
    ]


async def test_assistant_unparseable_reply_is_invalid_response():
    recorder = Recorder(_ok(_response(f"garbage {SECRET}")))
    provider = _assistant(recorder, _Policy(fail_parse=True))
    with pytest.raises(ProviderRuntimeError) as excinfo:
        await provider.assist(AssistRequest(target_message=_message()))
    assert excinfo.value.error.code is ProviderErrorCode.INVALID_RESPONSE
    assert excinfo.value.error.capability is ASSISTANT
    assert excinfo.value.__cause__ is None
    _assert_no_secret(str(excinfo.value), excinfo.value.error.details)


async def test_assistant_summarize_uses_reserved_tokens():
    recorder = Recorder(_ok(_response("- point")))
    provider = _assistant(recorder, generation=TextGenerationOptions(temperature=0.4))
    result = await provider.summarize(
        SummaryRequest(messages=(_message("a"), _message("b")), reserved_output_tokens=321)
    )
    assert result.summary_markdown == "- point"
    assert result.provider_event_id == "resp-1"
    assert recorder.body()["generationConfig"] == {"temperature": 0.4, "maxOutputTokens": 321}


async def test_assistant_stream_summary_yields_canonical_chunks():
    body = _sse(_chunk("Sum"), _chunk("mary", "STOP", usage=True))
    recorder = Recorder(httpx.Response(200, content=body))
    provider = _assistant(recorder)
    chunks = await _drain(provider.stream_summary(SummaryRequest(messages=(_message(),))))
    await provider.stop()
    assert [(c.delta_text, c.is_done) for c in chunks] == [
        ("Sum", False),
        ("mary", False),
        ("", True),
    ]
    assert recorder.body()["generationConfig"] == {"maxOutputTokens": 1000}


async def test_assistant_stream_summary_close_releases_response():
    stream = TrackingStream(_sse(_chunk("first")))

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    provider = GeminiAssistantProvider(
        _strategy(), "m", prompt_policy=_Policy(), http_transport=httpx.MockTransport(_handler)
    )
    iterator = provider.stream_summary(SummaryRequest(messages=(_message(),)))
    assert (await iterator.__anext__()).delta_text == "first"
    await asyncio.wait_for(iterator.aclose(), timeout=1.0)
    assert stream.closed.is_set()
    await provider.stop()


async def test_assistant_structured_output_passes_through_completion_primitive():
    recorder = Recorder(_ok(_response("{}")))
    provider = _assistant(recorder)
    await provider.complete_messages(MESSAGES, response_format=GeminiResponseFormat())
    assert recorder.body()["generationConfig"] == {"responseMimeType": "application/json"}


def test_assistant_requires_complete_prompt_policy():
    class _Partial:
        def build_assist_messages(self, request: AssistRequest) -> Sequence[TextMessage]:
            return ()

    with pytest.raises(ProviderRuntimeError) as excinfo:
        GeminiAssistantProvider(_strategy(), "m", prompt_policy=_Partial())  # type: ignore[arg-type]
    assert excinfo.value.error.details["field"] == "prompt_policy"
    assert excinfo.value.error.capability is ASSISTANT


async def test_health_checks_probe_model_listing_with_header_auth():
    recorder = Recorder(httpx.Response(200, json={"models": []}))
    provider = _assistant(recorder)
    healthy = await provider.health_check()
    assert healthy.status is ProviderHealthStatus.HEALTHY
    assert recorder.last.method == "GET"
    assert str(recorder.last.url) == f"{ENDPOINT}/models"
    assert recorder.last.headers["x-goog-api-key"] == SECRET

    denied = Recorder(httpx.Response(403, json=_google_error(403, "PERMISSION_DENIED")))
    translation = GeminiTranslationProvider(
        _strategy(),
        "m",
        message_builder=lambda r: MESSAGES,
        transport_options=FAST,
        http_transport=denied.transport(),
    )
    unhealthy = await translation.health_check()
    assert unhealthy.status is ProviderHealthStatus.UNHEALTHY
    assert unhealthy.details["code"] == "auth_failed"
    _assert_no_secret(unhealthy.message, unhealthy.details)


# ── Import hygiene and default ownership ─────────────────────────────────────


def test_gemini_modules_import_without_loading_http_or_sdk_clients():
    probe = (
        "import json, sys\n"
        "import engine.providers.text.gemini\n"
        "import engine.providers.translation.gemini\n"
        "import engine.providers.assistant.gemini\n"
        "watched = ('httpx', 'google.genai', 'google.generativeai', 'openai', 'requests')\n"
        "print(json.dumps(sorted(m for m in watched if m in sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_provider_defaults_live_only_in_manifests():
    registry = build_default_registry()
    recommended = set()
    for manifest in BUILTIN_MANIFESTS:
        if manifest.supports(TRANSLATION) or manifest.supports(ASSISTANT):
            recommended.update(v for v in (manifest.recommended_endpoint,) if v)
            recommended.update(v for v in (manifest.recommended_model,) if v)
    gemini = registry.get(ASSISTANT, "gemini")
    assert gemini.recommended_model in recommended
    for relative in (
        "engine/providers/text/gemini.py",
        "engine/providers/translation/gemini.py",
        "engine/providers/assistant/gemini.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "googleapis.com" not in source, relative
        for value in recommended:
            assert value not in source, (relative, value)


# ── Approved text matrix and registration ────────────────────────────────────

TEXT_MATRIX = {
    TRANSLATION: {
        "local",
        "openai",
        "gemini",
        "azure-openai",
        "assemblyai-llm-gateway",
        "custom-openai-compatible",
    },
    ASSISTANT: {
        "openai",
        "gemini",
        "azure-openai",
        "assemblyai-llm-gateway",
        "local-openai-compatible",
        "custom-openai-compatible",
    },
}

EXPECTED_TEXT_REGISTRATION = {
    "gemini": (
        ProtocolFamily.GEMINI_TEXT,
        {
            TRANSLATION: "engine.providers.translation.gemini:GeminiTranslationProvider",
            ASSISTANT: "engine.providers.assistant.gemini:GeminiAssistantProvider",
        },
    ),
    **{
        provider_id: (
            ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
            {
                TRANSLATION: (
                    "engine.providers.translation.openai_compatible:"
                    "OpenAICompatibleTranslationProvider"
                ),
                ASSISTANT: (
                    "engine.providers.assistant.openai_compatible:OpenAICompatibleAssistantProvider"
                ),
            },
        )
        for provider_id in (
            "openai",
            "azure-openai",
            "assemblyai-llm-gateway",
            "custom-openai-compatible",
        )
    },
    "local-openai-compatible": (
        ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        {
            ASSISTANT: (
                "engine.providers.assistant.openai_compatible:OpenAICompatibleAssistantProvider"
            )
        },
    ),
}


def _ref(version: int = 1) -> CredentialRef:
    return CredentialRef(config_id="main", slot="api_key", version=version)


@pytest.mark.parametrize("capability", [TRANSLATION, ASSISTANT])
def test_text_capability_matrix_is_exact(capability):
    registry = build_default_registry()
    listed = {m.provider_id for m in registry.providers_for(capability)}
    assert listed == TEXT_MATRIX[capability]


def test_text_providers_register_expected_protocol_and_factories():
    registry = build_default_registry()
    for provider_id, (family, factories) in EXPECTED_TEXT_REGISTRATION.items():
        manifest = registry.get(next(iter(factories)), provider_id)
        assert manifest.protocol_family is family, provider_id
        text_factories = {
            cap: ref for cap, ref in manifest.factories.items() if cap in (TRANSLATION, ASSISTANT)
        }
        assert text_factories == factories, provider_id
    local = registry.get(TRANSLATION, "local")
    assert local.protocol_family is ProtocolFamily.LOCAL
    assert local.factories[TRANSLATION] == (
        "engine.providers.translation.local:LocalTranslationProvider"
    )


def test_text_factories_resolve_to_capability_contracts():
    registry = build_default_registry()
    contracts = {TRANSLATION: TranslationProvider, ASSISTANT: AssistantProvider}
    for provider_id, (_, factories) in EXPECTED_TEXT_REGISTRATION.items():
        for capability in factories:
            factory = registry.load_factory(capability, provider_id)
            assert issubclass(factory, contracts[capability]), (provider_id, capability)
    assert registry.load_factory(TRANSLATION, "gemini") is GeminiTranslationProvider
    assert registry.load_factory(ASSISTANT, "gemini") is GeminiAssistantProvider


@pytest.mark.parametrize(
    ("capability", "provider_id", "codes"),
    [
        (ASSISTANT, "local", {RegistryErrorCode.UNSUPPORTED_CAPABILITY}),
        (TRANSLATION, "local-openai-compatible", {RegistryErrorCode.UNSUPPORTED_CAPABILITY}),
        (STT, "gemini", {RegistryErrorCode.UNSUPPORTED_CAPABILITY}),
        (STT, "azure-openai", {RegistryErrorCode.UNSUPPORTED_CAPABILITY}),
        (STT, "assemblyai-llm-gateway", {RegistryErrorCode.UNSUPPORTED_CAPABILITY}),
        (
            TRANSLATION,
            "deepgram",
            {RegistryErrorCode.UNKNOWN_PROVIDER, RegistryErrorCode.UNSUPPORTED_CAPABILITY},
        ),
        (
            ASSISTANT,
            "azure-speech",
            {RegistryErrorCode.UNKNOWN_PROVIDER, RegistryErrorCode.UNSUPPORTED_CAPABILITY},
        ),
        (
            TRANSLATION,
            "gemini-live",
            {RegistryErrorCode.UNSUPPORTED_CAPABILITY},
        ),
        (ASSISTANT, "google-translate", {RegistryErrorCode.UNKNOWN_PROVIDER}),
    ],
)
def test_unsupported_capability_provider_pairs_fail_early(capability, provider_id, codes):
    registry = build_default_registry()
    service = ServiceConfig(
        service=capability.value,
        provider_id=provider_id,
        endpoint="https://edited.example.invalid/v1",
        model_override="m",
        credential_refs={"api_key": _ref()},
    )
    for call in (
        lambda: registry.get(capability, provider_id),
        lambda: registry.resolve_service(service),
        lambda: registry.load_factory(capability, provider_id),
    ):
        with pytest.raises(ProviderRegistryError) as excinfo:
            call()
        assert excinfo.value.code in codes


def test_speech_only_vendors_never_serve_text():
    for manifest in BUILTIN_MANIFESTS:
        if manifest.supports(TRANSLATION) or manifest.supports(ASSISTANT):
            assert manifest.provider_id not in {"deepgram", "azure-speech", "assemblyai"}
            assert manifest.protocol_family in (
                ProtocolFamily.LOCAL,
                ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
                ProtocolFamily.GEMINI_TEXT,
            )


def test_recommended_endpoints_and_models_for_text_providers():
    registry = build_default_registry()
    gemini = registry.get(TRANSLATION, "gemini")
    assert gemini.recommended_endpoint == "https://generativelanguage.googleapis.com/v1beta"
    assert gemini.recommended_model == "gemini-3.8-flash"
    gateway = registry.get(ASSISTANT, "assemblyai-llm-gateway")
    assert gateway.recommended_endpoint == "https://llm-gateway.assemblyai.com/v1"
    assert gateway.recommended_model == "gpt-5.6-terra"
    azure = registry.get(ASSISTANT, "azure-openai")
    # Azure endpoints are per resource; there is nothing sensible to recommend.
    assert azure.recommended_endpoint == "" and azure.requires_endpoint
    assert azure.recommended_model == "gpt-5.6-terra"
    local = registry.get(ASSISTANT, "local-openai-compatible")
    assert local.requires_endpoint
    assert local.recommended_model == ""


def test_text_fields_are_provider_aware_typed_and_secret_only_as_slots():
    registry = build_default_registry()
    for provider_id in TEXT_MATRIX[TRANSLATION] | TEXT_MATRIX[ASSISTANT]:
        for capability in (TRANSLATION, ASSISTANT):
            try:
                manifest = registry.get(capability, provider_id)
            except ProviderRegistryError:
                continue
            assert manifest.endpoint_editable
            for spec in manifest.fields_for(capability):
                assert isinstance(spec.field_type, FieldType)
                if spec.secret:
                    assert spec.field_type is FieldType.SECRET and spec.default is None
            source = manifest.get_field("source_lang", capability)
            if capability is TRANSLATION:
                assert source is not None and source.default == "auto" and source.advanced
                target = manifest.get_field("target_lang", capability)
                assert target is not None and not target.advanced
            else:
                assert source is None
                assert manifest.get_field("target_lang", capability) is None
            if provider_id == "local":
                continue
            api_key = manifest.get_field("api_key", capability)
            assert api_key is not None and api_key.secret
            optional_key = provider_id in ("custom-openai-compatible", "local-openai-compatible")
            assert api_key.required is not optional_key, provider_id

    azure = registry.get(ASSISTANT, "azure-openai")
    deployment = azure.get_field("deployment")
    assert deployment.field_type is FieldType.STRING
    assert deployment.required and deployment.default is None and not deployment.advanced
    api_version = azure.get_field("api_version")
    assert api_version.field_type is FieldType.STRING
    assert (api_version.default, api_version.advanced) == ("v1", True)
    for other in ("openai", "gemini", "assemblyai-llm-gateway"):
        assert registry.get(ASSISTANT, other).get_field("deployment") is None


def test_azure_requires_resource_endpoint_and_deployment():
    registry = build_default_registry()
    base = ServiceConfig(
        service="translation", provider_id="azure-openai", credential_refs={"api_key": _ref()}
    )
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(base)
    assert excinfo.value.code is RegistryErrorCode.MISSING_ENDPOINT

    with_endpoint = base.with_changes(endpoint="https://my-resource.openai.azure.com")
    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(with_endpoint)
    assert excinfo.value.code is RegistryErrorCode.INVALID_OPTION
    assert excinfo.value.field_key == "deployment"

    resolved = registry.resolve_service(with_endpoint.with_changes(options={"deployment": "prod"}))
    assert resolved.options == {
        "deployment": "prod",
        "api_version": "v1",
        "source_lang": "auto",
        "target_lang": "vi",
        "temperature": 0.3,
        "max_output_tokens": 1000,
    }
    assert resolved.model == "gpt-5.6-terra"
    assert resolved.protocol_family is ProtocolFamily.OPENAI_COMPATIBLE_TEXT


def test_gemini_resolves_with_manifest_defaults_and_edited_endpoint():
    registry = build_default_registry()
    service = ServiceConfig(
        service="assistant", provider_id="gemini", credential_refs={"api_key": _ref()}
    )
    resolved = registry.resolve_service(service)
    assert resolved.endpoint == "https://generativelanguage.googleapis.com/v1beta"
    assert resolved.model == "gemini-3.8-flash"
    assert resolved.protocol_family is ProtocolFamily.GEMINI_TEXT
    assert resolved.options == {"temperature": 0.3, "max_output_tokens": 1000}

    edited = registry.resolve_service(
        service.with_changes(endpoint="https://proxy.example.invalid/v1beta")
    )
    assert edited.endpoint == "https://proxy.example.invalid/v1beta"

    with pytest.raises(ProviderRegistryError) as excinfo:
        registry.resolve_service(service.with_changes(credential_refs={}))
    assert excinfo.value.code is RegistryErrorCode.MISSING_CREDENTIAL


def test_local_openai_compatible_assistant_runs_keyless():
    registry = build_default_registry()
    service = ServiceConfig(
        service="assistant",
        provider_id="local-openai-compatible",
        model_override="llama-local",
    )
    resolved = registry.resolve_service(service)
    assert resolved.credential_refs == {}
    assert (
        resolved.endpoint == registry.get(ASSISTANT, "local-openai-compatible").recommended_endpoint
    )
    assert resolved.endpoint.startswith("http://localhost:")


@pytest.mark.parametrize("provider_id", ["gemini", "azure-openai", "assemblyai-llm-gateway"])
def test_secrets_in_options_are_rejected_without_echo(provider_id):
    registry = build_default_registry()
    for key in ("api_key", "apiKey", "token"):
        with pytest.raises(ProviderRegistryError) as excinfo:
            registry.validate_options(ASSISTANT, provider_id, {key: SECRET})
        assert excinfo.value.code is RegistryErrorCode.SECRET_IN_OPTIONS
        _assert_no_secret(str(excinfo.value), excinfo.value.field_key)
