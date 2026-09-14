"""Protocol and lifecycle tests for the OpenAI Realtime STT adapter.

Nothing here reaches a real provider: most tests drive a scripted in-memory connection,
and the end-to-end cases use a local ``websockets`` server bound to 127.0.0.1.
"""

import asyncio
import base64
import json
import logging
import math
import struct
from collections import deque
from http import HTTPStatus
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import pytest

from engine.providers.stt.openai_protocol import (
    MalformedEvent,
    OpenAIRealtimeSettings,
    PCM16Resampler,
    ServerError,
    TranscriptAssembler,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    build_connection_url,
    build_session_update,
    map_http_status,
    map_server_error,
    parse_server_event,
    redact_endpoint,
)
from engine.providers.stt.openai_realtime import (
    OpenAIRealtimeSTTProvider,
    RealtimeConnectionClosedError,
    RealtimeHandshakeRejectedError,
    RealtimeTransportOptions,
)
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

API_KEY = "sk-test-DO-NOT-LEAK-0123456789"
PROMPT = "secret meeting context about Project Nightjar"
ENDPOINT = "wss://realtime.example.test/v1/realtime"
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"


# ── Fakes and helpers ────────────────────────────────────────────────────────


class _Close:
    def __init__(self, code: int) -> None:
        self.code = code


class FakeConnection:
    """Scripted Realtime socket: answers session.update and replays pushed frames."""

    def __init__(
        self,
        *,
        auto_ready: bool = True,
        expires_at: Optional[float] = None,
        handshake_error: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.auto_ready = auto_ready
        self.expires_at = expires_at
        self.handshake_error = handshake_error
        self.sent: List[Dict[str, Any]] = []
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.send_gate: Optional[asyncio.Event] = None

    async def send(self, message: str) -> None:
        if self.closed:
            raise RealtimeConnectionClosedError(1006)
        if self.send_gate is not None:
            await self.send_gate.wait()
        payload = json.loads(message)
        self.sent.append(payload)
        if payload["type"] == "session.update" and self.auto_ready:
            if self.handshake_error is not None:
                self.push({"type": "error", "error": self.handshake_error})
                return
            session = {"id": "sess_1", "type": "transcription"}
            if self.expires_at is not None:
                session["expires_at"] = self.expires_at
            self.push({"type": "session.created", "session": session})
            self.push({"type": "session.updated", "session": session})

    async def recv(self) -> str | bytes:
        item = await self.incoming.get()
        if isinstance(item, _Close):
            raise RealtimeConnectionClosedError(item.code)
        return item

    async def close(self) -> None:
        self.closed = True
        self.incoming.put_nowait(_Close(1000))

    def push(self, event: Dict[str, Any] | str | bytes) -> None:
        self.incoming.put_nowait(event if isinstance(event, (str, bytes)) else json.dumps(event))

    def drop(self, code: int = 1006) -> None:
        self.incoming.put_nowait(_Close(code))

    def sent_types(self) -> List[str]:
        return [message["type"] for message in self.sent]


class FakeConnector:
    """Connect factory handing out scripted connections or failures in order."""

    def __init__(self, *plan: Any) -> None:
        self.plan: Deque[Any] = deque(plan)
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.opened: List[FakeConnection] = []

    async def __call__(self, url: str, headers: Dict[str, str], timeout: float) -> FakeConnection:
        self.calls.append((url, dict(headers)))
        item = self.plan.popleft() if self.plan else FakeConnection()
        if item == "hang":
            await asyncio.Event().wait()
        if isinstance(item, BaseException):
            raise item
        self.opened.append(item)
        return item


class TickClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def make_settings(**overrides: Any) -> OpenAIRealtimeSettings:
    values: Dict[str, Any] = {
        "endpoint": ENDPOINT,
        "model": "gpt-live-transcribe",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return OpenAIRealtimeSettings(**values)


def make_transport(**overrides: Any) -> RealtimeTransportOptions:
    values: Dict[str, Any] = {
        "handshake_timeout_s": 1.0,
        "close_timeout_s": 0.5,
        "reconnect_backoff_s": 0.0,
        "rotation_drain_s": 0.0,
    }
    values.update(overrides)
    return RealtimeTransportOptions(**values)


def make_provider(connector: FakeConnector, **kwargs: Any) -> OpenAIRealtimeSTTProvider:
    settings = kwargs.pop("settings", make_settings())
    transport = kwargs.pop("transport", make_transport())
    return OpenAIRealtimeSTTProvider(
        settings, transport=transport, connect_factory=connector, **kwargs
    )


def make_frame(stream_id: str = "default", seq: int = 0, samples: int = 320, rate: int = 16000):
    return AudioFrame(stream_id, seq, struct.pack("<h", seq) * samples, sample_rate=rate)


def delta(item_id: str, text: str, event_id: Optional[str] = None) -> Dict[str, Any]:
    event: Dict[str, Any] = {"type": DELTA, "item_id": item_id, "content_index": 0, "delta": text}
    if event_id:
        event["event_id"] = event_id
    return event


def completed(item_id: str, text: str, event_id: Optional[str] = None) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "type": COMPLETED,
        "item_id": item_id,
        "content_index": 0,
        "transcript": text,
    }
    if event_id:
        event["event_id"] = event_id
    return event


def server_error(code: str, error_type: str = "invalid_request_error") -> Dict[str, Any]:
    return {
        "type": "error",
        "event_id": "evt_err",
        "error": {
            "type": error_type,
            "code": code,
            "message": f"Incorrect API key provided: {API_KEY}. Context: {PROMPT}",
            "param": None,
        },
    }


async def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


async def next_event(events: Any, timeout: float = 2.0) -> STTTranscriptResult:
    return await asyncio.wait_for(events.__anext__(), timeout)


def live_adapter_tasks() -> List["asyncio.Task[Any]"]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("openai-stt") and not task.done()
    ]


def _decoded_appends(conn: FakeConnection) -> List[bytes]:
    return [
        base64.b64decode(message["audio"])
        for message in conn.sent
        if message["type"] == "input_audio_buffer.append"
    ]


# ── Protocol: requests and endpoints ─────────────────────────────────────────


def test_session_update_for_live_model_carries_languages_and_hints():
    settings = make_settings(
        languages=("en", "fr"), prompt=PROMPT, keywords=("AC-42",), delay="low"
    )
    assert build_session_update(settings) == {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": "gpt-live-transcribe",
                        "languages": ["en", "fr"],
                        "prompt": PROMPT,
                        "keywords": ["AC-42"],
                        "delay": "low",
                    },
                    "turn_detection": None,
                }
            },
        },
    }


def test_session_update_for_turn_model_uses_single_language_and_omits_absent_hints():
    message = build_session_update(make_settings(model="gpt-transcribe", languages=("vi",)))
    transcription = message["session"]["audio"]["input"]["transcription"]
    assert transcription == {"model": "gpt-transcribe", "language": "vi"}


def test_settings_repr_hides_key_and_context():
    text = repr(make_settings(prompt=PROMPT, keywords=("Nightjar",)))
    assert API_KEY not in text
    assert PROMPT not in text
    assert "Nightjar" not in text


def test_connection_url_adds_transcription_intent():
    assert build_connection_url(ENDPOINT) == f"{ENDPOINT}?intent=transcription"
    assert build_connection_url("ws://127.0.0.1:9000/rt?intent=transcription") == (
        "ws://127.0.0.1:9000/rt?intent=transcription"
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "https://api.example.test/v1/realtime",
        "ws://api.example.test/v1/realtime",
        f"wss://user:{API_KEY}@api.example.test/v1/realtime",
        "wss:///v1/realtime",
    ],
)
def test_connection_url_rejects_unsafe_endpoints(endpoint):
    with pytest.raises(ProviderRuntimeError) as exc_info:
        build_connection_url(endpoint)
    assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST
    assert API_KEY not in str(exc_info.value)


def test_redact_endpoint_drops_query_and_credentials():
    assert redact_endpoint(f"{ENDPOINT}?intent=transcription&key=abc") == ENDPOINT


def test_resampler_is_chunk_invariant_and_scales_to_24k():
    samples = [int(12000 * math.sin(i / 7.0)) for i in range(1600)]
    pcm = struct.pack(f"<{len(samples)}h", *samples)

    whole = PCM16Resampler(16000).process(pcm)
    chunked_resampler = PCM16Resampler(16000)
    chunked = b"".join(chunked_resampler.process(pcm[i : i + 640]) for i in range(0, len(pcm), 640))

    assert len(whole) == len(chunked)
    assert abs(len(whole) // 2 - 2400) <= 1
    a = struct.unpack(f"<{len(whole) // 2}h", whole)
    b = struct.unpack(f"<{len(chunked) // 2}h", chunked)
    assert max(abs(x - y) for x, y in zip(a, b, strict=True)) <= 1
    assert PCM16Resampler(24000).process(pcm) == pcm


# ── Protocol: event parsing and error mapping ────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        b"\xff\xfe",
        "[1, 2]",
        json.dumps({"item_id": "x"}),
        json.dumps({"type": DELTA, "delta": "hi"}),
        json.dumps({"type": DELTA, "item_id": "x", "delta": 5}),
        json.dumps({"type": DELTA, "item_id": "x", "delta": "a", "content_index": -1}),
        json.dumps({"type": COMPLETED, "item_id": "x"}),
        json.dumps({"type": "input_audio_buffer.speech_started", "item_id": "x"}),
        json.dumps({"type": "error", "error": "boom"}),
        json.dumps({"type": "session.updated"}),
    ],
)
def test_malformed_events_never_raise(raw):
    assert isinstance(parse_server_event(raw), MalformedEvent)


def test_unknown_event_types_are_ignored_not_malformed():
    event = parse_server_event(json.dumps({"type": "rate_limits.updated", "event_id": "e1"}))
    assert type(event).__name__ == "IgnoredEvent"


@pytest.mark.parametrize(
    ("code", "error_type", "expected", "retryable"),
    [
        ("invalid_api_key", "invalid_request_error", ProviderErrorCode.AUTH_FAILED, False),
        ("rate_limit_exceeded", "rate_limit_error", ProviderErrorCode.RATE_LIMITED, True),
        ("insufficient_quota", "invalid_request_error", ProviderErrorCode.QUOTA_EXCEEDED, False),
        ("model_not_found", "invalid_request_error", ProviderErrorCode.MODEL_UNAVAILABLE, False),
        ("session_expired", "invalid_request_error", ProviderErrorCode.CONNECTION_CLOSED, True),
        ("server_error", "server_error", ProviderErrorCode.INTERNAL_ERROR, True),
        ("something_new", "invalid_request_error", ProviderErrorCode.INVALID_REQUEST, False),
    ],
)
def test_server_errors_map_to_redacted_canonical_errors(code, error_type, expected, retryable):
    event = parse_server_event(json.dumps(server_error(code, error_type)))
    assert isinstance(event, ServerError)
    error = map_server_error(event)
    assert error.code == expected
    assert error.retryable is retryable
    assert error.capability == ProviderCapability.STT
    assert error.details == {"provider_code": code}
    rendered = f"{error} {error.details}"
    assert API_KEY not in rendered
    assert PROMPT not in rendered


def test_unsafe_provider_codes_are_not_copied():
    error = map_server_error(ServerError(error_type=None, code=f"bad code {API_KEY} with spaces"))
    assert error.details is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderErrorCode.AUTH_FAILED),
        (403, ProviderErrorCode.AUTH_FAILED),
        (429, ProviderErrorCode.RATE_LIMITED),
        (404, ProviderErrorCode.INVALID_REQUEST),
        (503, ProviderErrorCode.NETWORK_UNAVAILABLE),
    ],
)
def test_http_rejections_map_to_canonical_codes(status, expected):
    error = map_http_status(status)
    assert error.code == expected
    assert error.details == {"http_status": status}


# ── Protocol: transcript assembly ────────────────────────────────────────────


def test_assembler_partial_then_final_is_deterministic():
    def run() -> List[STTTranscriptResult]:
        assembler = TranscriptAssembler("mic", clock=TickClock())
        assembler.advance_audio(500.0)
        out = [assembler.on_delta(TranscriptDelta("item_1", "Hello"))]
        assembler.advance_audio(250.0)
        out.append(assembler.on_delta(TranscriptDelta("item_1", ", world")))
        assembler.mark_commit()
        assembler.on_committed("item_1")
        out.append(assembler.on_completed(TranscriptCompleted("item_1", "Hello, world.")))
        return [result for result in out if result is not None]

    first, second = run(), run()
    assert first == second
    assert first == [
        STTTranscriptResult("mic", 0, "item_1", "Hello", False, 1, 0.0, 500.0, 101.0),
        STTTranscriptResult("mic", 0, "item_1", "Hello, world", False, 2, 0.0, 750.0, 102.0),
        STTTranscriptResult(
            "mic", 0, "item_1", "Hello, world.", True, 3, 0.0, 750.0, 103.0, is_endpoint=True
        ),
    ]


def test_assembler_duplicate_and_late_events_never_regress_revision():
    assembler = TranscriptAssembler("mic", clock=TickClock())
    assembler.advance_audio(100.0)

    first = assembler.on_delta(TranscriptDelta("item_1", "one", event_id="e1"))
    assert first is not None and first.revision == 1
    assert assembler.on_delta(TranscriptDelta("item_1", "one", event_id="e1")) is None
    final = assembler.on_completed(TranscriptCompleted("item_1", "one two", event_id="e2"))
    assert final is not None and final.revision == 2 and final.is_final

    # Late delta, replayed completion, and a fresh-id duplicate completion are all inert.
    assert assembler.on_delta(TranscriptDelta("item_1", " three", event_id="e3")) is None
    assert assembler.on_completed(TranscriptCompleted("item_1", "one two", event_id="e2")) is None
    assert assembler.on_completed(TranscriptCompleted("item_1", "other", event_id="e4")) is None
    assert assembler.on_failed(TranscriptFailed("item_1", "x", event_id="e5")) is None


def test_assembler_out_of_order_items_keep_sequence_and_commit_intervals():
    assembler = TranscriptAssembler("system", clock=TickClock())
    assembler.advance_audio(1000.0)
    assembler.mark_commit()  # item_a covers 0-1000
    assembler.advance_audio(600.0)
    assembler.mark_commit()  # item_b covers 1000-1600

    assembler.on_committed("item_a")
    assembler.on_committed("item_b")
    # Provider completes the later item first.
    b_final = assembler.on_completed(TranscriptCompleted("item_b", "second"))
    a_partial = assembler.on_delta(TranscriptDelta("item_a", "fir"))
    a_final = assembler.on_completed(TranscriptCompleted("item_a", "first"))

    assert (a_partial.sequence, a_partial.revision) == (0, 1)
    assert (a_final.sequence, a_final.revision, a_final.is_final) == (0, 2, True)
    assert (a_final.start_time_ms, a_final.end_time_ms) == (0.0, 1000.0)
    assert (b_final.sequence, b_final.revision) == (1, 1)
    assert (b_final.start_time_ms, b_final.end_time_ms) == (1000.0, 1600.0)


def test_assembler_reset_discards_in_flight_items_until_clear_acknowledged():
    assembler = TranscriptAssembler("mic", clock=TickClock())
    assembler.advance_audio(100.0)
    assert assembler.on_delta(TranscriptDelta("old", "stale")) is not None

    assembler.reset()
    assert assembler.on_delta(TranscriptDelta("old", " more")) is None
    assert assembler.on_delta(TranscriptDelta("ghost", "committed before reset")) is None
    assembler.on_cleared()
    assert assembler.on_delta(TranscriptDelta("ghost", "late")) is None

    fresh = assembler.on_delta(TranscriptDelta("new", "fresh"))
    assert fresh is not None and (fresh.sequence, fresh.revision) == (1, 1)


def test_assembler_failed_item_keeps_shown_text_and_flush_finalizes():
    assembler = TranscriptAssembler("mic", clock=TickClock())
    assembler.advance_audio(100.0)
    assembler.on_delta(TranscriptDelta("shown", "visible"))
    kept = assembler.on_failed(TranscriptFailed("shown", "audio_unintelligible"))
    assert kept is not None and kept.is_final and kept.text == "visible"
    assert assembler.on_failed(TranscriptFailed("silent")) is None

    assembler.on_delta(TranscriptDelta("inflight", "half"))
    flushed = assembler.flush_pending()
    assert [(r.provider_event_id, r.is_final, r.revision) for r in flushed] == [
        ("inflight", True, 2)
    ]


# ── Adapter: handshake and streaming ─────────────────────────────────────────


async def test_invalid_settings_fail_before_any_connection():
    connector = FakeConnector()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        make_provider(connector, settings=make_settings(api_key=""))
    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    with pytest.raises(ProviderRuntimeError):
        make_provider(connector, settings=make_settings(delay="instant"))
    assert connector.calls == []


async def test_start_authenticates_and_configures_transcription_session():
    connector = FakeConnector()
    provider = make_provider(connector, settings=make_settings(languages=("en",)))
    assert provider.capability == ProviderCapability.STT

    await provider.start()
    try:
        url, headers = connector.calls[0]
        assert url == f"{ENDPOINT}?intent=transcription"
        assert headers == {"Authorization": f"Bearer {API_KEY}"}
        conn = connector.opened[0]
        assert conn.sent[0]["type"] == "session.update"
        assert conn.sent[0]["session"]["type"] == "transcription"
    finally:
        await provider.stop()


async def test_audio_commit_and_transcript_events_are_mapped():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic",), clock=TickClock())
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame("mic", samples=320))
        await provider.finalize_stream("mic")
        await wait_until(lambda: "input_audio_buffer.commit" in conn.sent_types())

        assert conn.sent_types() == [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        appended = _decoded_appends(conn)[0]
        assert len(appended) % 2 == 0 and abs(len(appended) // 2 - 480) <= 1

        conn.push(delta("item_1", "Hel", "e1"))
        conn.push({"type": "input_audio_buffer.committed", "item_id": "item_1"})
        conn.push(completed("item_1", "Hello", "e2"))

        partial = await next_event(events)
        final = await next_event(events)
        assert (partial.stream_id, partial.text, partial.is_final, partial.revision) == (
            "mic",
            "Hel",
            False,
            1,
        )
        assert (final.text, final.is_final, final.revision, final.is_endpoint) == (
            "Hello",
            True,
            2,
            True,
        )
        assert final.provider_event_id == "item_1" and final.sequence == 0
        assert (final.start_time_ms, final.end_time_ms) == (0.0, 20.0)
    finally:
        await provider.stop()


async def test_finalize_without_new_audio_sends_no_commit():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    try:
        await provider.finalize_stream("default")
        await asyncio.sleep(0.02)
        assert connector.opened[0].sent_types() == ["session.update"]
    finally:
        await provider.stop()


async def test_send_audio_never_blocks_and_drops_oldest_audio_but_keeps_commit():
    connector = FakeConnector()
    provider = make_provider(connector, transport=make_transport(max_queued_frames=3))
    await provider.start()
    conn = connector.opened[0]
    conn.send_gate = asyncio.Event()
    try:
        for seq in range(10):
            await asyncio.wait_for(
                provider.send_audio(make_frame(seq=seq, samples=240, rate=24000)), 0.05
            )
        await provider.finalize_stream("default")

        conn.send_gate.set()
        await wait_until(lambda: "input_audio_buffer.commit" in conn.sent_types())
        seqs = [struct.unpack("<h", pcm[:2])[0] for pcm in _decoded_appends(conn)]
        assert len(seqs) <= 4
        assert seqs[-3:] == [7, 8, 9]
        assert conn.sent_types()[-1] == "input_audio_buffer.commit"
    finally:
        await provider.stop()


async def test_reset_stream_clears_provider_buffer_and_ignores_stale_items():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame())
        conn.push(delta("item_r", "stale"))
        assert (await next_event(events)).text == "stale"

        await provider.reset_stream("default")
        await wait_until(lambda: "input_audio_buffer.clear" in conn.sent_types())
        conn.push(delta("item_r", " more"))
        conn.push(delta("item_ghost", "ghost"))
        conn.push({"type": "input_audio_buffer.cleared"})
        conn.push(delta("item_z", "fresh"))

        fresh = await next_event(events)
        assert (fresh.text, fresh.sequence, fresh.revision) == ("fresh", 1, 1)
    finally:
        await provider.stop()


async def test_multiple_streams_use_isolated_sessions():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic", "system"))
    await provider.start()
    mic, system = connector.opened
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame("system"))
        await wait_until(lambda: "input_audio_buffer.append" in system.sent_types())
        assert "input_audio_buffer.append" not in mic.sent_types()

        system.push(delta("item_s", "from system"))
        result = await next_event(events)
        assert (result.stream_id, result.sequence) == ("system", 0)

        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.send_audio(make_frame("speaker"))
        assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST
    finally:
        await provider.stop()


# ── Adapter: errors ──────────────────────────────────────────────────────────


async def test_handshake_server_error_is_redacted_and_closes_socket():
    conn = FakeConnection(handshake_error=server_error("invalid_api_key")["error"])
    provider = make_provider(FakeConnector(conn))

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert API_KEY not in str(exc_info.value)
    assert PROMPT not in repr(exc_info.value.error)
    assert conn.closed
    assert live_adapter_tasks() == []


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [(401, ProviderErrorCode.AUTH_FAILED, False), (429, ProviderErrorCode.RATE_LIMITED, True)],
)
async def test_handshake_http_rejection_maps_status(status, expected, retryable):
    provider = make_provider(FakeConnector(RealtimeHandshakeRejectedError(status)))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == expected
    assert exc_info.value.error.retryable is retryable


async def test_handshake_timeout_and_network_failure():
    provider = make_provider(
        FakeConnector(FakeConnection(auto_ready=False)),
        transport=make_transport(handshake_timeout_s=0.05),
    )
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == ProviderErrorCode.TIMEOUT

    provider = make_provider(FakeConnector(OSError(f"cannot reach {ENDPOINT}?key={API_KEY}")))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == ProviderErrorCode.NETWORK_UNAVAILABLE
    assert API_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


async def test_rate_limit_during_stream_surfaces_retryable_redacted_error():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    try:
        conn.push(server_error("rate_limit_exceeded", "rate_limit_error"))
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(provider.receive_events())
        error = exc_info.value.error
        assert (error.code, error.retryable) == (ProviderErrorCode.RATE_LIMITED, True)
        assert API_KEY not in str(exc_info.value) and PROMPT not in str(error.details)

        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.send_audio(make_frame())
        assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED
    finally:
        await provider.stop()


async def test_malformed_and_non_fatal_error_events_do_not_break_stream():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        conn.push("{not json")
        conn.push({"type": DELTA, "delta": "orphan"})
        conn.push({"type": "rate_limits.updated", "rate_limits": []})
        conn.push(server_error("input_audio_buffer_commit_empty"))
        conn.push(server_error("unknown_parameter"))
        conn.push(delta("item_ok", "still streaming"))

        result = await next_event(events)
        assert result.text == "still streaming"
    finally:
        await provider.stop()


# ── Adapter: close, expiry, and rotation ─────────────────────────────────────


async def test_unexpected_close_finalizes_in_flight_item_and_reconnects():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    first = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame())
        first.push(delta("item_a", "partial words"))
        assert (await next_event(events)).revision == 1

        first.drop(1006)
        flushed = await next_event(events)
        assert (flushed.provider_event_id, flushed.is_final, flushed.revision) == (
            "item_a",
            True,
            2,
        )

        # The connection is handed out before the handshake; pushing earlier would race it.
        await wait_until(
            lambda: len(connector.opened) == 2 and bool(connector.opened[1].sent_types())
        )
        second = connector.opened[1]
        assert second.sent_types()[0] == "session.update"
        assert first.closed

        second.push(delta("item_b", "after reconnect"))
        resumed = await next_event(events)
        assert (resumed.sequence, resumed.revision) == (1, 1)

        await provider.send_audio(make_frame(seq=1))
        await wait_until(lambda: "input_audio_buffer.append" in second.sent_types())
    finally:
        await provider.stop()


async def test_session_expired_error_rotates_to_a_new_session():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    first = connector.opened[0]
    try:
        first.push(server_error("session_expired"))
        await wait_until(lambda: len(connector.opened) == 2 and first.closed)
        second = connector.opened[1]
        await wait_until(lambda: second.sent_types() == ["session.update"])
        assert not second.closed
    finally:
        await provider.stop()


async def test_reconnect_exhaustion_reports_connection_closed():
    connector = FakeConnector(
        FakeConnection(), RealtimeHandshakeRejectedError(503), RealtimeHandshakeRejectedError(503)
    )
    provider = make_provider(connector, transport=make_transport(max_reconnect_attempts=2))
    await provider.start()
    try:
        connector.opened[0].drop(1011)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(provider.receive_events())
        assert exc_info.value.error.code == ProviderErrorCode.CONNECTION_CLOSED
        assert exc_info.value.error.retryable is True
        assert len(connector.calls) == 3
    finally:
        await provider.stop()


async def test_non_retryable_reconnect_failure_stops_retrying():
    connector = FakeConnector(FakeConnection(), RealtimeHandshakeRejectedError(401))
    provider = make_provider(connector, transport=make_transport(max_reconnect_attempts=5))
    await provider.start()
    try:
        connector.opened[0].drop(1006)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(provider.receive_events())
        assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
        assert len(connector.calls) == 2
    finally:
        await provider.stop()


async def test_proactive_rotation_near_expiry_is_make_before_break():
    first = FakeConnection(expires_at=1050.0)
    connector = FakeConnector(first)
    provider = make_provider(
        connector,
        transport=make_transport(rotation_margin_s=120.0, rotation_drain_s=0.2),
        wall_clock=lambda: 1000.0,
    )
    await provider.start()
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame())
        first.push(delta("item_x", "before rotation"))
        assert (await next_event(events)).is_final is False

        await provider.finalize_stream("default")
        await wait_until(lambda: len(connector.opened) == 2)
        second = connector.opened[1]
        await wait_until(lambda: second.sent_types() == ["session.update"])
        assert not first.closed

        # The retiring socket still delivers the completion for audio it received.
        first.push(completed("item_x", "before rotation."))
        final = await next_event(events)
        assert (final.provider_event_id, final.is_final, final.revision) == ("item_x", True, 2)

        await provider.send_audio(make_frame(seq=1))
        await wait_until(lambda: "input_audio_buffer.append" in second.sent_types())
        await wait_until(lambda: first.closed)
        assert first.sent_types().count("input_audio_buffer.append") == 1
        assert not second.closed
    finally:
        await provider.stop()


# ── Adapter: cancellation and cleanup ────────────────────────────────────────


async def test_stop_closes_sockets_cancels_tasks_and_ends_iteration():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic", "system"))
    await provider.start()
    received: List[STTTranscriptResult] = []

    async def consume() -> None:
        async for result in provider.receive_events():
            received.append(result)

    consumer = asyncio.ensure_future(consume())
    connector.opened[0].push(delta("item_1", "hi"))
    await wait_until(lambda: len(received) == 1)
    assert live_adapter_tasks()

    await provider.stop()
    await asyncio.wait_for(consumer, 1.0)

    assert all(conn.closed for conn in connector.opened)
    assert live_adapter_tasks() == []
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(make_frame("mic"))
    assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST


async def test_cancelled_consumer_does_not_leak_after_stop():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()

    async def consume() -> None:
        async for _ in provider.receive_events():
            pass

    consumer = asyncio.ensure_future(consume())
    await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    await provider.stop()
    assert connector.opened[0].closed
    assert live_adapter_tasks() == []


async def test_cancelled_start_closes_socket_and_leaks_no_task():
    conn = FakeConnection(auto_ready=False)
    provider = make_provider(FakeConnector(conn), transport=make_transport(handshake_timeout_s=5))

    start = asyncio.ensure_future(provider.start())
    await wait_until(lambda: conn.sent_types() == ["session.update"])
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert conn.closed
    assert live_adapter_tasks() == []


async def test_cancelled_start_during_second_stream_closes_first_session():
    first = FakeConnection()
    connector = FakeConnector(first, "hang")
    provider = make_provider(connector, stream_ids=("mic", "system"))

    start = asyncio.ensure_future(provider.start())
    await wait_until(lambda: len(connector.calls) == 2)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert first.closed
    assert live_adapter_tasks() == []


async def test_health_check_reports_status_without_leaking_secrets():
    healthy_conn = FakeConnection()
    provider = make_provider(FakeConnector(healthy_conn))
    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.HEALTHY
    assert healthy_conn.closed

    provider = make_provider(FakeConnector(RealtimeHandshakeRejectedError(401)))
    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.UNHEALTHY
    assert health.details == {"code": "auth_failed", "retryable": False}
    assert API_KEY not in health.message


async def test_logs_never_contain_secret_url_audio_or_transcript(caplog):
    caplog.set_level(logging.DEBUG, logger="echoflux")
    first = FakeConnection()
    connector = FakeConnector(first)
    provider = make_provider(connector, settings=make_settings(prompt=PROMPT))
    await provider.start()
    events = provider.receive_events()
    frame = make_frame(seq=1234)
    try:
        await provider.send_audio(frame)
        first.push(delta("item_1", "confidential transcript words"))
        await next_event(events)
        first.push("{garbage")
        first.drop(1006)
        await next_event(events)
        await wait_until(
            lambda: len(connector.opened) == 2 and bool(connector.opened[1].sent_types())
        )
        connector.opened[1].push(server_error("rate_limit_exceeded", "rate_limit_error"))
        with pytest.raises(ProviderRuntimeError):
            await next_event(events)
    finally:
        await provider.stop()

    text = caplog.text
    assert caplog.records
    for forbidden in (
        API_KEY,
        PROMPT,
        "confidential transcript words",
        "intent=transcription",
        "Bearer",
        base64.b64encode(frame.pcm_data[:60]).decode("ascii"),
    ):
        assert forbidden not in text


# ── Adapter against a local WebSocket server ─────────────────────────────────


@pytest.fixture
def no_proxy_env(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


async def test_local_websocket_server_end_to_end(no_proxy_env):
    from websockets.asyncio.server import serve

    received: List[Dict[str, Any]] = []
    request: Dict[str, Any] = {}
    server_closed = asyncio.Event()

    async def handler(ws: Any) -> None:
        request["authorization"] = ws.request.headers.get("Authorization")
        request["path"] = ws.request.path
        try:
            async for message in ws:
                payload = json.loads(message)
                received.append(payload)
                kind = payload["type"]
                if kind == "session.update":
                    session = {"id": "sess_local", "expires_at": 4102444800}
                    await ws.send(json.dumps({"type": "session.created", "session": session}))
                    await ws.send(json.dumps({"type": "session.updated", "session": session}))
                elif kind == "input_audio_buffer.append":
                    await ws.send(json.dumps(delta("item_1", "local", "e1")))
                elif kind == "input_audio_buffer.commit":
                    await ws.send(
                        json.dumps({"type": "input_audio_buffer.committed", "item_id": "item_1"})
                    )
                    await ws.send(json.dumps(completed("item_1", "local server", "e2")))
        finally:
            server_closed.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        provider = OpenAIRealtimeSTTProvider(
            make_settings(endpoint=f"ws://127.0.0.1:{port}/v1/realtime"),
            transport=make_transport(),
        )
        await provider.start()
        events = provider.receive_events()
        try:
            await provider.send_audio(make_frame())
            partial = await next_event(events)
            await provider.finalize_stream("default")
            final = await next_event(events)
        finally:
            await provider.stop()
        await asyncio.wait_for(server_closed.wait(), 2.0)

    assert request == {
        "authorization": f"Bearer {API_KEY}",
        "path": "/v1/realtime?intent=transcription",
    }
    assert [message["type"] for message in received] == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
    ]
    assert (partial.text, partial.is_final) == ("local", False)
    assert (final.text, final.is_final, final.revision) == ("local server", True, 2)
    assert live_adapter_tasks() == []


async def test_local_websocket_server_http_rejection_maps_to_auth_failed(no_proxy_env):
    from websockets.asyncio.server import serve

    def reject(connection: Any, request: Any) -> Any:
        return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")

    async def handler(ws: Any) -> None:  # pragma: no cover - never reached
        await ws.close()

    async with serve(handler, "127.0.0.1", 0, process_request=reject) as server:
        port = server.sockets[0].getsockname()[1]
        provider = OpenAIRealtimeSTTProvider(
            make_settings(endpoint=f"ws://127.0.0.1:{port}/v1/realtime"),
            transport=make_transport(),
        )
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert exc_info.value.error.details == {"http_status": 401}
    assert live_adapter_tasks() == []
