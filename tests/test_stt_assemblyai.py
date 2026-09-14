"""Protocol and lifecycle tests for the AssemblyAI Universal Streaming STT adapter.

Nothing here reaches a real provider: most tests drive a scripted in-memory connection,
and the end-to-end cases use a local ``websockets`` server bound to 127.0.0.1.
"""

import asyncio
import json
import logging
import struct
from collections import deque
from http import HTTPStatus
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import pytest

from engine.providers.stt.assemblyai import (
    AssemblyAIMalformed,
    AssemblyAIServerError,
    AssemblyAISettings,
    AssemblyAISTTProvider,
    AssemblyAITransportOptions,
    AssemblyAITurn,
    AssemblyAITurnAssembler,
    build_auth_headers,
    build_connection_url,
    map_close,
    parse_message,
)
from engine.providers.stt.streaming_ws import HandshakeRejectedError, SocketClosedError
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

API_KEY = "aai-test-DO-NOT-LEAK-0123456789abcdef"
KEYTERM = "Nightjar"
ENDPOINT = "wss://streaming.example.test/v3/ws"
TRANSCRIPT_SECRET = "confidential transcript words"
SESSION_ID = "sess-123"


# ── Fakes and helpers ────────────────────────────────────────────────────────


class _Close:
    def __init__(self, code: int, reason: str = "") -> None:
        self.code = code
        self.reason = reason


class FakeConnection:
    """Scripted v3 socket: sends ``Begin`` on open and confirms ``Terminate``."""

    def __init__(self, *, begin: bool = True, confirm_terminate: bool = True) -> None:
        self.sent: List[str | bytes] = []
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.send_gate: Optional[asyncio.Event] = None
        self.confirm_terminate = confirm_terminate
        self.on_text: Optional[Callable[[Dict[str, Any]], None]] = None
        if begin:
            self.push({"type": "Begin", "id": SESSION_ID, "expires_at": 4102444800})

    async def send(self, message: str | bytes) -> None:
        if self.closed:
            raise SocketClosedError(1006)
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.sent.append(message)
        if not isinstance(message, str):
            return
        payload = json.loads(message)
        if self.on_text is not None:
            self.on_text(payload)
        if payload["type"] == "Terminate" and self.confirm_terminate:
            self.push(
                {"type": "Termination", "audio_duration_seconds": 1, "session_duration_seconds": 2}
            )
            self.drop(1000)

    async def recv(self) -> str | bytes:
        item = await self.incoming.get()
        if isinstance(item, _Close):
            self.closed = True
            raise SocketClosedError(item.code, item.reason)
        return item

    async def close(self) -> None:
        self.closed = True
        self.incoming.put_nowait(_Close(1000))

    def push(self, event: Dict[str, Any] | str | bytes) -> None:
        self.incoming.put_nowait(event if isinstance(event, (str, bytes)) else json.dumps(event))

    def drop(self, code: int, reason: str = "") -> None:
        self.incoming.put_nowait(_Close(code, reason))

    def text_types(self) -> List[str]:
        return [json.loads(m)["type"] for m in self.sent if isinstance(m, str)]

    def audio(self) -> List[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


class FakeConnector:
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


def make_settings(**overrides: Any) -> AssemblyAISettings:
    values: Dict[str, Any] = {
        "endpoint": ENDPOINT,
        "speech_model": "universal-3-5-pro",
        "api_key": API_KEY,
        "sample_rate": 16000,
    }
    values.update(overrides)
    return AssemblyAISettings(**values)


def make_transport(**overrides: Any) -> AssemblyAITransportOptions:
    values: Dict[str, Any] = {
        "handshake_timeout_s": 1.0,
        "close_timeout_s": 0.5,
        "terminate_timeout_s": 1.0,
    }
    values.update(overrides)
    return AssemblyAITransportOptions(**values)


def make_provider(connector: FakeConnector, **kwargs: Any) -> AssemblyAISTTProvider:
    settings = kwargs.pop("settings", make_settings())
    transport = kwargs.pop("transport", make_transport())
    return AssemblyAISTTProvider(settings, transport=transport, connect_factory=connector, **kwargs)


def make_frame(stream_id: str = "default", seq: int = 0, ms: int = 20) -> AudioFrame:
    samples = 16 * ms
    return AudioFrame(stream_id, seq, struct.pack("<h", seq + 1) * samples, sample_rate=16000)


def turn(
    order: int,
    text: str,
    *,
    end: bool = False,
    formatted: bool = False,
    start_ms: int = 0,
    end_ms: Optional[int] = None,
) -> Dict[str, Any]:
    words = []
    if text:
        stop = end_ms if end_ms is not None else start_ms + 100 * len(text.split())
        words = [
            {"text": w, "start": start_ms, "end": stop, "confidence": 0.9} for w in text.split()
        ]
    return {
        "type": "Turn",
        "turn_order": order,
        "turn_is_formatted": formatted,
        "end_of_turn": end,
        "transcript": text,
        "end_of_turn_confidence": 0.5,
        "language_code": "en",
        "words": words,
    }


def parsed(event: Dict[str, Any]) -> AssemblyAITurn:
    result = parse_message(json.dumps(event))
    assert isinstance(result, AssemblyAITurn)
    return result


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
        if task.get_name().startswith("assemblyai-stt") and not task.done()
    ]


# ── Protocol: settings, URL, auth ────────────────────────────────────────────


def test_connection_url_carries_resolved_settings_and_omits_unset_options():
    url = build_connection_url(make_settings())
    assert url == f"{ENDPOINT}?sample_rate=16000&encoding=pcm_s16le&speech_model=universal-3-5-pro"
    assert build_auth_headers(make_settings()) == {"Authorization": API_KEY}

    full = build_connection_url(
        make_settings(
            format_turns=True,
            language_detection=True,
            language_codes=("en", "es"),
            keyterms_prompt=("AC-42", KEYTERM),
            end_of_turn_confidence_threshold=0.6,
            min_turn_silence_ms=200,
            max_turn_silence_ms=1500,
            inactivity_timeout_s=30,
        )
    )
    query = parse_qs(urlsplit(full).query)
    assert query["format_turns"] == ["true"]
    assert query["language_detection"] == ["true"]
    assert json.loads(query["language_codes"][0]) == ["en", "es"]
    assert json.loads(query["keyterms_prompt"][0]) == ["AC-42", KEYTERM]
    assert query["end_of_turn_confidence_threshold"] == ["0.6"]
    assert (query["min_turn_silence"], query["max_turn_silence"]) == (["200"], ["1500"])
    assert query["inactivity_timeout"] == ["30"]
    assert "token" not in query


def test_temporary_token_travels_in_query_not_header():
    settings = make_settings(auth_mode="temporary_token")
    assert build_auth_headers(settings) == {}
    assert parse_qs(urlsplit(build_connection_url(settings)).query)["token"] == [API_KEY]


def test_settings_repr_hides_key_and_keyterms():
    text = repr(make_settings(keyterms_prompt=(KEYTERM,)))
    assert API_KEY not in text and KEYTERM not in text


@pytest.mark.parametrize(
    ("settings_overrides", "transport_overrides", "code"),
    [
        ({"speech_model": ""}, {}, ProviderErrorCode.INVALID_REQUEST),
        ({"api_key": ""}, {}, ProviderErrorCode.AUTH_FAILED),
        ({"sample_rate": 4000}, {}, ProviderErrorCode.INVALID_REQUEST),
        ({"inactivity_timeout_s": 2}, {}, ProviderErrorCode.INVALID_REQUEST),
        ({"end_of_turn_confidence_threshold": 1.5}, {}, ProviderErrorCode.INVALID_REQUEST),
        ({"auth_mode": "bearer"}, {}, ProviderErrorCode.INVALID_REQUEST),
        ({"endpoint": "ws://streaming.example.test/v3/ws"}, {}, ProviderErrorCode.INVALID_REQUEST),
        ({}, {"chunk_ms": 20.0}, ProviderErrorCode.INVALID_REQUEST),
        ({}, {"chunk_ms": 1500.0}, ProviderErrorCode.INVALID_REQUEST),
    ],
)
async def test_invalid_settings_fail_before_any_connection(
    settings_overrides, transport_overrides, code
):
    connector = FakeConnector()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        make_provider(
            connector,
            settings=make_settings(**settings_overrides),
            transport=make_transport(**transport_overrides),
        )
    assert exc_info.value.error.code == code
    assert API_KEY not in str(exc_info.value)
    assert connector.calls == []


# ── Protocol: parsing and error mapping ──────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        b"\xff",
        "[]",
        json.dumps({"turn_order": 1}),
        json.dumps({**turn(0, "x"), "turn_order": -1}),
        json.dumps({**turn(0, "x"), "transcript": None}),
        json.dumps({**turn(0, "x"), "end_of_turn": "true"}),
    ],
)
def test_malformed_messages_never_raise(raw):
    assert isinstance(parse_message(raw), AssemblyAIMalformed)


def test_error_payload_and_unknown_types():
    assert isinstance(parse_message(json.dumps({"error": f"bad {API_KEY}"})), AssemblyAIServerError)
    heartbeat = parse_message(json.dumps({"type": "Heartbeat", "total_audio_received_ms": 10}))
    assert type(heartbeat).__name__ == "AssemblyAIIgnored"


@pytest.mark.parametrize(
    ("code", "reason", "expected", "retryable"),
    [
        (1008, "Unauthorized Connection: Invalid API key", ProviderErrorCode.AUTH_FAILED, False),
        (
            1008,
            "Unauthorized Connection: insufficient balance",
            ProviderErrorCode.QUOTA_EXCEEDED,
            False,
        ),
        (3009, "Too many concurrent sessions", ProviderErrorCode.RATE_LIMITED, True),
        (3005, "Session Cancelled: An error occurred", ProviderErrorCode.INTERNAL_ERROR, True),
        (3006, "Session terminated due to inactivity", ProviderErrorCode.TIMEOUT, True),
        (3006, "Invalid Message Type", ProviderErrorCode.INVALID_REQUEST, False),
        (3007, "Input duration violation", ProviderErrorCode.INVALID_REQUEST, False),
        (3008, "Session Expired", ProviderErrorCode.CONNECTION_CLOSED, True),
        (1011, "Internal error", ProviderErrorCode.INTERNAL_ERROR, True),
    ],
)
def test_close_codes_map_to_redacted_canonical_errors(code, reason, expected, retryable):
    error = map_close(code, f"{reason} {API_KEY}")
    assert (error.code, error.retryable) == (expected, retryable)
    assert error.capability == ProviderCapability.STT
    assert error.details == {"close_code": code}
    assert API_KEY not in f"{error} {error.details}" and reason not in error.message


# ── Protocol: turn finality and duplicates ───────────────────────────────────


def test_partials_then_end_of_turn_final_without_formatting():
    assembler = AssemblyAITurnAssembler("mic", expects_formatting=False, clock=TickClock())
    assembler.set_session(SESSION_ID)
    out: List[STTTranscriptResult] = []
    for event in (
        turn(0, "hello"),
        turn(0, "hello"),
        turn(0, "hello there", end_ms=900),
        turn(0, "hello there", end=True, start_ms=100, end_ms=1000),
        turn(0, "hello there", end=True, formatted=True, end_ms=1000),
        turn(1, "next", start_ms=1200),
    ):
        out.extend(assembler.on_turn(parsed(event)))

    assert [(r.sequence, r.text, r.is_final, r.revision) for r in out] == [
        (0, "hello", False, 1),
        (0, "hello there", False, 2),
        (0, "hello there", True, 3),
        (1, "next", False, 1),
    ]
    final = out[2]
    assert final.provider_event_id == f"assemblyai:{SESSION_ID}:0"
    assert (final.start_time_ms, final.end_time_ms, final.is_endpoint) == (0.0, 1000.0, True)
    assert final.detected_language == "en"
    assert final.words is not None and len(final.words) == 2


def test_format_turns_waits_for_formatted_copy_and_ignores_replays():
    assembler = AssemblyAITurnAssembler("mic", expects_formatting=True, clock=TickClock())
    out: List[STTTranscriptResult] = []
    for event in (
        turn(0, "hello world"),
        turn(0, "hello world", end=True),
        turn(0, "Hello world.", end=True, formatted=True),
        turn(0, "Hello world.", end=True, formatted=True),
        turn(0, "hello world"),
    ):
        out.extend(assembler.on_turn(parsed(event)))
    assert [(r.text, r.is_final) for r in out] == [("hello world", False), ("Hello world.", True)]


def test_missing_final_is_promoted_when_next_turn_starts_and_flush_finalizes():
    assembler = AssemblyAITurnAssembler("mic", expects_formatting=True, clock=TickClock())
    assembler.on_turn(parsed(turn(0, "never formatted", end=True)))
    out = assembler.on_turn(parsed(turn(1, "second")))
    assert [(r.sequence, r.text, r.is_final) for r in out] == [
        (0, "never formatted", True),
        (1, "second", False),
    ]
    assert assembler.on_turn(parsed(turn(0, "Never formatted.", end=True, formatted=True))) == []
    flushed = assembler.flush()
    assert [(r.sequence, r.is_final, r.revision) for r in flushed] == [(1, True, 2)]
    assert assembler.flush() == []


def test_reset_abandons_open_turn_and_earlier_audio():
    assembler = AssemblyAITurnAssembler("mic", expects_formatting=False, clock=TickClock())
    assert assembler.on_turn(parsed(turn(0, "stale")))
    assembler.reset(discard_before_ms=2000.0)

    assert assembler.on_turn(parsed(turn(0, "stale words", end=True))) == []
    assert assembler.on_turn(parsed(turn(1, "pre reset", end=True, end_ms=1500))) == []
    fresh = assembler.on_turn(parsed(turn(2, "fresh", start_ms=2100, end_ms=2300)))
    assert [(r.sequence, r.text) for r in fresh] == [(2, "fresh")]


# ── Adapter: handshake and streaming ─────────────────────────────────────────


async def test_start_waits_for_begin_and_authenticates():
    connector = FakeConnector()
    provider = make_provider(connector)
    assert provider.capability == ProviderCapability.STT
    await provider.start()
    try:
        url, headers = connector.calls[0]
        assert url == build_connection_url(make_settings())
        assert headers == {"Authorization": API_KEY}
    finally:
        await provider.stop()


async def test_begin_timeout_terminates_and_closes_session():
    conn = FakeConnection(begin=False)
    provider = make_provider(
        FakeConnector(conn), transport=make_transport(handshake_timeout_s=0.05)
    )
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == ProviderErrorCode.TIMEOUT
    assert conn.text_types() == ["Terminate"]
    assert conn.closed
    assert live_adapter_tasks() == []


async def test_close_during_handshake_maps_auth_failure_without_leaking():
    conn = FakeConnection(begin=False)
    conn.drop(1008, f"Unauthorized Connection: bad key {API_KEY}")
    provider = make_provider(FakeConnector(conn))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    error = exc_info.value.error
    assert (error.code, error.retryable, error.details) == (
        ProviderErrorCode.AUTH_FAILED,
        False,
        {"close_code": 1008},
    )
    assert API_KEY not in str(exc_info.value)
    assert live_adapter_tasks() == []


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [(401, ProviderErrorCode.AUTH_FAILED, False), (429, ProviderErrorCode.RATE_LIMITED, True)],
)
async def test_handshake_http_rejection_maps_status(status, expected, retryable):
    provider = make_provider(FakeConnector(HandshakeRejectedError(status)))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert (exc_info.value.error.code, exc_info.value.error.retryable) == (expected, retryable)


async def test_audio_is_rechunked_and_finalize_pads_then_forces_endpoint():
    connector = FakeConnector()
    provider = make_provider(connector, transport=make_transport(chunk_ms=100.0))
    await provider.start()
    conn = connector.opened[0]
    try:
        for seq in range(7):
            await provider.send_audio(make_frame(seq=seq, ms=20))
        await wait_until(lambda: len(conn.audio()) == 1)
        assert len(conn.audio()[0]) == 3200

        await provider.finalize_stream("default")
        await wait_until(lambda: conn.text_types() == ["ForceEndpoint"])
        tail = conn.audio()[1]
        assert len(tail) == 1600
        assert tail[1280:] == b"\x00" * 320
        assert isinstance(conn.sent[-1], str)

        await provider.finalize_stream("default")
        await asyncio.sleep(0.02)
        assert conn.text_types() == ["ForceEndpoint"]

        await provider.send_audio(make_frame(seq=9, ms=250))
        await wait_until(lambda: len(conn.audio()) == 4)
        assert [len(chunk) for chunk in conn.audio()[2:]] == [3200, 3200]
    finally:
        await provider.stop()


async def test_turn_events_are_mapped_through_receive_events():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic",))
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        conn.push(turn(0, "hel"))
        conn.push(turn(0, "hel"))
        conn.push({"type": "SpeechStarted", "timestamp": 10})
        conn.push(turn(0, "hello", end=True))
        partial = await next_event(events)
        final = await next_event(events)
        assert (partial.stream_id, partial.text, partial.is_final, partial.revision) == (
            "mic",
            "hel",
            False,
            1,
        )
        assert (final.text, final.is_final, final.revision) == ("hello", True, 2)
        assert final.provider_event_id == f"assemblyai:{SESSION_ID}:0"
    finally:
        await provider.stop()


async def test_send_audio_never_blocks_and_reports_backpressure():
    connector = FakeConnector()
    provider = make_provider(connector, transport=make_transport(max_queued_frames=2, chunk_ms=50))
    await provider.start()
    conn = connector.opened[0]
    conn.send_gate = asyncio.Event()
    try:
        for seq in range(8):
            await asyncio.wait_for(provider.send_audio(make_frame(seq=seq, ms=50)), 0.05)
        state = provider.send_queue_state("default")
        assert state.backpressured and state.dropped_audio == 5 and state.queued_audio == 2

        conn.send_gate.set()
        await wait_until(lambda: len(conn.audio()) == 3)
        seqs = [struct.unpack("<h", pcm[:2])[0] - 1 for pcm in conn.audio()]
        assert seqs == [0, 6, 7]
    finally:
        await provider.stop()


async def test_reset_stream_clears_audio_forces_endpoint_and_ignores_stale_turn():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame(ms=100))
        await wait_until(lambda: len(conn.audio()) == 1)
        conn.push(turn(0, "stale"))
        assert (await next_event(events)).text == "stale"

        await provider.send_audio(make_frame(ms=40))
        await provider.reset_stream("default")
        await wait_until(lambda: conn.text_types() == ["ForceEndpoint"])
        await provider.finalize_stream("default")
        await asyncio.sleep(0.02)
        assert len(conn.audio()) == 1 and conn.text_types() == ["ForceEndpoint"]

        conn.push(turn(0, "stale words", end=True))
        conn.push(turn(1, "fresh", start_ms=150, end_ms=300))
        fresh = await next_event(events)
        assert (fresh.sequence, fresh.text) == (1, "fresh")
    finally:
        await provider.stop()


# ── Adapter: errors ──────────────────────────────────────────────────────────


async def test_rate_limit_close_flushes_turn_then_surfaces_retryable_error():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        conn.push(turn(0, "kept words"))
        assert (await next_event(events)).is_final is False
        conn.push({"error": f"Too many concurrent sessions for {API_KEY}"})
        conn.drop(3009, "Unauthorized Connection: Too many concurrent sessions")

        flushed = await next_event(events)
        assert (flushed.text, flushed.is_final) == ("kept words", True)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(events)
        error = exc_info.value.error
        assert (error.code, error.retryable) == (ProviderErrorCode.RATE_LIMITED, True)

        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.send_audio(make_frame())
        assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED
    finally:
        await provider.stop()
    assert live_adapter_tasks() == []


async def test_server_error_close_is_retryable_internal_error():
    connector = FakeConnector()
    provider = make_provider(connector)
    await provider.start()
    try:
        connector.opened[0].drop(3005, "Session Cancelled")
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await next_event(provider.receive_events())
        assert (exc_info.value.error.code, exc_info.value.error.retryable) == (
            ProviderErrorCode.INTERNAL_ERROR,
            True,
        )
    finally:
        await provider.stop()


# ── Adapter: termination and cleanup ─────────────────────────────────────────


async def test_stop_flushes_audio_terminates_and_waits_for_termination():
    connector = FakeConnector()
    provider = make_provider(connector, stream_ids=("mic", "system"))
    await provider.start()
    mic = connector.opened[0]

    def server(message: Dict[str, Any]) -> None:
        if message["type"] == "Terminate":
            mic.push(turn(0, "last words", end=True))

    mic.on_text = server
    received: List[STTTranscriptResult] = []

    async def consume() -> None:
        async for result in provider.receive_events():
            received.append(result)

    consumer = asyncio.ensure_future(consume())
    await provider.send_audio(make_frame("mic", ms=30))
    await provider.stop()
    await asyncio.wait_for(consumer, 1.0)

    assert [len(chunk) for chunk in mic.audio()] == [1600]
    assert mic.text_types() == ["Terminate"]
    assert all(conn.text_types()[-1] == "Terminate" for conn in connector.opened)
    assert [(r.text, r.is_final) for r in received] == [("last words", True)]
    assert all(conn.closed for conn in connector.opened)
    assert live_adapter_tasks() == []
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(make_frame("mic"))
    assert exc_info.value.error.code == ProviderErrorCode.INVALID_REQUEST


async def test_stop_closes_when_termination_is_never_confirmed(caplog):
    caplog.set_level(logging.WARNING, logger="echoflux")
    conn = FakeConnection(confirm_terminate=False)
    provider = make_provider(
        FakeConnector(conn), transport=make_transport(terminate_timeout_s=0.05)
    )
    await provider.start()
    events = provider.receive_events()
    conn.push(turn(0, "pending"))
    assert (await next_event(events)).is_final is False

    await provider.stop()
    flushed = await next_event(events)
    assert (flushed.text, flushed.is_final) == ("pending", True)
    with pytest.raises(StopAsyncIteration):
        await next_event(events)
    assert conn.text_types() == ["Terminate"] and conn.closed
    assert "did not confirm termination" in caplog.text
    assert live_adapter_tasks() == []


async def test_failed_second_stream_terminates_first_billable_session():
    first = FakeConnection()
    connector = FakeConnector(first, HandshakeRejectedError(429))
    provider = make_provider(connector, stream_ids=("mic", "system"))
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED
    assert first.text_types() == ["Terminate"]
    assert first.closed
    assert live_adapter_tasks() == []


async def test_cancelled_start_closes_socket_and_leaks_no_task():
    conn = FakeConnection(begin=False)
    provider = make_provider(FakeConnector(conn), transport=make_transport(handshake_timeout_s=5))
    start = asyncio.ensure_future(provider.start())
    await asyncio.sleep(0.02)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    assert conn.closed
    assert live_adapter_tasks() == []


async def test_health_check_terminates_probe_session():
    conn = FakeConnection()
    provider = make_provider(FakeConnector(conn))
    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.HEALTHY
    assert conn.text_types() == ["Terminate"] and conn.closed

    provider = make_provider(FakeConnector(HandshakeRejectedError(401)))
    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.UNHEALTHY
    assert health.details == {"code": "auth_failed", "retryable": False}
    assert API_KEY not in health.message
    assert live_adapter_tasks() == []


async def test_logs_never_contain_token_query_audio_or_transcript(caplog):
    caplog.set_level(logging.DEBUG, logger="echoflux")
    connector = FakeConnector()
    provider = make_provider(
        connector,
        settings=make_settings(auth_mode="temporary_token", keyterms_prompt=(KEYTERM,)),
    )
    await provider.start()
    conn = connector.opened[0]
    events = provider.receive_events()
    try:
        await provider.send_audio(make_frame(seq=1234, ms=100))
        conn.push(turn(0, TRANSCRIPT_SECRET))
        await next_event(events)
        conn.push("{garbage")
        conn.push({"error": f"{API_KEY} {TRANSCRIPT_SECRET}"})
        conn.drop(1008, f"Unauthorized {API_KEY}")
        await next_event(events)
        with pytest.raises(ProviderRuntimeError):
            await next_event(events)
    finally:
        await provider.stop()

    text = caplog.text
    assert caplog.records
    for forbidden in (API_KEY, KEYTERM, TRANSCRIPT_SECRET, "token=", "sample_rate="):
        assert forbidden not in text


# ── Adapter against a local WebSocket server ─────────────────────────────────


@pytest.fixture
def no_proxy_env(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


async def test_local_websocket_server_end_to_end(no_proxy_env):
    from websockets.asyncio.server import serve

    request: Dict[str, Any] = {}
    frames: List[str] = []
    server_closed = asyncio.Event()

    async def handler(ws: Any) -> None:
        request["authorization"] = ws.request.headers.get("Authorization")
        request["path"] = ws.request.path
        await ws.send(json.dumps({"type": "Begin", "id": "local-session", "expires_at": 1}))
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    frames.append(f"audio:{len(message)}")
                    await ws.send(json.dumps(turn(0, "local")))
                    continue
                kind = json.loads(message)["type"]
                frames.append(kind)
                if kind == "ForceEndpoint":
                    await ws.send(json.dumps(turn(0, "local server", end=True)))
                elif kind == "Terminate":
                    await ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": 0}))
                    await ws.close()
        finally:
            server_closed.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        provider = AssemblyAISTTProvider(
            make_settings(endpoint=f"ws://127.0.0.1:{port}/v3/ws"),
            transport=make_transport(),
        )
        await provider.start()
        events = provider.receive_events()
        try:
            await provider.send_audio(make_frame(ms=100))
            partial = await next_event(events)
            await provider.finalize_stream("default")
            final = await next_event(events)
        finally:
            await provider.stop()
        await asyncio.wait_for(server_closed.wait(), 2.0)

    assert request["authorization"] == API_KEY
    assert request["path"] == (
        "/v3/ws?sample_rate=16000&encoding=pcm_s16le&speech_model=universal-3-5-pro"
    )
    assert frames == ["audio:3200", "ForceEndpoint", "Terminate"]
    assert (partial.text, partial.is_final) == ("local", False)
    assert (final.text, final.is_final, final.provider_event_id) == (
        "local server",
        True,
        "assemblyai:local-session:0",
    )
    assert live_adapter_tasks() == []


async def test_local_websocket_server_http_rejection_maps_to_auth_failed(no_proxy_env):
    from websockets.asyncio.server import serve

    def reject(connection: Any, request: Any) -> Any:
        return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")

    async def handler(ws: Any) -> None:  # pragma: no cover - never reached
        await ws.close()

    async with serve(handler, "127.0.0.1", 0, process_request=reject) as server:
        port = server.sockets[0].getsockname()[1]
        provider = AssemblyAISTTProvider(
            make_settings(endpoint=f"ws://127.0.0.1:{port}/v3/ws"),
            transport=make_transport(),
        )
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert exc_info.value.error.details == {"http_status": 401}
    assert live_adapter_tasks() == []
