"""Gemini Live streaming Speech-to-Text adapter.

Owns the connection lifecycle, PCM16 mono framing, transparent session rotation, and
the mapping to canonical ``STTTranscriptResult`` events. It is dormant: nothing
registers it or wires it into the engine, and it never decides fallback.

Each audio ``stream_id`` gets its own Live connection, because a Live session
transcribes a single audio stream.

Rotation. Gemini Live connections end after roughly ten minutes and announce the cut
with ``goAway``. The adapter keeps that out of the event stream: past a soft deadline
(or on ``goAway``) it opens a replacement in the background, swaps at the next utterance
boundary (or at the hard deadline, whichever comes first), and drains the old
connection so an in-flight final still arrives exactly once. Audio after the swap goes
only to the new connection, so the two never transcribe the same samples. Revisions and
the audio timeline are per stream, so they stay monotonic across connections.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

from engine.core.logging import get_logger
from engine.providers.contracts import StreamingSTTProvider
from engine.providers.stt import gemini_protocol as proto
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

logger = get_logger("stt.gemini_live")

AUTH_MODE_API_KEY = "api_key"
AUTH_MODE_ACCESS_TOKEN = "access_token"

WireFrame = Union[str, bytes]  # noqa: UP007 -- typing house style (rule 04)


# ── Connection seam ──────────────────────────────────────────────────────────


class LiveConnectionClosedError(Exception):
    """The socket closed. ``reason`` may echo provider text and must be redacted."""

    def __init__(self, code: Optional[int], reason: str = "") -> None:
        super().__init__(f"connection closed ({code})")
        self.code = code
        self.reason = reason


class LiveConnectError(Exception):
    """The WebSocket handshake failed before any application message."""

    def __init__(self, status: Optional[int], cause: str = "") -> None:
        super().__init__(f"connect failed ({status})")
        self.status = status
        self.cause = cause


class LiveConnection(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> WireFrame: ...

    async def close(self) -> None: ...


Connector = Callable[[str, Dict[str, str]], Awaitable[LiveConnection]]
Clock = Callable[[], float]

# websockets logs handshake details at debug level; keep them out of the app log.
_wire_logger = get_logger("stt.gemini_live.wire")
_wire_logger.setLevel("WARNING")


class _WebsocketsConnection:
    def __init__(self, socket: Any) -> None:
        self._socket = socket

    async def send(self, message: str) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            await self._socket.send(message)
        except ConnectionClosed as exc:
            raise _closed_from(exc) from None

    async def recv(self) -> WireFrame:
        from websockets.exceptions import ConnectionClosed

        try:
            return await self._socket.recv()
        except ConnectionClosed as exc:
            raise _closed_from(exc) from None

    async def close(self) -> None:
        await self._socket.close()


def _closed_from(exc: Any) -> LiveConnectionClosedError:
    frame = getattr(exc, "rcvd", None)
    if frame is None:
        return LiveConnectionClosedError(1006, "")
    return LiveConnectionClosedError(frame.code, frame.reason)


async def websockets_connector(url: str, headers: Dict[str, str]) -> LiveConnection:
    """Default connector. Imported lazily so local-only startup never loads websockets."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidHandshake, InvalidStatus

    try:
        socket = await connect(url, additional_headers=headers, logger=_wire_logger)
    except InvalidStatus as exc:
        raise LiveConnectError(exc.response.status_code, "handshake rejected") from None
    except (InvalidHandshake, OSError, asyncio.TimeoutError) as exc:
        # Exception text can include the URL; only the type name leaves this function.
        raise LiveConnectError(None, type(exc).__name__) from None
    return _WebsocketsConnection(socket)


# ── Configuration ────────────────────────────────────────────────────────────


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class GeminiLiveSTTConfig:
    """Resolved Gemini Live STT settings. ``api_key`` never appears in repr."""

    api_key: str = field(default="", repr=False)
    endpoint: str = proto.DEFAULT_ENDPOINT
    model: str = proto.DEFAULT_MODEL
    language_hints: Tuple[str, ...] = ()
    custom_vocabulary: Tuple[str, ...] = ()
    transcription_mode: Optional[str] = None
    # "api_key" sends the key in a header; "access_token" is an ephemeral token, which
    # the service only accepts in the query string.
    auth_mode: str = AUTH_MODE_API_KEY
    manual_activity: bool = False
    chunk_ms: float = 100.0
    setup_timeout_s: float = 10.0
    # The service cuts connections at about ten minutes; rotate well before that.
    rotate_after_s: float = 480.0
    max_connection_s: float = 570.0
    go_away_margin_s: float = 2.0
    drain_timeout_s: float = 5.0
    reconnect_backoff_s: float = 5.0
    provider_id: str = "gemini-live"

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "GeminiLiveSTTConfig":
        """Build from a resolved service config mapping. Unknown keys are ignored."""
        values: Dict[str, Any] = {}
        simple = {
            "api_key": "api_key",
            "endpoint": "endpoint",
            "model": "model",
            "mode": "transcription_mode",
            "transcription_mode": "transcription_mode",
            "auth_mode": "auth_mode",
            "manual_activity": "manual_activity",
            "chunk_ms": "chunk_ms",
            "setup_timeout_s": "setup_timeout_s",
            "rotate_after_s": "rotate_after_s",
            "max_connection_s": "max_connection_s",
            "go_away_margin_s": "go_away_margin_s",
            "drain_timeout_s": "drain_timeout_s",
            "reconnect_backoff_s": "reconnect_backoff_s",
            "provider_id": "provider_id",
        }
        for key, attr in simple.items():
            if options.get(key) not in (None, ""):
                values[attr] = options[key]
        if options.get("model_override"):
            values["model"] = options["model_override"]
        hints = options.get("language_hints", options.get("languages"))
        if hints is not None:
            values["language_hints"] = _as_tuple(hints)
        if options.get("custom_vocabulary") is not None:
            values["custom_vocabulary"] = _as_tuple(options["custom_vocabulary"])
        return cls(**values)


# ── Session state ────────────────────────────────────────────────────────────


class _SessionState(str, Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSED = "closed"


@dataclass(eq=False)
class _LiveSession:
    stream_id: str
    generation: int
    connection: LiveConnection
    opened_at: float
    state: _SessionState = _SessionState.ACTIVE
    reader: Optional["asyncio.Task[None]"] = None
    segment_index: int = 0
    segment_open: bool = False
    segment_start_ms: Optional[float] = None
    sent_end_ms: float = 0.0
    last_sequence: int = -1
    last_interim_text: str = ""
    last_final_text: str = ""
    activity_open: bool = False
    go_away: bool = False
    error_reported: bool = False
    closing: bool = False
    boundary: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(eq=False)
class _StreamState:
    stream_id: str
    active: Optional[_LiveSession] = None
    standby: Optional[Tuple[LiveConnection, float]] = None
    pending: Optional["asyncio.Task[Tuple[LiveConnection, float]]"] = None
    draining: List[_LiveSession] = field(default_factory=list)
    generation: int = 0
    revision: int = 0
    audio_ms: float = 0.0
    buffer: bytearray = field(default_factory=bytearray)
    buffer_rate: int = proto.DEFAULT_SAMPLE_RATE
    buffer_sequence: int = -1
    soft_deadline: float = float("inf")
    hard_deadline: float = float("inf")
    next_prepare_at: float = 0.0
    last_final_text: str = ""
    last_final_generation: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_STOP = object()


def _pcm_ms(byte_count: int, sample_rate: int) -> float:
    return byte_count / (proto.PCM16_SAMPLE_WIDTH_BYTES * sample_rate) * 1000.0


# ── Provider ─────────────────────────────────────────────────────────────────


class GeminiLiveSTTProvider(StreamingSTTProvider):
    """Streaming STT over the Gemini Live WebSocket API.

    Setup failures raise ``ProviderRuntimeError`` from ``send_audio`` (the caller is
    waiting on it). Failures after setup surface from ``receive_events``.
    """

    def __init__(
        self,
        config: GeminiLiveSTTConfig,
        connector: Optional[Connector] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._config = config
        self._connector: Connector = connector or websockets_connector
        self._clock: Clock = clock or time.monotonic
        self._streams: Dict[str, _StreamState] = {}
        self._events: Optional[asyncio.Queue] = None
        self._tasks: Set[asyncio.Task] = set()
        self._started = False
        self._stopping = False

    @property
    def provider_id(self) -> str:
        return self._config.provider_id

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._config.api_key:
            raise self._error(ProviderErrorCode.AUTH_FAILED, "Gemini Live credential is missing.")
        self._setup_message()
        self._events = asyncio.Queue()
        self._stopping = False
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        for state in list(self._streams.values()):
            await self._close_stream(state, graceful=True)
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._streams.clear()
        self._started = False
        if self._events is not None:
            self._events.put_nowait(_STOP)

    async def health_check(self) -> ProviderHealthCheck:
        started_at = self._clock()
        try:
            connection = await self._open_connection()
        except ProviderRuntimeError as exc:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=exc.error.message,
                details={"code": exc.error.code.value, "retryable": exc.error.retryable},
            )
        await self._safe_close(connection)
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Gemini Live accepted the session setup.",
            latency_ms=(self._clock() - started_at) * 1000.0,
        )

    # ── Streaming contract ───────────────────────────────────────────────────

    async def send_audio(self, frame: AudioFrame) -> None:
        self._require_started()
        self._validate_frame(frame)
        if not frame.pcm_data:
            return
        state = self._stream(frame.stream_id)
        async with state.lock:
            if state.buffer and state.buffer_rate != frame.sample_rate:
                await self._flush(state)
            state.buffer.extend(frame.pcm_data)
            state.buffer_rate = frame.sample_rate
            state.buffer_sequence = frame.sequence
            if _pcm_ms(len(state.buffer), state.buffer_rate) >= self._config.chunk_ms:
                await self._flush(state)

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        self._require_started()
        queue = self._events
        assert queue is not None
        while True:
            item = await queue.get()
            if item is _STOP:
                return
            if isinstance(item, ProviderError):
                raise ProviderRuntimeError(item)
            yield item

    async def finalize_stream(self, stream_id: str) -> None:
        self._require_started()
        state = self._streams.get(stream_id)
        if state is None:
            return
        async with state.lock:
            await self._flush(state)
            session = state.active
            if session is None:
                return
            await self._send_boundary(session)

    async def reset_stream(self, stream_id: str) -> None:
        """Drop the stream's connections and buffered audio.

        The revision counter and audio timeline are kept so results stay monotonic.
        """
        state = self._streams.get(stream_id)
        if state is None:
            return
        async with state.lock:
            await self._close_stream(state, graceful=False)

    # ── Connection handling ──────────────────────────────────────────────────

    def _setup_message(self) -> Dict[str, Any]:
        try:
            return proto.build_setup_message(
                self._config.model,
                self._config.language_hints,
                self._config.custom_vocabulary,
                self._config.transcription_mode,
                self._config.manual_activity,
            )
        except ValueError as exc:
            raise self._error(ProviderErrorCode.INVALID_REQUEST, str(exc)) from None

    def _connection_target(self) -> Tuple[str, Dict[str, str]]:
        if self._config.auth_mode == AUTH_MODE_ACCESS_TOKEN:
            url = proto.build_connect_url(self._config.endpoint, self._config.api_key)
            return url, {}
        return self._config.endpoint, {proto.API_KEY_HEADER: self._config.api_key}

    async def _open_connection(self) -> LiveConnection:
        """Connect, send setup as the first message, and wait for setupComplete."""
        setup = proto.encode_message(self._setup_message())
        url, headers = self._connection_target()
        logger.info(
            "Connecting to Gemini Live at %s (model %s)",
            proto.redact_url(url),
            proto.normalize_model_name(self._config.model),
        )
        try:
            connection = await self._connector(url, headers)
        except LiveConnectError as exc:
            code, retryable = proto.classify_http_status(exc.status)
            raise self._error(
                code,
                "Could not connect to Gemini Live.",
                retryable,
                {"http_status": exc.status},
            ) from None
        except OSError as exc:
            raise self._error(
                ProviderErrorCode.NETWORK_UNAVAILABLE,
                "Could not connect to Gemini Live.",
                True,
                {"exception": type(exc).__name__},
            ) from None

        try:
            await connection.send(setup)
            await asyncio.wait_for(
                self._await_setup_complete(connection), self._config.setup_timeout_s
            )
        except LiveConnectionClosedError as exc:
            await self._safe_close(connection)
            reason = self._redact(exc.reason)
            code, retryable = proto.classify_close(exc.code, reason)
            raise self._error(
                code,
                "Gemini Live rejected the session setup.",
                retryable,
                {"close_code": exc.code, "reason": reason[:200]},
            ) from None
        except asyncio.TimeoutError:
            await self._safe_close(connection)
            raise self._error(
                ProviderErrorCode.TIMEOUT, "Gemini Live did not confirm the session setup.", True
            ) from None
        except BaseException:
            await self._safe_close(connection)
            raise
        return connection

    async def _await_setup_complete(self, connection: LiveConnection) -> None:
        while True:
            raw = await connection.recv()
            try:
                events = proto.parse_server_message(raw)
            except proto.GeminiProtocolError as exc:
                logger.warning("Ignoring malformed Gemini Live frame during setup: %s", exc)
                continue
            for event in events:
                if event.kind is proto.ServerEventKind.SETUP_COMPLETE:
                    return
                if event.kind is proto.ServerEventKind.ERROR:
                    raise LiveConnectionClosedError(
                        event.error_code, f"{event.error_status} {event.error_message}"
                    )

    async def _prepare_replacement(self) -> Tuple[LiveConnection, float]:
        connection = await self._open_connection()
        return connection, self._clock()

    def _install(self, state: _StreamState, connection: LiveConnection, opened_at: float) -> None:
        state.generation += 1
        session = _LiveSession(
            stream_id=state.stream_id,
            generation=state.generation,
            connection=connection,
            opened_at=opened_at,
            sent_end_ms=state.audio_ms,
        )
        state.active = session
        state.soft_deadline = opened_at + self._config.rotate_after_s
        state.hard_deadline = opened_at + self._config.max_connection_s
        session.reader = self._spawn(self._read_loop(state, session))

    async def _ensure_active(self, state: _StreamState) -> _LiveSession:
        if state.active is None:
            if state.pending is not None:
                # A replacement is already connecting (e.g. after goAway); reuse it.
                await asyncio.wait({state.pending})
                self._collect_pending(state, self._clock())
            if state.standby is not None:
                connection, opened_at = state.standby
                state.standby = None
            else:
                connection = await self._open_connection()
                opened_at = self._clock()
            self._install(state, connection, opened_at)
        assert state.active is not None
        return state.active

    async def _maybe_rotate(self, state: _StreamState) -> None:
        session = state.active
        now = self._clock()
        if session is None or now < state.soft_deadline:
            return
        self._collect_pending(state, now)
        if state.standby is None and state.pending is None and now >= state.next_prepare_at:
            state.pending = self._spawn(self._prepare_replacement(), log_errors=False)
        if state.standby is None and state.pending is not None and now >= state.hard_deadline:
            # Out of time: wait for the replacement rather than feed a dying connection.
            await asyncio.wait({state.pending})
            self._collect_pending(state, self._clock())
        if state.standby is not None and (not session.segment_open or now >= state.hard_deadline):
            self._swap(state)

    def _collect_pending(self, state: _StreamState, now: float) -> None:
        task = state.pending
        if task is None or not task.done():
            return
        state.pending = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            state.standby = task.result()
            return
        state.next_prepare_at = now + self._config.reconnect_backoff_s
        if isinstance(exc, ProviderRuntimeError):
            logger.warning(
                "Gemini Live replacement connection failed for stream %s: %s",
                state.stream_id,
                exc.error,
            )
            if not exc.error.retryable:
                self._publish(exc.error)
        else:
            logger.warning(
                "Gemini Live replacement connection failed for stream %s: %s",
                state.stream_id,
                type(exc).__name__,
            )

    def _swap(self, state: _StreamState) -> None:
        old = state.active
        assert old is not None and state.standby is not None
        connection, opened_at = state.standby
        state.standby = None
        old.state = _SessionState.DRAINING
        state.draining.append(old)
        self._install(state, connection, opened_at)
        logger.info(
            "Gemini Live session rotated for stream %s (generation %d -> %d)",
            state.stream_id,
            old.generation,
            state.generation,
        )
        self._spawn(self._drain(state, old))

    async def _drain(self, state: _StreamState, session: _LiveSession) -> None:
        try:
            await self._send_boundary(session)
            if session.segment_open:
                try:
                    await asyncio.wait_for(session.boundary.wait(), self._config.drain_timeout_s)
                except asyncio.TimeoutError:
                    # The closing connection will never finalize this utterance; promote
                    # the last hypothesis so the text is not lost.
                    if session.segment_open and session.last_interim_text:
                        self._finalize_segment(
                            state, session, session.last_interim_text, None, promoted=True
                        )
        finally:
            await self._close_session(state, session)

    async def _flush(self, state: _StreamState) -> None:
        if not state.buffer:
            return
        pcm = bytes(state.buffer)
        rate = state.buffer_rate
        state.buffer.clear()
        session = await self._ensure_active(state)
        await self._maybe_rotate(state)
        session = state.active or session

        if session.segment_start_ms is None:
            session.segment_start_ms = state.audio_ms
        state.audio_ms += _pcm_ms(len(pcm), rate)
        session.sent_end_ms = state.audio_ms
        session.last_sequence = state.buffer_sequence
        try:
            if self._config.manual_activity and not session.activity_open:
                await session.connection.send(proto.encode_message(proto.build_activity_start()))
                session.activity_open = True
            await session.connection.send(
                proto.encode_message(proto.build_audio_message(pcm, rate))
            )
        except LiveConnectionClosedError:
            # The reader owns reporting; the next chunk reconnects.
            logger.debug("Gemini Live send failed on closed stream %s", state.stream_id)

    async def _send_boundary(self, session: _LiveSession) -> None:
        if self._config.manual_activity:
            if not session.activity_open:
                return
            message = proto.build_activity_end()
            session.activity_open = False
        else:
            message = proto.build_audio_stream_end()
        try:
            await session.connection.send(proto.encode_message(message))
        except LiveConnectionClosedError:
            logger.debug("Gemini Live boundary not sent; stream %s closed", session.stream_id)

    # ── Server events ────────────────────────────────────────────────────────

    async def _read_loop(self, state: _StreamState, session: _LiveSession) -> None:
        try:
            while True:
                raw = await session.connection.recv()
                try:
                    events = proto.parse_server_message(raw)
                except proto.GeminiProtocolError as exc:
                    logger.warning(
                        "Ignoring malformed Gemini Live frame on stream %s: %s",
                        state.stream_id,
                        exc,
                    )
                    continue
                for event in events:
                    self._handle_event(state, session, event)
        except LiveConnectionClosedError as exc:
            self._handle_closed(state, session, exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Gemini Live reader failed on stream %s: %s", state.stream_id, type(exc).__name__
            )
            self._detach(state, session)
            if not self._stopping and not session.closing:
                self._publish(
                    self._error(
                        ProviderErrorCode.INTERNAL_ERROR, "Gemini Live stream failed.", True
                    ).error
                )

    def _handle_event(
        self, state: _StreamState, session: _LiveSession, event: proto.ServerEvent
    ) -> None:
        kind = event.kind
        if kind is proto.ServerEventKind.INTERIM:
            text = event.text
            if not text.strip() or text == session.last_interim_text:
                return
            session.segment_open = True
            session.last_interim_text = text
            self._emit(state, session, text, False, event.language_code)
        elif kind is proto.ServerEventKind.FINAL:
            self._finalize_segment(state, session, event.text, event.language_code)
        elif kind is proto.ServerEventKind.TURN_COMPLETE:
            if not session.segment_open:
                session.boundary.set()
        elif kind is proto.ServerEventKind.GO_AWAY:
            self._handle_go_away(state, session, event.time_left_s)
        elif kind is proto.ServerEventKind.ERROR:
            reason = self._redact(f"{event.error_status} {event.error_message}")
            code, retryable = proto.classify_close(event.error_code, reason)
            session.error_reported = True
            self._publish(
                self._error(
                    code,
                    "Gemini Live reported a stream error.",
                    retryable,
                    {"status": event.error_status[:64]},
                ).error
            )

    def _finalize_segment(
        self,
        state: _StreamState,
        session: _LiveSession,
        text: str,
        language: Optional[str],
        promoted: bool = False,
    ) -> None:
        stripped = text.strip()
        if stripped and not session.segment_open:
            # A repeated final with no new hypothesis in between is a replay, not speech.
            # On a fresh connection's first segment, the previous connection's last final
            # is stale text; that connection never heard the audio it describes.
            repeat_on_session = text == session.last_final_text
            repeat_across_rotation = (
                session.segment_index == 0
                and session.generation != state.last_final_generation
                and text == state.last_final_text
            )
            if repeat_on_session or repeat_across_rotation:
                logger.debug("Dropped duplicate Gemini Live final on stream %s", state.stream_id)
                return
        if stripped:
            state.last_final_text = text
            state.last_final_generation = session.generation
            session.last_final_text = text
            self._emit(state, session, text, True, language)
            if promoted:
                logger.debug("Promoted last interim to final on stream %s", state.stream_id)
        if stripped or session.segment_open:
            session.segment_index += 1
        session.segment_open = False
        session.last_interim_text = ""
        session.segment_start_ms = None
        session.boundary.set()

    def _handle_go_away(
        self, state: _StreamState, session: _LiveSession, time_left_s: Optional[float]
    ) -> None:
        session.go_away = True
        if state.active is not session:
            return
        now = self._clock()
        state.soft_deadline = now
        if time_left_s is not None:
            cutoff = now + max(0.0, time_left_s - self._config.go_away_margin_s)
            state.hard_deadline = min(state.hard_deadline, cutoff)
        logger.info("Gemini Live announced session end for stream %s", state.stream_id)
        if state.pending is None and state.standby is None:
            state.pending = self._spawn(self._prepare_replacement(), log_errors=False)

    def _handle_closed(
        self, state: _StreamState, session: _LiveSession, exc: LiveConnectionClosedError
    ) -> None:
        was_draining = session.state is _SessionState.DRAINING
        self._detach(state, session)
        if self._stopping or session.closing or was_draining or session.error_reported:
            return
        if session.go_away or exc.code in (1000, 1001):
            logger.info(
                "Gemini Live connection ended for stream %s; reconnecting on next audio",
                state.stream_id,
            )
            return
        reason = self._redact(exc.reason)
        code, retryable = proto.classify_close(exc.code, reason)
        self._publish(
            self._error(
                code,
                "Gemini Live connection closed unexpectedly.",
                retryable,
                {"close_code": exc.code, "reason": reason[:200]},
            ).error
        )

    def _emit(
        self,
        state: _StreamState,
        session: _LiveSession,
        text: str,
        is_final: bool,
        language: Optional[str],
    ) -> STTTranscriptResult:
        state.revision += 1
        start_ms = session.segment_start_ms
        if start_ms is None:
            start_ms = session.sent_end_ms
        result = STTTranscriptResult(
            stream_id=state.stream_id,
            sequence=session.last_sequence,
            provider_event_id=(
                f"gemini:{state.stream_id}:{session.generation}:{session.segment_index}"
            ),
            text=text,
            is_final=is_final,
            revision=state.revision,
            start_time_ms=start_ms,
            end_time_ms=max(start_ms, session.sent_end_ms),
            monotonic_timestamp=self._clock(),
            detected_language=language,
            is_endpoint=is_final,
        )
        if self._events is not None:
            self._events.put_nowait(result)
        return result

    # ── Teardown ─────────────────────────────────────────────────────────────

    def _detach(self, state: _StreamState, session: _LiveSession) -> None:
        session.state = _SessionState.CLOSED
        session.boundary.set()
        if state.active is session:
            state.active = None
        if session in state.draining:
            state.draining.remove(session)

    async def _close_session(self, state: _StreamState, session: _LiveSession) -> None:
        session.closing = True
        self._detach(state, session)
        await self._safe_close(session.connection)
        reader = session.reader
        if reader is not None and reader is not asyncio.current_task() and not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _close_stream(self, state: _StreamState, graceful: bool) -> None:
        state.buffer.clear()
        if state.pending is not None:
            task = state.pending
            state.pending = None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if not task.cancelled() and task.exception() is None:
                await self._safe_close(task.result()[0])
        if state.standby is not None:
            await self._safe_close(state.standby[0])
            state.standby = None
        sessions = list(state.draining)
        if state.active is not None:
            sessions.append(state.active)
            if graceful:
                await self._send_boundary(state.active)
        for session in sessions:
            await self._close_session(state, session)
        state.soft_deadline = float("inf")
        state.hard_deadline = float("inf")

    @staticmethod
    async def _safe_close(connection: LiveConnection) -> None:
        try:
            await connection.close()
        except Exception as exc:
            logger.debug("Ignoring Gemini Live close failure: %s", type(exc).__name__)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _spawn(
        self, coro: Coroutine[Any, Any, Any], log_errors: bool = True
    ) -> "asyncio.Task[Any]":
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)

        def _done(finished: "asyncio.Task[Any]") -> None:
            self._tasks.discard(finished)
            if finished.cancelled():
                return
            exc = finished.exception()
            if exc is not None and log_errors:
                logger.error("Gemini Live background task failed: %s", type(exc).__name__)

        task.add_done_callback(_done)
        return task

    def _stream(self, stream_id: str) -> _StreamState:
        state = self._streams.get(stream_id)
        if state is None:
            state = _StreamState(stream_id=stream_id)
            self._streams[stream_id] = state
        return state

    def _require_started(self) -> None:
        if not self._started or self._events is None:
            raise self._error(ProviderErrorCode.INVALID_REQUEST, "Gemini Live STT is not started.")

    def _validate_frame(self, frame: AudioFrame) -> None:
        if frame.channels != 1:
            raise self._error(
                ProviderErrorCode.INVALID_REQUEST, "Gemini Live expects mono PCM16 audio."
            )
        if frame.sample_rate <= 0:
            raise self._error(ProviderErrorCode.INVALID_REQUEST, "Audio sample rate is invalid.")
        if len(frame.pcm_data) % proto.PCM16_SAMPLE_WIDTH_BYTES:
            raise self._error(
                ProviderErrorCode.INVALID_REQUEST, "PCM16 audio must have an even byte length."
            )

    def _redact(self, text: str) -> str:
        return proto.redact_text(text or "", (self._config.api_key,))

    def _publish(self, error: ProviderError) -> None:
        if self._events is not None:
            self._events.put_nowait(error)

    @staticmethod
    def _error(
        code: ProviderErrorCode,
        message: str,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> ProviderRuntimeError:
        return ProviderRuntimeError(
            ProviderError(
                code=code,
                capability=ProviderCapability.STT,
                message=message,
                retryable=retryable,
                details=details,
            )
        )
