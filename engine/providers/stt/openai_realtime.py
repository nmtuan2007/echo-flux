"""OpenAI Realtime-compatible streaming STT adapter.

Owns socket lifecycle only: handshake, non-blocking audio send, event dispatch, session
rotation, and cleanup. Wire formats and transcript revision rules live in
``openai_protocol``. The adapter is not registered anywhere and never chooses fallback;
unrecoverable problems surface as ``ProviderRuntimeError`` from ``receive_events``.

Lifecycle
- ``start`` opens one Realtime transcription session per configured stream id and waits
  for ``session.updated``. Any failure closes what was opened and re-raises.
- ``send_audio`` only enqueues. A per-stream sender task resamples to 24 kHz and sends;
  when the outbox is full the oldest audio is dropped, commits and clears never are.
- ``finalize_stream`` enqueues ``input_audio_buffer.commit``. ``reset_stream`` drops
  queued audio, discards in-flight items, and enqueues ``input_audio_buffer.clear``.
- Rotation: after a commit, if the session is within ``rotation_margin_s`` of its
  ``expires_at``, a new session is opened before the old one is closed; the old socket
  keeps delivering completions for ``rotation_drain_s``. An unexpected close or a
  ``session_expired`` error finalizes in-flight items and reconnects with bounded
  retries; exhausting them reports ``connection_closed``.
- ``stop`` closes every socket, then cancels and awaits every task it spawned.

Nothing here logs the API key, the full URL, audio, or transcript text.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from engine.core.logging import get_logger
from engine.providers.contracts import StreamingSTTProvider
from engine.providers.stt.openai_protocol import (
    COMMIT_EMPTY_CODE,
    SESSION_EXPIRED_CODE,
    BufferCleared,
    BufferCommitted,
    MalformedEvent,
    OpenAIRealtimeSettings,
    PCM16Resampler,
    ServerError,
    ServerEvent,
    SessionReady,
    SpeechStarted,
    SpeechStopped,
    TranscriptAssembler,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    build_append,
    build_clear,
    build_commit,
    build_connection_url,
    build_session_update,
    encode,
    is_benign_server_error,
    make_error,
    map_http_status,
    map_server_error,
    parse_server_event,
    redact_endpoint,
    validate_settings,
)
from engine.providers.types import (
    AudioFrame,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)

logger = get_logger("providers.stt.openai_realtime")

_AUDIO = "audio"
_COMMIT = "commit"
_CLEAR = "clear"
_STOP = object()


class RealtimeConnectionClosedError(Exception):
    """Transport-neutral close signal raised by ``RealtimeConnection.send``/``recv``."""

    def __init__(self, code: Optional[int] = None) -> None:
        super().__init__(f"connection closed (code={code})")
        self.code = code


class RealtimeHandshakeRejectedError(Exception):
    """The server refused the WebSocket upgrade with an HTTP status."""

    def __init__(self, status: int) -> None:
        super().__init__(f"handshake rejected (status={status})")
        self.status = status


class RealtimeConnection(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


ConnectFactory = Callable[[str, Dict[str, str], float], Awaitable[RealtimeConnection]]


@dataclass(frozen=True)
class RealtimeTransportOptions:
    """Transport tuning. Provider settings (endpoint, model, hints) are not defaulted here."""

    handshake_timeout_s: float = 10.0
    close_timeout_s: float = 2.0
    max_queued_frames: int = 250
    max_reconnect_attempts: int = 3
    reconnect_backoff_s: float = 0.5
    rotation_margin_s: float = 120.0
    rotation_drain_s: float = 5.0
    max_message_bytes: int = 4 * 1024 * 1024


class _WebsocketsConnection:
    def __init__(self, ws: Any, closed_exc: type) -> None:
        self._ws = ws
        self._closed_exc = closed_exc

    def _closed(self, exc: Any) -> RealtimeConnectionClosedError:
        rcvd = getattr(exc, "rcvd", None)
        return RealtimeConnectionClosedError(getattr(rcvd, "code", None))

    async def send(self, message: str) -> None:
        try:
            await self._ws.send(message)
        except self._closed_exc as exc:
            raise self._closed(exc) from None

    async def recv(self) -> str | bytes:
        try:
            return await self._ws.recv()
        except self._closed_exc as exc:
            raise self._closed(exc) from None

    async def close(self) -> None:
        await self._ws.close()


def make_websocket_factory(max_message_bytes: int) -> ConnectFactory:
    """Default factory; ``websockets`` is imported only when a session is opened."""

    async def _connect(url: str, headers: Dict[str, str], open_timeout: float) -> Any:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        try:
            ws = await connect(
                url,
                additional_headers=headers,
                open_timeout=open_timeout,
                max_size=max_message_bytes,
                compression=None,
            )
        except InvalidStatus as exc:
            raise RealtimeHandshakeRejectedError(exc.response.status_code) from None
        return _WebsocketsConnection(ws, ConnectionClosed)

    return _connect


class _StreamSession:
    def __init__(self, stream_id: str, assembler: TranscriptAssembler) -> None:
        self.stream_id = stream_id
        self.assembler = assembler
        self.outbox: Deque[Tuple[str, Optional[AudioFrame]]] = deque()
        self.audio_count = 0
        self.dropped_frames = 0
        self.wakeup = asyncio.Event()
        self.ready = asyncio.Event()
        self.conn: Optional[RealtimeConnection] = None
        self.connections: List[RealtimeConnection] = []
        self.generation = 0
        self.expires_at: Optional[float] = None
        self.reconnecting = False
        self.resampler: Optional[PCM16Resampler] = None
        self.sender: Optional[asyncio.Task[None]] = None


class OpenAIRealtimeSTTProvider(StreamingSTTProvider):
    """Streaming STT over an OpenAI Realtime-compatible transcription session."""

    def __init__(
        self,
        settings: OpenAIRealtimeSettings,
        *,
        provider_id: str = "openai-realtime",
        stream_ids: Sequence[str] = ("default",),
        transport: Optional[RealtimeTransportOptions] = None,
        connect_factory: Optional[ConnectFactory] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        validate_settings(settings)
        if not stream_ids or len(set(stream_ids)) != len(stream_ids):
            raise ProviderRuntimeError(
                make_error(ProviderErrorCode.INVALID_REQUEST, detail="stream_ids")
            )
        self._settings = settings
        self._provider_id = provider_id
        self._stream_ids = tuple(stream_ids)
        self._transport = transport or RealtimeTransportOptions()
        self._connect_factory = connect_factory or make_websocket_factory(
            self._transport.max_message_bytes
        )
        self._clock = clock
        self._wall_clock = wall_clock
        self._url = build_connection_url(settings.endpoint)
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._sessions: Dict[str, _StreamSession] = {}
        self._tasks: Set[asyncio.Task[Any]] = set()
        self._started = False
        self._stopping = False
        self._failure: Optional[ProviderError] = None

    # ── BaseProvider ─────────────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        if self._started:
            return
        while not self._events.empty():
            self._events.get_nowait()
        self._stopping = False
        self._failure = None
        self._sessions = {
            stream_id: _StreamSession(stream_id, TranscriptAssembler(stream_id, self._clock))
            for stream_id in self._stream_ids
        }
        self._started = True
        try:
            for session in self._sessions.values():
                conn, expires_at = await self._open_connection()
                self._attach(session, conn, expires_at)
        except BaseException:
            self._stopping = True
            try:
                await self._teardown()
            finally:
                self._started = False
            raise
        logger.info(
            "OpenAI Realtime STT started: provider=%s streams=%d endpoint=%s model=%s",
            self._provider_id,
            len(self._sessions),
            redact_endpoint(self._url),
            self._settings.model,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        try:
            await self._teardown()
        finally:
            self._started = False
            self._events.put_nowait(_STOP)
        logger.info("OpenAI Realtime STT stopped: provider=%s", self._provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        began = self._clock()
        try:
            conn, _ = await self._open_connection()
        except ProviderRuntimeError as err:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=err.error.message,
                details={"code": err.error.code.value, "retryable": err.error.retryable},
            )
        await self._close_quietly(conn)
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="OpenAI Realtime transcription session accepted.",
            latency_ms=(self._clock() - began) * 1000.0,
            details={"model": self._settings.model},
        )

    # ── StreamingSTTProvider ─────────────────────────────────────────────────

    async def send_audio(self, frame: AudioFrame) -> None:
        session = self._require_session(frame.stream_id)
        if frame.channels != 1 or frame.sample_rate <= 0:
            raise ProviderRuntimeError(
                make_error(ProviderErrorCode.INVALID_REQUEST, detail="audio_format")
            )
        if not frame.pcm_data:
            return
        sample_count = len(frame.pcm_data) // 2
        session.assembler.advance_audio(sample_count * 1000.0 / frame.sample_rate)
        if session.audio_count >= self._transport.max_queued_frames:
            self._drop_oldest_audio(session)
        session.outbox.append((_AUDIO, frame))
        session.audio_count += 1
        session.wakeup.set()

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        while True:
            item = await self._events.get()
            if item is _STOP:
                return
            if isinstance(item, ProviderError):
                raise ProviderRuntimeError(item)
            yield item

    async def finalize_stream(self, stream_id: str) -> None:
        session = self._require_session(stream_id)
        if not session.assembler.has_uncommitted_audio():
            return
        session.assembler.mark_commit()
        session.outbox.append((_COMMIT, None))
        session.wakeup.set()

    async def reset_stream(self, stream_id: str) -> None:
        session = self._require_session(stream_id)
        session.outbox.clear()
        session.audio_count = 0
        session.assembler.reset()
        session.outbox.append((_CLEAR, None))
        session.wakeup.set()

    # ── Connection lifecycle ─────────────────────────────────────────────────

    async def _open_connection(self) -> Tuple[RealtimeConnection, Optional[float]]:
        timeout = self._transport.handshake_timeout_s
        headers = {"Authorization": f"Bearer {self._settings.api_key}"}
        try:
            conn = await asyncio.wait_for(
                self._connect_factory(self._url, headers, timeout), timeout
            )
        except RealtimeHandshakeRejectedError as exc:
            raise ProviderRuntimeError(map_http_status(exc.status)) from None
        except asyncio.TimeoutError:
            raise ProviderRuntimeError(make_error(ProviderErrorCode.TIMEOUT)) from None
        except RealtimeConnectionClosedError:
            raise ProviderRuntimeError(make_error(ProviderErrorCode.CONNECTION_CLOSED)) from None
        except ProviderRuntimeError:
            raise
        except Exception as exc:
            # The exception text can carry the URL; only the type is safe to log.
            logger.warning("OpenAI Realtime connect failed: error=%s", type(exc).__name__)
            raise ProviderRuntimeError(make_error(ProviderErrorCode.NETWORK_UNAVAILABLE)) from None

        try:
            await conn.send(encode(build_session_update(self._settings)))
            expires_at = await asyncio.wait_for(self._await_session_ready(conn), timeout)
        except BaseException as exc:
            await self._close_quietly(conn)
            if isinstance(exc, asyncio.TimeoutError):
                raise ProviderRuntimeError(make_error(ProviderErrorCode.TIMEOUT)) from None
            if isinstance(exc, RealtimeConnectionClosedError):
                raise ProviderRuntimeError(
                    make_error(ProviderErrorCode.CONNECTION_CLOSED)
                ) from None
            raise
        return conn, expires_at

    async def _await_session_ready(self, conn: RealtimeConnection) -> Optional[float]:
        expires_at: Optional[float] = None
        while True:
            event = parse_server_event(await conn.recv())
            if isinstance(event, SessionReady):
                expires_at = event.expires_at or expires_at
                if event.kind == "updated":
                    return expires_at
            elif isinstance(event, ServerError):
                raise ProviderRuntimeError(map_server_error(event))
            elif isinstance(event, MalformedEvent):
                logger.debug("OpenAI Realtime malformed handshake event: %s", event.reason)

    def _attach(
        self, session: _StreamSession, conn: RealtimeConnection, expires_at: Optional[float]
    ) -> None:
        session.generation += 1
        session.conn = conn
        session.connections.append(conn)
        session.expires_at = expires_at
        session.ready.set()
        offset_ms = session.assembler.cursor_ms
        self._spawn(
            self._receive_loop(session, conn, session.generation, offset_ms),
            f"openai-stt-recv-{session.stream_id}-{session.generation}",
        )
        if session.sender is None or session.sender.done():
            session.sender = self._spawn(
                self._send_loop(session), f"openai-stt-send-{session.stream_id}"
            )

    async def _teardown(self) -> None:
        conns = [conn for session in self._sessions.values() for conn in session.connections]
        try:
            for conn in conns:
                await self._close_quietly(conn)
        finally:
            while self._tasks:
                pending = list(self._tasks)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                self._tasks.difference_update(pending)
            for session in self._sessions.values():
                session.connections.clear()
                session.conn = None
                session.outbox.clear()
                session.audio_count = 0

    async def _close_quietly(self, conn: RealtimeConnection) -> None:
        try:
            await asyncio.wait_for(conn.close(), self._transport.close_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("OpenAI Realtime close failed: error=%s", type(exc).__name__)

    def _spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> "asyncio.Task[Any]":
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ── Send path ────────────────────────────────────────────────────────────

    async def _send_loop(self, session: _StreamSession) -> None:
        while True:
            if not session.outbox:
                session.wakeup.clear()
                await session.wakeup.wait()
                continue
            conn = session.conn
            if conn is None:
                await session.ready.wait()
                continue
            kind, frame = session.outbox.popleft()
            if kind == _AUDIO and frame is not None:
                session.audio_count -= 1
                payload = build_append(self._resample(session, frame))
            elif kind == _COMMIT:
                payload = build_commit()
            else:
                payload = build_clear()
            try:
                await conn.send(encode(payload))
            except RealtimeConnectionClosedError:
                # The receive loop owns reconnection; this message is lost with the session.
                logger.debug("OpenAI Realtime send on closed session: stream=%s", session.stream_id)
                continue
            except Exception as exc:
                logger.debug(
                    "OpenAI Realtime send failed: stream=%s error=%s",
                    session.stream_id,
                    type(exc).__name__,
                )
                continue
            if kind == _COMMIT and self._should_rotate(session):
                await self._rotate_proactively(session)

    def _resample(self, session: _StreamSession, frame: AudioFrame) -> bytes:
        resampler = session.resampler
        if resampler is None or resampler.src_rate != frame.sample_rate:
            resampler = PCM16Resampler(frame.sample_rate)
            session.resampler = resampler
        return resampler.process(frame.pcm_data)

    def _drop_oldest_audio(self, session: _StreamSession) -> None:
        for index, (kind, _) in enumerate(session.outbox):
            if kind == _AUDIO:
                del session.outbox[index]
                session.audio_count -= 1
                session.dropped_frames += 1
                if session.dropped_frames % 100 == 1:
                    logger.debug(
                        "OpenAI Realtime outbox full, dropping audio: stream=%s dropped=%d",
                        session.stream_id,
                        session.dropped_frames,
                    )
                return

    # ── Rotation ─────────────────────────────────────────────────────────────

    def _should_rotate(self, session: _StreamSession) -> bool:
        if session.expires_at is None or session.reconnecting or self._stopping:
            return False
        return self._wall_clock() >= session.expires_at - self._transport.rotation_margin_s

    async def _rotate_proactively(self, session: _StreamSession) -> None:
        try:
            conn, expires_at = await self._open_connection()
        except ProviderRuntimeError as err:
            # The current session is still valid; an eventual expiry is handled reactively.
            session.expires_at = None
            logger.warning(
                "OpenAI Realtime rotation deferred: stream=%s code=%s",
                session.stream_id,
                err.error.code.value,
            )
            return
        old = session.conn
        self._attach(session, conn, expires_at)
        if old is not None:
            self._spawn(self._retire(session, old), f"openai-stt-retire-{session.stream_id}")
        logger.info("OpenAI Realtime session rotated: stream=%s", session.stream_id)

    async def _retire(self, session: _StreamSession, conn: RealtimeConnection) -> None:
        await asyncio.sleep(self._transport.rotation_drain_s)
        await self._close_quietly(conn)
        if conn in session.connections:
            session.connections.remove(conn)

    async def _reconnect(self, session: _StreamSession, generation: int) -> None:
        if session.reconnecting or generation != session.generation or self._stopping:
            return
        session.reconnecting = True
        old = session.conn
        session.conn = None
        session.ready.clear()
        try:
            for result in session.assembler.flush_pending():
                self._publish(result)
            if old is not None:
                await self._close_quietly(old)
                if old in session.connections:
                    session.connections.remove(old)
            last_error = make_error(ProviderErrorCode.CONNECTION_CLOSED)
            for attempt in range(1, self._transport.max_reconnect_attempts + 1):
                if self._stopping:
                    return
                if attempt > 1:
                    await asyncio.sleep(self._transport.reconnect_backoff_s * (attempt - 1))
                try:
                    conn, expires_at = await self._open_connection()
                except ProviderRuntimeError as err:
                    logger.warning(
                        "OpenAI Realtime reconnect failed: stream=%s attempt=%d code=%s",
                        session.stream_id,
                        attempt,
                        err.error.code.value,
                    )
                    if not err.error.retryable:
                        last_error = err.error
                        break
                    continue
                self._attach(session, conn, expires_at)
                logger.info(
                    "OpenAI Realtime reconnected: stream=%s attempt=%d", session.stream_id, attempt
                )
                return
            self._fail(last_error)
        finally:
            session.reconnecting = False

    # ── Receive path ─────────────────────────────────────────────────────────

    async def _receive_loop(
        self,
        session: _StreamSession,
        conn: RealtimeConnection,
        generation: int,
        offset_ms: float,
    ) -> None:
        try:
            while True:
                event = parse_server_event(await conn.recv())
                if not await self._handle_event(session, event, generation, offset_ms):
                    return
        except RealtimeConnectionClosedError as closed:
            if self._stopping or generation != session.generation:
                return
            logger.warning(
                "OpenAI Realtime connection closed unexpectedly: stream=%s code=%s",
                session.stream_id,
                closed.code,
            )
            await self._reconnect(session, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "OpenAI Realtime receive loop failed: stream=%s error=%s",
                session.stream_id,
                type(exc).__name__,
            )
            self._fail(make_error(ProviderErrorCode.INTERNAL_ERROR))

    async def _handle_event(
        self, session: _StreamSession, event: ServerEvent, generation: int, offset_ms: float
    ) -> bool:
        assembler = session.assembler
        if isinstance(event, TranscriptDelta):
            self._publish(assembler.on_delta(event))
        elif isinstance(event, TranscriptCompleted):
            self._publish(assembler.on_completed(event))
        elif isinstance(event, TranscriptFailed):
            logger.debug(
                "OpenAI Realtime item transcription failed: stream=%s code=%s",
                session.stream_id,
                event.error_code,
            )
            self._publish(assembler.on_failed(event))
        elif isinstance(event, BufferCommitted):
            assembler.on_committed(event.item_id)
        elif isinstance(event, BufferCleared):
            assembler.on_cleared()
        elif isinstance(event, SpeechStarted):
            assembler.on_speech_started(event.item_id, offset_ms + event.audio_start_ms)
        elif isinstance(event, SpeechStopped):
            assembler.on_speech_stopped(event.item_id, offset_ms + event.audio_end_ms)
        elif isinstance(event, SessionReady):
            if generation == session.generation and event.expires_at:
                session.expires_at = event.expires_at
        elif isinstance(event, ServerError):
            return await self._handle_server_error(session, event, generation)
        elif isinstance(event, MalformedEvent):
            logger.debug(
                "OpenAI Realtime malformed event ignored: stream=%s reason=%s",
                session.stream_id,
                event.reason,
            )
        return True

    async def _handle_server_error(
        self, session: _StreamSession, event: ServerError, generation: int
    ) -> bool:
        if is_benign_server_error(event):
            if event.code == COMMIT_EMPTY_CODE:
                session.assembler.drop_oldest_pending_commit()
            return True
        error = map_server_error(event)
        if event.code == SESSION_EXPIRED_CODE:
            if generation == session.generation and not self._stopping:
                logger.info("OpenAI Realtime session expired: stream=%s", session.stream_id)
                await self._reconnect(session, generation)
            return False
        if error.code == ProviderErrorCode.INVALID_REQUEST:
            # Session config was accepted at handshake; a rejected client event is not fatal.
            logger.warning(
                "OpenAI Realtime rejected a client event: stream=%s code=%s",
                session.stream_id,
                (error.details or {}).get("provider_code"),
            )
            return True
        self._fail(error)
        return False

    def _publish(self, result: Optional[STTTranscriptResult]) -> None:
        if result is not None:
            self._events.put_nowait(result)

    def _fail(self, error: ProviderError) -> None:
        if self._failure is not None or self._stopping:
            return
        self._failure = error
        logger.error(
            "OpenAI Realtime STT failed: provider=%s code=%s retryable=%s",
            self._provider_id,
            error.code.value,
            error.retryable,
        )
        self._events.put_nowait(error)

    def _require_session(self, stream_id: str) -> _StreamSession:
        if self._failure is not None:
            raise ProviderRuntimeError(self._failure)
        if not self._started or self._stopping:
            raise ProviderRuntimeError(
                make_error(ProviderErrorCode.INVALID_REQUEST, detail="not_started")
            )
        session = self._sessions.get(stream_id)
        if session is None:
            raise ProviderRuntimeError(
                make_error(ProviderErrorCode.INVALID_REQUEST, detail="stream_id")
            )
        return session
