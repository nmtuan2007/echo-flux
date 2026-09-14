"""Provider-neutral WebSocket plumbing for binary-audio streaming STT adapters.

Transport mechanics only. Nothing here knows a provider's event names, auth scheme,
finality rules, or fallback policy; adapters supply the wire payloads and interpret
every inbound frame themselves.

- ``make_websocket_factory`` opens sockets; ``websockets`` is imported lazily so a
  local-only engine never loads it. Tests inject their own ``ConnectFactory``.
- ``StreamingSocket`` owns one open connection: a bounded, never-blocking send queue
  (oldest audio dropped first, control messages never dropped), a receive task that
  hands every frame to a callback, an optional idle keepalive driven by an injectable
  clock, and cancellation-safe close.
- ``ResultChannel`` is the event queue behind ``receive_events``.
- URL and text helpers keep credentials and query strings out of logs.
"""

import asyncio
import re
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
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from engine.core.logging import get_logger
from engine.providers.types import (
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderRuntimeError,
    STTTranscriptResult,
)

logger = get_logger("providers.stt.streaming_ws")

# websockets logs request headers at debug level; keep them out of the app log.
_wire_logger = get_logger("providers.stt.streaming_ws.wire")
_wire_logger.setLevel("WARNING")

WireMessage = Union[str, bytes]  # noqa: UP007 -- typing house style (rule 04)

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
REDACTED = "***"

_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_SECRET_FRAGMENT_RE = re.compile(
    r"(?i)\b(key|api_key|apikey|access_token|token|authorization)([=:]\s*)(\S+)"
)
_AUTH_VALUE_RE = re.compile(r"(?i)\b(bearer|token|basic)\s+[A-Za-z0-9._~+/=\-]{8,}")


# ── Connection seam ──────────────────────────────────────────────────────────


class SocketClosedError(Exception):
    """The socket closed. ``reason`` may echo provider text and must be redacted."""

    def __init__(self, code: Optional[int] = None, reason: str = "") -> None:
        super().__init__(f"socket closed (code={code})")
        self.code = code
        self.reason = reason


class HandshakeRejectedError(Exception):
    """The server refused the WebSocket upgrade with an HTTP status."""

    def __init__(self, status: int) -> None:
        super().__init__(f"handshake rejected (status={status})")
        self.status = status


class SocketConnectError(Exception):
    """The connection could not be opened. Carries only the failure type name."""

    def __init__(self, cause: str) -> None:
        super().__init__(f"connect failed ({cause})")
        self.cause = cause


class SocketConnection(Protocol):
    async def send(self, message: WireMessage) -> None: ...

    async def recv(self) -> WireMessage: ...

    async def close(self) -> None: ...


ConnectFactory = Callable[[str, Dict[str, str], float], Awaitable[SocketConnection]]


class _WebsocketsConnection:
    def __init__(self, ws: Any, closed_exc: type) -> None:
        self._ws = ws
        self._closed_exc = closed_exc

    @staticmethod
    def _closed(exc: Any) -> SocketClosedError:
        frame = getattr(exc, "rcvd", None)
        if frame is None:
            return SocketClosedError(1006, "")
        return SocketClosedError(frame.code, frame.reason or "")

    async def send(self, message: WireMessage) -> None:
        try:
            await self._ws.send(message)
        except self._closed_exc as exc:
            raise self._closed(exc) from None

    async def recv(self) -> WireMessage:
        try:
            return await self._ws.recv()
        except self._closed_exc as exc:
            raise self._closed(exc) from None

    async def close(self) -> None:
        await self._ws.close()


def make_websocket_factory(max_message_bytes: int = 4 * 1024 * 1024) -> ConnectFactory:
    """Default factory. ``websockets`` is imported only when a session is opened."""

    async def _connect(url: str, headers: Dict[str, str], open_timeout: float) -> Any:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

        try:
            ws = await connect(
                url,
                additional_headers=headers,
                open_timeout=open_timeout,
                max_size=max_message_bytes,
                compression=None,
                logger=_wire_logger,
            )
        except InvalidStatus as exc:
            raise HandshakeRejectedError(exc.response.status_code) from None
        except (InvalidHandshake, OSError) as exc:
            # Exception text can include the URL; only the type name leaves here.
            raise SocketConnectError(type(exc).__name__) from None
        return _WebsocketsConnection(ws, ConnectionClosed)

    return _connect


# ── URLs, redaction, error helpers ───────────────────────────────────────────


def validate_endpoint(endpoint: str) -> str:
    """Return the trimmed endpoint or raise ``ValueError``.

    ``wss://`` is required except for loopback hosts, and credentials embedded in the
    URL are rejected so they can never reach a log line or a proxy.
    """
    if not endpoint or not isinstance(endpoint, str):
        raise ValueError("endpoint")
    try:
        parts = urlsplit(endpoint.strip())
        hostname = parts.hostname
        has_userinfo = parts.username is not None or parts.password is not None
    except ValueError:
        raise ValueError("endpoint") from None
    scheme = parts.scheme.lower()
    if not hostname or has_userinfo or parts.fragment:
        raise ValueError("endpoint")
    if scheme not in ("wss", "ws") or (scheme == "ws" and hostname not in LOOPBACK_HOSTS):
        raise ValueError("endpoint")
    return endpoint.strip()


def build_url(endpoint: str, params: Sequence[Tuple[str, str]]) -> str:
    """Append query parameters to a validated endpoint, keeping any existing ones."""
    parts = urlsplit(validate_endpoint(endpoint))
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.extend(params)
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, urlencode(query), ""))


def redact_url(url: str) -> str:
    """Scheme, host, port, and path only: safe for logs."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<invalid endpoint>"
    return f"{parts.scheme}://{host}{port}{parts.path}"


def redact_text(text: str, secrets: Sequence[str] = ()) -> str:
    """Mask known secret values, auth header values, and ``key=``-style fragments."""
    result = text or ""
    for secret in secrets:
        if secret and len(secret) >= 4:
            result = result.replace(secret, REDACTED)
    result = _AUTH_VALUE_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", result)
    return _SECRET_FRAGMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", result)


def sanitize_code(value: Any) -> Optional[str]:
    """A provider error identifier if it is short and plain, else ``None``."""
    if isinstance(value, str) and _SAFE_CODE_RE.match(value):
        return value
    return None


_RETRYABLE = frozenset(
    {
        ProviderErrorCode.RATE_LIMITED,
        ProviderErrorCode.NETWORK_UNAVAILABLE,
        ProviderErrorCode.TIMEOUT,
        ProviderErrorCode.CONNECTION_CLOSED,
        ProviderErrorCode.INTERNAL_ERROR,
    }
)

_MESSAGES: Dict[ProviderErrorCode, str] = {
    ProviderErrorCode.AUTH_FAILED: "{label} rejected the credentials.",
    ProviderErrorCode.RATE_LIMITED: "{label} rate or concurrency limit reached.",
    ProviderErrorCode.QUOTA_EXCEEDED: "{label} quota or account balance exhausted.",
    ProviderErrorCode.NETWORK_UNAVAILABLE: "{label} endpoint is unreachable.",
    ProviderErrorCode.TIMEOUT: "{label} did not respond in time.",
    ProviderErrorCode.CONNECTION_CLOSED: "{label} connection closed.",
    ProviderErrorCode.INVALID_REQUEST: "{label} request or configuration is invalid.",
    ProviderErrorCode.INVALID_RESPONSE: "{label} returned an unexpected response.",
    ProviderErrorCode.MODEL_UNAVAILABLE: "{label} model is unavailable.",
    ProviderErrorCode.UNSUPPORTED_LANGUAGE: "{label} does not support the language.",
    ProviderErrorCode.INTERNAL_ERROR: "{label} reported an internal error.",
    ProviderErrorCode.CANCELLED: "{label} session was cancelled.",
}


def make_stt_error(
    label: str,
    code: ProviderErrorCode,
    *,
    retryable: Optional[bool] = None,
    details: Optional[Dict[str, Any]] = None,
) -> ProviderError:
    """Canonical STT error with a fixed, non-leaking message."""
    clean = {key: value for key, value in (details or {}).items() if value is not None}
    return ProviderError(
        code=code,
        capability=ProviderCapability.STT,
        message=_MESSAGES.get(code, "{label} error.").format(label=label),
        retryable=code in _RETRYABLE if retryable is None else retryable,
        details=clean or None,
    )


def classify_handshake_status(status: Optional[int]) -> Tuple[ProviderErrorCode, bool]:
    """Map a rejected WebSocket upgrade to a canonical code and retryability."""
    if status is None:
        return ProviderErrorCode.NETWORK_UNAVAILABLE, True
    if status in (401, 403):
        return ProviderErrorCode.AUTH_FAILED, False
    if status == 402:
        return ProviderErrorCode.QUOTA_EXCEEDED, False
    if status == 408:
        return ProviderErrorCode.TIMEOUT, True
    if status == 429:
        return ProviderErrorCode.RATE_LIMITED, True
    if status in (400, 404, 405, 409, 410, 413, 415, 422, 426):
        return ProviderErrorCode.INVALID_REQUEST, False
    if 500 <= status < 600:
        return ProviderErrorCode.INTERNAL_ERROR, True
    return ProviderErrorCode.INVALID_RESPONSE, False


async def open_connection(
    factory: ConnectFactory, url: str, headers: Dict[str, str], timeout: float
) -> SocketConnection:
    """Open a socket, raising only transport-neutral exceptions.

    Raises ``HandshakeRejectedError``, ``SocketClosedError``, ``SocketConnectError``, or
    ``asyncio.TimeoutError``. Unknown failures are reduced to their type name.
    """
    try:
        return await asyncio.wait_for(factory(url, headers, timeout), timeout)
    except (HandshakeRejectedError, SocketClosedError, SocketConnectError, asyncio.TimeoutError):
        raise
    except Exception as exc:
        raise SocketConnectError(type(exc).__name__) from None


# ── Streaming socket ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SendQueueState:
    """Snapshot of a socket's outbound audio queue."""

    queued_audio: int
    dropped_audio: int
    backpressured: bool
    audio_bytes_dequeued: int


MessageHandler = Callable[[WireMessage], None]
CloseHandler = Callable[[SocketClosedError], None]


class StreamingSocket:
    """One open connection with a bounded outbox and background send/receive tasks.

    - ``enqueue_audio`` never blocks. At ``max_queued_audio`` the oldest queued audio is
      dropped and the socket reports backpressure until the queue falls to half.
    - ``enqueue_control`` is never dropped; it keeps its order relative to audio.
    - Every inbound frame goes to ``on_message`` on the receive task. ``on_closed`` is
      called exactly once, when the connection ends for any reason other than
      ``close()`` being called first.
    - With ``keepalive_message`` set, it is sent whenever nothing was sent for
      ``keepalive_interval_s`` according to ``clock``.
    """

    def __init__(
        self,
        conn: SocketConnection,
        *,
        label: str,
        on_message: MessageHandler,
        on_closed: CloseHandler,
        max_queued_audio: int,
        keepalive_message: Optional[str] = None,
        keepalive_interval_s: Optional[float] = None,
        keepalive_poll_s: float = 0.5,
        close_timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if max_queued_audio < 1:
            raise ValueError("max_queued_audio")
        if keepalive_message is not None and (
            keepalive_interval_s is None or keepalive_interval_s <= 0
        ):
            raise ValueError("keepalive_interval_s")
        self._conn = conn
        self._label = label
        self._on_message = on_message
        self._on_closed = on_closed
        self._max_audio = max_queued_audio
        self._keepalive_message = keepalive_message
        self._keepalive_interval_s = keepalive_interval_s or 0.0
        self._keepalive_poll_s = keepalive_poll_s
        self._close_timeout_s = close_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._outbox: Deque[Tuple[bool, WireMessage]] = deque()
        self._audio_count = 0
        self._dropped = 0
        self._backpressured = False
        self._audio_bytes_dequeued = 0
        self._wakeup = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = asyncio.Event()
        self._tasks: Set[asyncio.Task[Any]] = set()
        self._started = False
        self._local_close = False
        self._last_send = clock()
        self.close_code: Optional[int] = None
        self.close_reason = ""
        self.keepalives_sent = 0

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._last_send = self._clock()
        self._spawn(self._send_loop(), "send")
        self._spawn(self._receive_loop(), "recv")
        if self._keepalive_message is not None:
            self._spawn(self._keepalive_loop(), "keepalive")

    def enqueue_audio(self, data: bytes) -> bool:
        """Queue audio without blocking. ``False`` when it was not queued cleanly."""
        if self.closed:
            return False
        accepted = True
        if self._audio_count >= self._max_audio:
            self._drop_oldest_audio()
            accepted = False
        self._outbox.append((True, data))
        self._audio_count += 1
        self._signal()
        return accepted

    def enqueue_control(self, message: WireMessage) -> bool:
        if self.closed:
            return False
        self._outbox.append((False, message))
        self._signal()
        return True

    def clear_audio(self) -> int:
        """Remove queued, unsent audio. Control messages stay queued."""
        removed = self._audio_count
        if removed:
            self._outbox = deque(item for item in self._outbox if not item[0])
            self._audio_count = 0
            self._backpressured = False
        return removed

    def state(self) -> SendQueueState:
        return SendQueueState(
            queued_audio=self._audio_count,
            dropped_audio=self._dropped,
            backpressured=self._backpressured,
            audio_bytes_dequeued=self._audio_bytes_dequeued,
        )

    async def drain(self, timeout: float) -> bool:
        """Wait until the outbox is flushed or the socket closes."""
        if self.closed or (not self._outbox and self._idle.is_set()):
            return True
        return await self._wait_event(self._idle, timeout)

    async def wait_closed(self, timeout: float) -> bool:
        return await self._wait_event(self._closed, timeout)

    async def close(self) -> None:
        """Close the connection and cancel every task. Idempotent and cancel-safe."""
        if not self._closed.is_set():
            self._local_close = True
            self._mark_closed(None)
        try:
            await self._close_connection()
        finally:
            await self._cancel_tasks()

    # ── Internals ────────────────────────────────────────────────────────────

    def _signal(self) -> None:
        self._idle.clear()
        self._wakeup.set()

    def _drop_oldest_audio(self) -> None:
        for index, (is_audio, _) in enumerate(self._outbox):
            if is_audio:
                del self._outbox[index]
                self._audio_count -= 1
                self._dropped += 1
                break
        if not self._backpressured:
            logger.warning(
                "Streaming socket outbox full, dropping oldest audio: socket=%s", self._label
            )
        elif self._dropped % 100 == 1:
            logger.debug(
                "Streaming socket still dropping audio: socket=%s dropped=%d",
                self._label,
                self._dropped,
            )
        self._backpressured = True

    async def _send_loop(self) -> None:
        while not self._closed.is_set():
            if not self._outbox:
                self._idle.set()
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            is_audio, message = self._outbox.popleft()
            if is_audio:
                self._audio_count -= 1
                self._audio_bytes_dequeued += len(message)
            try:
                await self._conn.send(message)
            except SocketClosedError:
                # The receive task reports the close; nothing more can be sent.
                self._outbox.clear()
                self._audio_count = 0
                self._idle.set()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(
                    "Streaming socket send failed: socket=%s error=%s",
                    self._label,
                    type(exc).__name__,
                )
                continue
            self._last_send = self._clock()
            if self._backpressured and self._audio_count <= self._max_audio // 2:
                self._backpressured = False
        self._idle.set()

    async def _receive_loop(self) -> None:
        try:
            while True:
                message = await self._conn.recv()
                self._on_message(message)
        except SocketClosedError as exc:
            self._mark_closed(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Streaming socket receive failed: socket=%s error=%s",
                self._label,
                type(exc).__name__,
            )
            self._mark_closed(SocketClosedError(None, ""))
            self._spawn(self._close_connection(), "abort")

    async def _keepalive_loop(self) -> None:
        message = self._keepalive_message
        assert message is not None
        while not self._closed.is_set():
            await self._sleep(self._keepalive_poll_s)
            if self._closed.is_set():
                return
            if self._outbox:
                continue
            if self._clock() - self._last_send >= self._keepalive_interval_s:
                # Stamp now so a slow send does not queue a second keepalive.
                self._last_send = self._clock()
                self.keepalives_sent += 1
                self.enqueue_control(message)

    def _mark_closed(self, exc: Optional[SocketClosedError]) -> None:
        if self._closed.is_set():
            return
        if exc is not None:
            self.close_code = exc.code
            self.close_reason = exc.reason
        self._closed.set()
        self._idle.set()
        self._wakeup.set()
        if exc is None or self._local_close:
            return
        try:
            self._on_closed(exc)
        except Exception as handler_exc:
            logger.error(
                "Streaming socket close handler failed: socket=%s error=%s",
                self._label,
                type(handler_exc).__name__,
            )

    async def _close_connection(self) -> None:
        try:
            await asyncio.wait_for(self._conn.close(), self._close_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Streaming socket close failed: socket=%s error=%s",
                self._label,
                type(exc).__name__,
            )

    async def _cancel_tasks(self) -> None:
        current = asyncio.current_task()
        while True:
            pending = [task for task in self._tasks if task is not current]
            if not pending:
                return
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._tasks.difference_update(pending)

    async def _wait_event(self, event: asyncio.Event, timeout: float) -> bool:
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            return event.is_set()
        return True

    def _spawn(self, coro: Coroutine[Any, Any, Any], role: str) -> "asyncio.Task[Any]":
        task = asyncio.ensure_future(coro)
        task.set_name(f"{self._label}-{role}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


# ── Result channel ───────────────────────────────────────────────────────────

_END = object()


class ResultChannel:
    """Queue of canonical results behind ``receive_events``.

    The first ``fail`` wins and ends iteration with ``ProviderRuntimeError`` at the
    point it was published; ``end`` finishes iteration cleanly.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self.failure: Optional[ProviderError] = None

    def reset(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
        self.failure = None

    def publish(self, result: Optional[STTTranscriptResult]) -> None:
        if result is not None:
            self._queue.put_nowait(result)

    def fail(self, error: ProviderError) -> bool:
        if self.failure is not None:
            return False
        self.failure = error
        self._queue.put_nowait(error)
        return True

    def end(self) -> None:
        self._queue.put_nowait(_END)

    async def iterate(self) -> AsyncIterator[STTTranscriptResult]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, ProviderError):
                raise ProviderRuntimeError(item)
            yield item
