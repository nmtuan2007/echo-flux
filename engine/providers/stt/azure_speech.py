"""Azure Speech streaming Speech-to-Text adapter.

Wraps the Azure Speech SDK's continuous recognition behind ``StreamingSTTProvider``. It is
dormant: nothing registers it or wires it into the engine, and it never decides fallback.

SDK. ``azure-cognitiveservices-speech`` is optional and is imported by ``start`` and
``health_check`` only, never at module import, so local-only startup does not need it. A
missing package or native runtime is reported as ``model_unavailable``.

Configuration is validated in ``start``, before any recognizer exists:
- A key is required. A region alone builds ``SpeechConfig(subscription, region)``.
- A standard regional host (``<region>.stt.speech.microsoft.com`` or
  ``<region>.api.cognitive.microsoft.com``) also needs ``region``, and the two must match.
  Keys are region-scoped, so a mismatch would otherwise surface later as an auth failure.
- A Speech resource host (``<name>.cognitiveservices.azure.com``) is accepted. Any other
  host is a custom endpoint and needs ``allow_custom_endpoint`` from the provider manifest.
- Endpoints must be https or wss and must not carry credentials.
- Exactly one of a fixed recognition language or candidate languages. Candidates are full
  locales, one per base language, at most 4 at-start or 10 continuous. Continuous language
  identification needs an explicit endpoint, as the SDK documentation requires.

Threading. Each audio ``stream_id`` gets its own recognizer and push stream, opened on its
first audio. SDK construction, start, and stop block, so they run in the default executor.
SDK callbacks fire on SDK threads: they copy plain values out of the event, put them on a
thread-safe ``queue.Queue``, and wake the loop with ``call_soon_threadsafe``. Every state
change and published event happens on the loop, in callback order.

Mapping, per recognizer session:
- ``recognizing`` -> partial. A repeated hypothesis is skipped.
- ``recognized`` with text -> final with ``is_endpoint``. ``NoMatch`` after a partial emits
  an empty final, so every partial is closed by exactly one final.
- ``canceled`` with ``EndOfStream`` -> nothing. With ``Error`` -> the open partial is
  promoted to final, then a categorized ``ProviderError`` surfaces from ``receive_events``.
  A non-retryable error makes later ``send_audio`` calls raise; after a retryable one the
  stream reopens on its next audio.
- ``session_stopped`` -> completes a finalize. An unexpected stop reopens on next audio.

``finalize_stream`` closes the push stream so Azure flushes the last phrase, waits for the
session to stop (promoting the open partial on timeout), and releases the recognizer.
Revisions and the audio timeline are per stream, so they stay monotonic across recognizers.

Nothing here logs or returns the key, endpoint query strings, audio, transcript text, or the
SDK's free-form ``error_details``.
"""

import asyncio
import importlib
import queue
import re
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Mapping,
    Optional,
    Set,
    Tuple,
)
from urllib.parse import parse_qsl, urlsplit

from engine.core.logging import get_logger
from engine.providers.contracts import StreamingSTTProvider
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

logger = get_logger("providers.stt.azure_speech")

SDK_MODULE = "azure.cognitiveservices.speech"
SDK_PACKAGE = "azure-cognitiveservices-speech"

LANGUAGE_ID_AT_START = "AtStart"
LANGUAGE_ID_CONTINUOUS = "Continuous"
MAX_AT_START_CANDIDATES = 4
MAX_CONTINUOUS_CANDIDATES = 10

ENDPOINT_KIND_REGION = "region"
ENDPOINT_KIND_REGIONAL = "regional"
ENDPOINT_KIND_RESOURCE = "resource"
ENDPOINT_KIND_CUSTOM = "custom"

# Result offsets and durations are in 100-nanosecond ticks.
_TICKS_PER_MS = 10_000.0
_PCM16_WIDTH_BYTES = 2
_REGION_RE = re.compile(r"^[a-z][a-z0-9]{2,39}$")
_LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})+$")
_RESOURCE_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")
_REGIONAL_HOST_SUFFIXES = (".stt.speech.microsoft.com", ".api.cognitive.microsoft.com")
_RESOURCE_HOST_SUFFIX = ".cognitiveservices.azure.com"
_ENDPOINT_SCHEMES = ("https", "wss")
_CREDENTIAL_QUERY_KEYS = frozenset(
    {
        "key",
        "subscription-key",
        "subscription_key",
        "ocp-apim-subscription-key",
        "authorization",
        "token",
        "access_token",
        "auth_token",
    }
)

_SESSION_STARTED = "session_started"
_RECOGNIZING = "recognizing"
_RECOGNIZED = "recognized"
_CANCELED = "canceled"
_SESSION_STOPPED = "session_stopped"
_CONNECTED = "connected"
_DISCONNECTED = "disconnected"
_RECOGNIZER_SIGNALS = (_SESSION_STARTED, _RECOGNIZING, _RECOGNIZED, _CANCELED, _SESSION_STOPPED)
_CONNECTION_SIGNALS = (_CONNECTED, _DISCONNECTED)

_CANCEL_ERROR = "error"
_CANCEL_END_OF_STREAM = "end_of_stream"
_CANCEL_OTHER = "other"

_STOP = object()

SdkLoader = Callable[[], Any]
Clock = Callable[[], float]


def import_speech_sdk() -> Any:
    """Default SDK loader; only called when a provider starts or runs a health check."""
    return importlib.import_module(SDK_MODULE)


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


# ── Configuration ────────────────────────────────────────────────────────────


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class AzureSpeechSTTConfig:
    """Resolved Azure Speech STT settings. No region, endpoint, or language is defaulted."""

    api_key: str = field(default="", repr=False)
    region: str = ""
    # Validation rejects credential-bearing endpoints, but keep it out of repr regardless.
    endpoint: str = field(default="", repr=False)
    allow_custom_endpoint: bool = False
    recognition_language: str = ""
    candidate_languages: Tuple[str, ...] = ()
    language_id_mode: str = LANGUAGE_ID_AT_START
    start_timeout_s: float = 10.0
    stop_timeout_s: float = 5.0
    finalize_timeout_s: float = 5.0
    health_timeout_s: float = 5.0
    provider_id: str = "azure-speech"

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "AzureSpeechSTTConfig":
        """Build from a resolved service config mapping. Unknown keys are ignored."""
        values: Dict[str, Any] = {}
        simple = {
            "api_key": "api_key",
            "region": "region",
            "endpoint": "endpoint",
            "allow_custom_endpoint": "allow_custom_endpoint",
            "language": "recognition_language",
            "recognition_language": "recognition_language",
            "language_id_mode": "language_id_mode",
            "start_timeout_s": "start_timeout_s",
            "stop_timeout_s": "stop_timeout_s",
            "finalize_timeout_s": "finalize_timeout_s",
            "health_timeout_s": "health_timeout_s",
            "provider_id": "provider_id",
        }
        for key, attr in simple.items():
            if options.get(key) not in (None, ""):
                values[attr] = options[key]
        candidates = options.get("candidate_languages", options.get("language_hints"))
        if candidates is not None:
            values["candidate_languages"] = _as_tuple(candidates)
        return cls(**values)


@dataclass(frozen=True)
class AzureSpeechTarget:
    """Validated connection target. ``endpoint`` is empty when the region alone is used."""

    region: str
    endpoint: str
    endpoint_kind: str
    recognition_language: str
    candidate_languages: Tuple[str, ...]
    continuous_language_id: bool

    def describe(self) -> str:
        """Log-safe summary: region and host only, never the path or query."""
        host = urlsplit(self.endpoint).hostname if self.endpoint else None
        return f"kind={self.endpoint_kind} region={self.region or '-'} host={host or '-'}"


def validate_config(config: AzureSpeechSTTConfig) -> AzureSpeechTarget:
    """Check credentials, region/endpoint, and languages. Raises ``ProviderRuntimeError``."""
    if not config.api_key.strip():
        raise _error(ProviderErrorCode.AUTH_FAILED, "Azure Speech credential is missing.")
    region = config.region.strip().lower()
    if region and not _REGION_RE.match(region):
        raise _error(
            ProviderErrorCode.INVALID_REQUEST, "Azure Speech region is not a valid region name."
        )
    endpoint = config.endpoint.strip()
    if endpoint:
        kind = _validate_endpoint(endpoint, region, config.allow_custom_endpoint)
    elif region:
        kind = ENDPOINT_KIND_REGION
    else:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST, "Azure Speech needs a region or an endpoint."
        )
    for name in ("start_timeout_s", "stop_timeout_s", "finalize_timeout_s", "health_timeout_s"):
        if float(getattr(config, name)) <= 0:
            raise _error(
                ProviderErrorCode.INVALID_REQUEST, "Azure Speech timeouts must be positive."
            )
    language, candidates, continuous = _validate_languages(config, bool(endpoint))
    return AzureSpeechTarget(
        region=region,
        endpoint=endpoint,
        endpoint_kind=kind,
        recognition_language=language,
        candidate_languages=candidates,
        continuous_language_id=continuous,
    )


def _validate_endpoint(endpoint: str, region: str, allow_custom: bool) -> str:
    try:
        parts = urlsplit(endpoint)
        host = (parts.hostname or "").lower()
        _ = parts.port
    except ValueError:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST, "Azure Speech endpoint is not a valid URL."
        ) from None
    if parts.scheme.lower() not in _ENDPOINT_SCHEMES or not host:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST, "Azure Speech endpoint must be an https or wss URL."
        )
    has_query_credential = any(
        name.lower() in _CREDENTIAL_QUERY_KEYS
        for name, _ in parse_qsl(parts.query, keep_blank_values=True)
    )
    if parts.username is not None or parts.password is not None or has_query_credential:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST, "Azure Speech endpoint must not embed credentials."
        )
    if parts.fragment:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST, "Azure Speech endpoint must not have a fragment."
        )

    for suffix in _REGIONAL_HOST_SUFFIXES:
        if host.endswith(suffix):
            host_region = host[: -len(suffix)]
            if not _REGION_RE.match(host_region):
                raise _error(
                    ProviderErrorCode.INVALID_REQUEST,
                    "Azure Speech endpoint host does not name a region.",
                )
            if not region:
                raise _error(
                    ProviderErrorCode.INVALID_REQUEST,
                    "Azure Speech region is required for a regional endpoint.",
                )
            if region != host_region:
                raise _error(
                    ProviderErrorCode.INVALID_REQUEST,
                    "Azure Speech region does not match the endpoint region.",
                    details={"region": region, "endpoint_region": host_region},
                )
            return ENDPOINT_KIND_REGIONAL

    if host.endswith(_RESOURCE_HOST_SUFFIX) and _RESOURCE_NAME_RE.match(
        host[: -len(_RESOURCE_HOST_SUFFIX)]
    ):
        return ENDPOINT_KIND_RESOURCE
    if not allow_custom:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Custom Azure Speech endpoints are not enabled for this provider.",
        )
    return ENDPOINT_KIND_CUSTOM


def _validate_languages(
    config: AzureSpeechSTTConfig, has_endpoint: bool
) -> Tuple[str, Tuple[str, ...], bool]:
    mode = config.language_id_mode
    if mode not in (LANGUAGE_ID_AT_START, LANGUAGE_ID_CONTINUOUS):
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Azure Speech language identification mode must be AtStart or Continuous.",
        )
    language = config.recognition_language.strip()
    candidates = tuple(item.strip() for item in config.candidate_languages if item.strip())
    if language and candidates:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Set either a recognition language or candidate languages, not both.",
        )
    if not language and not candidates:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Azure Speech needs a recognition language or candidate languages.",
        )
    for locale in candidates or (language,):
        if not _LOCALE_RE.match(locale):
            raise _error(
                ProviderErrorCode.UNSUPPORTED_LANGUAGE,
                "Azure Speech languages must be full locales such as en-US.",
            )
    if not candidates:
        return language, (), False

    continuous = mode == LANGUAGE_ID_CONTINUOUS
    limit = MAX_CONTINUOUS_CANDIDATES if continuous else MAX_AT_START_CANDIDATES
    if len(candidates) > limit:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            f"Azure Speech allows at most {limit} candidate languages in {mode} mode.",
        )
    bases = [locale.split("-", 1)[0].lower() for locale in candidates]
    if len(set(bases)) != len(bases):
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Use one locale per base language in Azure Speech candidate languages.",
        )
    if continuous and not has_endpoint:
        raise _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Continuous language identification requires an Azure Speech endpoint.",
        )
    return "", candidates, continuous


# ── Error mapping ────────────────────────────────────────────────────────────

_CANCELLATION_ERRORS: Dict[str, Tuple[ProviderErrorCode, bool, str]] = {
    "AuthenticationFailure": (
        ProviderErrorCode.AUTH_FAILED,
        False,
        "Azure Speech rejected the credential or region.",
    ),
    "Forbidden": (ProviderErrorCode.AUTH_FAILED, False, "Azure Speech denied access."),
    "TooManyRequests": (
        ProviderErrorCode.RATE_LIMITED,
        True,
        "Azure Speech is rate limiting requests.",
    ),
    "BadRequest": (
        ProviderErrorCode.INVALID_REQUEST,
        False,
        "Azure Speech rejected the recognition request.",
    ),
    "ConnectionFailure": (
        ProviderErrorCode.NETWORK_UNAVAILABLE,
        True,
        "Could not connect to Azure Speech.",
    ),
    "ServiceTimeout": (ProviderErrorCode.TIMEOUT, True, "Azure Speech timed out."),
    "ServiceUnavailable": (
        ProviderErrorCode.NETWORK_UNAVAILABLE,
        True,
        "Azure Speech is unavailable.",
    ),
    "ServiceError": (
        ProviderErrorCode.INTERNAL_ERROR,
        True,
        "Azure Speech reported a service error.",
    ),
    "ServiceRedirectTemporary": (
        ProviderErrorCode.CONNECTION_CLOSED,
        True,
        "Azure Speech redirected the connection.",
    ),
    "ServiceRedirectPermanent": (
        ProviderErrorCode.CONNECTION_CLOSED,
        False,
        "Azure Speech redirected the connection.",
    ),
    "EmbeddedModelError": (
        ProviderErrorCode.MODEL_UNAVAILABLE,
        False,
        "Azure Speech model is unavailable.",
    ),
    "RuntimeError": (ProviderErrorCode.INTERNAL_ERROR, False, "Azure Speech SDK failed."),
}


def map_cancellation_error(error_code: str) -> ProviderError:
    """Map a ``CancellationErrorCode`` member name to a categorized provider error."""
    code, retryable, message = _CANCELLATION_ERRORS.get(
        error_code,
        (ProviderErrorCode.INTERNAL_ERROR, False, "Azure Speech canceled recognition."),
    )
    return ProviderError(
        code=code,
        capability=ProviderCapability.STT,
        message=message,
        retryable=retryable,
        details={"provider_code": error_code or "unknown"},
    )


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None) or str(value or "").rsplit(".", 1)[-1]
    # Only an identifier leaves here; anything else could be free-form SDK text.
    return name if re.fullmatch(r"[A-Za-z]{1,64}", name or "") else ""


# ── SDK callback bridge ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Signal:
    """Plain copy of an SDK event, safe to hand from an SDK thread to the loop."""

    kind: str
    generation: int
    text: str = ""
    offset_ticks: int = 0
    duration_ticks: int = 0
    language: Optional[str] = None
    no_match: bool = False
    cancel_reason: str = ""
    error_code: str = ""


def _detected_language(sdk: Any, result: Any) -> Optional[str]:
    try:
        language = sdk.AutoDetectSourceLanguageResult(result).language
    except Exception:
        return None
    return str(language) if language else None


def _extract_signal(
    sdk: Any, kind: str, generation: int, evt: Any, detect_language: bool
) -> _Signal:
    if kind in (_RECOGNIZING, _RECOGNIZED):
        result = evt.result
        return _Signal(
            kind=kind,
            generation=generation,
            text=str(result.text or ""),
            offset_ticks=int(result.offset or 0),
            duration_ticks=int(result.duration or 0),
            language=_detected_language(sdk, result) if detect_language else None,
            no_match=result.reason == sdk.ResultReason.NoMatch,
        )
    if kind == _CANCELED:
        details = evt.cancellation_details
        if details.reason == sdk.CancellationReason.Error:
            return _Signal(
                kind=kind,
                generation=generation,
                cancel_reason=_CANCEL_ERROR,
                error_code=_enum_name(getattr(details, "code", None)),
            )
        if details.reason == sdk.CancellationReason.EndOfStream:
            return _Signal(kind=kind, generation=generation, cancel_reason=_CANCEL_END_OF_STREAM)
        return _Signal(kind=kind, generation=generation, cancel_reason=_CANCEL_OTHER)
    return _Signal(kind=kind, generation=generation)


class _CallbackBridge:
    """Carries signals from SDK threads to the event loop, preserving arrival order."""

    def __init__(self, loop: asyncio.AbstractEventLoop, handler: Callable[[_Signal], None]) -> None:
        self._loop = loop
        self._handler = handler
        self._inbox: queue.Queue[_Signal] = queue.Queue()
        self._closed = False

    def post(self, signal: _Signal) -> None:
        """Runs on an SDK thread: enqueue and wake the loop, nothing else."""
        if self._closed:
            return
        self._inbox.put_nowait(signal)
        try:
            self._loop.call_soon_threadsafe(self._drain)
        except RuntimeError:
            # The loop already closed during shutdown; late SDK events are dropped.
            self._closed = True

    def _drain(self) -> None:
        while not self._closed:
            try:
                signal = self._inbox.get_nowait()
            except queue.Empty:
                return
            try:
                self._handler(signal)
            except Exception as exc:
                logger.error("Azure Speech event handling failed: %s", type(exc).__name__)

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                return


def _make_callback(
    bridge: _CallbackBridge, sdk: Any, kind: str, generation: int, detect_language: bool
) -> Callable[[Any], None]:
    def _callback(evt: Any) -> None:
        # SDK thread. Exceptions must not propagate back into the SDK.
        try:
            signal = _extract_signal(sdk, kind, generation, evt, detect_language)
        except Exception as exc:
            logger.debug(
                "Azure Speech event unreadable: kind=%s error=%s", kind, type(exc).__name__
            )
            reason = _CANCEL_ERROR if kind == _CANCELED else ""
            signal = _Signal(kind=kind, generation=generation, cancel_reason=reason)
        bridge.post(signal)

    return _callback


def _quietly(step: Callable[[], Any]) -> None:
    try:
        step()
    except Exception as exc:
        logger.debug("Azure Speech cleanup step failed: %s", type(exc).__name__)


# ── Session state ────────────────────────────────────────────────────────────


@dataclass(eq=False)
class _Session:
    stream_id: str
    generation: int
    sample_rate: int
    base_ms: float
    health: bool = False
    recognizer: Any = None
    push_stream: Any = None
    connection: Any = None
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    health_error: Optional[ProviderError] = None
    last_sequence: int = -1
    segment_index: int = 0
    segment_open: bool = False
    last_interim_text: str = ""
    last_interim_language: Optional[str] = None
    last_interim_start_ms: float = 0.0
    last_interim_end_ms: float = 0.0
    input_closed: bool = False
    discarded: bool = False
    ended: bool = False
    released: bool = False


@dataclass(eq=False)
class _StreamState:
    stream_id: str
    active: Optional[_Session] = None
    revision: int = 0
    audio_ms: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _pcm_ms(byte_count: int, sample_rate: int) -> float:
    return byte_count / (_PCM16_WIDTH_BYTES * sample_rate) * 1000.0


# ── Provider ─────────────────────────────────────────────────────────────────


class AzureSpeechSTTProvider(StreamingSTTProvider):
    """Streaming STT over Azure Speech continuous recognition.

    Configuration and SDK problems raise from ``start``. Recognizer start failures raise
    from ``send_audio``. Failures reported by the service surface from ``receive_events``.
    """

    def __init__(
        self,
        config: AzureSpeechSTTConfig,
        sdk_loader: Optional[SdkLoader] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._config = config
        self._sdk_loader: SdkLoader = sdk_loader or import_speech_sdk
        self._clock: Clock = clock or time.monotonic
        self._sdk: Any = None
        self._target: Optional[AzureSpeechTarget] = None
        self._bridge: Optional[_CallbackBridge] = None
        self._events: Optional[asyncio.Queue] = None
        self._streams: Dict[str, _StreamState] = {}
        self._sessions: Dict[int, _Session] = {}
        self._generation = 0
        self._tasks: Set[asyncio.Task] = set()
        self._started = False
        self._stopping = False
        self._failure: Optional[ProviderError] = None

    @property
    def provider_id(self) -> str:
        return self._config.provider_id

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._started:
            return
        target = validate_config(self._config)
        sdk = await self._load_sdk()
        self._target = target
        self._bridge = _CallbackBridge(asyncio.get_running_loop(), self._on_signal)
        self._events = asyncio.Queue()
        self._failure = None
        self._stopping = False
        self._started = True
        self._sdk = sdk
        logger.info("Azure Speech STT ready: provider=%s %s", self.provider_id, target.describe())

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        try:
            finishing = []
            for state in self._streams.values():
                session = self._detach(state)
                if session is not None:
                    finishing.append(self._finish(session, graceful=True))
            if finishing:
                await asyncio.gather(*finishing, return_exceptions=True)
            while self._tasks:
                pending = list(self._tasks)
                await asyncio.gather(*pending, return_exceptions=True)
                self._tasks.difference_update(pending)
            for session in list(self._sessions.values()):
                await self._release(session)
        finally:
            if self._bridge is not None:
                self._bridge.close()
                self._bridge = None
            self._streams.clear()
            self._sessions.clear()
            self._started = False
            if self._events is not None:
                self._events.put_nowait(_STOP)
        logger.info("Azure Speech STT stopped: provider=%s", self.provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        began = self._clock()
        try:
            target = validate_config(self._config)
            sdk = await self._load_sdk()
        except ProviderRuntimeError as exc:
            return self._unhealthy(exc.error)

        owns_bridge = self._bridge is None
        if owns_bridge:
            self._bridge = _CallbackBridge(asyncio.get_running_loop(), self._on_signal)
        session = self._new_session("health-check", 16000, 0.0, health=True)
        try:
            await self._build(session, sdk, target, with_connection=True)
            connection = session.connection
            await self._run_blocking(lambda: connection.open(True), self._config.health_timeout_s)
            await asyncio.wait_for(session.connected.wait(), self._config.health_timeout_s)
        except asyncio.TimeoutError:
            return self._unhealthy(
                _error(
                    ProviderErrorCode.TIMEOUT, "Azure Speech did not confirm the connection.", True
                ).error
            )
        except ProviderRuntimeError as exc:
            return self._unhealthy(exc.error)
        except Exception as exc:
            logger.warning("Azure Speech health check failed: %s", type(exc).__name__)
            return self._unhealthy(
                _error(
                    ProviderErrorCode.NETWORK_UNAVAILABLE,
                    "Could not connect to Azure Speech.",
                    True,
                ).error
            )
        finally:
            await self._release(session)
            if owns_bridge and self._bridge is not None and not self._started:
                self._bridge.close()
                self._bridge = None

        if session.health_error is not None:
            return self._unhealthy(session.health_error)
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Azure Speech accepted the connection.",
            latency_ms=(self._clock() - began) * 1000.0,
            details={"endpoint_kind": target.endpoint_kind},
        )

    # ── Streaming contract ───────────────────────────────────────────────────

    async def send_audio(self, frame: AudioFrame) -> None:
        self._require_started()
        self._validate_frame(frame)
        if not frame.pcm_data:
            return
        state = self._stream(frame.stream_id)
        async with state.lock:
            self._require_started()
            session = state.active
            if session is not None and session.sample_rate != frame.sample_rate:
                self._detach(state)
                await self._finish(session, graceful=True)
                session = None
            if session is None:
                session = await self._open_session(state, frame.sample_rate)
                state.active = session
            try:
                # write() copies into the SDK's buffer; it does not wait on the network.
                session.push_stream.write(frame.pcm_data)
            except Exception as exc:
                logger.warning(
                    "Azure Speech audio write failed: stream=%s error=%s",
                    state.stream_id,
                    type(exc).__name__,
                )
                self._detach(state)
                session.ended = True
                self._spawn(self._release(session))
                raise _error(
                    ProviderErrorCode.CONNECTION_CLOSED,
                    "Azure Speech audio stream is closed.",
                    True,
                ) from None
            session.last_sequence = frame.sequence
            state.audio_ms += _pcm_ms(len(frame.pcm_data), frame.sample_rate)

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        queue_ = self._events
        if queue_ is None:
            raise _error(ProviderErrorCode.INVALID_REQUEST, "Azure Speech STT is not started.")
        while True:
            item = await queue_.get()
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
            session = self._detach(state)
            if session is not None:
                await self._finish(session, graceful=True)

    async def reset_stream(self, stream_id: str) -> None:
        """Drop the stream's recognizer and any open partial without emitting a final.

        The revision counter and audio timeline are kept so results stay monotonic.
        """
        state = self._streams.get(stream_id)
        if state is None:
            return
        async with state.lock:
            session = self._detach(state)
            if session is not None:
                session.discarded = True
                await self._release(session)

    # ── Recognizer handling ──────────────────────────────────────────────────

    async def _load_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        loop = asyncio.get_running_loop()
        try:
            sdk = await loop.run_in_executor(None, self._sdk_loader)
        except (ImportError, OSError) as exc:
            # OSError covers a present package whose native runtime cannot load.
            logger.warning("Azure Speech SDK unavailable: %s", type(exc).__name__)
            raise _error(
                ProviderErrorCode.MODEL_UNAVAILABLE,
                "Azure Speech SDK is not installed.",
                False,
                {"dependency": SDK_PACKAGE},
            ) from None
        self._sdk = sdk
        return sdk

    def _new_session(
        self, stream_id: str, sample_rate: int, base_ms: float, health: bool = False
    ) -> _Session:
        self._generation += 1
        session = _Session(
            stream_id=stream_id,
            generation=self._generation,
            sample_rate=sample_rate,
            base_ms=base_ms,
            health=health,
        )
        self._sessions[session.generation] = session
        return session

    async def _open_session(self, state: _StreamState, sample_rate: int) -> _Session:
        assert self._target is not None
        session = self._new_session(state.stream_id, sample_rate, state.audio_ms)
        try:
            await self._build(session, self._sdk, self._target)
            recognizer = session.recognizer
            await self._run_blocking(
                lambda: recognizer.start_continuous_recognition_async().get(),
                self._config.start_timeout_s,
            )
        except asyncio.CancelledError:
            self._spawn(self._release(session))
            raise
        except ProviderRuntimeError:
            await self._release(session)
            raise
        except asyncio.TimeoutError:
            await self._release(session)
            raise _error(
                ProviderErrorCode.TIMEOUT, "Azure Speech did not start recognition in time.", True
            ) from None
        except Exception as exc:
            await self._release(session)
            logger.warning(
                "Azure Speech recognition failed to start: stream=%s error=%s",
                state.stream_id,
                type(exc).__name__,
            )
            raise _error(
                ProviderErrorCode.INTERNAL_ERROR,
                "Azure Speech could not start recognition.",
                False,
                {"exception": type(exc).__name__},
            ) from None
        logger.info(
            "Azure Speech recognition started: stream=%s generation=%d rate=%d",
            state.stream_id,
            session.generation,
            sample_rate,
        )
        return session

    async def _build(
        self,
        session: _Session,
        sdk: Any,
        target: AzureSpeechTarget,
        with_connection: bool = False,
    ) -> None:
        bridge = self._bridge
        assert bridge is not None
        api_key = self._config.api_key
        detect = bool(target.candidate_languages)
        generation = session.generation
        sample_rate = session.sample_rate

        def build() -> Tuple[Any, Any, Any]:
            if target.endpoint:
                speech_config = sdk.SpeechConfig(subscription=api_key, endpoint=target.endpoint)
            else:
                speech_config = sdk.SpeechConfig(subscription=api_key, region=target.region)
            if target.recognition_language:
                speech_config.speech_recognition_language = target.recognition_language
            if target.continuous_language_id:
                speech_config.set_property(
                    property_id=sdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                    value=LANGUAGE_ID_CONTINUOUS,
                )
            stream_format = sdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate, bits_per_sample=16, channels=1
            )
            push_stream = sdk.audio.PushAudioInputStream(stream_format=stream_format)
            kwargs: Dict[str, Any] = {
                "speech_config": speech_config,
                "audio_config": sdk.audio.AudioConfig(stream=push_stream),
            }
            if detect:
                kwargs["auto_detect_source_language_config"] = (
                    sdk.languageconfig.AutoDetectSourceLanguageConfig(
                        languages=list(target.candidate_languages)
                    )
                )
            try:
                recognizer = sdk.SpeechRecognizer(**kwargs)
            except BaseException:
                _quietly(push_stream.close)
                raise
            for name in _RECOGNIZER_SIGNALS:
                getattr(recognizer, name).connect(
                    _make_callback(bridge, sdk, name, generation, detect)
                )
            connection = None
            if with_connection:
                connection = sdk.Connection.from_recognizer(recognizer)
                for name in _CONNECTION_SIGNALS:
                    getattr(connection, name).connect(
                        _make_callback(bridge, sdk, name, generation, False)
                    )
            return recognizer, push_stream, connection

        loop = asyncio.get_running_loop()
        try:
            recognizer, push_stream, connection = await loop.run_in_executor(None, build)
        except Exception as exc:
            logger.warning("Azure Speech recognizer setup failed: %s", type(exc).__name__)
            raise _error(
                ProviderErrorCode.INTERNAL_ERROR,
                "Azure Speech could not create a recognizer.",
                False,
                {"exception": type(exc).__name__},
            ) from None
        session.recognizer = recognizer
        session.push_stream = push_stream
        session.connection = connection

    async def _finish(self, session: _Session, graceful: bool) -> None:
        """End a detached session: flush the last phrase when graceful, then release it."""
        if graceful and not session.ended and not session.released:
            await self._close_input(session)
            try:
                await asyncio.wait_for(session.stopped.wait(), self._config.finalize_timeout_s)
            except asyncio.TimeoutError:
                logger.warning("Azure Speech did not flush stream %s in time", session.stream_id)
            # No recognized result arrived for the open phrase; keep its last hypothesis.
            self._promote_interim(session)
        await self._release(session)

    async def _close_input(self, session: _Session) -> None:
        push_stream = session.push_stream
        if session.input_closed or push_stream is None:
            return
        session.input_closed = True
        try:
            await self._run_blocking(push_stream.close, self._config.stop_timeout_s)
        except Exception as exc:
            logger.debug("Azure Speech push stream close failed: %s", type(exc).__name__)

    async def _release(self, session: _Session) -> None:
        """Close the push stream, stop recognition, and disconnect every callback."""
        if session.released:
            return
        session.released = True
        session.ended = True
        recognizer, push_stream, connection = (
            session.recognizer,
            session.push_stream,
            session.connection,
        )
        session.recognizer = session.push_stream = session.connection = None
        close_input = not session.input_closed
        session.input_closed = True
        stop_recognition = not session.health

        def teardown() -> None:
            if push_stream is not None and close_input:
                _quietly(push_stream.close)
            if connection is not None:
                _quietly(connection.close)
                for name in _CONNECTION_SIGNALS:
                    _quietly(getattr(connection, name).disconnect_all)
            if recognizer is not None:
                if stop_recognition:
                    _quietly(lambda: recognizer.stop_continuous_recognition_async().get())
                for name in _RECOGNIZER_SIGNALS:
                    _quietly(getattr(recognizer, name).disconnect_all)

        try:
            await self._run_blocking(teardown, self._config.stop_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "Azure Speech recognizer did not stop in time: stream=%s", session.stream_id
            )
        except Exception as exc:
            logger.debug("Azure Speech teardown failed: %s", type(exc).__name__)
        finally:
            self._sessions.pop(session.generation, None)
            session.stopped.set()
            session.connected.set()

    # ── SDK events (event loop only) ─────────────────────────────────────────

    def _on_signal(self, signal: _Signal) -> None:
        session = self._sessions.get(signal.generation)
        if session is None or session.released:
            return
        if session.health:
            self._on_health_signal(session, signal)
            return
        kind = signal.kind
        if kind == _SESSION_STOPPED:
            self._on_session_stopped(session)
        elif kind == _SESSION_STARTED:
            logger.debug("Azure Speech session started: stream=%s", session.stream_id)
        elif session.discarded or session.ended:
            return
        elif kind == _RECOGNIZING:
            self._on_recognizing(session, signal)
        elif kind == _RECOGNIZED:
            self._on_recognized(session, signal)
        elif kind == _CANCELED:
            self._on_canceled(session, signal)

    def _on_recognizing(self, session: _Session, signal: _Signal) -> None:
        text = signal.text
        if not text.strip() or text == session.last_interim_text:
            return
        start_ms, end_ms = self._span(session, signal)
        session.segment_open = True
        session.last_interim_text = text
        session.last_interim_language = signal.language
        session.last_interim_start_ms = start_ms
        session.last_interim_end_ms = end_ms
        self._emit(session, text, False, signal.language, start_ms, end_ms)

    def _on_recognized(self, session: _Session, signal: _Signal) -> None:
        if signal.no_match or not signal.text.strip():
            if session.segment_open:
                self._emit(
                    session,
                    "",
                    True,
                    None,
                    session.last_interim_start_ms,
                    session.last_interim_end_ms,
                )
                self._close_segment(session)
            return
        start_ms, end_ms = self._span(session, signal)
        self._emit(session, signal.text, True, signal.language, start_ms, end_ms)
        self._close_segment(session)

    def _on_canceled(self, session: _Session, signal: _Signal) -> None:
        if signal.cancel_reason != _CANCEL_ERROR:
            logger.debug(
                "Azure Speech input ended: stream=%s reason=%s",
                session.stream_id,
                signal.cancel_reason,
            )
            return
        error = map_cancellation_error(signal.error_code)
        self._promote_interim(session)
        self._end_session(session)
        if self._stopping:
            return
        logger.warning(
            "Azure Speech canceled recognition: stream=%s code=%s provider_code=%s retryable=%s",
            session.stream_id,
            error.code.value,
            signal.error_code or "unknown",
            error.retryable,
        )
        if not error.retryable and self._failure is None:
            self._failure = error
        self._publish(error)

    def _on_session_stopped(self, session: _Session) -> None:
        session.stopped.set()
        if session.input_closed or session.ended or session.discarded:
            return
        # Azure ended the session without a finalize; keep the text, reopen on next audio.
        self._promote_interim(session)
        self._end_session(session)
        logger.info(
            "Azure Speech session ended for stream %s; reopening on next audio",
            session.stream_id,
        )

    def _on_health_signal(self, session: _Session, signal: _Signal) -> None:
        if session.connected.is_set():
            return
        if signal.kind == _CONNECTED:
            session.connected.set()
        elif signal.kind == _DISCONNECTED:
            session.health_error = _error(
                ProviderErrorCode.CONNECTION_CLOSED, "Azure Speech closed the connection.", True
            ).error
            session.connected.set()
        elif signal.kind == _CANCELED and signal.cancel_reason == _CANCEL_ERROR:
            session.health_error = map_cancellation_error(signal.error_code)
            session.connected.set()

    def _end_session(self, session: _Session) -> None:
        session.ended = True
        state = self._streams.get(session.stream_id)
        if state is not None and state.active is session:
            state.active = None
        self._spawn(self._release(session))

    def _promote_interim(self, session: _Session) -> None:
        if not session.segment_open or session.discarded:
            return
        self._emit(
            session,
            session.last_interim_text,
            True,
            session.last_interim_language,
            session.last_interim_start_ms,
            session.last_interim_end_ms,
        )
        self._close_segment(session)

    @staticmethod
    def _close_segment(session: _Session) -> None:
        session.segment_index += 1
        session.segment_open = False
        session.last_interim_text = ""
        session.last_interim_language = None

    @staticmethod
    def _span(session: _Session, signal: _Signal) -> Tuple[float, float]:
        start_ms = session.base_ms + max(0, signal.offset_ticks) / _TICKS_PER_MS
        return start_ms, start_ms + max(0, signal.duration_ticks) / _TICKS_PER_MS

    def _emit(
        self,
        session: _Session,
        text: str,
        is_final: bool,
        language: Optional[str],
        start_ms: float,
        end_ms: float,
    ) -> None:
        state = self._stream(session.stream_id)
        state.revision += 1
        self._publish(
            STTTranscriptResult(
                stream_id=session.stream_id,
                sequence=session.last_sequence,
                provider_event_id=(
                    f"azure:{session.stream_id}:{session.generation}:{session.segment_index}"
                ),
                text=text,
                is_final=is_final,
                revision=state.revision,
                start_time_ms=start_ms,
                end_time_ms=max(start_ms, end_ms),
                monotonic_timestamp=self._clock(),
                detected_language=language,
                is_endpoint=is_final,
            )
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detach(state: _StreamState) -> Optional[_Session]:
        session = state.active
        state.active = None
        return session

    @staticmethod
    async def _run_blocking(step: Callable[[], Any], timeout: float) -> Any:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, step), timeout)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> "asyncio.Task[Any]":
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)

        def _done(finished: "asyncio.Task[Any]") -> None:
            self._tasks.discard(finished)
            if not finished.cancelled() and finished.exception() is not None:
                logger.error(
                    "Azure Speech background task failed: %s",
                    type(finished.exception()).__name__,
                )

        task.add_done_callback(_done)
        return task

    def _stream(self, stream_id: str) -> _StreamState:
        state = self._streams.get(stream_id)
        if state is None:
            state = _StreamState(stream_id=stream_id)
            self._streams[stream_id] = state
        return state

    def _publish(self, item: Any) -> None:
        if self._events is not None:
            self._events.put_nowait(item)

    def _require_started(self) -> None:
        if self._failure is not None:
            raise ProviderRuntimeError(self._failure)
        if not self._started or self._stopping or self._events is None:
            raise _error(ProviderErrorCode.INVALID_REQUEST, "Azure Speech STT is not started.")

    @staticmethod
    def _validate_frame(frame: AudioFrame) -> None:
        if frame.channels != 1:
            raise _error(
                ProviderErrorCode.INVALID_REQUEST, "Azure Speech expects mono PCM16 audio."
            )
        if frame.sample_rate <= 0:
            raise _error(ProviderErrorCode.INVALID_REQUEST, "Audio sample rate is invalid.")
        if len(frame.pcm_data) % _PCM16_WIDTH_BYTES:
            raise _error(
                ProviderErrorCode.INVALID_REQUEST, "PCM16 audio must have an even byte length."
            )

    @staticmethod
    def _unhealthy(error: ProviderError) -> ProviderHealthCheck:
        return ProviderHealthCheck(
            status=ProviderHealthStatus.UNHEALTHY,
            message=error.message,
            details={"code": error.code.value, "retryable": error.retryable},
        )
