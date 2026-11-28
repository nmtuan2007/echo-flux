"""Realtime STT session orchestration: one Start-to-Stop run of the provider stack.

Lifecycle::

    resolve_session_config()        worker thread, before any provider or audio device starts
    STTSessionOrchestrator.start()  all-or-nothing; a failure releases everything it touched
        cloud primary: secrets for its credential refs (fail closed)
                       -> Local fallback readiness (health check, no model load)
                       -> one adapter per stream, started, epoch 1, event pump
        local primary: the shared Local provider is started, epoch 1 per stream
    engine opens audio devices      Process-{stream} -> submit_audio() / request_endpoint()
    STTSessionOrchestrator.stop()   exactly once, however often it is called
        stop taking audio -> drain inboxes -> stop every provider once -> drain trailing
        results -> one empty final per stream -> end controllers -> drop buffers and tasks

Threads and tasks. Providers are asyncio adapters and live on the engine event loop. The
``Process-{stream}`` thread calls ``submit_audio`` and ``request_endpoint``: they append to the
stream's ring buffer and a bounded ``queue.Queue`` inbox and never block; a full inbox drops the
frame. One owner task per stream drains that inbox with ``get_nowait`` polling, sends audio to
the stream's active provider, applies ledger decisions to provider results, calls the failover
controller's ``poll()``, and carries out reconnect and Local replay plans. Event pumps, one per
provider instance, only move results into the owning stream's deque. Adapter construction and
secret lookup run through ``asyncio.to_thread``.

Providers. A cloud primary gets one adapter instance per stream, so a failure, reconnect, or
fallback affects that stream alone. Local is one shared instance with per-stream workers (one
Whisper model, as before). Cloud failures follow ``STTFailoverController``: one reconnect within
the deadline, then sticky Local replaying from the last final boundary minus the overlap. Local
failures are reported, never retried. Adapters never choose fallback; this module does.

Audio timeline. Cloud-primary sessions send every captured frame, so a provider's clock is the
session timeline minus its epoch offset and ledger reconciliation stays exact. Audio missing
from an inbox (dropped frames, audio evicted during a slow Local start) is sent as silence of
the same length. A Local-primary session is a single epoch that can never fail over, so it
gates audio with VAD like the legacy pipeline, keeping up to three pre-speech frames, and
finalizes on the engine's silence endpoint.

Nothing here knows the UI. It emits ``STTTranscriptEvent`` and ``STTStatusEvent``; the engine
maps them onto the WebSocket contract. Logs carry ids, codes, and counts, never audio or text.
"""

import asyncio
import dataclasses
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Full, Queue
from types import MappingProxyType
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from engine.core.config import ConfigError, ServiceConfig
from engine.core.logging import get_logger
from engine.providers.contracts import StreamingSTTProvider
from engine.providers.manifests import ProtocolFamily
from engine.providers.registry import (
    ProviderRegistry,
    ProviderRegistryError,
    RegistryErrorCode,
    ResolvedService,
    build_default_registry,
)
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
)
from engine.session.audio_ring_buffer import DEFAULT_CAPACITY_MS, AudioRingBuffer
from engine.session.stt_failover import (
    FailoverAction,
    FailoverDecision,
    FailoverPolicy,
    FailoverState,
    ReplayPlan,
    STTFailoverController,
)
from engine.session.transcript_ledger import (
    ContinuityWarning,
    EpochHandle,
    EpochStart,
    LedgerAction,
    TranscriptLedger,
)

logger = get_logger("session.orchestrator")

# Start config key carrying a v2 ``services.stt`` section. Without it the legacy flat keys apply.
STT_SECTION_KEY = "services.stt"

DEFAULT_POLL_INTERVAL_S = 0.02
MAX_ITEMS_PER_PASS = 10
# The legacy pipeline kept about three pre-speech chunks.
PREROLL_FRAMES = 3
SAMPLE_WIDTH_BYTES = 2

ERROR_INVALID_REQUEST = ProviderErrorCode.INVALID_REQUEST.value
ERROR_INTERNAL = ProviderErrorCode.INTERNAL_ERROR.value
ERROR_TIMEOUT = ProviderErrorCode.TIMEOUT.value
ERROR_MISSING_CREDENTIAL = RegistryErrorCode.MISSING_CREDENTIAL.value
ERROR_STT_DISABLED = "stt_disabled"
ERROR_LOCAL_FALLBACK_UNAVAILABLE = "local_fallback_unavailable"
ERROR_INVALID_STATE = "invalid_state"

_ERROR_MESSAGES = {
    ERROR_STT_DISABLED: "Speech recognition is turned off in settings.",
    ERROR_MISSING_CREDENTIAL: "Speech recognition credentials are not available to the engine.",
    ERROR_LOCAL_FALLBACK_UNAVAILABLE: (
        "Local speech recognition, required as the fallback, is not available."
    ),
}

_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
# Same shape LocalSTTProvider accepts for a language hint.
_LOCAL_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?")

_ENDPOINT = object()


def _safe_code(value: Any, default: str) -> str:
    return value if isinstance(value, str) and _CODE_RE.fullmatch(value) else default


class STTSessionError(Exception):
    """Start was refused. ``code`` is a fixed identifier; the message never echoes a value."""

    def __init__(self, code: str) -> None:
        safe = _safe_code(code, ERROR_INTERNAL)
        message = _ERROR_MESSAGES.get(safe, f"Speech recognition could not start ({safe}).")
        super().__init__(message)
        self.code = safe


# ── Resolved session config ──────────────────────────────────────────────────


def _frozen(resolved: ResolvedService) -> ResolvedService:
    """Copy whose option and credential-ref maps are read-only views."""
    return dataclasses.replace(
        resolved,
        options=MappingProxyType(dict(resolved.options)),
        credential_refs=MappingProxyType(dict(resolved.credential_refs)),
    )


@dataclass(frozen=True)
class STTSessionConfig:
    """Everything one session needs, resolved before capture and immutable for the run."""

    stream_ids: Tuple[str, ...]
    primary: ResolvedService
    # Local STT. Equal to ``primary`` for a Local session.
    fallback: ResolvedService
    sample_rate: int = 16000
    channels: int = 1
    # Whisper size from a legacy flat start config; None for v2 service sections.
    legacy_model_size: Optional[str] = None
    policy: FailoverPolicy = field(default_factory=FailoverPolicy)
    ring_capacity_ms: float = DEFAULT_CAPACITY_MS
    max_pending_frames: int = 500
    replay_frame_ms: float = 100.0
    start_timeout_s: float = 15.0
    # None waits as long as the model load takes, like the legacy pipeline did.
    local_start_timeout_s: Optional[float] = None
    stop_timeout_s: float = 25.0
    drain_timeout_s: float = 2.0

    @property
    def cloud_primary(self) -> bool:
        return self.primary.protocol_family is not ProtocolFamily.LOCAL

    @property
    def language_hint(self) -> Optional[str]:
        value = self.primary.options.get("language")
        if isinstance(value, str) and value.strip() and value.strip().lower() != "auto":
            return value
        return None


def legacy_stt_service(settings: Mapping[str, Any], compute_type: str = "float16") -> ServiceConfig:
    """Transitional adapter (kept until EF-030): legacy flat ``asr.*`` keys -> Local STT."""
    options = {
        "language": settings.get("asr.language", "en"),
        "device": settings.get("asr.device", "auto"),
        "compute_type": compute_type,
    }
    try:
        return ServiceConfig(service="stt", provider_id="local", enabled=True, options=options)
    except ConfigError as exc:
        raise STTSessionError(exc.code) from None


def stt_service_from_start_config(
    settings: Mapping[str, Any], compute_type: str = "float16"
) -> Tuple[ServiceConfig, bool]:
    """The STT service a start config asks for, and whether the legacy adapter produced it."""
    section = settings.get(STT_SECTION_KEY)
    if section is None:
        services = settings.get("services")
        if isinstance(services, Mapping):
            section = services.get("stt")
    if section is None:
        return legacy_stt_service(settings, compute_type), True
    try:
        return ServiceConfig.from_dict("stt", section), False
    except ConfigError as exc:
        raise STTSessionError(exc.code) from None


def resolve_session_config(
    settings: Mapping[str, Any],
    stream_ids: Sequence[str],
    *,
    registry: Optional[ProviderRegistry] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    compute_type: str = "float16",
    policy: Optional[FailoverPolicy] = None,
) -> STTSessionConfig:
    """Resolve provider, endpoint, model, and options once. Raises ``STTSessionError``."""
    ids = tuple(stream_ids)
    if not ids or len(set(ids)) != len(ids) or not all(isinstance(s, str) and s for s in ids):
        raise STTSessionError(ERROR_INVALID_REQUEST)
    if not isinstance(settings, Mapping):
        raise STTSessionError(ERROR_INVALID_REQUEST)
    service, legacy = stt_service_from_start_config(settings, compute_type)
    if not service.enabled:
        raise STTSessionError(ERROR_STT_DISABLED)
    registry = registry or build_default_registry()
    try:
        primary = registry.resolve_service(service)
        if primary.protocol_family is ProtocolFamily.LOCAL:
            fallback = primary
        else:
            hint = primary.options.get("language")
            language = (
                hint if isinstance(hint, str) and _LOCAL_LANGUAGE_RE.fullmatch(hint) else "auto"
            )
            fallback = registry.resolve_service(
                ServiceConfig(
                    service="stt", provider_id="local", enabled=True, options={"language": language}
                )
            )
    except ProviderRegistryError as err:
        raise STTSessionError(err.code.value) from None
    model_size = settings.get("asr.model_size", "small") if legacy else None
    return STTSessionConfig(
        stream_ids=ids,
        primary=_frozen(primary),
        fallback=_frozen(fallback),
        sample_rate=sample_rate,
        channels=channels,
        legacy_model_size=model_size if isinstance(model_size, str) and model_size else None,
        policy=policy or FailoverPolicy(),
    )


# ── Events ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class STTTranscriptEvent:
    """A visible transcript change for one stream, after ledger decisions."""

    stream_id: str
    is_final: bool
    text: str
    item_key: str
    revision: int
    epoch_id: int
    provider_id: str
    language: Optional[str] = None
    start_ms: float = 0.0
    end_ms: float = 0.0
    # An empty final that closes an item without committing text (retraction, duplicate,
    # abandoned partial, session end).
    retracted: bool = False
    duplicate_of: Optional[str] = None
    continuity: Optional[ContinuityWarning] = None


class STTStatusKind(str, Enum):
    RECONNECTING = "stt_reconnecting"
    RECONNECTED = "stt_reconnected"
    FALLBACK = "stt_fallback"
    FAILED = "stt_failed"


@dataclass(frozen=True)
class STTStatusEvent:
    """A provider transition on one stream. ``reason`` is a fixed code, never provider text."""

    stream_id: str
    status: STTStatusKind
    provider_id: Optional[str]
    previous_provider_id: Optional[str]
    reason: str
    state: str
    replay_start_ms: Optional[float] = None
    replay_end_ms: Optional[float] = None
    gap: Optional[Tuple[float, float]] = None


# ── Builders ─────────────────────────────────────────────────────────────────

# (resolved service) -> secrets by credential slot. Runs on a worker thread; the orchestrator
# copies what it needs and clears its copy once the adapter exists.
SecretsResolver = Callable[[ResolvedService], Optional[Mapping[str, str]]]
# (resolved primary, secrets, stream ids) -> adapter. Blocking; runs on a worker thread.
CloudBuilder = Callable[[ResolvedService, Mapping[str, str], Tuple[str, ...]], StreamingSTTProvider]
# (session config) -> Local adapter for every stream id. Blocking; runs on a worker thread.
LocalBuilder = Callable[[STTSessionConfig], StreamingSTTProvider]
EventSink = Callable[[Any], None]


def _language_hints(resolved: ResolvedService) -> Tuple[str, ...]:
    value = resolved.options.get("language")
    if isinstance(value, str) and value.strip() and value.strip().lower() != "auto":
        return (value,)
    return ()


def default_cloud_builder(registry: ProviderRegistry) -> CloudBuilder:
    """Builds cloud STT adapters from a resolved service.

    Adapters exposing a ``from_resolved`` hook (Custom, Deepgram, AssemblyAI, Azure Speech)
    build through it. OpenAI Realtime and Gemini Live get their resolved settings objects.
    """

    def build(
        resolved: ResolvedService, secrets: Mapping[str, str], stream_ids: Tuple[str, ...]
    ) -> StreamingSTTProvider:
        factory = registry.load_factory(resolved.capability, resolved.provider_id)
        hook = getattr(factory, "from_resolved", None)
        if callable(hook):
            return hook(resolved, secrets, stream_ids=stream_ids)
        if resolved.protocol_family is ProtocolFamily.OPENAI_REALTIME:
            from engine.providers.stt.openai_protocol import OpenAIRealtimeSettings

            raw_prompt = resolved.options.get("prompt")
            prompt = raw_prompt.strip() if isinstance(raw_prompt, str) and raw_prompt.strip() else None
            raw_keywords = resolved.options.get("keywords")
            keywords = (
                tuple(p.strip() for p in raw_keywords.split(",") if p.strip())
                if isinstance(raw_keywords, str)
                else tuple(raw_keywords or ())
            )
            settings = OpenAIRealtimeSettings(
                endpoint=resolved.endpoint,
                model=resolved.model,
                api_key=secrets.get("api_key", ""),
                languages=_language_hints(resolved),
                prompt=prompt,
                keywords=keywords,
            )
            return factory(settings, provider_id=resolved.provider_id, stream_ids=stream_ids)
        if resolved.protocol_family is ProtocolFamily.GEMINI_LIVE:
            from engine.providers.stt.gemini_live import GeminiLiveSTTConfig

            raw_vocab = resolved.options.get("custom_vocabulary")
            vocab = (
                tuple(p.strip() for p in raw_vocab.split(",") if p.strip())
                if isinstance(raw_vocab, str)
                else tuple(raw_vocab or ())
            )
            config = GeminiLiveSTTConfig(
                api_key=secrets.get("api_key", ""),
                endpoint=resolved.endpoint,
                model=resolved.model,
                language_hints=_language_hints(resolved),
                custom_vocabulary=vocab,
                provider_id=resolved.provider_id,
            )
            return factory(config)
        raise ProviderRegistryError(
            RegistryErrorCode.FACTORY_UNAVAILABLE,
            provider_id=resolved.provider_id,
            capability=ProviderCapability.STT,
        )

    return build


def default_local_builder(config: STTSessionConfig) -> StreamingSTTProvider:
    from engine.providers.stt.local import LocalSTTProvider, LocalSTTSettings

    settings = LocalSTTSettings.from_options(dict(config.fallback.options))
    return LocalSTTProvider(
        settings, provider_id=config.fallback.provider_id, stream_ids=config.stream_ids
    )


# ── Runtime state ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _AudioItem:
    pcm: bytes
    start_ms: float
    end_ms: float
    speech: bool


@dataclass
class _Lease:
    """The connection currently serving one stream, bound to its ledger epoch."""

    provider: StreamingSTTProvider
    service: ResolvedService
    epoch: EpochHandle
    # Session-timeline position of the next audio this connection should receive.
    sent_end_ms: float = 0.0

    @property
    def is_local(self) -> bool:
        return self.service.protocol_family is ProtocolFamily.LOCAL


class _Stream:
    """Per-stream state. Everything but ``ring``, ``inbox`` and ``dropped_frames`` is touched
    only by the event loop."""

    def __init__(self, stream_id: str, ring: AudioRingBuffer, inbox_size: int) -> None:
        self.stream_id = stream_id
        self.ring = ring
        self.inbox: Queue[Any] = Queue(maxsize=inbox_size)
        self.controller: Optional[STTFailoverController] = None
        self.results: Deque[Tuple[EpochHandle, STTTranscriptResult]] = deque()
        self.failures: Deque[Tuple[EpochHandle, ProviderError]] = deque()
        self.preroll: Deque[_AudioItem] = deque(maxlen=PREROLL_FRAMES)
        self.lease: Optional[_Lease] = None
        self.task: Optional[asyncio.Task] = None
        self.sequence = 0
        self.dropped_frames = 0
        self.last_epoch_id = 0
        self.last_provider_id: Optional[str] = None


class _State(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


def _start_error(exc: BaseException) -> STTSessionError:
    if isinstance(exc, STTSessionError):
        return exc
    if isinstance(exc, ProviderRuntimeError):
        return STTSessionError(exc.error.code.value)
    if isinstance(exc, ProviderRegistryError):
        return STTSessionError(exc.code.value)
    if isinstance(exc, asyncio.TimeoutError):
        return STTSessionError(ERROR_TIMEOUT)
    logger.error("STT session start failed unexpectedly (%s)", type(exc).__name__)
    return STTSessionError(ERROR_INTERNAL)


def _failure_reason(exc: BaseException, default: str) -> str:
    if isinstance(exc, (STTSessionError, ProviderRegistryError)):
        return _safe_code(getattr(exc.code, "value", exc.code), default)
    if isinstance(exc, ProviderRuntimeError):
        return _safe_code(exc.error.code.value, default)
    if isinstance(exc, asyncio.TimeoutError):
        return ERROR_TIMEOUT
    return default


async def _wait(awaitable: Awaitable[Any], timeout: Optional[float]) -> Any:
    if timeout is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout)


async def _cancel_all(tasks: Iterable[asyncio.Task]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _provider_label(provider: Any) -> str:
    try:
        return _safe_label(provider.provider_id)
    except Exception:
        return "unknown"


def _safe_label(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", value)
        else "unknown"
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────


class STTSessionOrchestrator:
    """Owns the STT providers, ledger, ring buffers, and failover of one session."""

    def __init__(
        self,
        config: STTSessionConfig,
        on_event: EventSink,
        *,
        registry: Optional[ProviderRegistry] = None,
        cloud_builder: Optional[CloudBuilder] = None,
        local_builder: Optional[LocalBuilder] = None,
        secrets_resolver: Optional[SecretsResolver] = None,
        clock: Callable[[], float] = time.monotonic,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._config = config
        self._on_event = on_event
        self._registry = registry
        self._cloud_builder = cloud_builder
        self._local_builder = local_builder or default_local_builder
        self._secrets_resolver = secrets_resolver
        self._clock = clock
        self._poll_interval_s = poll_interval_s
        self._ledger = TranscriptLedger()
        self._streams: Dict[str, _Stream] = {}
        # Providers this session started (or is starting) and has not stopped yet.
        self._providers: Set[StreamingSTTProvider] = set()
        self._tasks: Set[asyncio.Task] = set()
        self._local: Optional[StreamingSTTProvider] = None
        self._local_started = False
        self._local_lock = asyncio.Lock()
        self._local_routes: Dict[str, EpochHandle] = {}
        self._state = _State.IDLE
        self._accepting = False
        self._draining = False
        self._stop_task: Optional[asyncio.Task] = None

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def config(self) -> STTSessionConfig:
        return self._config

    @property
    def state(self) -> str:
        return self._state.value

    def failover_state(self, stream_id: str) -> Optional[FailoverState]:
        stream = self._streams.get(stream_id)
        return stream.controller.state if stream is not None and stream.controller else None

    def active_provider_id(self, stream_id: str) -> Optional[str]:
        stream = self._streams.get(stream_id)
        if stream is None or stream.lease is None:
            return None
        return stream.lease.service.provider_id

    def dropped_frames(self, stream_id: str) -> int:
        stream = self._streams.get(stream_id)
        return stream.dropped_frames if stream is not None else 0

    def owned_tasks(self) -> Set[asyncio.Task]:
        return {task for task in self._tasks if not task.done()}

    def owned_providers(self) -> Set[StreamingSTTProvider]:
        return set(self._providers)

    # ── Capture side (any thread, never blocks) ──────────────────────────────

    def submit_audio(self, stream_id: str, pcm: bytes, speech: bool = True) -> bool:
        """Retain and queue one captured chunk. False when the chunk was not taken."""
        stream = self._streams.get(stream_id)
        if stream is None or not self._accepting:
            return False
        frame_bytes = self._config.channels * SAMPLE_WIDTH_BYTES
        usable = len(pcm) - len(pcm) % frame_bytes
        if usable <= 0:
            return False
        if usable != len(pcm):
            pcm = pcm[:usable]
        span = stream.ring.append(pcm)
        try:
            stream.inbox.put_nowait(_AudioItem(pcm, span.start_ms, span.end_ms, speech))
        except Full:
            stream.dropped_frames += 1
            if stream.dropped_frames == 1 or stream.dropped_frames % 100 == 0:
                logger.warning(
                    "STT session falling behind; dropping audio: stream=%s dropped=%d",
                    stream_id,
                    stream.dropped_frames,
                )
            return False
        return True

    def request_endpoint(self, stream_id: str) -> None:
        """The engine's silence endpoint. Finalizes the utterance on Local connections."""
        stream = self._streams.get(stream_id)
        if stream is None or not self._accepting:
            return
        try:
            stream.inbox.put_nowait(_ENDPOINT)
        except Full:
            logger.debug("STT endpoint dropped: stream=%s", stream_id)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Ready every provider the session needs, or release everything and raise."""
        if self._state is not _State.IDLE:
            raise STTSessionError(ERROR_INVALID_STATE)
        self._state = _State.STARTING
        try:
            await self._open()
        except BaseException as exc:
            await asyncio.shield(self._release(emit_terminal=False))
            self._state = _State.STOPPED
            if isinstance(exc, (asyncio.CancelledError, STTSessionError)) or not isinstance(
                exc, Exception
            ):
                raise
            raise _start_error(exc) from None
        for stream in self._streams.values():
            stream.task = self._spawn(self._run_stream(stream), f"stt-owner-{stream.stream_id}")
        self._state = _State.RUNNING
        self._accepting = True
        logger.info(
            "STT session started: provider=%s fallback=%s streams=%d",
            self._config.primary.provider_id,
            self._config.fallback.provider_id,
            len(self._streams),
        )

    async def stop(self) -> None:
        """Release the session exactly once. Concurrent and repeated calls share one stop."""
        if self._stop_task is None:
            if self._state in (_State.IDLE, _State.STOPPED):
                self._state = _State.STOPPED
                return
            self._stop_task = asyncio.get_running_loop().create_task(
                self._close(), name="stt-session-stop"
            )
        await asyncio.shield(self._stop_task)

    async def _open(self) -> None:
        cfg = self._config
        self._streams = {sid: self._new_stream(sid) for sid in cfg.stream_ids}
        if not cfg.cloud_primary:
            local = await self._ensure_local()
            for stream in self._streams.values():
                start = self._ledger.begin_epoch(stream.stream_id, cfg.fallback.provider_id)
                self._attach(stream, local, cfg.fallback, start, None)
            return
        secrets = await self._secrets_for(cfg.primary)
        try:
            await self._check_local_ready()
            for stream in self._streams.values():
                deadline = self._clock() + cfg.start_timeout_s
                provider = await self._start_cloud(stream.stream_id, secrets, deadline)
                start = self._ledger.begin_epoch(stream.stream_id, cfg.primary.provider_id)
                self._attach(stream, provider, cfg.primary, start, None)
        finally:
            secrets.clear()

    async def _close(self) -> None:
        self._state = _State.STOPPING
        self._accepting = False
        self._draining = True
        owners = [
            s.task for s in self._streams.values() if s.task is not None and not s.task.done()
        ]
        if owners:
            _, pending = await asyncio.wait(owners, timeout=self._config.drain_timeout_s)
            await _cancel_all(pending)
        await self._release(emit_terminal=True)
        self._state = _State.STOPPED
        logger.info("STT session stopped: provider=%s", self._config.primary.provider_id)

    async def _release(self, emit_terminal: bool) -> None:
        for provider in list(self._providers):
            await self._retire(provider)
        pumps = [task for task in self._tasks if not task.done()]
        if pumps:
            _, pending = await asyncio.wait(pumps, timeout=self._config.stop_timeout_s)
            await _cancel_all(pending)
        for stream in self._streams.values():
            # Stopping a provider flushes finals for anything it was still showing.
            self._drain_results(stream)
            if emit_terminal:
                self._emit_closing_final(stream)
            if stream.controller is not None:
                stream.controller.end_session()
            stream.lease = None
            stream.preroll.clear()
            stream.results.clear()
            stream.failures.clear()
            while True:
                try:
                    stream.inbox.get_nowait()
                except Empty:
                    break
            stream.ring.clear()
        self._local_routes.clear()
        self._local = None
        self._local_started = False
        await _cancel_all(list(self._tasks))

    # ── Providers ────────────────────────────────────────────────────────────

    def _new_stream(self, stream_id: str) -> _Stream:
        cfg = self._config
        ring = AudioRingBuffer(
            stream_id, cfg.sample_rate, cfg.channels, capacity_ms=cfg.ring_capacity_ms
        )
        stream = _Stream(stream_id, ring, cfg.max_pending_frames)
        ledger = self._ledger
        stream.controller = STTFailoverController(
            stream_id,
            ring,
            lambda: ledger.last_final_boundary_ms(stream_id),
            policy=cfg.policy,
            clock=self._clock,
            cloud_primary=cfg.cloud_primary,
        )
        return stream

    async def _secrets_for(self, resolved: ResolvedService) -> Dict[str, str]:
        if not resolved.credential_refs:
            return {}
        if self._secrets_resolver is None:
            raise STTSessionError(ERROR_MISSING_CREDENTIAL)
        try:
            supplied = await asyncio.to_thread(self._secrets_resolver, resolved)
        except Exception as exc:
            logger.warning(
                "STT credential lookup failed: provider=%s error=%s",
                resolved.provider_id,
                type(exc).__name__,
            )
            raise STTSessionError(ERROR_MISSING_CREDENTIAL) from None
        secrets: Dict[str, str] = {}
        for slot in resolved.credential_refs:
            value = supplied.get(slot) if isinstance(supplied, Mapping) else None
            if not isinstance(value, str) or not value.strip():
                secrets.clear()
                raise STTSessionError(ERROR_MISSING_CREDENTIAL)
            secrets[slot] = value
        return secrets

    async def _check_local_ready(self) -> None:
        """A cloud session needs Local as its fallback: prove it can load without loading it."""
        try:
            local = await asyncio.to_thread(self._local_builder, self._config)
            health = await asyncio.wait_for(local.health_check(), self._config.start_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Local STT fallback check failed (%s)", type(exc).__name__)
            raise STTSessionError(ERROR_LOCAL_FALLBACK_UNAVAILABLE) from None
        if health.status is ProviderHealthStatus.UNHEALTHY:
            raise STTSessionError(ERROR_LOCAL_FALLBACK_UNAVAILABLE)
        self._local = local

    async def _ensure_local(self) -> StreamingSTTProvider:
        """Start the shared Local provider once; later callers reuse it."""
        async with self._local_lock:
            if self._local_started and self._local is not None:
                return self._local
            if self._local is None:
                self._local = await asyncio.to_thread(self._local_builder, self._config)
            local = self._local
            self._providers.add(local)
            try:
                await _wait(local.start(), self._config.local_start_timeout_s)
            except BaseException:
                self._local = None
                await self._retire(local)
                raise
            self._local_started = True
            self._spawn(
                self._pump(local, self._local_routes.get, self._config.stream_ids),
                "stt-pump-local",
            )
            return local

    async def _start_cloud(
        self, stream_id: str, secrets: Mapping[str, str], deadline: float
    ) -> StreamingSTTProvider:
        builder = self._cloud_builder
        if builder is None:
            if self._registry is None:
                self._registry = await asyncio.to_thread(build_default_registry)
            builder = default_cloud_builder(self._registry)
        provider = await asyncio.to_thread(builder, self._config.primary, secrets, (stream_id,))
        self._providers.add(provider)
        try:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(provider.start(), remaining)
        except BaseException:
            await self._retire(provider)
            raise
        return provider

    async def _retire(self, provider: StreamingSTTProvider) -> None:
        """Stop a provider this session owns. Each provider is stopped at most once."""
        if provider not in self._providers:
            return
        self._providers.discard(provider)
        try:
            await asyncio.wait_for(provider.stop(), self._config.stop_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "STT provider stop failed: provider=%s error=%s",
                _provider_label(provider),
                type(exc).__name__,
            )

    def _attach(
        self,
        stream: _Stream,
        provider: StreamingSTTProvider,
        service: ResolvedService,
        start: EpochStart,
        plan: Optional[ReplayPlan],
    ) -> _Lease:
        if start.abandoned_entry_keys:
            # The previous connection's open partial will never finish; clear it.
            self._emit_transcript(stream, start.handle, "", True, retracted=True)
        lease = _Lease(
            provider=provider,
            service=service,
            epoch=start.handle,
            sent_end_ms=plan.replay_start_ms if plan is not None else 0.0,
        )
        stream.lease = lease
        stream.last_epoch_id = start.handle.epoch_id
        stream.last_provider_id = service.provider_id
        if lease.is_local:
            self._local_routes[stream.stream_id] = start.handle
        else:
            epoch, owner = start.handle, stream.stream_id
            self._spawn(
                self._pump(provider, lambda sid: epoch if sid == owner else None, (owner,)),
                f"stt-pump-{owner}",
            )
        return lease

    def _spawn(self, coro: Any, name: str) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _pump(
        self,
        provider: StreamingSTTProvider,
        route: Callable[[str], Optional[EpochHandle]],
        stream_ids: Tuple[str, ...],
    ) -> None:
        """Move one provider's results to their streams; report how the provider ended."""
        error: Optional[ProviderError] = None
        try:
            async for result in provider.receive_events():
                stream = self._streams.get(getattr(result, "stream_id", None))
                epoch = route(result.stream_id) if stream is not None else None
                if epoch is not None:
                    stream.results.append((epoch, result))
        except asyncio.CancelledError:
            raise
        except ProviderRuntimeError as err:
            error = err.error
        except Exception as exc:
            logger.error(
                "STT event pump failed: provider=%s error=%s",
                _provider_label(provider),
                type(exc).__name__,
            )
            error = ProviderError(
                ProviderErrorCode.INTERNAL_ERROR,
                ProviderCapability.STT,
                "Speech recognition failed.",
            )
        if provider not in self._providers:
            return  # retired by this session, so its end was expected
        if error is None:
            error = ProviderError(
                ProviderErrorCode.CONNECTION_CLOSED,
                ProviderCapability.STT,
                "Speech recognition connection closed.",
                retryable=True,
            )
        for sid in stream_ids:
            stream = self._streams.get(sid)
            epoch = route(sid)
            if stream is not None and epoch is not None:
                stream.failures.append((epoch, error))

    # ── Stream owner ─────────────────────────────────────────────────────────

    async def _run_stream(self, stream: _Stream) -> None:
        try:
            while True:
                worked = self._drain_results(stream)
                if self._draining:
                    fed = await self._feed_inbox(stream)
                    if not fed and not stream.results:
                        return
                    continue
                if stream.failures:
                    epoch, error = stream.failures.popleft()
                    await self._on_failure(stream, epoch, error)
                    continue
                decision = stream.controller.poll()
                if decision.action is not FailoverAction.NONE:
                    await self._apply(stream, decision)
                    continue
                fed = await self._feed_inbox(stream)
                if not (worked or fed):
                    await asyncio.sleep(self._poll_interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "STT stream owner failed: stream=%s error=%s", stream.stream_id, type(exc).__name__
            )
            await self._fail(stream, ERROR_INTERNAL)

    def _drain_results(self, stream: _Stream) -> bool:
        worked = False
        while stream.results:
            epoch, result = stream.results.popleft()
            worked = True
            lease = stream.lease
            if lease is None or lease.epoch.epoch_id != epoch.epoch_id:
                continue  # a retired or failed connection
            self._ingest(stream, epoch, result)
        return worked

    async def _feed_inbox(self, stream: _Stream) -> bool:
        fed = False
        for _ in range(MAX_ITEMS_PER_PASS):
            try:
                item = stream.inbox.get_nowait()
            except Empty:
                break
            fed = True
            lease = stream.lease
            if lease is None:
                continue  # failed stream: audio stays only in the ring buffer
            try:
                if item is _ENDPOINT:
                    if lease.is_local:
                        await lease.provider.finalize_stream(stream.stream_id)
                elif self._config.cloud_primary:
                    await self._send_span(stream, lease, item.pcm, item.start_ms, item.end_ms)
                else:
                    await self._send_gated(stream, lease, item)
            except ProviderRuntimeError as err:
                stream.failures.append((lease.epoch, err.error))
                break
        return fed

    async def _send_gated(self, stream: _Stream, lease: _Lease, item: _AudioItem) -> None:
        if not item.speech:
            stream.preroll.append(item)
            return
        while stream.preroll:
            await self._send_pcm(stream, lease, stream.preroll.popleft().pcm)
        await self._send_pcm(stream, lease, item.pcm)

    async def _send_span(
        self, stream: _Stream, lease: _Lease, pcm: bytes, start_ms: float, end_ms: float
    ) -> None:
        """Send audio so the provider's clock stays on the session timeline."""
        if end_ms <= lease.sent_end_ms:
            return  # already sent by a replay
        if start_ms < lease.sent_end_ms:
            pcm = pcm[self._ms_to_bytes(lease.sent_end_ms - start_ms) :]
        elif start_ms > lease.sent_end_ms:
            missing = min(start_ms - lease.sent_end_ms, self._config.ring_capacity_ms)
            silence = self._ms_to_bytes(missing)
            if silence:
                await self._send_pcm(stream, lease, bytes(silence))
        if pcm:
            await self._send_pcm(stream, lease, pcm)
        lease.sent_end_ms = end_ms

    async def _send_pcm(self, stream: _Stream, lease: _Lease, pcm: bytes) -> None:
        cfg = self._config
        frame_bytes = cfg.channels * SAMPLE_WIDTH_BYTES
        chunk = max(frame_bytes, self._ms_to_bytes(cfg.replay_frame_ms))
        view = memoryview(pcm)
        for offset in range(0, len(pcm), chunk):
            part = bytes(view[offset : offset + chunk])
            stream.sequence += 1
            await lease.provider.send_audio(
                AudioFrame(
                    stream_id=stream.stream_id,
                    sequence=stream.sequence,
                    pcm_data=part,
                    sample_rate=cfg.sample_rate,
                    channels=cfg.channels,
                    monotonic_timestamp=self._clock(),
                    duration_ms=(len(part) // frame_bytes) * 1000.0 / cfg.sample_rate,
                )
            )

    def _ms_to_bytes(self, ms: float) -> int:
        cfg = self._config
        frames = math.floor(max(0.0, ms) * cfg.sample_rate / 1000.0 + 1e-6)
        return frames * cfg.channels * SAMPLE_WIDTH_BYTES

    # ── Failover ─────────────────────────────────────────────────────────────

    async def _on_failure(self, stream: _Stream, epoch: EpochHandle, error: ProviderError) -> None:
        lease = stream.lease
        if lease is None or lease.epoch.epoch_id != epoch.epoch_id:
            return  # a connection this stream already left
        reason = _safe_code(error.code.value, ERROR_INTERNAL)
        logger.warning(
            "STT provider failed: stream=%s provider=%s code=%s retryable=%s",
            stream.stream_id,
            lease.service.provider_id,
            reason,
            error.retryable,
        )
        if lease.is_local:
            decision = stream.controller.report_local_failure(reason)
        else:
            decision = stream.controller.report_cloud_failure(
                retryable=error.retryable, reason=reason
            )
        await self._apply(stream, decision)

    async def _apply(self, stream: _Stream, decision: FailoverDecision) -> None:
        if decision.action is FailoverAction.RECONNECT:
            await self._reconnect(stream, decision)
        elif decision.action is FailoverAction.START_LOCAL_REPLAY:
            await self._fall_back(stream, decision)
        elif decision.action is FailoverAction.REPORT_UNRECOVERABLE:
            await self._fail(stream, decision.reason)

    async def _reconnect(self, stream: _Stream, decision: FailoverDecision) -> None:
        cfg = self._config
        previous, stream.lease = stream.lease, None
        if previous is not None:
            await self._retire(previous.provider)
        plan = decision.plan
        self._emit_status(
            stream,
            STTStatusKind.RECONNECTING,
            cfg.primary.provider_id,
            cfg.primary.provider_id,
            decision.reason,
            decision.state,
            plan,
        )
        deadline = decision.reconnect_deadline
        if deadline is None:
            deadline = self._clock() + cfg.policy.reconnect_timeout_s
        try:
            secrets = await self._secrets_for(cfg.primary)
            try:
                provider = await self._start_cloud(stream.stream_id, secrets, deadline)
            finally:
                secrets.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = _failure_reason(exc, "reconnect_failed")
            await self._apply(stream, stream.controller.report_cloud_failure(reason=reason))
            return
        resumed = stream.controller.report_reconnect_succeeded()
        if resumed.action is not FailoverAction.RESUME_CLOUD:
            await self._retire(provider)
            await self._apply(stream, resumed)
            return
        start = self._ledger.begin_epoch(
            stream.stream_id, cfg.primary.provider_id, plan.offset_ms, plan.overlap_window, plan.gap
        )
        lease = self._attach(stream, provider, cfg.primary, start, plan)
        self._emit_status(
            stream,
            STTStatusKind.RECONNECTED,
            cfg.primary.provider_id,
            cfg.primary.provider_id,
            resumed.reason,
            resumed.state,
            plan,
            start.gap_warning,
        )
        await self._replay(stream, lease, plan)

    async def _fall_back(self, stream: _Stream, decision: FailoverDecision) -> None:
        cfg = self._config
        previous, stream.lease = stream.lease, None
        if previous is not None and not previous.is_local:
            await self._retire(previous.provider)
        plan = decision.plan
        try:
            local = await self._ensure_local()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Local STT fallback failed to start (%s)", type(exc).__name__)
            reason = _failure_reason(exc, "local_start_failed")
            await self._apply(stream, stream.controller.report_local_failure(reason))
            return
        start = self._ledger.begin_epoch(
            stream.stream_id,
            cfg.fallback.provider_id,
            plan.offset_ms,
            plan.overlap_window,
            plan.gap,
        )
        lease = self._attach(stream, local, cfg.fallback, start, plan)
        committed = stream.controller.report_local_started()
        self._emit_status(
            stream,
            STTStatusKind.FALLBACK,
            cfg.fallback.provider_id,
            cfg.primary.provider_id,
            decision.reason,
            committed.state,
            plan,
            start.gap_warning,
        )
        await self._replay(stream, lease, plan)

    async def _replay(self, stream: _Stream, lease: _Lease, plan: ReplayPlan) -> None:
        """Resend retained audio from the plan, then whatever arrived while reconnecting."""
        try:
            audio = plan.audio
            await self._send_span(stream, lease, audio.pcm_data, audio.start_ms, audio.end_ms)
            catch_up = stream.ring.read(lease.sent_end_ms)
            await self._send_span(
                stream, lease, catch_up.pcm_data, catch_up.start_ms, catch_up.end_ms
            )
        except ProviderRuntimeError as err:
            stream.failures.append((lease.epoch, err.error))

    async def _fail(self, stream: _Stream, reason: str) -> None:
        lease, stream.lease = stream.lease, None
        self._local_routes.pop(stream.stream_id, None)
        if lease is not None:
            self._emit_transcript(stream, lease.epoch, "", True, retracted=True)
            if not lease.is_local:
                await self._retire(lease.provider)
        state = stream.controller.state if stream.controller is not None else FailoverState.FAILED
        logger.error("STT stream failed: stream=%s reason=%s", stream.stream_id, reason)
        self._emit_status(
            stream,
            STTStatusKind.FAILED,
            None,
            lease.service.provider_id if lease is not None else stream.last_provider_id,
            reason,
            state,
        )

    # ── Events ───────────────────────────────────────────────────────────────

    def _ingest(self, stream: _Stream, epoch: EpochHandle, result: STTTranscriptResult) -> None:
        decision = self._ledger.ingest(result, epoch)
        action = decision.action
        if action is LedgerAction.PARTIAL:
            is_final, text, retracted = False, decision.text, False
        elif action is LedgerAction.FINAL:
            is_final, text, retracted = True, decision.text, False
        elif action in (LedgerAction.RETRACTED, LedgerAction.DUPLICATE_FINAL):
            is_final, text, retracted = True, "", True
        else:
            return
        self._emit(
            STTTranscriptEvent(
                stream_id=stream.stream_id,
                is_final=is_final,
                text=text,
                item_key=decision.entry_key,
                revision=decision.revision,
                epoch_id=decision.epoch_id,
                provider_id=epoch.provider_id,
                language=result.detected_language or self._config.language_hint,
                start_ms=decision.start_ms,
                end_ms=decision.end_ms,
                retracted=retracted,
                duplicate_of=decision.duplicate_of,
                continuity=decision.warning,
            )
        )

    def _emit_transcript(
        self, stream: _Stream, epoch: EpochHandle, text: str, is_final: bool, retracted: bool
    ) -> None:
        self._emit(
            STTTranscriptEvent(
                stream_id=stream.stream_id,
                is_final=is_final,
                text=text,
                item_key="",
                revision=0,
                epoch_id=epoch.epoch_id,
                provider_id=epoch.provider_id,
                language=self._config.language_hint,
                retracted=retracted,
            )
        )

    def _emit_closing_final(self, stream: _Stream) -> None:
        # The legacy pipeline ended every stream with an empty final; the UI clears partials on it.
        self._emit(
            STTTranscriptEvent(
                stream_id=stream.stream_id,
                is_final=True,
                text="",
                item_key="",
                revision=0,
                epoch_id=stream.last_epoch_id,
                provider_id=stream.last_provider_id or self._config.primary.provider_id,
                language=self._config.language_hint,
                retracted=True,
            )
        )

    def _emit_status(
        self,
        stream: _Stream,
        kind: STTStatusKind,
        provider_id: Optional[str],
        previous_provider_id: Optional[str],
        reason: str,
        state: FailoverState,
        plan: Optional[ReplayPlan] = None,
        gap_warning: Optional[ContinuityWarning] = None,
    ) -> None:
        gap = None
        if gap_warning is not None:
            gap = (gap_warning.start_ms, gap_warning.end_ms)
        elif plan is not None and plan.gap is not None:
            gap = plan.gap
        self._emit(
            STTStatusEvent(
                stream_id=stream.stream_id,
                status=kind,
                provider_id=provider_id,
                previous_provider_id=previous_provider_id,
                reason=_safe_code(reason, "provider_failure"),
                state=state.value,
                replay_start_ms=plan.replay_start_ms if plan is not None else None,
                replay_end_ms=plan.replay_end_ms if plan is not None else None,
                gap=gap,
            )
        )

    def _emit(self, event: Any) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            logger.error("STT event sink failed (%s)", type(exc).__name__)
