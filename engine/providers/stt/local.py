"""Local STT provider: the legacy Faster-Whisper runtime behind ``StreamingSTTProvider``.

Pack resolution. The model is never named by the caller. ``start`` probes the hardware and asks
``select_default_pack`` for the STT pack. A ``LEGACY_RUNTIME`` pack maps to the Whisper size the
current runtime already uses; a catalog pack must be installed and active in the pack service,
and its directory becomes the model path. Anything else is ``MODEL_UNAVAILABLE``.

Language. The hint defaults to ``"auto"``, which Whisper treats as "detect". Each result carries
the language Whisper detected.

Threads. Nothing blocking runs on the event loop. Model load and unload run through
``asyncio.to_thread``. Each stream id (``mic``, ``system``) has its own daemon worker thread and
``queue.Queue``, mirroring the engine's ``Process-{stream}`` threads; the worker owns that
stream's utterance state and hands results back with ``call_soon_threadsafe``. ``send_audio``
only enqueues and drops frames when a stream falls too far behind.

The adapter is dormant: nothing wires it into the engine yet, and it never chooses fallback.
Nothing here logs audio or transcript text.
"""

import asyncio
import contextlib
import importlib.util
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Optional, Sequence, Tuple

from engine.core.logging import get_logger
from engine.local_models.catalog import PackCatalog, PackKind, PackSource, builtin_catalog
from engine.local_models.selector import (
    HardwareProbe,
    SystemHardwareProbe,
    compatibility_issues,
    select_default_pack,
)
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

logger = get_logger("providers.stt.local")

AUTO_LANGUAGE = "auto"
SAMPLE_RATE = 16000
DEVICES = ("auto", "cpu", "cuda")
COMPUTE_TYPES = ("float16", "int8_float16", "int8", "float32")
QUALITY_PRESETS = ("fast", "balanced", "accurate")
_QUALITY_MODEL_MAP: Dict[str, str] = {
    "fast": "tiny",
    "balanced": "small",
    "accurate": "medium",
}
# Mirrors the engine's per-stream capture queue bound.
DEFAULT_MAX_PENDING_FRAMES = 500

_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?")

_AUDIO = "audio"
_FINALIZE = "finalize"
_RESET = "reset"
_STOP = "stop"
_END = object()


def _error(
    code: ProviderErrorCode,
    message: str,
    retryable: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> ProviderError:
    return ProviderError(
        code=code,
        capability=ProviderCapability.STT,
        message=message,
        retryable=retryable,
        details=details,
    )


def _invalid(field_name: str) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Local speech recognition rejected the request.",
            details={"field": field_name},
        )
    )


def _is_auto(language: Optional[str]) -> bool:
    return language is None or language.strip().lower() in ("", AUTO_LANGUAGE)


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LocalSTTSettings:
    """Local STT options from the ``local`` manifest. There is deliberately no model field."""

    language: str = AUTO_LANGUAGE
    device: str = "auto"
    compute_type: str = "float16"
    quality_preset: str = "balanced"

    @classmethod
    def from_options(cls, options: Dict[str, Any]) -> "LocalSTTSettings":
        """Build from registry-resolved options; keys for other capabilities are ignored."""
        defaults = cls()
        return cls(
            language=options.get("language", defaults.language),
            device=options.get("device", defaults.device),
            compute_type=options.get("compute_type", defaults.compute_type),
            quality_preset=options.get("quality_preset", defaults.quality_preset),
        )


def validate_settings(settings: LocalSTTSettings) -> None:
    language = settings.language
    if not isinstance(language, str) or not (
        language.strip().lower() == AUTO_LANGUAGE or _LANGUAGE_RE.fullmatch(language)
    ):
        raise _invalid("language")
    if settings.device not in DEVICES:
        raise _invalid("device")
    if settings.compute_type not in COMPUTE_TYPES:
        raise _invalid("compute_type")
    if settings.quality_preset not in QUALITY_PRESETS:
        raise _invalid("quality_preset")


# ── Pack resolution ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedSTTPack:
    """What the runtime loads. ``model_path`` is set only for installed catalog packs."""

    pack_id: str
    version: str
    reason: str
    model_size: str
    model_path: Optional[str] = None


InstallerFactory = Callable[[PackCatalog, HardwareProbe], Any]


def _default_installer(catalog: PackCatalog, probe: HardwareProbe) -> Any:
    from engine.core.config import Config
    from engine.local_models.installer import LocalPackInstaller

    return LocalPackInstaller.from_config(Config(), catalog, probe)


def resolve_stt_pack(
    catalog: PackCatalog,
    probe: HardwareProbe,
    language: Optional[str],
    installer_factory: InstallerFactory = _default_installer,
    quality_preset: str = "balanced",
    legacy_model_size: Optional[str] = None,
) -> ResolvedSTTPack:
    """Pick the STT pack for this machine. Blocking (hardware probe, filesystem)."""
    profile = probe.probe()
    hint = None if _is_auto(language) else language
    selection = select_default_pack(catalog, PackKind.STT, profile, source_language=hint)
    pack = selection.pack
    if pack is None:
        # A pack that fits this machine but not the hint means the language is the problem.
        hardware_fits = any(
            not compatibility_issues(candidate, profile)
            for candidate in catalog.packs_for(PackKind.STT)
        )
        if hint is not None and hardware_fits:
            raise ProviderRuntimeError(
                _error(
                    ProviderErrorCode.UNSUPPORTED_LANGUAGE,
                    "No local speech recognition pack supports the requested language.",
                    details={"reason": selection.reason},
                )
            )
        raise ProviderRuntimeError(
            _error(
                ProviderErrorCode.MODEL_UNAVAILABLE,
                "No local speech recognition pack is compatible with this machine.",
                details={"reason": selection.reason},
            )
        )
    if pack.source is PackSource.LEGACY_RUNTIME:
        if isinstance(legacy_model_size, str) and legacy_model_size.strip():
            model_size = legacy_model_size.strip()
        else:
            model_size = _QUALITY_MODEL_MAP.get(quality_preset, pack.legacy_model_id)
        pack_id = (
            f"legacy-faster-whisper-{model_size}"
            if model_size in ("tiny", "small", "medium")
            else pack.pack_id
        )
        return ResolvedSTTPack(pack_id, pack.version, selection.reason, model_size)
    installed = installer_factory(catalog, probe).active(pack.pack_id)
    if pack.runtime != "ctranslate2" or installed is None:
        raise ProviderRuntimeError(
            _error(
                ProviderErrorCode.MODEL_UNAVAILABLE,
                "The selected local speech recognition pack is not installed.",
                details={"pack_id": pack.pack_id, "reason": "pack_not_installed"},
            )
        )
    return ResolvedSTTPack(
        installed.pack_id, installed.version, selection.reason, pack.pack_id, str(installed.path)
    )


def _default_backend_factory() -> Any:
    from engine.asr.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend()


def _runtime_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


# ── Provider ─────────────────────────────────────────────────────────────────


class _LocalStream:
    """One audio stream. Utterance fields are touched only by the stream's worker thread."""

    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self.commands: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.dropped_frames = 0
        self.ordinal = 0
        self.revision = 0
        self.last_text: Optional[str] = None
        self.received_ms = 0.0
        self.utterance_start_ms: Optional[float] = None
        self.language: Optional[str] = None

    def close_utterance(self) -> None:
        if self.revision:
            self.ordinal += 1
        self.revision = 0
        self.last_text = None
        self.utterance_start_ms = None
        self.language = None


class LocalSTTProvider(StreamingSTTProvider):
    """Faster-Whisper streaming STT with independent state per stream id.

    - ``sequence`` is the utterance ordinal per stream and ``provider_event_id`` is stable per
      utterance; ``revision`` strictly increases until the utterance's single final.
    - ``finalize_stream`` closes the utterance after the audio already queued for that stream;
      ``reset_stream`` drops queued audio and abandons the open utterance.
    - ``stop`` processes queued audio, finalizes anything shown, joins the workers, and unloads
      the model. ``start`` and ``stop`` are idempotent.
    """

    def __init__(
        self,
        settings: Optional[LocalSTTSettings] = None,
        *,
        provider_id: str = "local",
        stream_ids: Sequence[str] = ("default",),
        catalog: Optional[PackCatalog] = None,
        hardware_probe: Optional[HardwareProbe] = None,
        backend_factory: Callable[[], Any] = _default_backend_factory,
        installer_factory: InstallerFactory = _default_installer,
        runtime_available: Callable[[], bool] = _runtime_available,
        max_pending_frames: int = DEFAULT_MAX_PENDING_FRAMES,
        stop_timeout_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        legacy_model_size: Optional[str] = None,
    ) -> None:
        self._settings = settings or LocalSTTSettings()
        validate_settings(self._settings)
        if not stream_ids or len(set(stream_ids)) != len(stream_ids):
            raise _invalid("stream_ids")
        if max_pending_frames < 1:
            raise _invalid("max_pending_frames")
        self._provider_id = provider_id
        self._stream_ids = tuple(stream_ids)
        self._catalog = catalog or builtin_catalog()
        self._probe = hardware_probe or SystemHardwareProbe()
        self._backend_factory = backend_factory
        self._installer_factory = installer_factory
        self._runtime_available = runtime_available
        self._max_pending = max_pending_frames
        self._stop_timeout_s = stop_timeout_s
        self._clock = clock
        self._legacy_model_size = legacy_model_size
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._lifecycle = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._backend: Any = None
        self._pack: Optional[ResolvedSTTPack] = None
        self._streams: Dict[str, _LocalStream] = {}
        self._abort = threading.Event()
        self._failure: Optional[ProviderError] = None
        self._started = False
        self._stopping = False

    # ── BaseProvider ─────────────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def resolved_pack(self) -> Optional[ResolvedSTTPack]:
        return self._pack

    async def start(self) -> None:
        async with self._lifecycle:
            if self._started:
                return
            self._loop = asyncio.get_running_loop()
            while not self._events.empty():
                self._events.get_nowait()
            self._failure = None
            self._stopping = False
            self._abort.clear()
            self._backend, self._pack = await asyncio.to_thread(self._load)
            self._streams = {sid: _LocalStream(sid) for sid in self._stream_ids}
            for stream in self._streams.values():
                stream.thread = threading.Thread(
                    target=self._run_stream,
                    args=(stream,),
                    name=f"LocalSTT-{stream.stream_id}",
                    daemon=True,
                )
                stream.thread.start()
            self._started = True
            logger.info(
                "Local STT started: provider=%s streams=%d pack=%s@%s reason=%s device=%s",
                self._provider_id,
                len(self._streams),
                self._pack.pack_id,
                self._pack.version,
                self._pack.reason,
                getattr(self._backend, "device", None),
            )

    async def stop(self) -> None:
        async with self._lifecycle:
            if not self._started:
                return
            self._stopping = True
            streams = list(self._streams.values())
            for stream in streams:
                stream.commands.put_nowait((_STOP, None))
            try:
                all_joined = await asyncio.to_thread(self._join_workers, streams)
                backend, self._backend = self._backend, None
                if all_joined:
                    await asyncio.to_thread(self._unload, backend)
                else:
                    # A worker is stuck mid-inference; unloading under it could crash the
                    # runtime. The daemon thread drops the last reference when it returns.
                    logger.warning("Local STT workers did not stop in time; skipping unload")
            finally:
                self._streams = {}
                self._started = False
                self._events.put_nowait(_END)
            logger.info("Local STT stopped: provider=%s", self._provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        """Report readiness without loading a model or transcribing audio."""
        began = self._clock()
        if self._started and self._pack is not None and self._failure is None:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.HEALTHY,
                message="Local speech recognition is running.",
                latency_ms=(self._clock() - began) * 1000.0,
                details={"loaded": True, **self._pack_details(self._pack)},
            )
        try:
            pack = await asyncio.to_thread(self._check_ready)
        except ProviderRuntimeError as err:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=err.error.message,
                details={"code": err.error.code.value, "retryable": err.error.retryable},
            )
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Local speech recognition is available.",
            latency_ms=(self._clock() - began) * 1000.0,
            details={"loaded": False, **self._pack_details(pack)},
        )

    # ── StreamingSTTProvider ─────────────────────────────────────────────────

    async def send_audio(self, frame: AudioFrame) -> None:
        stream = self._require_stream(frame.stream_id)
        if frame.channels != 1 or frame.sample_rate != SAMPLE_RATE or len(frame.pcm_data) % 2:
            raise _invalid("audio_format")
        if not frame.pcm_data:
            return
        # qsize is approximate across threads, which is all a drop policy needs.
        if stream.commands.qsize() >= self._max_pending:
            stream.dropped_frames += 1
            if stream.dropped_frames == 1 or stream.dropped_frames % 100 == 0:
                logger.warning(
                    "Local STT falling behind; dropping audio: stream=%s dropped=%d",
                    stream.stream_id,
                    stream.dropped_frames,
                )
            return
        stream.commands.put_nowait((_AUDIO, bytes(frame.pcm_data)))

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        while True:
            item = await self._events.get()
            if item is _END:
                return
            if isinstance(item, ProviderError):
                raise ProviderRuntimeError(item)
            yield item

    async def finalize_stream(self, stream_id: str) -> None:
        stream = self._require_stream(stream_id)
        stream.commands.put_nowait((_FINALIZE, None))

    async def reset_stream(self, stream_id: str) -> None:
        stream = self._require_stream(stream_id)
        # Queued audio and pending finalizes belong to the abandoned utterance.
        while True:
            try:
                stream.commands.get_nowait()
            except queue.Empty:
                break
        stream.commands.put_nowait((_RESET, None))

    def dropped_frames(self, stream_id: str) -> int:
        stream = self._streams.get(stream_id)
        return stream.dropped_frames if stream is not None else 0

    # ── Blocking work (worker threads only) ──────────────────────────────────

    def _load(self) -> Tuple[Any, ResolvedSTTPack]:
        from engine.core.config import TranscriptionConfig

        pack = resolve_stt_pack(
            self._catalog,
            self._probe,
            self._settings.language,
            self._installer_factory,
            quality_preset=self._settings.quality_preset,
            legacy_model_size=self._legacy_model_size,
        )
        backend = None
        try:
            backend = self._backend_factory()
            backend.load_model(
                TranscriptionConfig(
                    model_size=pack.model_size,
                    # The backend maps "auto" to Whisper's own language detection.
                    language=self._settings.language,
                    device=self._settings.device,
                    compute_type=self._settings.compute_type,
                    model_path=pack.model_path,
                )
            )
        except Exception as exc:
            logger.error(
                "Local STT model load failed: pack=%s error=%s", pack.pack_id, type(exc).__name__
            )
            self._unload(backend)
            raise ProviderRuntimeError(
                _error(
                    ProviderErrorCode.MODEL_UNAVAILABLE,
                    "The local speech recognition model could not be loaded.",
                    details={"pack_id": pack.pack_id},
                )
            ) from None
        return backend, pack

    def _check_ready(self) -> ResolvedSTTPack:
        if not self._runtime_available():
            raise ProviderRuntimeError(
                _error(
                    ProviderErrorCode.MODEL_UNAVAILABLE,
                    "The local speech recognition runtime is not installed.",
                )
            )
        return resolve_stt_pack(
            self._catalog,
            self._probe,
            self._settings.language,
            self._installer_factory,
            quality_preset=self._settings.quality_preset,
            legacy_model_size=self._legacy_model_size,
        )

    @staticmethod
    def _unload(backend: Any) -> None:
        if backend is None:
            return
        try:
            backend.unload_model()
        except Exception as exc:
            logger.warning("Local STT unload failed: error=%s", type(exc).__name__)

    def _join_workers(self, streams: Sequence[_LocalStream]) -> bool:
        deadline = time.monotonic() + self._stop_timeout_s
        for stream in streams:
            if stream.thread is not None:
                stream.thread.join(max(0.0, deadline - time.monotonic()))
        if any(s.thread is not None and s.thread.is_alive() for s in streams):
            # Stop spending time on queued audio; finish the frame in flight and exit.
            self._abort.set()
            for stream in streams:
                if stream.thread is not None:
                    stream.thread.join(self._stop_timeout_s)
        return not any(s.thread is not None and s.thread.is_alive() for s in streams)

    def _run_stream(self, stream: _LocalStream) -> None:
        backend = self._backend
        try:
            while True:
                kind, payload = stream.commands.get()
                if kind == _AUDIO:
                    if not self._abort.is_set():
                        self._handle_audio(backend, stream, payload)
                elif kind == _FINALIZE:
                    self._emit_from(stream, backend.finalize_current(stream.stream_id), True)
                elif kind == _RESET:
                    backend.reset_stream(stream.stream_id)
                    stream.close_utterance()
                elif kind == _STOP:
                    if not self._abort.is_set():
                        self._emit_from(stream, backend.finalize_current(stream.stream_id), True)
                    return
        except Exception as exc:
            logger.error(
                "Local STT inference failed: stream=%s error=%s",
                stream.stream_id,
                type(exc).__name__,
            )
            self._post(
                _error(
                    ProviderErrorCode.INTERNAL_ERROR,
                    "Local speech recognition failed.",
                    details={"stream_id": stream.stream_id},
                )
            )

    def _handle_audio(self, backend: Any, stream: _LocalStream, pcm: bytes) -> None:
        if stream.utterance_start_ms is None:
            stream.utterance_start_ms = stream.received_ms
        stream.received_ms += len(pcm) / 2 / SAMPLE_RATE * 1000.0
        self._emit_from(stream, backend.transcribe_stream(pcm, stream.stream_id), False)

    def _emit_from(self, stream: _LocalStream, result: Any, endpoint: bool) -> None:
        """Map a legacy ``TranscriptResult`` (or None) onto canonical results."""
        if result is None:
            if endpoint:
                if stream.revision:
                    # Every partial was retracted by the final pass; close the utterance.
                    self._emit(stream, "", final=True)
                else:
                    stream.close_utterance()
            return
        text = (result.text or "").strip()
        if result.language:
            stream.language = result.language
        if result.is_final or endpoint:
            if text or stream.revision:
                self._emit(stream, text, final=True)
            else:
                stream.close_utterance()
            return
        if text and text != stream.last_text:
            self._emit(stream, text, final=False)

    def _emit(self, stream: _LocalStream, text: str, final: bool) -> None:
        stream.revision += 1
        stream.last_text = text
        start_ms = (
            stream.utterance_start_ms
            if stream.utterance_start_ms is not None
            else stream.received_ms
        )
        self._post(
            STTTranscriptResult(
                stream_id=stream.stream_id,
                sequence=stream.ordinal,
                provider_event_id=f"{self._provider_id}:{stream.stream_id}:{stream.ordinal}",
                text=text,
                is_final=final,
                revision=stream.revision,
                start_time_ms=start_ms,
                end_time_ms=max(start_ms, stream.received_ms),
                monotonic_timestamp=self._clock(),
                detected_language=stream.language,
                is_endpoint=final,
            )
        )
        if final:
            stream.close_utterance()

    def _post(self, item: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        # A RuntimeError means the loop closed under a daemon worker; nobody is listening.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._deliver, item)

    # ── Event loop side ──────────────────────────────────────────────────────

    def _deliver(self, item: Any) -> None:
        if isinstance(item, ProviderError):
            if self._failure is not None or self._stopping:
                return
            self._failure = item
            logger.error(
                "Local STT failed: provider=%s code=%s", self._provider_id, item.code.value
            )
        self._events.put_nowait(item)

    def _require_stream(self, stream_id: str) -> _LocalStream:
        if self._failure is not None:
            raise ProviderRuntimeError(self._failure)
        if not self._started or self._stopping:
            raise _invalid("not_started")
        stream = self._streams.get(stream_id)
        if stream is None:
            raise _invalid("stream_id")
        return stream

    @staticmethod
    def _pack_details(pack: ResolvedSTTPack) -> Dict[str, Any]:
        return {"pack_id": pack.pack_id, "pack_version": pack.version, "reason": pack.reason}
