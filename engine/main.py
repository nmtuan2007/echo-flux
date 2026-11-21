import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional, Set

from dotenv import load_dotenv

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    ConversationRecord,
    ProviderMetadata,
    SessionRecord,
    TranscriptMessage,
)
from engine.conversation.repository import (
    ConversationRepository,
    new_record_id,
)
from engine.core.progress import install_progress_hijack
from engine.server.protocol import (
    ASSISTANT_ERROR_MESSAGES,
    ERROR_ASSISTANT_UNAVAILABLE,
    ERROR_CANCELLED,
    ERROR_FORBIDDEN,
    ERROR_INTERNAL,
    ERROR_INVALID_REQUEST,
    BootstrapTokens,
    ProtocolError,
    SecurityMode,
    parse_assist_command,
    parse_summary_command,
    safe_command_id,
    suggestion_error_message,
    suggestion_result_message,
    summary_chunk_message,
    summary_done_message,
    summary_error_replies,
)
from engine.server.websocket_server import WebSocketServer
from engine.session.session_orchestrator import (
    STTSessionOrchestrator,
    STTStatusEvent,
    STTStatusKind,
    resolve_session_config,
)
from engine.session.translation_orchestrator import (
    TranslationErrorEvent,
    TranslationSessionOrchestrator,
    TranslationStatusEvent,
    TranslationStatusKind,
    resolve_translation_session_config,
)

logger = logging.getLogger("echoflux.engine")

SILENCE_FINALIZE_DELAY = 0.8


def _setup_cuda_paths():
    if sys.platform != "win32":
        return

    # Use sys.prefix to find the current venv root dynamically
    venv_base = Path(sys.prefix)
    nvidia_path = venv_base / "Lib" / "site-packages" / "nvidia"

    # Standard locations for pip-installed nvidia binaries
    targets = [
        nvidia_path / "cublas" / "bin",
        nvidia_path / "cudnn" / "bin",
        nvidia_path / "cuda_runtime" / "bin",
    ]

    for target in targets:
        if target.exists():
            # Windows requires absolute paths for DLL loading
            abs_path = str(target.resolve())

            with contextlib.suppress(Exception):
                os.add_dll_directory(abs_path)

            # Prepend to PATH so CTranslate2 finds these first
            os.environ["PATH"] = abs_path + os.pathsep + os.environ["PATH"]

    # Check for the manual dependency zlibwapi.dll
    # It's not in pip packages, usually in System32 or project root
    zlib_found = False
    for p in os.environ["PATH"].split(os.pathsep):
        if p and (Path(p) / "zlibwapi.dll").exists():
            zlib_found = True
            break

    if not zlib_found and (Path.cwd() / "zlibwapi.dll").exists():
        zlib_found = True

    if not zlib_found:
        logger.warning("zlibwapi.dll not found. CUDA might fail. Check System32.")


def _load_dotenv():
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _setup_logging():
    log_dir = Path.home() / ".echoflux" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{ts}.log"

    level_str = os.getenv("ECHOFLUX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging initialized - file: %s", log_file)


def _assist_failure(message_id: str, code: str, conversation_id: Optional[str] = None) -> dict:
    return suggestion_error_message(
        message_id, code, ASSISTANT_ERROR_MESSAGES[code], conversation_id
    )


def _assist_reply(command, outcome) -> dict:
    if not outcome.ok:
        error = outcome.error
        return suggestion_error_message(
            command.message_id,
            error.code.value,
            error.message,
            command.conversation_id,
            error.retryable,
            error.provider_code,
        )
    stored = outcome.stored
    return suggestion_result_message(
        conversation_id=command.conversation_id,
        message_id=command.message_id,
        assist_id=stored.assist_id,
        suggested_replies=[
            {"strategy": s.strategy, "text": s.text, "reasoning": s.reasoning}
            for s in stored.suggested_replies
        ],
        clarifying_questions=[
            {"question": q.question, "purpose": q.purpose} for q in stored.clarifying_questions
        ],
        provider=stored.provider.to_dict() if stored.provider is not None else None,
        context_window={
            "message_count": outcome.context_message_count,
            "omitted_message_count": outcome.omitted_message_count,
            "profile_count": outcome.profile_count,
            "omitted_profile_count": outcome.omitted_profile_count,
        },
    )


def _warning_payload(warning) -> Optional[dict]:
    if warning is None:
        return None
    return {
        "omitted_message_count": warning.omitted_message_count,
        "first_omitted_timestamp_ms": warning.first_omitted_timestamp_ms,
        "last_omitted_timestamp_ms": warning.last_omitted_timestamp_ms,
        "reason": warning.reason,
    }



def _continuity_payload(warning) -> Optional[dict]:
    if warning is None:
        return None
    return {
        "reason": warning.reason.value,
        "start_ms": warning.start_ms,
        "end_ms": warning.end_ms,
        "epoch": warning.epoch_id,
    }


def _stt_transcript_message(
    event,
    entry_id: Optional[str],
    translation_backend: Optional[str],
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict:
    """The legacy partial/final shape, then additive session metadata the store ignores."""
    msg = {
        "type": "final" if event.is_final else "partial",
        "text": event.text,
        "translation": None,
        "is_final": event.is_final,
        "timestamp": time.time(),
    }
    if event.stream_id in ("mic", "system"):
        msg["source"] = event.stream_id
    if entry_id:
        msg["entry_id"] = entry_id
    if translation_backend:
        msg["translation_backend"] = translation_backend
    msg.update(
        {
            "revision": event.revision,
            "item_id": event.item_key,
            "epoch": event.epoch_id,
            "stt_provider": event.provider_id,
            "language": event.language,
            "retracted": event.retracted,
            "duplicate_of": event.duplicate_of,
            "continuity": _continuity_payload(event.continuity),
            "conversation_id": conversation_id if conversation_id is not None else getattr(event, "conversation_id", None),
            "message_id": message_id if message_id is not None else getattr(event, "message_id", None),
        }
    )
    return msg


def _stt_status_message(event) -> dict:
    """Provider transitions ride on ``status`` with values the current store does not act on."""
    replay = None
    if event.replay_start_ms is not None:
        replay = {"start_ms": event.replay_start_ms, "end_ms": event.replay_end_ms}
    gap = None
    if event.gap is not None:
        gap = {"reason": "audio_gap", "start_ms": event.gap[0], "end_ms": event.gap[1]}
    return {
        "type": "status",
        "status": event.status.value,
        "source": event.stream_id,
        "stt_provider": event.provider_id,
        "previous_stt_provider": event.previous_provider_id,
        "reason": event.reason,
        "stt_state": event.state,
        "replay": replay,
        "continuity": gap,
    }


def _translation_metadata(event) -> dict:
    """Additive Translation keys; the current store reads none of them."""
    return {
        "translation_provider": event.provider_id,
        "translation_fallback": event.fallback,
        "source_lang": event.source_lang,
        "target_lang": event.target_lang,
        "detected_source_lang": event.detected_source_lang,
    }


def _translation_update_message(event) -> dict:
    """A final's translation in the legacy translation_update shape, matched on entry_id.

    ``revision`` is the final's own source revision, so the store's revision check passes.
    """
    msg = {
        "type": "translation_update",
        "entry_id": event.entry_id,
        "source_text": event.source_text,
        "translation": event.translation,
        "timestamp": time.time(),
        "is_final": True,
        "translation_backend": event.provider_id,
        "revision": event.source_revision,
        "item_id": event.item_key,
    }
    if event.stream_id in ("mic", "system"):
        msg["source"] = event.stream_id
    msg.update(_translation_metadata(event))
    return msg


def _translation_error_message(event) -> dict:
    """A final that will not be translated. The store ignores an update without text."""
    msg = {
        "type": "translation_update",
        "entry_id": event.entry_id,
        "source_text": event.source_text,
        "translation": None,
        "timestamp": time.time(),
        "is_final": True,
        "revision": event.source_revision,
        "item_id": event.item_key,
        "translation_provider": event.provider_id,
        "translation_error": {"code": event.code, "message": event.message},
    }
    if event.stream_id in ("mic", "system"):
        msg["source"] = event.stream_id
    return msg


def _translation_status_message(event) -> dict:
    """Translation transitions ride on ``status`` with values the store does not act on."""
    return {
        "type": "status",
        "status": event.status.value,
        "translation_provider": event.provider_id,
        "previous_translation_provider": event.previous_provider_id,
        "reason": event.reason,
        "translation_state": event.state,
    }


class EchoFluxEngine:
    def __init__(
        self,
        insecure_dev: bool = False,
        conversation_repo: Optional[ConversationRepository] = None,
    ):
        # Tokens are consumed from the environment here and live only inside the server,
        # so they never reach Config, config dumps, logs, or child processes.
        tokens = BootstrapTokens.from_environ(os.environ)
        self._ws_server = WebSocketServer(
            host=os.getenv("ECHOFLUX_HOST", "127.0.0.1"),
            port=int(os.getenv("ECHOFLUX_PORT", "8765")),
            tokens=tokens,
            insecure_dev=insecure_dev,
        )
        self._main_task: Optional[asyncio.Task] = None

        self._inputs = {}
        self._vads = {}
        self._audio_queues = {}

        self._session_config = None
        self._stt_session: Optional[STTSessionOrchestrator] = None
        self._last_entry_stamp = 0.0
        # Composition seams (default to built-in local/cloud builders and ephemeral start secrets).
        self._stt_local_builder = self._build_local_stt
        self._stt_cloud_builder = None
        self._stt_secrets_resolver = None
        self._translation_config = None
        self._translation_session: Optional[TranslationSessionOrchestrator] = None
        self._translation_local_builder = None
        self._translation_cloud_builder = None
        self._translation_secrets_resolver = None
        # Latest partial message per stream; a partial translation is re-sent in its shape.
        self._partial_messages: Dict[str, dict] = {}

        self._running = False

        self._result_queue = Queue()

        self._capture_threads = []
        self._process_threads = []
        self._result_task: Optional[asyncio.Task] = None

        self._client = None
        self._current_config = {}
        self._conversation_repo = (
            conversation_repo if conversation_repo is not None else ConversationRepository()
        )
        self._active_conversation_id: Optional[str] = None
        self._active_session_id: Optional[str] = None
        self._ephemeral_secrets: Dict[str, str] = {}
        self._service_secrets: Dict[str, Dict[str, str]] = {}
        self._assistant_service = None
        self._summary_service = None
        self._owned_assistant_provider = None
        self._assistant_tasks: Set[asyncio.Task] = set()
        self._summary_tasks: Dict[str, asyncio.Task] = {}

        from engine.server.control_dispatcher import ControlCommandDispatcher

        self._control_dispatcher = ControlCommandDispatcher(
            engine=self, conversation_repo=self._conversation_repo
        )
        self._ws_server.set_control_dispatcher(self._control_dispatcher)

    @property
    def control_dispatcher(self):
        return self._control_dispatcher

    async def start(self):
        self._loop = asyncio.get_running_loop()

        self._ws_server.on_start(self._on_client_start)
        self._ws_server.on_stop(self._on_client_stop)
        self._ws_server.on_suggestion(self._on_suggestion_request)
        self._ws_server.on_summary(self._on_summary_request)

        logger.info("EchoFlux Engine starting...")
        self._main_task = asyncio.current_task()
        await self._ws_server.start()
        if self._ws_server.is_authenticated:
            self._start_stdin_watch()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self._shutdown()

    def _start_stdin_watch(self):
        """Tauri requests shutdown by closing our stdin. Only armed when supervised
        (bootstrap tokens present): a detached legacy run may have no stdin at all."""
        if sys.stdin is None:
            return
        t = threading.Thread(target=self._watch_stdin, name="StdinWatch", daemon=True)
        t.start()

    def _watch_stdin(self):
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        try:
            while stream.read(1024):
                pass
        except (OSError, ValueError) as e:
            # Unreadable stdin is not a shutdown request; the supervisor can still kill us.
            logger.debug("Stdin watch disabled (%s)", type(e).__name__)
            return
        logger.info("Supervisor closed stdin; shutting down")
        self._loop.call_soon_threadsafe(self._request_shutdown)

    def _request_shutdown(self):
        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()

    def _extract_start_secrets(
        self, config: dict, secrets: Optional[dict]
    ) -> tuple[dict, Dict[str, str], Dict[str, Dict[str, str]]]:
        """Pull injected credential values out of start config before it reaches _current_config."""
        from engine.core.config import is_secret_key_name

        cleaned_config = dict(config) if isinstance(config, dict) else {}
        flat_secrets: Dict[str, str] = {}
        service_secrets: Dict[str, Dict[str, str]] = {"stt": {}, "translation": {}, "assistant": {}}

        if isinstance(secrets, dict):
            for key, value in secrets.items():
                if not isinstance(key, str) or not isinstance(value, str) or not value:
                    continue
                if "." in key:
                    cap, slot = key.split(".", 1)
                    if cap in service_secrets and slot:
                        service_secrets[cap][slot] = value
                else:
                    flat_secrets[key] = value

        for key in list(cleaned_config.keys()):
            if not isinstance(key, str) or "." not in key:
                continue
            prefix, leaf = key.split(".", 1)
            if prefix in service_secrets and is_secret_key_name(leaf.split(".")[-1]):
                val = cleaned_config.pop(key)
                if isinstance(val, str) and val:
                    service_secrets[prefix][leaf] = val
            elif key == "llm.api_key":
                val = cleaned_config.pop(key)
                if isinstance(val, str) and val:
                    flat_secrets["api_key"] = val
                    service_secrets["assistant"].setdefault("api_key", val)

        return cleaned_config, flat_secrets, service_secrets

    async def _on_client_start(self, config: dict, secrets: Optional[dict] = None):
        # Log the shape only: the start config carries provider settings and API keys.
        key_count = len(config) if isinstance(config, dict) else 0
        logger.info("Starting pipeline (%d client config keys)", key_count)
        self._client = self._ws_server.current_requester()
        cleaned_config, flat_secrets, service_secrets = self._extract_start_secrets(config, secrets)

        if self._running:
            await self._stop_pipeline()

        self._ephemeral_secrets = flat_secrets
        self._service_secrets = service_secrets

        try:
            await asyncio.to_thread(self._initialize_pipeline, cleaned_config)
            # Translation first: a refused Translation Start never starts an STT provider.
            await self._start_translation_session()
            # Providers and the mandatory Local fallback are ready before any device opens.
            await self._start_stt_session()
            # Assistant is optional and must never prevent STT/Translation from starting.
            await self._start_assistant_service(cleaned_config)
            self._start_threads()
        except Exception as e:
            logger.error("Failed to start pipeline: %s", e, exc_info=True)
            await self._abort_pipeline_start()
            if self._client:
                await self._client.send(json.dumps({"type": "error", "message": str(e)}))
            return False

    async def _on_client_stop(self):
        await self._cancel_assistant_work()
        await self._stop_pipeline()

    async def _shutdown(self):
        await self._cancel_assistant_work()
        await self._stop_pipeline()
        await self._close_owned_assistant_provider()
        await self._ws_server.stop()

    def _initialize_pipeline(self, settings: dict):
        self._current_config = settings

        sample_rate = int(os.getenv("ECHOFLUX_SAMPLE_RATE", "16000"))
        chunk_ms = int(os.getenv("ECHOFLUX_CHUNK_MS", "20"))

        audio_config = {
            "sample_rate": sample_rate,
            "channels": 1,
            "chunk_ms": chunk_ms,
        }

        # 1. Audio Source
        # Prefer settings dict (sent by the UI), fall back to env for CLI usage.
        audio_source = settings.get("audio.source", os.getenv("ECHOFLUX_AUDIO_SOURCE", "microphone"))
        mic_id_str = settings.get("audio.mic_device_id") or os.getenv("ECHOFLUX_MIC_DEVICE_ID")
        spk_id_str = settings.get("audio.speaker_device_id") or os.getenv("ECHOFLUX_SPEAKER_DEVICE_ID")
        # Legacy single-device env var (still supported for backwards-compat)
        legacy_device_id = os.getenv("ECHOFLUX_AUDIO_DEVICE_ID")

        self._inputs.clear()
        if audio_source == "both":
            from engine.audio.microphone import MicrophoneInput
            from engine.audio.system_audio import SystemAudioInput
            mic_dev = int(mic_id_str) if mic_id_str else None
            self._inputs["mic"] = MicrophoneInput(audio_config, device_id=mic_dev)
            self._inputs["system"] = SystemAudioInput(audio_config, device_id=spk_id_str)
            logger.info(
                "Audio source: True Dual Stream (mic_device=%s, speaker_device=%s)",
                mic_id_str or "default",
                spk_id_str or "default",
            )
        elif audio_source == "system":
            from engine.audio.system_audio import SystemAudioInput
            device_id = spk_id_str or legacy_device_id
            self._inputs["system"] = SystemAudioInput(audio_config, device_id=device_id)
        else:
            from engine.audio.microphone import MicrophoneInput
            raw_id = mic_id_str or legacy_device_id
            dev_id = int(raw_id) if raw_id else None
            self._inputs["mic"] = MicrophoneInput(audio_config, device_id=dev_id)

        # 2. VAD & Queues
        from engine.audio.vad import VAD
        self._vads.clear()
        self._audio_queues.clear()

        for stream_id in self._inputs:
            self._audio_queues[stream_id] = Queue(maxsize=500)
            self._vads[stream_id] = VAD({
                "enabled": settings.get("vad.enabled", True),
                "threshold": settings.get("vad.threshold", 0.5),
                "sample_rate": sample_rate,
            })

        # 3. STT session config. Resolved once, before any provider or audio device starts.
        self._session_config = resolve_session_config(
            settings,
            tuple(self._inputs),
            sample_rate=sample_rate,
            compute_type=os.getenv("ECHOFLUX_COMPUTE_TYPE", "float16"),
        )

        # 4. Translation.
        self._translation_config = resolve_translation_session_config(settings)

        # 5. Conversation and Session Tracking
        repo = self._conversation_repo
        if repo is not None:
            conv_id = settings.get("conversation_id") or settings.get("conversationId")
            if conv_id and repo.get_conversation(conv_id) is not None:
                self._active_conversation_id = conv_id
            else:
                conv_id = conv_id or new_record_id()
                now_ms = time.time() * 1000.0
                raw_labels = settings.get("source_labels")
                conv_kwargs: Dict[str, Any] = {
                    "conversation_id": conv_id,
                    "title": settings.get("conversation_title") or f"Conversation {conv_id[:8]}",
                    "created_at_ms": now_ms,
                    "updated_at_ms": now_ms,
                }
                if isinstance(raw_labels, dict):
                    conv_kwargs["source_labels"] = {
                        k: str(v) for k, v in raw_labels.items() if k in ("mic", "system") and str(v).strip()
                    }
                repo.create_conversation(ConversationRecord(**conv_kwargs))
                self._active_conversation_id = conv_id

            session_id = new_record_id()
            providers: Dict[str, ProviderMetadata] = {}
            if self._session_config and self._session_config.primary:
                providers["stt"] = ProviderMetadata(
                    provider_id=self._session_config.primary.provider_id,
                    model=self._session_config.primary.model or None,
                )
            if self._translation_config and self._translation_config.enabled and self._translation_config.primary:
                providers["translation"] = ProviderMetadata(
                    provider_id=self._translation_config.primary.provider_id,
                    model=self._translation_config.primary.model or None,
                )

            profile_svc = ContextProfileService(repo)
            raw_profiles = settings.get("attached_profile_ids")
            selected_profile_ids: Optional[list] = None
            if isinstance(raw_profiles, list) and raw_profiles:
                existing = {p.profile_id for p in profile_svc.list_profiles()}
                valid_ids = [
                    str(pid) for pid in raw_profiles if isinstance(pid, str) and pid in existing
                ]
                if valid_ids:
                    selected_profile_ids = valid_ids
                    if not profile_svc.attached_profile_ids(self._active_conversation_id):
                        with contextlib.suppress(Exception):
                            profile_svc.attach_profiles(self._active_conversation_id, valid_ids)

            session_record = SessionRecord(
                session_id=session_id,
                conversation_id=self._active_conversation_id,
                started_at_ms=time.time() * 1000.0,
                providers=providers,
            )
            profile_svc.start_session(session_record, profile_ids=selected_profile_ids)
            self._active_session_id = session_id

        if (
            self._assistant_service is None
            and self._conversation_repo is not None
            and getattr(self, "_assistant_provider", None) is not None
        ):
            self.configure_assistant_services(
                self._conversation_repo, self._assistant_provider
            )

    def _start_threads(self):
        self._running = True
        self._ws_server.is_capturing = True

        # Clear queues
        for q in self._audio_queues.values():
            with q.mutex:
                q.queue.clear()

        with self._result_queue.mutex:
            self._result_queue.queue.clear()

        # Start audio streams sequentially to avoid PortAudio concurrency issues
        for stream_id, audio_input in self._inputs.items():
            try:
                audio_input.start()
            except Exception as e:
                logger.error("Failed to start audio input %s: %s", stream_id, e)
                raise

        # Threads
        self._capture_threads = []
        self._process_threads = []

        for stream_id, audio_input in self._inputs.items():
            ct = threading.Thread(
                target=self._capture_loop,
                args=(stream_id, audio_input, self._audio_queues[stream_id]),
                name=f"Capture-{stream_id}"
            )
            ct.daemon = True
            ct.start()
            self._capture_threads.append(ct)

            pt = threading.Thread(
                target=self._session_process_loop,
                args=(
                    stream_id,
                    self._audio_queues[stream_id],
                    self._vads[stream_id],
                    self._stt_session,
                ),
                name=f"Process-{stream_id}"
            )
            pt.daemon = True
            pt.start()
            self._process_threads.append(pt)

        self._result_task = asyncio.create_task(self._broadcast_loop())

    async def _stop_pipeline(self):
        if not self._running:
            return

        logger.info("Stopping pipeline...")
        self._running = False
        self._ws_server.is_capturing = False

        for inp in self._inputs.values():
            inp.stop()

        for ct in self._capture_threads:
            ct.join(timeout=1.0)

        for pt in self._process_threads:
            pt.join(timeout=2.0)

        if self._stt_session is not None:
            # Stops every provider once and queues trailing finals before the broadcast flush.
            session, self._stt_session = self._stt_session, None
            await session.stop()
        self._session_config = None
        if self._translation_session is not None:
            # After STT, so it saw the trailing finals; requests still running are cancelled.
            translation, self._translation_session = self._translation_session, None
            await translation.stop()
        self._translation_config = None
        self._partial_messages.clear()
        if self._active_session_id and self._conversation_repo:
            try:
                self._conversation_repo.end_session(self._active_session_id)
            except Exception as e:
                logger.warning("Failed to end session in repo: %s", e)
            self._active_session_id = None
        self._ephemeral_secrets.clear()
        self._service_secrets.clear()

        # Give the broadcast loop a tiny moment to flush the final results
        await asyncio.sleep(0.1)

        if self._result_task:
            self._result_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._result_task

        # NOTE: _owned_assistant_provider is intentionally NOT cleared here so
        # Summarize/Assist continue working after the pipeline stops.

        self._inputs.clear()
        self._vads.clear()
        self._audio_queues.clear()
        self._capture_threads.clear()
        self._process_threads.clear()
        logger.info("Pipeline stopped.")

    def _capture_loop(self, stream_id: str, audio_input, audio_queue: Queue):
        logger.info("Capture thread [%s] started.", stream_id)
        chunk_count = 0
        try:
            while self._running:
                chunk = audio_input.read_chunk()
                if chunk:
                    chunk_count += 1
                    if chunk_count % 100 == 0:
                        logger.debug("Capture [%s]: Read 100 chunks. Current queue size: %d", stream_id, audio_queue.qsize())

                    if not audio_queue.full():
                        audio_queue.put(chunk)
                    else:
                        logger.warning("Capture [%s]: Queue is full! Dropping chunk", stream_id)
                else:
                    time.sleep(0.005)
        except Exception as e:
            logger.error("Capture thread [%s] error: %s", stream_id, e, exc_info=True)
        finally:
            logger.info("Capture thread [%s] ended.", stream_id)

    # --- Realtime STT session -------------------------------------------------------------
    #
    # Lifecycle: _initialize_pipeline resolves the session config on a worker thread ->
    # _start_stt_session readies providers (no device open yet) -> _start_threads opens audio ->
    # Process-{stream} runs VAD and hands audio to the session -> _stop_pipeline stops the
    # session once. Any Start failure goes through _abort_pipeline_start.

    async def _close_owned_assistant_provider(self) -> None:
        provider, self._owned_assistant_provider = self._owned_assistant_provider, None
        if provider is not None:
            with contextlib.suppress(Exception):
                await provider.stop()

    async def _start_assistant_service(self, settings: dict) -> None:
        """Compose repository-backed Assist and Summary from v2 ``services.assistant``."""
        if getattr(self, "_assistant_provider", None) is not None:
            return
        services = settings.get("services")
        section = settings.get("services.assistant")
        if section is None and isinstance(services, dict):
            section = services.get("assistant")
        if not isinstance(section, dict):
            return

        await self._close_owned_assistant_provider()
        if not section.get("enabled", False):
            self.configure_assistant_services(None, None)
            return

        assistant_secrets = dict(self._service_secrets.pop("assistant", {}))
        try:
            from engine.assistant.summary_service import (
                EngineAssistantPromptPolicy,
                SummaryPromptPolicy,
            )
            from engine.core.config import ServiceConfig
            from engine.providers.manifests import ProtocolFamily
            from engine.providers.registry import build_default_registry
            from engine.session.translation_orchestrator import _generation_options, _text_strategy

            service_cfg = ServiceConfig.from_dict("assistant", section)
            registry = build_default_registry()
            resolved = registry.resolve_service(service_cfg)
            for slot in resolved.credential_refs:
                if slot not in assistant_secrets and slot in self._ephemeral_secrets:
                    assistant_secrets[slot] = self._ephemeral_secrets[slot]

            factory = await asyncio.to_thread(
                registry.load_factory, resolved.capability, resolved.provider_id
            )
            strategy = _text_strategy(resolved, assistant_secrets)
            summary_policy = SummaryPromptPolicy()
            prompt_policy = EngineAssistantPromptPolicy(summary_policy=summary_policy)
            provider = factory(
                strategy,
                resolved.model,
                prompt_policy=prompt_policy,
                provider_id=resolved.provider_id,
                generation=_generation_options(resolved),
            )
            if resolved.protocol_family not in (
                ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
                ProtocolFamily.GEMINI_TEXT,
            ):
                logger.warning("Unsupported assistant protocol family: %s", resolved.protocol_family)
                return
            await provider.start()
            self._owned_assistant_provider = provider
            self.configure_assistant_services(
                self._conversation_repo, provider, model=resolved.model
            )
        except Exception as exc:
            # Assistant setup failure never halts STT/Translation capture.
            logger.warning("Assistant service could not be started (%s)", type(exc).__name__)
            self.configure_assistant_services(None, None)
        finally:
            assistant_secrets.clear()

    async def _start_stt_session(self):
        secrets_resolver = self._stt_secrets_resolver
        if secrets_resolver is None:
            def secrets_resolver(resolved):
                cap_secrets = self._service_secrets.get(resolved.capability.value, {})
                return cap_secrets if cap_secrets else self._ephemeral_secrets
        session = STTSessionOrchestrator(
            self._session_config,
            self._on_stt_event,
            cloud_builder=self._stt_cloud_builder,
            local_builder=self._stt_local_builder,
            secrets_resolver=secrets_resolver,
        )
        self._stt_session = session
        await session.start()

    async def _start_translation_session(self):
        """Ready the Translation session. Translation off or a legacy config starts nothing."""
        config = self._translation_config
        if config is None or not config.enabled:
            return
        secrets_resolver = self._translation_secrets_resolver
        if secrets_resolver is None:
            def secrets_resolver(resolved):
                cap_secrets = self._service_secrets.get(resolved.capability.value, {})
                return cap_secrets if cap_secrets else self._ephemeral_secrets
        session = TranslationSessionOrchestrator(
            config,
            self._on_translation_event,
            cloud_builder=self._translation_cloud_builder,
            local_builder=self._translation_local_builder,
            secrets_resolver=secrets_resolver,
        )
        self._translation_session = session
        await session.start()

    async def _abort_pipeline_start(self):
        """Undo a failed Start: nothing stays loaded, open, or running."""
        if self._running:
            await self._stop_pipeline()
            return
        if self._stt_session is not None:
            session, self._stt_session = self._stt_session, None
            await session.stop()
        if self._translation_session is not None:
            translation, self._translation_session = self._translation_session, None
            await translation.stop()
        self._translation_config = None
        self._session_config = None
        if self._active_session_id and self._conversation_repo:
            try:
                self._conversation_repo.end_session(self._active_session_id)
            except Exception as e:
                logger.warning("Failed to end session in repo during abort: %s", e)
            self._active_session_id = None
        self._ephemeral_secrets.clear()
        self._service_secrets.clear()
        self._inputs.clear()
        self._vads.clear()
        self._audio_queues.clear()

    def _build_local_stt(self, session_config):
        """Local STT for a session. A legacy start keeps the Whisper size the user picked."""
        from engine.providers.stt.local import LocalSTTProvider, LocalSTTSettings

        settings = LocalSTTSettings.from_options(dict(session_config.fallback.options))
        return LocalSTTProvider(
            settings,
            provider_id=session_config.fallback.provider_id,
            stream_ids=session_config.stream_ids,
        )

    def _session_process_loop(self, stream_id: str, audio_queue: Queue, vad, session):
        """VAD and silence endpointing for one stream; recognition runs in the session."""
        logger.info("Processing thread [%s] started.", stream_id)
        was_speech = False
        silence_start_time = None
        try:
            while self._running:
                chunks = []
                try:
                    chunks.append(audio_queue.get(timeout=0.1))
                    while not audio_queue.empty() and len(chunks) < 10:
                        try:
                            chunks.append(audio_queue.get_nowait())
                        except Empty:
                            break
                except Empty:
                    silent_for = time.time() - (silence_start_time or time.time())
                    if was_speech and silent_for >= SILENCE_FINALIZE_DELAY:
                        session.request_endpoint(stream_id)
                        was_speech = False
                        silence_start_time = None
                    continue

                combined_audio = b"".join(chunks)
                is_speech = vad.process(combined_audio)
                session.submit_audio(stream_id, combined_audio, speech=is_speech)
                if is_speech:
                    silence_start_time = None
                    was_speech = True
                elif was_speech:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    if time.time() - silence_start_time >= SILENCE_FINALIZE_DELAY:
                        logger.info("Process [%s]: silence timeout; finalizing.", stream_id)
                        session.request_endpoint(stream_id)
                        was_speech = False
                        silence_start_time = None
                        vad.reset()
        except Exception as e:
            logger.error("Processing thread [%s] error: %s", stream_id, e, exc_info=True)
        finally:
            logger.info("Processing thread [%s] ended.", stream_id)

    def _on_stt_event(self, event):
        """Session events arrive on the event loop; only non-blocking queue handoffs here."""
        if isinstance(event, STTStatusEvent):
            self._result_queue.put(_stt_status_message(event))
            if event.status is STTStatusKind.FAILED:
                self._result_queue.put(
                    {
                        "type": "error",
                        "message": "Speech recognition stopped for one audio source.",
                    }
                )
            return

        entry_id = None
        if event.is_final:
            # Unique per final, in the legacy e-<timestamp> form.
            stamp = max(time.time(), self._last_entry_stamp + 1e-6)
            self._last_entry_stamp = stamp
            entry_id = f"e-{stamp}"
            if self._conversation_repo and self._active_conversation_id:
                try:
                    msg_record = TranscriptMessage(
                        message_id=entry_id,
                        source=event.stream_id if event.stream_id in ("mic", "system") else "unknown",
                        text=event.text,
                        timestamp_ms=time.time() * 1000.0,
                        is_final=True,
                        revision=event.revision,
                        session_id=self._active_session_id,
                        detected_language=event.language,
                        stt_provider=ProviderMetadata(provider_id=event.provider_id) if event.provider_id else None,
                    )
                    self._conversation_repo.upsert_message(self._active_conversation_id, msg_record)
                except Exception as e:
                    logger.warning("Failed to upsert transcript message to repo: %s", e)

        msg = _stt_transcript_message(
            event,
            entry_id,
            None,
            conversation_id=self._active_conversation_id,
            message_id=entry_id if event.is_final else None,
        )
        self._result_queue.put(msg)
        session = self._translation_session
        if session is not None:
            if event.is_final:
                self._partial_messages.pop(event.stream_id, None)
            else:
                self._partial_messages[event.stream_id] = msg
            # Records the change and schedules work; translations arrive later on this loop.
            session.on_transcript(event, entry_id)

    def _on_translation_event(self, event):
        """Translation session events arrive on the event loop; only queue handoffs here."""
        if isinstance(event, TranslationStatusEvent):
            self._result_queue.put(_translation_status_message(event))
            if event.status is TranslationStatusKind.FAILED:
                self._result_queue.put(
                    {"type": "error", "message": "Translation stopped for this session."}
                )
            return
        if isinstance(event, TranslationErrorEvent):
            self._result_queue.put(_translation_error_message(event))
            return
        if event.is_final:
            self._result_queue.put(_translation_update_message(event))
            if self._conversation_repo and self._active_conversation_id and event.entry_id:
                try:
                    provider_meta = ProviderMetadata(provider_id=event.provider_id) if event.provider_id else None
                    self._conversation_repo.set_translation(
                        self._active_conversation_id,
                        event.entry_id,
                        event.translation,
                        provider=provider_meta,
                        revision=event.source_revision,
                    )
                except Exception as e:
                    logger.warning("Failed to attach translation in repo: %s", e)
            return
        # A partial translation rides on its caption, and only while that caption is shown.
        caption = self._partial_messages.get(event.stream_id)
        if caption is None or caption.get("item_id") != event.item_key:
            return
        if (caption.get("text") or "").strip() != event.source_text:
            return
        self._result_queue.put(
            {
                **caption,
                "translation": event.translation,
                "timestamp": time.time(),
                **_translation_metadata(event),
            }
        )

    async def _broadcast_loop(self):
        while self._running:
            try:
                count = 0
                while not self._result_queue.empty() and count < 10:
                    msg_dict = self._result_queue.get_nowait()
                    if self._ws_server:
                        await self._ws_server.broadcast(msg_dict)
                    count += 1

                if count == 0:
                    await asyncio.sleep(0.02)
            except Exception as e:
                logger.error("Broadcast error: %s", e)
                await asyncio.sleep(1)

    async def _on_suggestion_request(self, message: dict, websocket):
        """Handle request_suggestion from UI — routes to repository-backed assist."""
        await self._on_conversation_assist(message, websocket)

    async def _on_summary_request(self, message: dict, websocket):
        """Handle request_summary from UI — routes to repository-backed summary."""
        await self._on_conversation_summary(message, websocket)

    # --- Repository-backed Assist and Summary ---------------------------------------------
    #
    # This path reads context only from the conversation repository,
    # refuses client-assembled context, needs an authenticated control connection (or explicit
    # --insecure-dev), and always ends with exactly one suggestion_result or llm_done.

    def configure_assistant_services(
        self,
        repository: Any,
        provider: Any,
        model: Optional[str] = None,
        summary_budget: Any = None,
    ) -> None:
        """Compose repository-backed Assist and Summary around one Assistant provider.

        ``repository`` is a ConversationRepository; ``provider`` an AssistantProvider built with
        ``EngineAssistantPromptPolicy``. The caller owns the provider lifecycle and credentials:
        nothing here reads keys from commands. None for either clears the services.
        """
        if repository is None or provider is None:
            self._assistant_service = None
            self._summary_service = None
            logger.info("Repository-backed assistant services cleared")
            return

        from engine.assistant.service import AssistantService
        from engine.assistant.summary_service import SummaryService
        from engine.conversation.context_profiles import ContextProfileService

        self._assistant_service = AssistantService(
            repository, ContextProfileService(repository), provider, model=model
        )
        self._summary_service = SummaryService(
            repository, provider, model=model, budget=summary_budget
        )
        logger.info(
            "Repository-backed assistant services configured: provider=%s", provider.provider_id
        )

    def _conversation_requests_allowed(self) -> bool:
        # Event clients never reach dispatch. The unauthenticated legacy socket must not read
        # stored conversations.
        return (
            self._ws_server.is_authenticated
            or self._ws_server.security_mode is SecurityMode.INSECURE_DEV
        )

    async def _send_reply(self, websocket, message: dict) -> None:
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.debug("Dropped assistant reply for closed client (%s)", type(e).__name__)

    def _track_assistant_task(self, coro) -> None:
        # Detached so a slow provider never holds the control connection's dispatch loop.
        task = asyncio.create_task(coro)
        self._assistant_tasks.add(task)
        task.add_done_callback(self._assistant_tasks.discard)

    async def _cancel_assistant_work(self):
        """Cancel in-flight repository-backed requests; each still sends its terminal reply."""
        pending = [task for task in self._assistant_tasks if not task.done()]
        if not pending:
            return
        logger.info("Cancelling %d in-flight assistant request(s)", len(pending))
        for task in pending:
            task.cancel()
        await asyncio.wait(pending, timeout=2.0)

    async def _on_conversation_assist(self, message: dict, websocket):
        # Wire contract reply: {"type": "suggestion_result"}
        echo_id = safe_command_id(message, "message_id") or safe_command_id(message, "entry_id")
        if not self._conversation_requests_allowed():
            await self._send_reply(websocket, _assist_failure(echo_id, ERROR_FORBIDDEN))
            return
        try:
            command = parse_assist_command(message)
        except ProtocolError:
            await self._send_reply(websocket, _assist_failure(echo_id, ERROR_INVALID_REQUEST))
            return
        service = self._assistant_service
        if service is None:
            reply = _assist_failure(
                command.message_id, ERROR_ASSISTANT_UNAVAILABLE, command.conversation_id
            )
            await self._send_reply(websocket, reply)
            return
        self._track_assistant_task(self._run_assist(service, command, websocket))

    async def _run_assist(self, service, command, websocket):
        try:
            outcome = await service.request_assist(command.conversation_id, command.message_id)
        except asyncio.CancelledError:
            reply = _assist_failure(command.message_id, ERROR_CANCELLED, command.conversation_id)
            await asyncio.shield(self._send_reply(websocket, reply))
            raise
        except Exception as e:
            logger.error("Assist request failed unexpectedly (%s)", type(e).__name__)
            reply = _assist_failure(command.message_id, ERROR_INTERNAL, command.conversation_id)
        else:
            reply = _assist_reply(command, outcome)
        await asyncio.shield(self._send_reply(websocket, reply))

    async def _send_summary_failure(
        self,
        websocket,
        code: str,
        conversation_id: str,
        after_chunks: bool = False,
        message: Optional[str] = None,
        retryable: bool = False,
        provider_code: Optional[str] = None,
        warning: Optional[dict] = None,
    ) -> None:
        chunk, done = summary_error_replies(
            code,
            message or ASSISTANT_ERROR_MESSAGES[code],
            conversation_id,
            after_chunks=after_chunks,
            retryable=retryable,
            provider_code=provider_code,
            warning=warning,
        )
        await self._send_reply(websocket, chunk)
        await self._send_reply(websocket, done)

    def cancel_summary(self, conversation_id: Optional[str] = None) -> bool:
        """Cancel an in-flight summary task by conversation_id, or any active summary."""
        if conversation_id and conversation_id in self._summary_tasks:
            task = self._summary_tasks.get(conversation_id)
            if task is not None and not task.done():
                task.cancel()
                return True
            return False
        cancelled = False
        for task in list(self._summary_tasks.values()):
            if not task.done():
                task.cancel()
                cancelled = True
        return cancelled

    async def _on_conversation_summary(self, message: dict, websocket):
        # Wire contract replies: {"type": "llm_chunk"}, {"type": "llm_done"}
        echo_id = safe_command_id(message, "conversation_id")
        if not self._conversation_requests_allowed():
            await self._send_summary_failure(websocket, ERROR_FORBIDDEN, echo_id)
            return
        try:
            command = parse_summary_command(message)
        except ProtocolError:
            await self._send_summary_failure(websocket, ERROR_INVALID_REQUEST, echo_id)
            return
        service = self._summary_service
        if service is None:
            await self._send_summary_failure(
                websocket, ERROR_ASSISTANT_UNAVAILABLE, command.conversation_id
            )
            return
        task = asyncio.create_task(self._run_summary(service, command, websocket))
        self._assistant_tasks.add(task)
        self._summary_tasks[command.conversation_id] = task

        def _cleanup_summary(done_task: asyncio.Task) -> None:
            self._assistant_tasks.discard(done_task)
            if self._summary_tasks.get(command.conversation_id) is done_task:
                self._summary_tasks.pop(command.conversation_id, None)

        task.add_done_callback(_cleanup_summary)

    async def _run_summary(self, service, command, websocket):
        conversation_id = command.conversation_id
        streamed = []

        async def _on_chunk(text: str) -> None:
            streamed.append(len(text))
            chunk_msg = summary_chunk_message(text, conversation_id)
            await self._send_reply(websocket, chunk_msg)
            if self._ws_server and self._ws_server.is_authenticated:
                await self._ws_server.broadcast(chunk_msg)

        try:
            outcome = await service.request_summary(conversation_id, on_chunk=_on_chunk)
        except asyncio.CancelledError:
            failure = self._send_summary_failure(
                websocket, ERROR_CANCELLED, conversation_id, after_chunks=bool(streamed)
            )
            await asyncio.shield(failure)
            raise
        except Exception as e:
            logger.error("Summary request failed unexpectedly (%s)", type(e).__name__)
            failure = self._send_summary_failure(
                websocket, ERROR_INTERNAL, conversation_id, after_chunks=bool(streamed)
            )
            await asyncio.shield(failure)
            return

        warning = _warning_payload(outcome.warning)
        if not outcome.ok:
            error = outcome.error
            failure = self._send_summary_failure(
                websocket,
                error.code.value,
                conversation_id,
                after_chunks=bool(streamed),
                message=error.message,
                retryable=error.retryable,
                provider_code=error.provider_code,
                warning=warning,
            )
            await asyncio.shield(failure)
            return

        done = summary_done_message(
            conversation_id=conversation_id,
            summary_id=outcome.stored.summary_id,
            warning=warning,
            provider=outcome.provider.to_dict() if outcome.provider is not None else None,
            included_message_count=outcome.included_message_count,
            omitted_message_count=outcome.omitted_message_count,
        )
        await asyncio.shield(self._send_reply(websocket, done))


def run_engine(config_path=None, insecure_dev: bool = False):
    _load_dotenv()
    _setup_logging()

    # Eagerly initialize PyTorch cuDNN to prevent DLL collision with CTranslate2 later on Windows
    try:
        import torch
        if torch.cuda.is_available():
            torch.nn.functional.conv2d(torch.zeros(1, 1, 1, 1, device="cuda"), torch.zeros(1, 1, 1, 1, device="cuda"))
    except Exception:
        pass

    install_progress_hijack()

    engine = EchoFluxEngine(insecure_dev=insecure_dev)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(engine.start())


if __name__ == "__main__":
    run_engine()
