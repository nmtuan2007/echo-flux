import asyncio
import json
import logging
import os
import sys
import threading
import time
from queue import Queue, Empty, Full
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

from engine.core.config import TranscriptionConfig
from engine.server.websocket_server import WebSocketServer

logger = logging.getLogger("echoflux.engine")

SILENCE_FINALIZE_DELAY = 0.8


def _setup_cuda_paths():
    """
    On Windows, manually add the NVIDIA library paths from the venv site-packages to PATH.
    This fixes 'cublas64_12.dll not found' errors with CTranslate2/Faster-Whisper.
    """
    if sys.platform != "win32":
        return

    # Typical paths where pip installs nvidia libs in venv
    venv_base = Path(sys.prefix)
    site_packages = venv_base / "Lib" / "site-packages"

    nvidia_libs = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
    ]

    added_count = 0
    for lib_path in nvidia_libs:
        if lib_path.exists():
            try:
                os.add_dll_directory(str(lib_path))
            except Exception:
                pass
            os.environ["PATH"] = str(lib_path) + os.pathsep + os.environ["PATH"]
            added_count += 1


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
    logger.info("Logging initialized – file: %s", log_file)


class EchoFluxEngine:
    def __init__(self):
        self._ws_server = WebSocketServer(
            host=os.getenv("ECHOFLUX_HOST", "127.0.0.1"),
            port=int(os.getenv("ECHOFLUX_PORT", "8765")),
        )

        self._audio_input = None
        self._asr_backend = None
        self._translation_backend = None
        self._vad = None

        # REMOVED: SentenceAccumulator to prevent delay

        self._running = False

        self._audio_queue = Queue(maxsize=500)
        self._translation_queue = Queue(maxsize=100)
        self._result_queue = Queue()

        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._translation_thread: Optional[threading.Thread] = None
        self._result_task: Optional[asyncio.Task] = None

        self._client = None
        self._current_config = {}

    async def start(self):
        self._ws_server.on_start(self._on_client_start)
        self._ws_server.on_stop(self._on_client_stop)

        logger.info("EchoFlux Engine starting...")
        await self._ws_server.start()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self._shutdown()

    async def _on_client_start(self, config: dict):
        logger.info("Starting pipeline with client config: %s", config)
        self._client = list(self._ws_server._clients)[0] if self._ws_server._clients else None

        if self._running:
            await self._stop_pipeline()

        try:
            self._initialize_pipeline(config)
            self._start_threads()
        except Exception as e:
            logger.error("Failed to start pipeline: %s", e, exc_info=True)
            if self._client:
                await self._client.send(json.dumps({"type": "error", "message": str(e)}))

    async def _on_client_stop(self):
        await self._stop_pipeline()

    async def _shutdown(self):
        await self._stop_pipeline()
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
        audio_source = os.getenv("ECHOFLUX_AUDIO_SOURCE", "microphone")
        device_id = os.getenv("ECHOFLUX_AUDIO_DEVICE_ID")

        if audio_source == "system":
            from engine.audio.system_audio import SystemAudioInput
            self._audio_input = SystemAudioInput(audio_config, device_id=device_id)
        else:
            from engine.audio.microphone import MicrophoneInput
            dev_id = int(device_id) if device_id else None
            self._audio_input = MicrophoneInput(audio_config, device_id=dev_id)

        # 2. VAD
        from engine.audio.vad import VAD
        self._vad = VAD({
            "enabled": settings.get("vad.enabled", True),
            "threshold": settings.get("vad.threshold", 0.5),
            "sample_rate": sample_rate,
        })

        # 3. ASR Backend
        from engine.asr.faster_whisper_backend import FasterWhisperBackend
        self._asr_backend = FasterWhisperBackend()

        asr_config = TranscriptionConfig(
            model_size=settings.get("asr.model_size", "small"),
            language=settings.get("asr.language", "en"),
            device=settings.get("asr.device", "auto"),
            compute_type=os.getenv("ECHOFLUX_COMPUTE_TYPE", "float16"),
        )
        self._asr_backend.load_model(asr_config)

        # 4. Translation Backend
        if settings.get("translation.enabled", False):
            from engine.translation.fallback_backend import FallbackTranslationBackend

            source_lang = settings.get("translation.source_lang", "en")
            target_lang = settings.get("translation.target_lang", "vi")
            backend = settings.get("translation.backend", "online")
            custom_model = settings.get("translation.model")

            translation_config = {
                "translation.backend": backend,
                "translation.source_lang": source_lang,
                "translation.target_lang": target_lang,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "device": settings.get("asr.device", "auto"),
            }

            if custom_model:
                translation_config["translation.model"] = custom_model

            logger.info(
                "Translation config: backend=%s, pair=%s→%s, model=%s",
                backend, source_lang, target_lang,
                custom_model or "(default preset)",
            )

            self._translation_backend = FallbackTranslationBackend()
            self._translation_backend.load_model(translation_config)

    def _start_threads(self):
        self._running = True

        # Clear queues
        with self._audio_queue.mutex:
            self._audio_queue.queue.clear()
        with self._translation_queue.mutex:
            self._translation_queue.queue.clear()
        with self._result_queue.mutex:
            self._result_queue.queue.clear()

        # Threads
        self._capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self._capture_thread.daemon = True
        self._capture_thread.start()

        self._process_thread = threading.Thread(target=self._process_loop, name="ProcessThread")
        self._process_thread.daemon = True
        self._process_thread.start()

        if self._translation_backend:
            self._translation_thread = threading.Thread(target=self._translation_loop, name="TranslationThread")
            self._translation_thread.daemon = True
            self._translation_thread.start()

        self._result_task = asyncio.create_task(self._broadcast_loop())

    async def _stop_pipeline(self):
        if not self._running:
            return

        logger.info("Stopping pipeline...")
        self._running = False

        if self._audio_input:
            self._audio_input.stop()

        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)

        if self._process_thread:
            self._process_thread.join(timeout=2.0)

        if self._translation_thread:
            self._translation_thread.join(timeout=2.0)

        if self._result_task:
            self._result_task.cancel()
            try:
                await self._result_task
            except asyncio.CancelledError:
                pass

        if self._asr_backend:
            result = self._asr_backend.finalize_current()
            if result:
                self._enqueue_asr_result(result)
            self._asr_backend.unload_model()
            self._asr_backend = None

        if self._translation_backend:
            self._translation_backend.unload_model()
            self._translation_backend = None

        self._audio_input = None
        self._vad = None
        logger.info("Pipeline stopped.")

    def _capture_loop(self):
        logger.info("Capture thread started.")
        try:
            self._audio_input.start()
            while self._running:
                chunk = self._audio_input.read_chunk()
                if chunk:
                    if not self._audio_queue.full():
                        self._audio_queue.put(chunk)
                else:
                    time.sleep(0.005)
        except Exception as e:
            logger.error("Capture thread error: %s", e, exc_info=True)
        finally:
            logger.info("Capture thread ended.")

    def _process_loop(self):
        logger.info("Processing thread started.")
        was_speech = False
        silence_start_time = None
        has_pending_audio = False

        try:
            while self._running:
                chunks = []
                try:
                    first = self._audio_queue.get(timeout=0.1)
                    chunks.append(first)
                    while not self._audio_queue.empty() and len(chunks) < 10:
                        try:
                            chunks.append(self._audio_queue.get_nowait())
                        except Empty:
                            break
                except Empty:
                    if has_pending_audio and silence_start_time is not None:
                        elapsed = time.time() - silence_start_time
                        if elapsed >= SILENCE_FINALIZE_DELAY:
                            self._do_finalize()
                            has_pending_audio = False
                            silence_start_time = None
                            was_speech = False
                    continue

                combined_audio = b"".join(chunks)
                is_speech = self._vad.process(combined_audio)

                if is_speech:
                    silence_start_time = None
                    has_pending_audio = True
                    was_speech = True
                    result = self._asr_backend.transcribe_stream(combined_audio)
                    if result:
                        self._enqueue_asr_result(result)
                else:
                    if was_speech and silence_start_time is None:
                        silence_start_time = time.time()
                    if silence_start_time is not None:
                        elapsed = time.time() - silence_start_time
                        if elapsed >= SILENCE_FINALIZE_DELAY and has_pending_audio:
                            self._do_finalize()
                            has_pending_audio = False
                            silence_start_time = None
                            was_speech = False
                            self._vad.reset()

        except Exception as e:
            logger.error("Processing thread error: %s", e, exc_info=True)
        finally:
            logger.info("Processing thread ended.")

    def _translation_loop(self):
        logger.info("Translation thread started.")
        src_lang = self._current_config.get("translation.source_lang", "auto")
        tgt_lang = self._current_config.get("translation.target_lang", "vi")

        while self._running:
            try:
                task = self._translation_queue.get(timeout=0.5)
            except Empty:
                continue

            text = task.get("text", "")

            if not text:
                continue

            try:
                trans_res = self._translation_backend.translate(text, src_lang, tgt_lang)
                if trans_res.translated_text:
                    active_backend = getattr(self._translation_backend, "active_backend", None)

                    msg = {
                        "type": "translation_update",
                        "text": None,
                        "source_text": text,
                        "translation": trans_res.translated_text,
                        "timestamp": time.time(),
                        "is_final": True
                    }
                    if active_backend:
                        msg["translation_backend"] = active_backend

                    self._result_queue.put(msg)
                    logger.debug("Translated: '%s' -> '%s'", text, trans_res.translated_text)
            except Exception as e:
                logger.error("Translation loop error: %s", e)

        logger.info("Translation thread ended.")

    def _do_finalize(self):
        result = self._asr_backend.finalize_current()
        if result:
            self._enqueue_asr_result(result)

    def _enqueue_asr_result(self, asr_result):
        """
        Handle ASR result.
        IMPROVEMENT: Removed sentence accumulation.
        Every FINAL ASR segment is sent to translation queue immediately.
        """

        active_backend = None
        if self._translation_backend and hasattr(self._translation_backend, "active_backend"):
            active_backend = self._translation_backend.active_backend

        # Send ASR (English) to UI
        msg = {
            "type": "partial" if not asr_result.is_final else "final",
            "text": asr_result.text,
            "translation": None,
            "is_final": asr_result.is_final,
            "timestamp": time.time(),
        }

        if asr_result.is_final:
            msg["entry_id"] = f"e-{time.time()}"

        if active_backend:
            msg["translation_backend"] = active_backend

        self._result_queue.put(msg)

        # Translation Logic: Queue immediately if FINAL
        if self._translation_backend and self._translation_backend.is_loaded:
            if asr_result.is_final and asr_result.text.strip():
                try:
                    self._translation_queue.put({
                        "text": asr_result.text,
                        "is_final": True
                    }, timeout=0.5)
                except Full:
                    pass

    async def _broadcast_loop(self):
        while self._running:
            try:
                count = 0
                while not self._result_queue.empty() and count < 10:
                    msg_dict = self._result_queue.get_nowait()
                    if self._client:
                        await self._client.send(json.dumps(msg_dict))
                    count += 1

                if count == 0:
                    await asyncio.sleep(0.02)
            except Exception as e:
                logger.error("Broadcast error: %s", e)
                await asyncio.sleep(1)


def run_engine(config_path=None):
    _load_dotenv()
    _setup_cuda_paths()
    _setup_logging()

    engine = EchoFluxEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_engine()
