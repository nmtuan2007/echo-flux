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
from engine.core.progress import install_progress_hijack, set_progress_callback
from engine.server.websocket_server import WebSocketServer

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

            try:
                os.add_dll_directory(abs_path)
            except Exception:
                pass

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
    logger.info("Logging initialized – file: %s", log_file)


class EchoFluxEngine:
    def __init__(self):
        self._ws_server = WebSocketServer(
            host=os.getenv("ECHOFLUX_HOST", "127.0.0.1"),
            port=int(os.getenv("ECHOFLUX_PORT", "8765")),
        )

        self._inputs = {}
        self._vads = {}
        self._audio_queues = {}
        
        self._asr_backend = None
        self._translation_backend = None

        self._running = False

        self._translation_queue = Queue(maxsize=100)
        self._result_queue = Queue()

        self._capture_threads = []
        self._process_threads = []
        self._translation_thread: Optional[threading.Thread] = None
        self._result_task: Optional[asyncio.Task] = None

        self._client = None
        self._current_config = {}
        self._llm_assistant = None  # Created on each pipeline start if LLM is enabled

    async def start(self):
        self._loop = asyncio.get_running_loop()
        
        def _progress_cb(model_name: str, percent: int):
            if self._client:
                msg = {"type": "download_progress", "model": model_name, "percent": percent}
                asyncio.run_coroutine_threadsafe(self._client.send(json.dumps(msg)), self._loop)
                
        set_progress_callback(_progress_cb)

        self._ws_server.on_start(self._on_client_start)
        self._ws_server.on_stop(self._on_client_stop)
        self._ws_server.on_suggestion(self._on_suggestion_request)
        self._ws_server.on_summary(self._on_summary_request)

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
            await asyncio.to_thread(self._initialize_pipeline, config)
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

        # 5. LLM Assistant
        if settings.get("llm.enabled", False):
            from engine.llm.assistant import LLMAssistant
            api_key = settings.get("llm.api_key", "")
            model = settings.get("llm.model", "openai/gpt-4o-mini")
            provider_url = settings.get("llm.provider_url") or None
            if api_key and model:
                self._llm_assistant = LLMAssistant(
                    api_key=api_key,
                    model=model,
                    base_url=provider_url,
                )
                logger.info("LLM assistant initialized: model=%s", model)
            else:
                logger.warning("LLM enabled but api_key or model not set — skipping")
                self._llm_assistant = None
        else:
            self._llm_assistant = None

    def _start_threads(self):
        self._running = True
        self._ws_server.is_capturing = True

        # Clear queues
        for q in self._audio_queues.values():
            with q.mutex:
                q.queue.clear()
                
        with self._translation_queue.mutex:
            self._translation_queue.queue.clear()
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
                target=self._process_loop,
                args=(stream_id, self._audio_queues[stream_id], self._vads[stream_id]),
                name=f"Process-{stream_id}"
            )
            pt.daemon = True
            pt.start()
            self._process_threads.append(pt)

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
        self._ws_server.is_capturing = False

        for inp in self._inputs.values():
            inp.stop()

        for ct in self._capture_threads:
            ct.join(timeout=1.0)
            
        for pt in self._process_threads:
            pt.join(timeout=2.0)

        if self._translation_thread:
            self._translation_thread.join(timeout=2.0)

        if self._result_task:
            self._result_task.cancel()
            try:
                await self._result_task
            except asyncio.CancelledError:
                pass

        if self._asr_backend:
            for stream_id in self._inputs.keys():
                result = self._asr_backend.finalize_current(stream_id)
                if result:
                    self._enqueue_asr_result(result)
            self._asr_backend.unload_model()
            self._asr_backend = None

        if self._translation_backend:
            self._translation_backend.unload_model()
            self._translation_backend = None

        # NOTE: _llm_assistant is intentionally NOT cleared here.
        # It holds no audio resources and must remain available so the
        # "Summarize Meeting" button works after the pipeline has stopped.
        # It will be re-created on the next pipeline start if config changes.

        self._inputs.clear()
        self._vads.clear()
        self._audio_queues.clear()
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

    def _process_loop(self, stream_id: str, audio_queue: Queue, vad):
        logger.info("Processing thread [%s] started.", stream_id)
        was_speech = False
        silence_start_time = None
        has_pending_audio = False

        try:
            while self._running:
                chunks = []
                try:
                    first = audio_queue.get(timeout=0.1)
                    chunks.append(first)
                    while not audio_queue.empty() and len(chunks) < 10:
                        try:
                            chunks.append(audio_queue.get_nowait())
                        except Empty:
                            break
                except Empty:
                    if has_pending_audio and silence_start_time is not None:
                        elapsed = time.time() - silence_start_time
                        if elapsed >= SILENCE_FINALIZE_DELAY:
                            self._do_finalize(stream_id)
                            has_pending_audio = False
                            silence_start_time = None
                            was_speech = False
                    continue

                combined_audio = b"".join(chunks)
                is_speech = vad.process(combined_audio)

                if is_speech:
                    if not was_speech:
                        logger.info("Process [%s]: SPEECH STARTED", stream_id)
                    silence_start_time = None
                    has_pending_audio = True
                    was_speech = True
                    
                    result = self._asr_backend.transcribe_stream(combined_audio, stream_id)
                    if result:
                        logger.debug("Process [%s]: Received ASR Result -> '%s' (is_final=%s)", stream_id, result.text, result.is_final)
                        self._enqueue_asr_result(result)
                else:
                    if was_speech and silence_start_time is None:
                        logger.info("Process [%s]: SPEECH ENDED. Entering silence countdown.", stream_id)
                        silence_start_time = time.time()
                    
                    if silence_start_time is not None:
                        elapsed = time.time() - silence_start_time
                        if elapsed >= SILENCE_FINALIZE_DELAY and has_pending_audio:
                            logger.info("Process [%s]: SILENCE TIMEOUT REACHED (%.2fs). Finalizing...", stream_id, elapsed)
                            self._do_finalize(stream_id)
                            has_pending_audio = False
                            silence_start_time = None
                            was_speech = False
                            vad.reset()

        except Exception as e:
            logger.error("Processing thread [%s] error: %s", stream_id, e, exc_info=True)
        finally:
            logger.info("Processing thread [%s] ended.", stream_id)

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

    def _do_finalize(self, stream_id: str = "default"):
        result = self._asr_backend.finalize_current(stream_id)
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

        # True Dual Stream automatically tags every result with its originating stream_id
        audio_source = asr_result.stream_id if asr_result.stream_id in ["mic", "system"] else None

        # Send ASR (English) to UI
        msg = {
            "type": "partial" if not asr_result.is_final else "final",
            "text": asr_result.text,
            "translation": None,
            "is_final": asr_result.is_final,
            "timestamp": time.time(),
        }

        if audio_source:
            msg["source"] = audio_source

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

    def _ensure_llm_assistant(self, message: dict) -> bool:
        """
        Ensure _llm_assistant is initialized and available.
        If it's not set (pipeline never started or already stopped),
        try to create one from the llm_config embedded in the request message.
        Returns True if the assistant is ready, False otherwise.
        """
        if self._llm_assistant and self._llm_assistant.is_available():
            return True

        llm_cfg = message.get("llm_config", {})
        api_key = llm_cfg.get("api_key", "")
        model = llm_cfg.get("model", "")
        provider_url = llm_cfg.get("provider_url") or None

        if not api_key or not model:
            return False

        try:
            from engine.llm.assistant import LLMAssistant
            self._llm_assistant = LLMAssistant(
                api_key=api_key,
                model=model,
                base_url=provider_url,
            )
            logger.info("LLM assistant initialized on-demand: model=%s", model)
            return self._llm_assistant.is_available()
        except Exception as e:
            logger.error("Failed to initialize LLM assistant on-demand: %s", e)
            return False

    async def _on_suggestion_request(self, message: dict, websocket):
        """Handle request_suggestion from UI — calls LLM on a thread and forwards result."""
        if not self._ensure_llm_assistant(message):
            await websocket.send(json.dumps({
                "type": "suggestion_result",
                "entry_id": message.get("entry_id", ""),
                "error": "AI Assistant is not configured. Please add your API key in Settings.",
            }))
            return

        entry_id = message.get("entry_id", "")
        target_text = message.get("target_text", "")
        context = message.get("context", [])

        def _callback(result: dict):
            msg = {
                "type": "suggestion_result",
                **result,
            }
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps(msg)), self._loop
            )

        self._llm_assistant.request_suggestion(
            entry_id=entry_id,
            target_text=target_text,
            context=context,
            callback=_callback,
        )

    async def _on_summary_request(self, message: dict, websocket):
        """Handle request_summary from UI — streams LLM output back as llm_chunk messages."""
        if not self._ensure_llm_assistant(message):
            await websocket.send(json.dumps({
                "type": "llm_chunk",
                "text": "**Error:** AI Assistant is not configured. Please add your API key in Settings.",
            }))
            await websocket.send(json.dumps({"type": "llm_done"}))
            return

        entries = message.get("entries", [])
        transcript_text = "\n".join(
            f"[{e.get('source', 'speaker').upper()}] {e.get('text', '')}"
            for e in entries
            if e.get("text", "").strip()
        )

        if not transcript_text.strip():
            await websocket.send(json.dumps({
                "type": "llm_chunk",
                "text": "**Error:** Transcript is empty — nothing to summarize.",
            }))
            await websocket.send(json.dumps({"type": "llm_done"}))
            return

        def _chunk_cb(text: str):
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps({"type": "llm_chunk", "text": text})),
                self._loop,
            )

        def _done_cb():
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps({"type": "llm_done"})),
                self._loop,
            )

        self._llm_assistant.request_summary(
            transcript_text=transcript_text,
            chunk_callback=_chunk_cb,
            done_callback=_done_cb,
        )


def run_engine(config_path=None):
    _load_dotenv()
    _setup_cuda_paths()
    _setup_logging()
    
    install_progress_hijack()

    engine = EchoFluxEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_engine()
