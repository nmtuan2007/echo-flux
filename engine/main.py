import asyncio
import json
import logging
import os
import threading
import time
from queue import Queue, Empty
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from engine.core.config import TranscriptionConfig
from engine.server.websocket_server import WebSocketServer

logger = logging.getLogger("echoflux.engine")


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
        
        self._running = False
        self._audio_queue = Queue(maxsize=500)
        self._result_queue = Queue()
        
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
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

        # 1. Audio Source
        audio_source = os.getenv("ECHOFLUX_AUDIO_SOURCE", "microphone")
        if audio_source == "system":
            from engine.audio.system_audio import SystemAudioInput
            self._audio_input = SystemAudioInput(settings)
        else:
            from engine.audio.microphone import MicrophoneInput
            self._audio_input = MicrophoneInput(settings)
        
        # 2. VAD
        from engine.audio.vad import VAD
        self._vad = VAD({
            "enabled": settings.get("vad.enabled", True),
            "threshold": settings.get("vad.threshold", 0.5),
            "sample_rate": 16000
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
            backend_type = settings.get("translation.backend", "marian")
            
            if backend_type == "online":
                from engine.translation.online_backend import OnlineBackend
                self._translation_backend = OnlineBackend()
            else:
                from engine.translation.marian_backend import MarianBackend
                self._translation_backend = MarianBackend()
            
            self._translation_backend.load_model(settings)

    def _start_threads(self):
        self._running = True
        
        while not self._audio_queue.empty(): self._audio_queue.get()
        while not self._result_queue.empty(): self._result_queue.get()

        self._capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self._capture_thread.daemon = True
        self._capture_thread.start()

        self._process_thread = threading.Thread(target=self._process_loop, name="ProcessThread")
        self._process_thread.daemon = True
        self._process_thread.start()

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

        if self._result_task:
            self._result_task.cancel()
            try:
                await self._result_task
            except asyncio.CancelledError:
                pass
        
        if self._asr_backend:
            # Final flush
            result = self._asr_backend.finalize_current()
            if result:
                self._process_result(result) 
                
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
        
        silence_counter = 0
        max_silence_chunks = 100 # Approx 2 seconds
        
        try:
            while self._running:
                chunks_to_process = []
                try:
                    first_chunk = self._audio_queue.get(timeout=0.5)
                    chunks_to_process.append(first_chunk)
                    while not self._audio_queue.empty():
                        try:
                            chunks_to_process.append(self._audio_queue.get_nowait())
                        except Empty:
                            break
                except Empty:
                    continue

                if not chunks_to_process:
                    continue

                combined_audio = b"".join(chunks_to_process)
                is_speech = self._vad.process(combined_audio)
                
                if is_speech:
                    silence_counter = 0
                    result = self._asr_backend.transcribe_stream(combined_audio)
                    if result:
                        self._process_result(result)
                else:
                    silence_counter += len(chunks_to_process)
                    if silence_counter > max_silence_chunks:
                        result = self._asr_backend.finalize_current()
                        if result:
                            self._process_result(result)
                        silence_counter = 0
                    else:
                        self._asr_backend.transcribe_stream(combined_audio)

        except Exception as e:
            logger.error("Processing thread error: %s", e, exc_info=True)
        finally:
            logger.info("Processing thread ended.")

    def _process_result(self, asr_result):
        """Helper to attach translation and queue the result."""
        translated_text = None
        
        # Only translate if enabled AND model is loaded
        if self._translation_backend and self._translation_backend.is_loaded:
            if asr_result.text.strip():
                try:
                    src = self._current_config.get("translation.source_lang", "auto")
                    tgt = self._current_config.get("translation.target_lang", "vi")
                    
                    # --- CHANGE: Only translate if the segment is FINAL ---
                    # This prevents heavy CPU load and flickering text.
                    if asr_result.is_final:
                        trans_res = self._translation_backend.translate(
                            asr_result.text, src, tgt
                        )
                        translated_text = trans_res.translated_text
                        
                except Exception as e:
                    logger.error("Translation processing error: %s", e)

        # Enqueue as a dict to be broadcast
        self._result_queue.put({
            "type": "final" if asr_result.is_final else "partial",
            "text": asr_result.text,
            "translation": translated_text, # Will be None for partial results
            "is_final": asr_result.is_final,
            "timestamp": time.time(),
        })

    async def _broadcast_loop(self):
        while self._running:
            try:
                while not self._result_queue.empty():
                    msg_dict = self._result_queue.get_nowait()
                    if self._client:
                        await self._client.send(json.dumps(msg_dict))
                await asyncio.sleep(0.02)
            except Exception as e:
                logger.error("Broadcast error: %s", e)
                await asyncio.sleep(1)


if __name__ == "__main__":
    _load_dotenv()
    _setup_logging()
    
    engine = EchoFluxEngine()
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        pass
