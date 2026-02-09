"""EchoFlux Engine – Main Entry Point (Refactored for True Streaming)."""

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
        self._vad = None
        
        self._running = False
        self._audio_queue = Queue(maxsize=500)  # ~10 seconds of audio at 20ms chunks
        self._result_queue = Queue()
        
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._result_task: Optional[asyncio.Task] = None
        
        self._client = None

    async def start(self):
        """Start the engine and WebSocket server."""
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
        # 1. Audio Input
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

        # 3. ASR
        from engine.asr.faster_whisper_backend import FasterWhisperBackend
        self._asr_backend = FasterWhisperBackend()
        
        asr_config = TranscriptionConfig(
            model_size=settings.get("asr.model_size", "small"),
            language=settings.get("asr.language", "en"),
            device=settings.get("asr.device", "auto"),
            compute_type=os.getenv("ECHOFLUX_COMPUTE_TYPE", "float16"),
        )
        self._asr_backend.load_model(asr_config)

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
            self._asr_backend.unload_model()
            self._asr_backend = None
        
        self._audio_input = None
        self._vad = None
        logger.info("Pipeline stopped.")

    def _capture_loop(self):
        """Reads audio from device and puts into queue."""
        logger.info("Capture thread started.")
        try:
            self._audio_input.start()
            while self._running:
                chunk = self._audio_input.read_chunk()
                if chunk:
                    if not self._audio_queue.full():
                        self._audio_queue.put(chunk)
                    else:
                        # Log less frequently to avoid spamming
                        # logger.warning("Audio queue full! Dropping frame.")
                        pass
                else:
                    time.sleep(0.005)
        except Exception as e:
            logger.error("Capture thread error: %s", e, exc_info=True)
        finally:
            logger.info("Capture thread ended.")

    def _process_loop(self):
        """Reads audio from queue and runs Inference."""
        logger.info("Processing thread started.")
        
        try:
            while self._running:
                # 1. Drain the queue (Batch Processing)
                # Instead of getting 1 chunk and blocking, get ALL pending chunks
                # This fixes the 'Audio queue full' issue when inference is slow
                chunks_to_process = []
                
                try:
                    # Blocking wait for the first chunk
                    first_chunk = self._audio_queue.get(timeout=0.5)
                    chunks_to_process.append(first_chunk)
                    
                    # Non-blocking drain of the rest
                    while not self._audio_queue.empty():
                        try:
                            chunks_to_process.append(self._audio_queue.get_nowait())
                        except Empty:
                            break
                except Empty:
                    continue

                if not chunks_to_process:
                    continue

                # Combine chunks
                combined_audio = b"".join(chunks_to_process)

                # 2. Run VAD (Updates internal state)
                # We can process just the combined chunk or feed it piece by piece if VAD logic implies it
                # For Silero, feeding the whole blob is fine if we updated the logic, 
                # but to be safe with the VAD wrapper (which buffers internally), we just feed it.
                is_speech = self._vad.process(combined_audio)
                
                # 3. Pass to ASR Backend
                # If VAD is active or backend manages context, feed audio.
                # Currently we feed everything to maintain context.
                result = self._asr_backend.transcribe_stream(combined_audio)
                
                if result:
                    self._result_queue.put(result)

        except Exception as e:
            logger.error("Processing thread error: %s", e, exc_info=True)
        finally:
            logger.info("Processing thread ended.")

    async def _broadcast_loop(self):
        """Reads results from queue and sends via WebSocket."""
        logger.info("Broadcast loop started.")
        while self._running:
            try:
                while not self._result_queue.empty():
                    result = self._result_queue.get_nowait()
                    
                    if self._client:
                        msg = json.dumps({
                            "type": "final" if result.is_final else "partial",
                            "text": result.text,
                            "is_final": result.is_final,
                            "timestamp": time.time(),
                        })
                        await self._client.send(msg)
                
                await asyncio.sleep(0.05)
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
