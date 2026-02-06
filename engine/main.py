import asyncio
import time
import threading
from typing import Optional

from engine.core.config import Config
from engine.core.logging import setup_logging, get_logger
from engine.core.exceptions import EchoFluxError
from engine.audio.input_manager import InputManager
from engine.audio.microphone import MicrophoneInput
from engine.audio.vad import VAD
from engine.asr.base import ASRBackend
from engine.asr.faster_whisper_backend import FasterWhisperBackend
from engine.translation.base import TranslationBackend
from engine.translation.marian_backend import MarianBackend
from engine.server.websocket_server import WebSocketServer

logger = None


class EchoFluxEngine:

    def __init__(self, config: Config):
        self._config = config
        global logger
        logger = setup_logging(config)
        self._logger = get_logger("engine")

        audio_config = config.get("audio", {})
        self._input_manager = InputManager(audio_config)
        self._vad = VAD({
            **config.get("vad", {}),
            "sample_rate": audio_config.get("sample_rate", 16000),
        })

        self._asr: Optional[ASRBackend] = None
        self._translator: Optional[TranslationBackend] = None

        self._server = WebSocketServer(
            host=config.get("engine.host", "127.0.0.1"),
            port=config.get("engine.port", 8765),
        )
        self._server.on_start(self._handle_start)
        self._server.on_stop(self._handle_stop)

        self._capture_thread: Optional[threading.Thread] = None
        self._processing = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _init_audio_source(self):
        audio_config = self._config.get("audio", {})
        source_type = self._config.get("audio.source", "microphone")
        device_id = self._config.get("audio.device_id")

        if source_type == "system":
            try:
                from engine.audio.system_audio import SystemAudioInput
                source = SystemAudioInput(audio_config, device_id=device_id)
                self._logger.info("Using system audio (loopback) input")
            except Exception as e:
                self._logger.warning("System audio unavailable, falling back to microphone: %s", e)
                source = MicrophoneInput(audio_config, device_id=int(device_id) if device_id else None)
        else:
            mic_device = int(device_id) if device_id is not None else None
            source = MicrophoneInput(audio_config, device_id=mic_device)
            self._logger.info("Using microphone input")

        self._input_manager.set_source(source)

    def _init_asr(self) -> ASRBackend:
        backend_name = self._config.get("asr.backend", "faster_whisper")
        if backend_name == "faster_whisper":
            backend = FasterWhisperBackend()
        else:
            raise EchoFluxError(f"Unknown ASR backend: {backend_name}")

        backend.load_model({
            "model_size": self._config.get("asr.model_size", "small"),
            "language": self._config.get("asr.language", "en"),
            "compute_type": self._config.get("asr.compute_type", "float16"),
            "device": self._config.get("asr.device", "auto"),
            "model_path": self._config.get("asr.model_path"),
            "sample_rate": self._config.get("audio.sample_rate", 16000),
        })
        return backend

    def _init_translator(self) -> Optional[TranslationBackend]:
        if not self._config.get("translation.enabled", False):
            return None

        backend_name = self._config.get("translation.backend", "marian")
        if backend_name == "marian":
            backend = MarianBackend()
        else:
            raise EchoFluxError(f"Unknown translation backend: {backend_name}")

        backend.load_model({
            "source_lang": self._config.get("translation.source_lang", "en"),
            "target_lang": self._config.get("translation.target_lang", "vi"),
            "model_path": self._config.get("translation.model_path"),
            "device": self._config.get("asr.device", "auto"),
        })
        return backend

    async def _handle_start(self, client_config: dict):
        self._logger.info("Starting pipeline")

        for key, value in client_config.items():
            self._config.set(key, value)

        try:
            self._init_audio_source()
            self._asr = self._init_asr()
            self._translator = self._init_translator()
            self._start_capture()
        except EchoFluxError as e:
            self._logger.error("Failed to start pipeline: %s", e)
            await self._server.broadcast({
                "type": "error",
                "message": str(e),
            })

    async def _handle_stop(self):
        self._logger.info("Stopping pipeline")
        self._stop_capture()

        if self._asr:
            self._asr.reset_stream()
            self._asr.unload_model()
            self._asr = None

        if self._translator:
            self._translator.unload_model()
            self._translator = None

        self._vad.reset()

    def _start_capture(self):
        if self._processing:
            return

        self._processing = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="audio-capture",
        )
        self._capture_thread.start()

    def _stop_capture(self):
        self._processing = False
        self._input_manager.stop()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        self._capture_thread = None

    def _capture_loop(self):
        self._logger.info("Audio capture loop started")
        self._input_manager.start()

        while self._processing:
            try:
                chunk = self._input_manager.read_chunk()
                if not chunk:
                    continue

                is_speech = self._vad.process(chunk)
                if not is_speech:
                    continue

                if not self._asr:
                    continue

                result = self._asr.transcribe_stream(chunk)
                if not result.text:
                    continue

                message = {
                    "type": "final" if result.is_final else "partial",
                    "text": result.text,
                    "translation": None,
                    "timestamp": time.time(),
                }

                if result.is_final and self._translator:
                    try:
                        tr = self._translator.translate(
                            result.text,
                            self._config.get("translation.source_lang", "en"),
                            self._config.get("translation.target_lang", "vi"),
                        )
                        message["translation"] = tr.translated_text
                    except Exception as e:
                        self._logger.error("Translation failed: %s", e)

                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._server.broadcast(message),
                        self._loop,
                    )

            except Exception as e:
                self._logger.error("Capture loop error: %s", e)

        self._logger.info("Audio capture loop stopped")

    async def run(self):
        self._loop = asyncio.get_running_loop()
        await self._server.start()
        self._logger.info("EchoFlux engine running — waiting for client commands")

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            await self._handle_stop()
            await self._server.stop()
            self._logger.info("EchoFlux engine shut down")


def run_engine(config_path: Optional[str] = None):
    config = Config(config_path)
    engine = EchoFluxEngine(config)

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_engine()
