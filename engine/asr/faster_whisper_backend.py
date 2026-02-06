import io
import numpy as np
from typing import Optional

from engine.asr.base import ASRBackend, TranscriptResult
from engine.core.logging import get_logger
from engine.core.exceptions import ModelLoadError, ModelNotFoundError, ASRError

logger = get_logger("asr.faster_whisper")


class FasterWhisperBackend(ASRBackend):

    def __init__(self):
        self._model = None
        self._buffer = bytearray()
        self._sample_rate = 16000
        self._min_chunk_seconds = 1.0
        self._language: Optional[str] = None
        self._config: dict = {}

    def load_model(self, config: dict) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ModelLoadError("faster-whisper is not installed")

        model_size = config.get("model_size", "small")
        device = self._resolve_device(config.get("device", "auto"))
        compute_type = config.get("compute_type", "float16")
        model_path = config.get("model_path")

        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.info("CPU mode: falling back compute_type to int8")

        self._language = config.get("language")
        self._sample_rate = config.get("sample_rate", 16000)
        self._min_chunk_seconds = config.get("min_chunk_seconds", 1.0)
        self._config = config

        model_id = model_path or model_size
        logger.info(
            "Loading model: %s (device=%s, compute_type=%s)",
            model_id, device, compute_type,
        )

        try:
            self._model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
            )
        except Exception as e:
            self._model = None
            raise ModelLoadError(f"Failed to load faster-whisper model: {e}") from e

        logger.info("Model loaded successfully")

    def transcribe_stream(self, audio_chunk: bytes) -> TranscriptResult:
        if not self._model:
            raise ASRError("Model not loaded")

        self._buffer.extend(audio_chunk)

        min_bytes = int(self._sample_rate * 2 * self._min_chunk_seconds)
        if len(self._buffer) < min_bytes:
            return TranscriptResult(text="", is_final=False, confidence=0.0)

        audio_array = self._bytes_to_float32(bytes(self._buffer))

        try:
            segments, info = self._model.transcribe(
                audio_array,
                language=self._language,
                beam_size=1,
                best_of=1,
                vad_filter=False,
                without_timestamps=True,
            )

            text_parts = []
            total_confidence = 0.0
            segment_count = 0

            for segment in segments:
                text_parts.append(segment.text.strip())
                total_confidence += getattr(segment, "avg_logprob", 0.0)
                segment_count += 1

            text = " ".join(text_parts).strip()
            avg_confidence = (total_confidence / segment_count) if segment_count > 0 else 0.0

            is_final = self._should_finalize()

            if is_final:
                self._buffer.clear()

            return TranscriptResult(
                text=text,
                is_final=is_final,
                confidence=avg_confidence,
                language=info.language if hasattr(info, "language") else self._language,
            )

        except Exception as e:
            logger.error("Transcription error: %s", e)
            raise ASRError(f"Transcription failed: {e}") from e

    def reset_stream(self) -> None:
        self._buffer.clear()
        logger.debug("Stream buffer reset")

    def unload_model(self) -> None:
        self._model = None
        self._buffer.clear()
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _should_finalize(self) -> bool:
        max_seconds = self._config.get("max_buffer_seconds", 5.0)
        max_bytes = int(self._sample_rate * 2 * max_seconds)
        return len(self._buffer) >= max_bytes

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA available — using GPU")
                return "cuda"
        except ImportError:
            pass
        logger.info("Using CPU")
        return "cpu"
