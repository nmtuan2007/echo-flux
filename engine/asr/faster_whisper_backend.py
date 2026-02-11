"""FasterWhisper ASR backend – Speed Optimized."""

import logging
import time
from typing import Optional

import numpy as np

from engine.core.config import TranscriptionConfig
from engine.asr.base import ASRBackend, TranscriptResult

logger = logging.getLogger("echoflux.asr.faster_whisper")


class FasterWhisperBackend(ASRBackend):
    def __init__(self):
        self._model = None
        self._config: Optional[TranscriptionConfig] = None

        self._sample_rate = 16000
        self._audio_buffer = np.array([], dtype=np.float32)

        self._inference_interval = 0.2
        self._last_inference_time = 0

        self._finalization_threshold_seconds = 10.0

    def load_model(self, config: TranscriptionConfig) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        device = self._resolve_device(config.device)
        compute_type = config.compute_type

        if device == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"
            logger.info("Adjusted compute_type to '%s' for CPU", compute_type)

        model_path = config.model_path or config.model_size
        logger.info(
            "Loading model: %s (device=%s, compute_type=%s)",
            model_path, device, compute_type,
        )

        try:
            self._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
            )
        except Exception as e:
            if device == "cuda":
                logger.warning(
                    "Failed to load model on CUDA: %s. Falling back to CPU.", e
                )
                device = "cpu"
                compute_type = "int8"
                self._model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4,
                )
            else:
                raise

        logger.info("Model loaded successfully (device=%s, compute_type=%s)", device, compute_type)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload_model(self) -> None:
        if self._model:
            del self._model
        self._model = None
        self.reset_stream()

    def reset_stream(self) -> None:
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_inference_time = 0

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[TranscriptResult]:
        if not self._model:
            return None

        new_samples = self._bytes_to_float32(audio_chunk)
        self._audio_buffer = np.concatenate((self._audio_buffer, new_samples))

        now = time.time()
        if now - self._last_inference_time < self._inference_interval:
            return None
        self._last_inference_time = now

        if len(self._audio_buffer) < 4800:
            return None

        segments_gen, info = self._model.transcribe(
            self._audio_buffer,
            language=self._config.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        segments = list(segments_gen)
        if not segments:
            return None

        buffer_duration = len(self._audio_buffer) / self._sample_rate
        final_text_parts = []
        cut_off_time = 0.0

        should_finalize = False
        if len(segments) > 1:
            should_finalize = True
            segments_to_finalize = segments[:-1]
        elif buffer_duration > self._finalization_threshold_seconds:
            should_finalize = True
            segments_to_finalize = segments[:-1] if len(segments) > 1 else segments
        else:
            segments_to_finalize = []

        if should_finalize and segments_to_finalize:
            for seg in segments_to_finalize:
                if seg.text.strip():
                    final_text_parts.append(seg.text.strip())
                cut_off_time = seg.end

            cut_samples = int(cut_off_time * self._sample_rate)
            cut_samples = min(cut_samples, len(self._audio_buffer))
            self._audio_buffer = self._audio_buffer[cut_samples:]

            if final_text_parts:
                final_text = " ".join(final_text_parts)
                return TranscriptResult(
                    text=final_text,
                    is_final=True,
                    language=info.language,
                )

        full_text = " ".join([s.text.strip() for s in segments if s.text.strip()])
        if not full_text:
            return None

        return TranscriptResult(
            text=full_text,
            is_final=False,
            language=info.language,
        )

    def finalize_current(self) -> Optional[TranscriptResult]:
        if not self._model or len(self._audio_buffer) == 0:
            return None

        segments_gen, info = self._model.transcribe(
            self._audio_buffer,
            language=self._config.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = list(segments_gen)
        text = " ".join([s.text.strip() for s in segments if s.text.strip()])

        self.reset_stream()

        if text:
            return TranscriptResult(text=text, is_final=True, language=info.language)
        return None

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        return int16_arr.astype(np.float32) / 32768.0

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "cpu":
            return "cpu"

        # For "cuda" or "auto", probe actual CUDA availability via CTranslate2
        cuda_available = False
        try:
            import ctranslate2
            cuda_available = "cuda" in ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            # CTranslate2 probe failed; try torch as secondary check
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                pass

        if cuda_available:
            logger.info("CUDA is available — using GPU for ASR")
            return "cuda"

        if device_str == "cuda":
            logger.warning(
                "CUDA was explicitly requested but is not available. "
                "Common causes: missing CUDA Toolkit, missing cuBLAS/cuDNN libraries "
                "(nvidia-cublas-cu12, nvidia-cudnn-cu12), or incompatible driver. "
                "Falling back to CPU."
            )
        else:
            logger.info("CUDA not available — using CPU for ASR")

        return "cpu"
