import logging
import time
import zlib
import re
from typing import Optional, List

import numpy as np

from engine.core.config import TranscriptionConfig
from engine.asr.base import ASRBackend, TranscriptResult

logger = logging.getLogger("echoflux.asr.faster_whisper")

_MODEL_INFERENCE_INTERVAL = {
    "tiny": 0.15,
    "base": 0.2,
    "small": 0.3,
    "medium": 0.5,
    "large": 0.6,
}

_MODEL_MAX_BUFFER = {
    "tiny": 5.0,
    "base": 5.0,
    "small": 4.0,
    "medium": 3.0,
    "large": 3.0,
}

class FasterWhisperBackend(ASRBackend):
    def __init__(self):
        self._model = None
        self._config: Optional[TranscriptionConfig] = None

        self._sample_rate = 16000
        self._audio_buffer = np.array([], dtype=np.float32)

        self._inference_interval = 0.3
        self._max_buffer_duration = 4.0
        self._last_inference_time = 0.0

        self._last_final_text = ""

    def load_model(self, config: TranscriptionConfig) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        target_device = config.device
        if target_device == "auto":
             target_device = "cuda"

        compute_type = config.compute_type
        model_path = config.model_path or config.model_size
        model_size = config.model_size

        self._inference_interval = _MODEL_INFERENCE_INTERVAL.get(model_size, 0.3)
        self._max_buffer_duration = _MODEL_MAX_BUFFER.get(model_size, 4.0)

        if target_device == "cuda":
            logger.info("Loading Faster-Whisper on CUDA (compute_type=%s)...", compute_type)
            try:
                self._model = WhisperModel(
                    model_path,
                    device="cuda",
                    compute_type=compute_type,
                    cpu_threads=4,
                )
                logger.info("Faster-Whisper loaded successfully on CUDA.")
                return
            except Exception as e:
                logger.warning("Failed on CUDA: %s. Fallback to CPU.", e)

        logger.info("Loading Faster-Whisper on CPU (int8)...")
        self._model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
        logger.info("Faster-Whisper loaded successfully on CPU.")

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
        self._last_inference_time = 0.0
        self._last_final_text = ""

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[TranscriptResult]:
        if not self._model:
            return None

        new_samples = self._bytes_to_float32(audio_chunk)
        self._audio_buffer = np.concatenate((self._audio_buffer, new_samples))

        max_samples = int(self._max_buffer_duration * self._sample_rate)

        if len(self._audio_buffer) > max_samples:
            return self._run_finalize()

        now = time.time()
        if now - self._last_inference_time < self._inference_interval:
            return None
        self._last_inference_time = now

        if len(self._audio_buffer) < int(0.4 * self._sample_rate):
            return None

        return self._run_inference(is_final=False)

    def finalize_current(self) -> Optional[TranscriptResult]:
        return self._run_finalize()

    def _run_finalize(self) -> Optional[TranscriptResult]:
        if not self._model or len(self._audio_buffer) == 0:
            return None

        result = self._run_inference(is_final=True)
        self._audio_buffer = np.array([], dtype=np.float32)
        return result

    def _run_inference(self, is_final: bool) -> Optional[TranscriptResult]:
        buffer_duration = len(self._audio_buffer) / self._sample_rate

        # Fixed parameter name: log_prob_threshold instead of logprob_threshold
        segments_gen, info = self._model.transcribe(
            self._audio_buffer,
            language=self._config.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        segments = list(segments_gen)
        if not segments:
            return None

        valid_segments = []
        for s in segments:
            if self._validate_segment(s):
                valid_segments.append(s.text.strip())

        if not valid_segments:
            return None

        raw_text = " ".join(valid_segments)

        # Advanced Hallucination Filter
        if self._is_hallucination(raw_text, buffer_duration):
            if is_final:
                self._audio_buffer = np.array([], dtype=np.float32)
            return None

        # Inter-segment repetition check (stuck buffer loop)
        clean_text = raw_text.strip()
        if clean_text == self._last_final_text:
             if is_final:
                self._audio_buffer = np.array([], dtype=np.float32)
             return None

        if is_final:
            self._last_final_text = clean_text
            self._audio_buffer = np.array([], dtype=np.float32)
            return TranscriptResult(
                text=clean_text,
                is_final=True,
                language=info.language,
            )

        return TranscriptResult(
            text=clean_text,
            is_final=False,
            language=info.language,
        )

    def _validate_segment(self, segment) -> bool:
        # 1. No Speech Probability
        if segment.no_speech_prob > 0.6:
            return False

        # 2. Average Log Probability (Confidence)
        if segment.avg_logprob < -1.0:
            return False

        # 3. Compression Ratio (Internal Whisper Metric)
        if segment.compression_ratio > 2.4:
            return False

        return True

    def _is_hallucination(self, text: str, duration: float) -> bool:
        text_len = len(text)
        if text_len == 0:
            return True

        # 1. Speech Rate Physics Check
        # 40 chars/sec is a very generous upper bound for human speech
        chars_per_sec = text_len / duration
        if chars_per_sec > 40.0:
            return True

        # 2. Entropy / Compression Check (Zlib)
        compressed = zlib.compress(text.encode("utf-8"))
        compression_ratio = len(text) / len(compressed)

        # Real speech rarely exceeds 3.0 compression ratio unless repetitive
        if text_len > 10 and compression_ratio > 3.0:
            return True

        # 3. Character Repetition (Regex)
        # Detects patterns like "......." or "?????"
        if re.search(r'([^\w\s])\1{3,}', text):
            return True

        return False

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        return int16_arr.astype(np.float32) / 32768.0
