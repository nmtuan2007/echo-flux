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
    "tiny": 10.0,
    "base": 10.0,
    "small": 8.0,
    "medium": 6.0,
    "large": 6.0,
}

class FasterWhisperBackend(ASRBackend):
    def __init__(self):
        self._model = None
        self._config: Optional[TranscriptionConfig] = None
        self._device: Optional[str] = None

        self._sample_rate = 16000
        
        # Dictionary of streams to keep independent buffers and timers 
        # (e.g. {"mic": {...}, "system": {...}})
        self._streams = {}

        self._inference_interval = 0.3
        self._max_buffer_duration = 4.0

    def _get_stream(self, stream_id: str) -> dict:
        if stream_id not in self._streams:
            self._streams[stream_id] = {
                "audio_buffer": np.array([], dtype=np.float32),
                "last_inference_time": 0.0,
                "last_final_text": "",
                "dominant_language": None,
                "alt_lang": None,
                "alt_streak": 0,
            }
        return self._streams[stream_id]

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

        from engine.core.config import Config
        models_dir = str(Config().models_dir)

        if target_device == "cuda":
            logger.info("Loading Faster-Whisper on CUDA (compute_type=%s)...", compute_type)
            try:
                self._model = WhisperModel(
                    model_path,
                    device="cuda",
                    compute_type=compute_type,
                    cpu_threads=4,
                    download_root=models_dir
                )
                logger.info("Faster-Whisper loaded successfully on CUDA.")
                self._device = "cuda"
                return
            except Exception as e:
                logger.warning("Failed on CUDA: %s. Fallback to CPU.", e)

        logger.info("Loading Faster-Whisper on CPU (int8)...")
        self._model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            download_root=models_dir
        )
        logger.info("Faster-Whisper loaded successfully on CPU.")
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        """Device the model actually loaded on after any CUDA-to-CPU fallback."""
        return self._device

    @staticmethod
    def normalize_language(language: Optional[str]) -> Optional[str]:
        """Map the "auto" hint (and empty values) to None so Whisper detects the language."""
        if not language or language.strip().lower() == "auto":
            return None
        return language

    def unload_model(self) -> None:
        if self._model:
            del self._model
        self._model = None
        self._device = None
        self._streams.clear()

    def reset_stream(self, stream_id: str = "default") -> None:
        if stream_id in self._streams:
            del self._streams[stream_id]

    def transcribe_stream(self, audio_chunk: bytes, stream_id: str = "default") -> Optional[TranscriptResult]:
        if not self._model:
            return None

        stream = self._get_stream(stream_id)
        new_samples = self._bytes_to_float32(audio_chunk)
        stream["audio_buffer"] = np.concatenate((stream["audio_buffer"], new_samples))

        max_samples = int(self._max_buffer_duration * self._sample_rate)

        if len(stream["audio_buffer"]) > max_samples:
            logger.debug("ASR [%s]: Buffer exceeded max_samples, forcing finalize.", stream_id)
            return self._run_finalize(stream_id, stream)

        now = time.time()
        if now - stream["last_inference_time"] < self._inference_interval:
            return None
        stream["last_inference_time"] = now

        # Auto-detect needs a longer window than fixed-language mode to avoid wild guesses.
        target_lang = self.normalize_language(self._config.language) if self._config else None
        min_partial_sec = 0.8 if target_lang is None else 0.4
        if len(stream["audio_buffer"]) < int(min_partial_sec * self._sample_rate):
            return None

        return self._run_inference(stream_id, stream, is_final=False)

    def finalize_current(self, stream_id: str = "default") -> Optional[TranscriptResult]:
        return self._run_finalize(stream_id, self._get_stream(stream_id))

    def _run_finalize(self, stream_id: str, stream: dict) -> Optional[TranscriptResult]:
        if not self._model or len(stream["audio_buffer"]) == 0:
            return None

        logger.debug("ASR [%s]: Running finalize against %.2fs buffer.", stream_id, len(stream["audio_buffer"])/self._sample_rate)
        result = self._run_inference(stream_id, stream, is_final=True)
        stream["audio_buffer"] = np.array([], dtype=np.float32)
        return result

    def _run_inference(self, stream_id: str, stream: dict, is_final: bool) -> Optional[TranscriptResult]:
        buffer_duration = len(stream["audio_buffer"]) / self._sample_rate
        target_lang = self.normalize_language(self._config.language) if self._config else None

        # Fixed parameter name: log_prob_threshold instead of logprob_threshold
        transcribe_kwargs = dict(
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        segments_gen, info = self._model.transcribe(stream["audio_buffer"], language=target_lang, **transcribe_kwargs)

        if target_lang is None:
            detected_lang = getattr(info, "language", None)
            lang_prob = getattr(info, "language_probability", 1.0)
            dominant = stream.get("dominant_language")
            if dominant is None:
                if detected_lang and lang_prob >= 0.80:
                    stream["dominant_language"] = detected_lang
                elif not is_final and lang_prob < 0.50:
                    return None
            elif detected_lang == dominant:
                stream["alt_lang"] = None
                stream["alt_streak"] = 0
            else:
                if detected_lang and lang_prob >= 0.95:
                    streak = (stream.get("alt_streak", 0) + 1) if stream.get("alt_lang") == detected_lang else 1
                    stream["alt_lang"] = detected_lang
                    stream["alt_streak"] = streak
                else:
                    streak = 0
                    stream["alt_lang"] = None
                    stream["alt_streak"] = 0
                if streak >= 2:
                    stream["dominant_language"] = detected_lang
                    stream["alt_lang"] = None
                    stream["alt_streak"] = 0
                else:
                    segments_gen, info = self._model.transcribe(stream["audio_buffer"], language=dominant, **transcribe_kwargs)

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
            logger.debug("ASR [%s]: Dropped due to hallucination filter: '%s'", stream_id, raw_text)
            if is_final:
                stream["audio_buffer"] = np.array([], dtype=np.float32)
            return None

        # Record the last final text just in case we need it later, but DO NOT drop it simply because it matches.
        clean_text = raw_text.strip()

        if is_final:
            stream["last_final_text"] = clean_text
            stream["audio_buffer"] = np.array([], dtype=np.float32)
            return TranscriptResult(
                text=clean_text,
                is_final=True,
                language=info.language,
                stream_id=stream_id
            )

        return TranscriptResult(
            text=clean_text,
            is_final=False,
            language=info.language,
            stream_id=stream_id
        )

    def _validate_segment(self, segment) -> bool:
        # Relaxed Validation: VAD already ensures speech is present.
        # We only drop absolute garbage that the hallucination filter might miss.
        
        # 1. No Speech Probability
        if segment.no_speech_prob > 0.85:
            return False

        # 2. Average Log Probability (Confidence)
        if segment.avg_logprob < -1.8:
            return False

        # 3. Compression Ratio (Internal Whisper Metric)
        if segment.compression_ratio > 3.2:
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
