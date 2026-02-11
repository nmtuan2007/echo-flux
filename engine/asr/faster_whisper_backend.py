"""FasterWhisper ASR backend – Optimized streaming with anti-hallucination."""

import logging
import time
from collections import Counter
from typing import Optional

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

# Approximate max words per second of audio (generous upper bound)
MAX_WORDS_PER_SECOND = 5.0


class FasterWhisperBackend(ASRBackend):
    def __init__(self):
        self._model = None
        self._config: Optional[TranscriptionConfig] = None

        self._sample_rate = 16000
        self._audio_buffer = np.array([], dtype=np.float32)

        self._inference_interval = 0.3
        self._max_buffer_duration = 4.0
        self._last_inference_time = 0.0
        self._last_stable_text = ""

    def load_model(self, config: TranscriptionConfig) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        device = self._resolve_device(config.device)
        compute_type = config.compute_type

        if device == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"
            logger.info("Adjusted compute_type to '%s' for CPU", compute_type)

        model_path = config.model_path or config.model_size
        model_size = config.model_size

        self._inference_interval = _MODEL_INFERENCE_INTERVAL.get(model_size, 0.3)
        self._max_buffer_duration = _MODEL_MAX_BUFFER.get(model_size, 4.0)

        logger.info(
            "Loading model: %s (device=%s, compute_type=%s, "
            "inference_interval=%.2fs, max_buffer=%.1fs)",
            model_path, device, compute_type,
            self._inference_interval, self._max_buffer_duration,
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
        self._last_inference_time = 0.0
        self._last_stable_text = ""

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[TranscriptResult]:
        if not self._model:
            return None

        new_samples = self._bytes_to_float32(audio_chunk)
        self._audio_buffer = np.concatenate((self._audio_buffer, new_samples))

        max_samples = int(self._max_buffer_duration * self._sample_rate)
        if len(self._audio_buffer) > max_samples:
            result = self._run_finalize()
            if result:
                return result

        now = time.time()
        if now - self._last_inference_time < self._inference_interval:
            return None
        self._last_inference_time = now

        if len(self._audio_buffer) < int(0.3 * self._sample_rate):
            return None

        return self._run_inference(is_final=False)

    def finalize_current(self) -> Optional[TranscriptResult]:
        return self._run_finalize()

    def _run_finalize(self) -> Optional[TranscriptResult]:
        if not self._model or len(self._audio_buffer) == 0:
            return None

        result = self._run_inference(is_final=True)
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_stable_text = ""
        return result

    def _run_inference(self, is_final: bool) -> Optional[TranscriptResult]:
        buffer_duration = len(self._audio_buffer) / self._sample_rate

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

        raw_text = " ".join([s.text.strip() for s in segments if s.text.strip()])
        if not raw_text:
            return None

        # Layer 1: Remove repetitions
        cleaned_text = self._remove_repetitions(raw_text)

        # Layer 2: Length sanity check based on audio duration
        cleaned_text = self._enforce_length_limit(cleaned_text, buffer_duration)

        if not cleaned_text.strip():
            return None

        # Layer 3: Detect if hallucination occurred — force finalize
        is_hallucination = len(cleaned_text) < len(raw_text) * 0.7
        if is_hallucination:
            logger.warning(
                "Hallucination detected: raw=%d chars, cleaned=%d chars. Forcing finalize.",
                len(raw_text), len(cleaned_text),
            )
            is_final = True
            self._audio_buffer = np.array([], dtype=np.float32)

        if is_final:
            self._last_stable_text = ""
            self._audio_buffer = np.array([], dtype=np.float32)
            return TranscriptResult(
                text=cleaned_text,
                is_final=True,
                language=info.language,
            )

        self._last_stable_text = cleaned_text
        return TranscriptResult(
            text=cleaned_text,
            is_final=False,
            language=info.language,
        )

    @staticmethod
    def _remove_repetitions(text: str) -> str:
        if len(text) < 10:
            return text

        words = text.split()
        if len(words) < 3:
            return text

        # Pass 1: Single word consecutive repeats
        # "positive positive positive positive" → "positive"
        deduplicated = [words[0]]
        repeat_count = 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                repeat_count += 1
                if repeat_count <= 2:
                    deduplicated.append(words[i])
            else:
                repeat_count = 1
                deduplicated.append(words[i])

        words = deduplicated

        if len(words) < 4:
            return " ".join(words)

        # Pass 2: N-gram consecutive repeats (n = 2..10)
        # "work with humans and work with humans and" → "work with humans and"
        result = list(words)
        found = True
        while found:
            found = False
            for n in range(2, min(11, len(result) // 2 + 1)):
                i = 0
                new_result = []
                while i < len(result):
                    if i + n * 2 <= len(result):
                        pattern = result[i:i + n]
                        next_block = result[i + n:i + n * 2]
                        pattern_lower = [w.lower() for w in pattern]
                        next_lower = [w.lower() for w in next_block]

                        if pattern_lower == next_lower:
                            # Found repeat — keep pattern, skip all consecutive repeats
                            new_result.extend(pattern)
                            pos = i + n
                            while pos + n <= len(result):
                                candidate = [w.lower() for w in result[pos:pos + n]]
                                if candidate == pattern_lower:
                                    pos += n
                                else:
                                    break
                            # Append everything after the repeated block
                            new_result.extend(result[pos:])
                            found = True
                            result = new_result
                            break
                        else:
                            new_result.append(result[i])
                            i += 1
                    else:
                        new_result.append(result[i])
                        i += 1
                else:
                    result = new_result
                if found:
                    break

        if len(result) < 2:
            return " ".join(result)

        # Pass 3: Dominant word check
        # If a single word makes up >40% of all words, likely hallucination
        word_counts = Counter(w.lower() for w in result)
        total = len(result)
        for word, count in word_counts.most_common(1):
            if count > total * 0.4 and total > 5:
                # Keep only words up to the point where dominance starts
                trimmed = []
                seen_count = 0
                for w in result:
                    if w.lower() == word:
                        seen_count += 1
                    if seen_count > 3:
                        break
                    trimmed.append(w)
                logger.debug(
                    "Dominant word '%s' (%d/%d). Trimmed output.",
                    word, count, total,
                )
                result = trimmed
                break

        return " ".join(result)

    @staticmethod
    def _enforce_length_limit(text: str, audio_duration: float) -> str:
        """Trim output if it's unreasonably long for the given audio duration."""
        words = text.split()
        max_words = int(audio_duration * MAX_WORDS_PER_SECOND)
        max_words = max(max_words, 5)

        if len(words) > max_words:
            logger.debug(
                "Output too long (%d words for %.1fs audio, max %d). Trimming.",
                len(words), audio_duration, max_words,
            )
            return " ".join(words[:max_words])

        return text

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

        cuda_available = False
        try:
            import ctranslate2
            cuda_available = "cuda" in ctranslate2.get_supported_compute_types("cuda")
        except Exception:
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
