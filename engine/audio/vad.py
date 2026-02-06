import struct
from collections import deque
from typing import Optional

from engine.core.logging import get_logger

logger = get_logger("audio.vad")


class VAD:
    """
    Energy-based Voice Activity Detection.
    Serves as a lightweight default. Can be replaced with a model-based
    detector (e.g. Silero VAD) via the same interface.
    """

    def __init__(self, config: dict):
        self._enabled = config.get("enabled", True)
        self._threshold = config.get("threshold", 0.5)
        self._sample_rate = config.get("sample_rate", 16000)

        # Adaptive energy tracking
        self._window_size = config.get("window_frames", 30)
        self._energy_history: deque = deque(maxlen=self._window_size)
        self._speech_frames = 0
        self._silence_frames = 0

        # Require a few consecutive speech/silence frames to transition
        self._speech_pad_frames = config.get("speech_pad_frames", 3)
        self._silence_pad_frames = config.get("silence_pad_frames", 8)

        self._is_speech = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    def process(self, audio_chunk: bytes) -> bool:
        if not self._enabled:
            return True

        energy = self._compute_energy(audio_chunk)
        self._energy_history.append(energy)

        adaptive_threshold = self._get_adaptive_threshold()
        frame_is_speech = energy > adaptive_threshold

        if frame_is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            self._speech_frames = 0

        if not self._is_speech and self._speech_frames >= self._speech_pad_frames:
            self._is_speech = True
            logger.debug("Speech start detected (energy=%.4f, threshold=%.4f)", energy, adaptive_threshold)

        if self._is_speech and self._silence_frames >= self._silence_pad_frames:
            self._is_speech = False
            logger.debug("Speech end detected (energy=%.4f, threshold=%.4f)", energy, adaptive_threshold)

        return self._is_speech

    def reset(self):
        self._energy_history.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speech = False

    def _compute_energy(self, audio_chunk: bytes) -> float:
        if len(audio_chunk) < 2:
            return 0.0
        n_samples = len(audio_chunk) // 2
        samples = struct.unpack(f"<{n_samples}h", audio_chunk[:n_samples * 2])
        rms = (sum(s * s for s in samples) / n_samples) ** 0.5
        # Normalize to 0.0–1.0 range based on int16 max
        return min(rms / 32768.0, 1.0)

    def _get_adaptive_threshold(self) -> float:
        if len(self._energy_history) < 5:
            return self._threshold * 0.02

        avg_energy = sum(self._energy_history) / len(self._energy_history)
        return max(avg_energy * self._threshold, 0.005)
