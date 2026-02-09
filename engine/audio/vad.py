import logging
import os
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from engine.core.config import Config

logger = logging.getLogger("echoflux.audio.vad")

# Silero VAD v4 constants (Updated to stable tag)
MODEL_URL = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
# Supported window sizes for 16000Hz: 512, 1024, 1536
WINDOW_SIZE_SAMPLES = 512  # ~32ms at 16kHz


class VAD:
    """
    Neural Voice Activity Detection using Silero VAD (ONNX).
    Replaces the legacy Energy-based VAD.
    """

    def __init__(self, config: dict):
        self._enabled = config.get("enabled", True)
        self._threshold = config.get("threshold", 0.5)
        self._sample_rate = config.get("sample_rate", 16000)

        # Buffer for accumulating samples to match model window size
        self._buffer = np.array([], dtype=np.float32)
        
        # ONNX Runtime session
        self._session: Optional[ort.InferenceSession] = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

        # State tracking
        self._is_speech = False
        
        self._models_dir = Config().models_dir
        self._model_path = self._models_dir / "silero_vad.onnx"

        if self._enabled:
            self._init_model()

    def _init_model(self):
        """Download and load the ONNX model."""
        try:
            if not self._model_path.exists():
                logger.info("Downloading Silero VAD model to %s...", self._model_path)
                self._models_dir.mkdir(parents=True, exist_ok=True)
                # Set user agent to avoid 403 forbidden on some systems
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, self._model_path)
                logger.info("Download complete.")

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3  # Error only

            self._session = ort.InferenceSession(
                str(self._model_path), 
                sess_options=opts, 
                providers=["CPUExecutionProvider"]
            )
            logger.info("Silero VAD initialized (threshold=%.2f)", self._threshold)
        except Exception as e:
            logger.error("Failed to initialize Silero VAD: %s", e)
            # We explicitly disable VAD if init fails to prevent crashes,
            # but log clearly so user knows performance will degrade.
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    def process(self, audio_chunk: bytes) -> bool:
        """
        Process a chunk of audio bytes (int16).
        Returns True if speech is detected in the current buffer context.
        """
        if not self._enabled or self._session is None:
            return True

        # Convert bytes to float32 numpy array
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Append to internal buffer
        self._buffer = np.concatenate((self._buffer, audio_float32))

        # Process in chunks of WINDOW_SIZE_SAMPLES (512)
        has_speech_in_chunk = False

        while len(self._buffer) >= WINDOW_SIZE_SAMPLES:
            window = self._buffer[:WINDOW_SIZE_SAMPLES]
            self._buffer = self._buffer[WINDOW_SIZE_SAMPLES:]

            prob = self._inference(window)
            
            if prob > self._threshold:
                self._is_speech = True
                has_speech_in_chunk = True
            else:
                pass 
        
        return self._is_speech

    def _inference(self, window: np.ndarray) -> float:
        # Prepare inputs: input [1, N], sr [1], h, c
        x = window[np.newaxis, :] 
        sr = np.array([self._sample_rate], dtype=np.int64)

        ort_inputs = {
            "input": x,
            "sr": sr,
            "h": self._h,
            "c": self._c
        }

        out, self._h, self._c = self._session.run(None, ort_inputs)
        return out[0][0]

    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._buffer = np.array([], dtype=np.float32)
        self._is_speech = False
