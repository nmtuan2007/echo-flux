"""FasterWhisper ASR backend – Continuous Sliding Window Streaming."""

import logging
import time
from typing import Optional, List

import numpy as np

from engine.core.config import TranscriptionConfig
from engine.asr.base import ASRBackend, TranscriptResult

logger = logging.getLogger("echoflux.asr.faster_whisper")


class FasterWhisperBackend(ASRBackend):
    def __init__(self):
        self._model = None
        self._config: Optional[TranscriptionConfig] = None
        self._sample_rate = 16000

        # ── Sliding Window State ──
        # Ring buffer for audio: keeps the last 30 seconds of context
        # 16000 Hz * 30 sec = 480000 samples
        self._max_context_samples = 16000 * 30
        self._audio_buffer = np.array([], dtype=np.float32)
        
        # Cursor pointing to where "confirmed" (final) text ends in the audio buffer
        self._committed_cursor = 0
        
        # In-progress text state
        self._last_final_text = ""
        self._last_partial_time = 0
        
        # Performance tuning
        self._inference_interval = 0.3  # Run inference max every 300ms
        
        # VAD filtering helper inside Whisper
        self._vad_parameters = dict(
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )

    def load_model(self, config: TranscriptionConfig) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        device = self._resolve_device(config.device)
        compute_type = config.compute_type

        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.info("CPU mode: switched compute_type to int8 for performance")

        model_path = config.model_path or config.model_size
        logger.info(
            "Loading model: %s (device=%s, compute_type=%s)",
            model_path, device, compute_type,
        )
        
        self._model = WhisperModel(
            model_path, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=4 # Optimize for realtime on CPU
        )
        logger.info("Model loaded successfully")

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
        self._committed_cursor = 0
        self._last_final_text = ""
        self._last_partial_time = 0

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[TranscriptResult]:
        """
        Ingest audio, update buffer, and optionally run inference.
        Returns a TranscriptResult if new text is available.
        """
        if not self._model:
            return None

        # 1. Parse and append audio
        new_samples = self._bytes_to_float32(audio_chunk)
        self._audio_buffer = np.concatenate((self._audio_buffer, new_samples))

        # 2. Check timing (don't run inference on every single 20ms chunk)
        now = time.time()
        if now - self._last_partial_time < self._inference_interval:
            return None
        self._last_partial_time = now

        # 3. Get the "active" window
        # We only care about audio AFTER the committed cursor (plus a bit of context overlap)
        # However, Whisper works best with some context.
        # Strategy: Pass the full buffer (up to 30s), let Whisper decode, then diff against known text.
        
        # Optimization: If buffer is too long, trim the head (processed part)
        # but keep enough for Whisper context.
        if len(self._audio_buffer) > self._max_context_samples:
            # Shift buffer: Keep last 25s
            keep_samples = 16000 * 25
            trim_amount = len(self._audio_buffer) - keep_samples
            
            self._audio_buffer = self._audio_buffer[-keep_samples:]
            self._committed_cursor = max(0, self._committed_cursor - trim_amount)

        # 4. Run Inference
        segments, info = self._model.transcribe(
            self._audio_buffer,
            language=self._config.language,
            beam_size=1,      # Greedy decoding for speed
            best_of=1,
            temperature=0.0,
            vad_filter=True,  # Internal VAD to skip silent parts of the buffer
            vad_parameters=self._vad_parameters,
            condition_on_previous_text=False # Crucial for stability in streaming chunks
        )

        current_segments = list(segments)
        if not current_segments:
            return None

        # 5. Process Segments
        # Reconstruct the full text currently visible in the buffer
        full_text_in_window = " ".join([s.text.strip() for s in current_segments if s.text.strip()])
        
        if not full_text_in_window:
            return None

        # 6. Stability Check / Finalization Logic
        # If the last segment is completed (based on Whisper's timestamp logic), we can consider it "Final".
        # However, for low-latency visual feedback, we output everything as "Partial" 
        # until the buffer grows significantly or we explicitly detect a pause.
        
        # In this implementation, we rely on the main loop's VAD to finalize
        # OR we treat everything as partial updates until the main loop decides to "reset" or "finalize".
        # But to be robust, we can implement "Local Finalization":
        # If we have multiple segments, earlier ones are likely stable.
        
        is_final = False
        text_to_emit = full_text_in_window

        # Logic: If we have > 1 segment, the earlier ones are usually stable.
        # But simpler logic for the "Instant" feel: Always emit partial of the whole buffer.
        # Main.py will handle finalization on silence.
        
        return TranscriptResult(
            text=text_to_emit,
            is_final=False, # Let the UI show it as grey/partial
            language=info.language,
            confidence=current_segments[-1].avg_logprob
        )

    def finalize_current(self) -> Optional[TranscriptResult]:
        """Called when external VAD detects silence to force a commit."""
        if len(self._audio_buffer) == 0:
            return None
            
        # Run one last high-quality pass? 
        # Or just return what we have as final.
        # For speed, we assume the last transcribe_stream was close enough, 
        # or we re-run specific logic.
        
        # Simple approach: Clear buffer and start fresh next time.
        self.reset_stream()
        return None # We rely on the stream having emitted the text already

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        """Convert int16 bytes to float32 normalized array."""
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        # Use frombuffer to avoid copy
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        return int16_arr.astype(np.float32) / 32768.0

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device_str
