import logging
import time
from typing import Optional

import numpy as np

from engine.core.config import TranscriptionConfig
from engine.asr.base import ASRBackend, TranscriptResult

logger = logging.getLogger("echoflux.asr.transformers")

class TransformersASRAdapter(ASRBackend):
    def __init__(self):
        self._pipe = None
        self._config: Optional[TranscriptionConfig] = None
        self._streams = {}

    def load_model(self, config: TranscriptionConfig) -> None:
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            logger.error("transformers or torch not installed. Cannot use TransformersASRAdapter.")
            raise

        self._config = config
        model_id = config.model_path or config.model_size

        device = 0 if (config.device == "cuda" or config.device == "auto" and torch.cuda.is_available()) else -1
        
        logger.info(f"Loading HF transformers ASR pipeline for {model_id} on device {device}")
        
        # Determine data type based on config or device
        torch_dtype = torch.float16 if device == 0 else torch.float32

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype
        )
        logger.info("HF transformers ASR pipeline loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def unload_model(self) -> None:
        if self._pipe:
            del self._pipe
        self._pipe = None
        self._streams.clear()

    def reset_stream(self, stream_id: str = "default") -> None:
        if stream_id in self._streams:
            del self._streams[stream_id]

    def _get_stream(self, stream_id: str) -> dict:
        if stream_id not in self._streams:
            self._streams[stream_id] = {
                "audio_buffer": np.array([], dtype=np.float32),
            }
        return self._streams[stream_id]

    def transcribe_stream(self, audio_chunk: bytes, stream_id: str = "default") -> Optional[TranscriptResult]:
        if not self._pipe:
            return None

        stream = self._get_stream(stream_id)
        new_samples = self._bytes_to_float32(audio_chunk)
        stream["audio_buffer"] = np.concatenate((stream["audio_buffer"], new_samples))

        # We will only infer on finalize to keep transformers simple due to no native chunking stream
        return None

    def finalize_current(self, stream_id: str = "default") -> Optional[TranscriptResult]:
        if not self._pipe:
            return None

        stream = self._get_stream(stream_id)
        if len(stream["audio_buffer"]) == 0:
            return None

        logger.debug(f"Transformers ASR logic finalized on {len(stream['audio_buffer'])/16000:.2f}s audio.")
        
        try:
            # We must pass sampling rate to the pipeline
            result = self._pipe(
                {"sampling_rate": 16000, "raw": stream["audio_buffer"]},
                generate_kwargs={"language": self._config.language} if self._config.language != "auto" else {}
            )
            text = result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Transformers pipeline error: {e}")
            text = ""
            
        stream["audio_buffer"] = np.array([], dtype=np.float32)

        if text:
            return TranscriptResult(
                text=text,
                is_final=True,
                language=self._config.language,
                stream_id=stream_id
            )
        return None

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        return int16_arr.astype(np.float32) / 32768.0
