import logging
import threading
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
        self._lock = threading.Lock()

    def load_model(self, config: TranscriptionConfig) -> None:
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            logger.error("transformers or torch not installed. Cannot use TransformersASRAdapter.")
            raise

        self._config = config
        model_id = config.model_path or config.model_size

        device = "cpu"
        model_kwargs = {}
        
        cuda_avail = torch.cuda.is_available()
        logger.warning(f"CUDA Check: config.device='{config.device}', torch.cuda.is_available()={cuda_avail}")
        
        if config.device in ("cuda", "auto", "cuda:0", 0) and cuda_avail:
            device = "cuda:0"
            model_kwargs["torch_dtype"] = torch.float16
            
        # In case the user forces cuda and torch says false, log it!
        if config.device in ("cuda", "cuda:0", 0) and not cuda_avail:
            logger.error("User requested CUDA but torch.cuda.is_available() returned False! Falling back to CPU.")
            device = "cpu"
        
        logger.info(f"Loading HF transformers ASR pipeline for {model_id} on device {device}")
        
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            **model_kwargs
        )
        
        # Patch for outdated generation_configs in older Whisper models
        if hasattr(self._pipe.model, "generation_config"):
            gen_config = self._pipe.model.generation_config
            if not hasattr(gen_config, "lang_to_id") and hasattr(self._pipe.tokenizer, "get_vocab"):
                gen_config.lang_to_id = self._pipe.tokenizer.get_vocab()
            if not hasattr(gen_config, "task_to_id") and hasattr(self._pipe.tokenizer, "get_vocab"):
                gen_config.task_to_id = self._pipe.tokenizer.get_vocab()
            if not hasattr(gen_config, "is_multilingual"):
                gen_config.is_multilingual = True
        
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
                "last_inferred_len": 0
            }
        return self._streams[stream_id]

    def transcribe_stream(self, audio_chunk: bytes, stream_id: str = "default") -> Optional[TranscriptResult]:
        if not self._pipe:
            return None

        stream = self._get_stream(stream_id)
        new_samples = self._bytes_to_float32(audio_chunk)
        stream["audio_buffer"] = np.concatenate((stream["audio_buffer"], new_samples))

        max_buffer_duration = 5.0
        max_samples = int(max_buffer_duration * 16000)

        if len(stream["audio_buffer"]) > max_samples:
            logger.debug(f"ASR [{stream_id}]: Buffer exceeded max_samples, forcing finalize.")
            return self.finalize_current(stream_id)

        if len(stream["audio_buffer"]) - stream.get("last_inferred_len", 0) >= 16000:
            audio_data = stream["audio_buffer"]
            stream["last_inferred_len"] = len(audio_data)
            
            logger.info(f"[Partial Inference] Triggering on {len(audio_data)} samples. "
                         f"Audio max={np.max(audio_data):.4f}, min={np.min(audio_data):.4f}, mean={np.mean(audio_data):.4f}")
            
            try:
                gen_kwargs = {
                    "max_length": 64,  # Force maximum sequence length to be extremely short
                    "condition_on_prev_tokens": False
                }
                is_multilingual = getattr(self._pipe.model.generation_config, "is_multilingual", True)
                if self._config.language != "auto" and is_multilingual:
                    gen_kwargs["language"] = self._config.language
                
                logger.info(f"[Partial Inference] gen_kwargs: {gen_kwargs}")
                
                try:
                    t0 = time.time()
                    with self._lock:
                        if not self._pipe: return None
                        result = self._pipe(
                            {"sampling_rate": 16000, "raw": audio_data},
                            generate_kwargs=gen_kwargs
                        )
                    t1 = time.time()
                    logger.info(f"[Partial Inference] _pipe (attempt 1) finished in {t1 - t0:.2f}s. Raw result: {result}")
                except Exception as inner_e:
                    if "generation config is outdated" in str(inner_e) or "language" in str(inner_e):
                        logger.warning(f"Retrying without language kwarg due to issue: {inner_e}")
                        gen_kwargs.pop("language", None)
                        t0 = time.time()
                        with self._lock:
                            if not self._pipe: return None
                            result = self._pipe(
                                {"sampling_rate": 16000, "raw": audio_data},
                                generate_kwargs=gen_kwargs
                            )
                        t1 = time.time()
                        logger.info(f"[Partial Inference] _pipe (attempt 2) finished in {t1 - t0:.2f}s. Raw result: {result}")
                    else:
                        raise
                
                text = result.get("text", "").strip()
                if text:
                    return TranscriptResult(
                        text=text,
                        is_final=False,
                        language=self._config.language,
                        stream_id=stream_id
                    )
            except Exception as e:
                logger.error(f"Transformers partial transcribe error: {e}")

        return None

    def finalize_current(self, stream_id: str = "default") -> Optional[TranscriptResult]:
        if not self._pipe:
            return None

        stream = self._get_stream(stream_id)
        if len(stream["audio_buffer"]) == 0:
            return None

        audio_data = stream["audio_buffer"]
        logger.info(f"[Final Inference] Finalizing on {len(audio_data)/16000:.2f}s audio. "
                     f"Audio max={np.max(audio_data):.4f}, min={np.min(audio_data):.4f}, mean={np.mean(audio_data):.4f}")
        
        try:
            gen_kwargs = {
                "max_length": 64,  # Force maximum sequence length to be extremely short
                "condition_on_prev_tokens": False
            }
            is_multilingual = getattr(self._pipe.model.generation_config, "is_multilingual", True)
            if self._config.language != "auto" and is_multilingual:
                gen_kwargs["language"] = self._config.language
                
            logger.info(f"[Final Inference] gen_kwargs: {gen_kwargs}")
            
            try:
                t0 = time.time()
                with self._lock:
                    if not self._pipe: return None
                    result = self._pipe(
                        {"sampling_rate": 16000, "raw": audio_data},
                        generate_kwargs=gen_kwargs
                    )
                t1 = time.time()
                logger.info(f"[Final Inference] _pipe (attempt 1) finished in {t1 - t0:.2f}s. Raw result: {result}")
            except Exception as inner_e:
                if "generation config is outdated" in str(inner_e) or "language" in str(inner_e):
                    logger.warning(f"Retrying without language kwarg due to issue: {inner_e}")
                    gen_kwargs.pop("language", None)
                    t0 = time.time()
                    with self._lock:
                        if not self._pipe: return None
                        result = self._pipe(
                            {"sampling_rate": 16000, "raw": audio_data},
                            generate_kwargs=gen_kwargs
                        )
                    t1 = time.time()
                    logger.info(f"[Final Inference] _pipe (attempt 2) finished in {t1 - t0:.2f}s. Raw result: {result}")
                else:
                    raise
            text = result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Transformers pipeline error: {e}")
            text = ""
            
        stream["audio_buffer"] = np.array([], dtype=np.float32)
        stream["last_inferred_len"] = 0

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
