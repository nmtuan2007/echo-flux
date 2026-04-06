import logging
import threading
import time
import os
from queue import Queue, Empty, Full
from engine.core.event_bus import EventBus
from engine.asr.base import TranscriptResult

logger = logging.getLogger("echoflux.inference")

class InferenceController:
    """
    Manages loading and invoking ASR and Translation models.
    Takes discrete speech chunks from AudioPipeline, infers transcripts, 
    and pipes outputs to the EventBus.
    """
    def __init__(self, settings: dict, event_bus: EventBus):
        self._settings = settings
        self._event_bus = event_bus
        self._asr_backend = None
        self._translation_backend = None

        self._translation_queue = Queue(maxsize=100)
        self._running = False
        self._translation_thread = None

        self._initialize_models()

    def _initialize_models(self):
        from engine.asr import AutoModelAdapter
        from engine.core.config import TranscriptionConfig

        asr_config = TranscriptionConfig(
            model_size=self._settings.get("asr.model_size", "small"),
            language=self._settings.get("asr.language", "en"),
            device=self._settings.get("asr.device", "auto"),
            compute_type=os.getenv("ECHOFLUX_COMPUTE_TYPE", "float16"),
        )
        self._asr_backend = AutoModelAdapter.load(asr_config)
        self._asr_backend.load_model(asr_config)

        if self._settings.get("translation.enabled", False):
            from engine.translation.fallback_backend import FallbackTranslationBackend
            source_lang = self._settings.get("translation.source_lang", "en")
            target_lang = self._settings.get("translation.target_lang", "vi")
            backend = self._settings.get("translation.backend", "online")
            custom_model = self._settings.get("translation.model")

            translation_config = {
                "translation.backend": backend,
                "translation.source_lang": source_lang,
                "translation.target_lang": target_lang,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "device": self._settings.get("asr.device", "auto"),
            }
            if custom_model:
                translation_config["translation.model"] = custom_model

            self._translation_backend = FallbackTranslationBackend()
            self._translation_backend.load_model(translation_config)

    def start(self):
        self._running = True
        with self._translation_queue.mutex:
            self._translation_queue.queue.clear()
            
        if self._translation_backend:
            self._translation_thread = threading.Thread(
                target=self._translation_loop, 
                name="TranslationThread", 
                daemon=True
            )
            self._translation_thread.start()

    def stop(self, stream_ids: list):
        self._running = False
        if self._translation_thread:
            self._translation_thread.join(timeout=2.0)

        if self._asr_backend:
            for stream_id in stream_ids:
                result = self._asr_backend.finalize_current(stream_id)
                if result:
                    self._enqueue_asr_result(result)
                else:
                    self._enqueue_asr_result(TranscriptResult(text="", is_final=True, language="auto", stream_id=stream_id))
            self._asr_backend.unload_model()
            self._asr_backend = None

        if self._translation_backend:
            self._translation_backend.unload_model()
            self._translation_backend = None

    def handle_speech_chunk(self, stream_id: str, audio_data: bytes):
        if not self._asr_backend: return
        result = self._asr_backend.transcribe_stream(audio_data, stream_id)
        if result:
            self._enqueue_asr_result(result)

    def handle_finalize(self, stream_id: str):
        if not self._asr_backend: return
        result = self._asr_backend.finalize_current(stream_id)
        if result:
            self._enqueue_asr_result(result)
        else:
            self._enqueue_asr_result(TranscriptResult(text="", is_final=True, language="auto", stream_id=stream_id))

    def _enqueue_asr_result(self, asr_result):
        active_backend = None
        if self._translation_backend and hasattr(self._translation_backend, "active_backend"):
            active_backend = self._translation_backend.active_backend

        audio_source = asr_result.stream_id if asr_result.stream_id in ["mic", "system"] else None
        entry_id = f"e-{time.time()}" if asr_result.is_final else None

        msg = {
            "type": "partial" if not asr_result.is_final else "final",
            "text": asr_result.text,
            "translation": None,
            "is_final": asr_result.is_final,
            "timestamp": time.time(),
        }

        if audio_source: 
            msg["source"] = audio_source
        if entry_id: 
            msg["entry_id"] = entry_id
        if active_backend: 
            msg["translation_backend"] = active_backend

        self._event_bus.emit(msg)

        if self._translation_backend and self._translation_backend.is_loaded:
            if asr_result.is_final and asr_result.text.strip():
                try:
                    self._translation_queue.put({"text": asr_result.text, "entry_id": entry_id, "is_final": True}, timeout=0.5)
                except Full:
                    pass

    def _translation_loop(self):
        src_lang = self._settings.get("translation.source_lang", "auto")
        tgt_lang = self._settings.get("translation.target_lang", "vi")

        while self._running:
            try:
                task = self._translation_queue.get(timeout=0.5)
            except Empty:
                continue

            text = task.get("text", "")
            entry_id = task.get("entry_id")
            if not text:
                continue

            try:
                trans_res = self._translation_backend.translate(text, src_lang, tgt_lang)
                if trans_res.translated_text:
                    active_backend = getattr(self._translation_backend, "active_backend", None)
                    msg = {
                        "type": "translation_update",
                        "entry_id": entry_id,
                        "source_text": text,
                        "translation": trans_res.translated_text,
                        "timestamp": time.time(),
                        "is_final": True
                    }
                    if active_backend: 
                        msg["translation_backend"] = active_backend
                    self._event_bus.emit(msg)
            except Exception as e:
                logger.error("Translation loop error: %s", e)
