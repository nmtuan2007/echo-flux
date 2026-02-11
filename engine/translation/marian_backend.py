import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Tuple

from engine.translation.base import TranslationBackend, TranslationResult
from engine.core.config import Config
from engine.core.logging import get_logger
from engine.core.exceptions import ModelLoadError, TranslationError

logger = get_logger("translation.marian")

# Preset models mapping
_PRESET_MODELS = {
    ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
}

class MarianBackend(TranslationBackend):
    """
    Optimized MarianMT backend using CTranslate2.
    Automatically converts HuggingFace models to CTranslate2 INT8 format on first load.
    Robust GPU-to-CPU fallback included.
    """

    def __init__(self):
        self._translator = None
        self._tokenizer = None
        self._device = "cpu"
        self._source_lang: Optional[str] = None
        self._target_lang: Optional[str] = None
        self._model_name: Optional[str] = None
        self._ct2_model_path: Optional[Path] = None

    def load_model(self, config: dict) -> None:
        try:
            import ctranslate2
            from transformers import MarianTokenizer
        except ImportError:
            raise ModelLoadError("ctranslate2 or transformers not installed")

        self._source_lang = config.get("translation.source_lang", "en")
        self._target_lang = config.get("translation.target_lang", "vi")

        # 1. Resolve potential device
        device_str = config.get("device", "auto")
        potential_device = "cpu"
        if device_str == "cuda":
            potential_device = "cuda"
        elif device_str == "auto":
            try:
                if ctranslate2.get_cuda_device_count() > 0:
                    potential_device = "cuda"
            except Exception:
                potential_device = "cpu"

        # 2. Resolve model name
        custom_model = config.get("translation.model")
        if custom_model:
            self._model_name = custom_model
        else:
            pair = (self._source_lang, self._target_lang)
            self._model_name = _PRESET_MODELS.get(
                pair,
                f"Helsinki-NLP/opus-mt-{self._source_lang}-{self._target_lang}",
            )

        # 3. Prepare paths and convert if needed
        app_config = Config()
        ct2_dir = app_config.models_dir / "ct2"
        safe_name = self._model_name.replace("/", "_")
        self._ct2_model_path = ct2_dir / safe_name

        if not self._is_valid_ct2_model(self._ct2_model_path):
            self._convert_model(self._model_name, self._ct2_model_path)

        # 4. Load Tokenizer
        try:
            self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
        except Exception as e:
            raise ModelLoadError(f"Failed to load tokenizer for '{self._model_name}': {e}") from e

        # 5. Load and Test Translator (Robust Fallback Logic)
        self._load_translator_safe(str(self._ct2_model_path), potential_device)

    def _load_translator_safe(self, model_path: str, device: str):
        import ctranslate2

        # Try loading with the preferred device
        try:
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info("Attempting to load CTranslate2 model on %s (%s)...", device, compute_type)

            translator = ctranslate2.Translator(
                model_path,
                device=device,
                compute_type=compute_type
            )

            # SELF-TEST: Try to translate a dummy token to verify libraries are actually working
            # This catches missing DLLs (cublas64_12.dll) which only crash at runtime
            translator.translate_batch([["test"]], max_batch_size=1)

            # If successful
            self._translator = translator
            self._device = device
            logger.info("Translation model loaded and verified on %s.", device)
            return

        except Exception as e:
            error_msg = str(e).lower()
            is_library_error = "dll" in error_msg or "library" in error_msg or "cublas" in error_msg or "cudnn" in error_msg

            if device == "cuda" and is_library_error:
                logger.warning(
                    "Failed to initialize CUDA inference (missing libraries?): %s. "
                    "Falling back to CPU (INT8).", e
                )
                # Recursive call to load on CPU
                self._load_translator_safe(model_path, "cpu")
            elif device == "cuda":
                logger.warning("Unknown CUDA error: %s. Falling back to CPU.", e)
                self._load_translator_safe(model_path, "cpu")
            else:
                # If it failed on CPU, it's a fatal error
                raise ModelLoadError(f"Failed to load CTranslate2 model on CPU: {e}") from e

    def _is_valid_ct2_model(self, path: Path) -> bool:
        return path.exists() and (path / "model.bin").exists() and (path / "config.json").exists()

    def _convert_model(self, model_name: str, output_path: Path):
        import ctranslate2.converters

        logger.info("Converting '%s' to CTranslate2 INT8 format...", model_name)
        try:
            converter = ctranslate2.converters.TransformersConverter(model_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            converter.convert(str(output_path), quantization="int8", force=True)
            logger.info("Conversion complete.")
        except Exception as e:
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)
            raise ModelLoadError(f"Failed to convert model '{model_name}': {e}") from e

    def translate_raw(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not self._translator or not self._tokenizer:
            raise TranslationError("Translation model not loaded")

        if not text.strip():
            return TranslationResult(text, "", source_lang, target_lang)

        try:
            sentences = self.split_sentences(text, max_length=200)

            source_tokens = [
                self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(s))
                for s in sentences
            ]

            results = self._translator.translate_batch(
                source_tokens,
                batch_type="tokens",
                max_batch_size=2048,
                beam_size=2,
            )

            translated_sentences = []
            for res in results:
                decoded = self._tokenizer.decode(
                    self._tokenizer.convert_tokens_to_ids(res.hypotheses[0])
                )
                translated_sentences.append(decoded)

            translated_text = " ".join(translated_sentences)

            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        except Exception as e:
            logger.error("CTranslate2 inference error: %s", e)
            raise TranslationError(f"Translation failed: {e}") from e

    def unload_model(self) -> None:
        if self._translator:
            del self._translator
        self._translator = None
        self._tokenizer = None
        import gc
        gc.collect()
        logger.info("MarianBackend unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._translator is not None

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        if self._source_lang and self._target_lang:
            return [(self._source_lang, self._target_lang)]
        return []

    @staticmethod
    def get_preset_models() -> dict:
        return dict(_PRESET_MODELS)
