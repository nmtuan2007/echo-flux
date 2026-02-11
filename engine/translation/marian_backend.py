from typing import Optional, List, Tuple

from engine.translation.base import TranslationBackend, TranslationResult
from engine.core.logging import get_logger
from engine.core.exceptions import ModelLoadError, TranslationError

logger = get_logger("translation.marian")


class MarianBackend(TranslationBackend):

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._source_lang: Optional[str] = None
        self._target_lang: Optional[str] = None
        self._model_name: Optional[str] = None

    def load_model(self, config: dict) -> None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ModelLoadError("transformers library is not installed")

        self._source_lang = config.get("source_lang", "en")
        self._target_lang = config.get("target_lang", "vi")
        self._model_name = config.get(
            "model_name",
            f"Helsinki-NLP/opus-mt-{self._source_lang}-{self._target_lang}",
        )
        model_path = config.get("model_path", self._model_name)
        self._device = self._resolve_device(config.get("device", "auto"))

        logger.info(
            "Loading translation model: %s (device=%s)",
            model_path, self._device,
        )

        try:
            self._tokenizer = MarianTokenizer.from_pretrained(model_path)
            self._model = MarianMTModel.from_pretrained(model_path)
        except Exception as e:
            self._model = None
            self._tokenizer = None
            raise ModelLoadError(f"Failed to load Marian model: {e}") from e

        try:
            self._model.to(self._device)
        except Exception as e:
            if self._device == "cuda":
                logger.warning(
                    "Failed to move translation model to CUDA: %s. Falling back to CPU.", e
                )
                self._device = "cpu"
                self._model.to(self._device)
            else:
                raise ModelLoadError(f"Failed to place model on {self._device}: {e}") from e

        self._model.eval()
        logger.info("Translation model loaded successfully (device=%s)", self._device)

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not self._model or not self._tokenizer:
            raise TranslationError("Translation model not loaded")

        if not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
            )

        try:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                translated_ids = self._model.generate(**inputs)

            translated_text = self._tokenizer.decode(
                translated_ids[0], skip_special_tokens=True
            )

            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        except Exception as e:
            logger.error("Translation error: %s", e)
            raise TranslationError(f"Translation failed: {e}") from e

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        logger.info("Translation model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        if self._source_lang and self._target_lang:
            return [(self._source_lang, self._target_lang)]
        return []

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA available — using GPU for translation")
                return "cuda"
        except ImportError:
            pass

        if device == "cuda":
            logger.warning(
                "CUDA was explicitly requested but is not available. "
                "Ensure PyTorch is installed with CUDA support (e.g. pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121). Falling back to CPU."
            )
        else:
            logger.info("CUDA not available — using CPU for translation")

        return "cpu"
