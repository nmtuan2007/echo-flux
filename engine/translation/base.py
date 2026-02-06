from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: Optional[float] = None


class TranslationBackend(ABC):

    @abstractmethod
    def load_model(self, config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_pairs(self) -> list:
        """Return list of (source_lang, target_lang) tuples supported by this backend."""
        raise NotImplementedError
