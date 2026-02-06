from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    confidence: float = 0.0
    language: Optional[str] = None


class ASRBackend(ABC):

    @abstractmethod
    def load_model(self, config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def transcribe_stream(self, audio_chunk: bytes) -> TranscriptResult:
        raise NotImplementedError

    @abstractmethod
    def reset_stream(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError
