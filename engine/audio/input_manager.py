from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from engine.core.logging import get_logger

logger = get_logger("audio.input")


class AudioSourceType(Enum):
    MICROPHONE = "microphone"
    SYSTEM = "system"


@dataclass
class AudioDevice:
    id: str
    name: str
    source_type: AudioSourceType
    sample_rate: int = 16000
    channels: int = 1


class AudioInput(ABC):

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def is_active(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_devices(self) -> List[AudioDevice]:
        raise NotImplementedError


class InputManager:
    def __init__(self, config: dict):
        self._config = config
        self._source: Optional[AudioInput] = None
        self._device: Optional[AudioDevice] = None

    def set_source(self, source: AudioInput, device: Optional[AudioDevice] = None):
        if self._source and self._source.is_active():
            logger.info("Stopping current audio source before switching")
            self._source.stop()
        self._source = source
        self._device = device
        logger.info("Audio source set: %s", type(source).__name__)

    def start(self) -> None:
        if not self._source:
            raise RuntimeError("No audio source configured")
        logger.info("Starting audio capture")
        self._source.start()

    def stop(self) -> None:
        if self._source and self._source.is_active():
            logger.info("Stopping audio capture")
            self._source.stop()

    def read_chunk(self) -> bytes:
        if not self._source:
            raise RuntimeError("No audio source configured")
        return self._source.read_chunk()

    @property
    def is_active(self) -> bool:
        return self._source is not None and self._source.is_active()

    @property
    def current_device(self) -> Optional[AudioDevice]:
        return self._device

    @property
    def sample_rate(self) -> int:
        return self._config.get("sample_rate", 16000)

    @property
    def channels(self) -> int:
        return self._config.get("channels", 1)

    @property
    def chunk_size_bytes(self) -> int:
        chunk_ms = self._config.get("chunk_ms", 20)
        bytes_per_sample = 2  # int16
        return int(self.sample_rate * self.channels * bytes_per_sample * chunk_ms / 1000)
