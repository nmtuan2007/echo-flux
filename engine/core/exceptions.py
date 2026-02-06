class EchoFluxError(Exception):
    """Base exception for all EchoFlux errors."""


class ConfigError(EchoFluxError):
    """Configuration loading or validation error."""


class AudioError(EchoFluxError):
    """Audio capture or processing error."""


class ModelNotFoundError(EchoFluxError):
    """Requested model does not exist at the expected path."""


class ModelLoadError(EchoFluxError):
    """Failed to load a model into memory."""


class BackendError(EchoFluxError):
    """Generic backend initialization or runtime error."""


class ASRError(BackendError):
    """ASR-specific processing error."""


class TranslationError(BackendError):
    """Translation-specific processing error."""


class ServerError(EchoFluxError):
    """WebSocket server error."""
