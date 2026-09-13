"""Abstract provider lifecycle and capability contracts for EchoFlux.

Defines separate, distinct interfaces for Streaming Speech-to-Text,
Translation, and Assistant providers. All network and lifecycle methods are async
and orchestrator-compatible, strictly decoupling IO from blocking execution.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from engine.providers.types import (
    AssistRequest,
    AssistResult,
    AudioFrame,
    ProviderCapability,
    ProviderHealthCheck,
    STTTranscriptResult,
    SummaryChunk,
    SummaryRequest,
    SummaryResult,
    TranslationRequest,
    TranslationResult,
)


class BaseProvider(ABC):
    """Abstract base lifecycle contract implemented by all EchoFlux providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier of the provider instance or implementation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capability(self) -> ProviderCapability:
        """Specific capability handled by this provider."""
        raise NotImplementedError

    @abstractmethod
    async def start(self) -> None:
        """Initialize resources, connections, or background workers."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully terminate active connections and release resources."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> ProviderHealthCheck:
        """Perform a test connection or health check without running production workload."""
        raise NotImplementedError


class StreamingSTTProvider(BaseProvider, ABC):
    """Contract for realtime streaming Speech-to-Text providers."""

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability.STT

    @abstractmethod
    async def send_audio(self, frame: AudioFrame) -> None:
        """Enqueue or send an audio frame for streaming transcription."""
        raise NotImplementedError

    @abstractmethod
    def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        """Stream normalized transcript events (both partial and final).

        Implementations are `async def` generators consumed with `async for`.
        """
        raise NotImplementedError

    @abstractmethod
    async def finalize_stream(self, stream_id: str) -> None:
        """Signal endpoint/boundary for the given audio stream."""
        raise NotImplementedError

    @abstractmethod
    async def reset_stream(self, stream_id: str) -> None:
        """Clear stream state and buffers for the given audio stream."""
        raise NotImplementedError


class TranslationProvider(BaseProvider, ABC):
    """Contract for text translation providers."""

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability.TRANSLATION

    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text with normalized input/output and revision metadata."""
        raise NotImplementedError


class AssistantProvider(BaseProvider, ABC):
    """Contract for LLM assistant providers offering per-message assist and meeting summary."""

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability.ASSISTANT

    @abstractmethod
    async def assist(self, request: AssistRequest) -> AssistResult:
        """Generate structured suggestions and clarifying questions for a finalized message."""
        raise NotImplementedError

    @abstractmethod
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Generate a meeting summary from transcript messages."""
        raise NotImplementedError

    @abstractmethod
    def stream_summary(self, request: SummaryRequest) -> AsyncIterator[SummaryChunk]:
        """Stream a meeting summary chunk-by-chunk.

        Implementations are `async def` generators consumed with `async for`.
        """
        raise NotImplementedError
