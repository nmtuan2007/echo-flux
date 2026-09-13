"""Canonical domain types and error definitions for EchoFlux providers.

Contains dependency-light dataclasses and enums for Speech-to-Text (STT),
Translation, and Assistant capabilities. All types avoid importing heavy ML SDKs,
network clients, or secrets.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ProviderCapability(str, Enum):
    """Core capabilities supported across EchoFlux providers."""

    STT = "stt"
    TRANSLATION = "translation"
    ASSISTANT = "assistant"


class ProviderHealthStatus(str, Enum):
    """Health check outcome status for provider connectivity tests."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProviderErrorCode(str, Enum):
    """Categorized stable error codes across all provider implementations."""

    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK_UNAVAILABLE = "network_unavailable"
    TIMEOUT = "timeout"
    CONNECTION_CLOSED = "connection_closed"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    CONTEXT_OVERFLOW = "context_overflow"
    INTERNAL_ERROR = "internal_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProviderError:
    """Normalized provider error record.

    Ensures consistent error reporting with a safe, non-leaking user message.
    Raw requests, API keys, or raw configurations are strictly excluded.
    """

    code: ProviderErrorCode
    capability: ProviderCapability
    message: str
    retryable: bool = False
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"[{self.capability.value}:{self.code.value}] {self.message}"


class ProviderRuntimeError(Exception):
    """Standard exception wrapper for provider errors."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(str(error))
        self.error = error


# Backward-compatible alias
ProviderException = ProviderRuntimeError


@dataclass(frozen=True)
class ProviderHealthCheck:
    """Normalized test connection and health check result."""

    status: ProviderHealthStatus
    message: str = ""
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    checked_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AudioFrame:
    """Normalized audio frame payload for streaming STT ingestion."""

    stream_id: str
    sequence: int
    pcm_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    monotonic_timestamp: float = 0.0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class WordTiming:
    """Word-level alignment and timing information."""

    word: str
    start_time_ms: float
    end_time_ms: float
    confidence: Optional[float] = None


@dataclass(frozen=True)
class STTTranscriptResult:
    """Normalized Speech-to-Text event result with revision and interval tracking."""

    stream_id: str
    sequence: int
    provider_event_id: str
    text: str
    is_final: bool
    revision: int
    start_time_ms: float
    end_time_ms: float
    monotonic_timestamp: float
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    words: Optional[Tuple[WordTiming, ...]] = None
    is_endpoint: bool = False


@dataclass(frozen=True)
class TranslationRequest:
    """Request payload for translation providers."""

    text: str
    source_lang: str
    target_lang: str
    stream_id: str = "default"
    sequence: Optional[int] = None
    revision: Optional[int] = None
    is_final: bool = True


@dataclass(frozen=True)
class TranslationResult:
    """Normalized Translation result with revision tracking and token usage."""

    translated_text: str
    source_text: str
    source_lang: str
    target_lang: str
    stream_id: str = "default"
    sequence: Optional[int] = None
    revision: Optional[int] = None
    is_final: bool = True
    detected_source_lang: Optional[str] = None
    provider_event_id: Optional[str] = None
    monotonic_timestamp: float = field(default_factory=time.monotonic)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass(frozen=True)
class ConversationMessage:
    """Normalized conversation message for assistant context windows."""

    message_id: str
    speaker: str
    text: str
    timestamp_ms: float
    stream_id: str = "default"
    sequence: Optional[int] = None
    is_final: bool = True


@dataclass(frozen=True)
class ContextProfileSnapshot:
    """Immutable context profile snapshot attached to assistant queries."""

    profile_id: str
    name: str
    content: str


@dataclass(frozen=True)
class AssistSuggestion:
    """Single suggested reply strategy from the assistant."""

    strategy: str
    text: str
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class AssistQuestion:
    """Single clarifying or probing question from the assistant."""

    question: str
    purpose: Optional[str] = None


@dataclass(frozen=True)
class AssistRequest:
    """Input payload for a targeted assistant assist query."""

    target_message: ConversationMessage
    context_messages: Tuple[ConversationMessage, ...] = ()
    context_profiles: Tuple[ContextProfileSnapshot, ...] = ()
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class AssistResult:
    """Normalized Assist result containing both suggested replies and clarifying questions."""

    target_message_id: str
    suggested_replies: Tuple[AssistSuggestion, ...]
    clarifying_questions: Tuple[AssistQuestion, ...]
    provider_event_id: Optional[str] = None
    monotonic_timestamp: float = field(default_factory=time.monotonic)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass(frozen=True)
class SummaryRequest:
    """Input payload for full-transcript summarization."""

    messages: Tuple[ConversationMessage, ...]
    reserved_output_tokens: int = 1000
    custom_instructions: Optional[str] = None


@dataclass(frozen=True)
class SummaryWarning:
    """Warning metadata when transcript exceeds provider context and oldest messages are omitted."""

    omitted_message_count: int = 0
    first_omitted_timestamp_ms: Optional[float] = None
    last_omitted_timestamp_ms: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class SummaryResult:
    """Normalized full-transcript summary result."""

    summary_markdown: str
    bullet_points: Tuple[str, ...] = ()
    decisions: Tuple[str, ...] = ()
    action_items: Tuple[str, ...] = ()
    risks: Tuple[str, ...] = ()
    warning: Optional[SummaryWarning] = None
    provider_event_id: Optional[str] = None
    monotonic_timestamp: float = field(default_factory=time.monotonic)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass(frozen=True)
class SummaryChunk:
    """Incremental chunk streamed during summary generation."""

    delta_text: str
    is_done: bool = False
