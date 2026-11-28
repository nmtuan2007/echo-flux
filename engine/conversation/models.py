"""Versioned domain records for local conversation and Context Profile storage.

Records are validated on construction so the repository never persists a malformed row.
Validation errors carry a location and a fixed code, never the offending value, because
the values here are transcript text, profile content, or something a user pasted by mistake.

Nothing in this module stores raw audio, credential values, or credential references.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from engine.core.config import SERVICE_NAMES, is_secret_key_name, is_valid_id_segment
from engine.providers.types import (
    AssistQuestion,
    AssistSuggestion,
    ContextProfileSnapshot,
    ConversationMessage,
    SummaryWarning,
)

# Bumped whenever a record's persisted shape changes; stored per row so a later migration
# can tell which shape it is reading.
RECORD_VERSION = 1

SOURCE_MIC = "mic"
SOURCE_SYSTEM = "system"
SOURCE_BOTH = "both"
SOURCE_UNKNOWN = "unknown"
SOURCES = (SOURCE_MIC, SOURCE_SYSTEM, SOURCE_BOTH, SOURCE_UNKNOWN)

# D-024: no diarization. A stream maps to a speaker label, and the user may rename it.
DEFAULT_SOURCE_LABELS: Dict[str, str] = {
    SOURCE_MIC: "Me",
    SOURCE_SYSTEM: "Others",
    SOURCE_BOTH: "Mixed",
    SOURCE_UNKNOWN: "Unknown",
}

PROFILE_FORMAT_TEXT = "text"
PROFILE_FORMAT_MARKDOWN = "markdown"
PROFILE_FORMATS = (PROFILE_FORMAT_TEXT, PROFILE_FORMAT_MARKDOWN)

# Explicit Context Profile limits. Content is measured in UTF-8 bytes because that is what
# reaches a provider's context window and the database page cache.
MAX_PROFILE_NAME_CHARS = 120
MAX_PROFILE_CONTENT_BYTES = 64 * 1024
MAX_PROFILES_PER_SESSION = 8
MAX_SESSION_PROFILE_BYTES = 256 * 1024

MAX_TITLE_CHARS = 200
MAX_LABEL_CHARS = 60
MAX_ID_CHARS = 128

OUTBOUND_ASSIST = "assist"
OUTBOUND_SUMMARY = "summary"
OUTBOUND_PURPOSES = (OUTBOUND_ASSIST, OUTBOUND_SUMMARY)

# Legacy ids look like "e-1712345678.123" or "1712345678123"; import ids add a namespace.
_RECORD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}")
_KEY_LIKE_PREFIXES = ("sk-", "sk_", "aiza", "gsk_", "xai-", "ghp_", "bearer")

BROWSER_HISTORY_IMPORT_PREFIX = "browser-history:v1:"


class ConversationValidationError(ValueError):
    """Invalid conversation or profile data. Carries location and code, never the value."""

    def __init__(self, path: str, code: str) -> None:
        super().__init__(f"{path}: {code}")
        self.path = path
        self.code = code


def validate_record_id(path: str, value: Any) -> str:
    if not isinstance(value, str) or not _RECORD_ID_RE.fullmatch(value):
        raise ConversationValidationError(path, "invalid_id")
    return value


def _validate_text(path: str, value: Any, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ConversationValidationError(path, "invalid_text")
    if "\x00" in value:
        raise ConversationValidationError(path, "binary_content")
    if not allow_empty and not value.strip():
        raise ConversationValidationError(path, "empty_text")
    return value


def _validate_short_text(path: str, value: Any, max_chars: int) -> str:
    _validate_text(path, value, allow_empty=False)
    if len(value) > max_chars or "\n" in value or "\r" in value:
        raise ConversationValidationError(path, "invalid_label")
    return value


def _validate_ms(path: str, value: Any, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConversationValidationError(path, "invalid_timestamp")


def _validate_count(path: str, value: Any, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversationValidationError(path, "invalid_count")


def import_id_for_browser_history(legacy_conversation_id: str) -> str:
    """Stable import id for a conversation from the webview's localStorage history.

    The later frontend import task sends each legacy conversation once per attempt; reusing
    this id makes a retried or repeated import a no-op instead of a duplicate.
    """
    validate_record_id("legacy_conversation_id", legacy_conversation_id)
    import_id = f"{BROWSER_HISTORY_IMPORT_PREFIX}{legacy_conversation_id}"
    return validate_record_id("import_id", import_id)


@dataclass(frozen=True)
class ProviderMetadata:
    """Which provider and model produced a result. Ids only; never endpoints or credentials."""

    provider_id: str
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if not is_valid_id_segment(self.provider_id):
            raise ConversationValidationError("provider.provider_id", "invalid_id")
        if self.model is not None and (
            not isinstance(self.model, str)
            or not _MODEL_RE.fullmatch(self.model)
            or self.model.lower().startswith(_KEY_LIKE_PREFIXES)
        ):
            raise ConversationValidationError("provider.model", "invalid_model")

    def to_dict(self) -> Dict[str, Any]:
        return {"provider_id": self.provider_id, "model": self.model}

    @classmethod
    def from_dict(cls, data: Any) -> "ProviderMetadata":
        if not isinstance(data, dict) or set(data) - {"provider_id", "model"}:
            raise ConversationValidationError("provider", "invalid_metadata")
        return cls(provider_id=data.get("provider_id"), model=data.get("model"))


def _validate_source_labels(labels: Any) -> None:
    if not isinstance(labels, dict):
        raise ConversationValidationError("source_labels", "invalid_labels")
    for source, label in labels.items():
        if source not in SOURCES:
            raise ConversationValidationError("source_labels", "unknown_source")
        _validate_short_text(f"source_labels.{source}", label, MAX_LABEL_CHARS)


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    title: str
    created_at_ms: float
    updated_at_ms: float
    source_labels: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCE_LABELS))
    import_id: Optional[str] = None
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("conversation_id", self.conversation_id)
        _validate_short_text("title", self.title, MAX_TITLE_CHARS)
        _validate_ms("created_at_ms", self.created_at_ms)
        _validate_ms("updated_at_ms", self.updated_at_ms)
        _validate_source_labels(self.source_labels)
        if self.import_id is not None:
            validate_record_id("import_id", self.import_id)

    def label_for(self, source: str) -> str:
        return self.source_labels.get(source, DEFAULT_SOURCE_LABELS.get(source, source))


@dataclass(frozen=True)
class SessionRecord:
    """One Start-to-Stop run. Provider choices are captured once and never edited."""

    session_id: str
    conversation_id: str
    started_at_ms: float
    ended_at_ms: Optional[float] = None
    providers: Dict[str, ProviderMetadata] = field(default_factory=dict)
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("session_id", self.session_id)
        validate_record_id("conversation_id", self.conversation_id)
        _validate_ms("started_at_ms", self.started_at_ms)
        _validate_ms("ended_at_ms", self.ended_at_ms, optional=True)
        if not isinstance(self.providers, dict):
            raise ConversationValidationError("providers", "invalid_metadata")
        for service, metadata in self.providers.items():
            if service not in SERVICE_NAMES:
                raise ConversationValidationError("providers", "unknown_service")
            if not isinstance(metadata, ProviderMetadata):
                raise ConversationValidationError(f"providers.{service}", "invalid_metadata")


@dataclass(frozen=True)
class TranscriptMessage:
    """A persisted transcript line. ``sequence`` is assigned by the repository on insert."""

    message_id: str
    source: str
    text: str
    timestamp_ms: float
    is_final: bool = True
    revision: int = 0
    translation: Optional[str] = None
    session_id: Optional[str] = None
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    detected_language: Optional[str] = None
    provider_event_id: Optional[str] = None
    stt_provider: Optional[ProviderMetadata] = None
    translation_provider: Optional[ProviderMetadata] = None
    sequence: Optional[int] = None
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("message_id", self.message_id)
        if self.source not in SOURCES:
            raise ConversationValidationError("source", "unknown_source")
        _validate_text("text", self.text)
        _validate_ms("timestamp_ms", self.timestamp_ms)
        if not isinstance(self.is_final, bool):
            raise ConversationValidationError("is_final", "invalid_flag")
        _validate_count("revision", self.revision)
        if self.translation is not None:
            _validate_text("translation", self.translation)
        if self.session_id is not None:
            validate_record_id("session_id", self.session_id)
        _validate_ms("start_ms", self.start_ms, optional=True)
        _validate_ms("end_ms", self.end_ms, optional=True)
        if self.detected_language is not None:
            _validate_short_text("detected_language", self.detected_language, 35)
        if self.provider_event_id is not None:
            _validate_short_text("provider_event_id", self.provider_event_id, 200)
        for name in ("stt_provider", "translation_provider"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ProviderMetadata):
                raise ConversationValidationError(name, "invalid_metadata")
        _validate_count("sequence", self.sequence, optional=True)

    def to_provider_message(self, labels: Dict[str, str]) -> ConversationMessage:
        speaker = labels.get(self.source, DEFAULT_SOURCE_LABELS[self.source])
        return ConversationMessage(
            message_id=self.message_id,
            speaker=speaker,
            text=self.text,
            timestamp_ms=float(self.timestamp_ms),
            stream_id=self.source,
            sequence=self.sequence,
            is_final=self.is_final,
        )


@dataclass(frozen=True)
class StoredAssistResult:
    assist_id: str
    conversation_id: str
    target_message_id: str
    suggested_replies: Tuple[AssistSuggestion, ...]
    clarifying_questions: Tuple[AssistQuestion, ...]
    created_at_ms: float
    provider: Optional[ProviderMetadata] = None
    provider_event_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("assist_id", self.assist_id)
        validate_record_id("conversation_id", self.conversation_id)
        validate_record_id("target_message_id", self.target_message_id)
        for i, reply in enumerate(self.suggested_replies):
            if not isinstance(reply, AssistSuggestion):
                raise ConversationValidationError(f"suggested_replies.{i}", "invalid_record")
        for i, question in enumerate(self.clarifying_questions):
            if not isinstance(question, AssistQuestion):
                raise ConversationValidationError(f"clarifying_questions.{i}", "invalid_record")
        _validate_ms("created_at_ms", self.created_at_ms)
        _validate_count("input_tokens", self.input_tokens, optional=True)
        _validate_count("output_tokens", self.output_tokens, optional=True)
        _validate_ms("latency_ms", self.latency_ms, optional=True)


@dataclass(frozen=True)
class StoredSummary:
    summary_id: str
    conversation_id: str
    summary_markdown: str
    created_at_ms: float
    bullet_points: Tuple[str, ...] = ()
    decisions: Tuple[str, ...] = ()
    action_items: Tuple[str, ...] = ()
    risks: Tuple[str, ...] = ()
    warning: Optional[SummaryWarning] = None
    provider: Optional[ProviderMetadata] = None
    provider_event_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("summary_id", self.summary_id)
        validate_record_id("conversation_id", self.conversation_id)
        _validate_text("summary_markdown", self.summary_markdown)
        _validate_ms("created_at_ms", self.created_at_ms)
        for name in ("bullet_points", "decisions", "action_items", "risks"):
            for i, item in enumerate(getattr(self, name)):
                _validate_text(f"{name}.{i}", item)
        if self.warning is not None:
            if not isinstance(self.warning, SummaryWarning):
                raise ConversationValidationError("warning", "invalid_record")
            _validate_count("warning.omitted_message_count", self.warning.omitted_message_count)
        _validate_count("input_tokens", self.input_tokens, optional=True)
        _validate_count("output_tokens", self.output_tokens, optional=True)
        _validate_ms("latency_ms", self.latency_ms, optional=True)


@dataclass(frozen=True)
class StoredAskTurn:
    ask_id: str
    conversation_id: str
    question: str
    answer_markdown: str
    created_at_ms: float
    provider: Optional[ProviderMetadata] = None
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("ask_id", self.ask_id)
        validate_record_id("conversation_id", self.conversation_id)
        _validate_text("question", self.question, allow_empty=False)
        _validate_text("answer_markdown", self.answer_markdown, allow_empty=False)
        _validate_ms("created_at_ms", self.created_at_ms)
        if self.provider is not None and not isinstance(self.provider, ProviderMetadata):
            raise ConversationValidationError("provider", "invalid_metadata")


def validate_profile_name(name: Any) -> str:
    return _validate_short_text("profile.name", name, MAX_PROFILE_NAME_CHARS)


def validate_profile_content(content: Any, content_format: Any) -> str:
    """Text/Markdown only, UTF-8 encodable, within ``MAX_PROFILE_CONTENT_BYTES``."""
    if content_format not in PROFILE_FORMATS:
        raise ConversationValidationError("profile.content_format", "unsupported_format")
    _validate_text("profile.content", content)
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ConversationValidationError("profile.content", "invalid_text") from exc
    if size > MAX_PROFILE_CONTENT_BYTES:
        raise ConversationValidationError("profile.content", "content_too_large")
    return content


@dataclass(frozen=True)
class ContextProfile:
    """Reusable, editable Context Profile. ``revision`` increments on every content edit."""

    profile_id: str
    name: str
    content: str
    content_format: str
    revision: int
    created_at_ms: float
    updated_at_ms: float
    record_version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        validate_record_id("profile_id", self.profile_id)
        validate_profile_name(self.name)
        validate_profile_content(self.content, self.content_format)
        _validate_count("revision", self.revision)
        _validate_ms("created_at_ms", self.created_at_ms)
        _validate_ms("updated_at_ms", self.updated_at_ms)


@dataclass(frozen=True)
class ContextProfileInfo:
    """Profile metadata for listings. Deliberately carries no content."""

    profile_id: str
    name: str
    content_format: str
    revision: int
    content_bytes: int
    updated_at_ms: float


@dataclass(frozen=True)
class SessionProfileSnapshot:
    """Profile content frozen at session start. Later edits or deletes never reach it."""

    session_id: str
    position: int
    profile_id: str
    name: str
    content: str
    content_format: str
    profile_revision: int
    content_sha256: str
    captured_at_ms: float
    record_version: int = RECORD_VERSION

    def to_provider_snapshot(self) -> ContextProfileSnapshot:
        return ContextProfileSnapshot(
            profile_id=self.profile_id, name=self.name, content=self.content
        )


def reject_secret_keys(path: str, data: Any) -> None:
    """Raise if any nested mapping key names a raw credential or credential reference."""
    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(key, str) or is_secret_key_name(key) or "credential" in key.lower():
                raise ConversationValidationError(path, "secret_field")
            reject_secret_keys(f"{path}.{key}", value)
    elif isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            reject_secret_keys(f"{path}.{i}", value)
