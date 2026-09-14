"""Manual per-message Assist application service.

One explicit request targets one persisted, finalized message. The service reads the
conversation and the session's Context Profile snapshots off the event loop, calls the
configured Assistant provider once (plus at most one schema-repair attempt), re-validates the
reply, and stores it against the target message.

Failure is local to that message: nothing is written on any error, no other provider is
tried, and errors carry fixed messages with codes, never transcript or profile text.
"""

import asyncio
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Tuple

from engine.assistant.context_window import (
    AssistContextWindow,
    ContextBudget,
    ContextSelectionError,
    select_assist_context,
)
from engine.assistant.prompt_builder import (
    REPAIR_INSTRUCTION,
    AssistResponseSchemaError,
    normalize_assist_result,
)
from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    OUTBOUND_ASSIST,
    ConversationValidationError,
    ProviderMetadata,
    StoredAssistResult,
    validate_record_id,
)
from engine.conversation.repository import (
    ConversationRepository,
    RecordNotFoundError,
    RepositoryError,
)
from engine.core.logging import get_logger
from engine.providers.contracts import AssistantProvider
from engine.providers.types import (
    AssistRequest,
    AssistResult,
    ProviderErrorCode,
    ProviderRuntimeError,
)

logger = get_logger("assistant.service")

_ASSISTANT_SERVICE = "assistant"


class AssistErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_NOT_FINAL = "target_not_final"
    TARGET_EMPTY = "target_empty"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"
    STORAGE_ERROR = "storage_error"
    INTERNAL_ERROR = "internal_error"


_MESSAGES = {
    AssistErrorCode.INVALID_REQUEST: "Assist request is invalid.",
    AssistErrorCode.TARGET_NOT_FOUND: "The message to assist with was not found.",
    AssistErrorCode.TARGET_NOT_FINAL: "Assist is available only for finalized messages.",
    AssistErrorCode.TARGET_EMPTY: "The message to assist with has no text.",
    AssistErrorCode.PROVIDER_ERROR: "The Assistant provider could not complete the request.",
    AssistErrorCode.INVALID_RESPONSE: "The Assistant returned a response in an invalid format.",
    AssistErrorCode.STORAGE_ERROR: "The Assist result could not be read or saved locally.",
    AssistErrorCode.INTERNAL_ERROR: "Assist failed unexpectedly.",
}

_SELECTION_CODES = {
    "target_not_found": AssistErrorCode.TARGET_NOT_FOUND,
    "target_not_final": AssistErrorCode.TARGET_NOT_FINAL,
    "target_empty": AssistErrorCode.TARGET_EMPTY,
}


@dataclass(frozen=True)
class AssistError:
    """Safe, message-local failure. ``provider_code`` is a stable ProviderErrorCode value."""

    code: AssistErrorCode
    message: str
    retryable: bool = False
    provider_code: Optional[str] = None


@dataclass(frozen=True)
class AssistOutcome:
    conversation_id: str
    target_message_id: str
    stored: Optional[StoredAssistResult] = None
    error: Optional[AssistError] = None
    session_id: Optional[str] = None
    provider: Optional[ProviderMetadata] = None
    context_message_count: int = 0
    omitted_message_count: int = 0
    profile_count: int = 0
    omitted_profile_count: int = 0
    repair_attempted: bool = False

    @property
    def ok(self) -> bool:
        return self.stored is not None and self.error is None


def _error(
    code: AssistErrorCode, retryable: bool = False, provider_code: Optional[str] = None
) -> AssistError:
    return AssistError(code, _MESSAGES[code], retryable=retryable, provider_code=provider_code)


class _AssistFailedError(Exception):
    def __init__(self, error: AssistError, repaired: bool = False) -> None:
        super().__init__(error.code.value)
        self.error = error
        self.repaired = repaired


class _SchemaViolationError(Exception):
    """A reply broke the output contract; eligible for the single repair attempt."""


def _now_ms() -> float:
    return time.time() * 1000.0


class AssistantService:
    """Explicit Assist on one finalized message. Call ``request_assist`` from the event loop.

    Storage work runs through ``asyncio.to_thread``; the provider call is awaited directly
    because provider adapters are async and non-blocking.
    """

    def __init__(
        self,
        repository: ConversationRepository,
        profiles: ContextProfileService,
        provider: AssistantProvider,
        model: Optional[str] = None,
        budget: Optional[ContextBudget] = None,
        clock_ms: Callable[[], float] = _now_ms,
    ) -> None:
        self._repo = repository
        self._profiles = profiles
        self._provider = provider
        self._model = model
        self._budget = budget or ContextBudget()
        self._clock_ms = clock_ms

    async def request_assist(self, conversation_id: str, target_message_id: str) -> AssistOutcome:
        base = AssistOutcome(
            conversation_id=conversation_id if isinstance(conversation_id, str) else "",
            target_message_id=target_message_id if isinstance(target_message_id, str) else "",
        )
        try:
            validate_record_id("conversation_id", conversation_id)
            validate_record_id("target_message_id", target_message_id)
        except ConversationValidationError:
            return replace(base, error=_error(AssistErrorCode.INVALID_REQUEST))

        try:
            window, metadata = await asyncio.to_thread(
                self._prepare, conversation_id, target_message_id
            )
        except _AssistFailedError as failure:
            self._log_failure(conversation_id, target_message_id, failure.error)
            return replace(base, error=failure.error)

        outcome = replace(
            base,
            session_id=window.session_id if isinstance(window.session_id, str) else None,
            provider=metadata,
            context_message_count=len(window.preceding),
            omitted_message_count=window.omitted_message_count,
            profile_count=len(window.profiles),
            omitted_profile_count=window.omitted_profile_count,
        )
        request = AssistRequest(
            target_message=window.target,
            context_messages=window.preceding,
            context_profiles=window.profiles,
        )

        try:
            result, repaired = await self._invoke(request, target_message_id)
        except _AssistFailedError as failure:
            self._log_failure(conversation_id, target_message_id, failure.error)
            return replace(outcome, error=failure.error, repair_attempted=failure.repaired)

        try:
            stored = await asyncio.to_thread(
                self._repo.save_assist_result,
                conversation_id,
                result,
                metadata,
                None,
                self._clock_ms(),
            )
        except RecordNotFoundError:
            error = _error(AssistErrorCode.TARGET_NOT_FOUND)
        except (RepositoryError, ConversationValidationError):
            error = _error(AssistErrorCode.STORAGE_ERROR, retryable=True)
        else:
            logger.info(
                "Assist stored: conversation=%s target=%s provider=%s replies=%d questions=%d"
                " context=%d omitted=%d profiles=%d repaired=%s",
                conversation_id,
                target_message_id,
                metadata.provider_id,
                len(stored.suggested_replies),
                len(stored.clarifying_questions),
                len(window.preceding),
                window.omitted_message_count,
                len(window.profiles),
                repaired,
            )
            return replace(outcome, stored=stored, repair_attempted=repaired)
        self._log_failure(conversation_id, target_message_id, error)
        return replace(outcome, error=error, repair_attempted=repaired)

    # --- steps --------------------------------------------------------------------------------

    def _prepare(
        self, conversation_id: str, target_message_id: str
    ) -> Tuple[AssistContextWindow, ProviderMetadata]:
        """Blocking: read the conversation, select context, and release profile snapshots."""
        try:
            conversation = self._repo.get_conversation(conversation_id)
            if conversation is None:
                raise _AssistFailedError(_error(AssistErrorCode.TARGET_NOT_FOUND))
            messages = self._repo.list_messages(conversation_id)
            labels = dict(conversation.source_labels)
            # Select first so an invalid target never releases profile content.
            window = select_assist_context(messages, target_message_id, labels, (), self._budget)
            session = None
            if isinstance(window.session_id, str):
                session = self._repo.get_session(window.session_id)
                if session is not None:
                    snapshots = self._profiles.snapshots_for_outbound(
                        session.session_id, OUTBOUND_ASSIST
                    )
                    window = select_assist_context(
                        messages, target_message_id, labels, snapshots, self._budget
                    )
        except ContextSelectionError as exc:
            code = _SELECTION_CODES.get(exc.code, AssistErrorCode.INVALID_REQUEST)
            raise _AssistFailedError(_error(code)) from None
        except (RepositoryError, ConversationValidationError):
            raise _AssistFailedError(
                _error(AssistErrorCode.STORAGE_ERROR, retryable=True)
            ) from None
        return window, self._provider_metadata(session)

    def _provider_metadata(self, session: object) -> ProviderMetadata:
        provider_id = self._provider.provider_id
        model = self._model
        recorded = getattr(session, "providers", {}).get(_ASSISTANT_SERVICE)
        if recorded is not None and recorded.provider_id == provider_id and recorded.model:
            # The session snapshot is the authoritative model for the whole run.
            model = recorded.model
        try:
            return ProviderMetadata(provider_id=provider_id, model=model)
        except ConversationValidationError:
            return ProviderMetadata(provider_id=provider_id, model=None)

    async def _invoke(
        self, request: AssistRequest, target_message_id: str
    ) -> Tuple[AssistResult, bool]:
        try:
            return await self._attempt(request, target_message_id), False
        except _SchemaViolationError:
            logger.info("Assist reply failed the output contract; attempting one repair")
        repair_request = replace(request, system_prompt=REPAIR_INSTRUCTION)
        try:
            return await self._attempt(repair_request, target_message_id), True
        except _SchemaViolationError:
            raise _AssistFailedError(
                _error(AssistErrorCode.INVALID_RESPONSE), repaired=True
            ) from None
        except _AssistFailedError as failure:
            failure.repaired = True
            raise

    async def _attempt(self, request: AssistRequest, target_message_id: str) -> AssistResult:
        try:
            result = await self._provider.assist(request)
        except ProviderRuntimeError as err:
            if err.error.code is ProviderErrorCode.INVALID_RESPONSE:
                raise _SchemaViolationError() from None
            raise _AssistFailedError(
                _error(
                    AssistErrorCode.PROVIDER_ERROR,
                    retryable=bool(err.error.retryable),
                    provider_code=err.error.code.value,
                )
            ) from None
        except Exception as exc:
            logger.warning("Assist provider raised unexpectedly: %s", type(exc).__name__)
            raise _AssistFailedError(_error(AssistErrorCode.INTERNAL_ERROR)) from None
        try:
            return normalize_assist_result(result, target_message_id)
        except AssistResponseSchemaError:
            raise _SchemaViolationError() from None

    def _log_failure(
        self, conversation_id: str, target_message_id: str, error: AssistError
    ) -> None:
        logger.warning(
            "Assist failed: conversation=%s target=%s code=%s provider_code=%s",
            conversation_id,
            target_message_id,
            error.code.value,
            error.provider_code,
        )
