"""Deterministic Assist context selection and token budgeting.

Selection rules, in order:

1. The target must be a persisted, finalized, non-blank message of the conversation.
2. Candidates are finalized, non-blank messages of the same conversation whose ``sequence``
   is strictly lower than the target's. Partials, later ("future") messages, and anything
   from another conversation never qualify.
3. The newest ``max_preceding_messages`` candidates are kept, then trimmed from the oldest
   end until the transcript fits ``max_transcript_tokens``. The target is always kept.
4. Output is chronological (ascending ``sequence``).

Profiles are kept in attachment order until ``max_profile_tokens`` would be exceeded; a
profile is included whole or not at all so its meaning is never cut mid-sentence.

Token counts are a conservative character estimate. No tokenizer is loaded here.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from engine.conversation.models import TranscriptMessage
from engine.providers.types import ContextProfileSnapshot, ConversationMessage

DEFAULT_MAX_PRECEDING_MESSAGES = 20

# Roughly four characters per token for Latin text; CJK is denser, so the budget errs low.
_CHARS_PER_TOKEN = 4
# Per-message framing: ids, role, source, and ordering metadata in the prompt.
_MESSAGE_OVERHEAD_TOKENS = 16
_PROFILE_OVERHEAD_TOKENS = 8
_TRUNCATION_MARK = "…"


class ContextSelectionError(ValueError):
    """The request cannot form a valid Assist context. Carries a code, never message text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ContextBudget:
    max_preceding_messages: int = DEFAULT_MAX_PRECEDING_MESSAGES
    max_message_chars: int = 2000
    max_transcript_tokens: int = 6000
    max_profile_tokens: int = 12000

    def __post_init__(self) -> None:
        values = (
            self.max_preceding_messages,
            self.max_message_chars,
            self.max_transcript_tokens,
            self.max_profile_tokens,
        )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid_budget")
        if self.max_preceding_messages > DEFAULT_MAX_PRECEDING_MESSAGES:
            raise ValueError("invalid_budget")
        if self.max_message_chars < 1:
            raise ValueError("invalid_budget")


@dataclass(frozen=True)
class AssistContextWindow:
    target: ConversationMessage
    preceding: Tuple[ConversationMessage, ...]
    profiles: Tuple[ContextProfileSnapshot, ...]
    # None for imported messages, which carry no session and therefore no profiles.
    session_id: Optional[str] = None
    omitted_message_count: int = 0
    omitted_profile_count: int = 0
    truncated_message_count: int = 0


def estimate_tokens(text: str) -> int:
    return -(-len(text) // _CHARS_PER_TOKEN)


def _clip(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[: max(0, max_chars - 1)].rstrip() + _TRUNCATION_MARK, True


def _is_usable(message: TranscriptMessage) -> bool:
    return message.is_final and message.sequence is not None and bool(message.text.strip())


def select_assist_context(
    messages: Sequence[TranscriptMessage],
    target_message_id: str,
    labels: Dict[str, str],
    profiles: Sequence[ContextProfileSnapshot] = (),
    budget: ContextBudget = ContextBudget(),
) -> AssistContextWindow:
    """Build the Assist window from one conversation's stored messages.

    ``messages`` must all belong to the target's conversation; the service reads them with
    ``ConversationRepository.list_messages`` so that holds by construction.
    """
    target = next((m for m in messages if m.message_id == target_message_id), None)
    if target is None:
        raise ContextSelectionError("target_not_found")
    if not target.is_final or target.sequence is None:
        raise ContextSelectionError("target_not_final")
    if not target.text.strip():
        raise ContextSelectionError("target_empty")

    candidates = sorted(
        (m for m in messages if _is_usable(m) and m.sequence < target.sequence),
        key=lambda m: m.sequence,
    )
    newest = candidates[-budget.max_preceding_messages :] if budget.max_preceding_messages else []
    omitted = len(candidates) - len(newest)

    truncated = 0
    target_text, clipped = _clip(target.text, budget.max_message_chars)
    truncated += int(clipped)
    target_msg = replace(target.to_provider_message(labels), text=target_text)
    used_tokens = estimate_tokens(target_text) + _MESSAGE_OVERHEAD_TOKENS

    kept: List[ConversationMessage] = []
    for message in reversed(newest):
        text, clipped = _clip(message.text, budget.max_message_chars)
        cost = estimate_tokens(text) + _MESSAGE_OVERHEAD_TOKENS
        if used_tokens + cost > budget.max_transcript_tokens:
            break
        used_tokens += cost
        truncated += int(clipped)
        kept.append(replace(message.to_provider_message(labels), text=text))
    omitted += len(newest) - len(kept)
    kept.reverse()

    chosen_profiles: List[ContextProfileSnapshot] = []
    profile_tokens = 0
    for profile in profiles:
        cost = estimate_tokens(profile.content) + estimate_tokens(profile.name)
        cost += _PROFILE_OVERHEAD_TOKENS
        if profile_tokens + cost > budget.max_profile_tokens:
            break
        profile_tokens += cost
        chosen_profiles.append(profile)

    return AssistContextWindow(
        target=target_msg,
        preceding=tuple(kept),
        profiles=tuple(chosen_profiles),
        session_id=target.session_id,
        omitted_message_count=omitted,
        omitted_profile_count=len(profiles) - len(chosen_profiles),
        truncated_message_count=truncated,
    )
