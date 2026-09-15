"""Revision-aware hybrid Translation scheduling for one session.

Pure policy: no provider, no asyncio, no clock of its own. The session orchestrator feeds it
transcript changes and the current time, dispatches the jobs ``poll`` returns, and reports how
each one ended. Nothing here logs or stores more text than the requests it tracks.

Requests

- Final: every committed final with text creates one authoritative request at once, carrying
  the final's entry id and source revision. Finals are never debounced and are dispatched
  before any partial. A final (or an empty, retracting final) closes its stream's open partial.
- Partial: a stream's open partial is translated once its text has stayed unchanged for
  ``partial_debounce_s`` (about 600 ms). A newer revision with the same text keeps the timer; a
  text change restarts it. Text already requested for the open item is not requested again.

Revision rules

- A partial event whose revision is not newer than the open item's is ignored.
- Partial jobs are created only at dispatch time from the current state, so a superseded
  partial is never sent. An in-flight partial whose item was finalized or replaced, or whose
  text changed, is discarded as stale when it returns.
- A partial result is current only while its item is still the stream's open partial with the
  same text. Any final on the stream closes that item, so a late partial result can never
  overwrite a final translation.
- A final result is accepted once per entry id; a repeat or an older source revision is stale.

Bounds

- At most ``max_in_flight`` requests run at once. Partials may hold at most
  ``max_partial_in_flight`` of them, which is always fewer, so a final always finds a slot.
- At most ``max_pending_finals`` finals wait for a slot; a final beyond that is refused with
  ``translation_backlog``.
- Optional per-session caps: ``max_requests`` dispatched requests and ``max_tokens``
  provider-reported tokens. Once either is reached nothing more is dispatched and waiting or
  new finals are refused with ``translation_budget_exhausted``.
- ``usage()`` reports request, outcome, and token counts for the session. It is accounting,
  not billing.

Thread ownership: a single owner, the engine event loop. Not thread-safe.
"""

import dataclasses
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

DEFAULT_PARTIAL_DEBOUNCE_S = 0.6

REFUSED_BACKLOG = "translation_backlog"
REFUSED_BUDGET = "translation_budget_exhausted"

# Entry ids remembered for duplicate-final detection; older ones age out.
MAX_TRACKED_FINALS = 1024
# Clock arithmetic on floats must not push a due partial one poll later.
_EPSILON_S = 1e-9


class SchedulerPolicyError(ValueError):
    """A policy field is out of range. Carries the field name only."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"invalid translation scheduler policy field: {field_name}")
        self.field_name = field_name


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _usable_tokens(value: Any) -> int:
    return value if _is_int(value) and value > 0 else 0


@dataclass(frozen=True)
class SchedulerPolicy:
    """Debounce, concurrency, and budget bounds for one session."""

    partial_debounce_s: float = DEFAULT_PARTIAL_DEBOUNCE_S
    translate_partials: bool = True
    max_in_flight: int = 2
    max_partial_in_flight: int = 1
    max_pending_finals: int = 100
    max_requests: Optional[int] = None
    max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        debounce = self.partial_debounce_s
        if isinstance(debounce, bool) or not isinstance(debounce, (int, float)) or debounce < 0:
            raise SchedulerPolicyError("partial_debounce_s")
        if not isinstance(self.translate_partials, bool):
            raise SchedulerPolicyError("translate_partials")
        if not _is_int(self.max_in_flight) or self.max_in_flight < 1:
            raise SchedulerPolicyError("max_in_flight")
        partial_slots = self.max_partial_in_flight
        if not _is_int(partial_slots) or not 0 <= partial_slots < self.max_in_flight:
            raise SchedulerPolicyError("max_partial_in_flight")
        if not _is_int(self.max_pending_finals) or self.max_pending_finals < 1:
            raise SchedulerPolicyError("max_pending_finals")
        for name in ("max_requests", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise SchedulerPolicyError(name)


class JobKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass(frozen=True)
class TranslationJob:
    """One provider request: which transcript text, at which source revision."""

    job_id: int
    kind: JobKind
    stream_id: str
    item_key: str
    source_revision: int
    text: str
    # Finals only: the transcript entry the translation belongs to.
    entry_id: Optional[str] = None
    attempt: int = 1

    @property
    def is_final(self) -> bool:
        return self.kind is JobKind.FINAL


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    STALE = "stale"


@dataclass(frozen=True)
class SchedulerTick:
    """Jobs to dispatch now, and waiting finals refused with a reason code."""

    dispatch: Tuple[TranslationJob, ...] = ()
    refused: Tuple[Tuple[TranslationJob, str], ...] = ()


@dataclass(frozen=True)
class TranslationUsage:
    """Per-session request and token accounting."""

    partial_requests: int = 0
    final_requests: int = 0
    succeeded: int = 0
    failed: int = 0
    stale_discarded: int = 0
    finals_refused: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def requests(self) -> int:
        return self.partial_requests + self.final_requests

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class _OpenPartial:
    """The partial a stream is currently showing."""

    __slots__ = ("item_key", "requested_text", "revision", "stable_since", "text")

    def __init__(self, item_key: str, revision: int, text: str, now: float) -> None:
        self.item_key = item_key
        self.revision = revision
        self.text = text
        self.stable_since = now
        self.requested_text: Optional[str] = None


class TranslationScheduler:
    """Decides what to translate, when, and whether a returned result still applies."""

    def __init__(self, policy: Optional[SchedulerPolicy] = None) -> None:
        self._policy = policy or SchedulerPolicy()
        self._partials: Dict[str, _OpenPartial] = {}
        self._pending: Deque[TranslationJob] = deque()
        self._in_flight: Dict[int, TranslationJob] = {}
        self._final_revisions: OrderedDict[str, int] = OrderedDict()
        self._next_job_id = 1
        self._closed = False
        self._partial_requests = 0
        self._final_requests = 0
        self._succeeded = 0
        self._failed = 0
        self._stale = 0
        self._refused = 0
        self._input_tokens = 0
        self._output_tokens = 0

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def policy(self) -> SchedulerPolicy:
        return self._policy

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    @property
    def pending_final_count(self) -> int:
        return len(self._pending)

    @property
    def budget_exhausted(self) -> bool:
        policy = self._policy
        if policy.max_requests is not None and self._requests() >= policy.max_requests:
            return True
        tokens = self._input_tokens + self._output_tokens
        return policy.max_tokens is not None and tokens >= policy.max_tokens

    def usage(self) -> TranslationUsage:
        return TranslationUsage(
            partial_requests=self._partial_requests,
            final_requests=self._final_requests,
            succeeded=self._succeeded,
            failed=self._failed,
            stale_discarded=self._stale,
            finals_refused=self._refused,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    # ── Transcript changes ───────────────────────────────────────────────────

    def on_partial(
        self, stream_id: str, item_key: str, revision: int, text: str, now: float
    ) -> None:
        """Record what a stream's partial now shows."""
        if self._closed:
            return
        text = text.strip() if isinstance(text, str) else ""
        current = self._partials.get(stream_id)
        if current is not None and current.item_key == item_key and revision <= current.revision:
            return
        if not text:
            self._partials.pop(stream_id, None)
            return
        if current is None or current.item_key != item_key:
            self._partials[stream_id] = _OpenPartial(item_key, revision, text, now)
            return
        current.revision = revision
        if text != current.text:
            current.text = text
            current.stable_since = now

    def on_final(
        self,
        stream_id: str,
        item_key: str,
        revision: int,
        text: str,
        entry_id: Optional[str],
    ) -> Optional[str]:
        """Close the stream's partial and queue the authoritative request.

        Returns a refusal reason code, or None when the final was queued or has nothing to
        translate.
        """
        if self._closed:
            return None
        self._partials.pop(stream_id, None)
        text = text.strip() if isinstance(text, str) else ""
        if not text or not entry_id:
            return None
        if self.budget_exhausted:
            self._refused += 1
            return REFUSED_BUDGET
        if len(self._pending) >= self._policy.max_pending_finals:
            self._refused += 1
            return REFUSED_BACKLOG
        self._pending.append(
            TranslationJob(
                job_id=self._new_job_id(),
                kind=JobKind.FINAL,
                stream_id=stream_id,
                item_key=item_key,
                source_revision=revision,
                text=text,
                entry_id=entry_id,
            )
        )
        return None

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def poll(self, now: float) -> SchedulerTick:
        """Jobs that may start now. Finals first; then stable partials, oldest first."""
        if self._closed:
            return SchedulerTick()
        policy = self._policy
        dispatch: List[TranslationJob] = []
        while self._pending and len(self._in_flight) < policy.max_in_flight:
            if self.budget_exhausted:
                break
            job = self._pending.popleft()
            self._start(job)
            dispatch.append(job)
        refused: List[Tuple[TranslationJob, str]] = []
        if self.budget_exhausted:
            while self._pending:
                refused.append((self._pending.popleft(), REFUSED_BUDGET))
                self._refused += 1
        elif policy.translate_partials:
            dispatch.extend(self._due_partials(now))
        return SchedulerTick(tuple(dispatch), tuple(refused))

    def _due_partials(self, now: float) -> List[TranslationJob]:
        policy = self._policy
        due = sorted(
            (state.stable_since, stream_id)
            for stream_id, state in self._partials.items()
            if state.requested_text != state.text
            and now - state.stable_since + _EPSILON_S >= policy.partial_debounce_s
        )
        jobs: List[TranslationJob] = []
        for _, stream_id in due:
            if len(self._in_flight) >= policy.max_in_flight:
                break
            if self._partials_in_flight() >= policy.max_partial_in_flight or self.budget_exhausted:
                break
            state = self._partials[stream_id]
            state.requested_text = state.text
            job = TranslationJob(
                job_id=self._new_job_id(),
                kind=JobKind.PARTIAL,
                stream_id=stream_id,
                item_key=state.item_key,
                source_revision=state.revision,
                text=state.text,
            )
            self._start(job)
            jobs.append(job)
        return jobs

    # ── Outcomes ─────────────────────────────────────────────────────────────

    def complete(
        self,
        job: TranslationJob,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Verdict:
        """Release the job's slot, count its tokens, and decide whether its result applies."""
        started = self._in_flight.pop(job.job_id, None)
        # Tokens were spent whether or not the result is still wanted.
        self._input_tokens += _usable_tokens(input_tokens)
        self._output_tokens += _usable_tokens(output_tokens)
        if started is None or self._closed:
            self._stale += 1
            return Verdict.STALE
        if job.is_final:
            prior = self._final_revisions.get(job.entry_id)
            if prior is not None and prior >= job.source_revision:
                self._stale += 1
                return Verdict.STALE
            self._final_revisions[job.entry_id] = job.source_revision
            self._final_revisions.move_to_end(job.entry_id)
            while len(self._final_revisions) > MAX_TRACKED_FINALS:
                self._final_revisions.popitem(last=False)
            self._succeeded += 1
            return Verdict.ACCEPTED
        state = self._partials.get(job.stream_id)
        if state is None or state.item_key != job.item_key or state.text != job.text:
            self._stale += 1
            return Verdict.STALE
        self._succeeded += 1
        return Verdict.ACCEPTED

    def fail(self, job: TranslationJob, retry: bool = False) -> Optional[TranslationJob]:
        """Release a failed job's slot. With ``retry`` a final is requeued ahead of the rest.

        Returns the requeued job, if any. A failed partial is never retried: the next text
        change requests again.
        """
        started = self._in_flight.pop(job.job_id, None)
        if started is None or self._closed:
            return None
        self._failed += 1
        if not (retry and job.is_final):
            return None
        again = dataclasses.replace(job, job_id=self._new_job_id(), attempt=job.attempt + 1)
        self._pending.appendleft(again)
        return again

    def close(self) -> Tuple[TranslationJob, ...]:
        """End scheduling. Returns the finals that were still waiting; every later call is inert."""
        abandoned = tuple(self._pending)
        self._closed = True
        self._pending.clear()
        self._partials.clear()
        self._in_flight.clear()
        return abandoned

    # ── Internals ────────────────────────────────────────────────────────────

    def _new_job_id(self) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def _requests(self) -> int:
        return self._partial_requests + self._final_requests

    def _partials_in_flight(self) -> int:
        return sum(1 for job in self._in_flight.values() if not job.is_final)

    def _start(self, job: TranslationJob) -> None:
        self._in_flight[job.job_id] = job
        if job.is_final:
            self._final_requests += 1
        else:
            self._partial_requests += 1
