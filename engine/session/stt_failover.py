"""Reconnect / Local-fallback state machine for one cloud STT stream.

The controller is a pure decision object: callers report what happened (a cloud failure, a
reconnect result, Local starting) and act on the returned ``FailoverDecision``. It never
opens connections, sleeps, or touches an event loop; time comes from an injectable clock and
deadlines are enforced when the owner calls ``poll()``.

State machine (per stream, per session)::

    CLOUD_ACTIVE --failure (retryable, budget left)--> RECONNECTING   [RECONNECT + plan]
    CLOUD_ACTIVE --failure (non-retryable / no budget)--> FALLING_BACK [START_LOCAL_REPLAY]
    RECONNECTING --success within deadline--> CLOUD_ACTIVE             [RESUME_CLOUD]
    RECONNECTING --failure | late success | poll past deadline--> FALLING_BACK
    FALLING_BACK --local started--> LOCAL_ACTIVE                       [COMMIT_LOCAL]
    FALLING_BACK | LOCAL_ACTIVE --local failure--> FAILED              [REPORT_UNRECOVERABLE]
    any --end_session--> ENDED                                         [END]

``FALLING_BACK`` and ``LOCAL_ACTIVE`` are sticky: cloud signals are ignored and every
attempt to return to cloud is ``REJECTED``, so a stream can never flap or fail back within
a session. The reconnect budget (one attempt by default) is per session, so a second cloud
failure after a successful reconnect goes straight to Local.

Replay: every reconnect and fallback plan replays retained audio from the stream's last
final boundary minus ``replay_overlap_ms``. The new connection's time zero is the replay
start, which is therefore the ledger ``offset_ms``; the overlap window tells the ledger
where text reconciliation is allowed; and audio the ring buffer no longer holds is reported
as an explicit gap instead of an assumed-continuous transcript.

Thread ownership: one controller per stream, driven by the thread that owns that stream's
STT pipeline. An internal lock makes ``state`` reads and ``poll()`` safe from other threads.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

from engine.core.logging import get_logger
from engine.session.audio_ring_buffer import AudioRingBuffer, ReplaySlice

logger = get_logger("session.stt_failover")


class FailoverState(str, Enum):
    CLOUD_ACTIVE = "cloud_active"
    RECONNECTING = "reconnecting"
    FALLING_BACK = "falling_back"
    LOCAL_ACTIVE = "local_active"
    FAILED = "failed"
    ENDED = "ended"


class FailoverAction(str, Enum):
    NONE = "none"
    RECONNECT = "reconnect"
    RESUME_CLOUD = "resume_cloud"
    START_LOCAL_REPLAY = "start_local_replay"
    COMMIT_LOCAL = "commit_local"
    REPORT_UNRECOVERABLE = "report_unrecoverable"
    REJECTED = "rejected"
    END = "end"


class ReplayTarget(str, Enum):
    CLOUD_RECONNECT = "cloud_reconnect"
    LOCAL = "local"


@dataclass(frozen=True)
class FailoverPolicy:
    reconnect_timeout_s: float = 3.0
    max_reconnect_attempts: int = 1
    replay_overlap_ms: float = 2000.0


@dataclass(frozen=True)
class ReplayPlan:
    """Audio to resend to a new connection and how its timeline maps onto the session."""

    stream_id: str
    target: ReplayTarget
    final_boundary_ms: float
    audio: ReplaySlice

    @property
    def requested_start_ms(self) -> float:
        return self.audio.requested_start_ms

    @property
    def replay_start_ms(self) -> float:
        return self.audio.start_ms

    @property
    def replay_end_ms(self) -> float:
        """Live audio for the new connection continues from here."""
        return self.audio.end_ms

    @property
    def offset_ms(self) -> float:
        """Add to the new connection's provider-relative times to get session times."""
        return self.audio.start_ms

    @property
    def overlap_window(self) -> Optional[Tuple[float, float]]:
        """Replayed audio that committed finals already cover."""
        end = min(self.final_boundary_ms, self.audio.end_ms)
        if end <= self.audio.start_ms:
            return None
        return (self.audio.start_ms, end)

    @property
    def gap(self) -> Optional[Tuple[float, float]]:
        """Audio after the last final that the ring buffer no longer holds."""
        if self.audio.missing_ms <= 0.0:
            return None
        return (self.audio.requested_start_ms, self.audio.start_ms)


@dataclass(frozen=True)
class FailoverDecision:
    stream_id: str
    action: FailoverAction
    previous_state: FailoverState
    state: FailoverState
    reason: str = ""
    plan: Optional[ReplayPlan] = None
    reconnect_deadline: Optional[float] = None

    @property
    def changed(self) -> bool:
        return self.previous_state is not self.state


_STICKY_LOCAL = frozenset({FailoverState.FALLING_BACK, FailoverState.LOCAL_ACTIVE})


class STTFailoverController:
    """Bounded recovery for one stream: one reconnect, then sticky Local with replay."""

    def __init__(
        self,
        stream_id: str,
        ring_buffer: AudioRingBuffer,
        final_boundary_ms: Callable[[], float],
        policy: Optional[FailoverPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        cloud_primary: bool = True,
    ) -> None:
        if ring_buffer.stream_id != stream_id:
            raise ValueError("ring buffer belongs to a different stream")
        self.stream_id = stream_id
        self.policy = policy or FailoverPolicy()
        self._ring = ring_buffer
        self._final_boundary_ms = final_boundary_ms
        self._clock = clock
        self._lock = threading.Lock()
        self._state = FailoverState.CLOUD_ACTIVE if cloud_primary else FailoverState.LOCAL_ACTIVE
        self._reconnect_attempts = 0
        self._deadline: Optional[float] = None
        self._last_plan: Optional[ReplayPlan] = None

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def state(self) -> FailoverState:
        with self._lock:
            return self._state

    @property
    def reconnect_attempts_used(self) -> int:
        with self._lock:
            return self._reconnect_attempts

    @property
    def reconnect_deadline(self) -> Optional[float]:
        with self._lock:
            return self._deadline

    @property
    def last_plan(self) -> Optional[ReplayPlan]:
        with self._lock:
            return self._last_plan

    # ── Events ───────────────────────────────────────────────────────────────

    def report_cloud_failure(self, retryable: bool = True, reason: str = "") -> FailoverDecision:
        with self._lock:
            if self._state is FailoverState.CLOUD_ACTIVE:
                budget_left = self._reconnect_attempts < self.policy.max_reconnect_attempts
                if retryable and budget_left:
                    self._reconnect_attempts += 1
                    self._deadline = self._clock() + self.policy.reconnect_timeout_s
                    plan = self._plan(ReplayTarget.CLOUD_RECONNECT)
                    return self._move(
                        FailoverState.RECONNECTING,
                        FailoverAction.RECONNECT,
                        reason or "cloud_failure",
                        plan,
                    )
                why = "non_retryable_failure" if not retryable else "reconnect_budget_exhausted"
                return self._fall_back(reason or why)
            if self._state is FailoverState.RECONNECTING:
                return self._fall_back(reason or "reconnect_failed")
            return self._noop("cloud_failure_ignored")

    def report_reconnect_succeeded(self) -> FailoverDecision:
        with self._lock:
            if self._state is FailoverState.RECONNECTING:
                if self._deadline is not None and self._clock() <= self._deadline:
                    self._deadline = None
                    return self._move(
                        FailoverState.CLOUD_ACTIVE, FailoverAction.RESUME_CLOUD, "reconnected"
                    )
                return self._fall_back("reconnect_late")
            if self._state in _STICKY_LOCAL:
                return self._noop("failback_prohibited", FailoverAction.REJECTED)
            return self._noop("reconnect_ignored")

    def poll(self) -> FailoverDecision:
        """Enforce the reconnect deadline. Call periodically from the owning thread."""
        with self._lock:
            if (
                self._state is FailoverState.RECONNECTING
                and self._deadline is not None
                and self._clock() > self._deadline
            ):
                return self._fall_back("reconnect_timeout")
            return self._noop("")

    def report_local_started(self) -> FailoverDecision:
        with self._lock:
            if self._state is FailoverState.FALLING_BACK:
                return self._move(
                    FailoverState.LOCAL_ACTIVE, FailoverAction.COMMIT_LOCAL, "local_started"
                )
            return self._noop("local_start_ignored")

    def report_local_failure(self, reason: str = "") -> FailoverDecision:
        with self._lock:
            if self._state in _STICKY_LOCAL:
                return self._move(
                    FailoverState.FAILED,
                    FailoverAction.REPORT_UNRECOVERABLE,
                    reason or "local_failure",
                )
            return self._noop("local_failure_ignored")

    def request_cloud(self) -> FailoverDecision:
        """Any attempt to put a fallen-back stream back on cloud is refused."""
        with self._lock:
            if self._state in _STICKY_LOCAL:
                return self._noop("failback_prohibited", FailoverAction.REJECTED)
            return self._noop("cloud_request_ignored")

    def end_session(self) -> FailoverDecision:
        with self._lock:
            if self._state is FailoverState.ENDED:
                return self._noop("already_ended")
            self._deadline = None
            return self._move(FailoverState.ENDED, FailoverAction.END, "session_end")

    # ── Internals (lock held) ────────────────────────────────────────────────

    def _fall_back(self, reason: str) -> FailoverDecision:
        self._deadline = None
        plan = self._plan(ReplayTarget.LOCAL)
        return self._move(
            FailoverState.FALLING_BACK, FailoverAction.START_LOCAL_REPLAY, reason, plan
        )

    def _plan(self, target: ReplayTarget) -> ReplayPlan:
        boundary = max(0.0, self._final_boundary_ms())
        requested = max(0.0, boundary - self.policy.replay_overlap_ms)
        plan = ReplayPlan(
            stream_id=self.stream_id,
            target=target,
            final_boundary_ms=boundary,
            audio=self._ring.read(requested),
        )
        self._last_plan = plan
        return plan

    def _move(
        self,
        state: FailoverState,
        action: FailoverAction,
        reason: str,
        plan: Optional[ReplayPlan] = None,
    ) -> FailoverDecision:
        previous = self._state
        self._state = state
        decision = FailoverDecision(
            stream_id=self.stream_id,
            action=action,
            previous_state=previous,
            state=state,
            reason=reason,
            plan=plan,
            reconnect_deadline=self._deadline,
        )
        logger.info(
            "Stream %s STT failover %s -> %s (%s, action=%s, gap=%s)",
            self.stream_id,
            previous.value,
            state.value,
            reason,
            action.value,
            plan is not None and plan.gap is not None,
        )
        return decision

    def _noop(self, reason: str, action: FailoverAction = FailoverAction.NONE) -> FailoverDecision:
        return FailoverDecision(
            stream_id=self.stream_id,
            action=action,
            previous_state=self._state,
            state=self._state,
            reason=reason,
            reconnect_deadline=self._deadline,
        )
