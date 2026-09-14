"""Audio ring buffer bounds and the reconnect / sticky-Local failover state machine."""

from typing import List, Optional

import pytest

from engine.providers.types import STTTranscriptResult
from engine.session.audio_ring_buffer import AudioRingBuffer
from engine.session.stt_failover import (
    FailoverAction,
    FailoverPolicy,
    FailoverState,
    ReplayTarget,
    STTFailoverController,
)
from engine.session.transcript_ledger import ContinuityReason, LedgerAction, TranscriptLedger

BYTES_PER_SECOND = 16000 * 2  # 16 kHz mono int16


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _second(value: int) -> bytes:
    return bytes([value % 256]) * BYTES_PER_SECOND


def _ring(stream_id: str = "mic", seconds: int = 10) -> AudioRingBuffer:
    ring = AudioRingBuffer(stream_id)
    for index in range(seconds):
        ring.append(_second(index))
    return ring


def _controller(
    boundary_ms: float = 6000.0,
    seconds: int = 10,
    stream_id: str = "mic",
    clock: Optional[FakeClock] = None,
    **kwargs,
):
    clock = clock or FakeClock()
    ring = _ring(stream_id, seconds)
    ctrl = STTFailoverController(stream_id, ring, lambda: boundary_ms, clock=clock, **kwargs)
    return ctrl, clock


def _result(event: str, text: str, start: float, end: float, final: bool = True):
    return STTTranscriptResult(
        stream_id="mic",
        sequence=0,
        provider_event_id=event,
        text=text,
        is_final=final,
        revision=1,
        start_time_ms=start,
        end_time_ms=end,
        monotonic_timestamp=0.0,
    )


# ── Ring buffer ──────────────────────────────────────────────────────────────


def test_ring_capacity_matches_rate_channels_and_width():
    assert AudioRingBuffer("mic").capacity_bytes == 15 * 16000 * 1 * 2
    assert AudioRingBuffer("mic").capacity_ms == 15000.0
    stereo = AudioRingBuffer("system", sample_rate=48000, channels=2)
    assert stereo.capacity_bytes == 15 * 48000 * 2 * 2


def test_ring_memory_stays_bounded_and_keeps_newest_audio():
    ring = AudioRingBuffer("mic")
    capacity = ring.capacity_bytes

    for index in range(20):
        span = ring.append(_second(index))
        assert span.duration_ms == 1000.0

    assert ring.capacity_bytes == capacity
    retained = ring.retained_span()
    assert (retained.start_ms, retained.end_ms) == (5000.0, 20000.0)

    audio = ring.read(0.0)
    assert len(audio.pcm_data) == capacity
    assert audio.missing_ms == 5000.0
    assert not audio.is_complete
    for offset, second in enumerate(range(5, 20)):
        chunk = audio.pcm_data[offset * BYTES_PER_SECOND : (offset + 1) * BYTES_PER_SECOND]
        assert chunk == _second(second)


def test_ring_read_inside_window_is_complete():
    ring = _ring(seconds=20)

    audio = ring.read(12000.0)

    assert audio.is_complete
    assert (audio.start_ms, audio.end_ms) == (12000.0, 20000.0)
    assert audio.pcm_data[:BYTES_PER_SECOND] == _second(12)
    assert len(audio.pcm_data) == 8 * BYTES_PER_SECOND


def test_ring_single_oversized_append_keeps_tail():
    ring = AudioRingBuffer("mic")

    ring.append(b"".join(_second(index) for index in range(18)))

    audio = ring.read(0.0)
    assert (audio.start_ms, audio.end_ms) == (3000.0, 18000.0)
    assert audio.pcm_data[:BYTES_PER_SECOND] == _second(3)
    assert audio.pcm_data[-BYTES_PER_SECOND:] == _second(17)


def test_ring_rejects_partial_samples():
    with pytest.raises(ValueError):
        AudioRingBuffer("mic").append(b"\x00\x01\x02")


def test_ring_clear_drops_audio_but_keeps_timeline():
    ring = _ring(seconds=4)

    ring.clear()
    assert ring.retained_span().duration_ms == 0.0
    span = ring.append(_second(9))

    assert (span.start_ms, span.end_ms) == (4000.0, 5000.0)
    assert ring.read(0.0).start_ms == 4000.0


def test_replay_slice_splits_into_audio_frames():
    ring = _ring(seconds=2)

    frames = ring.read(1000.0).to_frames(frame_ms=100.0, first_sequence=7)

    assert len(frames) == 10
    assert [frame.sequence for frame in frames[:2]] == [7, 8]
    assert all(len(frame.pcm_data) == 3200 for frame in frames)
    assert all(frame.duration_ms == 100.0 and frame.stream_id == "mic" for frame in frames)


# ── Reconnect ────────────────────────────────────────────────────────────────


def test_one_reconnect_within_deadline_resumes_cloud():
    ctrl, clock = _controller()

    failure = ctrl.report_cloud_failure(retryable=True)

    assert failure.action is FailoverAction.RECONNECT
    assert failure.state is FailoverState.RECONNECTING
    assert failure.reconnect_deadline == pytest.approx(103.0)
    plan = failure.plan
    assert plan.target is ReplayTarget.CLOUD_RECONNECT
    assert (plan.replay_start_ms, plan.offset_ms, plan.replay_end_ms) == (4000.0, 4000.0, 10000.0)
    assert plan.overlap_window == (4000.0, 6000.0)
    assert plan.gap is None

    clock.advance(2.9)
    resumed = ctrl.report_reconnect_succeeded()

    assert resumed.action is FailoverAction.RESUME_CLOUD
    assert ctrl.state is FailoverState.CLOUD_ACTIVE
    assert ctrl.reconnect_attempts_used == 1


def test_poll_falls_back_only_after_deadline():
    ctrl, clock = _controller()
    ctrl.report_cloud_failure()

    clock.advance(3.0)
    assert ctrl.poll().action is FailoverAction.NONE

    clock.advance(0.01)
    decision = ctrl.poll()

    assert decision.action is FailoverAction.START_LOCAL_REPLAY
    assert decision.state is FailoverState.FALLING_BACK
    assert decision.reason == "reconnect_timeout"


def test_late_reconnect_success_is_not_accepted():
    ctrl, clock = _controller()
    ctrl.report_cloud_failure()
    clock.advance(3.5)

    decision = ctrl.report_reconnect_succeeded()

    assert decision.action is FailoverAction.START_LOCAL_REPLAY
    assert decision.reason == "reconnect_late"
    assert ctrl.state is FailoverState.FALLING_BACK


def test_failed_reconnect_falls_back_with_local_replay_plan():
    ctrl, _ = _controller()
    ctrl.report_cloud_failure()

    decision = ctrl.report_cloud_failure()

    plan = decision.plan
    assert decision.action is FailoverAction.START_LOCAL_REPLAY
    assert plan.target is ReplayTarget.LOCAL
    assert plan.final_boundary_ms == 6000.0
    assert plan.requested_start_ms == plan.replay_start_ms == 4000.0
    assert plan.overlap_window == (4000.0, 6000.0)
    assert len(plan.audio.pcm_data) == 6 * BYTES_PER_SECOND
    assert plan.audio.pcm_data[:BYTES_PER_SECOND] == _second(4)
    assert ctrl.last_plan == plan


def test_non_retryable_failure_skips_reconnect():
    ctrl, _ = _controller()

    decision = ctrl.report_cloud_failure(retryable=False)

    assert decision.action is FailoverAction.START_LOCAL_REPLAY
    assert decision.reason == "non_retryable_failure"
    assert ctrl.reconnect_attempts_used == 0


def test_reconnect_budget_is_per_session():
    ctrl, _ = _controller()
    ctrl.report_cloud_failure()
    ctrl.report_reconnect_succeeded()

    decision = ctrl.report_cloud_failure(retryable=True)

    assert decision.action is FailoverAction.START_LOCAL_REPLAY
    assert decision.reason == "reconnect_budget_exhausted"


def test_policy_overrides_timeout_and_overlap():
    policy = FailoverPolicy(reconnect_timeout_s=1.0, replay_overlap_ms=500.0)
    ctrl, clock = _controller(policy=policy)

    plan = ctrl.report_cloud_failure().plan
    clock.advance(1.5)

    assert plan.replay_start_ms == 5500.0
    assert ctrl.poll().action is FailoverAction.START_LOCAL_REPLAY


# ── Sticky Local, flapping, and terminal states ──────────────────────────────


def test_local_is_sticky_and_failback_is_rejected():
    ctrl, clock = _controller()
    ctrl.report_cloud_failure(retryable=False)

    assert ctrl.request_cloud().action is FailoverAction.REJECTED
    assert ctrl.report_local_started().action is FailoverAction.COMMIT_LOCAL

    assert ctrl.report_reconnect_succeeded().action is FailoverAction.REJECTED
    assert ctrl.request_cloud().action is FailoverAction.REJECTED
    assert ctrl.report_cloud_failure().action is FailoverAction.NONE
    clock.advance(60)
    assert ctrl.poll().action is FailoverAction.NONE
    assert ctrl.state is FailoverState.LOCAL_ACTIVE


def test_flapping_signals_never_return_stream_to_cloud():
    ctrl, clock = _controller()
    ctrl.report_cloud_failure()
    clock.advance(4)
    assert ctrl.poll().action is FailoverAction.START_LOCAL_REPLAY

    signals = [
        ctrl.report_reconnect_succeeded,
        ctrl.report_cloud_failure,
        ctrl.request_cloud,
        ctrl.poll,
        ctrl.report_local_started,
    ]
    actions: List[FailoverAction] = []
    for _ in range(50):
        for signal in signals:
            clock.advance(0.5)
            decision = signal()
            actions.append(decision.action)
            assert decision.state in (FailoverState.FALLING_BACK, FailoverState.LOCAL_ACTIVE)

    assert FailoverAction.RESUME_CLOUD not in actions
    assert FailoverAction.RECONNECT not in actions
    assert actions.count(FailoverAction.COMMIT_LOCAL) == 1
    assert ctrl.reconnect_attempts_used == 1


def test_local_failure_is_terminal_and_reported():
    ctrl, _ = _controller()
    ctrl.report_cloud_failure(retryable=False)
    ctrl.report_local_started()

    decision = ctrl.report_local_failure()

    assert decision.action is FailoverAction.REPORT_UNRECOVERABLE
    assert ctrl.state is FailoverState.FAILED
    assert ctrl.report_cloud_failure().action is FailoverAction.NONE
    assert ctrl.report_reconnect_succeeded().action is FailoverAction.NONE
    assert ctrl.state is FailoverState.FAILED


def test_session_end_is_final():
    ctrl, _ = _controller()
    ctrl.report_cloud_failure()

    assert ctrl.end_session().action is FailoverAction.END
    assert ctrl.reconnect_deadline is None
    for signal in (ctrl.report_cloud_failure, ctrl.report_reconnect_succeeded, ctrl.poll):
        assert signal().action is FailoverAction.NONE
    assert ctrl.end_session().action is FailoverAction.NONE
    assert ctrl.state is FailoverState.ENDED


def test_local_primary_session_ignores_cloud_signals():
    ctrl, _ = _controller(cloud_primary=False)

    assert ctrl.state is FailoverState.LOCAL_ACTIVE
    assert ctrl.report_cloud_failure().action is FailoverAction.NONE


def test_controller_rejects_ring_buffer_of_other_stream():
    with pytest.raises(ValueError):
        STTFailoverController("mic", AudioRingBuffer("system"), lambda: 0.0)


def test_mic_and_system_fail_over_independently():
    clock = FakeClock()
    mic, _ = _controller(stream_id="mic", clock=clock)
    system, _ = _controller(stream_id="system", clock=clock)

    mic.report_cloud_failure()
    clock.advance(4)

    assert mic.poll().action is FailoverAction.START_LOCAL_REPLAY
    assert system.poll().action is FailoverAction.NONE
    assert system.state is FailoverState.CLOUD_ACTIVE
    assert system.reconnect_attempts_used == 0


# ── Gaps ─────────────────────────────────────────────────────────────────────


def test_evicted_replay_audio_produces_explicit_gap():
    ledger = TranscriptLedger()
    ledger.begin_epoch("mic", "cloud")
    ctrl, _ = _controller(boundary_ms=4000.0, seconds=20)

    plan = ctrl.report_cloud_failure(retryable=False).plan

    assert plan.requested_start_ms == 2000.0
    assert plan.replay_start_ms == 5000.0
    assert plan.gap == (2000.0, 5000.0)
    assert plan.overlap_window is None

    start = ledger.begin_epoch("mic", "local", plan.offset_ms, plan.overlap_window, plan.gap)
    assert start.gap_warning.reason is ContinuityReason.AUDIO_GAP
    assert (start.gap_warning.start_ms, start.gap_warning.end_ms) == (2000.0, 5000.0)
    assert ledger.stream("mic").warning_count == 1


def test_no_final_yet_replays_from_session_start():
    ctrl, _ = _controller(boundary_ms=0.0, seconds=10)

    plan = ctrl.report_cloud_failure(retryable=False).plan

    assert plan.replay_start_ms == 0.0
    assert plan.gap is None
    assert plan.overlap_window is None


# ── End to end: ring + controller + ledger ───────────────────────────────────


def test_failover_overlap_leaves_no_duplicate_final():
    clock = FakeClock()
    ledger = TranscriptLedger()
    ring = _ring(seconds=12)
    ctrl = STTFailoverController(
        "mic", ring, lambda: ledger.last_final_boundary_ms("mic"), clock=clock
    )
    cloud = ledger.begin_epoch("mic", "cloud").handle
    committed: List[str] = []

    for result in (
        _result("c1", "Good morning everyone.", 0, 2000),
        _result("c2", "We should schedule the meeting", 2500, 6000),
        _result("c3", "for tomorrow", 6200, 8000, final=False),
    ):
        decision = ledger.ingest(result, cloud)
        if decision.action is LedgerAction.FINAL:
            committed.append(decision.text)

    assert ctrl.report_cloud_failure().action is FailoverAction.RECONNECT
    clock.advance(3.5)
    fallback = ctrl.poll()
    plan = fallback.plan
    assert fallback.action is FailoverAction.START_LOCAL_REPLAY
    assert (plan.offset_ms, plan.overlap_window, plan.gap) == (4000.0, (4000.0, 6000.0), None)

    local = ledger.begin_epoch("mic", "local", plan.offset_ms, plan.overlap_window, plan.gap)
    assert local.abandoned_entry_keys == ("mic:1:c3",)
    assert ctrl.report_local_started().action is FailoverAction.COMMIT_LOCAL

    outcomes = []
    for result in (
        _result("l1", "schedule the meeting", 0, 2200),
        _result("l2", "the meeting for tomorrow at noon", 1600, 4500),
        _result("l3", "see you then", 5000, 6500),
    ):
        decision = ledger.ingest(result, local.handle)
        outcomes.append(decision.action)
        if decision.action is LedgerAction.FINAL:
            committed.append(decision.text)

    assert outcomes == [LedgerAction.DUPLICATE_FINAL, LedgerAction.FINAL, LedgerAction.FINAL]
    assert committed == [
        "Good morning everyone.",
        "We should schedule the meeting",
        "for tomorrow at noon",
        "see you then",
    ]
    assert " ".join(committed).lower().count("the meeting") == 1

    late_cloud = ledger.ingest(_result("c3", "for tomorrow", 6200, 8000), cloud)
    assert late_cloud.action is LedgerAction.RETIRED_EPOCH
    assert ctrl.report_reconnect_succeeded().action is FailoverAction.REJECTED
    assert ledger.stream("mic").warning_count == 0
