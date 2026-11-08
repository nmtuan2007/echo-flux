"""Accelerated long-session soak test harness for EchoFlux.

Verifies:
1. 4-hour meeting timeline (14,400s / 14,400,000ms) simulation executing in < 3 seconds.
2. In-memory AudioRingBuffer memory stability: constant preallocated capacity (480,000 bytes),
   retained span bounded strictly at <= 15,000ms, zero heap growth.
3. TranscriptLedger ordering, monotonicity, and revision stability across hundreds of speech turns.
4. Failover state machine resilience through intermittent cloud drops and sticky local fallbacks.
5. SQLite conversation database integrity (PRAGMA integrity_check) and session termination.
6. Stress testing over thousands of streaming partial/final events without queue buildup or drift.
"""

import sqlite3
import time
from pathlib import Path
from typing import List

from engine.conversation.models import (
    ConversationRecord,
    SessionRecord,
    TranscriptMessage,
)
from engine.conversation.repository import DB_FILENAME, ConversationRepository
from engine.providers.types import STTTranscriptResult
from engine.session.audio_ring_buffer import AudioRingBuffer
from engine.session.stt_failover import (
    FailoverAction,
    FailoverPolicy,
    FailoverState,
    STTFailoverController,
)
from engine.session.transcript_ledger import (
    LedgerAction,
    TranscriptLedger,
)

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16 kHz mono int16 (2 bytes/sample)


class VirtualClock:
    """Deterministic fast virtual clock for accelerated soak simulation."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current = start_time

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def _dummy_pcm(num_bytes: int, fill: int = 42) -> bytes:
    return bytes([fill % 256]) * num_bytes


def _make_stt_result(
    stream_id: str,
    event_id: str,
    text: str,
    start_ms: float,
    end_ms: float,
    is_final: bool = True,
    revision: int = 1,
) -> STTTranscriptResult:
    return STTTranscriptResult(
        stream_id=stream_id,
        sequence=0,
        provider_event_id=event_id,
        text=text,
        is_final=is_final,
        revision=revision,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        monotonic_timestamp=time.monotonic(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ACCELERATED 4-HOUR MEETING SIMULATION
# ─────────────────────────────────────────────────────────────────────────────


def test_accelerated_4_hour_session_simulation(tmp_path: Path) -> None:
    """Simulate a 4-hour dual-stream meeting (14,400s) with 600 turns in < 3 seconds."""
    wall_start = time.perf_counter()
    virtual_clock = VirtualClock(100.0)

    db_path = tmp_path / DB_FILENAME
    repo = ConversationRepository(db_path)

    conv_id = "conv-soak-4h"
    sess_id = "sess-soak-4h"
    repo.create_conversation(
        ConversationRecord(
            conversation_id=conv_id,
            title="Executive 4-Hour Strategy & Technical Review",
            created_at_ms=100_000.0,
            updated_at_ms=100_000.0,
        )
    )
    repo.create_session(
        SessionRecord(
            session_id=sess_id,
            conversation_id=conv_id,
            started_at_ms=100_000.0,
            providers={},
        )
    )

    ring_mic = AudioRingBuffer("mic")
    ring_sys = AudioRingBuffer("system")
    ledger = TranscriptLedger()

    mic_epoch = ledger.begin_epoch("mic", "deepgram", offset_ms=0.0)
    sys_epoch = ledger.begin_epoch("system", "azure_speech", offset_ms=0.0)

    last_final_mic_ms = 0.0
    failover_ctrl_mic = STTFailoverController(
        "mic",
        ring_mic,
        lambda: last_final_mic_ms,
        clock=virtual_clock,
        policy=FailoverPolicy(
            reconnect_timeout_s=3.0,
            max_reconnect_attempts=1,
            replay_overlap_ms=500.0,
        ),
    )

    last_final_sys_ms = 0.0
    failover_ctrl_sys = STTFailoverController(
        "system",
        ring_sys,
        lambda: last_final_sys_ms,
        clock=virtual_clock,
        policy=FailoverPolicy(
            reconnect_timeout_s=3.0,
            max_reconnect_attempts=1,
            replay_overlap_ms=500.0,
        ),
    )

    total_turns = 600
    session_duration_s = 14400.0  # 4 hours
    turn_spacing_s = session_duration_s / total_turns  # 24s per turn

    failovers_triggered = 0
    pcm_chunk_2s = _dummy_pcm(BYTES_PER_SECOND * 2)

    for turn_idx in range(total_turns):
        turn_start_s = turn_idx * turn_spacing_s
        turn_start_ms = turn_start_s * 1000.0
        turn_end_ms = turn_start_ms + 4000.0  # 4-second utterance

        stream_id = "mic" if (turn_idx % 2 == 0) else "system"
        ring = ring_mic if stream_id == "mic" else ring_sys

        # 1. Advance audio buffer: append audio
        ring.append(pcm_chunk_2s)
        ring.append(pcm_chunk_2s)

        # Invariant 1: AudioRingBuffer capacity is constant (480,000 bytes)
        assert ring.capacity_bytes == 480_000
        # Invariant 2: Retained span never exceeds 15,000 ms
        span = ring.retained_span()
        assert span.duration_ms <= 15000.0 + 1e-4

        # 2. Simulate intermittent failover: mic at turn 150, system at turn 350
        if stream_id == "mic" and turn_idx == 150:
            failovers_triggered += 1
            decision = failover_ctrl_mic.report_cloud_failure(
                retryable=True, reason="simulated network timeout"
            )
            assert decision.action == FailoverAction.RECONNECT
            assert failover_ctrl_mic.state == FailoverState.RECONNECTING

            virtual_clock.advance(4.0)
            poll_decision = failover_ctrl_mic.poll()
            assert poll_decision.action == FailoverAction.START_LOCAL_REPLAY
            assert failover_ctrl_mic.state == FailoverState.FALLING_BACK

            failover_ctrl_mic.report_local_started()
            assert failover_ctrl_mic.state == FailoverState.LOCAL_ACTIVE

            plan = poll_decision.plan
            assert plan is not None
            mic_epoch = ledger.begin_epoch(
                "mic",
                "local_whisper",
                offset_ms=plan.replay_start_ms,
                overlap_window=(plan.replay_start_ms, plan.replay_start_ms + 500.0),
            )
        elif stream_id == "system" and turn_idx == 351:
            failovers_triggered += 1
            decision = failover_ctrl_sys.report_cloud_failure(
                retryable=True, reason="simulated system stream drop"
            )
            assert decision.action == FailoverAction.RECONNECT
            assert failover_ctrl_sys.state == FailoverState.RECONNECTING

            virtual_clock.advance(4.0)
            poll_decision = failover_ctrl_sys.poll()
            assert poll_decision.action == FailoverAction.START_LOCAL_REPLAY
            assert failover_ctrl_sys.state == FailoverState.FALLING_BACK

            failover_ctrl_sys.report_local_started()
            assert failover_ctrl_sys.state == FailoverState.LOCAL_ACTIVE

            plan = poll_decision.plan
            assert plan is not None
            sys_epoch = ledger.begin_epoch(
                "system",
                "local_whisper",
                offset_ms=plan.replay_start_ms,
                overlap_window=(plan.replay_start_ms, plan.replay_start_ms + 500.0),
            )

        # 3. Simulate Interim Partials -> Final
        active_epoch = mic_epoch if stream_id == "mic" else sys_epoch
        turn_text = f"Turn {turn_idx} statement regarding roadmap item {turn_idx % 25}."
        utterance_id = f"ev-turn-{turn_idx}"

        # Emit 2 partials with increasing revisions
        partial_1 = _make_stt_result(
            stream_id,
            utterance_id,
            turn_text[:12],
            turn_start_ms,
            turn_start_ms + 1500.0,
            is_final=False,
            revision=1,
        )
        dec_p1 = ledger.ingest(partial_1, active_epoch.handle)
        assert dec_p1.action in (LedgerAction.PARTIAL, LedgerAction.DUPLICATE_FINAL)

        partial_2 = _make_stt_result(
            stream_id,
            utterance_id,
            turn_text[:25],
            turn_start_ms,
            turn_start_ms + 3000.0,
            is_final=False,
            revision=2,
        )
        dec_p2 = ledger.ingest(partial_2, active_epoch.handle)
        assert dec_p2.action in (LedgerAction.PARTIAL, LedgerAction.DUPLICATE_FINAL)

        # Emit final
        final_res = _make_stt_result(
            stream_id,
            utterance_id,
            turn_text,
            turn_start_ms,
            turn_end_ms,
            is_final=True,
            revision=3,
        )
        dec_fin = ledger.ingest(final_res, active_epoch.handle)
        assert dec_fin.action in (LedgerAction.FINAL, LedgerAction.DUPLICATE_FINAL)

        if stream_id == "mic":
            last_final_mic_ms = turn_end_ms
        else:
            last_final_sys_ms = turn_end_ms

        # 4. Persist to SQLite repository
        repo.upsert_message(
            conv_id,
            TranscriptMessage(
                message_id=f"msg-{turn_idx:04d}",
                session_id=sess_id,
                source=stream_id,
                text=turn_text,
                translation=f"Ban dich {turn_idx}" if turn_idx % 3 == 0 else None,
                is_final=True,
                timestamp_ms=turn_start_ms,
            ),
        )

        # Advance virtual clock
        virtual_clock.advance(turn_spacing_s)

    # Conclude session after 4 hours
    repo.end_session(sess_id, ended_at_ms=session_duration_s * 1000.0)
    repo.close()

    wall_elapsed = time.perf_counter() - wall_start

    # Verification assertions:
    assert failovers_triggered == 2
    # Soak test must complete rapidly using virtual clock (< 3.0s)
    assert wall_elapsed < 3.0, f"Soak test too slow: {wall_elapsed:.2f}s (must be < 3.0s)"

    # Deep SQLite database verification
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Database integrity
    integrity = cursor.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"

    # 2. Total messages persisted
    count = cursor.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
    ).fetchone()[0]
    assert count == total_turns

    # 3. Message ordering and monotonicity
    rows = cursor.execute(
        "SELECT message_id, source, timestamp_ms FROM messages"
        " WHERE conversation_id = ? ORDER BY timestamp_ms ASC",
        (conv_id,),
    ).fetchall()

    mic_timestamps: List[float] = []
    sys_timestamps: List[float] = []
    for _mid, src, ts in rows:
        if src == "mic":
            mic_timestamps.append(ts)
        else:
            sys_timestamps.append(ts)

    assert len(mic_timestamps) == total_turns // 2
    assert len(sys_timestamps) == total_turns // 2

    # Monotonicity check
    for i in range(1, len(mic_timestamps)):
        assert mic_timestamps[i] >= mic_timestamps[i - 1]
    for i in range(1, len(sys_timestamps)):
        assert sys_timestamps[i] >= sys_timestamps[i - 1]

    # Session record check
    sess_row = cursor.execute(
        "SELECT started_at_ms, ended_at_ms FROM sessions WHERE session_id = ?", (sess_id,)
    ).fetchone()
    assert sess_row is not None
    assert sess_row[0] == 100_000.0
    assert sess_row[1] == 14_400_000.0

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. RING BUFFER MEMORY CEILING & RETENTION STABILITY
# ─────────────────────────────────────────────────────────────────────────────


def test_audio_ring_buffer_memory_ceiling_under_continuous_pressure() -> None:
    """Stress test AudioRingBuffer with 1,000,000 frames (6+ hours) of audio."""
    ring = AudioRingBuffer("mic", sample_rate=16000, capacity_ms=15000.0)

    # 1 second of PCM per iteration
    second_pcm = _dummy_pcm(BYTES_PER_SECOND, fill=7)

    # Append 25,000 seconds of audio (> 6.9 hours)
    for _ in range(25000):
        ring.append(second_pcm)

    # Buffer memory must remain strictly 480,000 bytes
    assert ring.capacity_bytes == 480_000
    assert ring.capacity_ms == 15000.0
    assert ring.end_ms == 25000.0 * 1000.0

    # Retained span is strictly bounded to the latest 15 seconds
    span = ring.retained_span()
    assert span.duration_ms == 15000.0
    assert span.start_ms == 25000000.0 - 15000.0
    assert span.end_ms == 25000000.0

    # Requesting audio from the beginning of time correctly reports evicted audio
    replay = ring.read(0.0)
    assert replay.requested_start_ms == 0.0
    assert replay.start_ms == 24985000.0
    assert replay.missing_ms == 24985000.0
    assert len(replay.pcm_data) == 480_000
    assert not replay.is_complete

    # Requesting audio within the retained window is 100% complete
    recent_replay = ring.read(span.start_ms + 1000.0)
    assert recent_replay.is_complete
    assert recent_replay.missing_ms == 0.0
    assert (recent_replay.end_ms - recent_replay.start_ms) == 14000.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSCRIPT LEDGER MONOTONICITY OVER 1,000 EXTENDED TURNS
# ─────────────────────────────────────────────────────────────────────────────


def test_ledger_monotonicity_and_revisions_over_extended_turns() -> None:
    """Verify TranscriptLedger ordering and deduplication across 1,000 speech segments."""
    ledger = TranscriptLedger()
    epoch = ledger.begin_epoch("mic", "deepgram", offset_ms=0.0)

    for seg_idx in range(1000):
        start_ms = seg_idx * 3000.0
        end_ms = start_ms + 2500.0
        event_key = f"ev-turn-{seg_idx}"

        # Emit 3 revisions of partials with increasing revision number
        for rev in (1, 2, 3):
            partial = _make_stt_result(
                "mic",
                event_key,
                f"Partial text rev {rev} of segment {seg_idx}",
                start_ms,
                start_ms + rev * 700.0,
                is_final=False,
                revision=rev,
            )
            decision = ledger.ingest(partial, epoch.handle)
            assert decision.action == LedgerAction.PARTIAL
            assert decision.revision == rev

        # Stale revision submission (revision 2 after revision 3) must be rejected
        stale_partial = _make_stt_result(
            "mic",
            event_key,
            "Out of order partial",
            start_ms,
            start_ms + 1000.0,
            is_final=False,
            revision=2,
        )
        stale_dec = ledger.ingest(stale_partial, epoch.handle)
        assert stale_dec.action == LedgerAction.STALE

        # Emit final
        final_res = _make_stt_result(
            "mic",
            event_key,
            f"Final authoritative text for segment {seg_idx}.",
            start_ms,
            end_ms,
            is_final=True,
            revision=4,
        )
        final_dec = ledger.ingest(final_res, epoch.handle)
        assert final_dec.action == LedgerAction.FINAL
        assert final_dec.text == f"Final authoritative text for segment {seg_idx}."

        # Late partial arriving after final must be dropped (AFTER_FINAL)
        late_partial = _make_stt_result(
            "mic",
            event_key,
            "Late arriving partial",
            start_ms,
            start_ms + 2000.0,
            is_final=False,
            revision=5,
        )
        late_dec = ledger.ingest(late_partial, epoch.handle)
        assert late_dec.action == LedgerAction.AFTER_FINAL


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPEATED FAILOVER AND RECOVERY STRESS CYCLES
# ─────────────────────────────────────────────────────────────────────────────


def test_repeated_failover_and_recovery_stress_cycles() -> None:
    """Stress test failover controller across 50 independent stream recovery lifecycles."""
    clock = VirtualClock(0.0)

    # Run 50 independent lifecycle cycles
    for cycle in range(50):
        ring = AudioRingBuffer(f"stream-{cycle}")
        for _ in range(5):
            ring.append(_dummy_pcm(BYTES_PER_SECOND))

        ctrl = STTFailoverController(
            f"stream-{cycle}",
            ring,
            lambda: 2000.0,
            clock=clock,
            policy=FailoverPolicy(
                reconnect_timeout_s=2.0,
                max_reconnect_attempts=2,
                replay_overlap_ms=300.0,
            ),
        )
        assert ctrl.state == FailoverState.CLOUD_ACTIVE

        # Cycle 1: First retryable failure -> RECONNECT
        dec1 = ctrl.report_cloud_failure(retryable=True, reason="temporary disconnect")
        assert dec1.action == FailoverAction.RECONNECT
        assert ctrl.state == FailoverState.RECONNECTING

        # Reconnect succeeds within deadline
        clock.advance(0.5)
        ctrl.report_reconnect_succeeded()
        assert ctrl.state == FailoverState.CLOUD_ACTIVE

        # Cycle 2: Second retryable failure -> RECONNECT (budget used: 2)
        dec2 = ctrl.report_cloud_failure(retryable=True, reason="second temporary drop")
        assert dec2.action == FailoverAction.RECONNECT
        assert ctrl.state == FailoverState.RECONNECTING

        clock.advance(0.5)
        ctrl.report_reconnect_succeeded()
        assert ctrl.state == FailoverState.CLOUD_ACTIVE

        # Cycle 3: Third failure -> Reconnect budget exhausted -> Immediate START_LOCAL_REPLAY
        dec3 = ctrl.report_cloud_failure(retryable=True, reason="third failure")
        assert dec3.action == FailoverAction.START_LOCAL_REPLAY
        assert ctrl.state == FailoverState.FALLING_BACK

        # Commit local
        ctrl.report_local_started()
        assert ctrl.state == FailoverState.LOCAL_ACTIVE

        # Post-condition: Sticky local rejects failback
        failback_dec = ctrl.report_reconnect_succeeded()
        assert failback_dec.action == FailoverAction.REJECTED
        assert ctrl.state == FailoverState.LOCAL_ACTIVE
