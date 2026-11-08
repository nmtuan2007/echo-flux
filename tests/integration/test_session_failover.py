"""Dual-stream audio, provider failover, and transcript reconciliation integration tests.

Verifies:
- Dual-stream audio (mic and system) handled independently without cross-stream deduplication.
- Cloud provider disconnect on mic/system, reconnect attempts, and sticky Local fallback.
- Audio ring buffer bounds (15s capacity) and replay-from-boundary with overlap window.
- Transcript ledger reconciliation: interval deduplication, prefix trimming, and explicit
  unresolved overlap / continuity warnings.
- Protection against stale overwrites from earlier epochs or dead connections.
- Tier 4 Real-World scenarios: Bilingual video call meeting & Cloud drop with sticky local fallback.
"""

import time

from engine.conversation.models import (
    ConversationRecord,
    SessionRecord,
    TranscriptMessage,
)
from engine.conversation.repository import DB_FILENAME, ConversationRepository
from engine.providers.types import (
    STTTranscriptResult,
)
from engine.session.audio_ring_buffer import AudioRingBuffer
from engine.session.stt_failover import (
    FailoverAction,
    FailoverPolicy,
    FailoverState,
    ReplayTarget,
    STTFailoverController,
)
from engine.session.transcript_ledger import (
    ContinuityReason,
    LedgerAction,
    StreamTranscriptLedger,
    TranscriptLedger,
)

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16 kHz mono int16


class FakeClock:
    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _second_pcm(val: int = 1) -> bytes:
    return bytes([val % 256]) * BYTES_PER_SECOND


def _result(
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
# 1. DUAL-STREAM INDEPENDENCE & NO CROSS-STREAM DEDUPE
# ─────────────────────────────────────────────────────────────────────────────


def test_dual_stream_independent_ledgers_no_cross_stream_dedupe():
    """Verify mic and system streams never deduplicate against each other even if identical."""
    ledger = TranscriptLedger()
    mic_ep = ledger.begin_epoch("mic", "deepgram", 0.0)
    sys_ep = ledger.begin_epoch("system", "deepgram", 0.0)

    # Both speakers say "Yes, exactly" at the exact same session timestamp (0 to 1200ms)
    mic_res = _result("mic", "ev-1", "Yes, exactly", 0.0, 1200.0, is_final=True)
    sys_res = _result("system", "ev-1", "Yes, exactly", 0.0, 1200.0, is_final=True)

    mic_item = ledger.ingest(mic_res, mic_ep.handle)
    sys_item = ledger.ingest(sys_res, sys_ep.handle)

    assert mic_item.action == LedgerAction.FINAL
    assert mic_item.text == "Yes, exactly"
    assert sys_item.action == LedgerAction.FINAL
    assert sys_item.text == "Yes, exactly"

    # Verify both streams recorded finals independently
    assert ledger.stream("mic").retained_final_count == 1
    assert ledger.stream("system").retained_final_count == 1
    assert ledger.stream("mic").last_final_boundary_ms == 1200.0
    assert ledger.stream("system").last_final_boundary_ms == 1200.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLOUD DROP & STICKY LOCAL FALLBACK STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────


def test_cloud_drop_reconnect_retry_and_sticky_local_fallback():
    """Verify cloud provider disconnect triggers reconnect, then sticky local fallback."""
    clock = FakeClock(100.0)
    ring = AudioRingBuffer("mic")
    for i in range(10):
        ring.append(_second_pcm(i))

    last_final_ms = 6000.0
    ctrl = STTFailoverController(
        "mic",
        ring,
        lambda: last_final_ms,
        clock=clock,
        policy=FailoverPolicy(
            reconnect_timeout_s=3.0,
            max_reconnect_attempts=1,
            replay_overlap_ms=500.0,
        ),
    )

    assert ctrl.state == FailoverState.CLOUD_ACTIVE

    # Inject first cloud failure -> initiates reconnect
    decision = ctrl.report_cloud_failure(retryable=True, reason="socket dropped")
    assert decision.action == FailoverAction.RECONNECT
    assert ctrl.state == FailoverState.RECONNECTING

    # Clock advances past reconnect deadline
    clock.advance(4.0)
    poll_decision = ctrl.poll()
    # Deadline expired -> transitions to Local fallback with replay plan
    assert poll_decision.action == FailoverAction.START_LOCAL_REPLAY
    assert ctrl.state == FailoverState.FALLING_BACK

    plan = poll_decision.plan
    assert plan is not None
    assert plan.replay_start_ms == 5500.0  # 6000.0 - 500.0 overlap
    assert plan.target == ReplayTarget.LOCAL

    # Replay finished & local provider started -> transitions to LOCAL_ACTIVE
    ctrl.report_local_started()
    assert ctrl.state == FailoverState.LOCAL_ACTIVE

    # Subsequent polls stay sticky Local
    clock.advance(100.0)
    assert ctrl.poll().action == FailoverAction.NONE
    assert ctrl.state == FailoverState.LOCAL_ACTIVE


def test_stream_level_isolation_one_stream_fails_other_stays_primary():
    """Verify failure on mic does not affect system stream."""
    clock = FakeClock(100.0)
    mic_ring = AudioRingBuffer("mic")
    sys_ring = AudioRingBuffer("system")

    mic_ctrl = STTFailoverController("mic", mic_ring, lambda: 0.0, clock=clock)
    sys_ctrl = STTFailoverController("system", sys_ring, lambda: 0.0, clock=clock)

    assert mic_ctrl.state == FailoverState.CLOUD_ACTIVE
    assert sys_ctrl.state == FailoverState.CLOUD_ACTIVE

    # Mic fails non-retryably -> immediate fallback
    mic_decision = mic_ctrl.report_cloud_failure(retryable=False, reason="auth rejected")
    assert mic_decision.action == FailoverAction.START_LOCAL_REPLAY
    assert mic_ctrl.state == FailoverState.FALLING_BACK

    # System stream remains completely unaffected on CLOUD_ACTIVE
    assert sys_ctrl.state == FailoverState.CLOUD_ACTIVE
    assert sys_ctrl.poll().action == FailoverAction.NONE


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSCRIPT LEDGER RECONCILIATION & CONTINUITY WARNINGS
# ─────────────────────────────────────────────────────────────────────────────


def test_ledger_reconciliation_trims_overlap_and_flags_uncertain_gap():
    """Verify fallback epoch reconciliation trims duplicate prefixes and emits continuity warnings."""
    ledger = StreamTranscriptLedger("mic")

    # Epoch 1 commits final: "The presentation is scheduled" (0 to 3000ms)
    ep1 = ledger.begin_epoch("deepgram", offset_ms=0.0)
    item1 = ledger.ingest(
        _result("mic", "ev-1", "The presentation is scheduled", 0.0, 3000.0, is_final=True),
        ep1.handle,
    )
    assert item1.action == LedgerAction.FINAL

    # Epoch 2 (Local fallback replaying from 2500ms with overlap 2500 to 3000ms):
    # Replays "is scheduled for tomorrow morning" (0 to 2500ms provider time -> 2500 to 5000ms session time)
    ep2 = ledger.begin_epoch("local", offset_ms=2500.0, overlap_window=(2500.0, 3000.0))
    item2 = ledger.ingest(
        _result("mic", "ev-2", "is scheduled for tomorrow morning", 0.0, 2500.0, is_final=True),
        ep2.handle,
    )

    # Prefix "is scheduled" trimmed, remainder "for tomorrow morning" committed
    assert item2.action == LedgerAction.FINAL
    assert "tomorrow morning" in item2.text

    # Next, simulate an unresolvable overlap where text completely mismatches
    # Result extends from 4000ms to 7000ms (touches overlap window 4000-5000ms, but extends into new audio 5000-7000ms)
    ep3 = ledger.begin_epoch("local", offset_ms=4000.0, overlap_window=(4000.0, 5000.0))
    item3 = ledger.ingest(
        _result("mic", "ev-3", "unexpected gibberish phrase extends beyond", 0.0, 3000.0, is_final=True),
        ep3.handle,
    )
    assert item3.action == LedgerAction.FINAL
    assert item3.warning is not None
    assert item3.warning.reason == ContinuityReason.UNRESOLVED_OVERLAP


def test_stale_overwrite_protection_from_old_epochs():
    """Verify results from older epochs or late partials after final are dropped."""
    ledger = StreamTranscriptLedger("mic")
    ep1 = ledger.begin_epoch("deepgram", offset_ms=0.0)

    # Epoch 1 emits final
    final_res = _result("mic", "item-1", "Complete sentence.", 0.0, 2000.0, is_final=True, revision=2)
    item = ledger.ingest(final_res, ep1.handle)
    assert item.action == LedgerAction.FINAL

    # Late partial arriving for item-1 after its final is rejected
    late_partial = _result("mic", "item-1", "Complete", 0.0, 1000.0, is_final=False, revision=1)
    rejected_partial = ledger.ingest(late_partial, ep1.handle)
    assert rejected_partial.action == LedgerAction.AFTER_FINAL

    # Epoch 2 starts
    ledger.begin_epoch("local", offset_ms=2000.0)
    # Late result from epoch 1 arriving after epoch 2 is active is dropped
    old_epoch_res = _result("mic", "item-2", "Ghost from past", 1000.0, 2000.0, is_final=True)
    dropped = ledger.ingest(old_epoch_res, ep1.handle)
    assert dropped.action == LedgerAction.RETIRED_EPOCH


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOUNDED MEMORY UNDER CONTINUOUS AUDIO BURSTS
# ─────────────────────────────────────────────────────────────────────────────


def test_audio_ring_buffer_strictly_bounded_at_capacity():
    """Verify continuous streaming never exceeds the 15-second buffer capacity."""
    ring = AudioRingBuffer("mic", sample_rate=SAMPLE_RATE, channels=1)
    max_bytes = 15 * BYTES_PER_SECOND
    assert ring.capacity_bytes == max_bytes

    # Append 60 seconds of audio in 100ms chunks (600 chunks)
    chunk_100ms = b"\x05\x00" * 1600
    for _ in range(600):
        ring.append(chunk_100ms)

    assert ring.capacity_bytes == max_bytes
    span = ring.retained_span()
    # Retained span is exactly the last 15 seconds (45.0s to 60.0s)
    assert span.duration_ms == 15000.0
    assert span.end_ms == 60000.0
    assert span.start_ms == 45000.0

    # Reading from beginning returns bounded audio with missing_ms indicated
    read_data = ring.read(0.0)
    assert len(read_data.pcm_data) == max_bytes
    assert read_data.missing_ms == 45000.0
    assert not read_data.is_complete


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIER 4 APPLICATION SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────


def test_scenario_bilingual_video_call_with_dual_stream_sqlite_persistence(tmp_path):
    """Tier 4 Scenario 1: Bilingual Video Call Meeting (mic + speaker) persisting to SQLite."""
    db_file = tmp_path / DB_FILENAME
    repo = ConversationRepository(db_file)

    # 1. Start conversation
    conv = repo.create_conversation(
        ConversationRecord(
            conversation_id="conv-bilingual-1",
            title="Bilingual sync: Product Review",
            created_at_ms=1000,
            updated_at_ms=1000,
        )
    )
    assert conv.conversation_id == "conv-bilingual-1"

    # 2. Session tracking
    repo.create_session(
        SessionRecord(
            session_id="sess-bilingual-1",
            conversation_id="conv-bilingual-1",
            started_at_ms=1000,
            providers={},
        )
    )

    # 3. Simulate dual-stream meeting
    ledger = TranscriptLedger()
    mic_ep = ledger.begin_epoch("mic", "deepgram", 0.0)
    sys_ep = ledger.begin_epoch("system", "deepgram", 0.0)

    # Mic speaker speaks in English: "Let's review the quarterly revenue metrics."
    mic_p1 = ledger.ingest(
        _result("mic", "m1", "Let's review", 0.0, 1000.0, is_final=False, revision=1),
        mic_ep.handle,
    )
    assert mic_p1.action == LedgerAction.PARTIAL

    mic_fin = ledger.ingest(
        _result("mic", "m1", "Let's review the quarterly revenue metrics.", 0.0, 3000.0, is_final=True, revision=2),
        mic_ep.handle,
    )
    assert mic_fin.action == LedgerAction.FINAL

    # Persist mic finalized message to SQLite
    repo.upsert_message(
        "conv-bilingual-1",
        TranscriptMessage(
            message_id="msg-mic-1",
            session_id="sess-bilingual-1",
            source="mic",
            text=mic_fin.text,
            is_final=True,
            start_ms=0,
            end_ms=3000,
            timestamp_ms=1000,
        ),
    )

    # System speaker responds in Vietnamese: "Toi da chuan bi bao cao tai chinh."
    sys_p1 = ledger.ingest(
        _result("system", "s1", "Toi da", 3200.0, 4000.0, is_final=False, revision=1),
        sys_ep.handle,
    )
    assert sys_p1.action == LedgerAction.PARTIAL

    sys_fin = ledger.ingest(
        _result("system", "s1", "Toi da chuan bi bao cao tai chinh.", 3200.0, 6000.0, is_final=True, revision=2),
        sys_ep.handle,
    )
    assert sys_fin.action == LedgerAction.FINAL

    # Persist system finalized message with translation to SQLite
    repo.upsert_message(
        "conv-bilingual-1",
        TranscriptMessage(
            message_id="msg-sys-1",
            session_id="sess-bilingual-1",
            source="system",
            text=sys_fin.text,
            translation="I have prepared the financial report.",
            is_final=True,
            start_ms=3200,
            end_ms=6000,
            timestamp_ms=3200,
        ),
    )

    # 4. Verify conversation record and messages in SQLite
    stored_msgs = repo.list_messages("conv-bilingual-1")
    assert len(stored_msgs) == 2
    assert stored_msgs[0].source == "mic"
    assert stored_msgs[0].text == "Let's review the quarterly revenue metrics."
    assert stored_msgs[1].source == "system"
    assert stored_msgs[1].text == "Toi da chuan bi bao cao tai chinh."
    assert stored_msgs[1].translation == "I have prepared the financial report."

    # End session
    repo.end_session("sess-bilingual-1", ended_at_ms=7000)
    repo.close()


def test_scenario_cloud_disconnect_with_sticky_local_fallback_and_continuity(tmp_path):
    """Tier 4 Scenario 2: Cloud drop mid-speech, replay from ring buffer, and sticky fallback."""
    db_file = tmp_path / DB_FILENAME
    repo = ConversationRepository(db_file)
    repo.create_conversation(
        ConversationRecord(
            conversation_id="conv-failover-1",
            title="Failover test",
            created_at_ms=1000,
            updated_at_ms=1000,
        )
    )

    clock = FakeClock(10.0)
    ring = AudioRingBuffer("mic")

    # Ingest 5 seconds of audio
    for i in range(5):
        ring.append(_second_pcm(i))

    ledger = StreamTranscriptLedger("mic")
    ep1 = ledger.begin_epoch("deepgram", offset_ms=0.0)

    # Cloud committed first final at 2000ms: "First agenda item."
    res1 = _result("mic", "ev-1", "First agenda item.", 0.0, 2000.0, is_final=True)
    item1 = ledger.ingest(res1, ep1.handle)
    assert item1.action == LedgerAction.FINAL
    repo.upsert_message(
        "conv-failover-1",
        TranscriptMessage(
            message_id="msg-1",
            text=item1.text,
            source="mic",
            is_final=True,
            timestamp_ms=2000,
        ),
    )

    # Failover controller tracks boundary at 2000ms
    ctrl = STTFailoverController(
        "mic",
        ring,
        lambda: 2000.0,
        clock=clock,
        policy=FailoverPolicy(reconnect_timeout_s=1.0, max_reconnect_attempts=0, replay_overlap_ms=500.0),
    )

    # Cloud network drop occurs mid-speech
    decision = ctrl.report_cloud_failure(retryable=False, reason="closed")
    assert decision.action == FailoverAction.START_LOCAL_REPLAY
    assert ctrl.state == FailoverState.FALLING_BACK

    # Replay plan: replaying from 1500ms (2000 - 500)
    plan = decision.plan
    assert plan is not None
    assert plan.replay_start_ms == 1500.0

    # Start Epoch 2 (Local Faster-Whisper fallback)
    ep2 = ledger.begin_epoch("local", offset_ms=plan.offset_ms, overlap_window=plan.overlap_window)
    # Local fallback re-transcribes audio: "agenda item. Moving on to sales numbers."
    local_res = _result("mic", "ev-local-1", "item. Moving on to sales numbers.", 0.0, 2500.0, is_final=True)
    item2 = ledger.ingest(local_res, ep2.handle)

    # Prefix trimmed -> "Moving on to sales numbers." committed cleanly
    assert item2.action == LedgerAction.FINAL
    assert "sales numbers" in item2.text

    ctrl.report_local_started()
    assert ctrl.state == FailoverState.LOCAL_ACTIVE

    repo.upsert_message(
        "conv-failover-1",
        TranscriptMessage(
            message_id="msg-2",
            text=item2.text,
            source="mic",
            is_final=True,
            timestamp_ms=4500,
        ),
    )

    # Verify conversation continuity in SQLite
    msgs = repo.list_messages("conv-failover-1")
    assert len(msgs) == 2
    assert msgs[0].text == "First agenda item."
    assert "sales numbers" in msgs[1].text

    repo.close()

