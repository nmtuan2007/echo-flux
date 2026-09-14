"""Transcript ledger: revision order, finality, interval ownership, and dedupe."""

import threading

import pytest

from engine.providers.types import STTTranscriptResult
from engine.session.transcript_ledger import (
    ContinuityReason,
    LedgerAction,
    LedgerThresholds,
    StreamTranscriptLedger,
    TranscriptLedger,
    normalize_text,
)


def _result(
    event: str = "a",
    revision: int = 1,
    text: str = "hello",
    final: bool = False,
    start: float = 0.0,
    end: float = 1000.0,
    stream: str = "mic",
) -> STTTranscriptResult:
    return STTTranscriptResult(
        stream_id=stream,
        sequence=0,
        provider_event_id=event,
        text=text,
        is_final=final,
        revision=revision,
        start_time_ms=start,
        end_time_ms=end,
        monotonic_timestamp=0.0,
    )


def _replay_setup(overlap: bool = True):
    """Cloud committed one final at 9-11 s, then Local replays from 9 s."""
    ledger = StreamTranscriptLedger("mic")
    cloud = ledger.begin_epoch("cloud").handle
    committed = ledger.ingest(
        _result("c1", 1, "we should schedule the meeting", True, 9000, 11000), cloud
    )
    assert committed.action is LedgerAction.FINAL
    window = (9000.0, 11000.0) if overlap else None
    local = ledger.begin_epoch("local", offset_ms=9000.0, overlap_window=window).handle
    return ledger, cloud, local


# ── Revision and finality ────────────────────────────────────────────────────


def test_partial_then_final_share_entry_key():
    ledger = StreamTranscriptLedger("mic")
    epoch = ledger.begin_epoch("cloud").handle

    partial = ledger.ingest(_result(revision=1, text="hel"), epoch)
    final = ledger.ingest(_result(revision=2, text="hello", final=True), epoch)

    assert partial.action is LedgerAction.PARTIAL
    assert final.action is LedgerAction.FINAL
    assert partial.entry_key == final.entry_key == "mic:1:a"
    assert final.text == "hello"
    assert ledger.last_final_boundary_ms == 1000.0


def test_stale_and_repeated_revisions_are_rejected():
    ledger = StreamTranscriptLedger("mic")
    epoch = ledger.begin_epoch("cloud").handle

    assert (
        ledger.ingest(_result(revision=2, text="hello wor"), epoch).action is LedgerAction.PARTIAL
    )
    stale = ledger.ingest(_result(revision=1, text="hel"), epoch)
    repeat = ledger.ingest(_result(revision=2, text="hello wor"), epoch)

    assert stale.action is LedgerAction.STALE
    assert repeat.action is LedgerAction.STALE
    assert not stale.is_visible
    assert stale.text == ""


def test_partial_never_overwrites_final():
    ledger = StreamTranscriptLedger("mic")
    epoch = ledger.begin_epoch("cloud").handle
    ledger.ingest(_result(revision=3, text="hello world", final=True), epoch)

    late_partial = ledger.ingest(_result(revision=4, text="hello"), epoch)
    second_final = ledger.ingest(_result(revision=5, text="other", final=True), epoch)

    assert late_partial.action is LedgerAction.AFTER_FINAL
    assert second_final.action is LedgerAction.AFTER_FINAL
    assert ledger.last_final_boundary_ms == 1000.0


def test_empty_final_retracts_without_advancing_boundary():
    ledger = StreamTranscriptLedger("mic")
    epoch = ledger.begin_epoch("deepgram").handle
    ledger.ingest(_result(revision=1, text="uh"), epoch)

    decision = ledger.ingest(_result(revision=2, text="  ", final=True), epoch)

    assert decision.action is LedgerAction.RETRACTED
    assert decision.is_visible
    assert ledger.last_final_boundary_ms == 0.0
    assert ledger.ingest(_result(revision=3, text="uh"), epoch).action is LedgerAction.AFTER_FINAL


def test_epoch_offset_maps_provider_time_onto_session_timeline():
    ledger = StreamTranscriptLedger("mic")
    epoch = ledger.begin_epoch("cloud", offset_ms=5000.0).handle

    decision = ledger.ingest(_result(text="hi", final=True, start=0, end=1000), epoch)

    assert (decision.start_ms, decision.end_ms) == (5000.0, 6000.0)
    assert ledger.last_final_boundary_ms == 6000.0


def test_new_epoch_retires_old_connection_and_reports_abandoned_partials():
    ledger = StreamTranscriptLedger("mic")
    old = ledger.begin_epoch("cloud").handle
    ledger.ingest(_result("done", 1, "finished", True), old)
    ledger.ingest(_result("open", 1, "still talking"), old)

    start = ledger.begin_epoch("cloud", offset_ms=1000.0)

    assert start.abandoned_entry_keys == ("mic:1:open",)
    late = ledger.ingest(_result("open", 2, "still talking now", True), old)
    assert late.action is LedgerAction.RETIRED_EPOCH
    assert ledger.active_epoch == start.handle


def test_same_epoch_finals_are_never_deduplicated():
    ledger = StreamTranscriptLedger("mic")
    epoch = ledger.begin_epoch("cloud").handle

    first = ledger.ingest(_result("a", 1, "yes", True, 0, 1000), epoch)
    second = ledger.ingest(_result("b", 1, "yes", True, 500, 1000), epoch)

    assert first.action is LedgerAction.FINAL
    assert second.action is LedgerAction.FINAL


# ── Duplicate reconciliation ─────────────────────────────────────────────────


def test_fully_covered_interval_is_duplicate_even_when_text_differs():
    ledger, _, local = _replay_setup()

    decision = ledger.ingest(
        _result("l1", 1, "we should schedule a meeting yes", True, 0, 2000), local
    )

    assert decision.action is LedgerAction.DUPLICATE_FINAL
    assert decision.duplicate_of == "mic:1:c1"
    assert ledger.duplicate_count == 1
    assert ledger.last_final_boundary_ms == 11000.0


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("the meeting", 1400, 2600),  # repeated suffix, nothing new
        ("schedule the", 1400, 2600),  # contained fragment
        ("We should schedule a meeting!", 1000, 2500),  # near-identical wording
    ],
)
def test_partial_interval_overlap_in_window_uses_text(text, start, end):
    ledger, _, local = _replay_setup()

    decision = ledger.ingest(_result("l1", 1, text, True, start, end), local)

    assert decision.action is LedgerAction.DUPLICATE_FINAL
    assert ledger.warning_count == 0


def test_repeated_prefix_is_trimmed_and_new_words_kept():
    ledger, _, local = _replay_setup()

    decision = ledger.ingest(
        _result("l1", 1, "Schedule the meeting, tomorrow at noon.", True, 1000, 4000), local
    )

    assert decision.action is LedgerAction.FINAL
    assert decision.text == "tomorrow at noon."
    assert decision.trimmed_prefix_tokens == 3
    assert decision.start_ms == 11000.0
    assert ledger.last_final_boundary_ms == 13000.0
    assert ledger.warning_count == 0


def test_unrelated_text_in_overlap_window_is_committed_with_warning():
    ledger, _, local = _replay_setup()

    decision = ledger.ingest(
        _result("l1", 1, "completely different words here", True, 1000, 3000), local
    )

    assert decision.action is LedgerAction.FINAL
    assert decision.warning is not None
    assert decision.warning.reason is ContinuityReason.UNRESOLVED_OVERLAP
    assert (decision.warning.start_ms, decision.warning.end_ms) == (10000.0, 12000.0)
    assert ledger.warnings() == [decision.warning]


def test_text_is_not_compared_outside_an_overlap_window():
    ledger, _, local = _replay_setup(overlap=False)

    decision = ledger.ingest(
        _result("l1", 1, "we should schedule the meeting", True, 1000, 3000), local
    )

    assert decision.action is LedgerAction.FINAL
    assert decision.warning is not None
    assert decision.warning.reason is ContinuityReason.UNRESOLVED_OVERLAP


def test_identical_text_without_interval_overlap_is_new_speech():
    ledger, _, local = _replay_setup()

    decision = ledger.ingest(
        _result("l1", 1, "we should schedule the meeting", True, 3000, 4000), local
    )

    assert decision.action is LedgerAction.FINAL
    assert decision.warning is None


def test_replayed_partials_are_suppressed_then_trimmed():
    ledger, _, local = _replay_setup()

    covered = ledger.ingest(_result("l1", 1, "we should", False, 0, 1500), local)
    growing = ledger.ingest(
        _result("l1", 2, "we should schedule the meeting for everyone", False, 0, 3000), local
    )

    assert covered.action is LedgerAction.SUPPRESSED_PARTIAL
    assert not covered.is_visible
    assert growing.action is LedgerAction.PARTIAL
    assert growing.text == "for everyone"


def test_gap_is_recorded_as_explicit_warning():
    ledger = StreamTranscriptLedger("mic")
    ledger.begin_epoch("cloud")

    start = ledger.begin_epoch("local", offset_ms=5000.0, gap=(0.0, 5000.0))

    assert start.gap_warning is not None
    assert start.gap_warning.reason is ContinuityReason.AUDIO_GAP
    assert start.gap_warning.epoch_id == start.handle.epoch_id
    assert ledger.warning_count == 1


# ── Stream independence ──────────────────────────────────────────────────────


def test_mic_and_system_never_reconcile_against_each_other():
    ledger = TranscriptLedger()
    mic = ledger.begin_epoch("mic", "cloud").handle
    ledger.ingest(_result("m1", 1, "hello there", True, 0, 1000, "mic"), mic)
    ledger.begin_epoch("system", "cloud")
    system_local = ledger.begin_epoch(
        "system", "local", overlap_window=(0.0, 1000.0), gap=None
    ).handle
    mic_local = ledger.begin_epoch("mic", "local", overlap_window=(0.0, 1000.0)).handle

    system_final = ledger.ingest(
        _result("s1", 1, "hello there", True, 0, 1000, "system"), system_local
    )
    mic_final = ledger.ingest(_result("m2", 1, "hello there", True, 0, 1000, "mic"), mic_local)

    assert system_final.action is LedgerAction.FINAL
    assert mic_final.action is LedgerAction.DUPLICATE_FINAL
    assert ledger.last_final_boundary_ms("system") == 1000.0
    assert ledger.stream("system").active_epoch == system_local
    assert sorted(ledger.stream_ids()) == ["mic", "system"]


def test_result_for_another_stream_is_rejected():
    ledger = TranscriptLedger()
    mic = ledger.begin_epoch("mic", "cloud").handle

    decision = ledger.ingest(_result(stream="system", final=True), mic)

    assert decision.action is LedgerAction.STREAM_MISMATCH
    assert ledger.last_final_boundary_ms("mic") == 0.0


def test_streams_ingest_concurrently_without_interference():
    ledger = TranscriptLedger()
    handles = {sid: ledger.begin_epoch(sid, "cloud").handle for sid in ("mic", "system")}
    outcomes = {"mic": [], "system": []}

    def run(stream_id: str) -> None:
        for index in range(200):
            result = _result(
                f"e{index}", 1, f"word {index}", True, index * 1000, index * 1000 + 900, stream_id
            )
            outcomes[stream_id].append(ledger.ingest(result, handles[stream_id]).action)

    threads = [threading.Thread(target=run, args=(sid,)) for sid in handles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for stream_id in handles:
        assert outcomes[stream_id] == [LedgerAction.FINAL] * 200
        assert ledger.last_final_boundary_ms(stream_id) == 199900.0


# ── Bounds and helpers ───────────────────────────────────────────────────────


def test_ledger_memory_is_bounded():
    thresholds = LedgerThresholds(max_items=8, max_committed_finals=4, max_warnings=3)
    ledger = StreamTranscriptLedger("mic", thresholds)
    epoch = ledger.begin_epoch("cloud").handle

    for index in range(100):
        ledger.ingest(_result(f"e{index}", 1, "x", True, index * 10, index * 10 + 5), epoch)
    for index in range(10):
        ledger.begin_epoch("local", gap=(0.0, float(index + 1)))

    assert ledger.retained_item_count <= 8
    assert ledger.retained_final_count == 4
    assert len(ledger.warnings()) == 3
    assert ledger.warning_count == 10


def test_normalize_text_folds_case_punctuation_and_width():
    assert normalize_text("  Hello,   WORLD!! ") == "hello world"
    fullwidth_abc_ellipsis = "".join(chr(code) for code in (0xFF21, 0xFF22, 0xFF23, 0x2026))
    assert normalize_text(fullwidth_abc_ellipsis) == "abc"
    assert normalize_text("?!") == ""
