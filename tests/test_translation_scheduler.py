"""Translation scheduling policy with a fake clock.

Pure: the scheduler has no clock, thread, or event loop of its own, so every test passes the
time explicitly and nothing sleeps. Covers the stable-partial debounce, authoritative finals,
revision-based staleness, concurrency bounds, backlog and budget refusal, and accounting.
"""

import pytest

from engine.session.translation_scheduler import (
    REFUSED_BACKLOG,
    REFUSED_BUDGET,
    JobKind,
    SchedulerPolicy,
    SchedulerPolicyError,
    TranslationScheduler,
    Verdict,
)


def _kinds(tick):
    return [(job.kind, job.stream_id, job.text) for job in tick.dispatch]


# ── Partial debounce ─────────────────────────────────────────────────────────


def test_stable_partial_is_requested_once_after_the_debounce():
    scheduler = TranslationScheduler()

    scheduler.on_partial("mic", "mic:1:a", 1, "hello", now=0.0)
    assert scheduler.poll(0.3).dispatch == ()
    scheduler.on_partial("mic", "mic:1:a", 2, "hello there", now=0.4)
    # The text changed at 0.4, so 0.9 is only 0.5 s of stability.
    assert scheduler.poll(0.9).dispatch == ()

    tick = scheduler.poll(1.0)
    assert _kinds(tick) == [(JobKind.PARTIAL, "mic", "hello there")]
    job = tick.dispatch[0]
    assert (job.item_key, job.source_revision, job.entry_id) == ("mic:1:a", 2, None)
    # The same text is never requested twice for the open item.
    assert scheduler.complete(job) is Verdict.ACCEPTED
    assert scheduler.poll(5.0).dispatch == ()


def test_a_newer_revision_with_the_same_text_keeps_the_debounce_timer():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 1, "good morning", now=0.0)
    scheduler.on_partial("mic", "mic:1:a", 2, "good morning ", now=0.5)

    tick = scheduler.poll(0.6)

    assert [job.source_revision for job in tick.dispatch] == [2]


def test_older_partial_revisions_are_ignored():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 3, "current words", now=0.0)
    scheduler.on_partial("mic", "mic:1:a", 2, "older words", now=0.1)

    tick = scheduler.poll(0.6)

    assert _kinds(tick) == [(JobKind.PARTIAL, "mic", "current words")]


def test_partials_can_be_turned_off():
    scheduler = TranslationScheduler(SchedulerPolicy(translate_partials=False))
    scheduler.on_partial("mic", "mic:1:a", 1, "hello", now=0.0)

    assert scheduler.poll(10.0).dispatch == ()


# ── Finals and staleness ─────────────────────────────────────────────────────


def test_final_is_dispatched_at_once_and_a_late_partial_result_is_stale():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 1, "we should meet", now=0.0)
    partial = scheduler.poll(0.6).dispatch[0]

    assert scheduler.on_final("mic", "mic:1:a", 2, "We should meet.", "e-1") is None
    final = scheduler.poll(0.61).dispatch[0]
    assert (final.kind, final.entry_id, final.source_revision, final.text) == (
        JobKind.FINAL,
        "e-1",
        2,
        "We should meet.",
    )

    assert scheduler.complete(final) is Verdict.ACCEPTED
    # The partial request returns after the final: it can never overwrite it.
    assert scheduler.complete(partial) is Verdict.STALE
    # Nothing for the closed item is requested again.
    assert scheduler.poll(9.0).dispatch == ()


def test_partial_result_returning_before_the_final_result_is_still_stale_once_final_arrived():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 1, "see you", now=0.0)
    partial = scheduler.poll(0.6).dispatch[0]
    scheduler.on_final("mic", "mic:1:a", 2, "see you then", "e-7")

    assert scheduler.complete(partial) is Verdict.STALE


def test_in_flight_partial_is_stale_when_the_text_changes_and_the_new_text_is_requested():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 1, "the meeting", now=0.0)
    first = scheduler.poll(0.6).dispatch[0]
    scheduler.on_partial("mic", "mic:1:a", 2, "the meeting for tomorrow", now=0.7)

    assert scheduler.complete(first) is Verdict.STALE
    assert scheduler.poll(1.2).dispatch == ()
    second = scheduler.poll(1.3).dispatch[0]
    assert (second.text, second.source_revision) == ("the meeting for tomorrow", 2)
    assert scheduler.complete(second) is Verdict.ACCEPTED


def test_retracting_empty_final_closes_the_partial_without_a_request():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 1, "uh", now=0.0)

    assert scheduler.on_final("mic", "mic:1:a", 2, "", "e-2") is None
    assert scheduler.poll(5.0).dispatch == ()


def test_streams_are_independent():
    scheduler = TranslationScheduler(SchedulerPolicy(max_in_flight=3, max_partial_in_flight=2))
    scheduler.on_partial("mic", "mic:1:a", 1, "from the mic", now=0.0)
    scheduler.on_partial("system", "system:1:b", 1, "from the speaker", now=0.1)
    mic, system = scheduler.poll(0.7).dispatch

    scheduler.on_final("mic", "mic:1:a", 2, "from the mic", "e-3")

    assert scheduler.complete(system) is Verdict.ACCEPTED
    assert scheduler.complete(mic) is Verdict.STALE


def test_a_final_result_is_accepted_once_and_a_failed_final_can_be_retried_first():
    scheduler = TranslationScheduler(SchedulerPolicy(max_in_flight=1, max_partial_in_flight=0))
    scheduler.on_final("mic", "mic:1:a", 1, "first", "e-1")
    scheduler.on_final("mic", "mic:1:b", 1, "second", "e-2")
    first = scheduler.poll(0.0).dispatch[0]

    retry = scheduler.fail(first, retry=True)
    assert (retry.entry_id, retry.attempt) == ("e-1", 2)
    assert retry.job_id != first.job_id
    again = scheduler.poll(0.0).dispatch[0]
    assert again == retry

    assert scheduler.complete(again) is Verdict.ACCEPTED
    # A repeated or unknown completion never applies twice.
    assert scheduler.complete(again) is Verdict.STALE
    assert scheduler.poll(0.0).dispatch[0].entry_id == "e-2"


def test_failed_partial_is_not_retried_until_the_text_changes():
    scheduler = TranslationScheduler()
    scheduler.on_partial("mic", "mic:1:a", 1, "hello", now=0.0)
    job = scheduler.poll(0.6).dispatch[0]

    assert scheduler.fail(job, retry=True) is None
    assert scheduler.poll(3.0).dispatch == ()
    scheduler.on_partial("mic", "mic:1:a", 2, "hello again", now=3.0)
    assert len(scheduler.poll(3.6).dispatch) == 1


# ── Bounds ───────────────────────────────────────────────────────────────────


def test_concurrency_is_bounded_and_finals_keep_priority_over_partials():
    scheduler = TranslationScheduler(SchedulerPolicy(max_in_flight=2, max_partial_in_flight=1))
    scheduler.on_partial("system", "system:1:p", 1, "caption", now=0.0)
    for index in range(3):
        scheduler.on_final("mic", f"mic:1:{index}", 1, f"final {index}", f"e-{index}")

    tick = scheduler.poll(1.0)
    assert [job.entry_id for job in tick.dispatch] == ["e-0", "e-1"]
    assert (scheduler.in_flight_count, scheduler.pending_final_count) == (2, 1)

    scheduler.complete(tick.dispatch[0])
    assert [job.entry_id for job in scheduler.poll(1.1).dispatch] == ["e-2"]

    scheduler.complete(tick.dispatch[1])
    assert _kinds(scheduler.poll(1.2)) == [(JobKind.PARTIAL, "system", "caption")]
    # One final and one partial are running; a second partial may not take the last slot.
    scheduler.on_partial("mic", "mic:2:p", 1, "another caption", now=1.2)
    assert scheduler.poll(5.0).dispatch == ()


def test_final_beyond_the_pending_bound_is_refused_with_backlog():
    scheduler = TranslationScheduler(
        SchedulerPolicy(max_in_flight=1, max_partial_in_flight=0, max_pending_finals=2)
    )
    assert scheduler.on_final("mic", "mic:1:a", 1, "one", "e-1") is None
    assert scheduler.on_final("mic", "mic:1:b", 1, "two", "e-2") is None

    assert scheduler.on_final("mic", "mic:1:c", 1, "three", "e-3") == REFUSED_BACKLOG
    assert scheduler.usage().finals_refused == 1


def test_request_budget_refuses_waiting_and_new_finals():
    scheduler = TranslationScheduler(
        SchedulerPolicy(max_in_flight=1, max_partial_in_flight=0, max_requests=1)
    )
    scheduler.on_final("mic", "mic:1:a", 1, "one", "e-1")
    scheduler.on_final("mic", "mic:1:b", 1, "two", "e-2")

    tick = scheduler.poll(0.0)
    assert [job.entry_id for job in tick.dispatch] == ["e-1"]
    assert [(job.entry_id, reason) for job, reason in tick.refused] == [("e-2", REFUSED_BUDGET)]
    assert scheduler.budget_exhausted
    assert scheduler.on_final("mic", "mic:1:c", 1, "three", "e-3") == REFUSED_BUDGET


def test_token_budget_stops_further_requests():
    scheduler = TranslationScheduler(SchedulerPolicy(max_tokens=10))
    scheduler.on_final("mic", "mic:1:a", 1, "one", "e-1")
    job = scheduler.poll(0.0).dispatch[0]
    scheduler.complete(job, input_tokens=6, output_tokens=5)

    scheduler.on_partial("mic", "mic:2:a", 1, "caption", now=0.0)
    assert scheduler.poll(1.0).dispatch == ()
    assert scheduler.on_final("mic", "mic:2:a", 2, "caption", "e-2") == REFUSED_BUDGET


def test_usage_accounts_requests_outcomes_and_tokens():
    scheduler = TranslationScheduler(SchedulerPolicy(max_in_flight=3, max_partial_in_flight=1))
    scheduler.on_partial("system", "system:1:p", 1, "caption", now=0.0)
    scheduler.on_final("mic", "mic:1:a", 1, "one", "e-1")
    scheduler.on_final("mic", "mic:1:b", 1, "two", "e-2")
    final_one, final_two, partial = scheduler.poll(0.6).dispatch

    scheduler.complete(final_one, input_tokens=12, output_tokens=4)
    scheduler.fail(final_two)
    scheduler.on_final("system", "system:1:p", 2, "caption.", "e-3")
    scheduler.complete(partial, input_tokens=3, output_tokens=True)

    usage = scheduler.usage()
    assert (usage.partial_requests, usage.final_requests, usage.requests) == (1, 2, 3)
    assert (usage.succeeded, usage.failed, usage.stale_discarded) == (1, 1, 1)
    assert (usage.input_tokens, usage.output_tokens, usage.tokens) == (15, 4, 19)


def test_close_returns_waiting_finals_and_makes_everything_inert():
    scheduler = TranslationScheduler(SchedulerPolicy(max_in_flight=1, max_partial_in_flight=0))
    scheduler.on_final("mic", "mic:1:a", 1, "one", "e-1")
    scheduler.on_final("mic", "mic:1:b", 1, "two", "e-2")
    running = scheduler.poll(0.0).dispatch[0]

    abandoned = scheduler.close()

    assert [job.entry_id for job in abandoned] == ["e-2"]
    assert scheduler.complete(running) is Verdict.STALE
    assert scheduler.on_final("mic", "mic:1:c", 1, "three", "e-3") is None
    assert scheduler.poll(1.0).dispatch == ()


@pytest.mark.parametrize(
    ("fields", "name"),
    [
        ({"partial_debounce_s": -0.1}, "partial_debounce_s"),
        ({"max_in_flight": 0}, "max_in_flight"),
        ({"max_in_flight": 2, "max_partial_in_flight": 2}, "max_partial_in_flight"),
        ({"max_pending_finals": 0}, "max_pending_finals"),
        ({"max_requests": -1}, "max_requests"),
        ({"max_tokens": True}, "max_tokens"),
    ],
)
def test_invalid_policy_is_rejected(fields, name):
    with pytest.raises(SchedulerPolicyError) as info:
        SchedulerPolicy(**fields)

    assert info.value.field_name == name
