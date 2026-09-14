"""Pins the local conversation repository: ordering, revisions, transactions, delete, export.

Every test uses a database under ``tmp_path``; none touches the real EchoFlux data dir,
contacts a provider, or uses a real credential.
"""

import json
import sqlite3
import threading
from dataclasses import fields

import pytest

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    ConversationRecord,
    ConversationValidationError,
    ProviderMetadata,
    SessionRecord,
    TranscriptMessage,
    import_id_for_browser_history,
)
from engine.conversation.repository import (
    DB_FILENAME,
    STORE_SCHEMA_VERSION,
    ConversationRepository,
    RecordConflictError,
    RecordNotFoundError,
    UnsupportedSchemaError,
    default_database_path,
)
from engine.providers.types import (
    AssistQuestion,
    AssistResult,
    AssistSuggestion,
    SummaryResult,
    SummaryWarning,
)

LOCAL_STT = ProviderMetadata(provider_id="local", model="small")
LOCAL_MT = ProviderMetadata(provider_id="local", model="opus-mt-en-vi")


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "store" / DB_FILENAME


@pytest.fixture
def repo(db_path):
    repository = ConversationRepository(db_path)
    yield repository
    repository.close()


def make_conversation(repo, conversation_id="c1", title="Weekly sync", **kwargs):
    return repo.create_conversation(
        ConversationRecord(
            conversation_id=conversation_id,
            title=title,
            created_at_ms=1000.0,
            updated_at_ms=1000.0,
            **kwargs,
        )
    )


def msg(message_id, source="mic", text="hello", revision=0, **kwargs):
    return TranscriptMessage(
        message_id=message_id,
        source=source,
        text=text,
        timestamp_ms=kwargs.pop("timestamp_ms", 2000.0),
        revision=revision,
        stt_provider=kwargs.pop("stt_provider", LOCAL_STT),
        **kwargs,
    )


def table_counts(repo, conversation_id):
    with repo.read() as conn:
        return {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            for table in ("messages", "sessions", "assist_results", "summaries")
        }


def test_schema_is_versioned_and_database_lives_where_asked(repo, db_path):
    assert repo.schema_version == STORE_SCHEMA_VERSION
    assert repo.db_path == db_path
    assert db_path.exists()


def test_default_database_path_is_in_the_isolated_data_dir(make_config, isolated_env):
    config = make_config()
    assert default_database_path(config) == isolated_env / DB_FILENAME


def test_message_order_sources_and_labels_survive_reload(repo, db_path):
    make_conversation(repo)
    repo.upsert_message("c1", msg("e-1", "mic", "first"))
    repo.upsert_message("c1", msg("e-2", "system", "second"))
    repo.upsert_message("c1", msg("e-3", "mic", "third"))
    repo.set_source_label("c1", "system", "Client team")
    repo.close()

    reopened = ConversationRepository(db_path)
    try:
        messages = reopened.list_messages("c1")
        conversation = reopened.get_conversation("c1")
    finally:
        reopened.close()

    assert [(m.message_id, m.sequence, m.source) for m in messages] == [
        ("e-1", 0, "mic"),
        ("e-2", 1, "system"),
        ("e-3", 2, "mic"),
    ]
    assert conversation.label_for("mic") == "Me"
    assert conversation.label_for("system") == "Client team"
    speakers = [m.to_provider_message(conversation.source_labels).speaker for m in messages]
    assert speakers == ["Me", "Client team", "Me"]
    assert messages[0].stt_provider == LOCAL_STT


def test_default_labels_are_me_and_others(repo):
    conversation = make_conversation(repo)
    assert conversation.label_for("mic") == "Me"
    assert conversation.label_for("system") == "Others"


def test_newer_revision_replaces_in_place_and_stale_revision_is_ignored(repo):
    make_conversation(repo)
    repo.upsert_message("c1", msg("e-1", text="draft", revision=0, is_final=False))
    repo.upsert_message("c1", msg("e-2", text="other"))
    updated = repo.upsert_message("c1", msg("e-1", text="final text", revision=2))
    stale = repo.upsert_message("c1", msg("e-1", text="late partial", revision=1))

    assert updated.sequence == 0
    assert stale.text == "final text"
    stored = repo.list_messages("c1")
    assert [(m.message_id, m.text, m.revision) for m in stored] == [
        ("e-1", "final text", 2),
        ("e-2", "other", 0),
    ]


def test_translation_is_attached_and_survives_idempotent_resend(repo):
    make_conversation(repo)
    repo.upsert_message("c1", msg("e-1", text="hello", revision=1))
    repo.set_translation("c1", "e-1", "xin chao", provider=LOCAL_MT, revision=1)
    ignored = repo.set_translation("c1", "e-1", "stale", provider=LOCAL_MT, revision=0)
    assert ignored.translation == "xin chao"

    resent = repo.upsert_message("c1", msg("e-1", text="hello", revision=1))
    assert resent.translation == "xin chao"
    assert resent.translation_provider == LOCAL_MT

    changed = repo.upsert_message("c1", msg("e-1", text="hello there", revision=2))
    assert changed.translation is None

    with pytest.raises(RecordNotFoundError):
        repo.set_translation("c1", "missing", "x")


def test_concurrent_writers_never_share_or_skip_sequence(repo, db_path):
    make_conversation(repo)
    second = ConversationRepository(db_path)
    errors = []

    def writer(target, worker):
        try:
            for i in range(25):
                target.upsert_message("c1", msg(f"w{worker}-{i}", text=f"{worker}:{i}"))
        except Exception as exc:  # surfaced below; a thread cannot fail the test directly
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(repo if n % 2 else second, n)) for n in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    second.close()

    assert errors == []
    messages = repo.list_messages("c1")
    assert [m.sequence for m in messages] == list(range(200))
    assert len({m.message_id for m in messages}) == 200
    for worker in range(8):
        own = [int(m.text.split(":")[1]) for m in messages if m.text.startswith(f"{worker}:")]
        assert own == list(range(25))


def test_failed_import_leaves_no_partial_conversation(repo):
    record = ConversationRecord(
        conversation_id="imp-1",
        title="Imported",
        created_at_ms=1.0,
        updated_at_ms=1.0,
        import_id=import_id_for_browser_history("1712345678"),
    )
    with pytest.raises(RecordConflictError):
        repo.import_conversation(record, [msg("e-1"), msg("e-2"), msg("e-1", text="dup")])

    assert repo.get_conversation("imp-1") is None
    assert table_counts(repo, "imp-1") == {
        "messages": 0,
        "sessions": 0,
        "assist_results": 0,
        "summaries": 0,
    }


def test_error_inside_outer_transaction_rolls_back_nested_writes(repo):
    with pytest.raises(RuntimeError), repo.transaction():
        make_conversation(repo)
        repo.upsert_message("c1", msg("e-1"))
        raise RuntimeError("simulated crash mid-write")

    assert repo.get_conversation("c1") is None
    assert repo.list_messages("c1") == []
    # The repository stays usable after the rollback.
    make_conversation(repo)
    assert repo.get_conversation("c1") is not None


def test_uncommitted_write_is_absent_after_reopen(repo, db_path):
    repo.close()
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    raw.execute("BEGIN IMMEDIATE")
    raw.execute("INSERT INTO conversations VALUES ('ghost', 'Ghost', 1, 1, '{}', NULL, 1)")
    raw.close()  # connection lost before COMMIT

    reopened = ConversationRepository(db_path)
    try:
        assert reopened.get_conversation("ghost") is None
        assert reopened.list_conversations() == []
    finally:
        reopened.close()


def test_browser_history_import_is_idempotent(repo):
    import_id = import_id_for_browser_history("1712345678")
    record = ConversationRecord(
        conversation_id="imp-1",
        title="Imported",
        created_at_ms=1.0,
        updated_at_ms=1.0,
        import_id=import_id,
    )
    first = repo.import_conversation(record, [msg("e-1", "mic"), msg("e-2", "system")])
    retry = repo.import_conversation(
        ConversationRecord(
            conversation_id="imp-other",
            title="Imported again",
            created_at_ms=2.0,
            updated_at_ms=2.0,
            import_id=import_id,
        ),
        [msg("e-9")],
    )

    assert first.imported is True
    assert retry.imported is False
    assert retry.conversation_id == "imp-1"
    assert [c.conversation_id for c in repo.list_conversations()] == ["imp-1"]
    assert [m.message_id for m in repo.list_messages("imp-1")] == ["e-1", "e-2"]

    with pytest.raises(ConversationValidationError):
        repo.import_conversation(ConversationRecord("no-key", "No key", 1.0, 1.0), [msg("e-1")])


def test_assist_result_and_summary_warning_round_trip(repo, db_path):
    make_conversation(repo)
    repo.upsert_message("c1", msg("e-1", "system", "Can you ship Friday?"))
    assist = AssistResult(
        target_message_id="e-1",
        suggested_replies=(AssistSuggestion("confirm", "Yes, Friday works.", "fits plan"),),
        clarifying_questions=(AssistQuestion("Which build?", "scope"),),
        provider_event_id="evt-1",
        input_tokens=120,
        output_tokens=30,
    )
    summary = SummaryResult(
        summary_markdown="## Summary\nShip Friday.",
        bullet_points=("Ship Friday",),
        decisions=("Release on Friday",),
        action_items=("Tag build",),
        risks=("QA time",),
        warning=SummaryWarning(
            omitted_message_count=4,
            first_omitted_timestamp_ms=10.0,
            last_omitted_timestamp_ms=40.0,
            reason="context_limit",
        ),
    )
    assistant = ProviderMetadata(provider_id="openai", model="gpt-4o-mini")
    repo.save_assist_result("c1", assist, provider=assistant, assist_id="a-1", created_at_ms=5.0)
    repo.save_summary("c1", summary, provider=assistant, summary_id="s-1", created_at_ms=6.0)
    repo.close()

    reopened = ConversationRepository(db_path)
    try:
        [stored_assist] = reopened.list_assist_results("c1", target_message_id="e-1")
        stored_summary = reopened.latest_summary("c1")
        with pytest.raises(RecordNotFoundError):
            reopened.save_assist_result("c1", AssistResult("missing", (), ()))
    finally:
        reopened.close()

    assert stored_assist.suggested_replies == assist.suggested_replies
    assert stored_assist.clarifying_questions == assist.clarifying_questions
    assert stored_assist.provider == assistant
    assert stored_assist.input_tokens == 120
    assert stored_summary.warning == summary.warning
    assert stored_summary.decisions == ("Release on Friday",)
    assert stored_summary.summary_markdown == summary.summary_markdown


def test_delete_is_scoped_to_one_conversation(repo):
    service = ContextProfileService(repo)
    profile = service.create_profile("Glossary", "ACME = client", profile_id="p-1")
    for cid in ("c1", "c2"):
        make_conversation(repo, cid)
        repo.upsert_message(cid, msg("e-1", text=f"text {cid}"))
        repo.save_assist_result(cid, AssistResult("e-1", (), ()), assist_id=f"a-{cid}")
        repo.save_summary(cid, SummaryResult(f"summary {cid}"), summary_id=f"s-{cid}")
        service.start_session(
            SessionRecord(f"sess-{cid}", cid, 10.0), profile_ids=[profile.profile_id]
        )

    assert repo.delete_conversation("c1") is True
    assert repo.delete_conversation("c1") is False

    assert repo.get_conversation("c1") is None
    assert set(table_counts(repo, "c1").values()) == {0}
    assert service.snapshot_info("sess-c1") == []
    assert table_counts(repo, "c2") == {
        "messages": 1,
        "sessions": 1,
        "assist_results": 1,
        "summaries": 1,
    }
    assert [s.name for s in service.snapshot_info("sess-c2")] == ["Glossary"]
    assert service.get_profile("p-1") is not None


def test_export_contains_one_conversation_and_no_secrets(repo):
    service = ContextProfileService(repo)
    service.create_profile("Secret-free notes", "PROFILE-BODY-SENTINEL", profile_id="p-1")
    make_conversation(repo, "c1", "Target")
    make_conversation(repo, "c2", "Unrelated")
    repo.upsert_message("c2", msg("e-x", text="UNRELATED-SENTINEL"))
    service.start_session(
        SessionRecord(
            "sess-1",
            "c1",
            10.0,
            providers={
                "stt": LOCAL_STT,
                "assistant": ProviderMetadata(provider_id="openai", model="gpt-4o-mini"),
            },
        ),
        profile_ids=["p-1"],
    )
    repo.upsert_message("c1", msg("e-1", "system", "hi", session_id="sess-1"))
    repo.set_translation("c1", "e-1", "xin chao", provider=LOCAL_MT)

    export = repo.export_conversation("c1", exported_at_ms=99.0)
    text = json.dumps(export)

    assert export["conversation"]["conversation_id"] == "c1"
    assert [m["speaker"] for m in export["messages"]] == ["Others"]
    assert export["sessions"][0]["providers"]["assistant"]["provider_id"] == "openai"
    assert export["sessions"][0]["context_profiles"][0]["name"] == "Secret-free notes"
    for forbidden in (
        "UNRELATED-SENTINEL",
        "Unrelated",
        "c2",
        "PROFILE-BODY-SENTINEL",
        "credential",
        "apiKey",
        "api_key",
        "endpoint",
    ):
        assert forbidden not in text
    with pytest.raises(RecordNotFoundError):
        repo.export_conversation("missing")


def test_records_reject_secret_shaped_provider_data_without_echoing_it():
    with pytest.raises(ConversationValidationError) as exc:
        ProviderMetadata(provider_id="sk-live-SENTINEL")
    assert "SENTINEL" not in str(exc.value)
    with pytest.raises(ConversationValidationError):
        ProviderMetadata(provider_id="openai", model="sk-proj-abcdef")
    with pytest.raises(ConversationValidationError):
        ProviderMetadata.from_dict({"provider_id": "openai", "api_key": "x"})
    with pytest.raises(ConversationValidationError):
        SessionRecord("s", "c", 1.0, providers={"diarization": LOCAL_STT})
    with pytest.raises(ConversationValidationError) as exc:
        ConversationRecord("c", "bad\x00TITLE-SENTINEL", 1.0, 1.0)
    assert "TITLE-SENTINEL" not in str(exc.value)


def test_message_session_must_belong_to_the_conversation(repo):
    make_conversation(repo, "c1")
    make_conversation(repo, "c2")
    repo.create_session(SessionRecord("sess-2", "c2", 1.0))
    with pytest.raises(RecordNotFoundError):
        repo.upsert_message("c1", msg("e-1", session_id="sess-2"))
    with pytest.raises(RecordNotFoundError):
        repo.upsert_message("missing", msg("e-1"))


def test_end_session_records_stop_time_once(repo):
    make_conversation(repo)
    repo.create_session(SessionRecord("sess-1", "c1", 10.0, providers={"stt": LOCAL_STT}))
    ended = repo.end_session("sess-1", ended_at_ms=20.0)
    again = repo.end_session("sess-1", ended_at_ms=30.0)
    assert ended.ended_at_ms == 20.0
    assert again.ended_at_ms == 20.0
    assert repo.get_session("sess-1").providers == {"stt": LOCAL_STT}
    with pytest.raises(RecordConflictError):
        repo.create_session(SessionRecord("sess-1", "c1", 11.0))


def test_newer_database_schema_is_refused_and_left_untouched(db_path):
    db_path.parent.mkdir(parents=True)
    raw = sqlite3.connect(str(db_path))
    raw.execute("PRAGMA user_version = 99")
    raw.close()

    with pytest.raises(UnsupportedSchemaError):
        ConversationRepository(db_path)

    raw = sqlite3.connect(str(db_path))
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 99
        assert raw.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    finally:
        raw.close()


def test_no_raw_audio_is_stored(repo):
    with repo.read() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            for column in conn.execute(f"PRAGMA table_info({table})"):
                assert column["type"].upper() != "BLOB"
                assert "audio" not in column["name"].lower()
                assert "pcm" not in column["name"].lower()
    assert not any(f.type is bytes for f in fields(TranscriptMessage))


def test_closed_repository_refuses_work(repo):
    repo.close()
    with pytest.raises(Exception, match="closed"):
        repo.list_conversations()
