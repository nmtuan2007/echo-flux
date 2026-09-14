"""Pins Context Profiles: Text/Markdown limits, attachments, and immutable session snapshots.

Every test uses a database under ``tmp_path``; none contacts a provider.
"""

import logging
import sqlite3
import threading

import pytest

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    MAX_PROFILE_CONTENT_BYTES,
    MAX_PROFILE_NAME_CHARS,
    MAX_PROFILES_PER_SESSION,
    ContextProfileInfo,
    ConversationRecord,
    ConversationValidationError,
    SessionRecord,
)
from engine.conversation.repository import (
    ConversationRepository,
    RecordConflictError,
    RecordNotFoundError,
)
from engine.providers.types import ContextProfileSnapshot


@pytest.fixture
def repo(tmp_path):
    repository = ConversationRepository(tmp_path / "profiles.sqlite3")
    for cid in ("c1", "c2"):
        repository.create_conversation(ConversationRecord(cid, f"Conversation {cid}", 1.0, 1.0))
    yield repository
    repository.close()


@pytest.fixture
def service(repo):
    return ContextProfileService(repo)


def snapshot_rows(repo, session_id):
    with repo.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM session_profile_snapshots WHERE session_id = ?", (session_id,)
        ).fetchone()[0]


def test_profile_crud_and_listing_carries_no_content(service):
    created = service.create_profile("Glossary", "# Terms\nACME = client", profile_id="p-1")
    service.create_profile("about me", "Plain text bio", content_format="text", profile_id="p-2")

    assert created.revision == 1
    listing = service.list_profiles()
    assert [p.profile_id for p in listing] == ["p-2", "p-1"]
    assert all(isinstance(p, ContextProfileInfo) for p in listing)
    assert not hasattr(listing[0], "content")

    updated = service.update_profile("p-1", content="# Terms\nACME = vendor", expected_revision=1)
    assert updated.revision == 2
    assert service.get_profile("p-1").content == "# Terms\nACME = vendor"
    with pytest.raises(RecordConflictError):
        service.update_profile("p-1", name="Renamed", expected_revision=1)
    with pytest.raises(RecordConflictError):
        service.create_profile("Dup", "x", profile_id="p-1")

    assert service.delete_profile("p-2") is True
    assert service.delete_profile("p-2") is False
    assert service.get_profile("p-2") is None
    with pytest.raises(RecordNotFoundError):
        service.update_profile("p-2", content="x")


def test_only_text_and_markdown_are_accepted(service):
    for bad_format in ("html", "pdf", "docx", ""):
        with pytest.raises(ConversationValidationError):
            service.create_profile("Doc", "content", content_format=bad_format)
    with pytest.raises(ConversationValidationError):
        service.create_profile("Binary", "abc\x00def")
    with pytest.raises(ConversationValidationError):
        service.create_profile("Bytes", b"raw bytes")
    assert service.list_profiles() == []


def test_content_and_name_size_limits_are_explicit(service):
    at_limit = service.create_profile("Max", "a" * MAX_PROFILE_CONTENT_BYTES, profile_id="p-max")
    assert service.list_profiles()[0].content_bytes == MAX_PROFILE_CONTENT_BYTES

    with pytest.raises(ConversationValidationError) as exc:
        service.create_profile("Over", "a" * (MAX_PROFILE_CONTENT_BYTES + 1))
    assert exc.value.code == "content_too_large"

    # The limit is in UTF-8 bytes, not characters.
    multibyte = "é" * (MAX_PROFILE_CONTENT_BYTES // 2 + 1)
    with pytest.raises(ConversationValidationError):
        service.create_profile("Accents", multibyte)

    with pytest.raises(ConversationValidationError):
        service.create_profile("n" * (MAX_PROFILE_NAME_CHARS + 1), "x")
    with pytest.raises(ConversationValidationError):
        service.update_profile(at_limit.profile_id, content=at_limit.content + "b")
    assert service.get_profile("p-max").revision == 1


def test_edits_and_deletes_do_not_change_past_snapshots(service):
    service.create_profile("Glossary", "v1 glossary", profile_id="p-1")
    service.create_profile("Agenda", "v1 agenda", profile_id="p-2")
    service.set_attached_profiles("c1", ["p-2", "p-1"])

    first = service.start_session(SessionRecord("s-1", "c1", 100.0))
    assert [(s.profile_id, s.position, s.profile_revision) for s in first] == [
        ("p-2", 0, 1),
        ("p-1", 1, 1),
    ]

    service.update_profile("p-1", name="Glossary v2", content="v2 glossary")
    service.delete_profile("p-2")

    assert service.snapshots_for_outbound("s-1", "assist") == (
        ContextProfileSnapshot("p-2", "Agenda", "v1 agenda"),
        ContextProfileSnapshot("p-1", "Glossary", "v1 glossary"),
    )
    assert service.attached_profile_ids("c1") == ["p-1"]

    service.start_session(SessionRecord("s-2", "c1", 200.0))
    assert service.snapshots_for_outbound("s-2", "summary") == (
        ContextProfileSnapshot("p-1", "Glossary v2", "v2 glossary"),
    )
    assert service.snapshots_for_outbound("s-1", "summary")[1].content == "v1 glossary"


def test_snapshot_rows_cannot_be_updated(service, repo):
    service.create_profile("Glossary", "original", profile_id="p-1")
    service.start_session(SessionRecord("s-1", "c1", 1.0), profile_ids=["p-1"])

    with (
        pytest.raises(sqlite3.DatabaseError, match="snapshot_immutable"),
        repo.transaction() as conn,
    ):
        conn.execute("UPDATE session_profile_snapshots SET content = 'tampered'")

    assert service.snapshots_for_outbound("s-1", "assist")[0].content == "original"


def test_tampered_snapshot_content_is_withheld(service, repo):
    service.create_profile("Glossary", "original", profile_id="p-1")
    service.start_session(SessionRecord("s-1", "c1", 1.0), profile_ids=["p-1"])
    with repo.transaction() as conn:
        conn.execute("DROP TRIGGER session_profile_snapshots_immutable")
        conn.execute("UPDATE session_profile_snapshots SET content = 'tampered'")

    with pytest.raises(RecordConflictError):
        service.snapshots_for_outbound("s-1", "assist")


def test_failed_snapshot_rolls_back_the_whole_session(service, repo):
    service.create_profile("Glossary", "text", profile_id="p-1")

    with pytest.raises(RecordNotFoundError):
        service.start_session(SessionRecord("s-1", "c1", 1.0), profile_ids=["p-1", "missing"])
    assert repo.get_session("s-1") is None
    assert snapshot_rows(repo, "s-1") == 0

    big = "a" * MAX_PROFILE_CONTENT_BYTES
    ids = [service.create_profile(f"Big {i}", big).profile_id for i in range(5)]
    with pytest.raises(ConversationValidationError) as exc:
        service.start_session(SessionRecord("s-2", "c1", 1.0), profile_ids=ids)
    assert exc.value.code == "session_profiles_too_large"
    assert repo.get_session("s-2") is None
    assert snapshot_rows(repo, "s-2") == 0

    too_many = [
        service.create_profile(f"Small {i}", "x").profile_id
        for i in range(MAX_PROFILES_PER_SESSION + 1)
    ]
    with pytest.raises(ConversationValidationError):
        service.start_session(SessionRecord("s-3", "c1", 1.0), profile_ids=too_many)
    with pytest.raises(ConversationValidationError):
        service.set_attached_profiles("c1", too_many)
    with pytest.raises(ConversationValidationError):
        service.set_attached_profiles("c1", ["p-1", "p-1"])
    assert repo.get_session("s-3") is None


def test_restarting_an_existing_session_keeps_original_snapshot(service, repo):
    service.create_profile("Glossary", "first", profile_id="p-1")
    service.start_session(SessionRecord("s-1", "c1", 1.0), profile_ids=["p-1"])
    service.update_profile("p-1", content="second")

    with pytest.raises(RecordConflictError):
        service.start_session(SessionRecord("s-1", "c1", 2.0), profile_ids=["p-1"])
    assert service.snapshots_for_outbound("s-1", "assist")[0].content == "first"


def test_attachment_failure_keeps_previous_attachments(service):
    service.create_profile("Glossary", "text", profile_id="p-1")
    service.set_attached_profiles("c1", ["p-1"])
    with pytest.raises(RecordNotFoundError):
        service.set_attached_profiles("c1", ["missing"])
    assert service.attached_profile_ids("c1") == ["p-1"]
    with pytest.raises(RecordNotFoundError):
        service.set_attached_profiles("no-such-conversation", ["p-1"])


def test_sessions_and_conversations_are_isolated(service, repo):
    service.create_profile("Shared", "shared text", profile_id="p-1")
    service.create_profile("Only c2", "c2 text", profile_id="p-2")
    service.set_attached_profiles("c1", ["p-1"])
    service.set_attached_profiles("c2", ["p-1", "p-2"])
    service.start_session(SessionRecord("s-c1", "c1", 1.0))
    service.start_session(SessionRecord("s-c2", "c2", 1.0))

    assert [s.profile_id for s in service.snapshots_for_outbound("s-c1", "assist")] == ["p-1"]
    assert [s.profile_id for s in service.snapshots_for_outbound("s-c2", "assist")] == [
        "p-1",
        "p-2",
    ]

    repo.delete_conversation("c1")
    assert service.snapshots_for_outbound("s-c1", "assist") == ()
    assert len(service.snapshots_for_outbound("s-c2", "assist")) == 2
    assert service.get_profile("p-1") is not None
    assert service.attached_profile_ids("c2") == ["p-1", "p-2"]


def test_outbound_content_requires_an_explicit_assist_or_summary_purpose(service):
    service.create_profile("Glossary", "text", profile_id="p-1")
    service.start_session(SessionRecord("s-1", "c1", 1.0), profile_ids=["p-1"])

    for purpose in ("export", "translation", "", None):
        with pytest.raises(ConversationValidationError):
            service.snapshots_for_outbound("s-1", purpose)
    info = service.snapshot_info("s-1")
    assert [(i.profile_id, i.revision, i.content_bytes) for i in info] == [("p-1", 1, 4)]
    assert not hasattr(info[0], "content")


def test_concurrent_edits_never_produce_a_torn_snapshot(service, repo):
    service.create_profile("Live", "rev-1", profile_id="p-1")
    service.set_attached_profiles("c1", ["p-1"])
    errors = []

    def editor():
        try:
            for _ in range(40):
                current = service.get_profile("p-1")
                service.update_profile(
                    "p-1",
                    name=f"Live {current.revision + 1}",
                    content=f"rev-{current.revision + 1}",
                    expected_revision=current.revision,
                )
        except Exception as exc:
            errors.append(exc)

    def starter():
        try:
            for i in range(40):
                service.start_session(SessionRecord(f"s-{i}", "c1", float(i)))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=editor), threading.Thread(target=starter)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    for i in range(40):
        [info] = service.snapshot_info(f"s-{i}")
        [snapshot] = service.snapshots_for_outbound(f"s-{i}", "assist")
        assert snapshot.content == f"rev-{info.revision}"
        if info.revision > 1:
            assert snapshot.name == f"Live {info.revision}"


def test_profile_content_never_reaches_logs(service, repo, caplog):
    caplog.set_level(logging.DEBUG, logger="echoflux")
    service.create_profile("Name", "PROFILE-SENTINEL", profile_id="p-1")
    service.update_profile("p-1", content="PROFILE-SENTINEL-2")
    service.start_session(SessionRecord("s-1", "c1", 1.0), profile_ids=["p-1"])
    service.snapshots_for_outbound("s-1", "summary")
    service.delete_profile("p-1")
    assert "PROFILE-SENTINEL" not in caplog.text
