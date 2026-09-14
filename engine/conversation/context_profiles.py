"""Reusable Context Profiles, conversation attachments, and immutable session snapshots.

Profiles are Text/Markdown only, bounded by the explicit limits in ``models``. When a session
starts, its attached profiles are copied into ``session_profile_snapshots`` in the same
transaction that creates the session: either the session and every snapshot exist, or
neither does. Snapshots are never updated (a database trigger rejects it), and they hold
their own copy of the content, so editing or deleting a profile cannot change what a past
session used.

Profile content leaves this module for provider use only through ``snapshots_for_outbound``,
which requires an explicit Assist or Summary purpose. Listings carry metadata, not content.
Like the repository, every call here is blocking and belongs off the asyncio loop.
"""

import hashlib
import sqlite3
import time
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from engine.conversation.models import (
    MAX_PROFILES_PER_SESSION,
    MAX_SESSION_PROFILE_BYTES,
    OUTBOUND_PURPOSES,
    PROFILE_FORMAT_MARKDOWN,
    ContextProfile,
    ContextProfileInfo,
    ConversationValidationError,
    SessionProfileSnapshot,
    SessionRecord,
    validate_record_id,
)
from engine.conversation.repository import (
    ConversationRepository,
    RecordConflictError,
    RecordNotFoundError,
    new_record_id,
)
from engine.core.logging import get_logger
from engine.providers.types import ContextProfileSnapshot

logger = get_logger("conversation.context_profiles")


def _now_ms() -> float:
    return time.time() * 1000.0


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _profile_from_row(row: sqlite3.Row) -> ContextProfile:
    return ContextProfile(
        profile_id=row["profile_id"],
        name=row["name"],
        content=row["content"],
        content_format=row["content_format"],
        revision=row["revision"],
        created_at_ms=row["created_at_ms"],
        updated_at_ms=row["updated_at_ms"],
        record_version=row["record_version"],
    )


def _snapshot_from_row(row: sqlite3.Row) -> SessionProfileSnapshot:
    return SessionProfileSnapshot(
        session_id=row["session_id"],
        position=row["position"],
        profile_id=row["profile_id"],
        name=row["name"],
        content=row["content"],
        content_format=row["content_format"],
        profile_revision=row["profile_revision"],
        content_sha256=row["content_sha256"],
        captured_at_ms=row["captured_at_ms"],
        record_version=row["record_version"],
    )


def _validate_profile_ids(profile_ids: Sequence[str]) -> Tuple[str, ...]:
    ids = tuple(profile_ids)
    for i, profile_id in enumerate(ids):
        validate_record_id(f"profile_ids.{i}", profile_id)
    if len(set(ids)) != len(ids):
        raise ConversationValidationError("profile_ids", "duplicate_profile")
    if len(ids) > MAX_PROFILES_PER_SESSION:
        raise ConversationValidationError("profile_ids", "too_many_profiles")
    return ids


class ContextProfileService:
    """Profile CRUD, attachment, and snapshot-at-session-start on a shared repository."""

    def __init__(self, repository: ConversationRepository) -> None:
        self._repo = repository

    # --- profile CRUD -------------------------------------------------------------------------

    def create_profile(
        self,
        name: str,
        content: str,
        content_format: str = PROFILE_FORMAT_MARKDOWN,
        profile_id: Optional[str] = None,
        now_ms: Optional[float] = None,
    ) -> ContextProfile:
        now = _now_ms() if now_ms is None else now_ms
        profile = ContextProfile(
            profile_id=profile_id or new_record_id(),
            name=name,
            content=content,
            content_format=content_format,
            revision=1,
            created_at_ms=now,
            updated_at_ms=now,
        )
        with self._repo.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM context_profiles WHERE profile_id = ?", (profile.profile_id,)
            ).fetchone()
            if exists is not None:
                raise RecordConflictError("profile_id", "duplicate_id")
            conn.execute(
                "INSERT INTO context_profiles (profile_id, name, content, content_format,"
                " revision, created_at_ms, updated_at_ms, record_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile.profile_id,
                    profile.name,
                    profile.content,
                    profile.content_format,
                    profile.revision,
                    profile.created_at_ms,
                    profile.updated_at_ms,
                    profile.record_version,
                ),
            )
        logger.debug("Created context profile %s", profile.profile_id)
        return profile

    def get_profile(self, profile_id: str) -> Optional[ContextProfile]:
        """Full profile for local editing. Not a path to providers; see snapshots_for_outbound."""
        validate_record_id("profile_id", profile_id)
        with self._repo.read() as conn:
            row = conn.execute(
                "SELECT * FROM context_profiles WHERE profile_id = ?", (profile_id,)
            ).fetchone()
        return None if row is None else _profile_from_row(row)

    def list_profiles(self) -> List[ContextProfileInfo]:
        with self._repo.read() as conn:
            rows = conn.execute(
                "SELECT profile_id, name, content_format, revision, updated_at_ms,"
                " length(CAST(content AS BLOB)) AS content_bytes FROM context_profiles"
                " ORDER BY name COLLATE NOCASE, profile_id"
            ).fetchall()
        return [
            ContextProfileInfo(
                profile_id=row["profile_id"],
                name=row["name"],
                content_format=row["content_format"],
                revision=row["revision"],
                content_bytes=row["content_bytes"],
                updated_at_ms=row["updated_at_ms"],
            )
            for row in rows
        ]

    def update_profile(
        self,
        profile_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        content_format: Optional[str] = None,
        expected_revision: Optional[int] = None,
        now_ms: Optional[float] = None,
    ) -> ContextProfile:
        """Edit a profile and bump its revision. Existing session snapshots are unaffected.

        ``expected_revision`` gives optimistic concurrency: a stale editor gets a conflict
        instead of silently overwriting a newer edit.
        """
        validate_record_id("profile_id", profile_id)
        with self._repo.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM context_profiles WHERE profile_id = ?", (profile_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("profile_id", "not_found")
            current = _profile_from_row(row)
            if expected_revision is not None and expected_revision != current.revision:
                raise RecordConflictError("profile.revision", "stale_revision")
            now = _now_ms() if now_ms is None else now_ms
            updated = replace(
                current,
                name=current.name if name is None else name,
                content=current.content if content is None else content,
                content_format=(
                    current.content_format if content_format is None else content_format
                ),
                revision=current.revision + 1,
                updated_at_ms=max(current.updated_at_ms, now),
            )
            conn.execute(
                "UPDATE context_profiles SET name = ?, content = ?, content_format = ?,"
                " revision = ?, updated_at_ms = ? WHERE profile_id = ?",
                (
                    updated.name,
                    updated.content,
                    updated.content_format,
                    updated.revision,
                    updated.updated_at_ms,
                    profile_id,
                ),
            )
        logger.debug("Updated context profile %s to revision %d", profile_id, updated.revision)
        return updated

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile and its attachments. Past session snapshots keep their copy."""
        validate_record_id("profile_id", profile_id)
        with self._repo.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM context_profiles WHERE profile_id = ?", (profile_id,)
            )
            deleted = cursor.rowcount > 0
        logger.debug("Delete context profile %s: %s", profile_id, deleted)
        return deleted

    # --- attachments --------------------------------------------------------------------------

    def set_attached_profiles(self, conversation_id: str, profile_ids: Sequence[str]) -> None:
        """Replace a conversation's ordered attachment list atomically."""
        ids = _validate_profile_ids(profile_ids)
        with self._repo.transaction() as conn:
            self._repo.require_conversation(conn, conversation_id)
            conn.execute(
                "DELETE FROM conversation_profile_attachments WHERE conversation_id = ?",
                (conversation_id,),
            )
            for position, profile_id in enumerate(ids):
                exists = conn.execute(
                    "SELECT 1 FROM context_profiles WHERE profile_id = ?", (profile_id,)
                ).fetchone()
                if exists is None:
                    raise RecordNotFoundError(f"profile_ids.{position}", "not_found")
                conn.execute(
                    "INSERT INTO conversation_profile_attachments"
                    " (conversation_id, profile_id, position) VALUES (?, ?, ?)",
                    (conversation_id, profile_id, position),
                )

    def attached_profile_ids(self, conversation_id: str) -> List[str]:
        validate_record_id("conversation_id", conversation_id)
        with self._repo.read() as conn:
            rows = conn.execute(
                "SELECT profile_id FROM conversation_profile_attachments"
                " WHERE conversation_id = ? ORDER BY position, profile_id",
                (conversation_id,),
            ).fetchall()
        return [row["profile_id"] for row in rows]

    # --- sessions and snapshots ---------------------------------------------------------------

    def start_session(
        self,
        session: SessionRecord,
        profile_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[SessionProfileSnapshot, ...]:
        """Create ``session`` and snapshot its profiles in one transaction.

        Uses ``profile_ids`` when given, otherwise the conversation's attachments at this
        moment. A missing profile or an over-limit total rolls back the session too.
        """
        with self._repo.transaction() as conn:
            self._repo.create_session(session)
            if profile_ids is None:
                rows = conn.execute(
                    "SELECT profile_id FROM conversation_profile_attachments"
                    " WHERE conversation_id = ? ORDER BY position, profile_id",
                    (session.conversation_id,),
                ).fetchall()
                profile_ids = [row["profile_id"] for row in rows]
            ids = _validate_profile_ids(profile_ids)

            snapshots = []
            total_bytes = 0
            for position, profile_id in enumerate(ids):
                row = conn.execute(
                    "SELECT * FROM context_profiles WHERE profile_id = ?", (profile_id,)
                ).fetchone()
                if row is None:
                    raise RecordNotFoundError(f"profile_ids.{position}", "not_found")
                profile = _profile_from_row(row)
                total_bytes += len(profile.content.encode("utf-8"))
                if total_bytes > MAX_SESSION_PROFILE_BYTES:
                    raise ConversationValidationError("profile_ids", "session_profiles_too_large")
                snapshot = SessionProfileSnapshot(
                    session_id=session.session_id,
                    position=position,
                    profile_id=profile.profile_id,
                    name=profile.name,
                    content=profile.content,
                    content_format=profile.content_format,
                    profile_revision=profile.revision,
                    content_sha256=_content_sha256(profile.content),
                    captured_at_ms=session.started_at_ms,
                )
                conn.execute(
                    "INSERT INTO session_profile_snapshots (session_id, position, profile_id,"
                    " name, content, content_format, profile_revision, content_sha256,"
                    " captured_at_ms, record_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.session_id,
                        snapshot.position,
                        snapshot.profile_id,
                        snapshot.name,
                        snapshot.content,
                        snapshot.content_format,
                        snapshot.profile_revision,
                        snapshot.content_sha256,
                        snapshot.captured_at_ms,
                        snapshot.record_version,
                    ),
                )
                snapshots.append(snapshot)
        logger.info(
            "Session %s started with %d context profile snapshot(s)",
            session.session_id,
            len(snapshots),
        )
        return tuple(snapshots)

    def snapshot_info(self, session_id: str) -> List[ContextProfileInfo]:
        """Which profiles a session captured, without their content."""
        validate_record_id("session_id", session_id)
        with self._repo.read() as conn:
            rows = conn.execute(
                "SELECT profile_id, name, content_format, profile_revision, captured_at_ms,"
                " length(CAST(content AS BLOB)) AS content_bytes"
                " FROM session_profile_snapshots WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        return [
            ContextProfileInfo(
                profile_id=row["profile_id"],
                name=row["name"],
                content_format=row["content_format"],
                revision=row["profile_revision"],
                content_bytes=row["content_bytes"],
                updated_at_ms=row["captured_at_ms"],
            )
            for row in rows
        ]

    def snapshots_for_outbound(
        self, session_id: str, purpose: str
    ) -> Tuple[ContextProfileSnapshot, ...]:
        """Session snapshot content for an explicit Assist or Summary request.

        Integrity is checked against the hash taken at capture; a mismatch means the row was
        altered outside this service and the content is withheld rather than sent.
        """
        if purpose not in OUTBOUND_PURPOSES:
            raise ConversationValidationError("purpose", "unsupported_purpose")
        validate_record_id("session_id", session_id)
        with self._repo.read() as conn:
            rows = conn.execute(
                "SELECT * FROM session_profile_snapshots WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        snapshots = [_snapshot_from_row(row) for row in rows]
        for snapshot in snapshots:
            if _content_sha256(snapshot.content) != snapshot.content_sha256:
                raise RecordConflictError("session_profile_snapshots", "integrity_mismatch")
        logger.debug(
            "Released %d profile snapshot(s) of session %s for %s",
            len(snapshots),
            session_id,
            purpose,
        )
        return tuple(snapshot.to_provider_snapshot() for snapshot in snapshots)
