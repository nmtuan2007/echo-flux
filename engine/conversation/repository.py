"""Local SQLite storage for conversations, sessions, transcripts, Assist results and summaries.

Everything here is blocking I/O. The repository is a synchronous component meant to be
called from worker threads (or via ``run_in_executor``), never directly on the asyncio loop.

Thread safety: one connection guarded by a re-entrant lock. Every write runs inside
``transaction()``, which opens ``BEGIN IMMEDIATE`` at the outermost level and a SAVEPOINT
when nested, so a failure anywhere rolls back the whole unit and no partial conversation,
session, or snapshot is ever committed. Several repositories on the same file are also safe:
SQLite's database lock plus ``busy_timeout`` serializes them.

Storage is local only. No raw audio, credential values, or credential references are stored.
"""

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from engine.conversation.models import (
    DEFAULT_SOURCE_LABELS,
    SOURCES,
    ConversationRecord,
    ConversationValidationError,
    ProviderMetadata,
    SessionRecord,
    StoredAssistResult,
    StoredSummary,
    TranscriptMessage,
    reject_secret_keys,
    validate_record_id,
)
from engine.core.config import Config
from engine.core.logging import get_logger
from engine.providers.types import (
    AssistQuestion,
    AssistResult,
    AssistSuggestion,
    SummaryResult,
    SummaryWarning,
)

logger = get_logger("conversation.repository")

DB_FILENAME = "conversations.sqlite3"
STORE_SCHEMA_VERSION = 1
EXPORT_FORMAT = "echoflux.conversation-export"
EXPORT_FORMAT_VERSION = 1
BUSY_TIMEOUT_MS = 5000


class RepositoryError(Exception):
    """Storage failure. Carries a location and fixed code, never stored content."""

    def __init__(self, path: str, code: str) -> None:
        super().__init__(f"{path}: {code}")
        self.path = path
        self.code = code


class RecordNotFoundError(RepositoryError, LookupError):
    pass


class RecordConflictError(RepositoryError):
    pass


class UnsupportedSchemaError(RepositoryError):
    """The database was written by a newer EchoFlux. It is left untouched."""


@dataclass(frozen=True)
class ImportOutcome:
    conversation_id: str
    imported: bool


def default_database_path(config: Optional[Config] = None) -> Path:
    if config is None:
        config = Config()
    return config.data_dir / DB_FILENAME


def _now_ms() -> float:
    return time.time() * 1000.0


def new_record_id() -> str:
    return uuid.uuid4().hex


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --- schema ----------------------------------------------------------------------------------

_SCHEMA_V1 = (
    """
    CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at_ms REAL NOT NULL,
        updated_at_ms REAL NOT NULL,
        source_labels TEXT NOT NULL,
        import_id TEXT UNIQUE,
        record_version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        started_at_ms REAL NOT NULL,
        ended_at_ms REAL,
        providers TEXT NOT NULL,
        record_version INTEGER NOT NULL
    )
    """,
    "CREATE INDEX sessions_by_conversation ON sessions(conversation_id, started_at_ms)",
    """
    CREATE TABLE messages (
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        message_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
        source TEXT NOT NULL,
        text TEXT NOT NULL,
        translation TEXT,
        is_final INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        timestamp_ms REAL NOT NULL,
        start_ms REAL,
        end_ms REAL,
        detected_language TEXT,
        provider_event_id TEXT,
        stt_provider TEXT,
        translation_provider TEXT,
        record_version INTEGER NOT NULL,
        PRIMARY KEY (conversation_id, message_id),
        UNIQUE (conversation_id, sequence)
    )
    """,
    """
    CREATE TABLE assist_results (
        assist_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        target_message_id TEXT NOT NULL,
        payload TEXT NOT NULL,
        provider TEXT,
        provider_event_id TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        latency_ms REAL,
        created_at_ms REAL NOT NULL,
        record_version INTEGER NOT NULL,
        FOREIGN KEY (conversation_id, target_message_id)
            REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX assist_by_conversation ON assist_results(conversation_id, created_at_ms)",
    """
    CREATE TABLE summaries (
        summary_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        summary_markdown TEXT NOT NULL,
        payload TEXT NOT NULL,
        warning TEXT,
        provider TEXT,
        provider_event_id TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        latency_ms REAL,
        created_at_ms REAL NOT NULL,
        record_version INTEGER NOT NULL
    )
    """,
    "CREATE INDEX summaries_by_conversation ON summaries(conversation_id, created_at_ms)",
    """
    CREATE TABLE context_profiles (
        profile_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        content_format TEXT NOT NULL CHECK (content_format IN ('text', 'markdown')),
        revision INTEGER NOT NULL,
        created_at_ms REAL NOT NULL,
        updated_at_ms REAL NOT NULL,
        record_version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE conversation_profile_attachments (
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL
            REFERENCES context_profiles(profile_id) ON DELETE CASCADE,
        position INTEGER NOT NULL,
        PRIMARY KEY (conversation_id, profile_id)
    )
    """,
    # No foreign key to context_profiles on purpose: deleting a profile must not reach
    # snapshots that past sessions already used.
    """
    CREATE TABLE session_profile_snapshots (
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        position INTEGER NOT NULL,
        profile_id TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        content_format TEXT NOT NULL,
        profile_revision INTEGER NOT NULL,
        content_sha256 TEXT NOT NULL,
        captured_at_ms REAL NOT NULL,
        record_version INTEGER NOT NULL,
        PRIMARY KEY (session_id, position),
        UNIQUE (session_id, profile_id)
    )
    """,
    """
    CREATE TRIGGER session_profile_snapshots_immutable
    BEFORE UPDATE ON session_profile_snapshots
    BEGIN
        SELECT RAISE(ABORT, 'snapshot_immutable');
    END
    """,
)


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_V1:
        conn.execute(statement)


# Migration hook: append (version, step) for each schema change. Steps run in order inside the
# same transaction that bumps PRAGMA user_version, so a failed step leaves the old schema intact.
MIGRATIONS: Tuple[Tuple[int, Callable[[sqlite3.Connection], None]], ...] = ((1, _migrate_to_v1),)


# --- row mapping ------------------------------------------------------------------------------


def _provider_to_json(provider: Optional[ProviderMetadata]) -> Optional[str]:
    return None if provider is None else _dumps(provider.to_dict())


def _provider_from_json(text: Optional[str]) -> Optional[ProviderMetadata]:
    return None if text is None else ProviderMetadata.from_dict(json.loads(text))


def _conversation_from_row(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        title=row["title"],
        created_at_ms=row["created_at_ms"],
        updated_at_ms=row["updated_at_ms"],
        source_labels=json.loads(row["source_labels"]),
        import_id=row["import_id"],
        record_version=row["record_version"],
    )


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    providers = {
        service: ProviderMetadata.from_dict(data)
        for service, data in json.loads(row["providers"]).items()
    }
    return SessionRecord(
        session_id=row["session_id"],
        conversation_id=row["conversation_id"],
        started_at_ms=row["started_at_ms"],
        ended_at_ms=row["ended_at_ms"],
        providers=providers,
        record_version=row["record_version"],
    )


def _message_from_row(row: sqlite3.Row) -> TranscriptMessage:
    return TranscriptMessage(
        message_id=row["message_id"],
        source=row["source"],
        text=row["text"],
        timestamp_ms=row["timestamp_ms"],
        is_final=bool(row["is_final"]),
        revision=row["revision"],
        translation=row["translation"],
        session_id=row["session_id"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        detected_language=row["detected_language"],
        provider_event_id=row["provider_event_id"],
        stt_provider=_provider_from_json(row["stt_provider"]),
        translation_provider=_provider_from_json(row["translation_provider"]),
        sequence=row["sequence"],
        record_version=row["record_version"],
    )


def _assist_payload(record: StoredAssistResult) -> Dict[str, Any]:
    return {
        "suggested_replies": [
            {"strategy": s.strategy, "text": s.text, "reasoning": s.reasoning}
            for s in record.suggested_replies
        ],
        "clarifying_questions": [
            {"question": q.question, "purpose": q.purpose} for q in record.clarifying_questions
        ],
    }


def _assist_from_row(row: sqlite3.Row) -> StoredAssistResult:
    payload = json.loads(row["payload"])
    return StoredAssistResult(
        assist_id=row["assist_id"],
        conversation_id=row["conversation_id"],
        target_message_id=row["target_message_id"],
        suggested_replies=tuple(AssistSuggestion(**s) for s in payload["suggested_replies"]),
        clarifying_questions=tuple(AssistQuestion(**q) for q in payload["clarifying_questions"]),
        created_at_ms=row["created_at_ms"],
        provider=_provider_from_json(row["provider"]),
        provider_event_id=row["provider_event_id"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        latency_ms=row["latency_ms"],
        record_version=row["record_version"],
    )


_SUMMARY_LISTS = ("bullet_points", "decisions", "action_items", "risks")


def _warning_to_dict(warning: SummaryWarning) -> Dict[str, Any]:
    return {
        "omitted_message_count": warning.omitted_message_count,
        "first_omitted_timestamp_ms": warning.first_omitted_timestamp_ms,
        "last_omitted_timestamp_ms": warning.last_omitted_timestamp_ms,
        "reason": warning.reason,
    }


def _summary_from_row(row: sqlite3.Row) -> StoredSummary:
    payload = json.loads(row["payload"])
    warning = None
    if row["warning"] is not None:
        warning = SummaryWarning(**json.loads(row["warning"]))
    return StoredSummary(
        summary_id=row["summary_id"],
        conversation_id=row["conversation_id"],
        summary_markdown=row["summary_markdown"],
        created_at_ms=row["created_at_ms"],
        warning=warning,
        provider=_provider_from_json(row["provider"]),
        provider_event_id=row["provider_event_id"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        latency_ms=row["latency_ms"],
        record_version=row["record_version"],
        **{name: tuple(payload[name]) for name in _SUMMARY_LISTS},
    )


# --- repository -------------------------------------------------------------------------------


class ConversationRepository:
    """Synchronous, thread-safe SQLite repository. Call from worker threads only."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is None:
            db_path = default_database_path()
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._depth = 0
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,
            check_same_thread=False,
            timeout=BUSY_TIMEOUT_MS / 1000.0,
        )
        conn.row_factory = sqlite3.Row
        self._conn: Optional[sqlite3.Connection] = conn
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            # Migrate before switching journal mode so a newer database is left untouched.
            self._migrate()
            conn.execute("PRAGMA journal_mode = WAL")
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_config(cls, config: Config) -> "ConversationRepository":
        return cls(default_database_path(config))

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def schema_version(self) -> int:
        with self.read() as conn:
            return conn.execute("PRAGMA user_version").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "ConversationRepository":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _require_open(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RepositoryError("repository", "closed")
        return self._conn

    def _migrate(self) -> None:
        with self.transaction() as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current > STORE_SCHEMA_VERSION:
                raise UnsupportedSchemaError("schema_version", "unsupported_schema_version")
            for version, step in MIGRATIONS:
                if version > current:
                    step(conn)
                    conn.execute(f"PRAGMA user_version = {int(version)}")
            if current < STORE_SCHEMA_VERSION:
                logger.info(
                    "Conversation store migrated from schema %d to %d",
                    current,
                    STORE_SCHEMA_VERSION,
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing unit of work. Nested calls become savepoints of the outer one."""
        with self._lock:
            conn = self._require_open()
            depth = self._depth
            savepoint = f"sp_{depth}"
            conn.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT {savepoint}")
            self._depth += 1
            try:
                yield conn
            except BaseException:
                self._depth -= 1
                if depth == 0:
                    conn.execute("ROLLBACK")
                else:
                    conn.execute(f"ROLLBACK TO {savepoint}")
                    conn.execute(f"RELEASE {savepoint}")
                raise
            self._depth -= 1
            try:
                conn.execute("COMMIT" if depth == 0 else f"RELEASE {savepoint}")
            except BaseException:
                if depth == 0 and conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._require_open()

    # --- conversations ------------------------------------------------------------------------

    def _insert_conversation(self, conn: sqlite3.Connection, record: ConversationRecord) -> None:
        try:
            conn.execute(
                "INSERT INTO conversations (conversation_id, title, created_at_ms, updated_at_ms,"
                " source_labels, import_id, record_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.conversation_id,
                    record.title,
                    record.created_at_ms,
                    record.updated_at_ms,
                    _dumps(record.source_labels),
                    record.import_id,
                    record.record_version,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RecordConflictError("conversation_id", "duplicate_id") from exc

    def create_conversation(self, record: ConversationRecord) -> ConversationRecord:
        with self.transaction() as conn:
            self._insert_conversation(conn, record)
        logger.debug("Created conversation %s", record.conversation_id)
        return record

    def require_conversation(
        self, conn: sqlite3.Connection, conversation_id: str
    ) -> ConversationRecord:
        validate_record_id("conversation_id", conversation_id)
        row = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError("conversation_id", "not_found")
        return _conversation_from_row(row)

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        with self.read() as conn:
            try:
                return self.require_conversation(conn, conversation_id)
            except RecordNotFoundError:
                return None

    def list_conversations(self) -> List[ConversationRecord]:
        with self.read() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY created_at_ms DESC, conversation_id"
            ).fetchall()
        return [_conversation_from_row(row) for row in rows]

    def list_conversations_with_stats(
        self, search_query: Optional[str] = None
    ) -> List[Tuple[ConversationRecord, int, bool]]:
        """List conversations along with message counts and summary presence in one query."""
        query = (search_query or "").strip().lower()
        params: Tuple[Any, ...] = ()
        where_clause = ""
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            where_clause = (
                " WHERE LOWER(c.title) LIKE ? ESCAPE '\\'"
                " OR EXISTS ("
                "SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id"
                " AND (LOWER(m.text) LIKE ? ESCAPE '\\' OR LOWER(COALESCE(m.translation, '')) LIKE ? ESCAPE '\\')"
                ")"
            )
            params = (pattern, pattern, pattern)

        with self.read() as conn:
            rows = conn.execute(
                "SELECT c.*,"
                " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count,"
                " EXISTS(SELECT 1 FROM summaries s WHERE s.conversation_id = c.conversation_id) AS has_summary"
                " FROM conversations c"
                f"{where_clause}"
                " ORDER BY c.created_at_ms DESC, c.conversation_id",
                params,
            ).fetchall()
        return [
            (_conversation_from_row(row), int(row["message_count"]), bool(row["has_summary"]))
            for row in rows
        ]

    def _touch(self, conn: sqlite3.Connection, conversation_id: str, now_ms: float) -> None:
        conn.execute(
            "UPDATE conversations SET updated_at_ms = MAX(updated_at_ms, ?)"
            " WHERE conversation_id = ?",
            (now_ms, conversation_id),
        )

    def _update_conversation(
        self, conversation_id: str, now_ms: Optional[float], **changes: Any
    ) -> ConversationRecord:
        with self.transaction() as conn:
            current = self.require_conversation(conn, conversation_id)
            now = _now_ms() if now_ms is None else now_ms
            updated = replace(current, updated_at_ms=max(current.updated_at_ms, now), **changes)
            conn.execute(
                "UPDATE conversations SET title = ?, source_labels = ?, updated_at_ms = ?"
                " WHERE conversation_id = ?",
                (
                    updated.title,
                    _dumps(updated.source_labels),
                    updated.updated_at_ms,
                    conversation_id,
                ),
            )
        return updated

    def rename_conversation(
        self, conversation_id: str, title: str, now_ms: Optional[float] = None
    ) -> ConversationRecord:
        return self._update_conversation(conversation_id, now_ms, title=title)

    def set_source_label(
        self, conversation_id: str, source: str, label: str, now_ms: Optional[float] = None
    ) -> ConversationRecord:
        """Rename a speaker label (e.g. ``mic`` -> "Alice"). Messages keep their source."""
        if source not in SOURCES:
            raise ConversationValidationError("source", "unknown_source")
        current = self.get_conversation(conversation_id)
        base = dict(DEFAULT_SOURCE_LABELS)
        if current is not None:
            base.update(current.source_labels)
        base[source] = label
        return self._update_conversation(conversation_id, now_ms, source_labels=base)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete one conversation and everything scoped to it, atomically.

        Sessions, messages, Assist results, summaries, attachments and session snapshots
        cascade. Other conversations and reusable Context Profiles are untouched.
        """
        validate_record_id("conversation_id", conversation_id)
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )
            deleted = cursor.rowcount > 0
        logger.debug("Delete conversation %s: %s", conversation_id, deleted)
        return deleted

    # --- sessions -----------------------------------------------------------------------------

    def create_session(self, session: SessionRecord) -> SessionRecord:
        with self.transaction() as conn:
            self.require_conversation(conn, session.conversation_id)
            providers = {service: meta.to_dict() for service, meta in session.providers.items()}
            try:
                conn.execute(
                    "INSERT INTO sessions (session_id, conversation_id, started_at_ms,"
                    " ended_at_ms, providers, record_version) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session.session_id,
                        session.conversation_id,
                        session.started_at_ms,
                        session.ended_at_ms,
                        _dumps(providers),
                        session.record_version,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RecordConflictError("session_id", "duplicate_id") from exc
            self._touch(conn, session.conversation_id, session.started_at_ms)
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        validate_record_id("session_id", session_id)
        with self.read() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return None if row is None else _session_from_row(row)

    def list_sessions(self, conversation_id: str) -> List[SessionRecord]:
        validate_record_id("conversation_id", conversation_id)
        with self.read() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE conversation_id = ?"
                " ORDER BY started_at_ms, session_id",
                (conversation_id,),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def end_session(self, session_id: str, ended_at_ms: Optional[float] = None) -> SessionRecord:
        """Record the Stop time. Only the end time changes; provider choices are fixed."""
        validate_record_id("session_id", session_id)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("session_id", "not_found")
            current = _session_from_row(row)
            if current.ended_at_ms is not None:
                return current
            ended = replace(current, ended_at_ms=_now_ms() if ended_at_ms is None else ended_at_ms)
            conn.execute(
                "UPDATE sessions SET ended_at_ms = ? WHERE session_id = ?",
                (ended.ended_at_ms, session_id),
            )
            self._touch(conn, current.conversation_id, ended.ended_at_ms)
        return ended

    # --- messages -----------------------------------------------------------------------------

    def _insert_message(
        self, conn: sqlite3.Connection, conversation_id: str, message: TranscriptMessage
    ) -> None:
        try:
            conn.execute(
                "INSERT INTO messages (conversation_id, message_id, sequence, session_id, source,"
                " text, translation, is_final, revision, timestamp_ms, start_ms, end_ms,"
                " detected_language, provider_event_id, stt_provider, translation_provider,"
                " record_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    message.message_id,
                    message.sequence,
                    message.session_id,
                    message.source,
                    message.text,
                    message.translation,
                    int(message.is_final),
                    message.revision,
                    message.timestamp_ms,
                    message.start_ms,
                    message.end_ms,
                    message.detected_language,
                    message.provider_event_id,
                    _provider_to_json(message.stt_provider),
                    _provider_to_json(message.translation_provider),
                    message.record_version,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RecordConflictError("message_id", "duplicate_id") from exc

    def _get_message_row(
        self, conn: sqlite3.Connection, conversation_id: str, message_id: str
    ) -> Optional[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? AND message_id = ?",
            (conversation_id, message_id),
        ).fetchone()

    def upsert_message(
        self, conversation_id: str, message: TranscriptMessage, now_ms: Optional[float] = None
    ) -> TranscriptMessage:
        """Insert a new message at the end, or apply a newer revision in place.

        New messages get the next ``sequence`` inside the write transaction, so concurrent
        writers can never share or skip an order slot. A revision older than the stored one is
        ignored and the stored message is returned. An equal-or-newer revision replaces text
        and metadata but keeps its sequence; the stored translation is kept unless the text
        changed or the new revision carries its own.
        """
        with self.transaction() as conn:
            self.require_conversation(conn, conversation_id)
            if message.session_id is not None:
                owner = conn.execute(
                    "SELECT conversation_id FROM sessions WHERE session_id = ?",
                    (message.session_id,),
                ).fetchone()
                if owner is None or owner[0] != conversation_id:
                    raise RecordNotFoundError("session_id", "not_found")
            row = self._get_message_row(conn, conversation_id, message.message_id)
            if row is None:
                next_sequence = conn.execute(
                    "SELECT COALESCE(MAX(sequence) + 1, 0) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
                stored = replace(message, sequence=next_sequence)
                self._insert_message(conn, conversation_id, stored)
            else:
                existing = _message_from_row(row)
                if message.revision < existing.revision:
                    return existing
                translation = message.translation
                translation_provider = message.translation_provider
                if translation is None and message.text == existing.text:
                    translation = existing.translation
                    translation_provider = existing.translation_provider
                stored = replace(
                    message,
                    sequence=existing.sequence,
                    translation=translation,
                    translation_provider=translation_provider,
                )
                conn.execute(
                    "UPDATE messages SET session_id = ?, source = ?, text = ?, translation = ?,"
                    " is_final = ?, revision = ?, timestamp_ms = ?, start_ms = ?, end_ms = ?,"
                    " detected_language = ?, provider_event_id = ?, stt_provider = ?,"
                    " translation_provider = ?, record_version = ?"
                    " WHERE conversation_id = ? AND message_id = ?",
                    (
                        stored.session_id,
                        stored.source,
                        stored.text,
                        stored.translation,
                        int(stored.is_final),
                        stored.revision,
                        stored.timestamp_ms,
                        stored.start_ms,
                        stored.end_ms,
                        stored.detected_language,
                        stored.provider_event_id,
                        _provider_to_json(stored.stt_provider),
                        _provider_to_json(stored.translation_provider),
                        stored.record_version,
                        conversation_id,
                        stored.message_id,
                    ),
                )
            self._touch(conn, conversation_id, _now_ms() if now_ms is None else now_ms)
        return stored

    def set_translation(
        self,
        conversation_id: str,
        message_id: str,
        translation: str,
        provider: Optional[ProviderMetadata] = None,
        revision: Optional[int] = None,
        now_ms: Optional[float] = None,
    ) -> TranscriptMessage:
        """Attach an asynchronous translation. Ignored if it targets a superseded revision."""
        validate_record_id("message_id", message_id)
        with self.transaction() as conn:
            self.require_conversation(conn, conversation_id)
            row = self._get_message_row(conn, conversation_id, message_id)
            if row is None:
                raise RecordNotFoundError("message_id", "not_found")
            existing = _message_from_row(row)
            if revision is not None and revision != existing.revision:
                return existing
            stored = replace(existing, translation=translation, translation_provider=provider)
            conn.execute(
                "UPDATE messages SET translation = ?, translation_provider = ?"
                " WHERE conversation_id = ? AND message_id = ?",
                (
                    stored.translation,
                    _provider_to_json(stored.translation_provider),
                    conversation_id,
                    message_id,
                ),
            )
            self._touch(conn, conversation_id, _now_ms() if now_ms is None else now_ms)
        return stored

    def update_message(
        self,
        conversation_id: str,
        message_id: str,
        *,
        text: Optional[str] = None,
        source: Optional[str] = None,
        now_ms: Optional[float] = None,
    ) -> TranscriptMessage:
        """Edit text and/or speaker source of an existing persisted message."""
        validate_record_id("message_id", message_id)
        if text is None and source is None:
            raise ConversationValidationError("message", "nothing_to_update")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ConversationValidationError("text", "empty_text")
        if source is not None and source not in SOURCES:
            raise ConversationValidationError("source", "unknown_source")

        with self.transaction() as conn:
            self.require_conversation(conn, conversation_id)
            row = self._get_message_row(conn, conversation_id, message_id)
            if row is None:
                raise RecordNotFoundError("message_id", "not_found")
            existing = _message_from_row(row)
            stored = replace(
                existing,
                text=text.strip() if text is not None else existing.text,
                source=source if source is not None else existing.source,
            )
            conn.execute(
                "UPDATE messages SET text = ?, source = ?"
                " WHERE conversation_id = ? AND message_id = ?",
                (stored.text, stored.source, conversation_id, message_id),
            )
            self._touch(conn, conversation_id, _now_ms() if now_ms is None else now_ms)
        return stored

    def list_messages(self, conversation_id: str) -> List[TranscriptMessage]:
        validate_record_id("conversation_id", conversation_id)
        with self.read() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence",
                (conversation_id,),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    # --- Assist results and summaries ---------------------------------------------------------

    def _insert_assist(self, conn: sqlite3.Connection, record: StoredAssistResult) -> None:
        if self._get_message_row(conn, record.conversation_id, record.target_message_id) is None:
            raise RecordNotFoundError("target_message_id", "not_found")
        try:
            conn.execute(
                "INSERT INTO assist_results (assist_id, conversation_id, target_message_id,"
                " payload, provider, provider_event_id, input_tokens, output_tokens, latency_ms,"
                " created_at_ms, record_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.assist_id,
                    record.conversation_id,
                    record.target_message_id,
                    _dumps(_assist_payload(record)),
                    _provider_to_json(record.provider),
                    record.provider_event_id,
                    record.input_tokens,
                    record.output_tokens,
                    record.latency_ms,
                    record.created_at_ms,
                    record.record_version,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RecordConflictError("assist_id", "duplicate_id") from exc

    def add_assist_result(self, record: StoredAssistResult) -> StoredAssistResult:
        with self.transaction() as conn:
            self.require_conversation(conn, record.conversation_id)
            self._insert_assist(conn, record)
            self._touch(conn, record.conversation_id, record.created_at_ms)
        return record

    def save_assist_result(
        self,
        conversation_id: str,
        result: AssistResult,
        provider: Optional[ProviderMetadata] = None,
        assist_id: Optional[str] = None,
        created_at_ms: Optional[float] = None,
    ) -> StoredAssistResult:
        record = StoredAssistResult(
            assist_id=assist_id or new_record_id(),
            conversation_id=conversation_id,
            target_message_id=result.target_message_id,
            suggested_replies=tuple(result.suggested_replies),
            clarifying_questions=tuple(result.clarifying_questions),
            created_at_ms=_now_ms() if created_at_ms is None else created_at_ms,
            provider=provider,
            provider_event_id=result.provider_event_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )
        return self.add_assist_result(record)

    def list_assist_results(
        self, conversation_id: str, target_message_id: Optional[str] = None
    ) -> List[StoredAssistResult]:
        validate_record_id("conversation_id", conversation_id)
        query = "SELECT * FROM assist_results WHERE conversation_id = ?"
        params: Tuple[Any, ...] = (conversation_id,)
        if target_message_id is not None:
            query += " AND target_message_id = ?"
            params += (target_message_id,)
        with self.read() as conn:
            rows = conn.execute(query + " ORDER BY created_at_ms, assist_id", params).fetchall()
        return [_assist_from_row(row) for row in rows]

    def _insert_summary(self, conn: sqlite3.Connection, record: StoredSummary) -> None:
        payload = {name: list(getattr(record, name)) for name in _SUMMARY_LISTS}
        warning = None if record.warning is None else _dumps(_warning_to_dict(record.warning))
        try:
            conn.execute(
                "INSERT INTO summaries (summary_id, conversation_id, summary_markdown, payload,"
                " warning, provider, provider_event_id, input_tokens, output_tokens, latency_ms,"
                " created_at_ms, record_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.summary_id,
                    record.conversation_id,
                    record.summary_markdown,
                    _dumps(payload),
                    warning,
                    _provider_to_json(record.provider),
                    record.provider_event_id,
                    record.input_tokens,
                    record.output_tokens,
                    record.latency_ms,
                    record.created_at_ms,
                    record.record_version,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RecordConflictError("summary_id", "duplicate_id") from exc

    def add_summary(self, record: StoredSummary) -> StoredSummary:
        with self.transaction() as conn:
            self.require_conversation(conn, record.conversation_id)
            self._insert_summary(conn, record)
            self._touch(conn, record.conversation_id, record.created_at_ms)
        return record

    def save_summary(
        self,
        conversation_id: str,
        result: SummaryResult,
        provider: Optional[ProviderMetadata] = None,
        summary_id: Optional[str] = None,
        created_at_ms: Optional[float] = None,
    ) -> StoredSummary:
        record = StoredSummary(
            summary_id=summary_id or new_record_id(),
            conversation_id=conversation_id,
            summary_markdown=result.summary_markdown,
            created_at_ms=_now_ms() if created_at_ms is None else created_at_ms,
            bullet_points=tuple(result.bullet_points),
            decisions=tuple(result.decisions),
            action_items=tuple(result.action_items),
            risks=tuple(result.risks),
            warning=result.warning,
            provider=provider,
            provider_event_id=result.provider_event_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )
        return self.add_summary(record)

    def list_summaries(self, conversation_id: str) -> List[StoredSummary]:
        validate_record_id("conversation_id", conversation_id)
        with self.read() as conn:
            rows = conn.execute(
                "SELECT * FROM summaries WHERE conversation_id = ?"
                " ORDER BY created_at_ms, summary_id",
                (conversation_id,),
            ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def latest_summary(self, conversation_id: str) -> Optional[StoredSummary]:
        summaries = self.list_summaries(conversation_id)
        return summaries[-1] if summaries else None

    # --- import and export --------------------------------------------------------------------

    def import_conversation(
        self,
        record: ConversationRecord,
        messages: Sequence[TranscriptMessage],
        assist_results: Sequence[StoredAssistResult] = (),
        summaries: Sequence[StoredSummary] = (),
    ) -> ImportOutcome:
        """Idempotently import one externally held conversation (e.g. browser history).

        ``record.import_id`` is the idempotency key: if it already exists the call changes
        nothing and reports ``imported=False``. Otherwise the conversation, its messages in
        the given order, and its Assist results and summaries are written in one transaction.
        """
        if record.import_id is None:
            raise ConversationValidationError("import_id", "missing_import_id")
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT conversation_id FROM conversations WHERE import_id = ?",
                (record.import_id,),
            ).fetchone()
            if existing is not None:
                return ImportOutcome(conversation_id=existing[0], imported=False)
            self._insert_conversation(conn, record)
            for sequence, message in enumerate(messages):
                if message.session_id is not None:
                    raise ConversationValidationError(
                        f"messages.{sequence}.session_id", "not_importable"
                    )
                self._insert_message(
                    conn, record.conversation_id, replace(message, sequence=sequence)
                )
            for assist in assist_results:
                if assist.conversation_id != record.conversation_id:
                    raise ConversationValidationError("assist_results", "wrong_conversation")
                self._insert_assist(conn, assist)
            for summary in summaries:
                if summary.conversation_id != record.conversation_id:
                    raise ConversationValidationError("summaries", "wrong_conversation")
                self._insert_summary(conn, summary)
        logger.info(
            "Imported conversation %s with %d messages", record.conversation_id, len(messages)
        )
        return ImportOutcome(conversation_id=record.conversation_id, imported=True)

    def export_conversation(
        self, conversation_id: str, exported_at_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """JSON-ready export of exactly one conversation.

        Built from an explicit field whitelist inside a single read, so it never includes
        other conversations, endpoints, credential values or credential references. Context
        Profile snapshots are listed by id, name and revision only; their content is read
        solely by explicit Assist/Summary calls.
        """
        with self.read() as conn:
            conversation = self.require_conversation(conn, conversation_id)
            session_rows = conn.execute(
                "SELECT * FROM sessions WHERE conversation_id = ?"
                " ORDER BY started_at_ms, session_id",
                (conversation_id,),
            ).fetchall()
            snapshot_rows = conn.execute(
                "SELECT s.session_id, s.position, s.profile_id, s.name, s.content_format,"
                " s.profile_revision, s.captured_at_ms FROM session_profile_snapshots s"
                " JOIN sessions ON sessions.session_id = s.session_id"
                " WHERE sessions.conversation_id = ? ORDER BY s.session_id, s.position",
                (conversation_id,),
            ).fetchall()
            message_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence",
                (conversation_id,),
            ).fetchall()
            assist_rows = conn.execute(
                "SELECT * FROM assist_results WHERE conversation_id = ?"
                " ORDER BY created_at_ms, assist_id",
                (conversation_id,),
            ).fetchall()
            summary_rows = conn.execute(
                "SELECT * FROM summaries WHERE conversation_id = ?"
                " ORDER BY created_at_ms, summary_id",
                (conversation_id,),
            ).fetchall()

        snapshots_by_session: Dict[str, List[Dict[str, Any]]] = {}
        for row in snapshot_rows:
            snapshots_by_session.setdefault(row["session_id"], []).append(
                {
                    "position": row["position"],
                    "profile_id": row["profile_id"],
                    "name": row["name"],
                    "content_format": row["content_format"],
                    "profile_revision": row["profile_revision"],
                    "captured_at_ms": row["captured_at_ms"],
                }
            )

        def provider(meta: Optional[ProviderMetadata]) -> Optional[Dict[str, Any]]:
            return None if meta is None else meta.to_dict()

        sessions = []
        for row in session_rows:
            session = _session_from_row(row)
            sessions.append(
                {
                    "session_id": session.session_id,
                    "started_at_ms": session.started_at_ms,
                    "ended_at_ms": session.ended_at_ms,
                    "providers": {k: v.to_dict() for k, v in session.providers.items()},
                    "context_profiles": snapshots_by_session.get(session.session_id, []),
                }
            )

        messages = []
        for row in message_rows:
            m = _message_from_row(row)
            messages.append(
                {
                    "message_id": m.message_id,
                    "sequence": m.sequence,
                    "session_id": m.session_id,
                    "source": m.source,
                    "speaker": conversation.label_for(m.source),
                    "text": m.text,
                    "translation": m.translation,
                    "is_final": m.is_final,
                    "revision": m.revision,
                    "timestamp_ms": m.timestamp_ms,
                    "start_ms": m.start_ms,
                    "end_ms": m.end_ms,
                    "detected_language": m.detected_language,
                    "stt_provider": provider(m.stt_provider),
                    "translation_provider": provider(m.translation_provider),
                }
            )

        assists = []
        for row in assist_rows:
            a = _assist_from_row(row)
            assists.append(
                {
                    "assist_id": a.assist_id,
                    "target_message_id": a.target_message_id,
                    "created_at_ms": a.created_at_ms,
                    "provider": provider(a.provider),
                    **_assist_payload(a),
                }
            )

        summaries = []
        for row in summary_rows:
            s = _summary_from_row(row)
            summaries.append(
                {
                    "summary_id": s.summary_id,
                    "created_at_ms": s.created_at_ms,
                    "summary_markdown": s.summary_markdown,
                    "provider": provider(s.provider),
                    "warning": None if s.warning is None else _warning_to_dict(s.warning),
                    **{name: list(getattr(s, name)) for name in _SUMMARY_LISTS},
                }
            )

        export = {
            "format": EXPORT_FORMAT,
            "format_version": EXPORT_FORMAT_VERSION,
            "schema_version": STORE_SCHEMA_VERSION,
            "exported_at_ms": _now_ms() if exported_at_ms is None else exported_at_ms,
            "conversation": {
                "conversation_id": conversation.conversation_id,
                "title": conversation.title,
                "created_at_ms": conversation.created_at_ms,
                "updated_at_ms": conversation.updated_at_ms,
                "source_labels": dict(conversation.source_labels),
            },
            "sessions": sessions,
            "messages": messages,
            "assist_results": assists,
            "summaries": summaries,
        }
        # Defense in depth: the whitelist above should make this unreachable.
        reject_secret_keys("export", export)
        return export
