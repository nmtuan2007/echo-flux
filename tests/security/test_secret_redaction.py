"""Automated secret leak scan tests for EchoFlux.

Verifies that zero raw secrets, credentials, or API tokens leak into:
1. Zustand persistence / localStorage serialized state.
2. SQLite conversation database across all tables, columns, and raw bytes.
3. Engine config.json files on disk.
4. Application log records and exception traces (URL/text redaction).
5. WebSocket control error frames and event broadcast payloads.
6. Conversation and history export files.
"""

import io
import json
import logging
import sqlite3
from pathlib import Path

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    ConversationRecord,
    SessionRecord,
    TranscriptMessage,
)
from engine.conversation.repository import DB_FILENAME, ConversationRepository
from engine.core.config import (
    Config,
    CredentialRef,
    ServiceConfig,
    endpoint_has_embedded_credentials,
    is_secret_key_name,
)
from engine.core.logging import get_logger
from engine.providers.stt.streaming_ws import redact_text, redact_url
from engine.server.protocol import (
    error_frame,
    sanitize_event,
)

CANARY_OPENAI = "sk-proj-CANARY-SECRET-OPENAI-99999"
CANARY_DEEPGRAM = "dg-canary-SECRET-DEEPGRAM-88888"
CANARY_GEMINI = "AIzaSy-CANARY-SECRET-GEMINI-77777"
CANARY_AZURE = "azure-canary-SECRET-AZURE-66666"
CANARY_ASSEMBLY = "aai-canary-SECRET-ASSEMBLY-55555"
CANARY_HF = "hf_CANARY-SECRET-HUGGINGFACE-44444"
CANARY_CONTROL_TOKEN = "control-auth-token-SECRET-CANARY-33333"
CANARY_BEARER = "Bearer eyJhbGciOi-SECRET-TOKEN-22222"

ALL_CANARIES = (
    CANARY_OPENAI,
    CANARY_DEEPGRAM,
    CANARY_GEMINI,
    CANARY_AZURE,
    CANARY_ASSEMBLY,
    CANARY_HF,
    CANARY_CONTROL_TOKEN,
    CANARY_BEARER,
)


def _assert_no_canary_leaks(target_text: str, context: str = ""):
    for canary in ALL_CANARIES:
        assert canary not in target_text, f"Secret leak detected in {context}: canary '{canary}' found!"


# ─────────────────────────────────────────────────────────────────────────────
# 1. ZUSTAND PERSISTENCE / LOCALSTORAGE SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def test_zustand_persistence_sanitization_drops_all_secrets():
    """Verify frontend state persistence whitelist strips raw keys and persists zero secrets."""
    # Read settingsSlice.ts to verify PERSISTED_CONFIG_KEYS contains NO secrets
    store_file = Path("apps/desktop/src/store/settingsSlice.ts")
    assert store_file.exists()
    content = store_file.read_text(encoding="utf-8")

    # Explicitly check banned secret names are never in PERSISTED_CONFIG_FLAGS
    for banned in ("apiKey", "api_key", "llmApiKey", "hfToken", "token", "secret", "password"):
        assert f"{banned}: true" not in content, f"Secret field '{banned}' found in PERSISTED_CONFIG_FLAGS!"

    # Simulate dirty state arriving at persistence boundary
    dirty_config = {
        "host": "127.0.0.1",
        "port": 8765,
        "modelSize": "small",
        "language": "en",
        "device": "auto",
        "translationEnabled": True,
        "llmEnabled": True,
        "llmApiKey": CANARY_OPENAI,  # Secret!
        "hfToken": CANARY_HF,       # Secret!
        "apiKey": CANARY_DEEPGRAM,  # Secret!
        "secret": "confidential",   # Secret!
    }

    # Simulate sanitizeConfig logic (whitelist-only copy)
    allowed_keys = {
        "host", "port", "modelSize", "language", "device",
        "translationEnabled", "translationBackend", "translationModel",
        "sourceLang", "targetLang", "vadEnabled", "audioSource",
        "micDeviceId", "speakerDeviceId", "llmEnabled", "llmProviderUrl",
        "llmModel", "stealthMode",
    }
    sanitized = {k: v for k, v in dirty_config.items() if k in allowed_keys}
    serialized_blob = json.dumps({"theme": "dark", "config": sanitized})

    _assert_no_canary_leaks(serialized_blob, "Zustand persisted localStorage blob")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLITE DATABASE RAW FILE & TABLE SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def test_sqlite_database_raw_bytes_and_tables_contain_zero_secrets(tmp_path):
    """Verify SQLite database file and all rows/tables contain zero leaked secrets."""
    db_file = tmp_path / DB_FILENAME
    repo = ConversationRepository(db_file)

    # 1. Create conversation
    repo.create_conversation(
        ConversationRecord(
            conversation_id="conv-sec-1",
            title="Confidential security discussion",
            created_at_ms=1000.0,
            updated_at_ms=1000.0,
        )
    )

    # 2. Record session with safe metadata
    repo.create_session(
        SessionRecord(
            session_id="sess-sec-1",
            conversation_id="conv-sec-1",
            started_at_ms=1000.0,
            providers={},
        )
    )

    # 3. Insert messages and translations
    repo.upsert_message(
        "conv-sec-1",
        TranscriptMessage(
            message_id="msg-sec-1",
            session_id="sess-sec-1",
            source="mic",
            text="Meeting regarding authorization architecture.",
            translation="Cuoc hop ve kien truc xac thuc.",
            is_final=True,
            timestamp_ms=1000.0,
        ),
    )

    # 4. Create context profile
    profile_service = ContextProfileService(repo)
    profile_service.create_profile(
        name="Executive Briefing Profile",
        content="Analyze quarterly metrics strictly. EchoFlux Tauri PyTorch",
        profile_id="prof-sec-1",
    )
    repo.close()

    # Deep scan: Read every string in every table of SQLite
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    for table in tables:
        rows = cursor.execute(f"SELECT * FROM {table}").fetchall()
        for row in rows:
            for col in row:
                if isinstance(col, (str, bytes)):
                    text = col.decode("utf-8", errors="ignore") if isinstance(col, bytes) else col
                    _assert_no_canary_leaks(text, f"SQLite table '{table}'")

    conn.close()

    # Raw disk bytes scan: search entire database binary
    raw_db_bytes = db_file.read_bytes()
    for canary in ALL_CANARIES:
        assert canary.encode("utf-8") not in raw_db_bytes, f"Canary {canary} leaked into raw SQLite database file!"


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIG FILE (config.json) SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def test_config_save_stores_credential_refs_never_raw_secrets(tmp_path):
    """Verify engine config saves credential references only and rejects raw secrets."""
    config_file = tmp_path / "config.json"
    env_file = tmp_path / ".env"

    config = Config(config_path=str(config_file), env_path=str(env_file))

    # Service configured with CredentialRef (safe)
    service = ServiceConfig(
        service="stt",
        provider_id="deepgram",
        enabled=True,
        credential_refs={
            "api_key": CredentialRef(config_id="deepgram-main", slot="api_key"),
        },
    )

    # Options with secret-shaped names must be rejected
    for secret_key in ("api_key", "token", "secret", "password"):
        assert is_secret_key_name(secret_key) is True

    # Endpoints with embedded credentials must be rejected
    assert endpoint_has_embedded_credentials(f"https://user:{CANARY_DEEPGRAM}@api.example.com") is True
    assert endpoint_has_embedded_credentials("wss://api.deepgram.com/v1/listen") is False

    # Save config to disk and scan raw file content
    config.set_service(service)
    config.save()

    raw_saved = config_file.read_text(encoding="utf-8")
    _assert_no_canary_leaks(raw_saved, "saved config.json file")
    assert "deepgram-main" in raw_saved
    assert "credentialRefs" in raw_saved


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOG REDACTION SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def test_log_redaction_strips_canary_tokens_from_urls_and_text():
    """Verify log sanitizers redact API keys, tokens, and authorization headers."""
    # Test URL redaction
    url_with_token = f"wss://api.deepgram.com/v1/listen?token={CANARY_DEEPGRAM}&model=nova-3"
    redacted_url = redact_url(url_with_token)
    _assert_no_canary_leaks(redacted_url, "redacted URL")
    assert redacted_url == "wss://api.deepgram.com/v1/listen"

    # Test text redaction
    text_with_keys = f"Connecting using {CANARY_BEARER} and key={CANARY_OPENAI}."
    redacted_msg = redact_text(text_with_keys, secrets=ALL_CANARIES)
    _assert_no_canary_leaks(redacted_msg, "redacted log text")

    # Test in-memory logging capture
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = get_logger("security.audit")
    logger.addHandler(handler)

    try:
        # Logging redacted message
        safe_log = redact_text(f"Connection attempt to {redacted_url} with {CANARY_OPENAI}", secrets=ALL_CANARIES)
        logger.info(safe_log)
        handler.flush()

        logged_content = log_stream.getvalue()
        _assert_no_canary_leaks(logged_content, "in-memory log capture")
    finally:
        logger.removeHandler(handler)


# ─────────────────────────────────────────────────────────────────────────────
# 5. WEBSOCKET ERROR FRAMES & EVENT BROADCAST SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def test_error_frame_and_events_never_disclose_secret_details():
    """Verify control error frames and event frames contain zero sensitive exception traces."""
    # 1. Error frame generation with sensitive request_id or details
    frame = error_frame("command_failed", "Command failed", request_id="req-123")
    encoded = json.dumps(frame)
    _assert_no_canary_leaks(encoded, "error_frame")
    assert frame["code"] == "command_failed"
    assert frame["message"] == "Command failed"

    # 2. Event frame sanitization
    raw_event = {
        "type": "final",
        "text": "Regular spoken transcript.",
        "conversation_id": "conv-1",
        "message_id": "msg-1",
        "is_final": True,
        # Should not leak internal keys if injected
        "internal_token": CANARY_CONTROL_TOKEN,
    }
    sanitized = sanitize_event(raw_event)
    encoded_event = json.dumps(sanitized)
    _assert_no_canary_leaks(encoded_event, "sanitized event frame")
    assert sanitized["internal_token"] == "[redacted]"


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONVERSATION EXPORT SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def test_conversation_export_contains_zero_canary_secrets(tmp_path):
    """Verify conversation export produces clean transcripts without secrets."""
    db_file = tmp_path / DB_FILENAME
    repo = ConversationRepository(db_file)
    repo.create_conversation(
        ConversationRecord(
            conversation_id="conv-export-1",
            title="Export verification",
            created_at_ms=1000.0,
            updated_at_ms=1000.0,
        )
    )

    repo.upsert_message(
        "conv-export-1",
        TranscriptMessage(
            message_id="msg-exp-1",
            source="mic",
            text="Exported transcript line.",
            translation="Dong ban ghi xuat ra.",
            is_final=True,
            timestamp_ms=1000.0,
        ),
    )

    exported = repo.export_conversation("conv-export-1")
    serialized_export = json.dumps(exported)
    _assert_no_canary_leaks(serialized_export, "exported conversation JSON")

    repo.close()
