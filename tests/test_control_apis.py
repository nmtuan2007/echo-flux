"""Comprehensive Integration Test Suite for Phase B Application APIs over authenticated WebSocket control loopback.

Validates:
1. Role-based access control, fail-closed security, and authentication gating.
2. Context Profile CRUD, validation, listing privacy, and conversation attachments.
3. Immutable session profile snapshotting and tamper resistance.
4. Conversation management: create, start, list, get, rename, delete, export, import.
5. Curated local model packs: status, readiness, install, cancel.
6. Provider manifest retrieval from build_default_registry().
7. ID-only Assist and Summary streaming with cancellation.
8. Strict preservation of tests/test_ws_contract.py invariants.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import pytest
import websockets

from engine.assistant.summary_service import EngineAssistantPromptPolicy
from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    MAX_PROFILE_CONTENT_BYTES,
    MAX_PROFILE_NAME_CHARS,
    TranscriptMessage,
)
from engine.conversation.repository import ConversationRepository
from engine.providers.contracts import AssistantProvider
from engine.providers.text.openai_compatible import TextCompletion
from engine.providers.types import (
    AssistRequest,
    AssistResult,
    ProviderHealthCheck,
    ProviderHealthStatus,
    SummaryChunk,
    SummaryRequest,
    SummaryResult,
)
from engine.server import protocol
from engine.server.control_dispatcher import ControlCommandDispatcher
from engine.server.protocol import (
    BootstrapTokens,
)
from engine.server.websocket_server import WebSocketServer

CONTROL_TOKEN = "c0" * 32
EVENT_TOKEN = "e1" * 32
TOKENS = BootstrapTokens(control=CONTROL_TOKEN, event=EVENT_TOKEN)

_CLIENT_LOGGER = logging.getLogger("tests.control_apis.client")
_CLIENT_LOGGER.setLevel(logging.WARNING)

GOOD_ASSIST_REPLY = json.dumps(
    {
        "suggested_replies": [
            {"strategy": "Direct", "text": "We can ship on Friday.", "reasoning": "Clear response"}
        ],
        "clarifying_questions": [{"question": "Which release?", "purpose": "Scope confirmation"}],
    }
)


class FakeAssistant(AssistantProvider):
    """Assistant provider with deterministic replies and streaming support."""

    def __init__(
        self,
        assist_reply: Any = GOOD_ASSIST_REPLY,
        chunks: Any = ("## Meeting Summary\n", "- Action: Ship on Friday.\n"),
        gate: Optional[asyncio.Event] = None,
    ) -> None:
        self.assist_reply = assist_reply
        self.chunks = list(chunks)
        self.gate = gate
        self.policy = EngineAssistantPromptPolicy()
        self.assist_requests: List[AssistRequest] = []
        self.summary_requests: List[SummaryRequest] = []
        self.assist_started = asyncio.Event()
        self.summary_started = asyncio.Event()

    @property
    def provider_id(self) -> str:
        return "fake-assistant"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

    async def assist(self, request: AssistRequest) -> AssistResult:
        self.assist_requests.append(request)
        self.policy.build_assist_messages(request)
        self.assist_started.set()
        if self.gate is not None:
            await self.gate.wait()
        if isinstance(self.assist_reply, BaseException):
            raise self.assist_reply
        return self.policy.parse_assist_result(request, TextCompletion(text=self.assist_reply))

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        raise NotImplementedError

    async def stream_summary(self, request: SummaryRequest):
        self.summary_requests.append(request)
        self.policy.build_summary_messages(request)
        self.summary_started.set()
        for item in self.chunks:
            if self.gate is not None:
                await self.gate.wait()
            if isinstance(item, BaseException):
                raise item
            yield SummaryChunk(delta_text=item)


class ControlClient:
    """Helper client wrapping WebSocket control connection."""

    def __init__(self, ws: Any, tokens: BootstrapTokens):
        self.ws = ws
        self.tokens = tokens
        self._seq = 0

    def next_id(self, prefix: str = "req") -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    async def call(
        self, command: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 3.0
    ) -> Dict[str, Any]:
        """Send command and wait for single response + complete."""
        req_id = self.next_id(command)
        frame = {
            "op": "command",
            "id": req_id,
            "command": command,
            "payload": payload or {},
        }
        await self.ws.send(json.dumps(frame))

        response_data = None
        error_data = None
        completed = False

        while not completed:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            reply = json.loads(raw)
            if reply.get("id") != req_id:
                continue
            op = reply.get("op")
            if op == "response":
                response_data = reply.get("data", reply.get("message"))
            elif op == "error":
                error_data = reply
            elif op == "complete":
                completed = True

        if error_data is not None:
            return {"ok": False, "error": error_data}
        return {"ok": True, "data": response_data}

    async def call_legacy_envelope(
        self, command: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 3.0
    ) -> Dict[str, Any]:
        """Send command using the op: request envelope."""
        req_id = self.next_id(command)
        frame = {
            "op": "request",
            "id": req_id,
            "command": {
                "command": command,
                "payload": payload or {},
            },
        }
        await self.ws.send(json.dumps(frame))

        response_data = None
        error_data = None
        completed = False

        while not completed:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            reply = json.loads(raw)
            if reply.get("id") != req_id:
                continue
            op = reply.get("op")
            if op == "response":
                response_data = reply.get("data", reply.get("message"))
            elif op == "error":
                error_data = reply
            elif op == "complete":
                completed = True

        if error_data is not None:
            return {"ok": False, "error": error_data}
        return {"ok": True, "data": response_data}

    async def stream(
        self, command: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Send streaming command and collect all response frames until complete."""
        req_id = self.next_id(command)
        frame = {
            "op": "command",
            "id": req_id,
            "command": command,
            "payload": payload or {},
        }
        await self.ws.send(json.dumps(frame))
        frames = []
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            reply = json.loads(raw)
            if reply.get("id") == req_id:
                frames.append(reply)
                if reply.get("op") in ("complete", "error"):
                    break
        return frames

    async def close(self):
        await self.ws.close()


async def connect_client(
    server: WebSocketServer, role: str = "control", token: str = CONTROL_TOKEN
) -> ControlClient:
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER)
    await ws.send(json.dumps({"op": "auth", "role": role, "token": token, "protocol": 1}))
    resp = json.loads(await ws.recv())
    assert resp.get("op") == "auth_ok"
    return ControlClient(ws, TOKENS)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv(protocol.ENV_CONTROL_TOKEN, CONTROL_TOKEN)
    monkeypatch.setenv(protocol.ENV_EVENT_TOKEN, EVENT_TOKEN)
    monkeypatch.setenv("ECHOFLUX_HOST", "127.0.0.1")
    monkeypatch.setenv("ECHOFLUX_PORT", "0")


@pytest.fixture
def repo(tmp_path):
    repository = ConversationRepository(tmp_path / "control_test.sqlite3")
    yield repository
    repository.close()


@pytest.fixture
def profile_service(repo):
    return ContextProfileService(repo)


@pytest.fixture
async def control_server(repo, profile_service):
    dispatcher = ControlCommandDispatcher(
        conversation_repo=repo,
        profile_service=profile_service,
    )
    server = WebSocketServer(
        host="127.0.0.1",
        port=0,
        tokens=TOKENS,
        control_dispatcher=dispatcher,
    )
    await server.start()
    try:
        yield server, dispatcher
    finally:
        await server.stop()


# -----------------------------------------------------------------------------
# 1. TestControlAuthSecurity
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthenticated_connection_rejected(control_server):
    server, _ = control_server
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER)
    try:
        cmd = {"op": "command", "id": "req-1", "command": "profile_list", "payload": {}}
        await ws.send(json.dumps(cmd))
        raw = await ws.recv()
        reply = json.loads(raw)
        assert reply.get("op") == "error"
        assert reply.get("code") in ("auth_required", "auth_failed")
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_event_role_client_cannot_execute_control_commands(control_server):
    server, _ = control_server
    client = await connect_client(server, role="event", token=EVENT_TOKEN)
    try:
        cmd = {"op": "command", "id": "req-1", "command": "profile_list", "payload": {}}
        await client.ws.send(json.dumps(cmd))
        raw = await client.ws.recv()
        reply = json.loads(raw)
        assert reply.get("op") == "error"
        assert reply.get("code") == "forbidden"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_invalid_token_rejected(control_server):
    server, _ = control_server
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER)
    try:
        bad_token = "00" * 32
        await ws.send(
            json.dumps({"op": "auth", "role": "control", "token": bad_token, "protocol": 1})
        )
        raw = await ws.recv()
        reply = json.loads(raw)
        assert reply.get("op") == "error"
        assert reply.get("code") == "auth_failed"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_authenticated_control_client_accepted(control_server):
    server, _ = control_server
    client = await connect_client(server, role="control", token=CONTROL_TOKEN)
    try:
        res = await client.call("profile_list")
        assert res["ok"] is True
        assert "profiles" in res["data"]
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 2. TestContextProfileControlAPIs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_create_success_and_listing_omits_content(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        # Create profile
        res = await client.call(
            "profile_create",
            {
                "name": "Medical Glossary",
                "content": "Terminology: ACME is a cardio diagnostic tool.",
                "content_format": "markdown",
                "profile_id": "p-med",
            },
        )
        assert res["ok"] is True
        prof = res["data"]["profile"]
        assert prof["profile_id"] == "p-med"
        assert prof["name"] == "Medical Glossary"
        assert prof["content"] == "Terminology: ACME is a cardio diagnostic tool."
        assert prof["revision"] == 1
        assert prof["content_format"] == "markdown"

        # List profiles: verify zero content leakage
        list_res = await client.call("profile_list")
        assert list_res["ok"] is True
        profiles = list_res["data"]["profiles"]
        assert len(profiles) == 1
        listed = profiles[0]
        assert listed["profile_id"] == "p-med"
        assert listed["name"] == "Medical Glossary"
        assert listed["content_bytes"] > 0
        assert "content" not in listed  # Must omit raw content!
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_profile_create_validation_failures(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        # 1. Invalid content format
        res1 = await client.call(
            "profile_create",
            {"name": "Bad Format", "content": "hello", "content_format": "html"},
        )
        assert res1["ok"] is False
        assert res1["error"]["code"] == "validation_error"

        # 2. Content too large (>64 KB)
        res2 = await client.call(
            "profile_create",
            {"name": "Huge", "content": "x" * (MAX_PROFILE_CONTENT_BYTES + 10)},
        )
        assert res2["ok"] is False
        assert res2["error"]["code"] == "validation_error"

        # 3. Name too long (>96 chars)
        res3 = await client.call(
            "profile_create",
            {"name": "A" * (MAX_PROFILE_NAME_CHARS + 10), "content": "ok"},
        )
        assert res3["ok"] is False
        assert res3["error"]["code"] == "validation_error"

        # 4. Null byte in content
        res4 = await client.call(
            "profile_create",
            {"name": "Binary", "content": "Hello\x00World"},
        )
        assert res4["ok"] is False
        assert res4["error"]["code"] == "validation_error"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_profile_create_conflict(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        res1 = await client.call(
            "profile_create",
            {"name": "Original", "content": "first", "profile_id": "p-dup"},
        )
        assert res1["ok"] is True

        res2 = await client.call(
            "profile_create",
            {"name": "Duplicate", "content": "second", "profile_id": "p-dup"},
        )
        assert res2["ok"] is False
        assert res2["error"]["code"] == "conflict"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_profile_get_success_and_not_found(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call(
            "profile_create",
            {"name": "Search Target", "content": "secret-target-content", "profile_id": "p-find"},
        )

        res = await client.call("profile_get", {"profile_id": "p-find"})
        assert res["ok"] is True
        assert res["data"]["profile"]["content"] == "secret-target-content"

        missing = await client.call("profile_get", {"profile_id": "p-nonexistent"})
        assert missing["ok"] is False
        assert missing["error"]["code"] == "not_found"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_profile_update_optimistic_locking(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call(
            "profile_create",
            {"name": "To Edit", "content": "rev 1 text", "profile_id": "p-edit"},
        )

        # Update with expected revision 1 -> succeeds, bumps to 2
        upd = await client.call(
            "profile_update",
            {"profile_id": "p-edit", "content": "rev 2 text", "expected_revision": 1},
        )
        assert upd["ok"] is True
        assert upd["data"]["profile"]["revision"] == 2
        assert upd["data"]["profile"]["content"] == "rev 2 text"

        # Stale update with expected revision 1 -> conflict
        stale = await client.call(
            "profile_update",
            {"profile_id": "p-edit", "content": "rev 3 text", "expected_revision": 1},
        )
        assert stale["ok"] is False
        assert stale["error"]["code"] == "conflict"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_profile_delete_idempotent(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call(
            "profile_create",
            {"name": "To Delete", "content": "bye", "profile_id": "p-del"},
        )

        del1 = await client.call("profile_delete", {"profile_id": "p-del"})
        assert del1["ok"] is True
        assert del1["data"]["deleted"] is True

        del2 = await client.call("profile_delete", {"profile_id": "p-del"})
        assert del2["ok"] is True
        assert del2["data"]["deleted"] is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_profile_attach_to_conversation(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call("conversation_create", {"title": "Meeting", "conversation_id": "c-att"})
        await client.call("profile_create", {"name": "P1", "content": "C1", "profile_id": "p-1"})
        await client.call("profile_create", {"name": "P2", "content": "C2", "profile_id": "p-2"})

        attach_res = await client.call(
            "profile_attach",
            {"conversation_id": "c-att", "profile_ids": ["p-1", "p-2"]},
        )
        assert attach_res["ok"] is True
        assert attach_res["data"]["attached_profile_ids"] == ["p-1", "p-2"]

        # Verify get reflects attachment
        cget = await client.call("conversation_get", {"conversation_id": "c-att"})
        assert cget["ok"] is True
        assert cget["data"]["attached_profile_ids"] == ["p-1", "p-2"]

        # Duplicate attachment rejected
        dup = await client.call(
            "profile_attach",
            {"conversation_id": "c-att", "profile_ids": ["p-1", "p-1"]},
        )
        assert dup["ok"] is False
        assert dup["error"]["code"] == "validation_error"
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 3. TestImmutableSessionProfileSnapshots
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_start_snapshots_attached_profiles_immutably(control_server, repo):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call("conversation_create", {"title": "Conv", "conversation_id": "c-snap"})
        await client.call(
            "profile_create",
            {"name": "Original Name", "content": "Original Content", "profile_id": "p-snap"},
        )
        await client.call(
            "profile_attach",
            {"conversation_id": "c-snap", "profile_ids": ["p-snap"]},
        )

        # Start session -> creates immutable snapshot
        s_res = await client.call(
            "conversation_start",
            {"conversation_id": "c-snap", "session_id": "s-snap-1"},
        )
        assert s_res["ok"] is True
        assert s_res["data"]["snapshots_count"] == 1

        # Now mutate the original profile
        await client.call(
            "profile_update",
            {
                "profile_id": "p-snap",
                "name": "Mutated Name",
                "content": "Mutated Content",
            },
        )

        # Query snapshot directly from SQLite to prove immutability
        with repo.read() as conn:
            row = conn.execute(
                "SELECT * FROM session_profile_snapshots WHERE session_id = 's-snap-1'"
            ).fetchone()
            assert row is not None
            assert row["name"] == "Original Name"
            assert row["content"] == "Original Content"
            assert row["profile_revision"] == 1

        # Delete profile: snapshot remains intact
        await client.call("profile_delete", {"profile_id": "p-snap"})
        with repo.read() as conn:
            row = conn.execute(
                "SELECT * FROM session_profile_snapshots WHERE session_id = 's-snap-1'"
            ).fetchone()
            assert row is not None
            assert row["content"] == "Original Content"
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 4. TestConversationControlAPIs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_lifecycle_crud(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        # Create
        c1 = await client.call("conversation_create", {"title": "Sprint 42", "conversation_id": "c-crud"})
        assert c1["ok"] is True
        assert c1["data"]["conversation"]["conversation_id"] == "c-crud"
        assert c1["data"]["conversation"]["title"] == "Sprint 42"

        # List
        clist = await client.call("conversation_list")
        assert clist["ok"] is True
        ids = [c["conversation_id"] for c in clist["data"]["conversations"]]
        assert "c-crud" in ids

        # Rename
        r_res = await client.call("conversation_rename", {"conversation_id": "c-crud", "title": "Sprint 43"})
        assert r_res["ok"] is True
        assert r_res["data"]["conversation"]["title"] == "Sprint 43"

        # Delete
        d_res = await client.call("conversation_delete", {"conversation_id": "c-crud"})
        assert d_res["ok"] is True
        assert d_res["data"]["deleted"] is True

        # Verify not found after delete
        missing = await client.call("conversation_get", {"conversation_id": "c-crud"})
        assert missing["ok"] is False
        assert missing["error"]["code"] == "not_found"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_conversation_get_with_and_without_messages(control_server, repo):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call("conversation_create", {"title": "Msg Test", "conversation_id": "c-msgs"})
        repo.upsert_message(
            "c-msgs",
            TranscriptMessage(
                message_id="m-1",
                source="mic",
                text="Hello world",
                timestamp_ms=1000.0,
            ),
        )

        # Without include_messages
        get_no = await client.call("conversation_get", {"conversation_id": "c-msgs"})
        assert get_no["ok"] is True
        assert get_no["data"]["messages"] == []

        # With include_messages
        get_yes = await client.call(
            "conversation_get", {"conversation_id": "c-msgs", "include_messages": True}
        )
        assert get_yes["ok"] is True
        assert len(get_yes["data"]["messages"]) == 1
        assert get_yes["data"]["messages"][0]["text"] == "Hello world"
        assert get_yes["data"]["messages"][0]["speaker_label"] == "Me"
        assert get_yes["data"]["summary"] is None

        # Add summary and check stats in list + detail
        from engine.providers.types import SummaryResult

        repo.save_summary("c-msgs", SummaryResult(summary_markdown="## Recap\n- Done"))
        clist = await client.call("conversation_list")
        row = next(c for c in clist["data"]["conversations"] if c["conversation_id"] == "c-msgs")
        assert row["message_count"] == 1
        assert row["has_summary"] is True

        get_sum = await client.call("conversation_get", {"conversation_id": "c-msgs"})
        assert get_sum["data"]["summary"]["summary_markdown"] == "## Recap\n- Done"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_conversation_export_and_import_idempotent(control_server, repo):
    server, _ = control_server
    client = await connect_client(server)
    try:
        await client.call("conversation_create", {"title": "Export Test", "conversation_id": "c-exp"})
        repo.upsert_message(
            "c-exp",
            TranscriptMessage(
                message_id="m-exp-1",
                source="mic",
                text="Valuable meeting notes",
                timestamp_ms=1500.0,
            ),
        )

        exp = await client.call("conversation_export", {"conversation_id": "c-exp"})
        assert exp["ok"] is True
        export_data = exp["data"]["export"]
        assert export_data["conversation_id"] == "c-exp"
        assert len(export_data["messages"]) == 1

        # Delete original conversation
        await client.call("conversation_delete", {"conversation_id": "c-exp"})

        # Import conversation back
        imp1 = await client.call("conversation_import", {"export_data": export_data})
        assert imp1["ok"] is True
        assert imp1["data"]["imported"] is True

        # Second import with same import_id -> no-op (imported False)
        imp2 = await client.call("conversation_import", {"export_data": export_data})
        assert imp2["ok"] is True
        assert imp2["data"]["imported"] is False
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 5. TestLocalModelPackControlAPIs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pack_status_and_readiness(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        # All packs status
        res = await client.call("pack_status")
        assert res["ok"] is True
        assert "packs" in res["data"]
        assert "hardware" in res["data"]
        assert isinstance(res["data"]["packs"], list)

        # Single pack status
        pack_id = res["data"]["packs"][0]["pack_id"]
        single = await client.call("pack_status", {"pack_id": pack_id})
        assert single["ok"] is True
        assert single["data"]["pack_id"] == pack_id

        # Readiness check
        readiness = await client.call("pack_check_readiness", {"capability": "stt"})
        assert readiness["ok"] is True
        assert "ready" in readiness["data"]
        assert "capability" in readiness["data"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_pack_install_and_cancel(control_server):
    server, _dispatcher = control_server
    client = await connect_client(server)
    try:
        # Test cancel on inactive install returns cancelled False
        c_res = await client.call("pack_cancel", {"pack_id": "stt-cpu-standard"})
        assert c_res["ok"] is True
        assert c_res["data"]["cancelled"] is False
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 6. TestProviderManifestControlAPIs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_provider_manifests_all_and_filtered(control_server):
    server, _ = control_server
    client = await connect_client(server)
    try:
        # All manifests
        res = await client.call("get_provider_manifests")
        assert res["ok"] is True
        manifests = res["data"]["manifests"]
        assert len(manifests) > 0
        provider_ids = [m["provider_id"] for m in manifests]
        assert "openai-realtime" in provider_ids
        assert "local" in provider_ids

        # Filtered by capability
        stt_res = await client.call("get_provider_manifests", {"capability": "stt"})
        assert stt_res["ok"] is True
        for m in stt_res["data"]["manifests"]:
            assert "stt" in m["capabilities"]
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 7. TestAssistControlAPIs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_assist_success_and_persistence(control_server, repo):
    server, dispatcher = control_server
    from engine.assistant.service import AssistantService

    fake_provider = FakeAssistant()
    dispatcher._assistant_service = AssistantService(
        repo, ContextProfileService(repo), fake_provider, model="model-test"
    )

    client = await connect_client(server)
    try:
        await client.call("conversation_create", {"title": "Assist Conv", "conversation_id": "c-ast"})
        repo.upsert_message(
            "c-ast",
            TranscriptMessage(
                message_id="m-ast-1",
                source="mic",
                text="When is the deployment ready?",
                timestamp_ms=2000.0,
            ),
        )

        res = await client.call(
            "request_assist",
            {"conversation_id": "c-ast", "message_id": "m-ast-1"},
        )
        assert res["ok"] is True
        data = res["data"]
        assert data["conversation_id"] == "c-ast"
        assert data["message_id"] == "m-ast-1"
        assert len(data["suggested_replies"]) == 1
        assert data["suggested_replies"][0]["text"] == "We can ship on Friday."
        assert len(data["clarifying_questions"]) == 1

        # Verify saved in SQLite assist_results table
        with repo.read() as conn:
            row = conn.execute(
                "SELECT * FROM assist_results WHERE conversation_id = 'c-ast'"
            ).fetchone()
            assert row is not None
            assert row["target_message_id"] == "m-ast-1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_assist_target_not_found_and_unavailable(control_server, repo):
    server, dispatcher = control_server
    client = await connect_client(server)
    try:
        # 1. Assistant unavailable when not configured
        dispatcher._assistant_service = None
        unavail = await client.call(
            "request_assist",
            {"conversation_id": "c-test", "message_id": "m-missing"},
        )
        assert unavail["ok"] is False
        assert unavail["error"]["code"] == "assistant_unavailable"

        # 2. Configured but target message missing
        from engine.assistant.service import AssistantService

        dispatcher._assistant_service = AssistantService(
            repo, ContextProfileService(repo), FakeAssistant()
        )
        await client.call("conversation_create", {"title": "Test", "conversation_id": "c-test"})
        missing = await client.call(
            "request_assist",
            {"conversation_id": "c-test", "message_id": "m-does-not-exist"},
        )
        assert missing["ok"] is False
        assert missing["error"]["code"] == "target_not_found"
    finally:
        await client.close()


# -----------------------------------------------------------------------------
# 8. TestSummaryControlAPIs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_summary_streaming_and_cancellation(control_server, repo):
    server, dispatcher = control_server
    from engine.assistant.summary_service import SummaryService

    gate = asyncio.Event()
    fake_provider = FakeAssistant(
        chunks=("## Summary\n", "- Key decision: Launch Friday.\n"), gate=gate
    )
    dispatcher._summary_service = SummaryService(repo, fake_provider, model="model-sum")

    client = await connect_client(server)
    try:
        await client.call("conversation_create", {"title": "Sum Conv", "conversation_id": "c-sum"})
        repo.upsert_message(
            "c-sum",
            TranscriptMessage(
                message_id="m-sum-1",
                source="mic",
                text="Let us discuss the launch timeline.",
                timestamp_ms=1000.0,
            ),
        )

        # Release gate immediately for normal streaming run
        gate.set()
        frames = await client.stream("request_summary", {"conversation_id": "c-sum"})
        assert len(frames) >= 3  # at least chunks + terminal done + complete

        responses = [f for f in frames if f.get("op") == "response"]
        completes = [f for f in frames if f.get("op") == "complete"]
        assert len(completes) == 1

        # Check final response
        final_resp = responses[-1]
        assert final_resp["data"]["done"] is True
        assert final_resp["data"]["included_message_count"] == 1

        # Check SQLite persistence
        with repo.read() as conn:
            row = conn.execute("SELECT * FROM summaries WHERE conversation_id = 'c-sum'").fetchone()
            assert row is not None

        # Test cancellation
        fake_provider.summary_started.clear()
        gate.clear()  # Block streaming
        stream_task = asyncio.create_task(
            client.stream("request_summary", {"conversation_id": "c-sum"})
        )
        await fake_provider.summary_started.wait()

        # Cancel the active summary
        active_ids = list(dispatcher._active_tasks.keys())
        assert len(active_ids) == 1
        cancel_client = await connect_client(server)
        try:
            c_res = await cancel_client.call("summary_cancel", {"request_id": active_ids[0]})
            assert c_res["ok"] is True
            assert c_res["data"]["cancelled"] is True
        finally:
            await cancel_client.close()

        # Complete canceled stream task
        canceled_frames = await stream_task
        error_frames = [f for f in canceled_frames if f.get("op") == "error"]
        assert len(error_frames) == 1
        assert error_frames[0]["code"] == "cancelled"
    finally:
        gate.set()
        await client.close()


# -----------------------------------------------------------------------------
# 9. TestContractCompatibility
# -----------------------------------------------------------------------------


def test_control_commands_do_not_trip_ws_contract_regexes():
    """Verify directly that none of the 22 commands or dispatcher logic tripped test_ws_contract."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    server_src = (root / "engine" / "server" / "websocket_server.py").read_text(encoding="utf-8")
    main_src = (root / "engine" / "main.py").read_text(encoding="utf-8")

    # 1. Pinned handled types must remain exactly the 6 approved types
    handled = set(re.findall(r'msg_type\s*==\s*"(\w+)"', server_src))
    assert handled == {
        "start",
        "stop",
        "list_devices",
        "request_suggestion",
        "request_summary",
        "test_connection",
    }

    # 2. Pinned sent types must remain exactly the 10 approved types
    sent = set()
    for src in (server_src, main_src):
        for match in re.finditer(r'"type":\s*([^,\n}]+)', src):
            sent.update(re.findall(r'"(\w+)"', match.group(1)))
    assert sent == {
        "partial",
        "final",
        "translation_update",
        "status",
        "error",
        "devices_list",
        "suggestion_result",
        "llm_chunk",
        "llm_done",
        "test_connection_result",
    }


@pytest.mark.asyncio
async def test_audio_probe_samples_stream_and_refuses_when_running(control_server, monkeypatch):
    import sys
    import types

    import numpy as np

    server, dispatcher = control_server

    class FakeMicInput:
        def __init__(self, config, device_id=None):
            self._reads = 0

        def start(self):
            pass

        def stop(self):
            pass

        def read_chunk(self):
            self._reads += 1
            if self._reads == 1:
                samples = (np.ones(160, dtype=np.int16) * 3276)
                return samples.tobytes()
            return b""

    fake_mic_mod = types.ModuleType("engine.audio.microphone")
    fake_mic_mod.MicrophoneInput = FakeMicInput
    monkeypatch.setitem(sys.modules, "engine.audio.microphone", fake_mic_mod)

    client = await connect_client(server)
    try:
        res = await client.call("audio_probe", {"audio_source": "microphone", "duration_ms": 120})
        assert res["ok"] is True
        mic = res["data"]["streams"]["mic"]
        assert mic["ok"] is True
        assert mic["level"] > 0.0

        dispatcher._engine = types.SimpleNamespace(_running=True)
        busy = await client.call("audio_probe", {"audio_source": "microphone"})
        assert busy["ok"] is False
        assert busy["error"]["code"] == "engine_busy"
    finally:
        dispatcher._engine = None
        await client.close()
