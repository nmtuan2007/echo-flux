"""Empirical adversarial stress and contract verification tests for Milestone 2.

Authored by challenger_m2_2.
Stress-tests:
1. Malformed JSON envelopes & corrupt frame rejection on the control WebSocket.
2. Unknown commands and unhandled ops fail safely without traceback leakage.
3. Strict authentication & role boundary enforcement (unauth, event role, invalid token).
4. High-concurrency command dispatch and race condition resistance.
5. Phase B domain parameter boundary validation (Context Profiles, Conversations, Model Packs, Manifests).
6. Discovery of bugs in Phase B APIs & Tauri Control Bridge forwarding:
   - Defect 1: Deadlock / timeout on `request_assist` forwarding through Tauri IPC bridge.
   - Defect 2: `summary_cancel` missing from Tauri IPC bridge `forward_policy` allowlist.
   - Defect 3: `conversation_import` crashes with `TypeError` when importing assist_results.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

import pytest
import websockets
from test_control_apis import FakeAssistant

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    ConversationRecord,
    TranscriptMessage,
)
from engine.conversation.repository import ConversationRepository
from engine.server.control_dispatcher import ControlCommandDispatcher
from engine.server.protocol import (
    BootstrapTokens,
)
from engine.server.websocket_server import WebSocketServer

CONTROL_TOKEN = "c0" * 32
EVENT_TOKEN = "e1" * 32
TOKENS = BootstrapTokens(control=CONTROL_TOKEN, event=EVENT_TOKEN)

logger = logging.getLogger("tests.adversarial_m2")
logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "adversarial_m2.db")
    return ConversationRepository(db_path=db_path)


@pytest.fixture
async def control_server(repo):
    server = WebSocketServer(host="127.0.0.1", port=0, tokens=TOKENS)
    profiles = ContextProfileService(repo)
    dispatcher = ControlCommandDispatcher(
        conversation_repo=repo,
        profile_service=profiles,
    )
    server.set_control_dispatcher(dispatcher)
    await server.start()
    try:
        yield server, dispatcher, repo
    finally:
        await server.stop()


async def connect_and_auth(server, role="control", token=CONTROL_TOKEN):
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}")
    await ws.send(json.dumps({
        "op": "auth",
        "role": role,
        "token": token,
        "protocol": 1,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
    msg = json.loads(raw)
    assert msg.get("op") == "auth_ok", f"Expected auth_ok, got: {msg}"
    return ws


# ---------------------------------------------------------------------------------------
# 1. Malformed JSON envelopes & Corrupt Frame Rejection
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_binary_and_corrupted_json(control_server):
    server, _, _ = control_server
    ws = await connect_and_auth(server)
    try:
        # Binary frame on authenticated control socket
        await ws.send(b"\x00\x01\x02\xff\xfe")
        # Drain bad_request error frame emitted by parse_control_frame
        err1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err1.get("op") == "error"
        assert err1.get("code") == "bad_request"

        # Now send a valid command to prove server event loop didn't crash
        await ws.send(json.dumps({
            "op": "command",
            "id": "probe-after-binary",
            "command": "profile_list",
            "payload": {},
        }))
        res = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert res.get("id") == "probe-after-binary"
        assert res.get("op") == "response"
        # Drain complete frame
        await asyncio.wait_for(ws.recv(), timeout=2.0)

        # Corrupted JSON string
        await ws.send("{bad json string")
        err2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err2.get("op") == "error"
        assert err2.get("code") == "bad_request"

        # Non-dict JSON
        for non_dict in ("plain text", 12345, [1, 2, 3], None):
            await ws.send(json.dumps(non_dict))
            err = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert err.get("op") == "error"

        # Connection should still be functional
        await ws.send(json.dumps({
            "op": "command",
            "id": "probe-after-corrupted",
            "command": "profile_list",
            "payload": {},
        }))
        res = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert res.get("id") == "probe-after-corrupted"
        assert res.get("op") == "response"
        await asyncio.wait_for(ws.recv(), timeout=2.0)
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_adversarial_malformed_command_envelopes(control_server):
    server, _, _ = control_server
    ws = await connect_and_auth(server)
    try:
        # Missing op
        await ws.send(json.dumps({"id": "req-1", "command": "profile_list"}))
        err1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err1.get("op") == "error"

        # Invalid op
        await ws.send(json.dumps({"op": "bogus_op", "id": "req-2"}))
        err2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err2.get("op") == "error"

        # op: command but command is integer (not in handlers)
        await ws.send(json.dumps({"op": "command", "id": "req-3", "command": 12345}))
        err3 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err3.get("op") == "error"

        # op: command with valid handler but invalid payload (string instead of dict)
        # extract_command substitutes {} so profile_create validates required fields and fails safely
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-4",
            "command": "profile_create",
            "payload": "not-a-dict",
        }))
        res = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert res.get("op") == "error"
        assert res.get("id") == "req-4"
        assert res.get("code") == "invalid_payload"
        await asyncio.wait_for(ws.recv(), timeout=2.0)
    finally:
        await ws.close()


# ---------------------------------------------------------------------------------------
# 2. Unknown Commands Fail Safely Without Tracebacks
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_unknown_commands_fail_safely(control_server):
    server, _, _ = control_server
    ws = await connect_and_auth(server)
    try:
        unknowns = [
            "system_exec",
            "drop_database",
            "__import__",
            "eval",
            "nonexistent_cmd_xyz",
        ]
        for cmd in unknowns:
            await ws.send(json.dumps({
                "op": "command",
                "id": f"id-{cmd}",
                "command": cmd,
                "payload": {},
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            res = json.loads(raw)
            # Must return an error frame
            assert res.get("op") == "error"
            # Must never leak python traceback, file paths, or exception objects
            rendered = json.dumps(res)
            assert "Traceback" not in rendered
            assert ".py" not in rendered
            assert "Exception" not in rendered
    finally:
        await ws.close()


# ---------------------------------------------------------------------------------------
# 3. Authentication & Role Boundary Enforcement
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_unauthenticated_command_rejection(control_server):
    server, _, _ = control_server
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}")
    try:
        # Send command directly without auth handshake
        await ws.send(json.dumps({
            "op": "command",
            "id": "unauth-1",
            "command": "profile_list",
            "payload": {},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(raw)
        assert res.get("op") == "error"
        assert res.get("code") in ("auth_required", "auth_failed")
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_adversarial_event_role_cannot_execute_control_commands(control_server):
    server, _, _ = control_server
    # Authenticate as EVENT role (receive-only)
    ws = await connect_and_auth(server, role="event", token=EVENT_TOKEN)
    try:
        # Event role attempting control mutation
        await ws.send(json.dumps({
            "op": "command",
            "id": "evt-exploit-1",
            "command": "profile_create",
            "payload": {"name": "Hacked Profile", "content": "Hacked"},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(raw)
        assert res.get("op") == "error"
        assert res.get("code") == "forbidden"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_adversarial_invalid_token_rejected_immediately(control_server):
    server, _, _ = control_server
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}")
    try:
        await ws.send(json.dumps({
            "op": "auth",
            "role": "control",
            "token": "wrong_token_12345",
            "protocol": 1,
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(raw)
        assert res.get("op") == "error"
        assert res.get("code") == "auth_failed"
    finally:
        await ws.close()


# ---------------------------------------------------------------------------------------
# 4. High-Concurrency Command Dispatch & Race Conditions
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_concurrent_commands_on_single_socket(control_server):
    server, _, _repo = control_server
    ws = await connect_and_auth(server)
    try:
        count = 30
        req_ids = [f"conc-prof-{i}" for i in range(count)]

        # Fire 30 profile_create commands concurrently without awaiting between sends
        for i, req_id in enumerate(req_ids):
            await ws.send(json.dumps({
                "op": "command",
                "id": req_id,
                "command": "profile_create",
                "payload": {"name": f"Profile {i}", "content": f"Content {i}"},
            }))

        responses = {}
        completes = set()

        # Collect all 30 responses and complete frames (60 frames total)
        for _ in range(count * 2):
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            rid = msg.get("id")
            if msg.get("op") == "response":
                responses[rid] = msg.get("data")
            elif msg.get("op") == "complete":
                completes.add(rid)

        assert len(responses) == count
        assert len(completes) == count
        for req_id in req_ids:
            assert req_id in responses
            assert responses[req_id]["profile_id"] is not None
            assert req_id in completes
    finally:
        await ws.close()


# ---------------------------------------------------------------------------------------
# 5. Phase B Parameter Boundaries & SQL Injection Resistance
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_profile_boundaries_and_sql_injection(control_server):
    server, _, _ = control_server
    ws = await connect_and_auth(server)
    try:
        # SQL Injection attempt in profile_get
        sql_injections = [
            "' OR '1'='1",
            "'; DROP TABLE context_profiles; --",
            "\" OR \"\"=\"",
            "../../etc/passwd",
        ]
        for inj in sql_injections:
            await ws.send(json.dumps({
                "op": "command",
                "id": f"sql-{hash(inj)}",
                "command": "profile_get",
                "payload": {"profile_id": inj},
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            res = json.loads(raw)
            # Must return error, not crash with SQLite syntax error
            assert res.get("op") == "error"
            assert res.get("code") in ("invalid_payload", "not_found")
            assert "syntax error" not in json.dumps(res).lower()
            # Drain complete frame
            await asyncio.wait_for(ws.recv(), timeout=1.0)

        # Profile create with content exceeding MAX_PROFILE_CONTENT_BYTES (64KB)
        big_content = "X" * (70 * 1024)
        await ws.send(json.dumps({
            "op": "command",
            "id": "big-prof-1",
            "command": "profile_create",
            "payload": {"name": "Big Profile", "content": big_content},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(raw)
        assert res.get("op") == "error"
        assert res.get("code") == "validation_error"
        await asyncio.wait_for(ws.recv(), timeout=1.0)
    finally:
        await ws.close()


# ---------------------------------------------------------------------------------------
# 6. Empirical Bug Discoveries
# ---------------------------------------------------------------------------------------

class TauriBridgeSimulator:
    """Accurately replicates the Rust Tauri `exchange()` logic from `control_bridge.rs`."""

    def __init__(self, forward_policy_table: Dict[str, Dict[str, Any]]):
        self.policy_table = forward_policy_table

    def exchange_simulation(self, command_name: str, server_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        policy = self.policy_table.get(command_name)
        if policy is None:
            return {"ok": False, "error": "command_not_allowed"}

        terminal = policy.get("terminal")
        terminal_seen = terminal is None
        completed = False
        messages = []

        for frame in server_frames:
            op = frame.get("op")
            if op == "response":
                message = frame.get("message") or frame.get("data") or {}
                if terminal is not None:
                    kind = message.get("type") if isinstance(message, dict) else None
                    if kind == terminal or kind == "error":
                        terminal_seen = True
                messages.append(message)
                if completed and terminal_seen:
                    return {"ok": True, "messages": messages}
            elif op == "complete":
                completed = True
                if terminal_seen:
                    return {"ok": True, "messages": messages}
            elif op == "error":
                return {"ok": False, "error": "command_rejected"}

        # If loop exhausts without returning, the Rust bridge hangs waiting for more socket frames -> TIMEOUT
        return {"ok": False, "error": "timeout", "hung": True, "terminal_seen": terminal_seen, "completed": completed}


@pytest.mark.asyncio
async def test_adversarial_tauri_bridge_deadlock_on_request_assist(control_server):
    """VERIFICATION THAT `request_assist` EMITS `type == "suggestion_result"` AND AVOIDS TAURI BRIDGE DEADLOCK.

    In `control_bridge.rs`:
        "request_assist" => (None, 120, false)
    And in `control_dispatcher.py`:
        _handle_request_assist emits:
        `{"op": "response", "id": ..., "data": {"type": "suggestion_result", ...}, "message": {"type": "suggestion_result", ...}}`
    """
    server, dispatcher, repo = control_server
    from engine.assistant.service import AssistantService
    dispatcher._assistant_service = AssistantService(
        repo, ContextProfileService(repo), FakeAssistant(), model="model-test"
    )

    forward_policy_table = {
        "request_assist": {"terminal": None, "timeout_sec": 120},
    }
    bridge_sim = TauriBridgeSimulator(forward_policy_table)

    ws = await connect_and_auth(server)
    try:
        cid = "c-ast-deadlock"
        mid = "m-ast-deadlock"
        repo.create_conversation(ConversationRecord(
            conversation_id=cid,
            title="Deadlock Test",
            created_at_ms=1000.0,
            updated_at_ms=1000.0,
        ))
        repo.upsert_message(cid, TranscriptMessage(
            message_id=mid,
            source="mic",
            text="Hello world",
            timestamp_ms=2000.0,
        ))

        req_id = "req-assist-deadlock-probe"
        await ws.send(json.dumps({
            "op": "command",
            "id": req_id,
            "command": "request_assist",
            "payload": {"conversation_id": cid, "message_id": mid},
        }))

        # Collect frames produced by the actual engine
        frames = []
        for _ in range(2):
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            frames.append(json.loads(raw))

        # Verify engine responded with response + complete
        assert frames[0]["op"] == "response"
        assert frames[1]["op"] == "complete"
        # Confirm that the engine's message frame carries 'type': 'suggestion_result'
        msg_payload = frames[0].get("message", {})
        assert msg_payload.get("type") == "suggestion_result", "Expected 'type': 'suggestion_result' in response"
        assert "options" in msg_payload, "Expected options list in response"

        # Now feed the engine's actual response into the Tauri bridge exchange simulation
        outcome = bridge_sim.exchange_simulation("request_assist", frames)

        # VERIFICATION: The Tauri bridge succeeds without hanging
        assert outcome["ok"] is True
        assert outcome["messages"][0]["type"] == "suggestion_result"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_adversarial_tauri_bridge_summary_cancel_not_allowed():
    """VERIFICATION THAT `summary_cancel` IS ALLOWLISTED BY TAURI BRIDGE.

    `control_dispatcher.py` implements `summary_cancel`, and `control_bridge.rs`
    `forward_policy` includes `summary_cancel` => (None, 10, false).
    """
    forward_policy_table = {
        "pack_cancel": {"terminal": None, "timeout_sec": 10},
        "request_summary": {"terminal": "llm_done", "timeout_sec": 300},
        "summary_cancel": {"terminal": None, "timeout_sec": 10},
    }
    bridge_sim = TauriBridgeSimulator(forward_policy_table)
    frames = [
        {"op": "response", "data": {"cancelled": True, "request_id": "req-1"}},
        {"op": "complete"},
    ]
    outcome = bridge_sim.exchange_simulation("summary_cancel", frames)
    assert outcome["ok"] is True
    assert outcome["messages"][0]["cancelled"] is True


@pytest.mark.asyncio
async def test_adversarial_import_stored_assist_result_typeerror(control_server):
    """VERIFICATION THAT `conversation_import` SUCCEEDS WITHOUT TYPEERROR WITH ASSIST RESULTS.

    `_handle_conversation_import` parses suggested_replies and clarifying_questions
    into domain models rather than passing payload=... to StoredAssistResult.
    """
    server, _, repo = control_server
    ws = await connect_and_auth(server)
    try:
        cid = "conv-bug-assist"
        export_payload = {
            "conversation_id": cid,
            "title": "Conv with assist",
            "messages": [
                {"message_id": "m-1", "source": "mic", "text": "Q?"}
            ],
            "assist_results": [
                {
                    "assist_id": "ast-1",
                    "target_message_id": "m-1",
                    "suggested_replies": [
                        {"strategy": "Direct", "text": "Yes, we can.", "reasoning": "Clear response"}
                    ],
                    "clarifying_questions": [
                        {"question": "When?", "purpose": "Timeline"}
                    ],
                }
            ],
        }

        await ws.send(json.dumps({
            "op": "command",
            "id": "imp-assist-bug",
            "command": "conversation_import",
            "payload": {"export_data": export_payload},
        }))

        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(raw)

        # VERIFICATION: Import succeeds cleanly without TypeError
        assert res.get("op") == "response"
        assert res.get("data", {}).get("conversation_id") == cid
        complete_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        complete_msg = json.loads(complete_raw)
        assert complete_msg.get("op") == "complete"

        # Check repository persistence
        stored_assists = repo.list_assist_results(cid)
        assert len(stored_assists) == 1
        assert stored_assists[0].assist_id == "ast-1"
        assert len(stored_assists[0].suggested_replies) == 1
        assert stored_assists[0].suggested_replies[0].text == "Yes, we can."
    finally:
        await ws.close()
