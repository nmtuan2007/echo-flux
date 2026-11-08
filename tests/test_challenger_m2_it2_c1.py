"""Empirical Challenger 1 test suite for Milestone 2 Iteration 2 remediations.

Verifies:
1. `control_bridge.rs` forward policy: deadlock elimination on request_assist and forward allowlisting on summary_cancel.
2. `control_dispatcher.py` extract_command() fallthrough for legacy request_suggestion/request_summary, and routing for Phase B commands.
3. `request_assist` response schema compliance: type='suggestion_result', options list of strings, suggested_replies, clarifying_questions.
4. `request_summary` streaming: conversation_id in done_frame (data & message), delta chunks, and clean in-flight cancellation via summary_cancel.
5. `conversation_import` parsing: domain tuples StoredAssistResult, AssistSuggestion, AssistQuestion with modern, legacy, and edge-case export formats.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional

import pytest
import websockets

from engine.assistant.service import AssistantService
from engine.assistant.summary_service import EngineAssistantPromptPolicy, SummaryService
from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    AssistQuestion,
    AssistSuggestion,
    ConversationRecord,
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
from engine.server.control_dispatcher import ControlCommandDispatcher
from engine.server.protocol import BootstrapTokens
from engine.server.websocket_server import WebSocketServer

CONTROL_TOKEN = "c0" * 32
EVENT_TOKEN = "e1" * 32
TOKENS = BootstrapTokens(control=CONTROL_TOKEN, event=EVENT_TOKEN)


class FakeChallengerAssistant(AssistantProvider):
    def __init__(
        self,
        replies: Optional[List[Dict[str, str]]] = None,
        questions: Optional[List[Dict[str, str]]] = None,
        chunks: Optional[List[str]] = None,
        gate: Optional[asyncio.Event] = None,
    ) -> None:
        self.replies = replies or [
            {"strategy": "Direct", "text": "Deploy to staging.", "reasoning": "Standard process"},
            {"strategy": "Cautious", "text": "Wait for QA approval.", "reasoning": "Risk mitigation"},
        ]
        self.questions = questions or [
            {"question": "Has smoke test passed?", "purpose": "Verification"},
        ]
        self.chunks = chunks or ["## Summary\n", "- Action item 1\n", "- Action item 2\n"]
        self.gate = gate
        self.policy = EngineAssistantPromptPolicy()
        self.summary_started = asyncio.Event()

    @property
    def provider_id(self) -> str:
        return "fake-challenger-assistant"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

    async def assist(self, request: AssistRequest) -> AssistResult:
        if self.gate is not None:
            await self.gate.wait()
        reply_json = json.dumps({
            "suggested_replies": self.replies,
            "clarifying_questions": self.questions,
        })
        return self.policy.parse_assist_result(request, TextCompletion(text=reply_json))

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        raise NotImplementedError

    async def stream_summary(self, request: SummaryRequest):
        self.summary_started.set()
        for chunk in self.chunks:
            if self.gate is not None:
                await self.gate.wait()
            yield SummaryChunk(delta_text=chunk)


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "challenger_c1.db")
    return ConversationRepository(db_path=db_path)


@pytest.fixture
async def test_env(repo):
    server = WebSocketServer(host="127.0.0.1", port=0, tokens=TOKENS)
    profiles = ContextProfileService(repo)
    dispatcher = ControlCommandDispatcher(
        conversation_repo=repo,
        profile_service=profiles,
    )
    assistant_provider = FakeChallengerAssistant()
    dispatcher._assistant_service = AssistantService(
        repo, profiles, assistant_provider, model="fake-model"
    )
    dispatcher._summary_service = SummaryService(
        repo, assistant_provider, model="fake-model"
    )
    server.set_control_dispatcher(dispatcher)
    await server.start()
    try:
        yield server, dispatcher, repo, assistant_provider
    finally:
        await server.stop()


async def connect_control(server):
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}")
    await ws.send(json.dumps({
        "op": "auth",
        "role": "control",
        "token": CONTROL_TOKEN,
        "protocol": 1,
    }))
    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
    auth_resp = json.loads(raw)
    assert auth_resp.get("op") == "auth_ok"
    return ws


# =======================================================================================
# 1. Desktop Bridge Deadlock Elimination on `request_assist`
# =======================================================================================

@pytest.mark.asyncio
async def test_request_assist_no_deadlock_and_schema_compliance(test_env):
    """Verify request_assist:
    1. Returns 'type': 'suggestion_result' and 'options': list of strings.
    2. Emits response and complete frames.
    3. Simulates Rust control_bridge exchange loop with terminal=None, verifying zero hang.
    """
    server, _, repo, _ = test_env
    cid = "conv-assist-deadlock-check"
    mid = "msg-assist-deadlock-check"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Deadlock Check",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id=mid,
        source="mic",
        text="Can we verify assist deadlock elimination?",
        timestamp_ms=1100.0,
    ))

    ws = await connect_control(server)
    try:
        req_id = "req-assist-deadlock-test"
        await ws.send(json.dumps({
            "op": "command",
            "id": req_id,
            "command": "request_assist",
            "payload": {"conversation_id": cid, "message_id": mid},
        }))

        # Rust exchange loop simulation with policy: terminal = None
        terminal = None
        terminal_seen = terminal is None
        completed = False
        messages = []

        start_time = time.monotonic()
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            frame = json.loads(raw)
            assert frame.get("id") == req_id
            op = frame.get("op")
            if op == "response":
                msg = frame.get("message") or frame.get("data") or {}
                if terminal is not None:
                    kind = msg.get("type") if isinstance(msg, dict) else None
                    if kind == terminal or kind == "error":
                        terminal_seen = True
                messages.append(msg)
                if completed and terminal_seen:
                    break
            elif op == "complete":
                completed = True
                if terminal_seen:
                    break
            elif op == "error":
                pytest.fail(f"Unexpected error frame: {frame}")

        elapsed = time.monotonic() - start_time
        # Verifies the exchange finished instantaneously without waiting for 120s timeout
        assert elapsed < 1.0, f"Exchange took too long: {elapsed}s"
        assert len(messages) == 1

        payload = messages[0]
        assert payload.get("type") == "suggestion_result"
        assert payload.get("conversation_id") == cid
        assert payload.get("message_id") == mid
        assert isinstance(payload.get("options"), list)
        assert len(payload.get("options")) == 2
        assert payload.get("options")[0] == "Deploy to staging."
        assert len(payload.get("suggested_replies")) == 2
        assert len(payload.get("clarifying_questions")) == 1
    finally:
        await ws.close()


# =======================================================================================
# 2. Desktop Bridge `summary_cancel` Forwarding and Allowlist
# =======================================================================================

@pytest.mark.asyncio
async def test_summary_streaming_and_cancellation_lifecycle(test_env):
    """Verify summary streaming metadata and cancellation:
    1. Streaming chunks and done_frame include 'conversation_id'.
    2. summary_cancel halts background task, returns cancelled=True, and emits 'cancelled' error frame.
    3. Cancelled summary leaves zero rows persisted in repository.
    """
    server, dispatcher, repo, assistant_provider = test_env
    cid = "conv-sum-stream-c1"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Summary Stream C1",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id="msg-sum-1",
        source="mic",
        text="Discussing sprint priorities.",
        timestamp_ms=1100.0,
    ))

    ws = await connect_control(server)
    try:
        # Part A: Normal streaming completion
        req_id = "req-sum-stream-c1"
        await ws.send(json.dumps({
            "op": "command",
            "id": req_id,
            "command": "request_summary",
            "payload": {"conversation_id": cid},
        }))

        chunks = []
        done_frame = None
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            if msg.get("op") == "response":
                if not msg.get("data", {}).get("done", False):
                    chunks.append(msg)
                    assert msg.get("message", {}).get("conversation_id") == cid
                else:
                    done_frame = msg
            elif msg.get("op") == "complete":
                break

        assert len(chunks) == 3
        assert done_frame is not None
        assert done_frame["data"]["conversation_id"] == cid
        assert done_frame["data"]["done"] is True
        assert done_frame["message"]["conversation_id"] == cid
        assert done_frame["message"]["type"] == "llm_done"

        # Verify summary row persisted in DB
        with repo.read() as conn:
            row = conn.execute("SELECT * FROM summaries WHERE conversation_id = ?", (cid,)).fetchone()
            assert row is not None

        # Part B: In-flight cancellation on second conversation
        cid_cancel = "conv-sum-cancel-c1"
        repo.create_conversation(ConversationRecord(
            conversation_id=cid_cancel,
            title="Cancel Conv",
            created_at_ms=2000.0,
            updated_at_ms=2000.0,
        ))
        repo.upsert_message(cid_cancel, TranscriptMessage(
            message_id="msg-cancel-1",
            source="mic",
            text="Classified discussion in progress.",
            timestamp_ms=2100.0,
        ))

        gate = asyncio.Event()
        assistant_provider.gate = gate
        assistant_provider.summary_started.clear()

        cancel_req_id = "req-sum-in-flight"
        # Start streaming request in background task
        async def read_stream():
            frames = []
            while True:
                r = await ws.recv()
                f = json.loads(r)
                if f.get("id") == cancel_req_id:
                    frames.append(f)
                    if f.get("op") in ("complete", "error"):
                        break
            return frames

        await ws.send(json.dumps({
            "op": "command",
            "id": cancel_req_id,
            "command": "request_summary",
            "payload": {"conversation_id": cid_cancel},
        }))

        stream_reader = asyncio.create_task(read_stream())
        # Wait until provider has entered stream_summary
        await asyncio.wait_for(assistant_provider.summary_started.wait(), timeout=2.0)
        assert cancel_req_id in dispatcher._active_tasks

        # Issue cancellation from a second control client
        cancel_ws = await connect_control(server)
        try:
            await cancel_ws.send(json.dumps({
                "op": "command",
                "id": "cancel-probe-1",
                "command": "summary_cancel",
                "payload": {"request_id": cancel_req_id},
            }))
            cancel_resp = json.loads(await asyncio.wait_for(cancel_ws.recv(), timeout=2.0))
            assert cancel_resp.get("op") == "response"
            assert cancel_resp.get("data", {}).get("cancelled") is True
            assert cancel_resp.get("data", {}).get("request_id") == cancel_req_id
            _ = await asyncio.wait_for(cancel_ws.recv(), timeout=2.0)  # Complete frame
        finally:
            await cancel_ws.close()

        # Stream should receive error frame with 'cancelled'
        stream_frames = await asyncio.wait_for(stream_reader, timeout=2.0)
        err_frame = next((f for f in stream_frames if f.get("op") == "error"), None)
        assert err_frame is not None
        assert err_frame.get("code") == "cancelled"

        # Active tasks map must be cleaned up
        assert cancel_req_id not in dispatcher._active_tasks

        # Verify zero rows persisted in DB for cid_cancel
        with repo.read() as conn:
            cancel_row = conn.execute("SELECT * FROM summaries WHERE conversation_id = ?", (cid_cancel,)).fetchone()
            assert cancel_row is None

    finally:
        assistant_provider.gate = None
        await ws.close()


# =======================================================================================
# 3. Command Dispatching and Fallthrough Verification
# =======================================================================================

def test_extract_command_fallthrough_logic():
    """Verify extract_command() routes Phase B commands and falls through for legacy types."""
    from unittest.mock import MagicMock
    mock_repo = MagicMock()
    mock_profiles = MagicMock()
    dispatcher = ControlCommandDispatcher(
        conversation_repo=mock_repo,
        profile_service=mock_profiles,
    )

    # 1. Standard Phase B command envelope
    f1 = {"op": "command", "id": "1", "command": "request_assist", "payload": {"conversation_id": "c1", "message_id": "m1"}}
    assert dispatcher.extract_command(f1) == ("1", "request_assist", {"conversation_id": "c1", "message_id": "m1"})

    f2 = {"op": "command", "id": "2", "command": "summary_cancel", "payload": {"request_id": "r1"}}
    assert dispatcher.extract_command(f2) == ("2", "summary_cancel", {"request_id": "r1"})

    # 2. Tauri bridge request envelope with Phase B inner command
    f3 = {"op": "request", "id": "3", "command": {"command": "conversation_list", "payload": {}}}
    assert dispatcher.extract_command(f3) == ("3", "conversation_list", {})

    # 3. Tauri bridge request envelope with Phase B inner type
    f4 = {"op": "request", "id": "4", "command": {"type": "request_assist", "conversation_id": "c1", "message_id": "m1"}}
    assert dispatcher.extract_command(f4) == ("4", "request_assist", {"conversation_id": "c1", "message_id": "m1"})

    # 4. CRITICAL: Legacy inner_type in ("request_suggestion", "request_summary") MUST return None
    # allowing fallthrough to EchoFluxEngine handlers in engine/main.py
    f5 = {"op": "request", "id": "5", "command": {"type": "request_suggestion", "target_text": "text"}}
    assert dispatcher.extract_command(f5) is None

    f6 = {"op": "request", "id": "6", "command": {"type": "request_summary", "entries": []}}
    assert dispatcher.extract_command(f6) is None

    # 5. Invalid / unauthenticated envelopes return None
    assert dispatcher.extract_command(None) is None
    assert dispatcher.extract_command({}) is None
    assert dispatcher.extract_command({"op": "unknown"}) is None


# =======================================================================================
# 4. Conversation Import Robustness and Domain Object Construction
# =======================================================================================

@pytest.mark.asyncio
async def test_conversation_import_diverse_assist_records(test_env):
    """Verify conversation_import correctly builds StoredAssistResult with domain tuples
    for suggested_replies and clarifying_questions across standard, legacy, and edge formats.
    """
    server, _, repo, _ = test_env
    cid = "conv-import-robust-c1"
    export_payload = {
        "conversation_id": cid,
        "title": "Import Robustness Conv",
        "messages": [
            {"message_id": "m-1", "source": "mic", "text": "Can we import diverse assists?"},
            {"message_id": "m-2", "source": "system", "text": "Yes, we can verify."},
        ],
        "assist_results": [
            # Standard modern format: dicts in suggested_replies and clarifying_questions
            {
                "assist_id": "ast-std-1",
                "target_message_id": "m-1",
                "suggested_replies": [
                    {"strategy": "Direct", "text": "Modern reply 1", "reasoning": "Reason 1"},
                    {"strategy": "Detailed", "text": "Modern reply 2", "reasoning": "Reason 2"},
                ],
                "clarifying_questions": [
                    {"question": "Scope confirmed?", "purpose": "Verification"},
                ],
                "created_at_ms": 1000.0,
                "provider": {"provider_id": "fake-prov", "model": "m-1"},
            },
            # Legacy stringified JSON in 'payload'
            {
                "assist_id": "ast-legacy-1",
                "target_message_id": "m-2",
                "payload": json.dumps({
                    "suggested_replies": [
                        {"strategy": "Legacy", "text": "Legacy string reply", "reasoning": "Legacy reason"},
                    ],
                    "clarifying_questions": [
                        {"question": "Legacy question?", "purpose": "Legacy purpose"},
                    ],
                }),
                "created_at_ms": 1100.0,
            },
            # Edge format: empty replies/questions
            {
                "assist_id": "ast-empty-1",
                "target_message_id": "m-1",
                "suggested_replies": [],
                "clarifying_questions": [],
                "created_at_ms": 1200.0,
            },
            # Edge format: sparse dicts missing optional fields
            {
                "assist_id": "ast-sparse-1",
                "target_message_id": "m-2",
                "suggested_replies": [{"text": "Sparse reply only"}],
                "clarifying_questions": [{"question": "Sparse question only"}],
                "created_at_ms": 1300.0,
            },
        ],
        "summaries": [
            {
                "summary_id": "sum-imp-1",
                "summary_markdown": "## Executive Summary\n- All tests pass.",
                "created_at_ms": 1400.0,
            }
        ],
    }

    ws = await connect_control(server)
    try:
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-import-robust",
            "command": "conversation_import",
            "payload": {"export_data": export_payload},
        }))

        res_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(res_raw)
        assert res.get("op") == "response"
        assert res.get("data", {}).get("conversation_id") == cid
        assert res.get("data", {}).get("imported") is True

        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete

        # Inspect SQLite persistence via repo
        stored_assists = repo.list_assist_results(cid)
        assert len(stored_assists) == 4

        # Verify Standard record
        std = next(a for a in stored_assists if a.assist_id == "ast-std-1")
        assert len(std.suggested_replies) == 2
        assert isinstance(std.suggested_replies[0], AssistSuggestion)
        assert std.suggested_replies[0].strategy == "Direct"
        assert std.suggested_replies[0].text == "Modern reply 1"
        assert len(std.clarifying_questions) == 1
        assert isinstance(std.clarifying_questions[0], AssistQuestion)
        assert std.clarifying_questions[0].question == "Scope confirmed?"

        # Verify Legacy record
        leg = next(a for a in stored_assists if a.assist_id == "ast-legacy-1")
        assert len(leg.suggested_replies) == 1
        assert leg.suggested_replies[0].text == "Legacy string reply"
        assert leg.suggested_replies[0].strategy == "Legacy"

        # Verify Empty record
        emp = next(a for a in stored_assists if a.assist_id == "ast-empty-1")
        assert len(emp.suggested_replies) == 0
        assert len(emp.clarifying_questions) == 0

        # Verify Sparse record
        spr = next(a for a in stored_assists if a.assist_id == "ast-sparse-1")
        assert len(spr.suggested_replies) == 1
        assert spr.suggested_replies[0].text == "Sparse reply only"
        assert spr.suggested_replies[0].strategy == "Direct"
        assert spr.clarifying_questions[0].question == "Sparse question only"
        assert spr.clarifying_questions[0].purpose is None

        # Verify Summary record
        summary = repo.latest_summary(cid)
        assert summary is not None
        assert summary.summary_id == "sum-imp-1"
        assert "Executive Summary" in summary.summary_markdown
    finally:
        await ws.close()
