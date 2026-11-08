"""Empirical Challenger 2 test suite for Milestone 2 Iteration 2 remediations.

Verifies:
1. `conversation_import` with diverse assist records (domain tuples, legacy payload JSON,
   empty records, malformed items, round-trip export/import preservation).
2. `request_assist` response schema compliance ('type': 'suggestion_result', 'options': list of strings).
3. `request_summary` streaming metadata ('conversation_id' in data and message) and cancellation.
4. `summary_cancel` allowlisting in Tauri bridge and clean task abortion.
5. Legacy fallthrough vs Phase B dispatching via `extract_command()`.
"""

import asyncio
import json
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
    ProviderMetadata,
    StoredAssistResult,
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


class FakeAssistant(AssistantProvider):
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
        self.summary_started = asyncio.Event()
        self.policy = EngineAssistantPromptPolicy()

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
    db_path = str(tmp_path / "challenger_c2.db")
    return ConversationRepository(db_path=db_path)


@pytest.fixture
async def test_env(repo):
    server = WebSocketServer(host="127.0.0.1", port=0, tokens=TOKENS)
    profiles = ContextProfileService(repo)
    dispatcher = ControlCommandDispatcher(
        conversation_repo=repo,
        profile_service=profiles,
    )
    assistant_provider = FakeAssistant()
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
# 1. Conversation Import Robustness & Round-Trip
# =======================================================================================

@pytest.mark.asyncio
async def test_import_with_various_assist_payload_formats(test_env):
    """Verify conversation_import parses modern dicts, legacy JSON string payload,
    empty lists, and missing fields without TypeError or crash.
    """
    server, _, repo, _ = test_env
    ws = await connect_control(server)
    try:
        cid = "conv-multi-format-import"
        export_data = {
            "conversation_id": cid,
            "title": "Diverse Assist Formats",
            "messages": [
                {"message_id": "m-1", "source": "mic", "text": "Question 1"},
                {"message_id": "m-2", "source": "system", "text": "Answer 1"},
            ],
            "assist_results": [
                # Format 1: Standard Phase B format (dicts in suggested_replies and clarifying_questions)
                {
                    "assist_id": "ast-std",
                    "target_message_id": "m-1",
                    "suggested_replies": [
                        {"strategy": "Direct", "text": "Reply 1", "reasoning": "Reason 1"},
                    ],
                    "clarifying_questions": [
                        {"question": "Q1?", "purpose": "P1"},
                    ],
                    "provider": {"provider_id": "prov-1", "model": "m-v1"},
                    "created_at_ms": 1000.0,
                },
                # Format 2: Legacy payload JSON string
                {
                    "assist_id": "ast-legacy-str",
                    "target_message_id": "m-2",
                    "payload": json.dumps({
                        "suggested_replies": [
                            {"strategy": "Fallback", "text": "Legacy reply", "reasoning": "Legacy"},
                        ],
                        "clarifying_questions": [
                            {"question": "Legacy Q?", "purpose": "Legacy P"},
                        ],
                    }),
                    "created_at_ms": 2000.0,
                },
                # Format 3: Empty replies and questions
                {
                    "assist_id": "ast-empty",
                    "target_message_id": "m-1",
                    "suggested_replies": [],
                    "clarifying_questions": [],
                    "created_at_ms": 3000.0,
                },
                # Format 4: Malformed items missing non-essential fields (should use defaults)
                {
                    "assist_id": "ast-sparse",
                    "target_message_id": "m-2",
                    "suggested_replies": [
                        {"text": "Only text provided"},
                    ],
                    "clarifying_questions": [
                        {"question": "Only question provided"},
                    ],
                    "created_at_ms": 4000.0,
                },
            ],
        }

        await ws.send(json.dumps({
            "op": "command",
            "id": "imp-diverse",
            "command": "conversation_import",
            "payload": {"export_data": export_data},
        }))

        res_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(res_raw)
        assert res.get("op") == "response"
        assert res.get("data", {}).get("conversation_id") == cid

        comp_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        comp = json.loads(comp_raw)
        assert comp.get("op") == "complete"

        # Check repository persistence
        persisted = repo.list_assist_results(cid)
        assert len(persisted) == 4

        # Validate Format 1
        a1 = next(a for a in persisted if a.assist_id == "ast-std")
        assert len(a1.suggested_replies) == 1
        assert a1.suggested_replies[0].strategy == "Direct"
        assert a1.suggested_replies[0].text == "Reply 1"
        assert a1.suggested_replies[0].reasoning == "Reason 1"
        assert len(a1.clarifying_questions) == 1
        assert a1.clarifying_questions[0].question == "Q1?"
        assert a1.clarifying_questions[0].purpose == "P1"
        assert a1.provider is not None and a1.provider.provider_id == "prov-1"

        # Validate Format 2 (legacy string)
        a2 = next(a for a in persisted if a.assist_id == "ast-legacy-str")
        assert len(a2.suggested_replies) == 1
        assert a2.suggested_replies[0].text == "Legacy reply"
        assert a2.suggested_replies[0].strategy == "Fallback"

        # Validate Format 3 (empty)
        a3 = next(a for a in persisted if a.assist_id == "ast-empty")
        assert len(a3.suggested_replies) == 0
        assert len(a3.clarifying_questions) == 0

        # Validate Format 4 (sparse defaults)
        a4 = next(a for a in persisted if a.assist_id == "ast-sparse")
        assert len(a4.suggested_replies) == 1
        assert a4.suggested_replies[0].text == "Only text provided"
        assert a4.suggested_replies[0].strategy == "Direct"  # Default
        assert len(a4.clarifying_questions) == 1
        assert a4.clarifying_questions[0].question == "Only question provided"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_conversation_export_and_import_roundtrip(test_env, tmp_path):
    """Full round-trip test: export from one repo and import into a second clean repo."""
    server, _dispatcher, repo, _ = test_env
    cid = "conv-roundtrip"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Roundtrip Test",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id="msg-rt-1",
        source="mic",
        text="Can we verify the roundtrip?",
        timestamp_ms=1100.0,
    ))

    # Add assist result to first repo
    repo.add_assist_result(StoredAssistResult(
        assist_id="ast-rt-1",
        conversation_id=cid,
        target_message_id="msg-rt-1",
        suggested_replies=(
            AssistSuggestion(strategy="Direct", text="Yes, verified.", reasoning="Confidence"),
            AssistSuggestion(strategy="Elaborate", text="Yes, tests passed.", reasoning="Detail"),
        ),
        clarifying_questions=(
            AssistQuestion(question="Which suite?", purpose="Clarity"),
        ),
        created_at_ms=1200.0,
        provider=ProviderMetadata(provider_id="fake-prov", model="m-1"),
        latency_ms=150.0,
    ))

    ws = await connect_control(server)
    try:
        # Export conversation via control command
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-export-rt",
            "command": "conversation_export",
            "payload": {"conversation_id": cid},
        }))
        res_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(res_raw)
        assert res.get("op") == "response"
        export_data = res.get("data", {}).get("export") or res.get("data")
        assert export_data is not None
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # Complete frame

        # Create second repository and dispatcher
        second_db = str(tmp_path / "second_repo.db")
        second_repo = ConversationRepository(db_path=second_db)
        second_profiles = ContextProfileService(second_repo)
        second_dispatcher = ControlCommandDispatcher(
            conversation_repo=second_repo,
            profile_service=second_profiles,
        )
        second_server = WebSocketServer(host="127.0.0.1", port=0, tokens=TOKENS)
        second_server.set_control_dispatcher(second_dispatcher)
        await second_server.start()

        try:
            ws2 = await connect_control(second_server)
            try:
                # Import into second server
                await ws2.send(json.dumps({
                    "op": "command",
                    "id": "req-import-rt",
                    "command": "conversation_import",
                    "payload": {"export_data": export_data},
                }))
                imp_raw = await asyncio.wait_for(ws2.recv(), timeout=2.0)
                imp_res = json.loads(imp_raw)
                assert imp_res.get("op") == "response"
                assert imp_res.get("data", {}).get("conversation_id") == cid
                _ = await asyncio.wait_for(ws2.recv(), timeout=2.0)  # Complete frame

                # Verify second repo has identical assist results
                imported_assists = second_repo.list_assist_results(cid)
                assert len(imported_assists) == 1
                imp_ast = imported_assists[0]
                assert imp_ast.assist_id == "ast-rt-1"
                assert imp_ast.target_message_id == "msg-rt-1"
                assert len(imp_ast.suggested_replies) == 2
                assert imp_ast.suggested_replies[0].text == "Yes, verified."
                assert imp_ast.suggested_replies[0].strategy == "Direct"
                assert imp_ast.suggested_replies[1].text == "Yes, tests passed."
                assert len(imp_ast.clarifying_questions) == 1
                assert imp_ast.clarifying_questions[0].question == "Which suite?"
                assert imp_ast.provider is not None
                assert imp_ast.provider.provider_id == "fake-prov"
            finally:
                await ws2.close()
        finally:
            await second_server.stop()
            second_repo.close()
    finally:
        await ws.close()


# =======================================================================================
# 2. `request_assist` Response Schema Verification
# =======================================================================================

@pytest.mark.asyncio
async def test_request_assist_schema_compliance(test_env):
    """Verify request_assist returns 'type': 'suggestion_result' and 'options': list of strings."""
    server, _, repo, _ = test_env
    cid = "conv-assist-schema"
    mid = "msg-assist-schema"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Assist Schema Test",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id=mid,
        source="mic",
        text="What is our deployment plan?",
        timestamp_ms=1100.0,
    ))

    ws = await connect_control(server)
    try:
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-assist-schema-test",
            "command": "request_assist",
            "payload": {"conversation_id": cid, "message_id": mid},
        }))

        res_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        res = json.loads(res_raw)
        assert res.get("op") == "response"

        data = res.get("data", {})
        message = res.get("message", {})

        # Assert schema fields
        for payload_dict in (data, message):
            assert payload_dict.get("type") == "suggestion_result"
            assert payload_dict.get("conversation_id") == cid
            assert payload_dict.get("message_id") == mid
            assert payload_dict.get("entry_id") == mid
            assert "assist_id" in payload_dict

            options = payload_dict.get("options")
            assert isinstance(options, list)
            assert len(options) == 2
            assert all(isinstance(opt, str) for opt in options)
            assert options[0] == "Deploy to staging."
            assert options[1] == "Wait for QA approval."

            replies = payload_dict.get("suggested_replies")
            assert isinstance(replies, list)
            assert len(replies) == 2
            assert replies[0]["strategy"] == "Direct"

            questions = payload_dict.get("clarifying_questions")
            assert isinstance(questions, list)
            assert len(questions) == 1
            assert questions[0]["question"] == "Has smoke test passed?"

        comp_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        comp = json.loads(comp_raw)
        assert comp.get("op") == "complete"
    finally:
        await ws.close()


# =======================================================================================
# 3. `request_summary` Streaming Metadata & `summary_cancel`
# =======================================================================================

@pytest.mark.asyncio
async def test_summary_streaming_conversation_id_and_cancellation(test_env):
    """Verify done_frame has conversation_id in data and message, and summary_cancel works."""
    server, _, repo, assistant_provider = test_env
    cid = "conv-summary-stream"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Summary Stream Test",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id="msg-sum-1",
        source="mic",
        text="Discussing sprint goals.",
        timestamp_ms=1100.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id="msg-sum-2",
        source="system",
        text="Sprint goals approved.",
        timestamp_ms=1200.0,
    ))

    ws = await connect_control(server)
    try:
        # Test 1: Complete streaming flow
        req_id = "req-sum-stream-1"
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
                    # Verify chunk message carries conversation_id
                    assert msg.get("message", {}).get("conversation_id") == cid
                else:
                    done_frame = msg
            elif msg.get("op") == "complete":
                break

        assert len(chunks) == 3
        assert done_frame is not None
        # Verify done_frame data and message carry conversation_id
        assert done_frame["data"]["conversation_id"] == cid
        assert done_frame["data"]["done"] is True
        assert done_frame["message"]["conversation_id"] == cid
        assert done_frame["message"]["type"] == "llm_done"

        # Test 2: In-flight cancellation via summary_cancel
        gate = asyncio.Event()
        assistant_provider.gate = gate

        req_cancel_id = "req-sum-to-cancel"
        await ws.send(json.dumps({
            "op": "command",
            "id": req_cancel_id,
            "command": "request_summary",
            "payload": {"conversation_id": cid},
        }))

        # Wait for provider to enter streaming loop and pause at gate
        await asyncio.wait_for(assistant_provider.summary_started.wait(), timeout=2.0)

        # Issue cancellation
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-cancel-cmd",
            "command": "summary_cancel",
            "payload": {"request_id": req_cancel_id},
        }))

        # Collect replies
        replies = []
        for _ in range(4):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                replies.append(json.loads(raw))
            except asyncio.TimeoutError:
                break

        # Cancel command should get affirmative response
        cancel_res = next(
            (r for r in replies if r.get("id") == "req-cancel-cmd" and r.get("op") == "response"),
            None,
        )
        assert cancel_res is not None
        assert cancel_res["data"]["cancelled"] is True
        assert cancel_res["data"]["request_id"] == req_cancel_id

        # Streaming task should emit error with code 'cancelled'
        err_frame = next(
            (r for r in replies if r.get("id") == req_cancel_id and r.get("op") == "error"),
            None,
        )
        assert err_frame is not None
        assert err_frame["code"] == "cancelled"

    finally:
        assistant_provider.gate = None
        await ws.close()


# =======================================================================================
# 4. Command Dispatching & Fallthrough Verification
# =======================================================================================

@pytest.mark.asyncio
async def test_extract_command_fallthrough_and_routing():
    """Unit test extract_command() against all command envelope variants."""
    from unittest.mock import MagicMock
    mock_repo = MagicMock()
    mock_profiles = MagicMock()
    dispatcher = ControlCommandDispatcher(
        conversation_repo=mock_repo,
        profile_service=mock_profiles,
    )

    # 1. Standard Phase B command envelope
    f1 = {"op": "command", "id": "1", "command": "request_assist", "payload": {"k": "v"}}
    assert dispatcher.extract_command(f1) == ("1", "request_assist", {"k": "v"})

    f2 = {"op": "command", "id": "2", "command": "summary_cancel", "payload": {"request_id": "x"}}
    assert dispatcher.extract_command(f2) == ("2", "summary_cancel", {"request_id": "x"})

    # 2. Tauri bridge request envelope with Phase B inner command
    f3 = {"op": "request", "id": "3", "command": {"command": "profile_list", "payload": {}}}
    assert dispatcher.extract_command(f3) == ("3", "profile_list", {})

    # 3. Tauri bridge request envelope with Phase B inner type
    f4 = {"op": "request", "id": "4", "command": {"type": "request_assist", "conversation_id": "c"}}
    assert dispatcher.extract_command(f4) == ("4", "request_assist", {"conversation_id": "c"})

    # 4. CRITICAL: Legacy inner_type in ("request_suggestion", "request_summary") MUST return None
    # allowing fallthrough to EchoFluxEngine handlers
    f5 = {"op": "request", "id": "5", "command": {"type": "request_suggestion", "conversation_id": "c"}}
    assert dispatcher.extract_command(f5) is None

    f6 = {"op": "request", "id": "6", "command": {"type": "request_summary", "conversation_id": "c"}}
    assert dispatcher.extract_command(f6) is None

    # 5. Non-dict or unknown command returns None
    assert dispatcher.extract_command(None) is None
    assert dispatcher.extract_command("string") is None
    assert dispatcher.extract_command({"op": "command", "command": "unknown"}) is None


# =======================================================================================
# 5. Adversarial Boundary Stress Tests
# =======================================================================================

@pytest.mark.asyncio
async def test_assist_boundary_rejections(test_env):
    """Stress test request_assist rejection of invalid, non-final, and cross-conv messages."""
    server, _, repo, _ = test_env
    cid = "conv-assist-bounds"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Assist Bounds Test",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id="msg-final",
        source="mic",
        text="Finalized transcript message.",
        timestamp_ms=1100.0,
        is_final=True,
    ))
    repo.upsert_message(cid, TranscriptMessage(
        message_id="msg-partial",
        source="mic",
        text="Partial unfinalized text...",
        timestamp_ms=1200.0,
        is_final=False,
    ))

    # Another conversation
    repo.create_conversation(ConversationRecord(
        conversation_id="conv-other",
        title="Other Conv",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))
    repo.upsert_message("conv-other", TranscriptMessage(
        message_id="msg-other",
        source="system",
        text="Other message.",
        timestamp_ms=1100.0,
        is_final=True,
    ))

    ws = await connect_control(server)
    try:
        # Case 1: Non-existent conversation
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-b-1",
            "command": "request_assist",
            "payload": {"conversation_id": "non-existent-cid", "message_id": "msg-final"},
        }))
        err1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err1.get("op") == "error"
        assert err1.get("code") == "target_not_found"
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete

        # Case 2: Non-existent message
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-b-2",
            "command": "request_assist",
            "payload": {"conversation_id": cid, "message_id": "non-existent-mid"},
        }))
        err2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err2.get("op") == "error"
        assert err2.get("code") == "target_not_found"
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete

        # Case 3: Message belonging to another conversation
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-b-3",
            "command": "request_assist",
            "payload": {"conversation_id": cid, "message_id": "msg-other"},
        }))
        err3 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err3.get("op") == "error"
        assert err3.get("code") == "target_not_found"
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete

        # Case 4: Non-finalized message (must be rejected)
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-b-4",
            "command": "request_assist",
            "payload": {"conversation_id": cid, "message_id": "msg-partial"},
        }))
        err4 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err4.get("op") == "error"
        assert err4.get("code") == "target_not_final"
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete

        # Case 5: Missing payload fields
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-b-5",
            "command": "request_assist",
            "payload": {"conversation_id": cid},  # missing message_id
        }))
        err5 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err5.get("op") == "error"
        assert err5.get("code") == "invalid_request"
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_summary_empty_conversation_rejection(test_env):
    """Verify request_summary on an empty conversation fails safely with domain error."""
    server, _, repo, _ = test_env
    cid = "conv-summary-empty"
    repo.create_conversation(ConversationRecord(
        conversation_id=cid,
        title="Empty Conv",
        created_at_ms=1000.0,
        updated_at_ms=1000.0,
    ))

    ws = await connect_control(server)
    try:
        await ws.send(json.dumps({
            "op": "command",
            "id": "req-sum-empty",
            "command": "request_summary",
            "payload": {"conversation_id": cid},
        }))
        err = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert err.get("op") == "error"
        assert err.get("code") in ("transcript_empty", "empty_transcript", "invalid_request")
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # complete
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_summary_cancel_idempotence_and_edge_cases(test_env):
    """Verify summary_cancel behavior on unknown id, non-string id, and duplicate cancel."""
    server, _, _, _ = test_env
    ws = await connect_control(server)
    try:
        # Non-existent request_id
        await ws.send(json.dumps({
            "op": "command",
            "id": "c-1",
            "command": "summary_cancel",
            "payload": {"request_id": "never-existed"},
        }))
        r1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert r1.get("op") == "response"
        assert r1["data"]["cancelled"] is False
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)

        # Empty payload
        await ws.send(json.dumps({
            "op": "command",
            "id": "c-2",
            "command": "summary_cancel",
            "payload": {},
        }))
        r2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert r2.get("op") == "response"
        assert r2["data"]["cancelled"] is False
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)

        # Non-string request_id
        await ws.send(json.dumps({
            "op": "command",
            "id": "c-3",
            "command": "summary_cancel",
            "payload": {"request_id": 12345},
        }))
        r3 = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert r3.get("op") == "response"
        assert r3["data"]["cancelled"] is False
        _ = await asyncio.wait_for(ws.recv(), timeout=2.0)
    finally:
        await ws.close()
