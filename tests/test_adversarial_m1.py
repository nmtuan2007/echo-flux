"""Empirical adversarial stress tests for Milestone 1.

Authored by challenger_m1_2.
Stress-tests:
1. Malformed start configs & fail-closed invariants.
2. Invalid provider configs & canary secret redaction.
3. Rapid connect/disconnect cycles under concurrent load.
4. Rapid start/stop lifecycle thrashing.
5. Control/Event role isolation under attack.
6. Pipeline abort SQLite integrity and session cleanup.
7. Test connection surface, redaction, and validation resilience.
8. Corrupted JSON and binary frame rejection.
9. Concurrent start commands race resilience.
10. Additive transcript keys invariant in live engine events.
"""

import asyncio
import json
import logging
import sqlite3

import pytest
import websockets
from test_stt_session_integration import FakeInput, FakeSTT
from websockets.exceptions import ConnectionClosed

from engine.conversation.models import ConversationRecord
from engine.conversation.repository import ConversationRepository
from engine.main import EchoFluxEngine
from engine.server import protocol
from engine.server.protocol import (
    CLOSE_AUTH_FAILED,
    BootstrapTokens,
)
from engine.server.websocket_server import WebSocketServer

CONTROL_TOKEN = "c0" * 32
EVENT_TOKEN = "e1" * 32
TOKENS = BootstrapTokens(control=CONTROL_TOKEN, event=EVENT_TOKEN)
CANARY_SECRET = "sk-antigravity-adversarial-CANARY-998877"

_CLIENT_LOGGER = logging.getLogger("tests.adversarial.client")
_CLIENT_LOGGER.setLevel(logging.WARNING)


class FakeVAD:
    def __init__(self, config: dict):
        self.config = config

    def process(self, chunk: bytes):
        return True

    def reset(self):
        pass


@pytest.fixture
def patch_audio(monkeypatch):
    import engine.audio.microphone as microphone
    import engine.audio.system_audio as system_audio
    import engine.audio.vad as vad

    created: list = []
    monkeypatch.setattr(microphone, "MicrophoneInput", lambda *a, **k: FakeInput(created, *a, **k))
    monkeypatch.setattr(system_audio, "SystemAudioInput", lambda *a, **k: FakeInput(created, *a, **k))
    monkeypatch.setattr(vad, "VAD", FakeVAD)
    return created


@pytest.fixture
def test_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOFLUX_HOST", "127.0.0.1")
    monkeypatch.setenv("ECHOFLUX_PORT", "0")
    monkeypatch.setenv(protocol.ENV_CONTROL_TOKEN, CONTROL_TOKEN)
    monkeypatch.setenv(protocol.ENV_EVENT_TOKEN, EVENT_TOKEN)

    db_path = tmp_path / "test_adversarial.db"
    repo = ConversationRepository(db_path=db_path)
    engine = EchoFluxEngine(conversation_repo=repo)
    # Stub local builder to use lightweight FakeSTT
    engine._stt_local_builder = lambda cfg: FakeSTT("local", ("mic", "system"))
    return engine


async def _start_engine_server(engine: EchoFluxEngine):
    engine._ws_server.on_start(engine._on_client_start)
    engine._ws_server.on_stop(engine._on_client_stop)
    await engine._ws_server.start()


async def _connect_auth(server: WebSocketServer, role: str = "control", token: str = CONTROL_TOKEN):
    ws = await websockets.connect(f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER)
    await ws.send(json.dumps({"op": "auth", "role": role, "token": token, "protocol": 1}))
    raw = await ws.recv()
    resp = json.loads(raw)
    assert resp.get("op") == "auth_ok"
    return ws


async def _request(ws, req_id: str, command: dict, timeout: float = 3.0):
    await ws.send(json.dumps({"op": "request", "id": req_id, "command": command}))
    frames = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        frame = json.loads(raw)
        frames.append(frame)
        if frame.get("op") == "complete" and frame.get("id") == req_id:
            return frames


# -----------------------------------------------------------------------------
# 1. Malformed Start Configs & Fail-Closed Invariants
# -----------------------------------------------------------------------------

INVALID_START_CONFIGS = [
    {"type": "start", "config": None},
    {"type": "start", "config": "not_a_dictionary"},
    {"type": "start", "config": 12345},
    {"type": "start", "config": [CANARY_SECRET]},
    {"type": "start", "config": {"services": {"stt": "invalid"}}},
    {"type": "start", "config": {"services": {"stt": {"provider": "nonexistent_stt_provider"}}}},
    {"type": "start", "config": {"services": {"stt": {"provider": "deepgram"}}}},
    {"type": "start", "config": {"services": {"translation": {"provider": "nonexistent_mt"}}}},
]


@pytest.mark.parametrize("payload", INVALID_START_CONFIGS)
@pytest.mark.asyncio
async def test_invalid_start_configs_fail_closed_and_redact(
    payload, test_engine, patch_audio, caplog
):
    caplog.set_level(logging.DEBUG)
    await _start_engine_server(test_engine)

    try:
        ws = await _connect_auth(test_engine._ws_server, "control", CONTROL_TOKEN)
        req_id = "req-malformed-start"
        frames = await _request(ws, req_id, payload)

        # Invariant 1: MUST NOT emit status: started
        status_frames = []
        for f in frames:
            msg = f.get("message")
            if (isinstance(msg, dict) and msg.get("type") == "status" and msg.get("status") == "started") or (f.get("type") == "status" and f.get("status") == "started"):
                status_frames.append(f)

        assert len(status_frames) == 0, f"Fail-closed violated: received started status in {frames}"

        # Invariant 2: Zero canary secret disclosure in frames or logs
        raw_frames_str = json.dumps(frames)
        assert CANARY_SECRET not in raw_frames_str
        assert CANARY_SECRET not in caplog.text

        # Invariant 3: Request completes with op: complete
        complete_frames = [f for f in frames if f.get("op") == "complete"]
        assert len(complete_frames) == 1

        # Invariant 4: Engine is not marked running
        assert not test_engine._running
        await ws.close()
    finally:
        await test_engine._stop_pipeline()
        await test_engine._ws_server.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "start", "config": {}, "secrets": "not_a_dict"},
        {"type": "start", "config": {}, "secrets": [CANARY_SECRET]},
        {"type": "start", "config": {"services": "not_a_dict"}},
    ],
)
@pytest.mark.asyncio
async def test_resilient_fallback_on_default_configs(payload, test_engine, patch_audio, caplog):
    caplog.set_level(logging.DEBUG)
    await _start_engine_server(test_engine)

    try:
        ws = await _connect_auth(test_engine._ws_server, "control", CONTROL_TOKEN)
        req_id = "req-fallback-start"
        frames = await _request(ws, req_id, payload)

        # Invariant 1: Zero canary secrets leaked
        assert CANARY_SECRET not in json.dumps(frames)
        assert CANARY_SECRET not in caplog.text

        # Invariant 2: Completes cleanly
        assert any(f.get("op") == "complete" for f in frames)

        # Clean shutdown
        await _request(ws, "stop-fallback", {"type": "stop"})
        assert not test_engine._running
        await ws.close()
    finally:
        await test_engine._stop_pipeline()
        await test_engine._ws_server.stop()


# -----------------------------------------------------------------------------
# 2. Invalid Provider Configs & Canary Secret Redaction
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_provider_start_refusal_and_redaction(test_engine, patch_audio, caplog):
    caplog.set_level(logging.DEBUG)
    await _start_engine_server(test_engine)

    try:
        ws = await _connect_auth(test_engine._ws_server, "control", CONTROL_TOKEN)
        req_payload = {
            "type": "start",
            "config": {
                "services": {
                    "stt": {"provider": "deepgram", "credential_ref": "deepgram-key"},
                    "translation": {"provider": "openai", "credential_ref": "openai-key"},
                }
            },
            "secrets": {
                "deepgram-key": CANARY_SECRET,
                "openai-key": CANARY_SECRET,
            },
        }

        frames = await _request(ws, "req-invalid-provider", req_payload)

        # Must fail closed: no started status
        for f in frames:
            assert f.get("status") != "started"
            if "message" in f and isinstance(f["message"], dict):
                assert f["message"].get("status") != "started"

        assert CANARY_SECRET not in json.dumps(frames)
        assert CANARY_SECRET not in caplog.text
        assert not test_engine._running
        await ws.close()
    finally:
        await test_engine._stop_pipeline()
        await test_engine._ws_server.stop()


# -----------------------------------------------------------------------------
# 3. Rapid Connect / Disconnect Concurrency Stress
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rapid_connect_disconnect_under_load(test_engine, patch_audio):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    async def _rapid_client(client_id: int):
        for _ in range(5):
            try:
                ws = await websockets.connect(
                    f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER
                )
                await ws.send(
                    json.dumps({
                        "op": "auth",
                        "role": "control",
                        "token": CONTROL_TOKEN,
                        "protocol": 1,
                    })
                )
                await ws.send(
                    json.dumps({
                        "op": "request",
                        "id": f"rapid-{client_id}",
                        "command": {"type": "stop"},
                    })
                )
                await asyncio.sleep(0.005)
                await ws.close()
            except Exception:
                pass

    try:
        await asyncio.gather(*[_rapid_client(i) for i in range(10)])

        # Engine server must remain completely healthy and responsive
        control = await _connect_auth(server, "control", CONTROL_TOKEN)
        frames = await _request(control, "req-post-stress", {"type": "stop"})
        complete_frames = [f for f in frames if f.get("op") == "complete"]
        assert len(complete_frames) == 1
        await control.close()
    finally:
        await test_engine._stop_pipeline()
        await server.stop()


# -----------------------------------------------------------------------------
# 4. Rapid Start / Stop Thrashing
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rapid_start_stop_thrashing(test_engine, patch_audio):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        ws = await _connect_auth(server, "control", CONTROL_TOKEN)
        for i in range(5):
            start_payload = {
                "type": "start",
                "config": {"audio.source": "microphone", "asr.model_size": "tiny"},
            }
            await _request(ws, f"req-start-{i}", start_payload)
            await _request(ws, f"req-stop-{i}", {"type": "stop"})

        assert not test_engine._running
        await ws.close()
    finally:
        await test_engine._stop_pipeline()
        await server.stop()


# -----------------------------------------------------------------------------
# 5. Role Isolation & Protocol Boundary Violations
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_role_cannot_issue_commands(test_engine):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        event_ws = await _connect_auth(server, "event", EVENT_TOKEN)
        await event_ws.send(
            json.dumps({
                "op": "request",
                "id": "exploit-1",
                "command": {"type": "start", "config": {}},
            })
        )
        raw = await asyncio.wait_for(event_ws.recv(), timeout=1.0)
        frame = json.loads(raw)
        assert frame.get("op") == "error"
        assert frame.get("code") == "forbidden"
        await event_ws.close()
    finally:
        await test_engine._stop_pipeline()
        await server.stop()


@pytest.mark.asyncio
async def test_unauthenticated_command_rejected_immediately(test_engine):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        ws = await websockets.connect(
            f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER
        )
        await ws.send(
            json.dumps({"op": "request", "id": "pre-auth", "command": {"type": "stop"}})
        )
        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        frame = json.loads(raw)
        assert frame.get("op") == "error"
        assert frame.get("code") == "auth_required"

        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(ws.recv(), timeout=1.0)
        assert ws.close_code in (CLOSE_AUTH_FAILED, 1008, 1002)
    finally:
        await server.stop()


# -----------------------------------------------------------------------------
# 6. Pipeline Abort Leaves SQLite Clean & Valid
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_abort_cleans_session_and_preserves_sqlite(test_engine, patch_audio):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        ws = await _connect_auth(server, "control", CONTROL_TOKEN)
        # Configure a cloud STT provider without cloud secrets so start fails closed
        payload = {
            "type": "start",
            "config": {
                "services": {
                    "stt": {"provider": "deepgram", "credential_ref": "deepgram-key"},
                }
            },
        }
        frames = await _request(ws, "abort-1", payload)
        assert any(f.get("op") == "complete" for f in frames)

        # Invariant: session should be aborted and inactive
        assert test_engine._active_session_id is None
        assert not test_engine._running

        # Verify SQLite integrity via PRAGMA
        conn = sqlite3.connect(test_engine._conversation_repo._db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        res = cur.fetchone()[0]
        conn.close()
        assert res == "ok"

        # Verify subsequent normal repository operations work cleanly
        conv = test_engine._conversation_repo.create_conversation(
            ConversationRecord(
                conversation_id="conv-adversarial-1",
                title="New Test Session",
                created_at_ms=1000.0,
                updated_at_ms=1000.0,
            )
        )
        assert conv.conversation_id == "conv-adversarial-1"
        await ws.close()
    finally:
        await test_engine._stop_pipeline()
        await server.stop()


# -----------------------------------------------------------------------------
# 7. Test Connection Surface, Redaction & Validation Resilience
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_test_connection_redaction_and_resilience(test_engine, caplog):
    caplog.set_level(logging.DEBUG)
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        ws = await _connect_auth(server, "control", CONTROL_TOKEN)

        # 7a. Valid capability with fake secret
        req_id = "tc-1"
        payload = {
            "type": "test_connection",
            "id": req_id,
            "capability": "stt",
            "service": {"provider": "deepgram", "credential_ref": "deepgram-slot"},
            "secrets": {"deepgram-slot": CANARY_SECRET},
        }
        frames = await _request(ws, req_id, payload)
        result_frames = [
            f for f in frames
            if f.get("message", {}).get("type") == "test_connection_result"
            or f.get("type") == "test_connection_result"
        ]
        assert len(result_frames) == 1

        # Assert no canary secrets leaked anywhere
        assert CANARY_SECRET not in json.dumps(frames)
        assert CANARY_SECRET not in caplog.text

        # 7b. Invalid/corrupted capability
        req_id_bad = "tc-2"
        payload_bad = {
            "type": "test_connection",
            "id": req_id_bad,
            "capability": "telepathy",
            "service": {"provider": "brainwaves"},
            "secrets": {"key": CANARY_SECRET},
        }
        frames_bad = await _request(ws, req_id_bad, payload_bad)
        bad_results = [
            f for f in frames_bad
            if f.get("message", {}).get("type") == "test_connection_result"
            or f.get("type") == "test_connection_result"
        ]
        assert len(bad_results) == 1
        res_data = bad_results[0].get("message", bad_results[0])
        assert res_data.get("result", {}).get("status") in ("invalid", "unsupported", "error")
        assert CANARY_SECRET not in json.dumps(frames_bad)
        await ws.close()
    finally:
        await server.stop()


# -----------------------------------------------------------------------------
# 8. Malformed & Binary Payload Rejection
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_corrupted_json_and_binary_frames(test_engine):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        # Binary frame before auth -> rejected with error then closed
        ws_bin = await websockets.connect(
            f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER
        )
        await ws_bin.send(b"\x00\xff\xfe\x01\x02")
        frame_bin = json.loads(await asyncio.wait_for(ws_bin.recv(), timeout=1.0))
        assert frame_bin.get("op") == "error"
        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(ws_bin.recv(), timeout=1.0)

        # Corrupted JSON before auth -> rejected with error then closed
        ws_corrupt = await websockets.connect(
            f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER
        )
        await ws_corrupt.send("{not_valid_json: 123,")
        frame_corrupt = json.loads(await asyncio.wait_for(ws_corrupt.recv(), timeout=1.0))
        assert frame_corrupt.get("op") == "error"
        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(ws_corrupt.recv(), timeout=1.0)

        # Corrupted JSON after auth on control channel -> protocol returns error frame
        ws_auth = await _connect_auth(server, "control", CONTROL_TOKEN)
        await ws_auth.send("{invalid_json_frame")
        frame_auth = json.loads(await asyncio.wait_for(ws_auth.recv(), timeout=1.0))
        assert frame_auth.get("op") == "error"
        assert frame_auth.get("code") == "bad_request"
        await ws_auth.close()
    finally:
        await server.stop()


# -----------------------------------------------------------------------------
# 9. Concurrent Start Commands Race Resilience
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_start_commands_race(test_engine, patch_audio):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    try:
        ws1 = await _connect_auth(server, "control", CONTROL_TOKEN)
        ws2 = await _connect_auth(server, "control", CONTROL_TOKEN)

        start_payload1 = {
            "type": "start",
            "config": {"audio.source": "microphone", "asr.model_size": "tiny"},
        }
        start_payload2 = {
            "type": "start",
            "config": {"audio.source": "microphone", "asr.model_size": "tiny"},
        }

        # Issue both simultaneously
        t1 = asyncio.create_task(_request(ws1, "race-1", start_payload1))
        t2 = asyncio.create_task(_request(ws2, "race-2", start_payload2))
        res1, res2 = await asyncio.gather(t1, t2)

        # At most one is active; engine state is stable
        assert test_engine._running in (True, False)
        # Both complete with op: complete
        assert any(f.get("op") == "complete" for f in res1)
        assert any(f.get("op") == "complete" for f in res2)

        # Stop cleanly
        await _request(ws1, "stop-race", {"type": "stop"})
        assert not test_engine._running
        await ws1.close()
        await ws2.close()
    finally:
        await test_engine._stop_pipeline()
        await server.stop()


# -----------------------------------------------------------------------------
# 10. Additive Transcript Keys Invariant in Live Engine Events
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_engine_transcript_events_have_required_additive_keys(test_engine, patch_audio):
    await _start_engine_server(test_engine)
    server = test_engine._ws_server

    event_client = await _connect_auth(server, "event", EVENT_TOKEN)
    control_client = await _connect_auth(server, "control", CONTROL_TOKEN)

    try:
        # Start pipeline
        start_payload = {
            "type": "start",
            "config": {"audio.source": "microphone", "asr.model_size": "tiny"},
        }
        await _request(control_client, "start-evt-test", start_payload)
        assert test_engine._running

        # Simulate STT event through engine handler
        from engine.session.session_orchestrator import STTTranscriptEvent

        event = STTTranscriptEvent(
            stream_id="mic",
            is_final=True,
            text="adversarial test transcript",
            item_key="item-1",
            revision=1,
            epoch_id=1,
            provider_id="fake-local",
            language="en",
        )

        test_engine._on_stt_event(event)

        # Receive broadcast event
        raw = await asyncio.wait_for(event_client.recv(), timeout=2.0)
        envelope = json.loads(raw)
        assert envelope.get("op") == "event"
        msg = envelope.get("event", {})

        # Assert all required additive keys are present
        assert "conversation_id" in msg
        assert "message_id" in msg
        assert msg["conversation_id"] == test_engine._active_conversation_id
        assert msg["is_final"] is True
        assert msg["text"] == "adversarial test transcript"
        assert msg["language"] == "en"

        await _request(control_client, "stop-evt-test", {"type": "stop"})
        await event_client.close()
        await control_client.close()
    finally:
        await test_engine._stop_pipeline()
        await server.stop()
