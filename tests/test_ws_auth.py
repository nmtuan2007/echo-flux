"""Authentication, authorization, redaction, and compatibility of the engine socket.

Every test runs a real ``WebSocketServer`` on an ephemeral loopback port with fake,
deterministic tokens. Handlers are recorders, so no audio, model, or provider is touched.
"""

import asyncio
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from engine.core.exceptions import ServerError
from engine.server import protocol
from engine.server.protocol import (
    ENV_CONTROL_TOKEN,
    ENV_EVENT_TOKEN,
    REDACTED,
    BootstrapError,
    BootstrapTokens,
    Role,
    SecurityMode,
)
from engine.server.websocket_server import WebSocketServer

CONTROL_TOKEN = "c0" * 32
EVENT_TOKEN = "e1" * 32
SECRET = "sk-fake-test-key-0000"
TOKENS = BootstrapTokens(control=CONTROL_TOKEN, event=EVENT_TOKEN)

# The websockets client logs raw frames at DEBUG; keep the test client's own auth frames
# out of caplog so log assertions only see what the engine wrote.
_CLIENT_LOGGER = logging.getLogger("tests.ws_auth.client")
_CLIENT_LOGGER.setLevel(logging.WARNING)

ALL_COMMANDS: List[Dict[str, Any]] = [
    {"type": "start", "config": {"llm.enabled": True, "llm.api_key": SECRET}},
    {"type": "stop"},
    {"type": "list_devices"},
    {"type": "request_models_list"},
    {"type": "download_model", "model_id": "m", "model_type": "asr", "hf_token": SECRET},
    {"type": "delete_model", "model_id": "m", "model_type": "asr"},
    {"type": "search_hub", "query": "whisper", "task": "asr", "hf_token": SECRET},
    {"type": "request_suggestion", "entry_id": "e-1", "target_text": "hi", "context": []},
    {"type": "request_summary", "entries": [], "llm_config": {"api_key": SECRET}},
]


class Recorder:
    """Stands in for EchoFluxEngine's handlers."""

    def __init__(self, server: WebSocketServer):
        self.calls: List[str] = []
        self.start_configs: List[dict] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        server.on_start(self._start)
        server.on_stop(self._stop)
        server.on_suggestion(self._suggestion)
        server.on_summary(self._record("request_summary"))
        server.on_request_models_list(self._record("request_models_list"))
        server.on_download_model(self._record("download_model"))
        server.on_delete_model(self._record("delete_model"))
        server.on_search_hub(self._record("search_hub"))
        self._server = server

    async def _start(self, config: dict):
        self.calls.append("start")
        self.start_configs.append(config)

    async def _stop(self):
        self.calls.append("stop")

    async def _suggestion(self, message: dict, websocket):
        self.calls.append("request_suggestion")
        loop = asyncio.get_running_loop()
        reply = {"type": "suggestion_result", "entry_id": message["entry_id"], "options": ["x"]}

        # Mirrors LLMAssistant: the reply is sent later, from a worker thread.
        def _worker():
            asyncio.run_coroutine_threadsafe(websocket.send(json.dumps(reply)), loop)

        threading.Thread(target=_worker, daemon=True).start()

    def _record(self, name: str):
        async def _handler(message: dict, websocket):
            self.calls.append(name)

        return _handler


@asynccontextmanager
async def serving(tokens: Optional[BootstrapTokens] = TOKENS, **kwargs):
    server = WebSocketServer("127.0.0.1", 0, tokens=tokens, **kwargs)
    # Enumeration would touch PyAudio; the contract under test is routing, not devices.
    server.is_capturing = True
    recorder = Recorder(server)
    await server.start()
    try:
        yield server, recorder
    finally:
        await server.stop()


async def connect(server: WebSocketServer):
    return await websockets.connect(f"ws://127.0.0.1:{server.bound_port}", logger=_CLIENT_LOGGER)


async def recv_text(ws, timeout: float = 2.0) -> str:
    return await asyncio.wait_for(ws.recv(), timeout)


async def recv_json(ws, timeout: float = 2.0) -> dict:
    return json.loads(await recv_text(ws, timeout))


async def authenticate(server: WebSocketServer, role: str, token: str):
    ws = await connect(server)
    await ws.send(json.dumps({"op": "auth", "role": role, "token": token, "protocol": 1}))
    assert await recv_json(ws) == {"op": "auth_ok", "role": role, "protocol": 1}
    return ws


async def request(ws, request_id: str, command: dict) -> List[dict]:
    """Send one control request and collect its frames up to and including completion."""
    await ws.send(json.dumps({"op": "request", "id": request_id, "command": command}))
    frames = []
    while True:
        frame = await recv_json(ws)
        frames.append(frame)
        if frame.get("op") == "complete" and frame.get("id") == request_id:
            return frames


async def assert_closed(ws, code: int):
    with pytest.raises(ConnectionClosed):
        await recv_text(ws)
    assert ws.close_code == code


# ---------------------------------------------------------------------------------------
# Bootstrap tokens
# ---------------------------------------------------------------------------------------


def test_absent_tokens_select_the_legacy_path():
    assert BootstrapTokens.from_environ({}) is None
    assert BootstrapTokens.from_environ({ENV_CONTROL_TOKEN: "", ENV_EVENT_TOKEN: ""}) is None


def test_tokens_are_consumed_from_the_environment():
    environ = {ENV_CONTROL_TOKEN: CONTROL_TOKEN, ENV_EVENT_TOKEN: EVENT_TOKEN, "OTHER": "1"}
    tokens = BootstrapTokens.from_environ(environ)
    assert tokens is not None
    assert environ == {"OTHER": "1"}
    assert CONTROL_TOKEN not in repr(tokens) and EVENT_TOKEN not in repr(tokens)


@pytest.mark.parametrize(
    "environ",
    [
        {ENV_CONTROL_TOKEN: CONTROL_TOKEN},
        {ENV_EVENT_TOKEN: EVENT_TOKEN},
        {ENV_CONTROL_TOKEN: CONTROL_TOKEN, ENV_EVENT_TOKEN: CONTROL_TOKEN},
        {ENV_CONTROL_TOKEN: "short", ENV_EVENT_TOKEN: EVENT_TOKEN},
        {ENV_CONTROL_TOKEN: CONTROL_TOKEN[:-1] + " ", ENV_EVENT_TOKEN: EVENT_TOKEN},
    ],
    ids=["control-only", "event-only", "identical", "too-short", "whitespace"],
)
def test_malformed_tokens_fail_closed_without_echoing_values(environ):
    values = [v for v in environ.values() if v]
    with pytest.raises(BootstrapError) as excinfo:
        BootstrapTokens.from_environ(environ)
    assert all(value not in str(excinfo.value) for value in values)
    assert ENV_CONTROL_TOKEN not in environ and ENV_EVENT_TOKEN not in environ


def test_token_verification_is_role_bound_and_constant_time(monkeypatch):
    compared = []
    real = protocol.hmac.compare_digest

    def _spy(a, b):
        compared.append(True)
        return real(a, b)

    monkeypatch.setattr(protocol.hmac, "compare_digest", _spy)
    assert TOKENS.verify(Role.CONTROL, CONTROL_TOKEN)
    assert TOKENS.verify(Role.EVENT, EVENT_TOKEN)
    assert not TOKENS.verify(Role.CONTROL, EVENT_TOKEN)
    assert not TOKENS.verify(Role.EVENT, CONTROL_TOKEN)
    assert not TOKENS.verify(Role.LEGACY, CONTROL_TOKEN)
    assert len(compared) == 4


# ---------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------


async def test_legacy_path_keeps_the_raw_contract_and_warns(caplog):
    caplog.set_level(logging.INFO)
    async with serving(tokens=None) as (server, recorder):
        assert server.security_mode is SecurityMode.LEGACY_TRANSITIONAL
        assert "DEPRECATED" in caplog.text

        ws = await connect(server)
        await ws.send(json.dumps(ALL_COMMANDS[0]))
        assert await recv_json(ws) == {"type": "status", "status": "started"}
        assert recorder.start_configs == [ALL_COMMANDS[0]["config"]]

        await server.broadcast({"type": "partial", "text": "hello"})
        assert await recv_json(ws) == {"type": "partial", "text": "hello"}

        await ws.send(json.dumps({"type": "bogus"}))
        assert (await recv_json(ws))["type"] == "error"
        await ws.close()

    assert SECRET not in caplog.text


async def test_insecure_dev_mode_is_explicit(caplog):
    caplog.set_level(logging.INFO)
    async with serving(tokens=None, insecure_dev=True) as (server, _recorder):
        assert server.security_mode is SecurityMode.INSECURE_DEV
        assert not server.is_authenticated
        assert "Insecure development mode" in caplog.text


def test_insecure_dev_cannot_be_combined_with_tokens():
    with pytest.raises(ServerError):
        WebSocketServer("127.0.0.1", 0, tokens=TOKENS, insecure_dev=True)


@pytest.mark.parametrize("tokens", [None, TOKENS], ids=["legacy", "authenticated"])
async def test_non_loopback_host_is_refused(tokens):
    server = WebSocketServer("0.0.0.0", 0, tokens=tokens)
    with pytest.raises(ServerError):
        await server.start()
    assert server._server is None


# ---------------------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "code", "close_code"),
    [
        ({"op": "auth", "role": "control", "token": "0" * 64}, "auth_failed", 4401),
        ({"op": "auth", "role": "control", "token": EVENT_TOKEN}, "auth_failed", 4401),
        ({"op": "auth", "role": "event", "token": CONTROL_TOKEN}, "auth_failed", 4401),
        ({"op": "auth", "role": "control"}, "auth_failed", 4401),
        ({"op": "auth", "role": "control", "token": ""}, "auth_failed", 4401),
        ({"op": "auth", "role": "admin", "token": CONTROL_TOKEN}, "auth_failed", 4401),
        (
            {"op": "auth", "role": "control", "token": CONTROL_TOKEN, "protocol": 99},
            "unsupported_protocol",
            4400,
        ),
        (ALL_COMMANDS[0], "auth_required", 4401),
        ("not json", "auth_required", 4401),
        (b"\x00binary", "auth_required", 4401),
    ],
    ids=[
        "wrong-token",
        "event-token-as-control",
        "control-token-as-event",
        "missing-token",
        "empty-token",
        "unknown-role",
        "unsupported-protocol",
        "command-before-auth",
        "invalid-json",
        "binary",
    ],
)
async def test_bad_handshakes_are_rejected_and_disconnected(frame, code, close_code, caplog):
    caplog.set_level(logging.DEBUG)
    async with serving() as (server, recorder):
        ws = await connect(server)
        if isinstance(frame, dict):
            await ws.send(json.dumps(frame))
        else:
            await ws.send(frame)

        raw = await recv_text(ws)
        assert json.loads(raw) == {
            "op": "error",
            "code": code,
            "message": json.loads(raw)["message"],
        }
        for secret in (SECRET, CONTROL_TOKEN, EVENT_TOKEN):
            assert secret not in raw
        await assert_closed(ws, close_code)

        assert recorder.calls == []
        assert server._subscribers == {}

    for secret in (SECRET, CONTROL_TOKEN, EVENT_TOKEN):
        assert secret not in caplog.text


async def test_silent_client_is_disconnected_after_the_auth_timeout():
    async with serving(auth_timeout=0.1) as (server, _recorder):
        ws = await connect(server)
        assert (await recv_json(ws))["code"] == "auth_timeout"
        await assert_closed(ws, 4408)


# ---------------------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ALL_COMMANDS, ids=[c["type"] for c in ALL_COMMANDS])
async def test_event_token_cannot_execute_commands(command):
    async with serving() as (server, recorder):
        ws = await authenticate(server, "event", EVENT_TOKEN)

        for frame in (command, {"op": "request", "id": "r-1", "command": command}):
            await ws.send(json.dumps(frame))
            raw = await recv_text(ws)
            assert json.loads(raw) == {
                "op": "error",
                "code": "forbidden",
                "message": "Event clients cannot send commands",
            }
            assert SECRET not in raw

        assert recorder.calls == []

        # Rejection does not cost the subscription.
        await server.broadcast({"type": "status", "status": "stopped"})
        assert await recv_json(ws) == {
            "op": "event",
            "event": {"type": "status", "status": "stopped"},
        }
        await ws.close()


async def test_control_requests_receive_correlated_responses():
    async with serving() as (server, recorder):
        ws = await authenticate(server, "control", CONTROL_TOKEN)

        frames = await request(ws, "req-1", ALL_COMMANDS[0])
        assert frames == [
            {"op": "response", "id": "req-1", "message": {"type": "status", "status": "started"}},
            {"op": "complete", "id": "req-1"},
        ]
        assert recorder.start_configs == [ALL_COMMANDS[0]["config"]]

        frames = await request(ws, "req-2", {"type": "nope"})
        assert frames[0]["id"] == "req-2"
        assert frames[0]["message"]["type"] == "error"
        await ws.close()


async def test_thread_replies_stay_correlated_after_the_request_completes():
    async with serving() as (server, _recorder):
        ws = await authenticate(server, "control", CONTROL_TOKEN)
        command = {"type": "request_suggestion", "entry_id": "e-9", "context": []}
        await ws.send(json.dumps({"op": "request", "id": "sug-1", "command": command}))

        frames = [await recv_json(ws), await recv_json(ws)]
        replies = [f for f in frames if f["op"] == "response"]
        assert replies == [
            {
                "op": "response",
                "id": "sug-1",
                "message": {"type": "suggestion_result", "entry_id": "e-9", "options": ["x"]},
            }
        ]
        assert {"op": "complete", "id": "sug-1"} in frames
        await ws.close()


@pytest.mark.parametrize(
    ("frame", "code", "request_id"),
    [
        (ALL_COMMANDS[0], "bad_request", None),
        ({"op": "request", "command": ALL_COMMANDS[0]}, "bad_request_id", None),
        ({"op": "request", "id": "bad id!", "command": ALL_COMMANDS[0]}, "bad_request_id", None),
        ({"op": "request", "id": "r-3", "command": [SECRET]}, "bad_request", "r-3"),
    ],
    ids=["bare-legacy-command", "missing-id", "invalid-id", "non-object-command"],
)
async def test_control_rejects_malformed_envelopes_without_echo(frame, code, request_id):
    async with serving() as (server, recorder):
        ws = await authenticate(server, "control", CONTROL_TOKEN)
        await ws.send(json.dumps(frame))
        raw = await recv_text(ws)
        reply = json.loads(raw)
        assert reply["op"] == "error"
        assert reply["code"] == code
        assert reply.get("id") == request_id
        assert SECRET not in raw
        assert recorder.calls == []
        await ws.close()


async def test_handler_failure_is_reported_without_details(caplog):
    caplog.set_level(logging.DEBUG)
    async with serving() as (server, _recorder):

        async def _explode(config: dict):
            raise RuntimeError(f"boom {SECRET}")

        server.on_start(_explode)
        ws = await authenticate(server, "control", CONTROL_TOKEN)
        frames = await request(ws, "r-5", ALL_COMMANDS[0])
        assert frames == [
            {"op": "error", "code": "command_failed", "message": "Command failed", "id": "r-5"},
            {"op": "complete", "id": "r-5"},
        ]
        await ws.close()

    assert SECRET not in caplog.text


async def test_event_subscribers_do_not_receive_control_replies():
    async with serving() as (server, _recorder):
        events = [
            await authenticate(server, "event", EVENT_TOKEN),
            await authenticate(server, "event", EVENT_TOKEN),
        ]
        control = await authenticate(server, "control", CONTROL_TOKEN)

        await request(control, "r-1", ALL_COMMANDS[0])
        await request(control, "r-2", {"type": "stop"})

        marker = {"type": "final", "entry_id": "e-marker", "text": "done", "translation": None}
        await server.broadcast(marker)

        # Replies were sent before the marker, so any leak would arrive first.
        for ws in events:
            assert await recv_json(ws) == {"op": "event", "event": marker}

        # Broadcasts are for the event plane; the control connection sees only its replies.
        frames = await request(control, "r-3", {"type": "stop"})
        assert all(frame.get("id") == "r-3" for frame in frames)

        for ws in [*events, control]:
            await ws.close()


# ---------------------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------------------


async def test_tokens_and_secrets_never_reach_events_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    async with serving() as (server, recorder):
        rejected = await connect(server)
        await rejected.send(
            json.dumps({"op": "auth", "role": "control", "token": CONTROL_TOKEN[:-1] + "x"})
        )
        await recv_text(rejected)

        event = await authenticate(server, "event", EVENT_TOKEN)
        control = await authenticate(server, "control", CONTROL_TOKEN)
        await request(control, "r-1", ALL_COMMANDS[0])
        assert recorder.start_configs[0]["llm.api_key"] == SECRET

        await server.broadcast(
            {
                "type": "model_action_result",
                "error": f"failed with {CONTROL_TOKEN} and {EVENT_TOKEN}",
                "hf_token": SECRET,
                "details": [{"llm_config": {"api_key": SECRET}, "Authorization": SECRET}],
            }
        )
        raw = await recv_text(event)
        for secret in (SECRET, CONTROL_TOKEN, EVENT_TOKEN):
            assert secret not in raw
        frame = json.loads(raw)["event"]
        assert frame["hf_token"] == REDACTED
        assert frame["details"] == [{"llm_config": REDACTED, "Authorization": REDACTED}]
        assert frame["error"] == f"failed with {REDACTED} and {REDACTED}"

        for ws in (event, control):
            await ws.close()

    for secret in (SECRET, CONTROL_TOKEN, EVENT_TOKEN):
        assert secret not in caplog.text


# ---------------------------------------------------------------------------------------
# Engine bootstrap and CLI
# ---------------------------------------------------------------------------------------


@pytest.fixture
def bootstrap_env(monkeypatch):
    monkeypatch.setenv(ENV_CONTROL_TOKEN, CONTROL_TOKEN)
    monkeypatch.setenv(ENV_EVENT_TOKEN, EVENT_TOKEN)
    monkeypatch.setenv("ECHOFLUX_HOST", "127.0.0.1")
    monkeypatch.setenv("ECHOFLUX_PORT", "0")


def test_engine_consumes_bootstrap_tokens_without_logging(bootstrap_env, caplog):
    import os

    from engine.main import EchoFluxEngine

    caplog.set_level(logging.DEBUG)
    engine = EchoFluxEngine()
    assert engine._ws_server.is_authenticated
    assert ENV_CONTROL_TOKEN not in os.environ
    assert ENV_EVENT_TOKEN not in os.environ
    assert CONTROL_TOKEN not in caplog.text and EVENT_TOKEN not in caplog.text


def test_engine_refuses_insecure_dev_with_tokens(bootstrap_env):
    from engine.main import EchoFluxEngine

    with pytest.raises(ServerError):
        EchoFluxEngine(insecure_dev=True)


def test_engine_without_tokens_uses_the_legacy_path():
    from engine.main import EchoFluxEngine

    assert EchoFluxEngine()._ws_server.security_mode is SecurityMode.LEGACY_TRANSITIONAL
    assert EchoFluxEngine(insecure_dev=True)._ws_server.security_mode is SecurityMode.INSECURE_DEV


async def test_supervised_engine_shuts_down_when_stdin_closes(bootstrap_env, monkeypatch):
    import engine.main as engine_main

    monkeypatch.setattr(engine_main, "set_progress_callback", lambda cb: None)
    monkeypatch.setattr(engine_main.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"")))

    engine = engine_main.EchoFluxEngine()
    await asyncio.wait_for(engine.start(), timeout=5.0)
    assert engine._ws_server._server is None


async def test_unsupervised_engine_does_not_watch_stdin(monkeypatch):
    import engine.main as engine_main

    monkeypatch.setenv("ECHOFLUX_PORT", "0")
    monkeypatch.setattr(engine_main, "set_progress_callback", lambda cb: None)
    engine = engine_main.EchoFluxEngine()
    watched = []
    monkeypatch.setattr(engine, "_start_stdin_watch", lambda: watched.append(True))

    task = asyncio.create_task(engine.start())
    for _ in range(200):
        if engine._ws_server._server is not None:
            break
        await asyncio.sleep(0.01)
    assert engine._ws_server._server is not None

    task.cancel()
    await asyncio.wait_for(task, timeout=5.0)
    assert watched == []


@pytest.mark.parametrize(("argv", "expected"), [([], False), (["--insecure-dev"], True)])
def test_cli_passes_insecure_dev_to_the_engine(monkeypatch, argv, expected):
    import apps.cli.main as cli
    import engine.core.logging as engine_logging

    calls = []
    monkeypatch.setattr(cli.sys, "argv", ["echoflux", *argv])
    monkeypatch.setattr(cli, "Config", lambda path: type("C", (), {"get": lambda s, k: None})())
    monkeypatch.setattr(cli, "apply_overrides", lambda config, args: None)
    monkeypatch.setattr(
        engine_logging, "setup_logging", lambda config: logging.getLogger("tests.cli")
    )
    monkeypatch.setattr(cli, "run_engine", lambda path, insecure_dev: calls.append(insecure_dev))

    cli.main()
    assert calls == [expected]
