import asyncio
import contextlib
import contextvars
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from engine.core.exceptions import ServerError
from engine.core.logging import get_logger
from engine.server.protocol import (
    AUTH_TIMEOUT_SECONDS,
    CLOSE_AUTH_FAILED,
    CLOSE_AUTH_TIMEOUT,
    BootstrapTokens,
    ProtocolError,
    Role,
    SecurityMode,
    auth_ok_frame,
    complete_frame,
    encode_frame,
    error_frame,
    event_frame,
    is_loopback_host,
    parse_auth_frame,
    parse_control_frame,
    response_frame,
    sanitize_event,
)

logger = get_logger("server.websocket")

# websockets logs raw frames at DEBUG, which would put auth tokens and start configs in the
# log whenever the engine runs with ECHOFLUX_LOG_LEVEL=DEBUG. Pin its logger above that.
_transport_logger = get_logger("server.transport")
_transport_logger.setLevel(logging.INFO)

# The connection whose command is being dispatched, so handlers that only receive a config
# can still reply to the requester rather than to an arbitrary client.
_current_requester: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "echoflux_current_requester", default=None
)


class _CorrelatedReplyChannel:
    """Compatibility adapter: legacy handlers call ``send(json)`` on what they think is the
    socket. On a control connection each reply is wrapped in a response frame carrying
    the originating request id, including replies sent later from worker threads."""

    def __init__(self, websocket, request_id: str, tokens: BootstrapTokens):
        self._websocket = websocket
        self._request_id = request_id
        self._tokens = tokens
        self.remote_address = getattr(websocket, "remote_address", None)

    async def send(self, payload) -> None:
        message = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        frame = response_frame(self._request_id, message)
        try:
            await self._websocket.send(encode_frame(frame, self._tokens))
        except Exception as e:
            logger.debug("Dropped reply for closed control client (%s)", type(e).__name__)

    async def close(self, *args, **kwargs) -> None:
        await self._websocket.close(*args, **kwargs)


class WebSocketServer:

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        tokens: Optional[BootstrapTokens] = None,
        insecure_dev: bool = False,
        auth_timeout: float = AUTH_TIMEOUT_SECONDS,
        control_dispatcher: Optional[Any] = None,
    ):
        if tokens is not None and insecure_dev:
            raise ServerError("Insecure development mode cannot be combined with bootstrap tokens")

        self._host = host
        self._port = port
        self._tokens = tokens
        self._auth_timeout = auth_timeout
        self._control_dispatcher = control_dispatcher
        if tokens is not None:
            self._mode = SecurityMode.AUTHENTICATED
        elif insecure_dev:
            self._mode = SecurityMode.INSECURE_DEV
        else:
            self._mode = SecurityMode.LEGACY_TRANSITIONAL

        self._server = None
        self._clients = set()
        # Connections that receive broadcasts: authenticated event clients and legacy clients.
        self._subscribers: Dict[Any, Role] = {}
        self._running = False
        self.is_capturing = False
        self._cached_devices = None

        self._on_start: Optional[Callable[[dict], Awaitable[None]]] = None
        self._on_stop: Optional[Callable[[], Awaitable[None]]] = None
        self._on_suggestion: Optional[Callable[[dict, object], Awaitable[None]]] = None
        self._on_summary: Optional[Callable[[dict, object], Awaitable[None]]] = None

    def on_start(self, handler: Callable[[dict], Awaitable[None]]):
        self._on_start = handler

    def on_stop(self, handler: Callable[[], Awaitable[None]]):
        self._on_stop = handler

    def on_suggestion(self, handler: Callable[[dict, object], Awaitable[None]]):
        self._on_suggestion = handler

    def on_summary(self, handler: Callable[[dict, object], Awaitable[None]]):
        self._on_summary = handler

    def set_control_dispatcher(self, dispatcher: Any) -> None:
        self._control_dispatcher = dispatcher

    @property
    def control_dispatcher(self) -> Optional[Any]:
        return self._control_dispatcher

    async def start(self):
        try:
            import websockets
        except ImportError:
            raise ServerError("websockets library is not installed") from None

        if not is_loopback_host(self._host):
            raise ServerError("The engine socket must bind to a loopback address")

        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self._host,
            self._port,
            logger=_transport_logger,
        )
        if self._mode is SecurityMode.AUTHENTICATED:
            logger.info(
                "WebSocket server listening on ws://%s:%d (authenticated control/event roles)",
                self._host,
                self.bound_port,
            )
        elif self._mode is SecurityMode.INSECURE_DEV:
            logger.warning(
                "Insecure development mode: unauthenticated WebSocket server on ws://%s:%d",
                self._host,
                self.bound_port,
            )
        else:
            logger.warning(
                "DEPRECATED: bootstrap tokens are absent, serving the unauthenticated loopback "
                "path on ws://%s:%d. This transitional path will be removed; use --insecure-dev "
                "for explicit development runs.",
                self._host,
                self.bound_port,
            )

    async def stop(self):
        self._running = False
        for client in list(self._clients):
            with contextlib.suppress(Exception):
                await client.close()
        self._clients.clear()
        self._subscribers.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("WebSocket server stopped")

    async def broadcast(self, message: dict):
        if not self._subscribers:
            return

        legacy_payload = None
        event_payload = None
        disconnected = []

        # Snapshot: clients may connect or authenticate while a send is awaiting.
        for client, role in list(self._subscribers.items()):
            if role is Role.EVENT:
                if event_payload is None:
                    event_payload = encode_frame(
                        event_frame(sanitize_event(message)), self._tokens
                    )
                payload = event_payload
            else:
                if legacy_payload is None:
                    legacy_payload = json.dumps(message)
                payload = legacy_payload
            try:
                await client.send(payload)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self._subscribers.pop(client, None)
            self._clients.discard(client)

    def current_requester(self) -> Optional[Any]:
        """The reply target of the command currently being dispatched, if any."""
        return _current_requester.get()

    async def _handle_client(self, websocket, path=None):
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)

        try:
            if self._tokens is None:
                self._subscribers[websocket] = Role.LEGACY
                async for raw_message in websocket:
                    await self._process_message(raw_message, websocket)
            else:
                role = await self._authenticate(websocket)
                if role is Role.EVENT:
                    await self._serve_event_client(websocket)
                elif role is Role.CONTROL:
                    await self._serve_control_client(websocket)
        except Exception as e:
            logger.debug("Client disconnected: %s (%s)", remote, type(e).__name__)
        finally:
            self._clients.discard(websocket)
            self._subscribers.pop(websocket, None)
            logger.info("Client removed: %s", remote)

    async def _authenticate(self, websocket) -> Optional[Role]:
        remote = websocket.remote_address
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=self._auth_timeout)
        except asyncio.TimeoutError:
            logger.warning("Client %s did not authenticate in time", remote)
            await self._reject(
                websocket,
                ProtocolError("auth_timeout", "Authentication timed out", CLOSE_AUTH_TIMEOUT),
            )
            return None

        try:
            auth = parse_auth_frame(raw)
            if not self._tokens.verify(auth.role, auth.token):
                raise ProtocolError("auth_failed", "Authentication failed", CLOSE_AUTH_FAILED)
        except ProtocolError as err:
            # Log the code only: the frame may hold a token or a command payload.
            logger.warning("Rejected client %s: %s", remote, err.code)
            await self._reject(websocket, err)
            return None

        # Subscribe before acknowledging, so every event after auth_ok reaches the client.
        if auth.role is Role.EVENT:
            self._subscribers[websocket] = Role.EVENT
        await websocket.send(encode_frame(auth_ok_frame(auth.role), self._tokens))
        logger.info("Client authenticated: %s role=%s", remote, auth.role.value)
        return auth.role

    async def _reject(self, websocket, err: ProtocolError):
        try:
            await websocket.send(encode_frame(error_frame(err.code, err.message), self._tokens))
            await websocket.close(code=err.close_code or CLOSE_AUTH_FAILED, reason=err.code)
        except Exception as e:
            logger.debug("Reject failed for closed client (%s)", type(e).__name__)

    async def _serve_event_client(self, websocket):
        remote = websocket.remote_address
        async for _raw in websocket:
            logger.warning("Rejected command from event client %s", remote)
            await websocket.send(
                encode_frame(
                    error_frame("forbidden", "Event clients cannot send commands"), self._tokens
                )
            )

    async def _serve_control_client(self, websocket):
        remote = websocket.remote_address
        async for raw in websocket:
            if self._control_dispatcher is not None:
                try:
                    decoded = json.loads(raw) if isinstance(raw, str) else None
                except Exception:
                    decoded = None
                extracted = self._control_dispatcher.extract_command(decoded)
                if extracted is not None:
                    req_id, cmd_name, payload = extracted
                    await self._control_dispatcher.dispatch(
                        req_id, cmd_name, payload, websocket, self._tokens
                    )
                    continue

            try:
                request = parse_control_frame(raw)
            except ProtocolError as err:
                logger.warning("Rejected control frame from %s: %s", remote, err.code)
                await websocket.send(
                    encode_frame(error_frame(err.code, err.message, err.request_id), self._tokens)
                )
                continue

            channel = _CorrelatedReplyChannel(websocket, request.request_id, self._tokens)
            try:
                await self._dispatch(request.command, channel)
            except Exception as e:
                logger.error(
                    "Control request %s failed (%s)", request.request_id, type(e).__name__
                )
                await websocket.send(
                    encode_frame(
                        error_frame("command_failed", "Command failed", request.request_id),
                        self._tokens,
                    )
                )
            await websocket.send(encode_frame(complete_frame(request.request_id), self._tokens))

    async def _process_message(self, raw: str, websocket):
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON from client")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON",
            }))
            return

        await self._dispatch(message, websocket)

    async def _dispatch(self, message: Any, websocket):
        if not isinstance(message, dict):
            await websocket.send(json.dumps({"type": "error", "message": "Invalid message"}))
            return

        reset_token = _current_requester.set(websocket)
        try:
            await self._dispatch_command(message, websocket)
        finally:
            _current_requester.reset(reset_token)

    async def _dispatch_command(self, message: dict, websocket):
        msg_type = message.get("type")

        if msg_type == "start":
            config = message.get("config", {})
            secrets = message.get("secrets", {})
            # Never log the config itself: it carries provider settings and API keys.
            key_count = len(config) if isinstance(config, dict) else 0
            logger.info("Received start command (%d config keys)", key_count)
            if self._on_start:
                sig = inspect.signature(self._on_start)
                if "secrets" in sig.parameters:
                    res = await self._on_start(config, secrets=secrets)
                else:
                    res = await self._on_start(config)
                if res is False:
                    raise ServerError("Start callback indicated failure")
            await websocket.send(json.dumps({"type": "status", "status": "started"}))

        elif msg_type == "test_connection":
            capability = message.get("capability")
            service_data = message.get("service")
            secrets = message.get("secrets", {})
            req_id = message.get("id")
            logger.info("Received test_connection command for capability=%s", capability)
            from engine.core.config import ServiceConfig
            from engine.providers.registry import build_default_registry
            from engine.providers.validation import (
                ERROR_INTERNAL,
                VALIDATION_INVALID,
                _unvalidated,
                validate_service,
            )
            try:
                service = ServiceConfig.from_dict(capability, service_data)
                val_res = await validate_service(
                    service, dict(secrets or {}), registry=build_default_registry()
                )
                await websocket.send(json.dumps({
                    "type": "test_connection_result",
                    "id": req_id,
                    "result": val_res.to_dict(),
                }))
            except Exception as exc:
                logger.error("Test connection failed: %s", type(exc).__name__)
                val_res = _unvalidated(req_id, capability, None, ERROR_INTERNAL, status=VALIDATION_INVALID)
                await websocket.send(json.dumps({
                    "type": "test_connection_result",
                    "id": req_id,
                    "result": val_res.to_dict(),
                }))

        elif msg_type == "stop":
            logger.info("Received stop command")
            if self._on_stop:
                await self._on_stop()
            await websocket.send(json.dumps({"type": "status", "status": "stopped"}))

        elif msg_type == "list_devices":
            logger.info("Received list_devices request")
            await websocket.send(json.dumps(await self._list_devices()))

        elif msg_type == "request_suggestion":
            logger.info("Received request_suggestion for entry_id=%s", message.get("entry_id"))
            if self._on_suggestion:
                await self._on_suggestion(message, websocket)
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "LLM not configured.",
                }))

        elif msg_type == "request_summary":
            logger.info("Received request_summary")
            if self._on_summary:
                await self._on_summary(message, websocket)
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "LLM not configured.",
                }))

        else:
            logger.warning("Unknown message type: %s", msg_type)
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown type: {msg_type}",
            }))

    async def _list_devices(self) -> dict:
        """Enumerate available microphone and speaker (loopback) devices.

        Runs blocking PyAudio calls in a thread executor to avoid blocking the
        event loop. Prevents PyAudio crashes by using cached devices or a
        placeholder when the engine is actively capturing.
        """
        if self.is_capturing:
            if self._cached_devices:
                return self._cached_devices
            return {
                "type": "devices_list",
                "microphones": [{"id": "default", "name": "Stop Engine to safely refresh"}],
                "speakers": [{"id": "default", "name": "Stop Engine to safely refresh"}],
            }

        loop = asyncio.get_running_loop()

        def _blocking_enumerate():
            mics = []
            speakers = []

            # -- Microphones (pyaudio) --
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                try:
                    default_host_api = pa.get_default_host_api_info()["index"]
                    for i in range(pa.get_device_count()):
                        info = pa.get_device_info_by_index(i)
                        if info.get("hostApi") == default_host_api and info.get("maxInputChannels", 0) > 0:
                            mics.append({"id": str(i), "name": info.get("name", f"Device {i}")})
                finally:
                    pa.terminate()
            except Exception as e:
                logger.warning("Failed to enumerate microphones: %s", e)

            # -- Speakers / loopback devices --
            import sys

            if sys.platform == "win32":
                try:
                    import pyaudiowpatch as pawp
                    pa2 = pawp.PyAudio()
                    try:
                        wasapi_info = pa2.get_host_api_info_by_type(pawp.paWASAPI)
                        for i in range(pa2.get_device_count()):
                            info = pa2.get_device_info_by_index(i)
                            if info["hostApi"] != wasapi_info["index"]:
                                continue
                            if info.get("isLoopbackDevice", False):
                                speakers.append({
                                    "id": str(info["index"]),
                                    "name": f"{info['name']} (Loopback)",
                                })
                    except Exception as e:
                        logger.warning("Failed to enumerate loopback devices: %s", e)
                    finally:
                        pa2.terminate()
                except Exception as e:
                    logger.warning("pyaudiowpatch not available: %s", e)
            else:
                try:
                    from engine.audio.system_audio import SystemAudioInput

                    for dev in SystemAudioInput({}).list_devices():
                        speakers.append({"id": dev.id, "name": dev.name})
                except Exception as e:
                    logger.warning("Failed to enumerate virtual loopback devices: %s", e)

            return mics, speakers

        try:
            mics, speakers = await loop.run_in_executor(None, _blocking_enumerate)
        except Exception as e:
            logger.error("Device enumeration failed: %s", e)
            mics, speakers = [], []

        self._cached_devices = {
            "type": "devices_list",
            "microphones": mics,
            "speakers": speakers,
        }
        return self._cached_devices

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_authenticated(self) -> bool:
        return self._mode is SecurityMode.AUTHENTICATED

    @property
    def security_mode(self) -> SecurityMode:
        return self._mode

    @property
    def bound_port(self) -> int:
        """The listening port; differs from the configured one when that was 0."""
        if self._server is not None:
            for sock in self._server.sockets:
                return sock.getsockname()[1]
        return self._port

    @property
    def client_count(self) -> int:
        return len(self._clients)
