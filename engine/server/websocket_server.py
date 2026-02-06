import json
import asyncio
from typing import Optional, Set, Callable, Awaitable

from engine.core.logging import get_logger
from engine.core.exceptions import ServerError

logger = get_logger("server.websocket")


class WebSocketServer:

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._server = None
        self._clients: Set = set()
        self._on_start: Optional[Callable[[dict], Awaitable[None]]] = None
        self._on_stop: Optional[Callable[[], Awaitable[None]]] = None
        self._running = False

    def on_start(self, handler: Callable[[dict], Awaitable[None]]):
        self._on_start = handler

    def on_stop(self, handler: Callable[[], Awaitable[None]]):
        self._on_stop = handler

    async def start(self):
        try:
            import websockets
        except ImportError:
            raise ServerError("websockets library is not installed")

        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self._host,
            self._port,
        )
        logger.info("WebSocket server listening on ws://%s:%d", self._host, self._port)

    async def stop(self):
        self._running = False
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("WebSocket server stopped")

    async def broadcast(self, message: dict):
        if not self._clients:
            return

        payload = json.dumps(message)
        disconnected = set()

        for client in self._clients:
            try:
                await client.send(payload)
            except Exception:
                disconnected.add(client)

        self._clients -= disconnected

    async def _handle_client(self, websocket, path=None):
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)

        try:
            async for raw_message in websocket:
                await self._process_message(raw_message, websocket)
        except Exception as e:
            logger.debug("Client disconnected: %s (%s)", remote, e)
        finally:
            self._clients.discard(websocket)
            logger.info("Client removed: %s", remote)

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

        msg_type = message.get("type")

        if msg_type == "start":
            config = message.get("config", {})
            logger.info("Received start command with config: %s", config)
            if self._on_start:
                await self._on_start(config)
            await websocket.send(json.dumps({"type": "status", "status": "started"}))

        elif msg_type == "stop":
            logger.info("Received stop command")
            if self._on_stop:
                await self._on_stop()
            await websocket.send(json.dumps({"type": "status", "status": "stopped"}))

        else:
            logger.warning("Unknown message type: %s", msg_type)
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown type: {msg_type}",
            }))

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)
