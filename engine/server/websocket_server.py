import json
import asyncio
from typing import Optional, Set, Callable, Awaitable

from engine.core.logging import get_logger
from engine.core.exceptions import ServerError

logger = get_logger("server.websocket")


class WebSocketServer:

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self._host = host
        self._port = port
        self._server = None
        self._clients = set()
        self._running = False
        self.is_capturing = False
        self._cached_devices = None

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

        elif msg_type == "list_devices":
            logger.info("Received list_devices request")
            await websocket.send(json.dumps(await self._list_devices()))

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

        loop = asyncio.get_event_loop()

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

            # -- Speakers / loopback devices (pyaudiowpatch) --
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
    def client_count(self) -> int:
        return len(self._clients)
