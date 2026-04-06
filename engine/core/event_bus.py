import asyncio
from typing import AsyncGenerator

class EventBus:
    """
    Bridges synchronous background threads with the asyncio event loop.
    Threads call emit() with a message dictionary, which is safely pushed 
    into an asyncio.Queue, and consumed natively by async tasks.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue = asyncio.Queue()

    def emit(self, msg: dict):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    async def consume(self) -> AsyncGenerator[dict, None]:
        while True:
            try:
                msg = await self._queue.get()
                yield msg
                self._queue.task_done()
            except asyncio.CancelledError:
                break

    def clear(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
