"""Inbound WebSocket server for robot initiated connections."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import websockets
from websockets import WebSocketServerProtocol


logger = logging.getLogger(__name__)

FrameCallback = Callable[[Dict[str, Any]], None]
StatusCallback = Callable[[Dict[str, Any]], None]


class WSServer:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        path: str = "/stream",
        token: Optional[str],
        loop: asyncio.AbstractEventLoop,
        on_frame: FrameCallback,
        on_status: StatusCallback,
    ) -> None:
        self._host = host
        self._port = port
        self._path = path
        self._token = token
        self._loop = loop
        self._on_frame = on_frame
        self._on_status = on_status

        self._stop_event = asyncio.Event()
        self._send_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._active_ws: Optional[WebSocketServerProtocol] = None

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(lambda: self._stop_event.set())

    async def run(self) -> None:
        async with websockets.serve(self._handler, self._host, self._port, ping_interval=None):
            self._emit_status(
                {
                    "transport": "ws_server",
                    "state": "listening",
                    "endpoint": f"ws://{self._host}:{self._port}{self._path}",
                }
            )
            await self._stop_event.wait()
        self._emit_status(
            {
                "transport": "ws_server",
                "state": "stopped",
                "endpoint": f"ws://{self._host}:{self._port}{self._path}",
            }
        )

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        parsed = urlparse(ws.path)
        if parsed.path != self._path:
            await ws.close(code=4000, reason="Invalid path")
            return
        if self._token and not self._check_token(parsed, ws):
            await ws.close(code=4001, reason="Unauthorized")
            return
        if self._active_ws is not None:
            await ws.close(code=4002, reason="Already connected")
            return

        self._active_ws = ws
        sender = asyncio.create_task(self._sender(ws))
        receiver = asyncio.create_task(self._receiver(ws))
        self._emit_status(
            {
                "transport": "ws_server",
                "state": "connected",
                "endpoint": f"ws://{self._host}:{self._port}{self._path}",
            }
        )
        try:
            done, pending = await asyncio.wait(
                {sender, receiver},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            self._active_ws = None
            self._emit_status(
                {
                    "transport": "ws_server",
                    "state": "ready",
                    "endpoint": f"ws://{self._host}:{self._port}{self._path}",
                }
            )

    async def _sender(self, ws: WebSocketServerProtocol) -> None:
        while not ws.closed:
            msg = await self._send_queue.get()
            await ws.send(msg)

    async def _receiver(self, ws: WebSocketServerProtocol) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    self._emit_status(
                        {
                            "transport": "ws_server",
                            "state": "bad_frame",
                            "raw": line,
                        }
                    )
                    continue
                self._on_frame(frame)

    def _check_token(self, parsed, ws: WebSocketServerProtocol) -> bool:
        query = parse_qs(parsed.query)
        if "token" in query and query["token"]:
            return query["token"][0] == self._token
        auth = ws.request_headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1] == self._token
        return False

    def queue_json(self, payload: Dict[str, Any]) -> None:
        text = json.dumps(payload)
        self.queue_text(text)

    def queue_text(self, text: str) -> None:
        async def _enqueue() -> None:
            await self._send_queue.put(text)

        asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)

    def _emit_status(self, status: Dict[str, Any]) -> None:
        self._on_status(status)

