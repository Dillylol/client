"""Async outbound WebSocket client for robot telemetry."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import websockets
from websockets import WebSocketClientProtocol


logger = logging.getLogger(__name__)

FrameCallback = Callable[[Dict[str, Any]], None]
StatusCallback = Callable[[Dict[str, Any]], None]


class WSClient:
    """Manage an outbound WebSocket connection with reconnection."""

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str],
        on_frame: FrameCallback,
        on_status: StatusCallback,
        loop: asyncio.AbstractEventLoop,
        backoff_min: float = 0.2,
        backoff_max: float = 5.0,
    ) -> None:
        self._raw_url = url
        self._token = token
        self._on_frame = on_frame
        self._on_status = on_status
        self._loop = loop
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max

        self._stop_event = asyncio.Event()
        self._send_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._connected = asyncio.Event()
        self._current_ws: Optional[WebSocketClientProtocol] = None

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(lambda: self._stop_event.set())

    async def run(self) -> None:
        backoff = self._backoff_min
        while not self._stop_event.is_set():
            try:
                url, headers = self._build_url_and_headers()
                self._emit_status(
                    {
                        "transport": "ws",
                        "state": "connecting",
                        "endpoint": url,
                    }
                )
                async with websockets.connect(url, extra_headers=headers, ping_interval=None) as ws:
                    self._current_ws = ws
                    self._connected.set()
                    self._emit_status(
                        {
                            "transport": "ws",
                            "state": "connected",
                            "endpoint": url,
                        }
                    )
                    backoff = self._backoff_min
                    await self._pump(ws)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("WebSocket client error: %s", exc)
                self._emit_status(
                    {
                        "transport": "ws",
                        "state": "error",
                        "endpoint": self._raw_url,
                        "last_error": str(exc),
                    }
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._backoff_max)
            finally:
                self._connected.clear()
                self._current_ws = None
                if not self._stop_event.is_set():
                    self._emit_status(
                        {
                            "transport": "ws",
                            "state": "reconnecting",
                            "endpoint": self._raw_url,
                        }
                    )

        self._emit_status(
            {
                "transport": "ws",
                "state": "stopped",
                "endpoint": self._raw_url,
            }
        )

    async def _pump(self, ws: WebSocketClientProtocol) -> None:
        sender = asyncio.create_task(self._sender(ws))
        receiver = asyncio.create_task(self._receiver(ws))
        done, pending = await asyncio.wait(
            {sender, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _sender(self, ws: WebSocketClientProtocol) -> None:
        while not self._stop_event.is_set():
            message = await self._send_queue.get()
            await ws.send(message)

    async def _receiver(self, ws: WebSocketClientProtocol) -> None:
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
                            "transport": "ws",
                            "state": "bad_frame",
                            "raw": line,
                        }
                    )
                    continue
                self._on_frame(frame)

    def _build_url_and_headers(self) -> tuple[str, Dict[str, str]]:
        if not self._token:
            return self._raw_url, {}
        parsed = urlparse(self._raw_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "token" not in query:
            query["token"] = self._token
        new_query = urlencode(query)
        new_parts = parsed._replace(query=new_query)
        url = urlunparse(new_parts)
        headers = {"Authorization": f"Bearer {self._token}"}
        return url, headers

    def _emit_status(self, status: Dict[str, Any]) -> None:
        status["transport"] = status.get("transport", "ws")
        self._on_status(status)

    def queue_json(self, payload: Dict[str, Any]) -> None:
        text = json.dumps(payload)
        self.queue_text(text)

    def queue_text(self, text: str) -> None:
        async def _enqueue() -> None:
            await self._send_queue.put(text)

        asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)

    async def wait_connected(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

