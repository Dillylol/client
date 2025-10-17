"""Transport orchestration for robot telemetry links."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlparse

from .http_poll import HTTPPoller
from .udp_discovery import UDPDiscovery
from .ws_client import WSClient
from .ws_server import WSServer


logger = logging.getLogger(__name__)

FrameCallback = Callable[[dict], None]
StatusCallback = Callable[[dict], None]


class LinkManager:
    def __init__(self) -> None:
        self.on_frame: Optional[FrameCallback] = None
        self.on_status: Optional[StatusCallback] = None

        self._url: Optional[str] = None
        self._token: Optional[str] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_async: Optional[asyncio.Event] = None

        self._ws_client: Optional[WSClient] = None
        self._ws_client_task: Optional[asyncio.Task] = None

        self._ws_server: Optional[WSServer] = None
        self._ws_server_task: Optional[asyncio.Task] = None

        self._http_poller: Optional[HTTPPoller] = None
        self._http_task: Optional[asyncio.Task] = None

        self._udp_listener: Optional[UDPDiscovery] = None
        self._udp_task: Optional[asyncio.Task] = None

        self._tasks: list[asyncio.Task] = []

        self._lock = threading.Lock()
        self._dispatch_frame: Optional[FrameCallback] = None
        self._dispatch_status: Optional[StatusCallback] = None

    # ------------------------------------------------------------------
    def configure(self, url: str, token: Optional[str]) -> None:
        with self._lock:
            self._url = url.strip()
            self._token = token.strip() if token else None

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if not self._url:
                raise RuntimeError("LinkManager.configure must be called before start")
            self._thread = threading.Thread(target=self._run_loop, name="LinkManager", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._ws_client:
            self._ws_client.stop()
        if self._ws_server:
            self._ws_server.stop()
        if self._http_poller:
            self._http_poller.stop()
        if self._udp_listener:
            self._udp_listener.stop()
        if self._stop_async:
            loop.call_soon_threadsafe(self._stop_async.set)
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._loop = None
        self._stop_async = None
        self._ws_client = None
        self._ws_client_task = None
        self._ws_server = None
        self._ws_server_task = None
        self._http_poller = None
        self._http_task = None
        self._udp_listener = None
        self._udp_task = None
        self._tasks = []
        self._dispatch_frame = None
        self._dispatch_status = None

    def manual_ping(self) -> None:
        payload = {"type": "ping", "id": int(time.time() * 1000), "t0": int(time.time() * 1000)}
        self._send_ws_json(payload)

    def send_cmd(self, text: str) -> None:
        payload = {"type": "cmd", "text": text, "id": int(time.time() * 1000)}
        self._send_ws_json(payload)

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        url = self._url
        token = self._token
        if not url:
            return
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stop_async = asyncio.Event()
        self._tasks = []
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        def dispatch_frame(frame: dict) -> None:
            if self.on_frame:
                self.on_frame(frame)

        def dispatch_status(status: dict) -> None:
            status.setdefault("timestamp", time.time())
            if self.on_status:
                self.on_status(status)

        self._dispatch_frame = dispatch_frame
        self._dispatch_status = dispatch_status

        # Always start UDP discovery
        self._udp_listener = UDPDiscovery(loop=loop, on_frame=lambda f: self._handle_udp_frame(f))
        self._udp_task = loop.create_task(self._udp_listener.run())
        self._tasks.append(self._udp_task)

        if scheme in {"ws", "wss"}:
            self._start_ws_client(loop, url, token)
        elif scheme == "listen":
            host = parsed.hostname or "0.0.0.0"
            port = parsed.port or 8765
            path = parsed.path or "/stream"
            self._ws_server = WSServer(
                host=host,
                port=port,
                path=path,
                token=token,
                loop=loop,
                on_frame=dispatch_frame,
                on_status=dispatch_status,
            )
            self._ws_server_task = loop.create_task(self._ws_server.run())
            self._tasks.append(self._ws_server_task)
        elif scheme in {"http", "https"}:
            self._http_poller = HTTPPoller(
                url=url,
                token=token,
                loop=loop,
                on_frame=dispatch_frame,
                on_status=dispatch_status,
            )
            self._http_task = loop.create_task(self._http_poller.run())
            self._tasks.append(self._http_task)
        elif scheme == "udp":
            dispatch_status({"transport": "udp", "state": "listening", "endpoint": url})
        else:
            dispatch_status({"transport": "unknown", "state": "error", "endpoint": url, "last_error": f"Unsupported scheme: {scheme}"})

        try:
            loop.run_until_complete(self._stop_async.wait())
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                loop.run_until_complete(asyncio.gather(*self._tasks, return_exceptions=True))
            loop.stop()
            loop.close()

    def _start_ws_client(self, loop: asyncio.AbstractEventLoop, url: str, token: Optional[str]) -> None:
        if self._dispatch_frame is None or self._dispatch_status is None:
            return
        self._ws_client = WSClient(
            url,
            token=token,
            on_frame=self._dispatch_frame,
            on_status=self._dispatch_status,
            loop=loop,
        )
        self._ws_client_task = loop.create_task(self._ws_client.run())
        self._tasks.append(self._ws_client_task)

    def _send_ws_json(self, payload: dict) -> None:
        if self._ws_client:
            self._ws_client.queue_json(payload)
        elif self._ws_server:
            self._ws_server.queue_json(payload)

    def _handle_udp_frame(self, frame: dict) -> None:
        if self._dispatch_frame:
            self._dispatch_frame(frame)
        if frame.get("type") == "heartbeat":
            ws_url = frame.get("ws_url") or frame.get("wsUrl")
            if ws_url and not self._ws_client and self._loop:
                logger.info("Discovered WS URL via UDP: %s", ws_url)
                self._loop.call_soon_threadsafe(self._start_ws_client, self._loop, ws_url, self._token)

