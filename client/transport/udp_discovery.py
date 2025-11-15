"""UDP discovery listener for robot heartbeats."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)

FrameCallback = Callable[[Dict[str, Any]], None]


class UDPDiscovery:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 27182,
        loop: asyncio.AbstractEventLoop,
        on_frame: FrameCallback,
    ) -> None:
        self._host = host
        self._port = port
        self._loop = loop
        self._on_frame = on_frame

        self._transport: Optional[asyncio.DatagramTransport] = None
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(lambda: self._stop_event.set())

    async def run(self) -> None:
        loop = self._loop

        class _Protocol(asyncio.DatagramProtocol):
            def __init__(self, cb: FrameCallback) -> None:
                self._cb = cb

            def datagram_received(self, data: bytes, addr) -> None:  # type: ignore[override]
                try:
                    text = data.decode("utf-8", errors="ignore").strip()
                except Exception:  # noqa: BLE001
                    return
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Malformed UDP frame: %s", line)
                        continue
                    self._cb(frame)

        transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self._on_frame),
            local_addr=(self._host, self._port),
            reuse_port=True,
        )
        self._transport = transport
        try:
            await self._stop_event.wait()
        finally:
            transport.close()
            self._transport = None

