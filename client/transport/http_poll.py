"""HTTP polling transport for read-only vitals."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import requests


logger = logging.getLogger(__name__)

FrameCallback = Callable[[Dict[str, Any]], None]
StatusCallback = Callable[[Dict[str, Any]], None]


class HTTPPoller:
    def __init__(
        self,
        url: str,
        *,
        token: Optional[str],
        loop: asyncio.AbstractEventLoop,
        interval: float = 0.5,
        on_frame: FrameCallback,
        on_status: StatusCallback,
    ) -> None:
        self._raw_url = url
        self._token = token
        self._loop = loop
        self._interval = interval
        self._on_frame = on_frame
        self._on_status = on_status
        self._stop_event = asyncio.Event()
        self._session = requests.Session()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(lambda: self._stop_event.set())

    async def run(self) -> None:
        url = self._build_url()
        headers = self._build_headers()
        self._emit_status({"transport": "http", "state": "connecting", "endpoint": url})
        while not self._stop_event.is_set():
            try:
                data = await self._loop.run_in_executor(None, self._fetch, url, headers)
                if data is not None:
                    frame = {
                        "type": "snapshot",
                        "ts_ms": int(time.time() * 1000),
                        "data": data,
                    }
                    self._on_frame(frame)
                    self._emit_status({"transport": "http", "state": "connected", "endpoint": url})
            except Exception as exc:  # noqa: BLE001
                logger.exception("HTTP polling error: %s", exc)
                self._emit_status(
                    {
                        "transport": "http",
                        "state": "error",
                        "endpoint": url,
                        "last_error": str(exc),
                    }
                )
            await asyncio.sleep(self._interval)
        self._emit_status({"transport": "http", "state": "stopped", "endpoint": url})

    def _fetch(self, url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        response = self._session.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return json.loads(text)

    def _build_url(self) -> str:
        if not self._token:
            return self._raw_url
        parsed = urlparse(self._raw_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "token" not in query:
            query["token"] = self._token
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _build_headers(self) -> Dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def _emit_status(self, status: Dict[str, Any]) -> None:
        self._on_status(status)

