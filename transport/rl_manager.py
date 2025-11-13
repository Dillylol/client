"""Background runner for the residual learning WebSocket server."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from client.config import ClientConfig
from client.rpm_model import RpmModel

from .ws_server import ShotServer


logger = logging.getLogger(__name__)


class RLManager:
    """Run the :class:`ShotServer` in a background thread for the UI."""

    def __init__(self, config_path: Path | str = Path("config.yml")) -> None:
        self._config_path = Path(config_path)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._lock = threading.Lock()

        self.on_event: Optional[Callable[[Dict[str, Any]], None]] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the RL server loop if it is not already running."""

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run_loop, name="RLManager", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the RL server loop."""

        loop: Optional[asyncio.AbstractEventLoop]
        stop_event: Optional[asyncio.Event]
        thread: Optional[threading.Thread]
        with self._lock:
            loop = self._loop
            stop_event = self._stop_event
            thread = self._thread
        if loop and stop_event:
            loop.call_soon_threadsafe(stop_event.set)
        if thread:
            thread.join(timeout=2.0)
        with self._lock:
            self._loop = None
            self._stop_event = None
            self._thread = None

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop_event = stop_event

        self._emit_status("starting")

        try:
            config = ClientConfig.load(self._config_path)
            model = RpmModel.load(config.paths.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load RL configuration")
            self._emit_error(f"RL config error: {exc}")
            self._emit_status("stopped")
            loop.close()
            with self._lock:
                self._loop = None
                self._stop_event = None
            return

        server = ShotServer(
            config,
            model,
            on_event=self._handle_server_event,
            on_status=self._handle_server_status,
        )

        try:
            loop.run_until_complete(server.run(stop_event=stop_event))
        except Exception as exc:  # noqa: BLE001
            logger.exception("RL server crashed")
            self._emit_error(f"RL server crashed: {exc}")
        finally:
            self._emit_status("stopped")
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            with contextlib.suppress(Exception):
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            with self._lock:
                self._loop = None
                self._stop_event = None

    # ------------------------------------------------------------------
    def _emit(self, payload: Dict[str, Any]) -> None:
        callback = self.on_event
        if not callback:
            return
        try:
            callback(payload)
        except Exception:  # noqa: BLE001
            logger.exception("RLManager callback failed")

    def _emit_status(self, state: str, **details: Any) -> None:
        payload: Dict[str, Any] = {
            "kind": "status",
            "state": state,
            "timestamp": time.time(),
        }
        if details:
            payload["details"] = details
        self._emit(payload)

    def _emit_error(self, message: str) -> None:
        self._emit({"kind": "error", "timestamp": time.time(), "message": str(message)})

    def _handle_server_event(self, event: Dict[str, Any]) -> None:
        payload = {
            "kind": "event",
            "timestamp": time.time(),
            "event": dict(event),
        }
        self._emit(payload)

    def _handle_server_status(self, status: Dict[str, Any]) -> None:
        payload = {
            "kind": "status",
            "timestamp": time.time(),
            "state": status.get("state", "unknown"),
            "details": dict(status),
        }
        self._emit(payload)
