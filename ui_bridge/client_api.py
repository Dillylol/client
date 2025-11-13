"""Glue between transport, ingestion, metrics, and the GUI."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Dict, Optional

from ingest.ingester import Ingester
from metrics.computations import MetricsComputer
from transport.link_manager import LinkManager
from transport.rl_manager import RLManager


class ClientAPI:
    def __init__(self) -> None:
        self.link = LinkManager()
        self.ingester = Ingester()
        self.metrics = MetricsComputer(on_metrics=self._emit_metrics)
        self.rl_manager = RLManager()

        self.on_status: Optional[Callable[[Dict[str, object]], None]] = None
        self.on_metrics: Optional[Callable[[Dict[str, object]], None]] = None
        self.on_telemetry_tree: Optional[Callable[[Dict[str, object]], None]] = None
        self.on_flat: Optional[Callable[[Dict[str, object]], None]] = None
        self.on_stdout: Optional[Callable[[str], None]] = None
        self.on_raw_frame: Optional[Callable[[Dict[str, object]], None]] = None
        self.on_rl_event: Optional[Callable[[Dict[str, object]], None]] = None

        self.link.on_frame = self._handle_frame
        self.link.on_status = self._handle_status
        self.ingester.on_tree = self._handle_tree_update
        self.ingester.on_flat = self._handle_flat_update
        self.ingester.on_stdout = self._handle_stdout
        self.rl_manager.on_event = self._handle_rl_event

        self.rl_manager.start()

        self._metrics_lock = threading.Lock()

        self._status_queue: "queue.SimpleQueue[Dict[str, object]]" = queue.SimpleQueue()
        self._stdout_queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._frame_queue: "queue.SimpleQueue[Dict[str, object]]" = queue.SimpleQueue()
        self._rl_queue: "queue.SimpleQueue[Dict[str, object]]" = queue.SimpleQueue()

        self._latest_tree: Optional[Dict[str, object]] = None
        self._latest_flat: Optional[Dict[str, object]] = None
        self._tree_dirty = False
        self._flat_dirty = False
        self._flat_pending_metrics = False

        self._last_tree_emit = 0.0
        self._last_flat_emit = 0.0

        self._tick_thread: Optional[threading.Thread] = None
        self._tick_stop = threading.Event()

        self._url: Optional[str] = None
        self._token: Optional[str] = None

    # ------------------------------------------------------------------
    def configure(self, url: str, token: Optional[str]) -> None:
        self._url = url
        self._token = token
        self.link.configure(url, token)

    def start(self) -> None:
        if not self._url:
            raise RuntimeError("Call configure() before start().")
        self._tick_stop.clear()
        if not self._tick_thread or not self._tick_thread.is_alive():
            self._tick_thread = threading.Thread(target=self._tick_loop, name="ClientAPI", daemon=True)
            self._tick_thread.start()
        self.link.start()

    def stop(self) -> None:
        self.link.stop()
        self._tick_stop.set()
        if self._tick_thread and self._tick_thread.is_alive():
            self._tick_thread.join(timeout=1.0)
        self._tick_thread = None

    def send_cmd(self, text: str) -> None:
        self.link.send_cmd(text)

    def manual_ping(self) -> None:
        self.link.manual_ping()

    def flush_rl_events(self) -> None:
        self._drain_rl_events()

    # ------------------------------------------------------------------
    def _handle_frame(self, frame: Dict[str, object]) -> None:
        with self._metrics_lock:
            self.metrics.update_from_frame(frame)
        self.ingester.handle_frame(frame)
        self._frame_queue.put(frame)

    def _handle_status(self, status: Dict[str, object]) -> None:
        state = str(status.get("state", "")).lower()
        if state in {"connected"}:
            with self._metrics_lock:
                self.metrics.set_connection_state(True)
        elif state in {"error", "stopped", "reconnecting", "connecting"}:
            with self._metrics_lock:
                self.metrics.set_connection_state(False)
        self._status_queue.put(status)

    def _handle_tree_update(self, tree: Dict[str, object]) -> None:
        self._latest_tree = tree
        self._tree_dirty = True

    def _handle_flat_update(self, flat: Dict[str, object]) -> None:
        self._latest_flat = flat
        self._flat_dirty = True
        self._flat_pending_metrics = True

    def _handle_stdout(self, line: str) -> None:
        self._stdout_queue.put(line)

    def _handle_rl_event(self, event: Dict[str, object]) -> None:
        if self.on_rl_event:
            self.on_rl_event(event)
        else:
            self._rl_queue.put(event)

    def _emit_metrics(self, metrics: Dict[str, object]) -> None:
        if self.on_metrics:
            self.on_metrics(metrics)

    # ------------------------------------------------------------------
    def _tick_loop(self) -> None:
        tree_interval = 0.5
        flat_interval = 0.3
        while not self._tick_stop.is_set():
            self._drain_status()
            self._drain_stdout()
            self._drain_frames()
            self._drain_rl_events()

            if self._flat_pending_metrics and self._latest_flat is not None:
                with self._metrics_lock:
                    self.metrics.update_from_flat(self._latest_flat)
                self._flat_pending_metrics = False

            now = time.time()
            if self._tree_dirty and self._latest_tree is not None and now - self._last_tree_emit >= tree_interval:
                if self.on_telemetry_tree:
                    self.on_telemetry_tree(self._latest_tree)
                self._tree_dirty = False
                self._last_tree_emit = now

            if self._flat_dirty and self._latest_flat is not None and now - self._last_flat_emit >= flat_interval:
                if self.on_flat:
                    self.on_flat(self._latest_flat)
                self._flat_dirty = False
                self._last_flat_emit = now

            with self._metrics_lock:
                self.metrics.compute_tick()

            time.sleep(0.05)

    def _drain_status(self) -> None:
        while True:
            try:
                status = self._status_queue.get_nowait()
            except queue.Empty:
                break
            if self.on_status:
                self.on_status(status)

    def _drain_stdout(self) -> None:
        while True:
            try:
                line = self._stdout_queue.get_nowait()
            except queue.Empty:
                break
            if self.on_stdout:
                self.on_stdout(line)

    def _drain_frames(self) -> None:
        if not self.on_raw_frame:
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            return
        while True:
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            self.on_raw_frame(frame)

    def _drain_rl_events(self) -> None:
        if not self.on_rl_event:
            return
        while True:
            try:
                event = self._rl_queue.get_nowait()
            except queue.Empty:
                break
            self.on_rl_event(event)

