"""Schema-agnostic telemetry ingestion pipeline."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .state_store import StateStore


class Ingester:
    def __init__(
        self,
        *,
        series_capacity: int = 10_000,
        throttle_ms: int = 100,
        global_cap: int = 2_000_000,
    ) -> None:
        self._store = StateStore(
            series_capacity=series_capacity,
            throttle_ms=throttle_ms,
            global_cap=global_cap,
        )
        self._lock = threading.Lock()
        self.on_tree: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_flat: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_stdout: Optional[Callable[[str], None]] = None
        self.on_unknown: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_change_ts = 0

    # ------------------------------------------------------------------
    def handle_frame(self, frame: Dict[str, Any]) -> None:
        frame_type = frame.get("type")
        ts_ms = self._extract_ts(frame)
        changed_keys: List[str] = []

        with self._lock:
            if frame_type == "snapshot":
                data = frame.get("data")
                if isinstance(data, dict):
                    changed_keys = self._store.handle_snapshot(data, ts_ms)
                else:
                    # treat entire frame minus metadata as snapshot
                    payload = {k: v for k, v in frame.items() if k not in {"type", "ts", "ts_ms"}}
                    if payload:
                        changed_keys = self._store.handle_snapshot(payload, ts_ms)
            elif frame_type == "diff":
                patch = frame.get("patch")
                if isinstance(patch, dict):
                    changed_keys = self._store.handle_diff(patch, ts_ms)
            elif frame_type == "heartbeat":
                changed_keys = self._store.update_branch("meta.heartbeat", frame, ts_ms)
            elif frame_type in {"stdout", "stderr", "log"}:
                self._emit_stdout(frame)
            else:
                if self.on_unknown:
                    self.on_unknown(frame)
                self._emit_stdout_raw(frame)

        if changed_keys:
            self._last_change_ts = ts_ms
            if self.on_tree:
                self.on_tree(self._store.get_live_tree())
            if self.on_flat:
                self.on_flat(self._store.get_flat())

    # ------------------------------------------------------------------
    def get_live_tree(self) -> Dict[str, Any]:
        with self._lock:
            return self._store.get_live_tree()

    def get_flat(self) -> Dict[str, Any]:
        with self._lock:
            return self._store.get_flat()

    def get_series(self, key: str) -> List[tuple[int, Any]]:
        with self._lock:
            return self._store.get_series(key)

    def export_csv(self, keys: List[str], window_ms: int) -> bytes:
        with self._lock:
            return self._store.export_csv(keys, window_ms)

    # ------------------------------------------------------------------
    def _emit_stdout(self, frame: Dict[str, Any]) -> None:
        if not self.on_stdout:
            return
        line = frame.get("line") or frame.get("message") or frame.get("text")
        if isinstance(line, str):
            prefix = frame.get("type", "stdout")
            self.on_stdout(f"[{prefix}] {line}")

    def _emit_stdout_raw(self, frame: Dict[str, Any]) -> None:
        if not self.on_stdout:
            return
        try:
            text = json.dumps(frame)
        except TypeError:
            text = repr(frame)
        self.on_stdout(f"[unknown] {text}")

    def _extract_ts(self, frame: Dict[str, Any]) -> int:
        ts = frame.get("ts_ms") or frame.get("timestamp") or frame.get("ts")
        if isinstance(ts, (int, float)):
            if ts > 1_000_000_000_000:  # probably milliseconds
                return int(ts)
            return int(ts * 1000)
        return int(time.time() * 1000)

