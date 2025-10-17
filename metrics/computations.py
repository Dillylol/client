"""Telemetry vitals and derived metric computations."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional


class MetricsComputer:
    def __init__(
        self,
        *,
        on_metrics: Optional[Callable[[Dict[str, Any]], None]] = None,
        emit_interval: float = 0.25,
    ) -> None:
        self.on_metrics = on_metrics
        self.emit_interval = emit_interval

        self._battery_v: Optional[float] = None
        self._active_opmode: Optional[str] = None
        self._last_heartbeat_ts: Optional[int] = None
        self._last_ping_ms: Optional[float] = None
        self._connected: bool = False

        self.last_metrics: Dict[str, Any] = {}
        self._last_emit = 0.0
        self._dirty = False

    # ------------------------------------------------------------------
    def update_from_frame(self, frame: Dict[str, Any]) -> None:
        frame_type = frame.get("type")
        if frame_type == "heartbeat":
            ts_ms = self._extract_ts(frame)
            self._last_heartbeat_ts = ts_ms
            battery = frame.get("battery_v") or frame.get("batteryV")
            if battery is not None:
                try:
                    self._battery_v = float(battery)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    pass
            opmode = frame.get("active_opmode") or frame.get("activeOpMode")
            if isinstance(opmode, str):
                self._active_opmode = opmode
            self._dirty = True
        elif frame_type == "pong":
            t0 = frame.get("t0")
            if isinstance(t0, (int, float)):
                now_ms = int(time.time() * 1000)
                ping = now_ms - int(t0)
                if ping >= 0:
                    self._last_ping_ms = ping
                    self._dirty = True
        elif frame_type == "snapshot":
            data = frame.get("data")
            if isinstance(data, dict):
                self._extract_from_tree(data)
        elif frame_type == "diff":
            patch = frame.get("patch")
            if isinstance(patch, dict):
                self._extract_from_patch(patch)

    def update_from_flat(self, flat: Dict[str, Any]) -> None:
        battery = self._first_present(flat, ["vitals.battery_v", "battery_v", "power.battery_v"])
        if battery is not None:
            try:
                value = float(battery)
            except (TypeError, ValueError):
                value = None
            if value is not None and value != self._battery_v:
                self._battery_v = value
                self._dirty = True
        opmode = self._first_present(flat, ["meta.active_opmode", "active_opmode", "robot.active_opmode"])
        if isinstance(opmode, str) and opmode != self._active_opmode:
            self._active_opmode = opmode
            self._dirty = True

    def set_connection_state(self, connected: bool) -> None:
        if connected != self._connected:
            self._connected = connected
            self._dirty = True

    def compute_tick(self) -> None:
        now = time.time()
        if not self._dirty and now - self._last_emit < self.emit_interval:
            return
        metrics = self._build_metrics()
        if metrics != self.last_metrics or now - self._last_emit >= self.emit_interval:
            self.last_metrics = metrics
            self._last_emit = now
            self._dirty = False
            if self.on_metrics:
                self.on_metrics(metrics)

    # ------------------------------------------------------------------
    def _build_metrics(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        heartbeat_age = None
        if self._last_heartbeat_ts is not None:
            heartbeat_age = max(0, now_ms - self._last_heartbeat_ts)
        connected = self._connected or (heartbeat_age is not None and heartbeat_age < 3000)
        return {
            "battery_v": self._battery_v,
            "heartbeat_age_ms": heartbeat_age,
            "active_opmode": self._active_opmode,
            "ping_ms": self._last_ping_ms,
            "connected": bool(connected),
        }

    def _extract_from_tree(self, tree: Dict[str, Any]) -> None:
        vitals = tree.get("vitals")
        if isinstance(vitals, dict):
            battery = vitals.get("battery_v") or vitals.get("batteryV")
            if battery is not None:
                try:
                    self._battery_v = float(battery)
                    self._dirty = True
                except (TypeError, ValueError):
                    pass
        meta = tree.get("meta")
        if isinstance(meta, dict):
            opmode = meta.get("active_opmode") or meta.get("activeOpMode")
            if isinstance(opmode, str):
                self._active_opmode = opmode
                self._dirty = True

    def _extract_from_patch(self, patch: Dict[str, Any]) -> None:
        for key, value in patch.items():
            if key.endswith("battery_v") or key.endswith("batteryV"):
                try:
                    self._battery_v = float(value)
                    self._dirty = True
                except (TypeError, ValueError):
                    continue
            if key.endswith("active_opmode") or key.endswith("activeOpMode"):
                if isinstance(value, str):
                    self._active_opmode = value
                    self._dirty = True

    def _first_present(self, flat: Dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in flat and flat[key] is not None:
                return flat[key]
        return None

    def _extract_ts(self, frame: Dict[str, Any]) -> int:
        ts = frame.get("ts_ms") or frame.get("timestamp") or frame.get("ts")
        if isinstance(ts, (int, float)):
            if ts > 1_000_000_000_000:
                return int(ts)
            return int(ts * 1000)
        return int(time.time() * 1000)

