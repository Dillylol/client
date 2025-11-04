"""Shot evaluation logic coordinating RPM model updates and planning."""
from __future__ import annotations

import csv
import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

from .config import ClientConfig
from .rpm_model import RpmModel


@dataclass
class BandStats:
    """Simple accumulator for hit-rate tracking within a band."""

    hits: int = 0
    tries: int = 0

    def update(self, hit: bool) -> None:
        self.tries += 1
        if hit:
            self.hits += 1

    @property
    def p_hit(self) -> float:
        if self.tries == 0:
            return 0.0
        return self.hits / self.tries


@dataclass
class PendingCommand:
    """Track a command that has been issued but not yet resolved."""

    cmd_id: str
    shot_id: str
    distance_in: float
    voltage: float
    rpm_base: float
    rpm_bias: int
    rpm_target: float
    range_delta_in: int
    sent_ts_ms: int
    expires_ts_ms: int
    request_seq: int
    rpm_tgt_cmd: Optional[float] = None
    rpm_at_fire: Optional[float] = None
    metadata: Dict[str, Any] | None = None


class ShotEvaluator:
    """Handle shot planning, residual learning, and persistence."""

    def __init__(self, config: ClientConfig, model: RpmModel) -> None:
        self.config = config
        self.model = model
        self.model.ensure_grid(config.distance_bins, config.voltage_bins)

        self.session_id: Optional[str] = None
        self._shot_counter = 0
        self._command_history: Dict[int, Dict[str, object]] = {}
        self._pending_commands: Dict[str, PendingCommand] = {}
        self._pending_by_shot: Dict[str, PendingCommand] = {}
        self._processed_shots: set[str] = set()
        self._band_stats: Dict[Tuple[float, float], BandStats] = {}
        self._band_bias_adjust: Dict[Tuple[float, float], int] = {}
        self._band_residual_sign: Dict[Tuple[float, float], int] = {}
        self._pending_step: Dict[Tuple[float, float], float] = {}
        self._last_outcome: Dict[Tuple[float, float], str] = {}
        self._residual_history: Deque[float] = deque(maxlen=200)
        self._shots_since_update = 0
        self._total_shots = 0
        self._last_update_shot_index = 0
        self._last_update_rmse = math.inf

        self._anchor_band: Optional[Tuple[float, float]] = (
            tuple(self.config.distance_bins[0]) if self.config.distance_bins else None
        )
        if not self.config.policy.start_with_anchor:
            self._anchor_band = None
        self._anchor_completed = not self.config.policy.start_with_anchor
        self._anchor_shots = 0
        self._anchor_first_command_sent = False

        self._logger = logging.getLogger(__name__)

        self._shots_log_path = self.config.paths.logs_dir / "shots.csv"
        if not self._shots_log_path.exists():
            self._write_shots_header()

    # -----------------------------------------------------------------
    def start_session(self, session_id: str) -> None:
        """Reset rolling state for a new robot session."""
        self.session_id = session_id
        self._shot_counter = 0
        self._command_history.clear()
        self._pending_commands.clear()
        self._pending_by_shot.clear()
        self._processed_shots.clear()
        self._band_bias_adjust.clear()
        self._band_residual_sign.clear()
        self._pending_step.clear()
        self._last_outcome.clear()
        self._residual_history.clear()
        self._shots_since_update = 0
        self._total_shots = 0
        self._last_update_shot_index = 0
        self._last_update_rmse = math.inf
        self._band_stats = {tuple(band): BandStats() for band in self.config.distance_bins}
        if self.config.policy.start_with_anchor and self.config.distance_bins:
            self._anchor_band = tuple(self.config.distance_bins[0])
            self._anchor_completed = False
        else:
            self._anchor_band = None
            self._anchor_completed = True
        self._anchor_shots = 0
        self._anchor_first_command_sent = False

        manifest_path = self.config.paths.logs_dir / f"session_{session_id}.yml"
        manifest = {
            "session_id": session_id,
            "start_ts": datetime.now(timezone.utc).isoformat(),
            "model_version_start": self.model.model_version,
            "bins_config": {
                "distance": [list(b) for b in self.config.distance_bins],
                "voltage": [list(b) for b in self.config.voltage_bins],
            },
            "policy_name": self.config.policy.name,
            "policy_flags": {
                "send_abs_rpm": self.config.policy.send_abs_rpm,
                "bias_step_rpm": self.config.policy.bias_step_rpm,
                "bias_cap_rpm": self.config.policy.bias_cap_rpm,
                "start_with_anchor": self.config.policy.start_with_anchor,
                "anchor_samples_target": self.config.policy.anchor_samples_target,
            },
        }
        manifest_path.write_text(_to_yaml(manifest))

    # -----------------------------------------------------------------
    def plan_shot(self, request: Dict[str, object], now_ms: int) -> Dict[str, object]:
        """Produce a command in response to ``request_shot_plan``."""
        if self.session_id is None:
            raise RuntimeError("Session not initialized")

        raw_seq = request.get("seq")
        try:
            seq = int(raw_seq)
        except (TypeError, ValueError):
            seq = -1
        if seq in self._command_history and seq >= 0:
            return self._command_history[seq]

        distance = request.get("distance_in")
        voltage = request.get("v_batt_load")
        if distance is None or voltage is None:
            self._logger.warning("request_shot_plan missing distance or voltage; issuing noop")
            return self._build_noop_cmd(now_ms, reason="missing_fields")

        distance_f = float(distance)
        voltage_f = float(voltage)
        band = self._distance_band(distance_f)
        loiter = self._last_outcome.get(band) == "miss"

        self._expire_commands(now_ms)

        rpm_base = _coerce_float(request.get("rpm_base"), self.model.predict(distance_f, voltage_f))
        delta = self.model.delta(distance_f, voltage_f)
        rpm_bias_base = int(round(delta))
        rpm_bias_adjust = self._band_bias_adjust.get(band, 0)
        rpm_bias = rpm_bias_base + rpm_bias_adjust

        if self._anchor_band == band and not self._anchor_first_command_sent:
            rpm_bias = 0
            self._anchor_first_command_sent = True

        range_delta = 0.0
        if not loiter:
            pending_step = self._pending_step.pop(band, 0.0)
            if pending_step and (self._anchor_completed or band != self._anchor_band):
                range_delta = pending_step
            else:
                if pending_step:
                    self._pending_step[band] = pending_step
                range_delta = 0.0

        cmd_id = str(uuid.uuid4())
        shot_id = str(request.get("shot_id") or f"{self.session_id}-{self._shot_counter + 1}")
        rpm_target = rpm_base + rpm_bias
        range_delta_int = int(round(range_delta))

        command: Dict[str, Any] = {
            "type": "cmd",
            "session_id": self.session_id,
            "cmd_id": cmd_id,
            "shot_id": shot_id,
            "valid_ms": self.config.cmd_valid_ms,
            "ts_ms": now_ms,
            "fire_now": True,
            "loiter": bool(loiter),
            "range_delta_in": range_delta_int,
            "request_seq": seq,
        }
        if self.config.policy.send_abs_rpm:
            command["rpm_target_abs"] = int(round(rpm_target))
        else:
            command["rpm_bias"] = int(round(rpm_bias))

        expires = now_ms + self.config.cmd_valid_ms
        pending = PendingCommand(
            cmd_id=cmd_id,
            shot_id=shot_id,
            distance_in=distance_f,
            voltage=voltage_f,
            rpm_base=rpm_base,
            rpm_bias=int(round(rpm_bias)),
            rpm_target=rpm_target,
            range_delta_in=range_delta_int,
            sent_ts_ms=now_ms,
            expires_ts_ms=expires,
            request_seq=seq,
            metadata={"request": request},
        )
        self._pending_commands[cmd_id] = pending
        self._pending_by_shot[shot_id] = pending
        if seq >= 0:
            self._command_history[seq] = command
            if len(self._command_history) > 128:
                for old_seq in sorted(self._command_history.keys())[:-128]:
                    self._command_history.pop(old_seq, None)
        self._shot_counter += 1
        return command

    def record_shot_fired(self, payload: Dict[str, object]) -> None:
        """Annotate pending command with the detected shot-fire token."""
        shot_id = str(payload.get("shot_id")) if payload.get("shot_id") else None
        if shot_id and shot_id in self._pending_by_shot:
            pending = self._pending_by_shot[shot_id]
            # Extend validity slightly to wait for result
            pending.expires_ts_ms += self.config.cmd_valid_ms
            distance = _coerce_float(
                payload.get("range_in") or payload.get("distance_in"), pending.distance_in
            )
            voltage = _coerce_float(payload.get("v_batt_load"), pending.voltage)
            pending.distance_in = distance
            pending.voltage = voltage
            pending.rpm_tgt_cmd = _coerce_float(
                payload.get("rpm_tgt_cmd") or payload.get("rpm_target"), pending.rpm_target
            )
            pending.rpm_at_fire = _coerce_float(payload.get("rpm_at_fire"), pending.rpm_at_fire or pending.rpm_target)

    def apply_shot_result(self, payload: Dict[str, object], now_ms: int) -> Optional[Dict[str, object]]:
        """Process the ``shot_result`` message and optionally trigger a model update."""
        if self.session_id is None:
            raise RuntimeError("Session not initialized")
        shot_id = str(payload.get("shot_id"))
        if not shot_id or shot_id in self._processed_shots:
            return None
        pending = self._pending_by_shot.pop(shot_id, None)
        if pending:
            self._pending_commands.pop(pending.cmd_id, None)
        self._processed_shots.add(shot_id)

        distance = _coerce_float(
            payload.get("distance_in") or payload.get("range_in"),
            pending.distance_in if pending else 0.0,
        )
        voltage = _coerce_float(payload.get("v_batt_load"), pending.voltage if pending else 12.0)
        rpm_base = _coerce_float(
            payload.get("rpm_base"),
            pending.rpm_base if pending else self.model.predict(distance, voltage),
        )
        if pending and pending.rpm_tgt_cmd is not None:
            rpm_tgt_default = pending.rpm_tgt_cmd
        elif pending:
            rpm_tgt_default = pending.rpm_target
        else:
            rpm_tgt_default = rpm_base
        rpm_tgt_cmd = _coerce_float(payload.get("rpm_tgt_cmd"), rpm_tgt_default)
        rpm_at_fire = _coerce_float(
            payload.get("rpm_at_fire"), pending.rpm_at_fire if pending else rpm_tgt_cmd
        )
        hit_raw = payload.get("hit")
        if isinstance(hit_raw, bool):
            hit = hit_raw
        else:
            hit = bool(int(hit_raw or 0))
        pose_x = payload.get("pose_x")
        pose_y = payload.get("pose_y")
        heading = payload.get("heading_to_tag")

        band = self._distance_band(distance)
        residual = rpm_at_fire - rpm_base
        residual_sign = 1 if residual >= 0 else -1
        self._band_stats.setdefault(band, BandStats()).update(hit)
        self._last_outcome[band] = "hit" if hit else "miss"
        if hit:
            self._band_bias_adjust[band] = 0
            if self._anchor_band == band and not self._anchor_completed:
                self._anchor_shots += 1
                if self._anchor_shots >= self.config.policy.anchor_samples_target:
                    self._anchor_completed = True
                    self._pending_step[band] = float(self.config.distance_step_in)
            else:
                self._pending_step[band] = float(self.config.distance_step_in)
        else:
            prev = self._band_bias_adjust.get(band, 0)
            direction = residual_sign
            step = self.config.policy.bias_step_rpm * direction
            cap = self.config.policy.bias_cap_rpm
            updated = prev + step
            updated = max(-cap, min(cap, updated))
            self._band_bias_adjust[band] = updated

        self._band_residual_sign[band] = residual_sign
        self._residual_history.append(residual)

        self.model.update_residual(
            distance_in=distance,
            v_batt_load=voltage,
            rpm_at_fire=rpm_at_fire,
            alpha=self.config.alpha_ewma,
            distance_bins=self.config.distance_bins,
            voltage_bins=self.config.voltage_bins,
        )
        self.model.save(self.config.paths.model_path)

        self._shots_since_update += 1
        self._total_shots += 1

        self._log_shot(
            session_id=self.session_id,
            shot_id=shot_id,
            timestamp_ms=int(payload.get("ts_ms", now_ms)),
            distance=distance,
            voltage=voltage,
            rpm_base=rpm_base,
            rpm_bias=(pending.rpm_bias if pending else int(round(rpm_tgt_cmd - rpm_base))),
            rpm_tgt_cmd=rpm_tgt_cmd,
            rpm_at_fire=rpm_at_fire,
            hit=hit,
            pose_x=pose_x,
            pose_y=pose_y,
            heading=heading,
            notes=payload.get("notes"),
        )

        if self._should_emit_update():
            self.model.bump_version(
                note=f"fit@{datetime.now(timezone.utc).isoformat()} n={len(self._residual_history)}"
            )
            self.model.save(self.config.paths.model_path)
            self._shots_since_update = 0
            self._last_update_shot_index = self._total_shots
            self._last_update_rmse = _rmse(self._residual_history)
            return self.model.as_update_payload()
        return None

    # -----------------------------------------------------------------
    def _build_noop_cmd(self, now_ms: int, reason: str) -> Dict[str, object]:
        cmd = {
            "type": "cmd",
            "session_id": self.session_id,
            "cmd_id": str(uuid.uuid4()),
            "shot_id": f"noop-{now_ms}",
            "valid_ms": self.config.cmd_valid_ms,
            "ts_ms": now_ms,
            "fire_now": False,
            "loiter": False,
            "range_delta_in": 0,
            "reason": reason,
            "request_seq": None,
        }
        if self.config.policy.send_abs_rpm:
            cmd["rpm_target_abs"] = 0
        else:
            cmd["rpm_bias"] = 0
        return cmd

    def safe_noop(self, now_ms: int, reason: str) -> Dict[str, object]:
        """Public helper to provide a safe, no-op command."""

        return self._build_noop_cmd(now_ms, reason=reason)

    def _distance_band(self, distance_in: float) -> Tuple[float, float]:
        for band in self.config.distance_bins:
            lo, hi = band
            if lo <= distance_in < hi:
                return band
        # fallback to nearest band by center
        centers = [(abs(((lo + hi) / 2.0) - distance_in), (lo, hi)) for lo, hi in self.config.distance_bins]
        return min(centers, key=lambda x: x[0])[1]

    def _expire_commands(self, now_ms: int) -> None:
        expired = [cmd_id for cmd_id, pending in self._pending_commands.items() if pending.expires_ts_ms < now_ms]
        for cmd_id in expired:
            pending = self._pending_commands.pop(cmd_id)
            self._pending_by_shot.pop(pending.shot_id, None)

    def _write_shots_header(self) -> None:
        with self._shots_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "session_id",
                    "shot_id",
                    "ts",
                    "range_in",
                    "v_batt_load",
                    "rpm_base",
                    "rpm_bias",
                    "rpm_tgt_cmd",
                    "rpm_at_fire",
                    "hit",
                    "pose_x",
                    "pose_y",
                    "heading_to_tag",
                    "notes",
                ]
            )

    def _log_shot(
        self,
        *,
        session_id: str,
        shot_id: str,
        timestamp_ms: int,
        distance: float,
        voltage: float,
        rpm_base: float,
        rpm_bias: int,
        rpm_tgt_cmd: float,
        rpm_at_fire: float,
        hit: bool,
        pose_x: Optional[float],
        pose_y: Optional[float],
        heading: Optional[float],
        notes: Optional[object],
    ) -> None:
        with self._shots_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    session_id,
                    shot_id,
                    timestamp_ms,
                    distance,
                    voltage,
                    rpm_base,
                    rpm_bias,
                    rpm_tgt_cmd,
                    rpm_at_fire,
                    int(hit),
                    pose_x,
                    pose_y,
                    heading,
                    notes,
                ]
            )

    def _should_emit_update(self) -> bool:
        if self._shots_since_update < self.config.update_cadence_shots:
            return False
        if (self._total_shots - self._last_update_shot_index) < self.config.update_rate_limit:
            return False
        current_rmse = _rmse(self._residual_history)
        if not math.isfinite(current_rmse):
            return False
        if not math.isfinite(self._last_update_rmse):
            return True
        improvement = self._last_update_rmse - current_rmse
        threshold = self._last_update_rmse * (self.config.update_rmse_improvement_pct / 100.0)
        return improvement >= threshold


def _rmse(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.inf
    return math.sqrt(sum(v * v for v in values) / len(values))


def _to_yaml(payload: Dict[str, object]) -> str:
    try:
        import yaml

        return yaml.safe_dump(payload, sort_keys=True)
    except Exception:
        # Fallback minimal YAML/JSON hybrid if PyYAML unavailable
        import json

        return json.dumps(payload, indent=2, sort_keys=True)


def _coerce_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)
