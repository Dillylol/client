"""Shot evaluation logic coordinating RPM model updates."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

from .rpm_model import RpmModel


class ShotEvaluator:
    """Handle shot planning and observation updates."""

    def __init__(
        self,
        model_path: Path,
        log_path: Path | None = None,
        update_interval: int = 10,
    ) -> None:
        self.model_path = model_path
        self.log_path = log_path or model_path.with_name("shot_log.csv")
        self.update_interval = update_interval
        self.model = RpmModel.load(model_path)

        self._shots_since_update = 0
        self._last_outcome: Dict[float, str] = {}
        self._last_residual: Dict[float, float] = {}

        if not self.log_path.exists():
            self._write_csv_header()

    # --- public API -------------------------------------------------
    def handle_request_shot_plan(self, payload: Dict[str, float]) -> Dict[str, object]:
        """Return the shot plan to send back to the robot."""
        distance = float(payload.get("distance_in", 0.0))
        voltage = float(payload.get("v_batt_load", 12.0))
        ts_ms = int(payload.get("ts_ms", time.time() * 1000))

        rpm_base = self.model.predict(distance, voltage)
        rpm_total = rpm_base + self.model.delta(distance, voltage)

        bin_key = self.model.bin_identifier(distance)
        last_outcome = self._last_outcome.get(bin_key)
        loiter = last_outcome == "miss"
        rpm_bias = 0
        if loiter:
            residual = self._last_residual.get(bin_key, 0.0)
            if residual < 0:
                rpm_bias = +20
            elif residual > 0:
                rpm_bias = -20

        plan = {
            "type": "shot_plan",
            "ts_ms": ts_ms,
            "rpm_bias": rpm_bias,
            "fire_now": True,
            "loiter": loiter,
            "rpm_tgt": round(rpm_total + rpm_bias, 2),
        }
        return plan

    def handle_obs_shot(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        """Process an observation from the robot.

        Returns an ``rpm_model_update`` message when enough samples have been
        accumulated, otherwise ``None``.
        """
        distance = float(payload.get("distance_in", 0.0))
        voltage = float(payload.get("v_batt_load", 12.0))
        rpm_at_fire = float(payload.get("rpm_at_fire", 0.0))
        hit = int(payload.get("hit", 0))

        self._log_shot(payload)

        bin_key = self.model.bin_identifier(distance)
        predicted = self.model.target(distance, voltage)
        residual = rpm_at_fire - predicted

        if hit:
            self._last_outcome[bin_key] = "hit"
            self._last_residual.pop(bin_key, None)
            self.model.update_residual(distance, voltage, rpm_at_fire)
            self.model.save(self.model_path)
            self._shots_since_update += 1
        else:
            self._last_outcome[bin_key] = "miss"
            self._last_residual[bin_key] = residual

        if self._shots_since_update >= self.update_interval:
            self._shots_since_update = 0
            return self.model.as_update_payload()
        return None

    # --- helpers ----------------------------------------------------
    def _write_csv_header(self) -> None:
        with self.log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "ts",
                    "distance_in",
                    "v_batt_load",
                    "rpm_tgt",
                    "rpm_at_fire",
                    "hit",
                    "pose_x",
                    "pose_y",
                    "heading_to_tag",
                ]
            )

    def _log_shot(self, payload: Dict[str, object]) -> None:
        with self.log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    payload.get("ts_ms", int(time.time() * 1000)),
                    payload.get("distance_in"),
                    payload.get("v_batt_load"),
                    payload.get("rpm_tgt"),
                    payload.get("rpm_at_fire"),
                    payload.get("hit"),
                    payload.get("pose_x"),
                    payload.get("pose_y"),
                    payload.get("heading_to_tag"),
                ]
            )
