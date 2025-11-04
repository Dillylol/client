"""RPM model utilities for Pedro-shot evaluation client."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping

import json
import math


@dataclass
class RpmModel:
    """Linear-in-distance RPM model with optional correction bins.

    The model follows the form::

        rpm(distance, voltage) = (m * feet + b) * (voltage / v_nom) ** gamma + delta(distance)

    where ``delta`` is a learned bias looked up in a small table of bins.
    """

    m_rpm_per_ft: float
    b_rpm_offset: float
    v_nom: float
    gamma: float
    delta_bins: MutableMapping[float, float] = field(default_factory=dict)
    bin_width_ft: float = 0.5
    model_name: str = "linear_ft"
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "RpmModel":
        """Load a model from ``path``."""
        data = json.loads(path.read_text())
        bins = cls._parse_bins(data.get("delta_bins", []))
        model = cls(
            m_rpm_per_ft=data["m_rpm_per_ft"],
            b_rpm_offset=data["b_rpm_offset"],
            v_nom=data.get("v_nom", 12.0),
            gamma=data.get("gamma", 1.0),
            delta_bins=bins,
            bin_width_ft=data.get("bin_width_ft", 0.5),
            model_name=data.get("model", "linear_ft"),
            path=path,
        )
        return model

    @staticmethod
    def _parse_bins(raw: Iterable) -> MutableMapping[float, float]:
        bins: Dict[float, float] = {}
        for entry in raw:
            if isinstance(entry, dict):
                distance_ft = float(entry.get("distance_ft"))
                value = float(entry.get("value", 0.0))
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                distance_ft = float(entry[0])
                value = float(entry[1])
            else:
                continue
            bins[distance_ft] = value
        return bins

    def save(self, path: Path | None = None) -> None:
        """Persist the model to disk."""
        target = path or self.path
        if target is None:
            raise ValueError("No path specified for saving RPM model")
        payload = {
            "model": self.model_name,
            "m_rpm_per_ft": self.m_rpm_per_ft,
            "b_rpm_offset": self.b_rpm_offset,
            "v_nom": self.v_nom,
            "gamma": self.gamma,
            "bin_width_ft": self.bin_width_ft,
            "delta_bins": self._dump_bins(),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _dump_bins(self) -> List[Dict[str, float]]:
        return [
            {"distance_ft": distance_ft, "value": value}
            for distance_ft, value in sorted(self.delta_bins.items())
        ]

    def predict(self, distance_in: float, v_batt_load: float) -> float:
        """Return the base RPM (without bias table) for ``distance_in`` and ``v_batt_load``."""
        feet = distance_in / 12.0
        voltage_ratio = (v_batt_load / self.v_nom) if self.v_nom else 1.0
        voltage_ratio = max(voltage_ratio, 0.0)
        return (self.m_rpm_per_ft * feet + self.b_rpm_offset) * (voltage_ratio ** self.gamma)

    def delta(self, distance_in: float, _v_batt_load: float | None = None) -> float:
        """Return the learned correction for ``distance_in``."""
        key = self._bin_key(distance_in)
        return self.delta_bins.get(key, 0.0)

    def target(self, distance_in: float, v_batt_load: float) -> float:
        """Return the total RPM target including delta."""
        return self.predict(distance_in, v_batt_load) + self.delta(distance_in, v_batt_load)

    def update_residual(self, distance_in: float, v_batt_load: float, rpm_at_fire: float, alpha: float = 0.15) -> float:
        """Update the correction bin using an exponentially weighted residual.

        Returns the new delta value for the affected bin.
        """
        key = self._bin_key(distance_in)
        base = self.predict(distance_in, v_batt_load)
        desired_delta = rpm_at_fire - base
        prev = self.delta_bins.get(key, 0.0)
        updated = prev + alpha * (desired_delta - prev)
        if not math.isfinite(updated):
            return prev
        self.delta_bins[key] = updated
        return updated

    def as_update_payload(self) -> Dict[str, object]:
        """Return a dict suitable for sending as ``rpm_model_update``."""
        return {
            "type": "rpm_model_update",
            "model": self.model_name,
            "m_rpm_per_ft": self.m_rpm_per_ft,
            "b_rpm_offset": self.b_rpm_offset,
            "v_nom": self.v_nom,
            "gamma": self.gamma,
            "delta_bins": self._dump_bins(),
        }

    def _bin_key(self, distance_in: float) -> float:
        feet = distance_in / 12.0
        bin_index = round(feet / self.bin_width_ft)
        return round(bin_index * self.bin_width_ft, 3)

    def bin_identifier(self, distance_in: float) -> float:
        """Expose the bin key for external book-keeping."""
        return self._bin_key(distance_in)
