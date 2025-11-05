"""RPM model utilities for the Pedro evaluation client.

This module implements a persistent RPM model that follows the schema
``rpm_model/v1``.  Two base model families are supported:

* ``quadratic`` – ``f(d) = b0 + b1 * d + b2 * d^2``
* ``power`` – ``f(d) = alpha * d^p + beta``

Both families are scaled by the battery voltage according to the
``(v_nom / V)^gamma`` term.  A compact correction table (``delta_bins``)
allows the client to learn residual biases across the distance/voltage grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import os
import tempfile

SCHEMA = "rpm_model/v1"


def _normalize_range(pair: Sequence[float]) -> Tuple[float, float]:
    if len(pair) != 2:
        raise ValueError(f"Range must have exactly 2 elements: {pair}")
    lo, hi = float(pair[0]), float(pair[1])
    if lo <= hi:
        return lo, hi
    # Some configs list voltage bands in descending order; normalize.
    return hi, lo


@dataclass
class DeltaBin:
    """Representation of a residual correction bin."""

    distance: Tuple[float, float]
    voltage: Tuple[float, float]
    delta: float = 0.0

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "DeltaBin":
        try:
            d_range = payload.get("d") or payload.get("distance") or payload.get("distance_in")
            v_range = payload.get("v") or payload.get("voltage") or payload.get("voltage_v")
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid delta bin payload: {payload}") from exc
        if d_range is None or v_range is None:
            raise ValueError(f"Delta bin missing distance/voltage ranges: {payload}")
        distance = _normalize_range(d_range)
        voltage = _normalize_range(v_range)
        delta = float(payload.get("delta", payload.get("value", 0.0)))
        return cls(distance=distance, voltage=voltage, delta=delta)

    def to_payload(self) -> Dict[str, object]:
        return {"d": list(self.distance), "v": list(self.voltage), "delta": self.delta}

    def contains(self, distance_in: float, voltage: float) -> bool:
        d_lo, d_hi = self.distance
        v_lo, v_hi = self.voltage
        return d_lo <= distance_in < d_hi and v_lo <= voltage < v_hi

    def center(self) -> Tuple[float, float]:
        d_lo, d_hi = self.distance
        v_lo, v_hi = self.voltage
        return (d_lo + d_hi) / 2.0, (v_lo + v_hi) / 2.0


@dataclass
class RpmModel:
    """Persistent RPM model with residual correction bins."""

    model_type: str
    params: Dict[str, float]
    v_nom: float
    gamma: float
    delta_bins: List[DeltaBin] = field(default_factory=list)
    model_version: int = 1
    note: Optional[str] = None
    schema: str = SCHEMA
    path: Optional[Path] = None

    # --- Persistence -------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RpmModel":
        data = json.loads(path.read_text())
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"Unsupported RPM model schema: {schema}")
        bins_raw = data.get("delta_bins", [])
        bins = [DeltaBin.from_payload(entry) for entry in bins_raw]
        model = cls(
            model_type=str(data.get("model", "quadratic")),
            params={k: float(v) for k, v in (data.get("params") or {}).items()},
            v_nom=float(data.get("v_nom", 12.0)),
            gamma=float(data.get("gamma", 1.0)),
            delta_bins=bins,
            model_version=int(data.get("model_version", 1)),
            note=data.get("note"),
            path=path,
        )
        return model

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self.path
        if target is None:
            raise ValueError("No path provided to save RPM model")
        payload = {
            "schema": self.schema,
            "model_version": self.model_version,
            "model": self.model_type,
            "params": self.params,
            "v_nom": self.v_nom,
            "gamma": self.gamma,
            "delta_bins": [bin_.to_payload() for bin_ in self.delta_bins],
        }
        if self.note is not None:
            payload["note"] = self.note
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = target.parent
        with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(target)

    # --- Base curve --------------------------------------------------
    def base_rpm(self, distance_in: float) -> float:
        if self.model_type == "quadratic":
            b0 = float(self.params.get("b0", 0.0))
            b1 = float(self.params.get("b1", 0.0))
            b2 = float(self.params.get("b2", 0.0))
            return b0 + b1 * distance_in + b2 * (distance_in ** 2)
        if self.model_type == "power":
            alpha = float(self.params.get("alpha", 0.0))
            beta = float(self.params.get("beta", 0.0))
            power = float(self.params.get("p", 1.0))
            return alpha * (distance_in ** power) + beta
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, distance_in: float, v_batt_load: float) -> float:
        base = self.base_rpm(distance_in)
        voltage = max(float(v_batt_load), 1e-3)
        scale = (self.v_nom / voltage) ** self.gamma if voltage > 0 else 1.0
        return base * scale

    # --- Residual corrections ---------------------------------------
    def delta(self, distance_in: float, v_batt_load: float) -> float:
        bin_ = self._select_bin(distance_in, v_batt_load)
        return bin_.delta if bin_ else 0.0

    def ensure_grid(self, distance_bins: Iterable[Sequence[float]], voltage_bins: Iterable[Sequence[float]]) -> None:
        existing = {(bin_.distance, bin_.voltage) for bin_ in self.delta_bins}
        for d_range in distance_bins:
            d_norm = _normalize_range(d_range)
            for v_range in voltage_bins:
                v_norm = _normalize_range(v_range)
                if (d_norm, v_norm) not in existing:
                    self.delta_bins.append(DeltaBin(distance=d_norm, voltage=v_norm, delta=0.0))
                    existing.add((d_norm, v_norm))

    def update_residual(
        self,
        distance_in: float,
        v_batt_load: float,
        rpm_at_fire: float,
        alpha: float,
        distance_bins: Iterable[Sequence[float]],
        voltage_bins: Iterable[Sequence[float]],
    ) -> float:
        base = self.predict(distance_in, v_batt_load)
        residual = rpm_at_fire - base
        bin_ = self._select_bin(distance_in, v_batt_load)
        if bin_ is None:
            # create bin aligned with configured ranges
            d_range = self._range_for_value(distance_in, distance_bins)
            v_range = self._range_for_value(v_batt_load, voltage_bins)
            bin_ = DeltaBin(distance=d_range, voltage=v_range, delta=0.0)
            self.delta_bins.append(bin_)
        prev = bin_.delta
        updated = prev + alpha * (residual - prev)
        if math.isfinite(updated):
            bin_.delta = updated
        return bin_.delta

    def _range_for_value(self, value: float, ranges: Iterable[Sequence[float]]) -> Tuple[float, float]:
        best: Optional[Tuple[float, float]] = None
        best_distance = math.inf
        for candidate in ranges:
            c_norm = _normalize_range(candidate)
            lo, hi = c_norm
            if lo <= value < hi:
                return c_norm
            center = (lo + hi) / 2.0
            distance = abs(center - value)
            if distance < best_distance:
                best = c_norm
                best_distance = distance
        if best is None:
            raise ValueError(f"No range configured for value {value}")
        return best

    def _select_bin(self, distance_in: float, v_batt_load: float) -> Optional[DeltaBin]:
        containing = [bin_ for bin_ in self.delta_bins if bin_.contains(distance_in, v_batt_load)]
        if containing:
            # Prefer the bin with the smallest area (most specific)
            return min(containing, key=lambda b: (b.distance[1] - b.distance[0]) * (b.voltage[1] - b.voltage[0]))
        if not self.delta_bins:
            return None
        # Fallback to nearest center distance/voltage (L2)
        best = min(
            self.delta_bins,
            key=lambda b: math.hypot(b.center()[0] - distance_in, b.center()[1] - v_batt_load),
        )
        return best

    # --- Model versioning -------------------------------------------
    def bump_version(self, note: Optional[str] = None) -> None:
        self.model_version += 1
        if note is not None:
            self.note = note

    def as_update_payload(self) -> Dict[str, object]:
        payload = {
            "schema": self.schema,
            "model_version": self.model_version,
            "model": self.model_type,
            "params": self.params,
            "v_nom": self.v_nom,
            "gamma": self.gamma,
            "delta_bins": [bin_.to_payload() for bin_ in self.delta_bins],
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload
