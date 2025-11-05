"""Shot evaluation logic coordinating RPM model updates and planning."""
from __future__ import annotations

import csv
import math
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

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
    sent_ts_ms: int
    expires_ts_ms: int
    loiter: bool


@dataclass
class HitSample:
    """A successful shot captured for model fitting."""

    distance_in: float
    voltage: float
    rpm_at_fire: float
    rpm_base: float


class ShotEvaluator:
    """Handle shot planning, residual learning, and persistence."""

    def __init__(self, config: ClientConfig, model: RpmModel) -> None:
        self.config = config
        self.model = model
        self.model.ensure_grid(config.distance_bins, config.voltage_bins)

        self.session_id: Optional[str] = None
        self._shot_counter = 0
        self._pending_commands: Dict[str, PendingCommand] = {}
        self._pending_by_shot: Dict[str, PendingCommand] = {}
        self._processed_shots: set[str] = set()
        self._band_stats: Dict[Tuple[float, float], BandStats] = {}
        self._retry_bias: Dict[Tuple[float, float], int] = {}
        self._last_outcome: Dict[Tuple[float, float], str] = {}
        self._residual_history: Deque[float] = deque(maxlen=200)
        self._hit_residuals: Deque[float] = deque(maxlen=200)
        self._hit_samples: List[HitSample] = []
        self._shots_since_update = 0
        self._total_shots = 0
        self._last_update_shot_index = 0
        self._last_update_rmse = math.inf
        self._last_update_rmse_hits = math.inf
        self._range_step_pending: float = 0.0
        self._anchor_band: Optional[Tuple[float, float]] = None
        self._anchor_shots: int = 0
        self._anchor_complete: bool = False

        self._shots_log_path = self.config.paths.logs_dir / "shots.csv"
        if not self._shots_log_path.exists():
            self._write_shots_header()

    # -----------------------------------------------------------------
    def start_session(self, session_id: str) -> None:
        """Reset rolling state for a new robot session."""
        self.session_id = session_id
        self._shot_counter = 0
        self._pending_commands.clear()
        self._pending_by_shot.clear()
        self._processed_shots.clear()
        self._retry_bias.clear()
        self._last_outcome.clear()
        self._residual_history.clear()
        self._hit_residuals.clear()
        self._hit_samples.clear()
        self._shots_since_update = 0
        self._total_shots = 0
        self._last_update_shot_index = 0
        self._last_update_rmse = math.inf
        self._last_update_rmse_hits = math.inf
        self._range_step_pending = 0.0
        self._anchor_band = None
        self._anchor_shots = 0
        self._anchor_complete = False
        self._band_stats = {tuple(band): BandStats() for band in self.config.distance_bins}
        manifest_path = self.config.paths.logs_dir / f"session_{session_id}.yml"
        manifest = {
            "session_id": session_id,
            "start_ts": datetime.now(timezone.utc).isoformat(),
            "model_version_start": self.model.model_version,
            "bins_config": {
                "distance": [list(b) for b in self.config.distance_bins],
                "voltage": [list(b) for b in self.config.voltage_bins],
            },
            "policy_flags": {
                "name": self.config.policy.name,
                "send_abs_rpm": self.config.policy.send_abs_rpm,
                "bias_step_rpm": self.config.policy.bias_step_rpm,
                "bias_cap_rpm": self.config.policy.bias_cap_rpm,
                "start_with_anchor": self.config.policy.start_with_anchor,
                "anchor_shots_required": self.config.policy.anchor_shots_required,
            },
        }
        manifest_path.write_text(_to_yaml(manifest))

    # -----------------------------------------------------------------
    def plan_shot(self, request: Dict[str, object], now_ms: int, *, offline: bool) -> Dict[str, object]:
        """Produce a command in response to ``request_shot_plan``."""
        if self.session_id is None:
            raise RuntimeError("Session not initialized")
        if offline:
            return self.build_noop_cmd(now_ms, reason="offline")

        distance = request.get("distance_in")
        voltage = request.get("v_batt_load")
        if distance is None or voltage is None:
            return self.build_noop_cmd(now_ms, reason="missing_fields")

        distance_f = float(distance)
        voltage_f = float(voltage)
        rpm_base_req = request.get("rpm_base")
        rpm_base = _coerce_float(rpm_base_req, self.model.predict(distance_f, voltage_f))

        band = self._distance_band(distance_f)
        if self.config.policy.start_with_anchor and self._anchor_band is None:
            self._anchor_band = band
        anchor_active = (
            self.config.policy.start_with_anchor
            and not self._anchor_complete
            and self._anchor_band == band
        )

        self._expire_commands(now_ms)

        delta = self.model.delta(distance_f, voltage_f)
        retry_bias = self._retry_bias.get(band, 0)
        rpm_bias = int(round(delta)) + retry_bias
        rpm_bias = int(max(-self.config.policy.bias_cap_rpm, min(self.config.policy.bias_cap_rpm, rpm_bias)))

        loiter = self._last_outcome.get(band) == "miss"
        range_delta = 0.0
        if loiter:
            range_delta = 0.0
        else:
            if not anchor_active and self._range_step_pending:
                range_delta = self._range_step_pending
                self._range_step_pending = 0.0
            elif anchor_active:
                range_delta = 0.0

        cmd_id = str(uuid.uuid4())
        shot_id = str(request.get("shot_id") or f"{self.session_id}-{self._shot_counter + 1}")
        rpm_target = rpm_base + rpm_bias

        command = {
            "type": "cmd",
            "session_id": self.session_id,
            "cmd_id": cmd_id,
            "shot_id": shot_id,
            "valid_ms": self.config.cmd_valid_ms,
            "ts_ms": now_ms,
            "range_delta_in": int(round(range_delta)),
            "loiter": bool(loiter),
        }
        if self.config.policy.send_abs_rpm:
            command["rpm_target_abs"] = int(round(rpm_target))
        else:
            command["rpm_bias"] = int(rpm_bias)

        expires = now_ms + self.config.cmd_valid_ms
        if shot_id in self._pending_by_shot:
            prior = self._pending_by_shot.pop(shot_id)
            self._pending_commands.pop(prior.cmd_id, None)
        pending = PendingCommand(
            cmd_id=cmd_id,
            shot_id=shot_id,
            distance_in=distance_f,
            voltage=voltage_f,
            rpm_base=rpm_base,
            rpm_bias=int(rpm_bias),
            rpm_target=rpm_target,
            sent_ts_ms=now_ms,
            expires_ts_ms=expires,
            loiter=bool(loiter),
        )
        self._pending_commands[cmd_id] = pending
        self._pending_by_shot[shot_id] = pending
        self._shot_counter += 1
        return command

    def record_shot_fired(self, payload: Dict[str, object]) -> None:
        """Annotate pending command with the detected shot-fire token."""
        shot_id = str(payload.get("shot_id")) if payload.get("shot_id") else None
        if shot_id and shot_id in self._pending_by_shot:
            pending = self._pending_by_shot[shot_id]
            # Extend validity slightly to wait for result
            pending.expires_ts_ms += self.config.cmd_valid_ms

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

        distance = _coerce_float(payload.get("distance_in"), pending.distance_in if pending else 0.0)
        voltage = _coerce_float(payload.get("v_batt_load"), pending.voltage if pending else 12.0)
        rpm_at_fire = _coerce_float(payload.get("rpm_at_fire"), pending.rpm_target if pending else 0.0)
        rpm_base_payload = payload.get("rpm_base")
        rpm_base = _coerce_float(rpm_base_payload, pending.rpm_base if pending else self.model.predict(distance, voltage))
        rpm_tgt_cmd = _coerce_float(payload.get("rpm_tgt_cmd"), pending.rpm_target if pending else rpm_base)
        hit = _coerce_bool(payload.get("hit"))
        pose_x = payload.get("pose_x")
        pose_y = payload.get("pose_y")
        heading = payload.get("heading_to_tag")

        band = self._distance_band(distance)
        self._band_stats.setdefault(band, BandStats()).update(hit)
        self._last_outcome[band] = "hit" if hit else "miss"

        anchor_active = (
            self.config.policy.start_with_anchor
            and not self._anchor_complete
            and self._anchor_band == band
        )
        if anchor_active:
            self._anchor_shots += 1
            if self._anchor_shots >= self.config.policy.anchor_shots_required:
                self._anchor_complete = True
                if not self._range_step_pending:
                    self._range_step_pending = float(self.config.distance_step_in)

        if hit:
            self._retry_bias[band] = 0
            if not anchor_active or self._anchor_complete:
                if not self._range_step_pending:
                    self._range_step_pending = float(self.config.distance_step_in)
        else:
            self._retry_bias[band] = min(
                self._retry_bias.get(band, 0) + self.config.policy.bias_step_rpm,
                self.config.policy.bias_cap_rpm,
            )

        residual = rpm_at_fire - self.model.predict(distance, voltage)
        self._residual_history.append(residual)
        if hit:
            self._hit_residuals.append(residual)
            self._hit_samples.append(
                HitSample(
                    distance_in=distance,
                    voltage=voltage,
                    rpm_at_fire=rpm_at_fire,
                    rpm_base=rpm_base,
                )
            )
            if len(self._hit_samples) > 500:
                self._hit_samples.pop(0)

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
            rpm_tgt_cmd=rpm_tgt_cmd,
            rpm_at_fire=rpm_at_fire,
            hit=hit,
            pose_x=pose_x,
            pose_y=pose_y,
            heading=heading,
            notes=payload.get("notes"),
        )

        if self._ready_for_update():
            fit = self._fit_model_update()
            if fit:
                model_type, params, gamma, rmse = fit
                improvement_threshold = 0.0
                if math.isfinite(self._last_update_rmse_hits):
                    improvement_threshold = self._last_update_rmse_hits * (
                        self.config.update_rmse_improvement_pct / 100.0
                    )
                if not math.isfinite(self._last_update_rmse_hits) or (
                    self._last_update_rmse_hits - rmse >= improvement_threshold
                ):
                    self.model.model_type = model_type
                    self.model.params = params
                    self.model.gamma = gamma
                    self.model.bump_version(
                        note=f"fit@{datetime.now(timezone.utc).isoformat()} n={len(self._hit_samples)}"
                    )
                    self.model.save(self.config.paths.model_path)
                    self._shots_since_update = 0
                    self._last_update_shot_index = self._total_shots
                    self._last_update_rmse_hits = rmse
                    self._last_update_rmse = _rmse(self._residual_history)
                    return self.model.as_update_payload()
        return None

    # -----------------------------------------------------------------
    def build_noop_cmd(self, now_ms: int, reason: str) -> Dict[str, object]:
        cmd = {
            "type": "cmd",
            "session_id": self.session_id,
            "cmd_id": str(uuid.uuid4()),
            "shot_id": f"noop-{now_ms}",
            "valid_ms": self.config.cmd_valid_ms,
            "ts_ms": now_ms,
            "range_delta_in": 0,
            "loiter": False,
            "reason": reason,
        }
        if self.config.policy.send_abs_rpm:
            cmd["rpm_target_abs"] = 0
        else:
            cmd["rpm_bias"] = 0
        return cmd

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
                    rpm_tgt_cmd,
                    rpm_at_fire,
                    int(hit),
                    pose_x,
                    pose_y,
                    heading,
                    notes,
                ]
            )

    def _ready_for_update(self) -> bool:
        if self._shots_since_update < self.config.update_cadence_shots:
            return False
        if (self._total_shots - self._last_update_shot_index) < self.config.update_rate_limit:
            return False
        if len(self._hit_samples) < 5:
            return False
        return True

    def _fit_model_update(self) -> Optional[Tuple[str, Dict[str, float], float, float]]:
        """Return the best-fit model tuple ``(type, params, gamma, rmse)``."""

        samples = list(self._hit_samples)
        if len(samples) < 5:
            return None

        gamma_candidates = _gamma_candidates(self.model.gamma)
        best: Optional[Tuple[str, Dict[str, float], float, float]] = None
        best_rmse = math.inf

        for gamma in gamma_candidates:
            scaled = [_scaled_sample(sample, gamma, self.model.v_nom) for sample in samples]
            quadratic = _evaluate_quadratic(scaled)
            power = _evaluate_power(scaled)
            for candidate in filter(None, [quadratic, power]):
                model_type, params, rmse = candidate
                if rmse < best_rmse:
                    if model_type == "quadratic" and not _quadratic_monotonic(params, scaled):
                        continue
                    if model_type == "power" and not _power_monotonic(params):
                        continue
                    best = (model_type, params, gamma, rmse)
                    best_rmse = rmse
        return best


def _rmse(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.inf
    return math.sqrt(sum(v * v for v in values) / len(values))


def _gamma_candidates(current: float) -> List[float]:
    candidates = {max(0.5, round(current, 3))}
    candidates.add(max(0.5, round(current + 0.1, 3)))
    candidates.add(max(0.5, round(current - 0.1, 3)))
    return sorted(candidates)


def _scaled_sample(sample: HitSample, gamma: float, v_nom: float) -> Tuple[float, float]:
    voltage = max(sample.voltage, 1e-3)
    scale = (voltage / v_nom) ** gamma
    return sample.distance_in, sample.rpm_at_fire * scale


def _evaluate_quadratic(samples: Sequence[Tuple[float, float]]) -> Optional[Tuple[str, Dict[str, float], float]]:
    if len(samples) < 3:
        return None

    def fit(data: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        return _solve_quadratic(data)

    def predict(params: Tuple[float, float, float], distance: float) -> float:
        b0, b1, b2 = params
        return b0 + b1 * distance + b2 * (distance ** 2)

    rmse = _kfold_rmse(samples, fit, predict)
    if not math.isfinite(rmse):
        return None
    params = _solve_quadratic(samples)
    if params is None:
        return None
    b0, b1, b2 = params
    return "quadratic", {"b0": b0, "b1": b1, "b2": b2}, rmse


def _evaluate_power(samples: Sequence[Tuple[float, float]]) -> Optional[Tuple[str, Dict[str, float], float]]:
    if len(samples) < 2:
        return None
    p_grid = [round(p, 2) for p in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]]
    best: Optional[Tuple[float, Tuple[float, float], float]] = None  # (p, params, rmse)

    for p in p_grid:
        def fit(data: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
            return _solve_power(data, p)

        def predict(params: Tuple[float, float], distance: float) -> float:
            alpha, beta = params
            return alpha * (distance ** p) + beta

        rmse = _kfold_rmse(samples, fit, predict)
        if not math.isfinite(rmse):
            continue
        params = _solve_power(samples, p)
        if params is None:
            continue
        if best is None or rmse < best[2]:
            best = (p, params, rmse)

    if best is None:
        return None
    p, params, rmse = best
    alpha, beta = params
    return "power", {"alpha": alpha, "beta": beta, "p": p}, rmse


def _kfold_rmse(
    samples: Sequence[Tuple[float, float]],
    fit_fn,
    predict_fn,
    folds: int = 3,
) -> float:
    n = len(samples)
    if n < folds:
        folds = n
    if folds < 2:
        return math.inf

    total_sq = 0.0
    total_n = 0
    for fold in range(folds):
        train = [samples[i] for i in range(n) if i % folds != fold]
        test = [samples[i] for i in range(n) if i % folds == fold]
        if not train or not test:
            continue
        params = fit_fn(train)
        if params is None:
            return math.inf
        for distance, target in test:
            pred = predict_fn(params, distance)
            total_sq += (pred - target) ** 2
            total_n += 1
    if total_n == 0:
        return math.inf
    return math.sqrt(total_sq / total_n)


def _solve_quadratic(data: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    if len(data) < 3:
        return None
    s0 = float(len(data))
    s1 = sum(d for d, _ in data)
    s2 = sum((d ** 2) for d, _ in data)
    s3 = sum((d ** 3) for d, _ in data)
    s4 = sum((d ** 4) for d, _ in data)
    sy = sum(y for _, y in data)
    s1y = sum(d * y for d, y in data)
    s2y = sum((d ** 2) * y for d, y in data)

    matrix = [
        [s0, s1, s2],
        [s1, s2, s3],
        [s2, s3, s4],
    ]
    rhs = [sy, s1y, s2y]
    return _solve_3x3(matrix, rhs)


def _solve_power(data: Sequence[Tuple[float, float]], p: float) -> Optional[Tuple[float, float]]:
    if len(data) < 2:
        return None
    xs = [d ** p for d, _ in data]
    ys = [y for _, y in data]
    sum_x = sum(xs)
    sum_x2 = sum(x * x for x in xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    n = len(data)
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-6:
        return None
    alpha = (n * sum_xy - sum_x * sum_y) / denom
    beta = (sum_y - alpha * sum_x) / n
    return alpha, beta


def _solve_3x3(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    a = [list(row) for row in matrix]
    b = list(rhs)
    n = 3
    for i in range(n):
        pivot = i + max(range(n - i), key=lambda k: abs(a[i + k][i]))
        if abs(a[pivot][i]) < 1e-9:
            return None
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
        pivot_val = a[i][i]
        for j in range(i, n):
            a[i][j] /= pivot_val
        b[i] /= pivot_val
        for k in range(n):
            if k == i:
                continue
            factor = a[k][i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]
    return b[0], b[1], b[2]


def _quadratic_monotonic(params: Dict[str, float], samples: Sequence[Tuple[float, float]]) -> bool:
    if not samples:
        return True
    b1 = float(params.get("b1", 0.0))
    b2 = float(params.get("b2", 0.0))
    distances = [d for d, _ in samples]
    lo = min(distances)
    hi = max(distances)
    for point in (lo, hi, (lo + hi) / 2.0):
        derivative = b1 + 2.0 * b2 * point
        if derivative < -1e-6:
            return False
    return True


def _power_monotonic(params: Dict[str, float]) -> bool:
    alpha = float(params.get("alpha", 0.0))
    p = float(params.get("p", 1.0))
    return alpha >= 0.0 and p > 0.0


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


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "hit"}:
            return True
        if lowered in {"0", "false", "no", "miss"}:
            return False
    return False
