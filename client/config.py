"""Configuration loader for the Pedro shot evaluation client."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class ClientPaths:
    """Convenience container for file system paths used by the client."""

    artifacts_dir: Path
    logs_dir: Path
    model_path: Path


@dataclass(frozen=True)
class ClientConfig:
    """Strongly-typed configuration for the evaluation workflow."""

    schema: str
    policy: str
    cmd_valid_ms: int
    distance_step_in: float
    alpha_ewma: float
    update_cadence_shots: int
    update_rate_limit: int
    update_rmse_improvement_pct: float
    distance_bins: List[Tuple[float, float]]
    voltage_bins: List[Tuple[float, float]]
    paths: ClientPaths
    require_token: bool
    token: str

    @classmethod
    def load(cls, path: Path) -> "ClientConfig":
        data = yaml.safe_load(path.read_text())
        if data.get("schema") != "config/v1":
            raise ValueError(f"Unsupported config schema: {data.get('schema')}")
        policy = data.get("policy", "simple_bandit_v1")
        security = data.get("security", {})
        paths_raw = data.get("paths", {})
        artifacts_dir = Path(paths_raw.get("artifacts_dir", "artifacts")).expanduser()
        logs_dir = Path(paths_raw.get("logs_dir", "logs")).expanduser()
        model_path_raw = paths_raw.get("model_path", "rpm_model.json")
        model_path = Path(model_path_raw).expanduser()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        if not model_path.is_absolute():
            model_path = (artifacts_dir / model_path).resolve()
        distance_bins = [tuple(_normalize_range(pair)) for pair in data["bins"]["distance"]]
        voltage_bins = [tuple(_normalize_range(pair)) for pair in data["bins"]["voltage"]]
        return cls(
            schema=data["schema"],
            policy=policy,
            cmd_valid_ms=int(data.get("cmd_valid_ms", 800)),
            distance_step_in=float(data.get("distance_step_in", 4.0)),
            alpha_ewma=float(data.get("alpha_ewma", 0.15)),
            update_cadence_shots=int(data.get("update_cadence_shots", 10)),
            update_rate_limit=int(_parse_rate_limit(data.get("update_rate_limit", 10))),
            update_rmse_improvement_pct=float(data.get("update_rmse_improvement_pct", 5.0)),
            distance_bins=distance_bins,
            voltage_bins=voltage_bins,
            paths=ClientPaths(artifacts_dir=artifacts_dir, logs_dir=logs_dir, model_path=model_path),
            require_token=bool(security.get("require_token", False)),
            token=str(security.get("token", "")),
        )


def _normalize_range(pair: Sequence[float]) -> Tuple[float, float]:
    if len(pair) != 2:
        raise ValueError(f"Range must have exactly 2 elements: {pair}")
    lo, hi = float(pair[0]), float(pair[1])
    if lo <= hi:
        return lo, hi
    return hi, lo


def _parse_rate_limit(raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.endswith("_per_10"):
        prefix = raw.split("_per_10")[0]
        try:
            return int(prefix) * 10
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid rate limit spec: {raw}") from exc
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid rate limit spec: {raw}") from exc
    raise ValueError(f"Unsupported rate limit spec: {raw}")
