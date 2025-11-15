
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

@dataclass
class Metrics:
    # Common top-level fields we expect from the robot
    t: Optional[float] = None
    ts: Optional[float] = None
    jsonData: Optional[str] = None

    # Common telemetry (optional; will be filled if present)
    battery_V: Optional[float] = None
    battery_v: Optional[float] = None  # allow either key convention
    heading_deg: Optional[float] = None
    vel_ips: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    heading: Optional[float] = None

    # IMU-ish (optional)
    pitch: Optional[float] = None
    roll: Optional[float] = None
    yawRate: Optional[float] = None
    pitchRate: Optional[float] = None
    rollRate: Optional[float] = None

    # Catch-all for anything else
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def set_dynamic(self, key: str, value: Any) -> None:
        setattr(self, key, value)
        self._extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return a merged dict including dynamic fields."""
        d = asdict(self)
        extra = d.pop("_extra", {}) or {}
        d.update(extra)
        return d
