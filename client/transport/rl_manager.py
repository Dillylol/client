"""RL bridge that listens to the main JULES stream instead of its own WS port."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from client.client.config import ClientConfig
    from client.client.rpm_model import RpmModel
except ModuleNotFoundError:  # pragma: no cover - repo root execution fallback
    from client.config import ClientConfig
    from client.rpm_model import RpmModel

logger = logging.getLogger(__name__)


class RLManager:
    """
    Minimal RL manager that:

    - Knows which bridge URL we are connected to (for display).
    - Listens to frames coming from the main link (ClientAPI hands them in).
    - Tracks shot events and requests reward/feedback from the GUI.
    - Updates the RPM model incrementally when rewards are submitted.
    hi
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        config_path = Path(__.pop("config_path", "config.yml"))

        self.on_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self.send_cmd: Optional[Callable[[Dict[str, Any]], None]] = None

        self._state: str = "stopped"
        self._bridge_url: Optional[str] = None
        self._last_shot_ts: Optional[float] = None
        self._pending_shots: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        self._config_path = config_path
        self._config: Optional[ClientConfig] = None
        self.model: Optional[RpmModel] = None
        self._alpha: float = 0.15
        self._distance_bins: Optional[Any] = None
        self._voltage_bins: Optional[Any] = None
        self._shots_log_path: Optional[Path] = None

        self._load_config_and_model()

        self._emit_status(
            state="stopped",
            details={
                "reason": "not attached",
                "endpoint": "bridge://(not configured)",
            },
        )

    # ------------------------------------------------------------------ public API

    def set_bridge_url(self, url: str) -> None:
        """Called by ClientAPI.configure() so RL tab shows the real endpoint."""
        self._bridge_url = url
        self._emit_status(
            state=self._state,
            details={
                "endpoint": url,
            },
        )

    def handle_status(self, status: Dict[str, Any]) -> None:
        """
        Called from ClientAPI when the main link status changes.
        We mirror that into the RL tab.
        """
        state = str(status.get("state", "")).lower()

        if state in {"connected"}:
            self._state = "online"
        elif state in {"connecting", "listening", "ready"}:
            self._state = "starting"
        elif state in {"reconnecting"}:
            self._state = "reconnecting"
        elif state in {"error", "stopped", "failed"}:
            self._state = "error"

        details = dict(status)
        if self._bridge_url:
            details.setdefault("endpoint", self._bridge_url)

        self._emit_status(self._state, details)

    def handle_frame(self, frame: Dict[str, Any]) -> None:
        """
        Called from ClientAPI for every frame on the main JULES bus.
        """
        ftype = str(frame.get("type", "")).lower()

        if ftype == "shot":
            self._handle_shot_frame(frame)
        elif ftype in {"snapshot", "heartbeat"}:
            event = {
                "direction": "inbound",
                "type": ftype,
                "payload": dict(frame),
            }
            self._emit_event(event)
        else:
            logger.debug("RLManager ignoring frame type: %s", ftype)

    def record_reward(self, shot_id: str, hit: bool) -> None:
        """
        Called by the GUI once the user answers “Did it score?”
        """
        if not shot_id:
            logger.warning("record_reward called without a shot_id")
            return
        normalized = str(shot_id)
        shot = self._pending_shots.pop(normalized, None)
        if not shot:
            logger.warning("Shot %s is no longer pending; ignoring reward", normalized)
            return

        shot = dict(shot)
        shot["shot_id"] = normalized
        shot["hit"] = bool(hit)
        reward = 1.0 if hit else 0.0
        shot["reward"] = reward
        features = self._apply_reward_to_model(shot, reward)
        self._append_reward_log(shot, reward, features)

        self._emit_event(
            {
                "direction": "local",
                "type": "reward_recorded",
                "shot_id": normalized,
                "reward": reward,
                "hit": bool(hit),
                "shot": shot,
                "features": features,
            }
        )

    # ------------------------------------------------------------------ helpers

    def _handle_shot_frame(self, frame: Dict[str, Any]) -> None:
        shot = dict(frame)
        shot_id = self._extract_shot_id(shot)
        if not shot_id:
            fallback = shot.get("ts_ms") or shot.get("ts") or int(time.time() * 1000)
            shot_id = str(fallback)
            shot["shot_id"] = shot_id
        if shot_id in self._pending_shots:
            self._pending_shots.pop(shot_id, None)
        self._pending_shots[shot_id] = shot
        while len(self._pending_shots) > 50:
            self._pending_shots.popitem(last=False)
        self._last_shot_ts = time.time()
        self._emit_event(
            {
                "direction": "inbound",
                "type": "shot_pending",
                "shot_id": shot_id,
                "payload": dict(shot),
            }
        )

    def _load_config_and_model(self) -> None:
        try:
            self._config = ClientConfig.load(self._config_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLManager could not load config %s: %s", self._config_path, exc)
            self._config = None
            return

        self._alpha = float(self._config.alpha_ewma)
        self._distance_bins = list(self._config.distance_bins)
        self._voltage_bins = list(self._config.voltage_bins)
        self._shots_log_path = (self._config.paths.logs_dir / "rl_rewards.csv").resolve()
        self._ensure_reward_log_header()

        try:
            self.model = RpmModel.load(self._config.paths.model_path)
        except FileNotFoundError:
            logger.warning("RPM model not found at %s", self._config.paths.model_path)
            self.model = None
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load RPM model")
            self.model = None

    def _ensure_reward_log_header(self) -> None:
        path = self._shots_log_path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "ts_ms",
                    "shot_id",
                    "session_id",
                    "bot_id",
                    "distance_in",
                    "v_batt_load",
                    "rpm_at_fire",
                    "rpm_target",
                    "rpm_manual",
                    "reward",
                    "hit",
                    "raw",
                ]
            )

    def _apply_reward_to_model(self, shot: Dict[str, Any], reward: float) -> Optional[Dict[str, float]]:
        if not self.model or not self._distance_bins or not self._voltage_bins:
            return RpmModel.extract_shot_features(shot)  # best effort for logging
        try:
            return self.model.update_from_shot_context(
                shot,
                reward=reward,
                alpha=self._alpha,
                distance_bins=self._distance_bins,
                voltage_bins=self._voltage_bins,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update RPM model from shot %s", shot.get("shot_id"))
            return RpmModel.extract_shot_features(shot)

    def _append_reward_log(
        self,
        shot: Dict[str, Any],
        reward: float,
        features: Optional[Dict[str, float]],
    ) -> None:
        path = self._shots_log_path
        if not path:
            return
        ts_ms = self._coerce_int(shot.get("ts_ms") or shot.get("ts"))
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        ctx = shot.get("context") if isinstance(shot.get("context"), dict) else {}
        row = [
            ts_ms,
            shot.get("shot_id"),
            shot.get("session_id") or ctx.get("session_id"),
            shot.get("bot_id") or ctx.get("bot_id"),
            (features or {}).get("distance_in"),
            (features or {}).get("v_batt_load"),
            (features or {}).get("rpm_at_fire") or self._coerce_float(shot.get("rpm_at_fire")),
            self._coerce_float(ctx.get("target_rpm")),
            self._coerce_float(ctx.get("manual_target_rpm")),
            reward,
            1 if shot.get("hit") else 0,
            json.dumps(shot, ensure_ascii=False, default=str),
        ]
        try:
            with path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(row)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to append RL reward log")

    def _extract_shot_id(self, shot: Dict[str, Any]) -> Optional[str]:
        value = shot.get("shot_id")
        if value is not None:
            return str(value)
        payload = shot.get("payload")
        if isinstance(payload, dict) and payload.get("shot_id") is not None:
            return str(payload.get("shot_id"))
        return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _emit(self, payload: Dict[str, Any]) -> None:
        cb = self.on_event
        if not cb:
            return
        try:
            cb(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Error emitting RL event")

    def _emit_event(self, event: Dict[str, Any]) -> None:
        wrapped = {
            "kind": "event",
            "timestamp": time.time(),
            "event": event,
        }
        self._emit(wrapped)

    def _emit_status(self, state: str, details: Dict[str, Any]) -> None:
        wrapped = {
            "kind": "status",
            "timestamp": time.time(),
            "state": state,
            "details": details,
        }
        self._emit(wrapped)
