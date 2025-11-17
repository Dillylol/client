"""Client-side artifact detection pipeline fed by streamed video frames."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - optional heavy dependency
    import cv2  # type: ignore
    import numpy as np
except Exception:  # noqa: BLE001 - we fall back gracefully if OpenCV is missing
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    bbox: List[float]
    center: List[float]
    area: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [float(v) for v in self.bbox],
            "center": [float(v) for v in self.center],
            "area": float(self.area),
        }


class ArtifactDetector:
    """Run lightweight color-based heuristics to find DECODE artifacts."""

    def __init__(
        self,
        *,
        min_area: float = 600.0,
        publish: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.min_area = float(min_area)
        self._publish = publish
        self._warned_missing_cv = False

    # ------------------------------------------------------------------
    def handle_frame(self, frame: Dict[str, Any]) -> None:
        """Process a `video_frame` packet."""
        if cv2 is None or np is None:
            self._warn_once("OpenCV (cv2) or NumPy is not available; artifact detection disabled.")
            return

        jpeg_bytes = self._extract_bytes(frame)
        if not jpeg_bytes:
            return

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.debug("artifact_detector: failed to decode JPEG payload.")
            return

        detections = [det.as_dict() for det in self._detect(image)]
        event = {
            "type": "artifact_detections",
            "ts_ms": self._extract_ts(frame),
            "detections": detections,
        }
        self._emit(event)

    # ------------------------------------------------------------------
    def _detect(self, image: "np.ndarray") -> List[Detection]:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Purple-ish range for DECODE artifacts (two overlapping ranges improve robustness).
        lower_primary = (120, 60, 40)
        upper_primary = (160, 255, 255)
        mask = cv2.inRange(hsv, lower_primary, upper_primary)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    bbox=[float(x), float(y), float(w), float(h)],
                    center=[float(x + w / 2), float(y + h / 2)],
                    area=area,
                )
            )
        return detections

    # ------------------------------------------------------------------
    def _extract_ts(self, frame: Dict[str, Any]) -> int:
        ts = frame.get("ts_ms") or frame.get("ts") or frame.get("timestamp")
        if isinstance(ts, (int, float)):
            if ts > 1_000_000_000_000:
                return int(ts)
            return int(ts * 1000)
        return 0

    def _extract_bytes(self, frame: Dict[str, Any]) -> Optional[bytes]:
        payloads = [
            frame.get("jpeg"),
            frame.get("jpeg_b64"),
            frame.get("image"),
            frame.get("payload"),
        ]
        for candidate in payloads:
            buf = self._coerce_bytes(candidate)
            if buf:
                return buf
        return None

    def _coerce_bytes(self, value: Any) -> Optional[bytes]:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return base64.b64decode(value, validate=False)
            except Exception:  # noqa: BLE001
                logger.debug("artifact_detector: failed to base64 decode payload preview.")
                return None
        if isinstance(value, list):
            try:
                return bytes(int(v) & 0xFF for v in value)
            except Exception:  # noqa: BLE001
                return None
        if isinstance(value, dict):
            jpeg = value.get("jpeg") or value.get("data")
            return self._coerce_bytes(jpeg)
        return None

    def _emit(self, payload: Dict[str, Any]) -> None:
        if not self._publish:
            return
        try:
            self._publish(payload)
        except Exception:  # noqa: BLE001
            logger.exception("artifact_detector: failed to emit detection payload.")

    def _warn_once(self, message: str) -> None:
        if self._warned_missing_cv:
            return
        self._warned_missing_cv = True
        logger.warning(message)


__all__ = ["ArtifactDetector"]

