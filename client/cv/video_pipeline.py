"""Decode video frames and run artifact detection."""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from .artifact_detector import ArtifactDetector, ArtifactDetection

logger = logging.getLogger(__name__)


@dataclass
class VideoFramePayload:
    ts_ms: int
    image: Image.Image
    detections: List[ArtifactDetection]


class VideoStreamProcessor:
    def __init__(self, detector: Optional[ArtifactDetector] = None) -> None:
        self.detector = detector or ArtifactDetector()
        self.on_frame: Optional[Callable[[VideoFramePayload], None]] = None
        self.on_artifact_event: Optional[Callable[[Dict[str, Any]], None]] = None

    def handle_packet(self, packet: Dict[str, Any]) -> None:
        try:
            frame = self._decode_packet(packet)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to decode video frame")
            return
        if frame is None:
            return
        try:
            detections = self.detector.process(frame)
        except Exception:  # noqa: BLE001
            logger.exception("Artifact detection failed")
            detections = []
        payload = VideoFramePayload(
            ts_ms=self._coerce_ts(packet),
            image=frame,
            detections=detections,
        )
        if self.on_frame:
            try:
                self.on_frame(payload)
            except Exception:  # noqa: BLE001
                logger.exception("Video frame callback failed")
        if self.on_artifact_event:
            event = {
                "type": "artifact_detections",
                "ts_ms": payload.ts_ms,
                "detections": [det.to_payload() for det in detections],
            }
            try:
                self.on_artifact_event(event)
            except Exception:  # noqa: BLE001
                logger.exception("Artifact event callback failed")

    def _decode_packet(self, packet: Dict[str, Any]) -> Optional[Image.Image]:
        jpeg_field = packet.get("jpeg_b64") or packet.get("jpeg")
        if jpeg_field is None:
            return None
        if isinstance(jpeg_field, str):
            jpeg_bytes = base64.b64decode(jpeg_field)
        elif isinstance(jpeg_field, (bytes, bytearray)):
            jpeg_bytes = bytes(jpeg_field)
        else:
            return None
        with io.BytesIO(jpeg_bytes) as buffer:
            image = Image.open(buffer)
            image = image.convert("RGB")
        return image

    def _coerce_ts(self, packet: Dict[str, Any]) -> int:
        raw = packet.get("ts_ms") or packet.get("ts")
        if isinstance(raw, (int, float)):
            if raw > 1_000_000_000_000:
                return int(raw)
            return int(raw * 1000)
        return int(time.time() * 1000)

