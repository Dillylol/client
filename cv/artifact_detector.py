"""Heuristic artifact (game-piece) detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ArtifactDetection:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float

    def to_payload(self) -> dict:
        return {
            "bbox": list(map(float, self.bbox)),
            "center": [float(self.center[0]), float(self.center[1])],
            "area": float(self.area),
        }


class ArtifactDetector:
    """Simple HSV-based color segmentation placeholder."""

    def __init__(
        self,
        *,
        min_area: float = 600.0,
        hsv_ranges: Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] | None = None,
    ) -> None:
        self.min_area = min_area
        self.hsv_ranges = hsv_ranges or (
            ((120, 70, 70), (160, 255, 255)),  # purple range A
            ((160, 40, 60), (180, 255, 255)),  # purple range B
        )

    def process(self, image: Image.Image) -> List[ArtifactDetection]:
        rgb = image.convert("RGB")
        np_img = np.array(rgb)
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            return []
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = None
        for lower, upper in self.hsv_ranges:
            current = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask = current if mask is None else cv2.bitwise_or(mask, current)

        if mask is None:
            return []

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[ArtifactDetection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2.0
            cy = y + h / 2.0
            detections.append(
                ArtifactDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    center=(cx, cy),
                    area=float(area),
                )
            )
        return detections

