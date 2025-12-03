"""Computer vision helpers for the JULES client."""

from .artifact_detector import ArtifactDetector, ArtifactDetection
from .video_pipeline import VideoFramePayload, VideoStreamProcessor

__all__ = [
    "ArtifactDetector",
    "ArtifactDetection",
    "VideoFramePayload",
    "VideoStreamProcessor",
]

