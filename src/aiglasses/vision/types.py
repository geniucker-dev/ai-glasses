from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    area_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaskSummary:
    label: str
    area_ratio: float
    center_offset: float
    vertical_position: float
    angle_deg: float = 0.0
    confidence: float = 0.0
    contour: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameAnalysis:
    blind_path: MaskSummary | None = None
    crosswalk_detection: Detection | None = None
    obstacles: list[Detection] = field(default_factory=list)
    traffic_light: str | None = None
    traffic_light_confidence: float = 0.0
    traffic_light_detection: Detection | None = None
    traffic_light_candidates: list[Detection] = field(default_factory=list)
    traffic_light_debug: dict[str, Any] = field(default_factory=dict)
    model_status: dict[str, str] = field(default_factory=dict)
    frame_width: int | None = None
    frame_height: int | None = None

    def to_observation(self) -> dict[str, Any]:
        nearest = max(self.obstacles, key=lambda item: item.area_ratio, default=None)
        return {
            "blind_path": self.blind_path.to_dict() if self.blind_path else None,
            "crosswalk_detection": (
                self.crosswalk_detection.to_dict() if self.crosswalk_detection else None
            ),
            "nearest_obstacle": nearest.to_dict() if nearest else None,
            "obstacles": [item.to_dict() for item in self.obstacles],
            "traffic_light": self.traffic_light,
            "traffic_light_confidence": self.traffic_light_confidence,
            "traffic_light_detection": (
                self.traffic_light_detection.to_dict() if self.traffic_light_detection else None
            ),
            "traffic_light_candidates": [item.to_dict() for item in self.traffic_light_candidates],
            "traffic_light_debug": self.traffic_light_debug,
            "model_status": self.model_status,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
