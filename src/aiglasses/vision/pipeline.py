from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from aiglasses.config import AppConfig

from .ncnn_yolo import ModelUnavailable, NcnnYoloModel, filter_detections
from .types import FrameAnalysis


OBSTACLE_LABELS = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "animal",
    "scooter",
    "stroller",
    "dog",
    "pole",
    "post",
    "column",
    "pillar",
    "bollard",
    "bench",
    "chair",
    "potted plant",
    "hydrant",
    "cone",
    "stone",
    "box",
}

NON_SIGNAL_TRAFFIC_LABELS = {None, "blank", "countdown_blank", "crossing"}


@dataclass
class VisionPipeline:
    config: AppConfig

    def __post_init__(self) -> None:
        size = (self.config.models.image_width, self.config.models.image_height)
        thresholds = self.config.vision_thresholds
        self.model_status: dict[str, str] = {}
        self.blind_model = self._optional_model(
            "blind_path",
            self.config.models.blind_path,
            image_size=size,
            confidence=thresholds.blind_path_conf,
            kind="segment",
        )
        self.obstacle_model = self._optional_model(
            "obstacle",
            self.config.models.obstacle,
            image_size=size,
            confidence=thresholds.obstacle_conf,
            kind="segment",
        )
        self.traffic_model = self._optional_model(
            "traffic_light",
            self.config.models.traffic_light,
            image_size=size,
            confidence=thresholds.traffic_light_conf,
            kind="detect",
        )

    def _optional_model(self, name: str, path: str, **kwargs: Any) -> NcnnYoloModel | None:
        try:
            model = NcnnYoloModel(
                path,
                min_mask_area=self.config.vision_thresholds.mask_min_area,
                ncnn_device=self.config.models.ncnn_device,
                ncnn_device_index=self.config.models.ncnn_device_index,
                **kwargs,
            )
            self.model_status[name] = "configured"
            return model
        except ModelUnavailable as exc:
            self.model_status[name] = str(exc)
            return None

    def analyze_jpeg(self, payload: bytes) -> FrameAnalysis:
        data = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return FrameAnalysis(model_status={**self.model_status, "frame": "decode_failed"})
        return self.analyze_frame(frame)

    def analyze_frame(self, frame: np.ndarray) -> FrameAnalysis:
        status = dict(self.model_status)
        blind_summary = None
        crosswalk_summary = None
        obstacles = []
        traffic_light = None
        traffic_conf = 0.0
        traffic_detection = None

        if self.blind_model:
            try:
                result = self.blind_model.predict(frame)
                blind_summary = result.masks.get("blind_path")
                crosswalk_summary = result.masks.get("road_crossing") or result.masks.get("crossing")
                status["blind_path"] = self.blind_model.status
            except Exception as exc:
                status["blind_path"] = f"error: {exc}"

        if self.obstacle_model:
            try:
                result = self.obstacle_model.predict(frame)
                obstacles = filter_detections(result.detections, OBSTACLE_LABELS)
                status["obstacle"] = self.obstacle_model.status
            except Exception as exc:
                status["obstacle"] = f"error: {exc}"

        if self.traffic_model:
            try:
                result = self.traffic_model.predict(frame)
                traffic_detection = max(
                    result.detections,
                    key=lambda det: det.confidence,
                    default=None,
                )
                if traffic_detection and traffic_detection.label in NON_SIGNAL_TRAFFIC_LABELS:
                    traffic_detection = None
                if result.top_label not in NON_SIGNAL_TRAFFIC_LABELS:
                    traffic_light = result.top_label
                    traffic_conf = result.top_confidence
                status["traffic_light"] = self.traffic_model.status
            except Exception as exc:
                status["traffic_light"] = f"error: {exc}"
                traffic_detection = None

        analysis = FrameAnalysis(
            blind_path=blind_summary,
            crosswalk=crosswalk_summary,
            obstacles=obstacles,
            traffic_light=traffic_light,
            traffic_light_confidence=traffic_conf,
            traffic_light_detection=traffic_detection,
            model_status=status,
        )
        self.model_status = status
        return analysis
