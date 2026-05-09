from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from aiglasses.config import AppConfig

from .obstacle_classes import OBSTACLE_LABELS, YOLOE_OBSTACLE_CLASS_NAMES
from .torch_yolo import TorchYoloModel
from .tuning import VisionTuning, default_vision_tuning, select_traffic_signal
from .types import Detection, FrameAnalysis
from .yolo_postprocess import ModelUnavailable, filter_detections


@dataclass
class VisionPipeline:
    config: AppConfig

    def __post_init__(self) -> None:
        size = (self.config.models.image_width, self.config.models.image_height)
        thresholds = self.config.vision_thresholds
        self.model_status: dict[str, str] = {}
        self.tuning: VisionTuning = default_vision_tuning(thresholds.traffic_light_conf)
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
            class_names=YOLOE_OBSTACLE_CLASS_NAMES,
        )
        self.traffic_model = self._optional_model(
            "traffic_light",
            self.config.models.traffic_light,
            image_size=size,
            confidence=thresholds.traffic_light_conf,
            kind="detect",
        )

    def _optional_model(self, name: str, path: str, **kwargs: Any) -> TorchYoloModel | None:
        try:
            model = TorchYoloModel(
                path,
                min_mask_area=self.config.vision_thresholds.mask_min_area,
                torch_device=self.config.models.torch_device,
                torch_half=self.config.models.torch_half,
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
        if not hasattr(self, "tuning"):
            self.tuning = default_vision_tuning(self.config.vision_thresholds.traffic_light_conf)
        blind_summary = None
        crosswalk_summary = None
        obstacles = []
        traffic_light = None
        traffic_conf = 0.0
        traffic_detection = None
        traffic_candidates: list[Detection] = []
        traffic_debug: dict[str, Any] = {"thresholds": self.tuning.to_dict()}

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
                self.traffic_model.confidence = self.tuning.traffic_light_conf
                result = self.traffic_model.predict(frame)
                traffic_candidates = result.detections
                traffic_detection, traffic_debug = select_traffic_signal(
                    result.detections,
                    self.tuning,
                    width=self.config.models.image_width,
                    height=self.config.models.image_height,
                )
                if traffic_detection:
                    traffic_light = traffic_detection.label
                    traffic_conf = traffic_detection.confidence
                status["traffic_light"] = self.traffic_model.status
            except Exception as exc:
                status["traffic_light"] = f"error: {exc}"
                traffic_detection = None
                traffic_debug = {"error": str(exc), "thresholds": self.tuning.to_dict()}

        analysis = FrameAnalysis(
            blind_path=blind_summary,
            crosswalk=crosswalk_summary,
            obstacles=obstacles,
            traffic_light=traffic_light,
            traffic_light_confidence=traffic_conf,
            traffic_light_detection=traffic_detection,
            traffic_light_candidates=traffic_candidates,
            traffic_light_debug=traffic_debug,
            model_status=status,
            frame_width=self.config.models.image_width,
            frame_height=self.config.models.image_height,
        )
        self.model_status = status
        return analysis
