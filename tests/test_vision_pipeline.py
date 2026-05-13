from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from aiglasses.config import AppConfig
from aiglasses.vision.pipeline import VisionPipeline
from aiglasses.vision.tuning import VisionTuning, select_traffic_signal
from aiglasses.vision.types import Detection, MaskSummary
from aiglasses.vision.yolo_postprocess import YoloOutput


@dataclass
class FakeBlindModel:
    masks: dict[str, MaskSummary]
    status: str = "ready"

    def predict(self, frame: np.ndarray) -> YoloOutput:
        return YoloOutput(detections=[], masks=self.masks)


@dataclass
class FakeTrafficModel:
    detections: list[Detection]
    status: str = "ready"

    def predict(self, frame: np.ndarray) -> YoloOutput:
        return YoloOutput(detections=self.detections, masks={})


class VisionPipelineTrafficLightTests(unittest.TestCase):
    def _pipeline_with_traffic(self, detections: list[Detection]) -> VisionPipeline:
        pipeline = object.__new__(VisionPipeline)
        pipeline.config = AppConfig(path="config.toml")
        pipeline.model_status = {}
        pipeline.blind_model = None
        pipeline.obstacle_model = None
        pipeline.traffic_model = FakeTrafficModel(detections)
        pipeline.tuning = VisionTuning(
            traffic_go_min_conf=0.0,
            traffic_stop_min_conf=0.99,
            traffic_yellow_min_conf=0.99,
        )
        return pipeline

    def test_traffic_light_ignores_weak_signal_under_much_stronger_blank(self) -> None:
        pipeline = self._pipeline_with_traffic(
            [
                Detection("blank", 0.95, (0.0, 0.0, 4.0, 4.0), 0.1),
                Detection("go", 0.70, (2.0, 2.0, 6.0, 6.0), 0.1),
                Detection("stop", 0.65, (4.0, 4.0, 8.0, 8.0), 0.1),
            ]
        )

        analysis = pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertIsNone(analysis.traffic_light)
        self.assertEqual(analysis.traffic_light_confidence, 0.0)
        self.assertIsNone(analysis.traffic_light_detection)

    def test_traffic_light_selects_clear_signal_over_blank(self) -> None:
        pipeline = self._pipeline_with_traffic(
            [
                Detection("blank", 0.80, (0.0, 0.0, 4.0, 4.0), 0.1),
                Detection("go", 0.92, (2.0, 2.0, 6.0, 6.0), 0.1),
                Detection("stop", 0.65, (4.0, 4.0, 8.0, 8.0), 0.1),
            ]
        )

        analysis = pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(analysis.traffic_light, "go")
        self.assertEqual(analysis.traffic_light_confidence, 0.92)
        self.assertIsNotNone(analysis.traffic_light_detection)
        self.assertEqual(analysis.traffic_light_detection.label, "go")

    def test_traffic_light_ignores_non_signal_detections(self) -> None:
        pipeline = self._pipeline_with_traffic(
            [
                Detection("blank", 0.95, (0.0, 0.0, 4.0, 4.0), 0.1),
                Detection("crossing", 0.90, (2.0, 2.0, 6.0, 6.0), 0.1),
            ]
        )

        analysis = pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertIsNone(analysis.traffic_light)
        self.assertEqual(analysis.traffic_light_confidence, 0.0)
        self.assertIsNone(analysis.traffic_light_detection)

    def test_center_weight_uses_frame_size_for_pixel_boxes(self) -> None:
        tuning = VisionTuning(
            traffic_go_min_conf=0.0,
            traffic_prefer_center_weight=1.0,
            traffic_roi_enabled=False,
        )
        left = Detection("go", 0.93, (0.0, 0.0, 20.0, 20.0), 0.01)
        center = Detection("go", 0.86, (310.0, 0.0, 330.0, 20.0), 0.01)

        selected, _debug = select_traffic_signal([left, center], tuning, width=640, height=480)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.box, center.box)

    def test_crosswalk_segmentation_confidence_filters_false_positive_mask(self) -> None:
        pipeline = object.__new__(VisionPipeline)
        pipeline.config = AppConfig(path="config.toml")
        pipeline.model_status = {}
        pipeline.blind_model = FakeBlindModel(
            {
                "blind_path": MaskSummary(
                    "blind_path",
                    area_ratio=0.08,
                    center_offset=0.0,
                    vertical_position=0.70,
                    confidence=0.92,
                ),
                "crossing": MaskSummary(
                    "crossing",
                    area_ratio=0.06,
                    center_offset=0.0,
                    vertical_position=0.60,
                    confidence=0.40,
                ),
            }
        )
        pipeline.obstacle_model = None
        pipeline.traffic_model = None
        pipeline.tuning = VisionTuning(crosswalk_conf=0.65)

        analysis = pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertIsNotNone(analysis.blind_path)
        self.assertIsNone(analysis.crosswalk)

    def test_crosswalk_segmentation_confidence_keeps_strong_mask(self) -> None:
        pipeline = object.__new__(VisionPipeline)
        pipeline.config = AppConfig(path="config.toml")
        pipeline.model_status = {}
        pipeline.blind_model = FakeBlindModel(
            {
                "crossing": MaskSummary(
                    "crossing",
                    area_ratio=0.06,
                    center_offset=0.0,
                    vertical_position=0.60,
                    confidence=0.86,
                )
            }
        )
        pipeline.obstacle_model = None
        pipeline.traffic_model = None
        pipeline.tuning = VisionTuning(crosswalk_conf=0.65)

        analysis = pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertIsNotNone(analysis.crosswalk)
        self.assertEqual(analysis.crosswalk.confidence, 0.86)


if __name__ == "__main__":
    unittest.main()
