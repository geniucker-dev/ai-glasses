from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from aiglasses.config import AppConfig
from aiglasses.vision.pipeline import VisionPipeline
from aiglasses.vision.tuning import VisionTuning, select_traffic_signal
from aiglasses.vision.types import Detection
from aiglasses.vision.yolo_postprocess import YoloOutput


@dataclass
class FakeTrafficModel:
    detections: list[Detection]
    status: str = "ready"
    confidence: float = 0.0
    confidence_by_label: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.confidence_by_label = dict(self.confidence_by_label or {})

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
        self.assertIsNotNone(analysis.crosswalk_detection)
        self.assertEqual(analysis.crosswalk_detection.label, "crossing")

    def test_analysis_reports_source_frame_dimensions(self) -> None:
        pipeline = self._pipeline_with_traffic([])

        analysis = pipeline.analyze_frame(np.zeros((6, 8, 3), dtype=np.uint8))

        self.assertEqual(analysis.frame_width, 8)
        self.assertEqual(analysis.frame_height, 6)

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

    def test_crosswalk_detection_uses_traffic_model_crossing_box(self) -> None:
        pipeline = self._pipeline_with_traffic(
            [
                Detection("crossing", 0.82, (100.0, 120.0, 500.0, 320.0), 0.21),
                Detection("crossing", 0.92, (10.0, 10.0, 30.0, 40.0), 0.01),
            ]
        )
        pipeline.tuning.crosswalk_detection_conf = 0.50

        analysis = pipeline.analyze_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertIsNotNone(analysis.crosswalk_detection)
        self.assertEqual(analysis.crosswalk_detection.box, (100.0, 120.0, 500.0, 320.0))

    def test_crosswalk_detection_ignores_tiny_box(self) -> None:
        pipeline = self._pipeline_with_traffic(
            [Detection("crossing", 0.92, (310.0, 440.0, 330.0, 460.0), 0.90)]
        )
        pipeline.tuning.crosswalk_detection_conf = 0.50
        pipeline.tuning.crosswalk_detection_min_area_ratio = 0.005

        analysis = pipeline.analyze_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertIsNone(analysis.crosswalk_detection)

    def test_traffic_model_uses_separate_crosswalk_detection_threshold(self) -> None:
        pipeline = self._pipeline_with_traffic([])
        pipeline.tuning.traffic_light_conf = 0.20
        pipeline.tuning.crosswalk_detection_conf = 0.08

        pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(pipeline.traffic_model.confidence, 0.20)
        self.assertEqual(pipeline.traffic_model.confidence_by_label, {"crossing": 0.08})

    def test_traffic_model_preserves_existing_label_thresholds(self) -> None:
        pipeline = self._pipeline_with_traffic([])
        pipeline.traffic_model.confidence_by_label = {"blank": 0.70}
        pipeline.tuning.crosswalk_detection_conf = 0.08

        pipeline.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(
            pipeline.traffic_model.confidence_by_label,
            {"blank": 0.70, "crossing": 0.08},
        )


if __name__ == "__main__":
    unittest.main()
