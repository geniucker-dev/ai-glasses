from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from aiglasses.config import AppConfig
from aiglasses.vision.pipeline import VisionPipeline
from aiglasses.vision.types import Detection
from aiglasses.vision.yolo_postprocess import YoloOutput


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


if __name__ == "__main__":
    unittest.main()
