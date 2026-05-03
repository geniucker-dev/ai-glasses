from __future__ import annotations

import unittest

import numpy as np

from aiglasses.vision.yolo_postprocess import postprocess_yolo_outputs


class YoloPostprocessTests(unittest.TestCase):
    def test_allowed_class_ids_keep_mask_coefficients_at_model_class_boundary(self) -> None:
        pred = np.zeros((4 + 80 + 32, 200), dtype=np.float32)
        pred[:4, :2] = np.array([[4.0], [4.0], [4.0], [4.0]])
        pred[4 + 79, 0] = 0.99
        pred[4 + 1, 0] = 0.50
        pred[4 + 1, 1] = 0.90
        pred[4 + 80 :, :2] = 1.0
        proto = np.ones((32, 4, 4), dtype=np.float32)

        result = postprocess_yolo_outputs(
            pred,
            proto,
            names={1: "car"},
            num_classes=80,
            allowed_class_ids={1},
            width=8,
            height=8,
            confidence=0.35,
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["car"])
        self.assertIn("car", result.masks)


if __name__ == "__main__":
    unittest.main()
