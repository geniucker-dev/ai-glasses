from __future__ import annotations

import unittest

import numpy as np

from aiglasses.vision.yolo_postprocess import LetterboxTransform, postprocess_yolo_outputs


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

    def test_letterbox_maps_boxes_back_to_source_frame(self) -> None:
        pred = np.zeros((5, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 5.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.90

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "blind_path"},
            num_classes=1,
            width=8,
            height=10,
            confidence=0.35,
            min_mask_area=0.0,
            letterbox=LetterboxTransform(
                source_width=8,
                source_height=8,
                scale=1.0,
                pad_left=0,
                pad_top=1,
                content_width=8,
                content_height=8,
            ),
        )

        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].box, (2.0, 2.0, 6.0, 6.0))
        self.assertAlmostEqual(result.detections[0].area_ratio, 0.25)

    def test_letterbox_mask_summary_ignores_padding(self) -> None:
        pred = np.zeros((6, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 5.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.90
        pred[5, 0] = 1.0
        proto = np.full((1, 10, 8), -10.0, dtype=np.float32)
        proto[0, 0, :] = 10.0
        proto[0, 9, :] = 10.0

        result = postprocess_yolo_outputs(
            pred,
            proto,
            names={0: "blind_path"},
            num_classes=1,
            width=8,
            height=10,
            confidence=0.35,
            min_mask_area=0.0,
            letterbox=LetterboxTransform(
                source_width=8,
                source_height=8,
                scale=1.0,
                pad_left=0,
                pad_top=1,
                content_width=8,
                content_height=8,
            ),
        )

        self.assertEqual(len(result.detections), 1)
        self.assertNotIn("blind_path", result.masks)

    def test_mask_is_cropped_to_detection_box_before_summary(self) -> None:
        pred = np.zeros((7, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.90
        pred[5, 0] = 1.0
        proto = np.zeros((2, 8, 8), dtype=np.float32)
        proto[0] = 1.0

        result = postprocess_yolo_outputs(
            pred,
            proto,
            names={0: "blind_path"},
            num_classes=1,
            width=8,
            height=8,
            confidence=0.35,
            min_mask_area=0.0,
        )

        self.assertIn("blind_path", result.masks)
        self.assertAlmostEqual(result.masks["blind_path"].area_ratio, 0.25)

    def test_letterbox_drops_boxes_that_only_hit_padding(self) -> None:
        pred = np.zeros((5, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 0.5, 4.0, 1.0], dtype=np.float32)
        pred[4, 0] = 0.90

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "blind_path"},
            num_classes=1,
            width=8,
            height=10,
            confidence=0.35,
            min_mask_area=0.0,
            letterbox=LetterboxTransform(
                source_width=8,
                source_height=8,
                scale=1.0,
                pad_left=0,
                pad_top=1,
                content_width=8,
                content_height=8,
            ),
        )

        self.assertEqual(result.detections, [])

    def test_confidence_by_label_overrides_default_threshold(self) -> None:
        pred = np.zeros((6, 10), dtype=np.float32)
        pred[:4, 0] = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        pred[4, 0] = 0.30
        pred[:4, 1] = np.array([6.0, 6.0, 2.0, 2.0], dtype=np.float32)
        pred[5, 1] = 0.30

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "go", 1: "crossing"},
            num_classes=2,
            width=8,
            height=8,
            confidence=0.50,
            confidence_by_label={"crossing": 0.20},
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["crossing"])

    def test_confidence_by_label_blocks_argmax_below_override_threshold(self) -> None:
        pred = np.zeros((6, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 2.0, 2.0], dtype=np.float32)
        pred[5, 0] = 0.30

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "go", 1: "crossing"},
            num_classes=2,
            width=8,
            height=8,
            confidence=0.20,
            confidence_by_label={"crossing": 0.80},
            min_mask_area=0.0,
        )

        self.assertEqual(result.detections, [])

    def test_confidence_by_label_does_not_duplicate_kept_argmax_row(self) -> None:
        pred = np.zeros((6, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 2.0, 2.0], dtype=np.float32)
        pred[4, 0] = 0.90
        pred[5, 0] = 0.25

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "go", 1: "crossing"},
            num_classes=2,
            width=8,
            height=8,
            confidence=0.20,
            confidence_by_label={"crossing": 0.20},
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["go"])

    def test_confidence_by_label_keeps_non_argmax_class_above_its_threshold(self) -> None:
        pred = np.zeros((6, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.30
        pred[5, 0] = 0.25

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "go", 1: "crossing"},
            num_classes=2,
            width=8,
            height=8,
            confidence=0.50,
            confidence_by_label={"crossing": 0.20},
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["crossing"])
        self.assertEqual(result.detections[0].confidence, 0.25)

    def test_confidence_by_label_does_not_expand_non_overridden_classes(self) -> None:
        pred = np.zeros((7, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.90
        pred[5, 0] = 0.80

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "go", 1: "stop", 2: "crossing"},
            num_classes=3,
            width=8,
            height=8,
            confidence=0.20,
            confidence_by_label={"crossing": 0.20},
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["go"])

    def test_confidence_by_label_does_not_duplicate_argmax_override_label(self) -> None:
        pred = np.zeros((6, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.25
        pred[5, 0] = 0.90

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "go", 1: "crossing"},
            num_classes=2,
            width=8,
            height=8,
            confidence=0.20,
            confidence_by_label={"crossing": 0.20},
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["crossing"])

    def test_confidence_by_label_keeps_one_best_override_per_row(self) -> None:
        pred = np.zeros((7, 10), dtype=np.float32)
        pred[:4, 0] = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
        pred[4, 0] = 0.30
        pred[5, 0] = 0.25
        pred[6, 0] = 0.24

        result = postprocess_yolo_outputs(
            pred,
            None,
            names={0: "blank", 1: "crossing", 2: "countdown_blank"},
            num_classes=3,
            width=8,
            height=8,
            confidence=0.50,
            confidence_by_label={"crossing": 0.20, "countdown_blank": 0.20},
            min_mask_area=0.0,
        )

        self.assertEqual([item.label for item in result.detections], ["crossing"])
        self.assertEqual(result.detections[0].confidence, 0.25)


if __name__ == "__main__":
    unittest.main()
