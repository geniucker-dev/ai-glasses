from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from aiglasses.vision.torch_yolo import TorchYoloModel
from aiglasses.vision.yolo_postprocess import ModelUnavailable


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeYolo:
    def __init__(self) -> None:
        self.names = {0: "0", 1: "1"}
        self.model = self


class FakeYoloE:
    def __init__(self, names: dict[int, str]) -> None:
        self.names = names
        self.model = self
        self.set_classes_calls: list[list[str]] = []

    def set_classes(self, names: list[str]) -> None:
        self.set_classes_calls.append(names)
        self.names = dict(enumerate(names))


class TorchYoloTests(unittest.TestCase):
    def test_to_raw_outputs_extracts_segment_proto_from_head_tuple(self) -> None:
        pred = np.zeros((1, 2, 3), dtype=np.float32)
        proto = np.ones((1, 4, 5, 6), dtype=np.float32)

        raw0, raw1 = TorchYoloModel._to_raw_outputs(
            object(),
            ((FakeTensor(pred), FakeTensor(proto)), ("unused",)),
        )

        np.testing.assert_array_equal(raw0, pred)
        np.testing.assert_array_equal(raw1, proto)

    def test_configure_class_names_overlays_selected_class_ids(self) -> None:
        model = object.__new__(TorchYoloModel)
        model._model = FakeYolo()
        model.class_id_names = {1: "bicycle"}
        model.class_names = ()

        TorchYoloModel._configure_class_names(model)

        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.names, {0: "0", 1: "bicycle"})

    def test_configure_class_names_rejects_unfixed_yoloe_prompts(self) -> None:
        model = object.__new__(TorchYoloModel)
        model.path = "fake-yoloe.pt"
        model._model = FakeYoloE({0: "0", 1: "1"})
        model.class_id_names = {}
        model.class_names = ("bicycle", "car")

        with self.assertRaises(ModelUnavailable) as ctx:
            TorchYoloModel._configure_class_names(model)

        self.assertIn("export_yoloe_obstacle", str(ctx.exception))
        self.assertEqual(model._model.set_classes_calls, [])

    def test_configure_class_names_skips_yoloe_prompts_when_already_fixed(self) -> None:
        model = object.__new__(TorchYoloModel)
        model._model = FakeYoloE({0: "bicycle", 1: "car"})
        model.class_id_names = {}
        model.class_names = ("bicycle", "car")

        TorchYoloModel._configure_class_names(model)

        self.assertEqual(model._model.set_classes_calls, [])
        self.assertEqual(model.names, {0: "bicycle", 1: "car"})

    def test_load_clears_partial_model_after_class_validation_failure(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            raw_yoloe = FakeYoloE({0: "0", 1: "1"})
            model = TorchYoloModel(
                tmp.name,
                image_size=(640, 480),
                confidence=0.35,
                kind="segment",
                torch_device="cpu",
                torch_half=False,
                class_names=("bicycle", "car"),
            )

            modules = {
                "torch": types.SimpleNamespace(),
                "ultralytics": types.SimpleNamespace(YOLO=lambda _: raw_yoloe),
            }
            with patch.dict(sys.modules, modules):
                with self.assertRaises(ModelUnavailable):
                    model.load()

        self.assertIsNone(model._model)
        self.assertEqual(model.names, {})
        self.assertIsNone(model.num_classes)
        self.assertEqual(model.status, "not_loaded")
        self.assertEqual(raw_yoloe.set_classes_calls, [])


if __name__ == "__main__":
    unittest.main()
