from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .yolo_postprocess import (
    LetterboxTransform,
    ModelUnavailable,
    YoloOutput,
    postprocess_yolo_outputs,
)


LETTERBOX_PAD_VALUE = 114


class TorchYoloModel:
    def __init__(
        self,
        path: str | Path,
        *,
        image_size: tuple[int, int],
        confidence: float,
        kind: str,
        torch_device: str = "cuda:0",
        torch_half: bool = True,
        min_mask_area: float = 0.01,
        class_id_names: dict[int, str] | None = None,
        class_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        model_path = Path(path)
        if model_path.suffix != ".pt":
            raise ModelUnavailable(f"torch runtime requires a .pt model: {model_path}")
        if not model_path.exists():
            raise ModelUnavailable(f"missing PyTorch model: {model_path}")

        self.path = model_path
        self.width, self.height = image_size
        self.confidence = confidence
        self.kind = kind
        self.torch_device = torch_device
        self.torch_half = torch_half
        self.min_mask_area = min_mask_area
        self.class_id_names = dict(class_id_names or {})
        self.class_names = tuple(class_names or ())
        self.names: dict[int, str] = {}
        self.num_classes: int | None = None
        self._model = None
        self.status = "not_loaded"

    def load(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on host install
            raise ModelUnavailable(f"torch/ultralytics import failed: {exc}") from exc

        self._torch = torch
        model = YOLO(str(self.path))
        self._model = model
        try:
            self._configure_class_names()
            model.model.to(self.torch_device)
            if self.torch_half and self.torch_device != "cpu":
                model.model.half()
            else:
                model.model.float()
            model.model.eval()
        except Exception:
            self._model = None
            self.names = {}
            self.num_classes = None
            self.status = "not_loaded"
            raise
        self.status = "ready"

    def _configure_class_names(self) -> None:
        self._refresh_class_names()
        if self.class_names:
            expected = list(self.class_names)
            current = [self.names[i] for i in sorted(self.names)] if self.names else []
            if current != expected:
                raise ModelUnavailable(
                    f"{self.path} has classes {current[:5]}..., expected the fixed "
                    "YOLOE obstacle classes. Run: uv run python -m "
                    "aiglasses.vision.export_yoloe_obstacle --source "
                    "models/yoloe-11l-seg.pt --output models/yoloe-11l-seg-obstacle.pt"
                )

        if self.class_id_names:
            self.names.update(self.class_id_names)

    def _refresh_class_names(self) -> None:
        names = getattr(self._model.model, "names", None) or getattr(self._model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.names = {int(key): str(value) for key, value in names.items()}
        self.num_classes = self._model_num_classes() or len(self.names) or None

    def _model_num_classes(self) -> int | None:
        model_layers = getattr(getattr(self._model, "model", None), "model", None)
        if isinstance(model_layers, (list, tuple)) and model_layers:
            nc = getattr(model_layers[-1], "nc", None)
            if isinstance(nc, int) and nc > 0:
                return nc
        return None

    def predict(self, bgr: np.ndarray) -> YoloOutput:
        if self._model is None:
            self.load()

        image, letterbox = self._letterbox(bgr)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None].copy()
        tensor = self._torch.from_numpy(chw).to(self.torch_device)
        if self.torch_half and self.torch_device != "cpu":
            tensor = tensor.half()

        with self._torch.inference_mode():
            output = self._model.model(tensor)

        raw0, raw1 = self._to_raw_outputs(output)
        return postprocess_yolo_outputs(
            raw0,
            raw1,
            names=self.names,
            num_classes=self.num_classes,
            allowed_class_ids=set(self.class_id_names) if self.class_id_names else None,
            width=self.width,
            height=self.height,
            confidence=self.confidence,
            min_mask_area=self.min_mask_area,
            letterbox=letterbox,
        )

    def _letterbox(self, bgr: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        source_height, source_width = bgr.shape[:2]
        scale = min(self.width / source_width, self.height / source_height)
        content_width = min(self.width, max(1, int(round(source_width * scale))))
        content_height = min(self.height, max(1, int(round(source_height * scale))))
        if (content_width, content_height) == (source_width, source_height):
            resized = bgr
        else:
            resized = cv2.resize(
                bgr,
                (content_width, content_height),
                interpolation=cv2.INTER_LINEAR,
            )

        pad_width = max(0, self.width - content_width)
        pad_height = max(0, self.height - content_height)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        if pad_width or pad_height:
            image = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
            )
        else:
            image = resized

        return image, LetterboxTransform(
            source_width=source_width,
            source_height=source_height,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            content_width=content_width,
            content_height=content_height,
        )

    def _to_raw_outputs(self, output) -> tuple[np.ndarray, np.ndarray | None]:
        proto = None
        if isinstance(output, (list, tuple)):
            head = output[0]
            if isinstance(head, (list, tuple)):
                if len(head) > 1:
                    proto = head[1]
                head = head[0]
            elif len(output) > 1 and isinstance(output[1], tuple):
                proto = output[1][-1]
        else:
            head = output

        if isinstance(head, (list, tuple)):
            head = head[0]
        raw0 = head.detach().float().cpu().numpy()

        raw1 = None
        if proto is not None:
            if isinstance(proto, (list, tuple)):
                proto = proto[-1]
            raw1 = proto.detach().float().cpu().numpy()

        return raw0, raw1
