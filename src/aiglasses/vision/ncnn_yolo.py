from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

import cv2
import numpy as np

from .types import Detection, MaskSummary


class ModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class YoloOutput:
    detections: list[Detection]
    masks: dict[str, MaskSummary]
    top_label: str | None = None
    top_confidence: float = 0.0


def validate_ncnn_model_dir(path: str | Path) -> Path:
    model_dir = Path(path)
    if model_dir.suffix == ".pt":
        raise ModelUnavailable(f"runtime cannot load PyTorch model: {model_dir}")
    if not model_dir.exists():
        raise ModelUnavailable(f"missing model directory: {model_dir}")
    if not (model_dir / "model.ncnn.param").exists() or not (model_dir / "model.ncnn.bin").exists():
        raise ModelUnavailable(f"missing NCNN param/bin in: {model_dir}")
    return model_dir


def _load_names(path: Path) -> dict[int, str]:
    metadata = path / "metadata.yaml"
    if not metadata.exists():
        return {}
    names: dict[int, str] = {}
    in_names = False
    for raw_line in metadata.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line == "names:":
            in_names = True
            continue
        if in_names:
            if line and not line.startswith(" "):
                break
            stripped = line.strip()
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                try:
                    names[int(key)] = value.strip().strip("'\"")
                except ValueError:
                    continue
    return names


def _load_output_blobs(path: Path) -> set[str]:
    param = path / "model.ncnn.param"
    outputs: set[str] = set()
    for raw_line in param.read_text(encoding="utf-8").splitlines()[2:]:
        parts = raw_line.split()
        if len(parts) < 4:
            continue
        try:
            input_count = int(parts[2])
            output_count = int(parts[3])
        except ValueError:
            continue
        output_start = 4 + input_count
        outputs.update(parts[output_start : output_start + output_count])
    return outputs


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def _summarize_mask(label: str, mask: np.ndarray, min_area: float) -> MaskSummary | None:
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    area_ratio = float(xs.size / max(h * w, 1))
    if area_ratio < min_area:
        return None
    center_offset = float((xs.mean() / max(w - 1, 1)) - 0.5) * 2.0
    vertical_position = float(ys.mean() / max(h - 1, 1))
    angle_deg = 0.0
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if points.shape[0] >= 16:
        vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle_deg = float(math.degrees(math.atan2(vx, vy)))

    contour_points: list[tuple[float, float]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(1.5, perimeter * 0.01), True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] > 80:
            step = int(math.ceil(pts.shape[0] / 80))
            pts = pts[::step]
        contour_points = [
            (float(x) / max(w - 1, 1), float(y) / max(h - 1, 1))
            for x, y in pts
        ]

    return MaskSummary(label, area_ratio, center_offset, vertical_position, angle_deg, contour_points)


class NcnnYoloModel:
    def __init__(
        self,
        path: str | Path,
        *,
        image_size: tuple[int, int],
        confidence: float,
        kind: str,
        ncnn_device: str = "vulkan",
        min_mask_area: float = 0.01,
    ) -> None:
        self.path = validate_ncnn_model_dir(path)
        self.width, self.height = image_size
        self.confidence = confidence
        self.kind = kind
        self.ncnn_device = ncnn_device
        self.min_mask_area = min_mask_area
        self.names = _load_names(self.path)
        self.output_blobs = _load_output_blobs(self.path)
        self._net = None
        self.status = "not_loaded"

    def load(self) -> None:
        try:
            import ncnn
        except Exception as exc:  # pragma: no cover - depends on host install
            raise ModelUnavailable(f"ncnn import failed: {exc}") from exc

        self._ncnn = ncnn
        self._net = ncnn.Net()
        if hasattr(self._net, "opt"):
            opt = self._net.opt
            if hasattr(opt, "use_vulkan_compute"):
                opt.use_vulkan_compute = self.ncnn_device in {"auto", "gpu", "vulkan"}
            if hasattr(opt, "num_threads"):
                opt.num_threads = 4
        self._net.load_param(str(self.path / "model.ncnn.param"))
        self._net.load_model(str(self.path / "model.ncnn.bin"))
        self.status = "ready"

    def predict(self, bgr: np.ndarray) -> YoloOutput:
        if self._net is None:
            self.load()
        resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1)).copy()

        with self._net.create_extractor() as ex:
            ex.input("in0", self._ncnn.Mat(chw).clone())
            _, out0 = ex.extract("out0")
            raw0 = np.array(out0)
            raw1 = None
            if "out1" in self.output_blobs:
                _, out1 = ex.extract("out1")
                raw1 = np.array(out1)

        return self._postprocess(raw0, raw1)

    def _postprocess(self, raw0: np.ndarray, raw1: np.ndarray | None) -> YoloOutput:
        pred = np.squeeze(raw0)
        if pred.ndim != 2:
            return YoloOutput([], {})
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        num_classes = len(self.names) or max(1, pred.shape[1] - 4)
        has_masks = raw1 is not None and pred.shape[1] > 4 + num_classes
        class_scores = pred[:, 4 : 4 + num_classes]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(pred.shape[0]), class_ids]
        keep = scores >= self.confidence
        if not keep.any():
            return YoloOutput([], {})

        pred = pred[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        xywh = pred[:, :4]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.height)

        final_idx: list[int] = []
        for class_id in sorted(set(class_ids.tolist())):
            idx = np.where(class_ids == class_id)[0]
            final_idx.extend(idx[i] for i in _nms(boxes[idx], scores[idx]))

        detections: list[Detection] = []
        masks: dict[str, MaskSummary] = {}
        for i in final_idx:
            label = self.names.get(int(class_ids[i]), str(int(class_ids[i])))
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(self.width * self.height, 1))
            detections.append(Detection(label, float(scores[i]), (x1, y1, x2, y2), area_ratio))

        if has_masks:
            proto = np.squeeze(raw1)
            if proto.ndim == 3:
                coeffs = pred[:, 4 + num_classes :]
                for i in final_idx:
                    label = self.names.get(int(class_ids[i]), str(int(class_ids[i])))
                    mask = np.tensordot(coeffs[i], proto, axes=(0, 0))
                    mask = 1.0 / (1.0 + np.exp(-mask))
                    mask = cv2.resize(mask, (self.width, self.height))
                    summary = _summarize_mask(label, (mask > 0.5).astype(np.uint8), self.min_mask_area)
                    if summary and (
                        label not in masks or summary.area_ratio > masks[label].area_ratio
                    ):
                        masks[label] = summary

        top = max(detections, key=lambda det: det.confidence, default=None)
        return YoloOutput(
            detections=detections,
            masks=masks,
            top_label=top.label if top else None,
            top_confidence=top.confidence if top else 0.0,
        )


def filter_detections(detections: Iterable[Detection], labels: set[str]) -> list[Detection]:
    return [item for item in detections if item.label in labels]
