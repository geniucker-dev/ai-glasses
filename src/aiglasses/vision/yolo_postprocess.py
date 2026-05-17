from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    scale: float
    pad_left: int
    pad_top: int
    content_width: int
    content_height: int


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


def _output_size(
    width: int,
    height: int,
    letterbox: LetterboxTransform | None,
) -> tuple[int, int]:
    if letterbox is None:
        return width, height
    return letterbox.source_width, letterbox.source_height


def _map_boxes_to_output(
    boxes: np.ndarray,
    *,
    width: int,
    height: int,
    letterbox: LetterboxTransform | None,
) -> np.ndarray:
    mapped = boxes.astype(np.float32, copy=True)
    if letterbox is not None:
        scale = max(letterbox.scale, 1e-6)
        mapped[:, [0, 2]] = (mapped[:, [0, 2]] - letterbox.pad_left) / scale
        mapped[:, [1, 3]] = (mapped[:, [1, 3]] - letterbox.pad_top) / scale
        out_w, out_h = letterbox.source_width, letterbox.source_height
    else:
        out_w, out_h = width, height
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, out_w)
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, out_h)
    return mapped


def _crop_mask_to_box(
    mask: np.ndarray,
    box: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    h, w = mask.shape[:2]
    x_scale = w / max(width, 1)
    y_scale = h / max(height, 1)
    x1 = int(round(float(box[0]) * x_scale))
    y1 = int(round(float(box[1]) * y_scale))
    x2 = int(round(float(box[2]) * x_scale))
    y2 = int(round(float(box[3]) * y_scale))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    cropped = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def _map_mask_to_output(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    letterbox: LetterboxTransform | None,
) -> np.ndarray:
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if letterbox is None:
        return mask
    x1 = letterbox.pad_left
    y1 = letterbox.pad_top
    x2 = x1 + letterbox.content_width
    y2 = y1 + letterbox.content_height
    content = mask[y1:y2, x1:x2]
    if content.shape[:2] == (letterbox.source_height, letterbox.source_width):
        return content
    return cv2.resize(
        content,
        (letterbox.source_width, letterbox.source_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _summarize_mask(
    label: str,
    mask: np.ndarray,
    min_area: float,
    confidence: float,
) -> MaskSummary | None:
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
            (float(x) / max(w - 1, 1), float(y) / max(h - 1, 1)) for x, y in pts
        ]

    return MaskSummary(
        label,
        area_ratio,
        center_offset,
        vertical_position,
        angle_deg,
        float(confidence),
        contour_points,
    )


def postprocess_yolo_outputs(
    raw0: np.ndarray,
    raw1: np.ndarray | None,
    *,
    names: dict[int, str],
    num_classes: int | None = None,
    allowed_class_ids: set[int] | None = None,
    width: int,
    height: int,
    confidence: float,
    confidence_by_label: dict[str, float] | None = None,
    min_mask_area: float,
    letterbox: LetterboxTransform | None = None,
) -> YoloOutput:
    pred = np.squeeze(raw0)
    if pred.ndim != 2:
        return YoloOutput([], {})
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    proto = np.squeeze(raw1) if raw1 is not None else None
    if num_classes is None:
        if proto is not None and proto.ndim == 3:
            num_classes = max(1, pred.shape[1] - 4 - proto.shape[0])
        else:
            num_classes = len(names) or max(1, pred.shape[1] - 4)
    if num_classes <= 0 or pred.shape[1] < 4 + num_classes:
        return YoloOutput([], {})

    valid_allowed_class_ids: set[int] | None = None
    if allowed_class_ids:
        valid_allowed_class_ids = {item for item in allowed_class_ids if 0 <= item < num_classes}
        if not valid_allowed_class_ids:
            return YoloOutput([], {})

    has_masks = raw1 is not None and pred.shape[1] > 4 + num_classes
    class_scores = pred[:, 4 : 4 + num_classes]
    pred_indices, class_ids, scores = _candidate_classes(
        class_scores,
        names=names,
        default_confidence=confidence,
        confidence_by_label=confidence_by_label,
        allowed_class_ids=valid_allowed_class_ids,
    )
    if pred_indices.size == 0:
        return YoloOutput([], {})

    pred = pred[pred_indices]
    xywh = pred[:, :4]
    input_boxes = np.empty_like(xywh)
    input_boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    input_boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    input_boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    input_boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    input_boxes[:, [0, 2]] = np.clip(input_boxes[:, [0, 2]], 0, width)
    input_boxes[:, [1, 3]] = np.clip(input_boxes[:, [1, 3]], 0, height)
    boxes = _map_boxes_to_output(input_boxes, width=width, height=height, letterbox=letterbox)
    out_w, out_h = _output_size(width, height, letterbox)
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid_boxes.any():
        return YoloOutput([], {})
    pred = pred[valid_boxes]
    scores = scores[valid_boxes]
    class_ids = class_ids[valid_boxes]
    boxes = boxes[valid_boxes]
    input_boxes = input_boxes[valid_boxes]

    final_idx: list[int] = []
    for class_id in sorted(set(class_ids.tolist())):
        idx = np.where(class_ids == class_id)[0]
        final_idx.extend(idx[i] for i in _nms(boxes[idx], scores[idx]))

    detections: list[Detection] = []
    masks: dict[str, MaskSummary] = {}
    for i in final_idx:
        label = names.get(int(class_ids[i]), str(int(class_ids[i])))
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(out_w * out_h, 1))
        detections.append(Detection(label, float(scores[i]), (x1, y1, x2, y2), area_ratio))

    if has_masks:
        if proto is not None and proto.ndim == 3:
            coeffs = pred[:, 4 + num_classes :]
            if coeffs.shape[1] == proto.shape[0]:
                for i in final_idx:
                    label = names.get(int(class_ids[i]), str(int(class_ids[i])))
                    mask = np.tensordot(coeffs[i], proto, axes=(0, 0))
                    mask = _crop_mask_to_box(mask, input_boxes[i], width=width, height=height)
                    mask = _map_mask_to_output(
                        mask,
                        width=width,
                        height=height,
                        letterbox=letterbox,
                    )
                    summary = _summarize_mask(
                        label,
                        (mask > 0.0).astype(np.uint8),
                        min_mask_area,
                        float(scores[i]),
                    )
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


def _candidate_classes(
    class_scores: np.ndarray,
    *,
    names: dict[int, str],
    default_confidence: float,
    confidence_by_label: dict[str, float] | None,
    allowed_class_ids: set[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Preserve normal YOLO single-label behavior for ordinary classes. Explicit
    # per-label thresholds also apply to argmax detections for those labels.
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    thresholds = np.full(scores.shape, float(default_confidence), dtype=np.float32)
    if confidence_by_label:
        for label, threshold in confidence_by_label.items():
            override_class_ids = [class_id for class_id, name in names.items() if name == label]
            for class_id in override_class_ids:
                thresholds[class_ids == class_id] = float(threshold)
    keep = scores >= thresholds
    if allowed_class_ids is not None:
        keep &= np.isin(class_ids, list(allowed_class_ids))
    pred_indices = np.nonzero(keep)[0]
    kept_class_ids = class_ids[keep]
    kept_scores = scores[keep]

    if confidence_by_label:
        best_override_class_ids = np.full(class_ids.shape, -1, dtype=class_ids.dtype)
        best_override_scores = np.full(scores.shape, -np.inf, dtype=np.float32)
        unused_rows = ~keep
        for label, threshold in confidence_by_label.items():
            override_class_ids = [class_id for class_id, name in names.items() if name == label]
            for class_id in override_class_ids:
                if allowed_class_ids is not None and class_id not in allowed_class_ids:
                    continue
                override_scores = class_scores[:, class_id]
                override_keep = override_scores >= float(threshold)
                override_keep &= unused_rows
                better_override = override_keep & (override_scores > best_override_scores)
                best_override_class_ids[better_override] = class_id
                best_override_scores[better_override] = override_scores[better_override]
        extra_indices = np.nonzero(best_override_class_ids >= 0)[0]
        if extra_indices.size:
            pred_indices = np.concatenate((pred_indices, extra_indices))
            kept_class_ids = np.concatenate((kept_class_ids, best_override_class_ids[extra_indices]))
            kept_scores = np.concatenate((kept_scores, best_override_scores[extra_indices]))

    return pred_indices, kept_class_ids, kept_scores


def filter_detections(detections: Iterable[Detection], labels: set[str]) -> list[Detection]:
    return [item for item in detections if item.label in labels]
