from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .types import Detection


SIGNAL_TRAFFIC_LABELS = {"go", "stop", "countdown_go", "countdown_stop"}
NON_SIGNAL_TRAFFIC_LABELS = {"blank", "countdown_blank", "crossing"}
YELLOW_TRAFFIC_LABELS = {"countdown_go", "countdown_stop"}
CROSSING_BOTTOM_FIELDS = {
    "crossing_completion_bottom_max",
    "crossing_mid_bottom_min",
    "crosswalk_detection_stop_bottom_min",
    "crossing_start_bottom_min",
}


@dataclass
class VisionTuning:
    traffic_filter_enabled: bool = True
    traffic_light_conf: float = 0.20
    traffic_signal_clear_margin: float = 0.10
    traffic_go_min_conf: float = 0.20
    traffic_stop_min_conf: float = 0.20
    traffic_yellow_min_conf: float = 0.90
    traffic_conflict_margin: float = 0.10
    traffic_roi_enabled: bool = False
    traffic_roi_x_min: float = 0.15
    traffic_roi_x_max: float = 0.85
    traffic_roi_y_min: float = 0.00
    traffic_roi_y_max: float = 0.65
    traffic_min_area_ratio: float = 0.00005
    traffic_max_area_ratio: float = 0.10
    traffic_prefer_center_weight: float = 0.00
    crossing_green_required_frames: int = 2
    crossing_obstacles_enabled: bool = False
    crossing_alignment_offset_max: float = 0.15
    crossing_start_bottom_min: float = 0.60
    crossing_mid_bottom_min: float = 0.45
    crossing_completion_bottom_max: float = 0.35
    crossing_completion_min_active_frames: int = 4
    crossing_completion_min_active_seconds: float = 3.0
    crossing_completion_lost_frames: int = 10
    crossing_completion_required_frames: int = 3
    crossing_wait_signal_suppress_frames: int = 3
    crossing_obstacle_suppress_frames: int = 3
    crossing_active_timeout_seconds: float = 45.0
    crosswalk_detection_conf: float = 0.20
    crosswalk_detection_min_area_ratio: float = 0.005
    crosswalk_detection_x_min: float = 0.05
    crosswalk_detection_x_max: float = 0.95
    crosswalk_detection_alert_bottom_min: float = 0.25
    crosswalk_detection_stop_bottom_min: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, values: dict[str, Any]) -> None:
        updated = self.updated(values)
        for name, value in updated.to_dict().items():
            setattr(self, name, value)

    def updated(self, values: dict[str, Any]) -> VisionTuning:
        changes: dict[str, Any] = {}
        for name, value in values.items():
            if not hasattr(self, name):
                continue
            current = getattr(self, name)
            if isinstance(current, bool):
                changes[name] = _coerce_bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                changes[name] = int(value)
            else:
                changes[name] = float(value)
        tuning = replace(self, **changes)
        tuning.clamp(updated_fields=set(changes))
        return tuning

    def clamp(self, *, updated_fields: set[str] | None = None) -> None:
        self.traffic_light_conf = _clamp(self.traffic_light_conf, 0.0, 1.0)
        self.traffic_signal_clear_margin = _clamp(self.traffic_signal_clear_margin, 0.0, 1.0)
        self.traffic_go_min_conf = _clamp(self.traffic_go_min_conf, 0.0, 1.0)
        self.traffic_stop_min_conf = _clamp(self.traffic_stop_min_conf, 0.0, 1.0)
        self.traffic_yellow_min_conf = _clamp(self.traffic_yellow_min_conf, 0.0, 1.0)
        self.traffic_conflict_margin = _clamp(self.traffic_conflict_margin, 0.0, 1.0)
        self.traffic_roi_x_min = _clamp(self.traffic_roi_x_min, 0.0, 1.0)
        self.traffic_roi_x_max = _clamp(self.traffic_roi_x_max, 0.0, 1.0)
        self.traffic_roi_y_min = _clamp(self.traffic_roi_y_min, 0.0, 1.0)
        self.traffic_roi_y_max = _clamp(self.traffic_roi_y_max, 0.0, 1.0)
        if self.traffic_roi_x_min > self.traffic_roi_x_max:
            self.traffic_roi_x_min, self.traffic_roi_x_max = (
                self.traffic_roi_x_max,
                self.traffic_roi_x_min,
            )
        if self.traffic_roi_y_min > self.traffic_roi_y_max:
            self.traffic_roi_y_min, self.traffic_roi_y_max = (
                self.traffic_roi_y_max,
                self.traffic_roi_y_min,
            )
        self.traffic_min_area_ratio = _clamp(self.traffic_min_area_ratio, 0.0, 1.0)
        self.traffic_max_area_ratio = _clamp(self.traffic_max_area_ratio, 0.0, 1.0)
        if self.traffic_min_area_ratio > self.traffic_max_area_ratio:
            self.traffic_min_area_ratio, self.traffic_max_area_ratio = (
                self.traffic_max_area_ratio,
                self.traffic_min_area_ratio,
            )
        self.traffic_prefer_center_weight = _clamp(self.traffic_prefer_center_weight, 0.0, 1.0)
        self.crossing_green_required_frames = max(
            1,
            min(10, int(self.crossing_green_required_frames)),
        )
        self.crossing_alignment_offset_max = _clamp(self.crossing_alignment_offset_max, 0.0, 1.0)
        self.crossing_completion_bottom_max = _clamp(
            self.crossing_completion_bottom_max,
            0.0,
            1.0,
        )
        self.crossing_mid_bottom_min = _clamp(self.crossing_mid_bottom_min, 0.0, 1.0)
        self.crossing_start_bottom_min = _clamp(self.crossing_start_bottom_min, 0.0, 1.0)
        self.crossing_completion_min_active_frames = max(
            1,
            min(300, int(self.crossing_completion_min_active_frames)),
        )
        self.crossing_completion_min_active_seconds = _clamp(
            self.crossing_completion_min_active_seconds,
            0.0,
            120.0,
        )
        self.crossing_completion_lost_frames = max(
            1,
            min(300, int(self.crossing_completion_lost_frames)),
        )
        self.crossing_completion_required_frames = max(
            1,
            min(60, int(self.crossing_completion_required_frames)),
        )
        self.crossing_wait_signal_suppress_frames = max(
            0,
            min(120, int(self.crossing_wait_signal_suppress_frames)),
        )
        self.crossing_obstacle_suppress_frames = max(
            0,
            min(120, int(self.crossing_obstacle_suppress_frames)),
        )
        self.crossing_active_timeout_seconds = _clamp(
            self.crossing_active_timeout_seconds,
            1.0,
            300.0,
        )
        self.crosswalk_detection_conf = _clamp(self.crosswalk_detection_conf, 0.0, 1.0)
        self.crosswalk_detection_min_area_ratio = _clamp(
            self.crosswalk_detection_min_area_ratio,
            0.0,
            1.0,
        )
        self.crosswalk_detection_x_min = _clamp(self.crosswalk_detection_x_min, 0.0, 1.0)
        self.crosswalk_detection_x_max = _clamp(self.crosswalk_detection_x_max, 0.0, 1.0)
        if self.crosswalk_detection_x_min > self.crosswalk_detection_x_max:
            self.crosswalk_detection_x_min, self.crosswalk_detection_x_max = (
                self.crosswalk_detection_x_max,
                self.crosswalk_detection_x_min,
            )
        self.crosswalk_detection_alert_bottom_min = _clamp(
            self.crosswalk_detection_alert_bottom_min,
            0.0,
            1.0,
        )
        self.crosswalk_detection_stop_bottom_min = _clamp(
            self.crosswalk_detection_stop_bottom_min,
            0.0,
            1.0,
        )
        self._clamp_crossing_bottom_order(updated_fields=updated_fields)
        self.crosswalk_detection_alert_bottom_min = min(
            self.crosswalk_detection_alert_bottom_min,
            self.crosswalk_detection_stop_bottom_min,
        )

    def _clamp_crossing_bottom_order(self, *, updated_fields: set[str] | None = None) -> None:
        crossing_updates = (updated_fields or set()) & CROSSING_BOTTOM_FIELDS
        if len(crossing_updates) == 1:
            self._clamp_single_crossing_bottom_field(next(iter(crossing_updates)))
            return
        completion, middle, stop, start = sorted(
            [
                self.crossing_completion_bottom_max,
                self.crossing_mid_bottom_min,
                self.crosswalk_detection_stop_bottom_min,
                self.crossing_start_bottom_min,
            ]
        )
        self.crossing_completion_bottom_max = completion
        self.crossing_mid_bottom_min = middle
        self.crosswalk_detection_stop_bottom_min = stop
        self.crossing_start_bottom_min = start

    def _clamp_single_crossing_bottom_field(self, field: str) -> None:
        if field == "crossing_completion_bottom_max":
            self.crossing_completion_bottom_max = min(
                self.crossing_completion_bottom_max,
                self.crossing_mid_bottom_min,
            )
        elif field == "crossing_mid_bottom_min":
            self.crossing_mid_bottom_min = _clamp(
                self.crossing_mid_bottom_min,
                self.crossing_completion_bottom_max,
                self.crosswalk_detection_stop_bottom_min,
            )
        elif field == "crosswalk_detection_stop_bottom_min":
            self.crosswalk_detection_stop_bottom_min = _clamp(
                self.crosswalk_detection_stop_bottom_min,
                self.crossing_mid_bottom_min,
                self.crossing_start_bottom_min,
            )
        elif field == "crossing_start_bottom_min":
            self.crossing_start_bottom_min = max(
                self.crossing_start_bottom_min,
                self.crosswalk_detection_stop_bottom_min,
            )


def default_vision_tuning(traffic_light_conf: float) -> VisionTuning:
    tuning = VisionTuning(traffic_light_conf=float(traffic_light_conf))
    tuning.clamp()
    return tuning


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    return bool(value)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def select_traffic_signal(
    detections: list[Detection],
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[Detection | None, dict[str, Any]]:
    candidates = [det for det in detections if det.confidence >= tuning.traffic_light_conf]
    debug: dict[str, Any] = {
        "filter_enabled": tuning.traffic_filter_enabled,
        "selected": None,
        "reason": "no_candidates_after_model_threshold" if not candidates else "raw_top_confidence",
        "thresholds": tuning.to_dict(),
        "candidates": [
            _candidate_debug(det, tuning, width=width, height=height)
            for det in detections
        ],
    }
    if not candidates:
        return None, debug

    if not tuning.traffic_filter_enabled:
        selected = max(candidates, key=lambda det: det.confidence)
        debug["selected"] = selected.to_dict()
        return selected, debug

    eligible = [
        det
        for det in candidates
        if _passes_traffic_filters(det, tuning, width=width, height=height)
    ]
    debug["filtered_candidates"] = [
        _candidate_debug(det, tuning, width=width, height=height) for det in eligible
    ]
    if not eligible:
        debug["reason"] = "no_candidates_after_spatial_filters"
        return None, debug

    stop = _best_label(eligible, {"stop"}, tuning, width=width, height=height)
    yellow = _best_label(eligible, YELLOW_TRAFFIC_LABELS, tuning, width=width, height=height)
    go = _best_label(eligible, {"go"}, tuning, width=width, height=height)
    non_signal = max(
        (det for det in eligible if det.label in NON_SIGNAL_TRAFFIC_LABELS),
        key=lambda det: det.confidence,
        default=None,
    )

    if stop is not None:
        debug["selected"] = stop.to_dict()
        debug["reason"] = "stop_priority"
        return stop, debug
    if yellow is not None:
        debug["selected"] = yellow.to_dict()
        debug["reason"] = "yellow_priority"
        return yellow, debug
    if go is None:
        debug["reason"] = "no_signal_candidate"
        return None, debug
    if (
        non_signal is not None
        and go.confidence + tuning.traffic_signal_clear_margin < non_signal.confidence
    ):
        debug["reason"] = "non_signal_stronger_than_go"
        return None, debug
    conflict = max(
        (
            det
            for det in eligible
            if det.label in {"stop", "countdown_go", "countdown_stop"}
            and det.confidence + tuning.traffic_conflict_margin >= go.confidence
        ),
        key=lambda det: det.confidence,
        default=None,
    )
    if conflict is not None:
        debug["selected"] = conflict.to_dict()
        debug["reason"] = "go_conflicts_with_stop_or_yellow"
        return conflict, debug
    debug["selected"] = go.to_dict()
    debug["reason"] = "go_passed_filters"
    return go, debug


def select_crosswalk_detection(
    detections: list[Detection],
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Detection | None:
    candidates = [
        det
        for det in detections
        if det.label == "crossing"
        and det.confidence >= tuning.crosswalk_detection_conf
        and _normalised_box_area(_normalised_box(det, width=width, height=height))
        >= tuning.crosswalk_detection_min_area_ratio
        and _crosswalk_detection_box_passes(det, tuning, width=width, height=height)
    ]
    return max(
        candidates,
        key=lambda det: _crosswalk_detection_priority(det, width=width, height=height),
        default=None,
    )


def _crosswalk_detection_box_passes(
    det: Detection,
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    x1, _, x2, _ = _normalised_box(det, width=width, height=height)
    center_x = (x1 + x2) / 2.0
    return tuning.crosswalk_detection_x_min <= center_x <= tuning.crosswalk_detection_x_max


def _crosswalk_detection_priority(
    det: Detection,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[float, float, float]:
    box = _normalised_box(det, width=width, height=height)
    _, _, _, y2 = box
    return (y2, det.confidence, _normalised_box_area(box))


def _best_label(
    detections: list[Detection],
    labels: set[str],
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Detection | None:
    min_conf = 0.0
    if labels == {"go"}:
        min_conf = tuning.traffic_go_min_conf
    elif labels == {"stop"}:
        min_conf = tuning.traffic_stop_min_conf
    elif labels == YELLOW_TRAFFIC_LABELS:
        min_conf = tuning.traffic_yellow_min_conf
    return max(
        (det for det in detections if det.label in labels and det.confidence >= min_conf),
        key=lambda det: _traffic_score(det, tuning, width=width, height=height),
        default=None,
    )


def _passes_traffic_filters(
    det: Detection,
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    if det.label not in SIGNAL_TRAFFIC_LABELS | NON_SIGNAL_TRAFFIC_LABELS:
        return False
    if det.area_ratio < tuning.traffic_min_area_ratio:
        return False
    if det.area_ratio > tuning.traffic_max_area_ratio:
        return False
    if not tuning.traffic_roi_enabled:
        return True
    x1, y1, x2, y2 = _normalised_box(det, width=width, height=height)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return (
        tuning.traffic_roi_x_min <= center_x <= tuning.traffic_roi_x_max
        and tuning.traffic_roi_y_min <= center_y <= tuning.traffic_roi_y_max
    )


def _traffic_score(
    det: Detection,
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> float:
    x1, _, x2, _ = _normalised_box(det, width=width, height=height)
    center_distance = abs(((x1 + x2) / 2.0) - 0.5)
    return det.confidence - center_distance * tuning.traffic_prefer_center_weight


def _candidate_debug(
    det: Detection,
    tuning: VisionTuning,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    return {
        **det.to_dict(),
        "passes_filters": _passes_traffic_filters(det, tuning, width=width, height=height),
        "score": round(_traffic_score(det, tuning, width=width, height=height), 4),
    }


def _normalised_box(
    det: Detection,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = det.box
    x1, y1, x2, y2 = (float(x1), float(y1), float(x2), float(y2))
    if width and width > 1.0 and max(abs(x1), abs(x2)) > 1.0:
        x1 /= float(width)
        x2 /= float(width)
    if height and height > 1.0 and max(abs(y1), abs(y2)) > 1.0:
        y1 /= float(height)
        y2 /= float(height)
    return (
        _clamp(x1, 0.0, 1.0),
        _clamp(y1, 0.0, 1.0),
        _clamp(x2, 0.0, 1.0),
        _clamp(y2, 0.0, 1.0),
    )


def _normalised_box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
