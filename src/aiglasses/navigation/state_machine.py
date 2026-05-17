from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import time
from typing import Any, Callable

from aiglasses.vision.tuning import VisionTuning


class NavigationMode(StrEnum):
    IDLE = "idle"
    BLIND_PATH = "blind_path"
    CROSSING = "crossing"
    TRAFFIC_LIGHT = "traffic_light"


class CrossingPhase(StrEnum):
    SEARCHING = "searching"
    ALIGNING = "aligning"
    READY = "ready"
    ACTIVE = "crossing_active"
    SUSPECTED_COMPLETED = "suspected_completed"


@dataclass(frozen=True)
class CrossingDetectionInfo:
    box: tuple[float, float, float, float]
    area: float
    bottom: float
    center_y: float
    center_offset: float
    aligned: bool


@dataclass(frozen=True)
class CrossingFrameContext:
    has_pure_go: bool
    has_wait_caution: bool
    wait_label: str | None
    crosswalk: CrossingDetectionInfo | None
    obstacle_hazard: dict[str, Any] | None


OBSTACLE_SPEECH_LABELS = {
    "bicycle": "自行车",
    "car": "车",
    "motorcycle": "摩托车",
    "bus": "公交车",
    "truck": "卡车",
    "animal": "动物",
    "scooter": "滑板车",
    "stroller": "婴儿车",
    "dog": "狗",
    "pole": "杆子",
    "post": "柱子",
    "column": "柱子",
    "pillar": "柱子",
    "stanchion": "立柱",
    "bollard": "路桩",
    "utility pole": "电线杆",
    "telegraph pole": "电线杆",
    "light pole": "灯杆",
    "street pole": "路灯杆",
    "signpost": "标志杆",
    "support post": "支撑柱",
    "vertical post": "立柱",
    "bench": "长椅",
    "chair": "椅子",
    "potted plant": "盆栽",
    "hydrant": "消防栓",
    "cone": "锥桶",
    "stone": "石头",
    "box": "箱子",
}
BLIND_PATH_MIN_CONFIDENCE = 0.35
BLIND_PATH_MIN_AREA_RATIO = 0.003
BLIND_PATH_NEAR_AREA_RATIO = 0.015
BLIND_PATH_AHEAD_BOTTOM_MIN = 0.45
BLIND_PATH_CLOSE_BOTTOM_MIN = 0.60
BLIND_PATH_CORRIDOR_HALF_WIDTH = 0.38
BLIND_PATH_STRICT_HALF_WIDTH = 0.28

CROSSING_MIN_CONFIDENCE = 0.35
CROSSING_MIN_AREA_RATIO = 0.002
CROSSING_AHEAD_BOTTOM_MIN = 0.25
CROSSING_CORRIDOR_HALF_WIDTH = 0.55
CROSSING_GO_LABELS = {"go"}
CROSSING_WAIT_CAUTION_LABELS = {"stop", "countdown_go", "countdown_stop", "yellow"}


@dataclass(frozen=True)
class NavigationResult:
    mode: NavigationMode
    speech: str | None = None
    overlay: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


class NavigationStateMachine:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        tuning: VisionTuning | None = None,
    ) -> None:
        self.mode = NavigationMode.IDLE
        self.tuning = tuning or VisionTuning()
        self.last_speech = ""
        self._clock = clock
        self._candidate_speech = ""
        self._candidate_frames = 0
        self._last_guidance_spoken_at: float | None = None
        self._crossing_phase = CrossingPhase.SEARCHING
        self._crossing_green_frames = 0
        self._crossing_active_frames = 0
        self._crossing_lost_crosswalk_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        self._crossing_saw_near_crosswalk = False
        self._crossing_saw_mid_crosswalk = False
        self._crossing_started_at: float | None = None
        self._crossing_wait_signal_recent_frames = 0
        self._crossing_recent_obstacle_wait_frames = 0
        self._crossing_completion_announced = False
        self._crossing_timeout_announced = False
        self._crossing_last_crosswalk_bottom: float | None = None
        self._crossing_last_crosswalk_area: float | None = None
        self._crossing_last_crosswalk_center_y: float | None = None
        self._blind_path_road_stop_latched = False

    def command(self, text: str) -> NavigationResult:
        normalized = text.strip()
        speech: str | None = None
        if any(k in normalized for k in ("开始过马路", "帮我过马路")):
            self.mode = NavigationMode.CROSSING
            self._reset_guidance_debounce()
            speech = "过马路模式已启动。"
        elif any(k in normalized for k in ("过马路结束", "结束过马路", "停止过马路", "取消过马路")):
            speech = self._stop_current_mode()
        elif any(k in normalized for k in ("检测红绿灯", "看红绿灯")):
            self.mode = NavigationMode.TRAFFIC_LIGHT
            self._reset_guidance_debounce()
            speech = "红绿灯检测已启动。"
        elif any(k in normalized for k in ("停止检测", "取消检测", "停止红绿灯", "取消红绿灯")):
            speech = self._stop_current_mode()
        elif any(k in normalized for k in ("开始导航", "盲道导航", "帮我导航")):
            self.mode = NavigationMode.BLIND_PATH
            self._reset_guidance_debounce()
            speech = "盲道导航已启动。"
        elif any(k in normalized for k in ("停止导航", "结束导航", "取消导航")):
            speech = self._stop_current_mode()
        return NavigationResult(self.mode, speech=speech, state=self.snapshot())

    def process_observation(self, observation: dict[str, Any]) -> NavigationResult:
        speech = None
        overlay: dict[str, Any] = {"mode": self.mode.value, "observation": observation}
        if self.mode == NavigationMode.BLIND_PATH:
            speech = self._blind_path_guidance(observation)
        elif self.mode == NavigationMode.CROSSING:
            speech = self._crossing_guidance(observation)
        elif self.mode == NavigationMode.TRAFFIC_LIGHT:
            speech = self._traffic_light_guidance(observation)

        speech = self._debounced_speech(speech)
        return NavigationResult(self.mode, speech=speech, overlay=overlay, state=self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "last_speech": self.last_speech,
            "candidate_speech": self._candidate_speech,
            "candidate_frames": self._candidate_frames,
            "tuning": self.tuning.to_dict(),
            "crossing_phase": self._crossing_phase.value,
            "crossing_green_frames": self._crossing_green_frames,
            "crossing_active": self._crossing_phase == CrossingPhase.ACTIVE,
            "crossing_active_frames": self._crossing_active_frames,
            "crossing_lost_crosswalk_frames": self._crossing_lost_crosswalk_frames,
            "crossing_clear_path_frames": self._crossing_clear_path_frames,
            "crossing_completion_candidate_frames": self._crossing_completion_candidate_frames,
            "crossing_saw_near_crosswalk": self._crossing_saw_near_crosswalk,
            "crossing_saw_mid_crosswalk": self._crossing_saw_mid_crosswalk,
            "crossing_wait_signal_recent_frames": self._crossing_wait_signal_recent_frames,
            "crossing_recent_obstacle_wait_frames": self._crossing_recent_obstacle_wait_frames,
            "crossing_completion_announced": self._crossing_completion_announced,
            "crossing_timeout_announced": self._crossing_timeout_announced,
            "crossing_last_crosswalk_bottom": self._crossing_last_crosswalk_bottom,
            "crossing_last_crosswalk_area": self._crossing_last_crosswalk_area,
            "crossing_last_crosswalk_center_y": self._crossing_last_crosswalk_center_y,
        }

    def _debounced_speech(self, speech: str | None) -> str | None:
        if not speech:
            self._candidate_speech = ""
            self._candidate_frames = 0
            return None

        if speech == self._candidate_speech:
            self._candidate_frames += 1
        else:
            self._candidate_speech = speech
            self._candidate_frames = 1

        required_frames = 3 if self._is_detection_loss_speech(speech) else 1
        if self._candidate_frames < required_frames:
            return None
        if speech == self.last_speech:
            return None

        now = float(self._clock())
        if (
            not self._is_urgent_speech(speech)
            and self._last_guidance_spoken_at is not None
            and now - self._last_guidance_spoken_at < 2.0
        ):
            return None

        self.last_speech = speech
        self._last_guidance_spoken_at = now
        return speech

    def _reset_guidance_debounce(self) -> None:
        self.last_speech = ""
        self._candidate_speech = ""
        self._candidate_frames = 0
        self._last_guidance_spoken_at = None
        self._reset_crossing_progress()
        self._blind_path_road_stop_latched = False

    def _reset_crossing_progress(self) -> None:
        self._crossing_phase = CrossingPhase.SEARCHING
        self._crossing_green_frames = 0
        self._crossing_active_frames = 0
        self._crossing_lost_crosswalk_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        self._crossing_saw_near_crosswalk = False
        self._crossing_saw_mid_crosswalk = False
        self._crossing_started_at = None
        self._crossing_wait_signal_recent_frames = 0
        self._crossing_recent_obstacle_wait_frames = 0
        self._crossing_completion_announced = False
        self._crossing_timeout_announced = False
        self._crossing_last_crosswalk_bottom = None
        self._crossing_last_crosswalk_area = None
        self._crossing_last_crosswalk_center_y = None

    def _stop_current_mode(self) -> str:
        mode_to_stop = self.mode
        if mode_to_stop == NavigationMode.TRAFFIC_LIGHT:
            speech = "红绿灯检测已停止。"
        elif mode_to_stop == NavigationMode.CROSSING:
            speech = "过马路模式已停止。"
        elif mode_to_stop == NavigationMode.BLIND_PATH:
            speech = "盲道导航已停止。"
        else:
            speech = "当前没有正在进行的检测。"
        self.mode = NavigationMode.IDLE
        self._reset_guidance_debounce()
        return speech

    @staticmethod
    def _is_detection_loss_speech(speech: str) -> bool:
        return speech.startswith("没看到")

    @staticmethod
    def _is_urgent_speech(speech: str) -> bool:
        return (
            speech.startswith("前方有")
            or speech.startswith("前方疑似有")
            or speech.startswith("前方盲道上")
            or speech.startswith("绿灯，但斑马线附近")
            or speech.startswith("斑马线附近疑似有")
            or speech
            in {
                "红灯。",
                "绿灯。",
                "黄灯。",
                "前方发现路口。",
                "前方到马路了，请先停下。",
                "绿灯稳定，开始通行。",
                "疑似已通过人行横道，请确认安全后停止过马路模式。",
                "过马路时间较长，请确认周围安全，必要时停止过马路模式。",
            }
        )

    def _blind_path_guidance(self, obs: dict[str, Any]) -> str | None:
        blind = obs.get("blind_path")
        crosswalk_detection = obs.get("crosswalk_detection")
        crosswalk_detection_visible = self._is_blind_path_crosswalk_detection_visible(
            crosswalk_detection,
            obs,
        )
        crosswalk_detection_near_stop = self._is_blind_path_crosswalk_detection_near_stop(
            crosswalk_detection,
            obs,
        )
        if not crosswalk_detection_visible:
            self._blind_path_road_stop_latched = False
        if not blind:
            obstacle = self._find_centered_near_obstacle(obs)
            if obstacle:
                label = self._obstacle_speech_label(obstacle)
                return f"前方疑似有{label}，请先停下。"
            if crosswalk_detection_near_stop:
                self._blind_path_road_stop_latched = True
                return "前方到马路了，请先停下。"
            if self._blind_path_road_stop_latched and crosswalk_detection_visible:
                return "前方到马路了，请先停下。"
            if crosswalk_detection_visible:
                return "前方发现路口。"
            return "没看到盲道，请原地小幅转动。"
        obstacle = self._find_blind_path_obstacle(obs)
        if obstacle:
            return f"前方盲道上疑似有{self._obstacle_speech_label(obstacle)}，请先停下。"
        if crosswalk_detection_near_stop:
            self._blind_path_road_stop_latched = True
            return "前方到马路了，请先停下。"
        if self._blind_path_road_stop_latched and crosswalk_detection_visible:
            return "前方到马路了，请先停下。"
        if crosswalk_detection_visible:
            return "前方发现路口。"
        offset = self._float_value(blind.get("center_offset"), 0.0)
        angle = self._float_value(blind.get("angle_deg"), 0.0)
        if offset < -0.18:
            return "请向左微调，对准盲道。"
        if offset > 0.18:
            return "请向右微调，对准盲道。"
        if angle < -12:
            return "请向左转动。"
        if angle > 12:
            return "请向右转动。"
        return "保持直行。"

    def _is_blind_path_crosswalk_detection_visible(
        self,
        crosswalk_detection: Any,
        obs: dict[str, Any],
    ) -> bool:
        box = self._blind_path_crosswalk_detection_box(crosswalk_detection, obs)
        return box is not None and box[3] >= self.tuning.crosswalk_detection_alert_bottom_min

    def _is_blind_path_crosswalk_detection_near_stop(
        self,
        crosswalk_detection: Any,
        obs: dict[str, Any],
    ) -> bool:
        box = self._blind_path_crosswalk_detection_box(crosswalk_detection, obs)
        return box is not None and box[3] >= self.tuning.crosswalk_detection_stop_bottom_min

    def _blind_path_crosswalk_detection_box(
        self,
        crosswalk_detection: Any,
        obs: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        if not isinstance(crosswalk_detection, dict):
            return None
        if crosswalk_detection.get("label") != "crossing":
            return None
        confidence = self._float_value(crosswalk_detection.get("confidence"), 0.0)
        if confidence < self.tuning.crosswalk_detection_conf:
            return None
        box = self._normalised_box(crosswalk_detection, obs)
        if box is None:
            return None
        if self._normalised_box_area(box) < self.tuning.crosswalk_detection_min_area_ratio:
            return None
        x1, _, x2, _ = box
        center_x = (x1 + x2) / 2.0
        if not (
            self.tuning.crosswalk_detection_x_min
            <= center_x
            <= self.tuning.crosswalk_detection_x_max
        ):
            return None
        return box

    @staticmethod
    def _obstacle_speech_label(obstacle: dict[str, Any]) -> str:
        label = str(obstacle.get("label") or "").strip()
        return OBSTACLE_SPEECH_LABELS.get(label, label or "障碍物")

    def _find_blind_path_obstacle(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        blind = obs.get("blind_path")
        if not isinstance(blind, dict):
            return None
        path_center = self._mask_center_offset(blind)
        candidates = [
            obstacle
            for obstacle in self._obstacles(obs)
            if self._is_blind_path_obstacle(obstacle, obs, path_center)
        ]
        return max(candidates, key=self._obstacle_priority, default=None)

    def _find_centered_near_obstacle(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        candidates = [
            obstacle
            for obstacle in self._obstacles(obs)
            if self._is_centered_near_obstacle(obstacle, obs)
        ]
        return max(candidates, key=self._obstacle_priority, default=None)

    def _is_centered_near_obstacle(self, obstacle: dict[str, Any], obs: dict[str, Any]) -> bool:
        if self._float_value(obstacle.get("confidence"), 0.0) < BLIND_PATH_MIN_CONFIDENCE:
            return False
        if self._float_value(obstacle.get("area_ratio"), 0.0) < BLIND_PATH_NEAR_AREA_RATIO:
            return False
        if not self._is_obstacle_ahead(
            obstacle,
            obs,
            min_bottom=BLIND_PATH_AHEAD_BOTTOM_MIN,
            close_bottom=BLIND_PATH_CLOSE_BOTTOM_MIN,
            near_area=BLIND_PATH_NEAR_AREA_RATIO,
        ):
            return False
        return self._is_obstacle_in_corridor(
            obstacle,
            obs,
            center_offset=0.0,
            half_width=BLIND_PATH_STRICT_HALF_WIDTH,
            centerline_margin=0.10,
        )

    def _is_blind_path_obstacle(
        self,
        obstacle: dict[str, Any],
        obs: dict[str, Any],
        path_center: float,
    ) -> bool:
        if self._float_value(obstacle.get("confidence"), 0.0) < BLIND_PATH_MIN_CONFIDENCE:
            return False
        area_ratio = self._float_value(obstacle.get("area_ratio"), 0.0)
        if area_ratio < BLIND_PATH_MIN_AREA_RATIO:
            return False
        if not self._is_obstacle_ahead(
            obstacle,
            obs,
            min_bottom=BLIND_PATH_AHEAD_BOTTOM_MIN,
            close_bottom=BLIND_PATH_CLOSE_BOTTOM_MIN,
            near_area=BLIND_PATH_NEAR_AREA_RATIO,
        ):
            return False
        return self._is_obstacle_in_corridor(
            obstacle,
            obs,
            center_offset=path_center,
            half_width=(
                BLIND_PATH_STRICT_HALF_WIDTH
                if area_ratio < BLIND_PATH_NEAR_AREA_RATIO
                else BLIND_PATH_CORRIDOR_HALF_WIDTH
            ),
            centerline_margin=0.12,
        )

    @staticmethod
    def _obstacles(obs: dict[str, Any]) -> list[dict[str, Any]]:
        obstacles = obs.get("obstacles")
        if isinstance(obstacles, list):
            return [item for item in obstacles if isinstance(item, dict)]
        nearest = obs.get("nearest_obstacle")
        return [nearest] if isinstance(nearest, dict) else []

    @staticmethod
    def _mask_center_offset(mask: Any) -> float:
        if isinstance(mask, dict):
            return NavigationStateMachine._float_value(mask.get("center_offset"), 0.0)
        return 0.0

    def _is_obstacle_ahead(
        self,
        obstacle: dict[str, Any],
        obs: dict[str, Any],
        *,
        min_bottom: float,
        close_bottom: float,
        near_area: float,
    ) -> bool:
        bottom = self._box_bottom_ratio(obstacle, obs)
        area_ratio = self._float_value(obstacle.get("area_ratio"), 0.0)
        return bottom >= (min_bottom if area_ratio >= near_area else close_bottom)

    def _is_obstacle_in_corridor(
        self,
        obstacle: dict[str, Any],
        obs: dict[str, Any],
        *,
        center_offset: float,
        half_width: float,
        centerline_margin: float,
    ) -> bool:
        box = self._normalised_box(obstacle, obs)
        if box is None:
            return False
        x1, _, x2, _ = box
        left = self._x_to_offset(x1)
        right = self._x_to_offset(x2)
        center = self._x_to_offset((x1 + x2) / 2.0)
        center_aligned = abs(center - center_offset) <= half_width
        overlaps_centerline = (
            left <= center_offset + centerline_margin
            and right >= center_offset - centerline_margin
        )
        return center_aligned or overlaps_centerline

    def _box_bottom_ratio(self, obstacle: dict[str, Any], obs: dict[str, Any]) -> float:
        box = self._normalised_box(obstacle, obs)
        return box[3] if box is not None else 0.0

    @staticmethod
    def _x_to_offset(x: float) -> float:
        return max(-1.0, min(1.0, (x - 0.5) * 2.0))

    @staticmethod
    def _normalised_box(
        obstacle: dict[str, Any],
        obs: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        raw_box = obstacle.get("box")
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_box)
        except (TypeError, ValueError):
            return None
        width = NavigationStateMachine._float_value(obs.get("frame_width"), 0.0)
        height = NavigationStateMachine._float_value(obs.get("frame_height"), 0.0)
        if width > 1.0 and max(abs(x1), abs(x2)) > 1.0:
            x1 /= width
            x2 /= width
        if height > 1.0 and max(abs(y1), abs(y2)) > 1.0:
            y1 /= height
            y2 /= height
        return (
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )

    @staticmethod
    def _normalised_box_area(box: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _float_value(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _obstacle_priority(self, obstacle: dict[str, Any]) -> tuple[float, float, float]:
        box = obstacle.get("box")
        bottom = (
            self._float_value(box[3], 0.0)
            if isinstance(box, (list, tuple)) and len(box) == 4
            else 0.0
        )
        return (
            self._float_value(obstacle.get("area_ratio"), 0.0),
            self._float_value(obstacle.get("confidence"), 0.0),
            bottom,
        )

    def _crossing_guidance(self, obs: dict[str, Any]) -> str | None:
        has_pure_go, has_wait_caution, wait_label = self._crossing_signal_state(obs)
        self._update_crossing_wait_signal_window(has_wait_caution)
        crosswalk = self._crossing_crosswalk_info(obs)
        obstacle_hazard = (
            self._find_crossing_obstacle_hazard(obs, crosswalk)
            if self.tuning.crossing_obstacles_enabled
            else None
        )
        self._update_crossing_obstacle_window(obstacle_hazard is not None)
        ctx = CrossingFrameContext(
            has_pure_go=has_pure_go,
            has_wait_caution=has_wait_caution,
            wait_label=wait_label,
            crosswalk=crosswalk,
            obstacle_hazard=obstacle_hazard,
        )

        event_speech, entered_ready = self._advance_crossing_phase(ctx)
        if event_speech is not None:
            return event_speech
        self._update_crossing_ready_green_frames(ctx, entered_ready=entered_ready)
        return self._crossing_continuous_speech(ctx)

    def _crossing_signal_state(self, obs: dict[str, Any]) -> tuple[bool, bool, str | None]:
        labels = self._traffic_candidate_labels(obs)
        has_wait_caution = any(label in CROSSING_WAIT_CAUTION_LABELS for label in labels)
        has_go = any(label in CROSSING_GO_LABELS for label in labels)
        wait_label = next((label for label in labels if label in CROSSING_WAIT_CAUTION_LABELS), None)
        return has_go and not has_wait_caution, has_wait_caution, wait_label

    def _traffic_candidate_labels(self, obs: dict[str, Any]) -> list[str]:
        debug = obs.get("traffic_light_debug")
        candidate_labels = self._traffic_detection_labels(
            obs.get("traffic_light_candidates"),
            apply_label_thresholds=True,
        )
        if isinstance(debug, dict):
            filtered_labels = self._traffic_detection_labels(
                debug.get("filtered_candidates"),
                apply_label_thresholds=True,
            )
            if filtered_labels:
                # Thresholded wait/caution candidates are conservative blockers, but a filtered
                # go only counts after the traffic selector has accepted it as selected.
                if any(label in CROSSING_WAIT_CAUTION_LABELS for label in filtered_labels):
                    return filtered_labels
                selected_label, _has_debug_selection = self._traffic_debug_selected_label(obs)
                return [selected_label] if selected_label is not None else []
            if any(label in CROSSING_WAIT_CAUTION_LABELS for label in candidate_labels):
                return candidate_labels

        selected_label, has_debug_selection = self._traffic_debug_selected_label(obs)
        if selected_label is not None:
            return [selected_label]
        if has_debug_selection:
            return []
        if candidate_labels:
            return candidate_labels
        fallback = obs.get("traffic_light")
        return [str(fallback)] if fallback else []

    def _traffic_detection_labels(
        self,
        candidates: Any,
        *,
        apply_label_thresholds: bool = False,
    ) -> list[str]:
        if not isinstance(candidates, list):
            return []
        labels: list[str] = []
        for candidate in candidates:
            label = self._detection_label(candidate)
            if not label:
                continue
            if apply_label_thresholds and not self._detection_passes_signal_threshold(
                candidate,
                label,
            ):
                continue
            labels.append(label)
        return labels

    @staticmethod
    def _detection_label(candidate: Any) -> str:
        if isinstance(candidate, dict):
            return str(candidate.get("label") or "")
        return str(getattr(candidate, "label", "") or "")

    def _detection_passes_signal_threshold(self, candidate: Any, label: str) -> bool:
        threshold = self._signal_label_threshold(label)
        if threshold is None:
            return True
        confidence = (
            candidate.get("confidence")
            if isinstance(candidate, dict)
            else getattr(candidate, "confidence", None)
        )
        return self._float_value(confidence, 0.0) >= threshold

    def _signal_label_threshold(self, label: str) -> float | None:
        if label == "go":
            return self.tuning.traffic_go_min_conf
        if label == "stop":
            return self.tuning.traffic_stop_min_conf
        if label in {"countdown_go", "countdown_stop", "yellow"}:
            return self.tuning.traffic_yellow_min_conf
        return None

    def _traffic_debug_selected_label(self, obs: dict[str, Any]) -> tuple[str | None, bool]:
        debug = obs.get("traffic_light_debug")
        if not isinstance(debug, dict) or "selected" not in debug:
            return None, False
        selected = debug.get("selected")
        if not isinstance(selected, dict):
            return None, True
        label = str(selected.get("label") or "")
        if label in CROSSING_WAIT_CAUTION_LABELS:
            return label, True
        if label and not self._detection_passes_signal_threshold(selected, label):
            return None, True
        return (label or None), True

    def _update_crossing_wait_signal_window(self, has_wait_caution: bool) -> None:
        if has_wait_caution:
            self._crossing_wait_signal_recent_frames = (
                self.tuning.crossing_wait_signal_suppress_frames
            )
            return
        self._crossing_wait_signal_recent_frames = max(
            0,
            self._crossing_wait_signal_recent_frames - 1,
        )

    def _update_crossing_obstacle_window(self, has_obstacle_wait: bool) -> None:
        if has_obstacle_wait:
            self._crossing_recent_obstacle_wait_frames = (
                self.tuning.crossing_obstacle_suppress_frames
            )
            return
        self._crossing_recent_obstacle_wait_frames = max(
            0,
            self._crossing_recent_obstacle_wait_frames - 1,
        )

    def _advance_crossing_phase(self, ctx: CrossingFrameContext) -> tuple[str | None, bool]:
        entered_ready = False
        if self._crossing_phase == CrossingPhase.SEARCHING:
            if ctx.crosswalk is not None:
                self._crossing_phase = CrossingPhase.ALIGNING
            return None, entered_ready

        if self._crossing_phase == CrossingPhase.ALIGNING:
            if ctx.crosswalk is None:
                self._crossing_phase = CrossingPhase.SEARCHING
                self._crossing_green_frames = 0
                return None, entered_ready
            if (
                ctx.crosswalk.aligned
                and ctx.crosswalk.bottom >= self.tuning.crosswalk_detection_stop_bottom_min
                and self._crossing_wait_signal_recent_frames == 0
            ):
                self._crossing_phase = CrossingPhase.READY
                self._crossing_green_frames = 0
                entered_ready = True
            return None, entered_ready

        if self._crossing_phase == CrossingPhase.READY:
            if ctx.crosswalk is None:
                self._crossing_phase = CrossingPhase.SEARCHING
                self._crossing_green_frames = 0
                return None, entered_ready
            if not ctx.crosswalk.aligned:
                self._crossing_phase = CrossingPhase.ALIGNING
                self._crossing_green_frames = 0
                return None, entered_ready
            if self._ready_crossing_can_start(ctx):
                self._start_active_crossing(ctx.crosswalk)
                return "绿灯稳定，开始通行。", entered_ready
            return None, entered_ready

        if self._crossing_phase == CrossingPhase.ACTIVE:
            return self._advance_active_crossing(ctx), entered_ready

        return None, entered_ready

    def _ready_crossing_can_start(self, ctx: CrossingFrameContext) -> bool:
        if ctx.crosswalk is None:
            return False
        return (
            ctx.has_pure_go
            and self._crossing_green_frames >= self.tuning.crossing_green_required_frames
            and self._crossing_wait_signal_recent_frames == 0
            and self._crossing_recent_obstacle_wait_frames == 0
            and ctx.obstacle_hazard is None
            and ctx.crosswalk.aligned
            and ctx.crosswalk.bottom >= self.tuning.crossing_start_bottom_min
            and ctx.crosswalk.area >= self.tuning.crosswalk_detection_min_area_ratio
        )

    def _update_crossing_ready_green_frames(
        self,
        ctx: CrossingFrameContext,
        *,
        entered_ready: bool,
    ) -> None:
        if self._crossing_phase != CrossingPhase.READY:
            self._crossing_green_frames = 0
            return
        if entered_ready:
            self._crossing_green_frames = 0
            return
        if (
            ctx.has_pure_go
            and self._crossing_wait_signal_recent_frames == 0
            and self._crossing_recent_obstacle_wait_frames == 0
            and ctx.obstacle_hazard is None
        ):
            self._crossing_green_frames += 1
        else:
            self._crossing_green_frames = 0

    def _start_active_crossing(self, crosswalk: CrossingDetectionInfo) -> None:
        self._crossing_phase = CrossingPhase.ACTIVE
        self._crossing_started_at = float(self._clock())
        self._crossing_active_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        self._crossing_completion_announced = False
        self._crossing_timeout_announced = False
        self._crossing_green_frames = 0
        self._crossing_saw_near_crosswalk = True
        self._crossing_saw_mid_crosswalk = (
            crosswalk.bottom >= self.tuning.crossing_mid_bottom_min
        )
        self._crossing_lost_crosswalk_frames = 0
        self._remember_crosswalk(crosswalk)

    def _advance_active_crossing(self, ctx: CrossingFrameContext) -> str | None:
        self._crossing_active_frames += 1
        if ctx.has_wait_caution or ctx.obstacle_hazard is not None:
            self._crossing_clear_path_frames = 0
        else:
            self._crossing_clear_path_frames += 1
        self._update_active_crosswalk_history(ctx.crosswalk)

        if self._is_crossing_completion_candidate(ctx):
            self._crossing_completion_candidate_frames += 1
        else:
            self._crossing_completion_candidate_frames = 0

        if self._crossing_completed():
            self._crossing_completion_candidate_frames = 0
            self._crossing_completion_announced = True
            self._crossing_phase = CrossingPhase.SUSPECTED_COMPLETED
            return "疑似已通过人行横道，请确认安全后停止过马路模式。"
        return None

    def _update_active_crosswalk_history(self, crosswalk: CrossingDetectionInfo | None) -> None:
        if crosswalk is None:
            self._crossing_lost_crosswalk_frames += 1
            return
        self._crossing_lost_crosswalk_frames = 0
        self._remember_crosswalk(crosswalk)
        if crosswalk.bottom >= self.tuning.crossing_start_bottom_min:
            self._crossing_saw_near_crosswalk = True
        if crosswalk.bottom >= self.tuning.crossing_mid_bottom_min:
            self._crossing_saw_mid_crosswalk = True

    def _remember_crosswalk(self, crosswalk: CrossingDetectionInfo) -> None:
        self._crossing_last_crosswalk_bottom = crosswalk.bottom
        self._crossing_last_crosswalk_area = crosswalk.area
        self._crossing_last_crosswalk_center_y = crosswalk.center_y

    def _is_crossing_completion_candidate(self, ctx: CrossingFrameContext) -> bool:
        crosswalk = ctx.crosswalk
        if (
            self._crossing_phase != CrossingPhase.ACTIVE
            or self._crossing_completion_announced
            or not self._crossing_saw_near_crosswalk
            or not self._crossing_saw_mid_crosswalk
            or self._crossing_active_frames < self.tuning.crossing_completion_min_active_frames
            or self._crossing_active_seconds() < self.tuning.crossing_completion_min_active_seconds
            or self._crossing_timed_out()
            or self._crossing_wait_signal_recent_frames > 0
            or self._crossing_recent_obstacle_wait_frames > 0
            or ctx.has_wait_caution
            or ctx.obstacle_hazard is not None
        ):
            return False
        if crosswalk is not None and crosswalk.bottom >= self.tuning.crossing_mid_bottom_min:
            return False
        if crosswalk is not None:
            return crosswalk.bottom <= self.tuning.crossing_completion_bottom_max
        return self._crossing_lost_crosswalk_frames >= self.tuning.crossing_completion_lost_frames

    def _crossing_crosswalk_info(self, obs: dict[str, Any]) -> CrossingDetectionInfo | None:
        crosswalk = obs.get("crosswalk_detection")
        if not isinstance(crosswalk, dict):
            return None
        if crosswalk.get("label") != "crossing":
            return None
        if (
            self._float_value(crosswalk.get("confidence"), 0.0)
            < self.tuning.crosswalk_detection_conf
        ):
            return None
        box = self._normalised_box(crosswalk, obs)
        if box is None:
            return None
        area = self._normalised_box_area(box)
        if area < self.tuning.crosswalk_detection_min_area_ratio:
            return None
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        if not (
            self.tuning.crosswalk_detection_x_min
            <= center_x
            <= self.tuning.crosswalk_detection_x_max
        ):
            return None
        center_offset = self._x_to_offset(center_x)
        return CrossingDetectionInfo(
            box=box,
            area=area,
            bottom=y2,
            center_y=(y1 + y2) / 2.0,
            center_offset=center_offset,
            aligned=abs(center_offset) <= self.tuning.crossing_alignment_offset_max,
        )

    def _crossing_completed(self) -> bool:
        return (
            self._crossing_completion_candidate_frames
            >= self.tuning.crossing_completion_required_frames
        )

    def _crossing_active_seconds(self) -> float:
        if self._crossing_started_at is None:
            return 0.0
        return max(0.0, float(self._clock()) - self._crossing_started_at)

    def _crossing_timed_out(self) -> bool:
        return self._crossing_active_seconds() >= self.tuning.crossing_active_timeout_seconds

    def _crossing_continuous_speech(self, ctx: CrossingFrameContext) -> str | None:
        if ctx.has_wait_caution:
            return "红灯。" if ctx.wait_label == "stop" else "黄灯。"
        if ctx.obstacle_hazard is not None:
            light = "go" if ctx.has_pure_go else None
            return self._crossing_obstacle_wait_message(ctx.obstacle_hazard, light)
        if self._crossing_phase == CrossingPhase.ALIGNING and ctx.crosswalk is not None:
            if ctx.crosswalk.center_offset < -self.tuning.crossing_alignment_offset_max:
                return "请向左转动。"
            if ctx.crosswalk.center_offset > self.tuning.crossing_alignment_offset_max:
                return "请向右转动。"
            if ctx.crosswalk.bottom >= self.tuning.crosswalk_detection_stop_bottom_min:
                return None
            return "发现斑马线，对准方向。"
        if self._crossing_phase == CrossingPhase.SEARCHING and ctx.crosswalk is None:
            return "没看到斑马线，请原地小幅转动。"
        if (
            self._crossing_phase == CrossingPhase.ACTIVE
            and not self._crossing_timeout_announced
            and self._crossing_timed_out()
        ):
            self._crossing_timeout_announced = True
            return "过马路时间较长，请确认周围安全，必要时停止过马路模式。"
        return None

    def _crossing_obstacle_wait_message(self, obstacle: dict[str, Any], light: Any) -> str:
        label = self._obstacle_speech_label(obstacle)
        if light == "go":
            return f"绿灯，但斑马线附近疑似有{label}，请先等待，确认安全后再过街。"
        return f"斑马线附近疑似有{label}，请先等待。"

    def _find_crossing_obstacle_hazard(
        self,
        obs: dict[str, Any],
        crosswalk: CrossingDetectionInfo | None = None,
    ) -> dict[str, Any] | None:
        center = (
            crosswalk.center_offset
            if crosswalk is not None
            else self._mask_center_offset(obs.get("blind_path"))
        )
        candidates = [
            obstacle
            for obstacle in self._obstacles(obs)
            if self._is_crossing_obstacle_hazard(obstacle, obs, center)
        ]
        return max(candidates, key=self._obstacle_priority, default=None)

    def _is_crossing_obstacle_hazard(
        self,
        obstacle: dict[str, Any],
        obs: dict[str, Any],
        center_offset: float,
    ) -> bool:
        if self._float_value(obstacle.get("confidence"), 0.0) < CROSSING_MIN_CONFIDENCE:
            return False
        if self._float_value(obstacle.get("area_ratio"), 0.0) < CROSSING_MIN_AREA_RATIO:
            return False
        if self._box_bottom_ratio(obstacle, obs) < CROSSING_AHEAD_BOTTOM_MIN:
            return False
        return self._is_obstacle_in_corridor(
            obstacle,
            obs,
            center_offset=center_offset,
            half_width=CROSSING_CORRIDOR_HALF_WIDTH,
            centerline_margin=0.18,
        )

    def _traffic_light_guidance(self, obs: dict[str, Any]) -> str | None:
        light = obs.get("traffic_light")
        if light == "go":
            return "绿灯。"
        if light == "stop":
            return "红灯。"
        if light in {"countdown_go", "countdown_stop"}:
            return "黄灯。"
        return None
