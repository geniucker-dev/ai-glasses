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
CROSSING_GREEN_REQUIRED_FRAMES = 2
CROSSING_COMPLETION_MIN_ACTIVE_FRAMES = 4
CROSSING_COMPLETION_BOTTOM_MAX = 0.35
CROSSING_COMPLETION_MAX_AREA_RATIO = 0.08
CROSSING_COMPLETION_REQUIRED_FRAMES = 3
CROSSING_PROGRESS_BOTTOM_MIN = 0.70
CROSSING_PROGRESS_MIN_AREA_RATIO = 0.12
CROSSING_MAX_ACTIVE_SECONDS = 45.0

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
        self._crossing_green_frames = 0
        self._crossing_active = False
        self._crossing_active_frames = 0
        self._crossing_lost_crosswalk_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        self._crossing_saw_near_crosswalk = False
        self._crossing_started_at: float | None = None
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
            "crossing_green_frames": self._crossing_green_frames,
            "crossing_active": self._crossing_active,
            "crossing_active_frames": self._crossing_active_frames,
            "crossing_lost_crosswalk_frames": self._crossing_lost_crosswalk_frames,
            "crossing_clear_path_frames": self._crossing_clear_path_frames,
            "crossing_completion_candidate_frames": self._crossing_completion_candidate_frames,
            "crossing_saw_near_crosswalk": self._crossing_saw_near_crosswalk,
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
        self._crossing_green_frames = 0
        self._crossing_active = False
        self._crossing_active_frames = 0
        self._crossing_lost_crosswalk_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        self._crossing_saw_near_crosswalk = False
        self._crossing_started_at = None

    def _pause_crossing_completion(self, *, reset_progress: bool = True) -> None:
        self._crossing_green_frames = 0
        self._crossing_lost_crosswalk_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        if reset_progress:
            self._crossing_saw_near_crosswalk = False

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
                "前方发现路口，请注意红绿灯。",
                "前方到马路了，请先停下。",
                "绿灯稳定，开始通行。",
                "疑似已通过人行横道，请确认安全后停止过马路模式。",
                "过马路时间较长，请确认周围安全，必要时停止过马路模式。",
            }
        )

    def _blind_path_guidance(self, obs: dict[str, Any]) -> str | None:
        blind = obs.get("blind_path")
        crosswalk = obs.get("crosswalk")
        crosswalk_visible = self._is_blind_path_crosswalk_visible(crosswalk)
        crosswalk_near_stop = self._is_blind_path_crosswalk_near_stop(crosswalk)
        if not crosswalk_visible:
            self._blind_path_road_stop_latched = False
        if not blind:
            obstacle = self._find_centered_near_obstacle(obs)
            if obstacle:
                label = self._obstacle_speech_label(obstacle)
                return f"前方疑似有{label}，请先停下。"
            if crosswalk_near_stop:
                self._blind_path_road_stop_latched = True
                return "前方到马路了，请先停下。"
            if self._blind_path_road_stop_latched and crosswalk_visible:
                return "前方到马路了，请先停下。"
            if crosswalk_visible:
                return "前方发现路口，请注意红绿灯。"
            return "没看到盲道，请原地小幅转动。"
        obstacle = self._find_blind_path_obstacle(obs)
        if obstacle:
            return f"前方盲道上疑似有{self._obstacle_speech_label(obstacle)}，请先停下。"
        if crosswalk_near_stop:
            self._blind_path_road_stop_latched = True
            return "前方到马路了，请先停下。"
        if self._blind_path_road_stop_latched and crosswalk_visible:
            return "前方到马路了，请先停下。"
        if crosswalk_visible:
            return "前方发现路口，请注意红绿灯。"
        offset = float(blind.get("center_offset", 0.0))
        angle = float(blind.get("angle_deg", 0.0))
        if offset < -0.18:
            return "请向左微调，对准盲道。"
        if offset > 0.18:
            return "请向右微调，对准盲道。"
        if angle < -12:
            return "请向左转动。"
        if angle > 12:
            return "请向右转动。"
        return "保持直行。"

    def _is_blind_path_crosswalk_visible(self, crosswalk: Any) -> bool:
        if not isinstance(crosswalk, dict):
            return False
        area_ratio = self._mask_area_ratio(crosswalk)
        return area_ratio is None or area_ratio >= self.tuning.road_alert_area_ratio

    def _is_blind_path_crosswalk_near_stop(self, crosswalk: Any) -> bool:
        if not isinstance(crosswalk, dict):
            return False
        area_ratio = self._mask_area_ratio(crosswalk)
        if area_ratio is None or area_ratio < self.tuning.road_stop_area_ratio:
            return False
        bottom = self._mask_bottom(crosswalk)
        return bottom is not None and bottom >= self.tuning.road_stop_bottom_min

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
        if float(obstacle.get("confidence", 0.0)) < BLIND_PATH_MIN_CONFIDENCE:
            return False
        if float(obstacle.get("area_ratio", 0.0)) < BLIND_PATH_NEAR_AREA_RATIO:
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
        if float(obstacle.get("confidence", 0.0)) < BLIND_PATH_MIN_CONFIDENCE:
            return False
        area_ratio = float(obstacle.get("area_ratio", 0.0))
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
            return float(mask.get("center_offset", 0.0))
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
        area_ratio = float(obstacle.get("area_ratio", 0.0))
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
        overlaps_centerline = left <= center_offset + centerline_margin and right >= center_offset - centerline_margin
        return center_aligned or overlaps_centerline

    def _box_bottom_ratio(self, obstacle: dict[str, Any], obs: dict[str, Any]) -> float:
        box = self._normalised_box(obstacle, obs)
        return float(box[3]) if box is not None else 0.0

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
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        width = float(obs.get("frame_width") or 0.0)
        height = float(obs.get("frame_height") or 0.0)
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

    def _obstacle_priority(self, obstacle: dict[str, Any]) -> tuple[float, float, float]:
        box = obstacle.get("box")
        bottom = float(box[3]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0.0
        return (
            float(obstacle.get("area_ratio", 0.0)),
            float(obstacle.get("confidence", 0.0)),
            bottom,
        )

    def _crossing_guidance(self, obs: dict[str, Any]) -> str | None:
        crosswalk = obs.get("crosswalk")
        light = obs.get("traffic_light")
        obstacle_hazard = (
            self._find_crossing_obstacle_hazard(obs)
            if self.tuning.crossing_obstacles_enabled
            else None
        )

        if light == "stop":
            self._pause_crossing_completion(reset_progress=not self._crossing_active)
            return "红灯。"
        if light in {"countdown_go", "countdown_stop"}:
            self._pause_crossing_completion(reset_progress=not self._crossing_active)
            return "黄灯。"
        if obstacle_hazard:
            self._pause_crossing_completion(reset_progress=not self._crossing_active)
            return self._crossing_obstacle_wait_message(obstacle_hazard, light)

        if not self._crossing_active:
            return self._pre_crossing_guidance(crosswalk, light)
        return self._active_crossing_guidance(crosswalk, light)

    def _pre_crossing_guidance(self, crosswalk: Any, light: Any) -> str | None:
        if light != "go":
            self._crossing_green_frames = 0
        if not crosswalk:
            self._crossing_green_frames = 0
            return "没看到斑马线，请原地小幅转动。"
        if light == "go":
            if self._is_crossing_progress_evidence(crosswalk):
                self._crossing_green_frames += 1
                if self._crossing_green_frames >= self.tuning.crossing_green_required_frames:
                    self._start_active_crossing(crosswalk)
                    return "绿灯稳定，开始通行。"
                return None
            self._crossing_green_frames = 0
        offset = float(crosswalk.get("center_offset", 0.0))
        if offset < -0.15:
            return "请向左转动。"
        if offset > 0.15:
            return "请向右转动。"
        return "发现斑马线，对准方向。"

    def _start_active_crossing(self, crosswalk: Any) -> None:
        self._crossing_active = True
        self._crossing_active_frames = 0
        self._crossing_lost_crosswalk_frames = 0
        self._crossing_clear_path_frames = 0
        self._crossing_completion_candidate_frames = 0
        self._crossing_saw_near_crosswalk = self._is_crossing_progress_evidence(crosswalk)
        self._crossing_started_at = float(self._clock())

    def _active_crossing_guidance(self, crosswalk: Any, light: Any) -> str | None:
        self._crossing_green_frames = self.tuning.crossing_green_required_frames
        self._crossing_active_frames += 1
        self._crossing_clear_path_frames += 1
        if crosswalk:
            self._crossing_lost_crosswalk_frames = 0
        else:
            self._crossing_lost_crosswalk_frames += 1

        if self._is_crossing_progress_evidence(crosswalk):
            self._crossing_saw_near_crosswalk = True

        if self._crossing_timed_out():
            self._crossing_completion_candidate_frames = 0
            return "过马路时间较长，请确认周围安全，必要时停止过马路模式。"
        if self._is_crossing_completion_candidate(crosswalk, light):
            self._crossing_completion_candidate_frames += 1
        else:
            self._crossing_completion_candidate_frames = 0

        if self._crossing_completed():
            self._crossing_completion_candidate_frames = 0
            return "疑似已通过人行横道，请确认安全后停止过马路模式。"
        return None

    def _is_crossing_progress_evidence(self, crosswalk: Any) -> bool:
        if not isinstance(crosswalk, dict):
            return False
        bottom = self._mask_bottom(crosswalk)
        if bottom is None or bottom < CROSSING_PROGRESS_BOTTOM_MIN:
            return False
        area_ratio = self._mask_area_ratio(crosswalk)
        return area_ratio is not None and area_ratio >= CROSSING_PROGRESS_MIN_AREA_RATIO

    def _is_crossing_completion_candidate(self, crosswalk: Any, light: Any) -> bool:
        if light in {"stop", "countdown_go", "countdown_stop"} or not self._crossing_saw_near_crosswalk:
            return False
        if not isinstance(crosswalk, dict):
            return False
        if self._crossing_active_frames < CROSSING_COMPLETION_MIN_ACTIVE_FRAMES:
            return False
        bottom = self._mask_bottom(crosswalk)
        if bottom is None or bottom > CROSSING_COMPLETION_BOTTOM_MAX:
            return False
        area_ratio = self._mask_area_ratio(crosswalk)
        return area_ratio is not None and area_ratio <= CROSSING_COMPLETION_MAX_AREA_RATIO

    @staticmethod
    def _mask_area_ratio(mask: dict[str, Any]) -> float | None:
        try:
            return float(mask["area_ratio"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _mask_bottom(mask: dict[str, Any]) -> float | None:
        contour = mask.get("contour")
        if not isinstance(contour, list) or not contour:
            return None
        ys = [
            float(point[1])
            for point in contour
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        return max(ys, default=None)

    def _crossing_completed(self) -> bool:
        return self._crossing_completion_candidate_frames >= CROSSING_COMPLETION_REQUIRED_FRAMES

    def _crossing_timed_out(self) -> bool:
        if self._crossing_started_at is None:
            return False
        return float(self._clock()) - self._crossing_started_at >= CROSSING_MAX_ACTIVE_SECONDS

    def _crossing_obstacle_wait_message(self, obstacle: dict[str, Any], light: Any) -> str:
        label = self._obstacle_speech_label(obstacle)
        if light == "go":
            return f"绿灯，但斑马线附近疑似有{label}，请先等待，确认安全后再过街。"
        return f"斑马线附近疑似有{label}，请先等待。"

    def _find_crossing_obstacle_hazard(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        center = self._mask_center_offset(obs.get("crosswalk") or obs.get("blind_path"))
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
        if float(obstacle.get("confidence", 0.0)) < CROSSING_MIN_CONFIDENCE:
            return False
        if float(obstacle.get("area_ratio", 0.0)) < CROSSING_MIN_AREA_RATIO:
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
