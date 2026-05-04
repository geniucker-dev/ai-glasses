from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import time
from typing import Any, Callable


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


@dataclass(frozen=True)
class NavigationResult:
    mode: NavigationMode
    speech: str | None = None
    overlay: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


class NavigationStateMachine:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.mode = NavigationMode.IDLE
        self.last_speech = ""
        self._clock = clock
        self._candidate_speech = ""
        self._candidate_frames = 0
        self._last_guidance_spoken_at: float | None = None

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
        elif any(k in normalized for k in ("继续", "立即通过", "现在通过")):
            speech = "收到。"
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
        return speech.startswith("前方有") or speech in {
            "红灯。",
            "绿灯。",
            "黄灯。",
            "绿灯稳定，开始通行。",
        }

    def _blind_path_guidance(self, obs: dict[str, Any]) -> str | None:
        obstacle = obs.get("nearest_obstacle")
        if obstacle:
            return f"前方有{self._obstacle_speech_label(obstacle)}，停一下。"
        blind = obs.get("blind_path")
        if not blind:
            return "没看到盲道，请原地小幅转动。"
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

    @staticmethod
    def _obstacle_speech_label(obstacle: dict[str, Any]) -> str:
        label = str(obstacle.get("label") or "").strip()
        return OBSTACLE_SPEECH_LABELS.get(label, label or "障碍物")

    def _crossing_guidance(self, obs: dict[str, Any]) -> str | None:
        crosswalk = obs.get("crosswalk")
        light = obs.get("traffic_light")
        if not crosswalk:
            return "没看到斑马线，请原地小幅转动。"
        if light == "stop":
            return "红灯。"
        if light == "go":
            return "绿灯稳定，开始通行。"
        offset = float(crosswalk.get("center_offset", 0.0))
        if offset < -0.15:
            return "请向左转动。"
        if offset > 0.15:
            return "请向右转动。"
        return "发现斑马线，对准方向。"

    def _traffic_light_guidance(self, obs: dict[str, Any]) -> str | None:
        light = obs.get("traffic_light")
        if light == "go":
            return "绿灯。"
        if light == "stop":
            return "红灯。"
        if light in {"countdown_go", "countdown_stop"}:
            return "黄灯。"
        return None
