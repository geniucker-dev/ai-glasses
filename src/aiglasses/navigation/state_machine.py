from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class NavigationMode(StrEnum):
    IDLE = "idle"
    BLIND_PATH = "blind_path"
    CROSSING = "crossing"
    TRAFFIC_LIGHT = "traffic_light"


@dataclass(frozen=True)
class NavigationResult:
    mode: NavigationMode
    speech: str | None = None
    overlay: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


class NavigationStateMachine:
    def __init__(self) -> None:
        self.mode = NavigationMode.IDLE
        self.last_speech = ""

    def command(self, text: str) -> NavigationResult:
        normalized = text.strip()
        speech: str | None = None
        if any(k in normalized for k in ("开始过马路", "帮我过马路")):
            self.mode = NavigationMode.CROSSING
            speech = "过马路模式已启动。"
        elif any(k in normalized for k in ("过马路结束", "结束过马路", "停止过马路", "取消过马路")):
            self.mode = NavigationMode.IDLE
            speech = "已停止导航。"
        elif any(k in normalized for k in ("检测红绿灯", "看红绿灯")):
            self.mode = NavigationMode.TRAFFIC_LIGHT
            speech = "红绿灯检测已启动。"
        elif any(k in normalized for k in ("停止检测", "取消检测", "停止红绿灯", "取消红绿灯")):
            self.mode = NavigationMode.IDLE
            speech = "红绿灯检测已停止。"
        elif any(k in normalized for k in ("开始导航", "盲道导航", "帮我导航")):
            self.mode = NavigationMode.BLIND_PATH
            speech = "盲道导航已启动。"
        elif any(k in normalized for k in ("停止导航", "结束导航", "取消导航")):
            self.mode = NavigationMode.IDLE
            speech = "已停止导航。"
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

        if speech == self.last_speech:
            speech = None
        elif speech:
            self.last_speech = speech
        return NavigationResult(self.mode, speech=speech, overlay=overlay, state=self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "last_speech": self.last_speech}

    def _blind_path_guidance(self, obs: dict[str, Any]) -> str | None:
        obstacle = obs.get("nearest_obstacle")
        if obstacle:
            return f"前方有{obstacle.get('label', '障碍物')}，停一下。"
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
