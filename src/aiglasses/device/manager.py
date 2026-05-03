from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from aiglasses.navigation import NavigationStateMachine
from aiglasses.protocol import Packet, PacketType, ProtocolError
from aiglasses.speech import SpeechHub
from aiglasses.vision import FrameAnalysis, VisionPipeline

MAX_TARGET_VIDEO_FPS = 1000


def clamp_target_video_fps(value: int) -> int:
    return min(MAX_TARGET_VIDEO_FPS, max(1, int(value)))


@dataclass
class UiClient:
    websocket: WebSocket


@dataclass
class DeviceManager:
    vision: VisionPipeline
    navigation: NavigationStateMachine
    speech: SpeechHub
    ui_clients: set[WebSocket] = field(default_factory=set)
    control_ws: WebSocket | None = None
    video_ws: WebSocket | None = None
    audio_ws: WebSocket | None = None
    last_frame_jpeg: bytes | None = None
    last_analysis: FrameAnalysis | None = None
    last_imu: dict[str, Any] | None = None
    target_video_fps: int = 1
    frame_count: int = 0
    audio_bytes: int = 0
    speech_seq: int = 0
    started_at: float = field(default_factory=time.time)

    async def add_ui(self, ws: WebSocket) -> None:
        self.ui_clients.add(ws)
        await self.broadcast({"kind": "snapshot", "state": self.snapshot()})

    def remove_ui(self, ws: WebSocket) -> None:
        self.ui_clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        text = json.dumps(message, ensure_ascii=False)
        for ws in list(self.ui_clients):
            if ws.client_state != WebSocketState.CONNECTED:
                dead.append(ws)
                continue
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ui_clients.discard(ws)

    async def handle_command_text(self, text: str, *, source: str = "debug") -> dict[str, Any]:
        result = self.navigation.command(text)
        event = {
            "kind": "command",
            "source": source,
            "text": text,
            "navigation": result.state,
            "created_at": time.time(),
        }
        await self.broadcast(event)
        if result.speech:
            await self.speech.say(result.speech, source="command")
        return event

    def device_config_payload(self) -> dict[str, Any]:
        return {"kind": "config", "target_fps": clamp_target_video_fps(self.target_video_fps)}

    async def send_device_config(self) -> bool:
        ws = self.control_ws
        if ws is None:
            return False
        try:
            await ws.send_text(json.dumps(self.device_config_payload(), ensure_ascii=False))
            return True
        except Exception:
            if self.control_ws is ws:
                self.control_ws = None
                await self.broadcast(
                    {"kind": "device", "channel": "control", "connected": False}
                )
            return False

    async def update_device_config(self, *, target_fps: int) -> dict[str, Any]:
        self.target_video_fps = clamp_target_video_fps(target_fps)
        sent = await self.send_device_config()
        config = self.device_config_payload()
        await self.broadcast({"kind": "device_config", "config": config, "sent": sent})
        return {"config": config, "sent": sent}

    async def sync_device_config(self) -> bool:
        sent = await self.send_device_config()
        if sent:
            await self.broadcast(
                {"kind": "device_config", "config": self.device_config_payload(), "sent": True}
            )
        return sent

    async def send_speech_pcm16(self, pcm16: bytes, *, chunk_bytes: int = 3200) -> bool:
        if not pcm16:
            return False
        sent = False
        for offset in range(0, len(pcm16), chunk_bytes):
            ws = self.control_ws
            if ws is None:
                return sent
            chunk = pcm16[offset : offset + chunk_bytes]
            packet = Packet(
                PacketType.SPEECH_PCM16,
                self.speech_seq,
                int(time.time() * 1000),
                chunk,
            )
            self.speech_seq += 1
            try:
                await ws.send_bytes(packet.pack())
                sent = True
            except Exception:
                if self.control_ws is ws:
                    self.control_ws = None
                    await self.broadcast(
                        {"kind": "device", "channel": "control", "connected": False}
                    )
                return sent
        return sent

    async def handle_control_packet(self, packet: Packet) -> None:
        if packet.packet_type == PacketType.IMU_JSON:
            self.last_imu = json.loads(packet.payload.decode("utf-8"))
            await self.broadcast({"kind": "imu", "data": self.last_imu})
        elif packet.packet_type in {PacketType.HELLO, PacketType.CONTROL_JSON}:
            payload = json.loads(packet.payload.decode("utf-8") or "{}")
            await self.broadcast({"kind": "device", "data": payload, "state": self.snapshot()})
        elif packet.packet_type == PacketType.PING:
            if self.control_ws:
                await self.control_ws.send_bytes(
                    Packet(PacketType.PONG, packet.seq, int(time.time() * 1000)).pack()
                )

    async def handle_video_packet(self, packet: Packet) -> None:
        if packet.packet_type != PacketType.VIDEO_JPEG:
            raise ProtocolError(f"unexpected packet on video channel: {packet.packet_type}")
        self.last_frame_jpeg = packet.payload
        self.frame_count += 1
        if self.frame_count % 2 == 1:
            analysis = await asyncio.to_thread(self.vision.analyze_jpeg, packet.payload)
            self.last_analysis = analysis
            observation = analysis.to_observation()
            nav = self.navigation.process_observation(observation)
            await self.broadcast(
                {
                    "kind": "analysis",
                    "frame_count": self.frame_count,
                    "observation": observation,
                    "navigation": nav.state,
                    "overlay": nav.overlay,
                }
            )
            if nav.speech:
                await self.speech.say(nav.speech)
        await self.broadcast({"kind": "frame", "frame_count": self.frame_count})

    async def handle_audio_packet(self, packet: Packet) -> None:
        if packet.packet_type != PacketType.AUDIO_PCM16:
            raise ProtocolError(f"unexpected packet on audio channel: {packet.packet_type}")
        self.audio_bytes += len(packet.payload)
        if self.audio_bytes % (16000 * 2 * 5) < len(packet.payload):
            await self.broadcast({"kind": "audio", "bytes": self.audio_bytes})

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "device": {
                "control": self.control_ws is not None,
                "video": self.video_ws is not None,
                "audio": self.audio_ws is not None,
            },
            "frame_count": self.frame_count,
            "audio_bytes": self.audio_bytes,
            "device_config": self.device_config_payload(),
            "navigation": self.navigation.snapshot(),
            "imu": self.last_imu,
            "model_status": self.vision.model_status,
        }
