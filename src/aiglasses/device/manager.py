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
    frame_count: int = 0
    audio_bytes: int = 0
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
            "navigation": self.navigation.snapshot(),
            "imu": self.last_imu,
            "model_status": self.vision.model_status,
        }
