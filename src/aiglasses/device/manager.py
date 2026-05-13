from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
from fastapi import WebSocket
import numpy as np
from starlette.websockets import WebSocketState

from aiglasses.navigation import NavigationStateMachine
from aiglasses.protocol import Packet, PacketType, ProtocolError
from aiglasses.speech import SpeechHub
from aiglasses.vision import FrameAnalysis, VisionPipeline

MAX_TARGET_VIDEO_FPS = 1000
BENCHMARK_JPEG_QUALITY = 85
RECORDINGS_DIR = Path("recordings")
RECORDING_FORMAT_VERSION = 1
SUPERSEDED_WS_CLOSE_TIMEOUT_S = 1.0

logger = logging.getLogger("aiglasses.device")


def clamp_target_video_fps(value: int) -> int:
    return min(MAX_TARGET_VIDEO_FPS, max(1, int(value)))


def _device_ws_locks() -> dict[str, asyncio.Lock]:
    return {
        "control": asyncio.Lock(),
        "video": asyncio.Lock(),
        "audio": asyncio.Lock(),
    }


def _device_ws_close_tasks() -> dict[str, set[asyncio.Task[None]]]:
    return {
        "control": set(),
        "video": set(),
        "audio": set(),
    }


def _device_ws_generations() -> dict[str, int]:
    return {
        "control": 0,
        "video": 0,
        "audio": 0,
    }


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
    video_udp_last_seen_at: float | None = None
    last_frame_jpeg: bytes | None = None
    last_analysis: FrameAnalysis | None = None
    last_imu: dict[str, Any] | None = None
    target_video_fps: int = 1
    device_jpeg_quality: int = 12
    device_camera_profile: str = "traffic_signal"
    device_ae_level: int = -1
    device_saturation: int = 1
    device_contrast: int = 1
    device_sharpness: int = 1
    device_gainceiling: int = 4
    frame_count: int = 0
    audio_bytes: int = 0
    speech_seq: int = 0
    started_at: float = field(default_factory=time.time)
    frame_received_at: list[float] = field(default_factory=list)
    analysis_elapsed_ms: list[float] = field(default_factory=list)
    backend_benchmark: dict[str, Any] = field(
        default_factory=lambda: {"status": "pending"}
    )
    recording_active: bool = False
    recording_dir: Path | None = None
    recording_frames_dir: Path | None = None
    recording_session_id: str | None = None
    recording_started_at: str | None = None
    recording_frame_count: int = 0
    device_ws_locks: dict[str, asyncio.Lock] = field(default_factory=_device_ws_locks)
    device_ws_close_tasks: dict[str, set[asyncio.Task[None]]] = field(
        default_factory=_device_ws_close_tasks
    )
    device_ws_generations: dict[str, int] = field(default_factory=_device_ws_generations)

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

    async def broadcast_bytes(self, data: bytes) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.ui_clients):
            if ws.client_state != WebSocketState.CONNECTED:
                dead.append(ws)
                continue
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ui_clients.discard(ws)

    async def replace_device_ws(self, channel: str, ws: WebSocket) -> int | None:
        async with self.device_ws_locks[channel]:
            previous = self._device_ws(channel)
            if previous is ws:
                return None
            self._set_device_ws(channel, ws)
            generation = self._next_device_ws_generation(channel)
        if previous is not None:
            self._close_ws_soon(channel, previous)
        return generation

    def _close_ws_soon(self, channel: str, ws: WebSocket) -> asyncio.Task[None]:
        async def close() -> None:
            try:
                await asyncio.wait_for(
                    self._close_ws(ws),
                    timeout=SUPERSEDED_WS_CLOSE_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning("timed out closing superseded %s websocket", channel)
            except Exception:
                logger.exception("failed to close superseded %s websocket", channel)

        task = asyncio.create_task(close())
        self.device_ws_close_tasks[channel].add(task)
        task.add_done_callback(self.device_ws_close_tasks[channel].discard)
        return task

    async def _drain_ws_close_tasks(self, channel: str) -> None:
        tasks = list(self.device_ws_close_tasks[channel])
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def device_ws_is_current(self, channel: str, ws: WebSocket) -> bool:
        async with self.device_ws_locks[channel]:
            return self._device_ws(channel) is ws

    def _next_device_ws_generation(self, channel: str) -> int:
        self.device_ws_generations[channel] += 1
        return self.device_ws_generations[channel]

    def device_connection_event(
        self,
        *,
        channel: str,
        connected: bool,
        generation: int,
    ) -> dict[str, Any]:
        return {
            "kind": "device",
            "channel": channel,
            "connected": connected,
            "generation": generation,
        }

    async def disconnect_device(self) -> dict[str, Any]:
        disconnected: list[str] = []
        events: list[dict[str, Any]] = []
        for channel in ("control", "video", "audio"):
            async with self.device_ws_locks[channel]:
                ws = self._device_ws(channel)
                generation: int | None = None
                if ws is not None:
                    self._set_device_ws(channel, None)
                    generation = self._next_device_ws_generation(channel)
            if ws is None:
                await self._drain_ws_close_tasks(channel)
                continue
            await self._close_ws(ws)
            await self._drain_ws_close_tasks(channel)
            disconnected.append(channel)
            if generation is not None:
                events.append(
                    self.device_connection_event(
                        channel=channel,
                        connected=False,
                        generation=generation,
                    )
                )
        for event in events:
            await self.broadcast(event)
        return {"disconnected": disconnected, "state": self.snapshot()}

    async def clear_device_ws(self, channel: str, ws: WebSocket) -> int | None:
        async with self.device_ws_locks[channel]:
            return self._clear_device_ws_locked(channel, ws)

    def _clear_device_ws_locked(self, channel: str, ws: WebSocket) -> int | None:
        if self._device_ws(channel) is not ws:
            return None
        self._set_device_ws(channel, None)
        return self._next_device_ws_generation(channel)

    async def _broadcast_device_disconnect_for_current(self, channel: str, ws: WebSocket) -> bool:
        generation = await self.clear_device_ws(channel, ws)
        if generation is None:
            return False
        await self.broadcast(
            self.device_connection_event(
                channel=channel,
                connected=False,
                generation=generation,
            )
        )
        return True

    def _device_ws(self, channel: str) -> WebSocket | None:
        if channel == "control":
            return self.control_ws
        if channel == "video":
            return self.video_ws
        if channel == "audio":
            return self.audio_ws
        raise ValueError(f"unknown device channel: {channel}")

    def _set_device_ws(self, channel: str, ws: WebSocket | None) -> None:
        if channel == "control":
            self.control_ws = ws
        elif channel == "video":
            self.video_ws = ws
        elif channel == "audio":
            self.audio_ws = ws
        else:
            raise ValueError(f"unknown device channel: {channel}")

    @staticmethod
    async def _close_ws(ws: WebSocket) -> None:
        try:
            await ws.close(code=1012)
        except Exception:
            pass

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
        return {
            "kind": "config",
            "target_fps": clamp_target_video_fps(self.target_video_fps),
            "jpeg_quality": self.device_jpeg_quality,
            "camera_profile": self.device_camera_profile,
            "ae_level": self.device_ae_level,
            "saturation": self.device_saturation,
            "contrast": self.device_contrast,
            "sharpness": self.device_sharpness,
            "gainceiling": self.device_gainceiling,
        }

    def mark_video_udp_seen(self) -> None:
        self.video_udp_last_seen_at = time.monotonic()

    def video_transport_connected(self) -> bool:
        if self.video_ws is not None:
            return True
        return (
            self.video_udp_last_seen_at is not None
            and time.monotonic() - self.video_udp_last_seen_at <= 5
        )

    async def send_device_config(self) -> bool:
        ws = self.control_ws
        if ws is None:
            return False
        try:
            await ws.send_text(json.dumps(self.device_config_payload(), ensure_ascii=False))
            return True
        except Exception:
            await self._broadcast_device_disconnect_for_current("control", ws)
            return False

    async def update_device_config(
        self,
        *,
        target_fps: int | None = None,
        jpeg_quality: int | None = None,
        camera_profile: str | None = None,
        ae_level: int | None = None,
        saturation: int | None = None,
        contrast: int | None = None,
        sharpness: int | None = None,
        gainceiling: int | None = None,
    ) -> dict[str, Any]:
        if target_fps is not None:
            self.target_video_fps = clamp_target_video_fps(target_fps)
        if jpeg_quality is not None:
            self.device_jpeg_quality = min(63, max(1, int(jpeg_quality)))
        if camera_profile is not None:
            profile = str(camera_profile).strip()
            if profile in {"default", "traffic_signal"}:
                self.device_camera_profile = profile
        if ae_level is not None:
            self.device_ae_level = min(2, max(-2, int(ae_level)))
        if saturation is not None:
            self.device_saturation = min(2, max(-2, int(saturation)))
        if contrast is not None:
            self.device_contrast = min(2, max(-2, int(contrast)))
        if sharpness is not None:
            self.device_sharpness = min(2, max(-2, int(sharpness)))
        if gainceiling is not None:
            self.device_gainceiling = min(128, max(2, int(gainceiling)))
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

    def recording_status(self) -> dict[str, Any]:
        return {
            "active": self.recording_active,
            "session_id": self.recording_session_id,
            "recording_dir": str(self.recording_dir) if self.recording_dir is not None else None,
            "frame_count": self.recording_frame_count,
            "started_at": self.recording_started_at,
        }

    async def start_recording(self) -> dict[str, Any]:
        if self.recording_active:
            return self.recording_status()

        now = datetime.now(timezone.utc)
        session_id = now.strftime("%Y%m%d-%H%M%S")
        recording_dir = RECORDINGS_DIR / session_id
        suffix = 1
        while recording_dir.exists():
            suffix += 1
            recording_dir = RECORDINGS_DIR / f"{session_id}-{suffix}"
        frames_dir = recording_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=False)

        self.recording_active = True
        self.recording_dir = recording_dir
        self.recording_frames_dir = frames_dir
        self.recording_session_id = recording_dir.name
        self.recording_started_at = now.isoformat()
        self.recording_frame_count = 0
        self._write_recording_session(stopped_at=None)
        status = self.recording_status()
        await self.broadcast({"kind": "recording", "recording": status})
        return status

    async def stop_recording(self) -> dict[str, Any]:
        was_active = self.recording_active
        if was_active:
            self._write_recording_session(stopped_at=datetime.now(timezone.utc).isoformat())
        self.recording_active = False
        status = self.recording_status()
        if was_active:
            await self.broadcast({"kind": "recording", "recording": status})
        return status

    def _write_recording_session(self, *, stopped_at: str | None) -> None:
        if self.recording_dir is None:
            return
        payload = {
            "format": "aiglasses_raw_jpeg_recording",
            "version": RECORDING_FORMAT_VERSION,
            "session_id": self.recording_session_id,
            "started_at": self.recording_started_at,
            "stopped_at": stopped_at,
            "frame_count": self.recording_frame_count,
        }
        tmp_path = self.recording_dir / "session.json.tmp"
        final_path = self.recording_dir / "session.json"
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(final_path)

    async def _record_frame(self, jpeg_bytes: bytes) -> None:
        if not self.recording_active or self.recording_frames_dir is None or self.recording_dir is None:
            return
        frame_index = self.recording_frame_count + 1
        filename = f"{frame_index:08d}.jpg"
        relative_frame = f"frames/{filename}"
        frame_path = self.recording_frames_dir / filename
        tmp_path = frame_path.with_name(f"{filename}.tmp")
        metadata = {
            "recording_frame_index": frame_index,
            "global_frame_count": self.frame_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_file": relative_frame,
            "jpeg_bytes": len(jpeg_bytes),
            "analysis": self.last_analysis.to_observation() if self.last_analysis is not None else None,
            "navigation": self.navigation.snapshot(),
            "tuning": self._vision_tuning_payload(),
            "imu": self.last_imu,
        }
        try:
            tmp_path.write_bytes(jpeg_bytes)
            tmp_path.replace(frame_path)
            with (self.recording_dir / "metadata.jsonl").open("a", encoding="utf-8") as metadata_file:
                metadata_file.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
                metadata_file.write("\n")
            self.recording_frame_count = frame_index
        except Exception:
            logger.exception("failed to write recording frame")
            self.recording_active = False
            await self.broadcast({"kind": "recording", "recording": self.recording_status()})

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
                await self._broadcast_device_disconnect_for_current("control", ws)
                return sent
        return sent

    async def handle_control_packet(self, packet: Packet) -> None:
        if packet.packet_type == PacketType.IMU_JSON:
            payload = await self._decode_device_json(packet, channel="control")
            if payload is None:
                return
            self.last_imu = payload
            await self.broadcast({"kind": "imu", "data": self.last_imu})
        elif packet.packet_type in {PacketType.HELLO, PacketType.CONTROL_JSON}:
            payload = await self._decode_device_json(packet, channel="control")
            if payload is None:
                return
            await self.broadcast({"kind": "device", "data": payload, "state": self.snapshot()})
        elif packet.packet_type == PacketType.PING:
            if self.control_ws:
                await self.control_ws.send_bytes(
                    Packet(PacketType.PONG, packet.seq, int(time.time() * 1000)).pack()
                )

    async def _decode_device_json(
        self,
        packet: Packet,
        *,
        channel: str,
    ) -> dict[str, Any] | None:
        try:
            payload = json.loads(packet.payload.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self.broadcast(
                {
                    "kind": "device_error",
                    "channel": channel,
                    "packet_type": packet.packet_type.name,
                    "error": f"malformed device JSON: {exc}",
                }
            )
            return None
        if not isinstance(payload, dict):
            await self.broadcast(
                {
                    "kind": "device_error",
                    "channel": channel,
                    "packet_type": packet.packet_type.name,
                    "error": "malformed device JSON: payload must be an object",
                }
            )
            return None
        return payload

    async def handle_video_packet(self, packet: Packet) -> None:
        if packet.packet_type != PacketType.VIDEO_JPEG:
            raise ProtocolError(f"unexpected packet on video channel: {packet.packet_type}")
        self.last_frame_jpeg = packet.payload
        self.frame_count += 1
        self._record_frame_received()
        if self.frame_count % 2 == 1:
            started = time.perf_counter()
            analysis = await asyncio.to_thread(self.vision.analyze_jpeg, packet.payload)
            self._record_analysis_elapsed((time.perf_counter() - started) * 1000)
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
        await self._record_frame(packet.payload)
        await self.broadcast_bytes(
            Packet(
                PacketType.VIDEO_JPEG,
                self.frame_count,
                int(time.time() * 1000),
                packet.payload,
            ).pack()
        )
        await self.broadcast(
            {
                "kind": "frame",
                "frame_count": self.frame_count,
                "video_stats": self.video_stats(),
                "recording": self.recording_status(),
            }
        )

    def _record_frame_received(self) -> None:
        now = time.monotonic()
        self.frame_received_at.append(now)
        cutoff = now - 10
        while self.frame_received_at and self.frame_received_at[0] < cutoff:
            self.frame_received_at.pop(0)

    def _record_analysis_elapsed(self, elapsed_ms: float) -> None:
        self.analysis_elapsed_ms.append(elapsed_ms)
        del self.analysis_elapsed_ms[:-20]

    def video_stats(self) -> dict[str, Any]:
        now = time.monotonic()
        recent = [item for item in self.frame_received_at if now - item <= 3]
        fps = 0.0
        if len(recent) >= 2:
            elapsed = recent[-1] - recent[0]
            fps = (len(recent) - 1) / elapsed if elapsed > 0 else 0.0
        analysis_ms = self.analysis_elapsed_ms[-1] if self.analysis_elapsed_ms else None
        avg_analysis_ms = (
            sum(self.analysis_elapsed_ms) / len(self.analysis_elapsed_ms)
            if self.analysis_elapsed_ms
            else None
        )
        return {
            "received_fps_3s": round(fps, 2),
            "last_analysis_ms": round(analysis_ms, 1) if analysis_ms is not None else None,
            "avg_analysis_ms": round(avg_analysis_ms, 1) if avg_analysis_ms is not None else None,
        }

    def benchmark_processing_capacity(
        self,
        *,
        warmup_runs: int = 5,
        measured_runs: int = 20,
        seed: int = 20260504,
    ) -> dict[str, Any]:
        if warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if measured_runs <= 0:
            raise ValueError("measured_runs must be positive")

        self.backend_benchmark = {
            "status": "running",
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
        }
        payload = self._benchmark_jpeg_payload(seed)
        started = time.perf_counter()

        for _ in range(warmup_runs):
            self.vision.analyze_jpeg(payload)

        samples = [self._time_analysis(payload) for _ in range(measured_runs)]
        elapsed_ms = (time.perf_counter() - started) * 1000
        ordered = sorted(samples)
        p50_ms = statistics.median(ordered)
        mean_ms = statistics.mean(ordered)
        p90_ms = ordered[round((len(ordered) - 1) * 0.90)]
        p95_ms = ordered[round((len(ordered) - 1) * 0.95)]
        self.backend_benchmark = {
            "status": "ready",
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "image_width": self.vision.config.models.image_width,
            "image_height": self.vision.config.models.image_height,
            "min_ms": round(min(ordered), 1),
            "p50_ms": round(p50_ms, 1),
            "mean_ms": round(mean_ms, 1),
            "p90_ms": round(p90_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "max_ms": round(max(ordered), 1),
            "fps_p50": round(1000 / p50_ms, 2) if p50_ms > 0 else None,
            "fps_mean": round(1000 / mean_ms, 2) if mean_ms > 0 else None,
            "elapsed_ms": round(elapsed_ms, 1),
            "created_at": time.time(),
        }
        return self.backend_benchmark

    def _benchmark_jpeg_payload(self, seed: int) -> bytes:
        width = self.vision.config.models.image_width
        height = self.vision.config.models.image_height
        rng = np.random.default_rng(seed)
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), BENCHMARK_JPEG_QUALITY],
        )
        if not ok:
            raise ValueError("failed to encode benchmark frame")
        return encoded.tobytes()

    def _time_analysis(self, payload: bytes) -> float:
        started = time.perf_counter()
        self.vision.analyze_jpeg(payload)
        return (time.perf_counter() - started) * 1000

    async def handle_audio_packet(self, packet: Packet) -> None:
        if packet.packet_type != PacketType.AUDIO_PCM16:
            raise ProtocolError(f"unexpected packet on audio channel: {packet.packet_type}")
        self.audio_bytes += len(packet.payload)
        if self.audio_bytes % (16000 * 2 * 5) < len(packet.payload):
            await self.broadcast({"kind": "audio", "bytes": self.audio_bytes})

    def _vision_tuning_payload(self) -> dict[str, Any]:
        tuning = getattr(getattr(self.vision, "tuning", None), "to_dict", None)
        payload = tuning() if callable(tuning) else {}
        return payload if isinstance(payload, dict) else {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "device": {
                "control": self.control_ws is not None,
                "video": self.video_transport_connected(),
                "audio": self.audio_ws is not None,
                "generation": dict(self.device_ws_generations),
            },
            "frame_count": self.frame_count,
            "audio_bytes": self.audio_bytes,
            "device_config": self.device_config_payload(),
            "navigation": self.navigation.snapshot(),
            "imu": self.last_imu,
            "model_status": self.vision.model_status,
            "vision": self.vision_frame_size(),
            "video_stats": self.video_stats(),
            "backend_benchmark": self.backend_benchmark,
            "tuning": self._vision_tuning_payload(),
            "recording": self.recording_status(),
        }

    def vision_frame_size(self) -> dict[str, int | None]:
        models = getattr(getattr(self.vision, "config", None), "models", None)
        return {
            "image_width": getattr(models, "image_width", None),
            "image_height": getattr(models, "image_height", None),
        }
