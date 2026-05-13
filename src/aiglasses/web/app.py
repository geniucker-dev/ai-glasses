from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import struct
import time
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from aiglasses.asr import AsrService
from aiglasses.config import AppConfig
from aiglasses.device import DeviceManager, clamp_target_video_fps
from aiglasses.navigation import NavigationStateMachine
from aiglasses.protocol import MAX_PAYLOAD_BYTES, Packet, PacketType, ProtocolError
from aiglasses.speech import DashscopeTtsSpeechSink, LocalTtsSpeechSink, SpeechHub, UiSpeechSink
from aiglasses.vision import VisionPipeline


logger = logging.getLogger("aiglasses.web")

CONTROL_MAX_PAYLOAD_BYTES = 8 * 1024
AUDIO_MAX_PAYLOAD_BYTES = 64 * 1024
VIDEO_MAX_PAYLOAD_BYTES = MAX_PAYLOAD_BYTES
VIDEO_FRAGMENT_HEADER = struct.Struct("<IQIIHH")


def _is_websocket_state_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "WebSocket is not connected" in message or "Cannot call" in message


async def _unpack_device_packet(
    data: bytes,
    *,
    channel: str,
    max_payload_bytes: int,
    manager: DeviceManager | None = None,
) -> Packet | None:
    try:
        return Packet.unpack(data, max_payload_bytes=max_payload_bytes)
    except ProtocolError as exc:
        logger.warning("dropping bad %s packet: %s", channel, exc)
        if manager is not None:
            await manager.broadcast({"kind": "device_error", "channel": channel, "error": str(exc)})
        return None


async def _receive_current_device_packet(
    ws: WebSocket,
    *,
    manager: DeviceManager,
    channel: str,
    max_payload_bytes: int,
) -> Packet | None:
    data = await ws.receive_bytes()
    if not await manager.device_ws_is_current(channel, ws):
        return None
    packet = await _unpack_device_packet(
        data,
        channel=channel,
        max_payload_bytes=max_payload_bytes,
        manager=manager,
    )
    if packet is None or not await manager.device_ws_is_current(channel, ws):
        return None
    return packet


class UdpVideoReassembler:
    def __init__(self, manager: DeviceManager, config: AppConfig):
        self.manager = manager
        self.config = config
        self.frames: dict[int, dict[str, Any]] = {}
        self.active_session_id: int | None = None
        self.retired_session_ids: set[int] = set()
        self.last_completed_frame_id = -1
        self.lock = asyncio.Lock()

    async def handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        max_fragment_payload = (
            VIDEO_FRAGMENT_HEADER.size + self.config.device.transport.video_payload_bytes
        )
        try:
            packet = Packet.unpack(data, max_payload_bytes=max_fragment_payload)
            if packet.packet_type != PacketType.VIDEO_JPEG_FRAGMENT:
                raise ProtocolError(f"unexpected udp video packet: {packet.packet_type}")
            if len(packet.payload) < VIDEO_FRAGMENT_HEADER.size:
                raise ProtocolError("video fragment payload too short")
            (
                session_id,
                frame_id,
                frame_len,
                offset,
                chunk_index,
                chunk_count,
            ) = VIDEO_FRAGMENT_HEADER.unpack(packet.payload[: VIDEO_FRAGMENT_HEADER.size])
            chunk = packet.payload[VIDEO_FRAGMENT_HEADER.size :]
            chunk_bytes = self.config.device.transport.video_payload_bytes
            expected_chunk_count = (frame_len + chunk_bytes - 1) // chunk_bytes
            expected_offset = chunk_index * chunk_bytes
            expected_chunk_len = (
                chunk_bytes if chunk_index + 1 < chunk_count else frame_len - expected_offset
            )
            if frame_len == 0 or frame_len > VIDEO_MAX_PAYLOAD_BYTES:
                raise ProtocolError(f"video frame too large: {frame_len}")
            if chunk_count == 0 or chunk_index >= chunk_count:
                raise ProtocolError("invalid video fragment index")
            if chunk_count != expected_chunk_count:
                raise ProtocolError("invalid video fragment count")
            if offset != expected_offset:
                raise ProtocolError("misaligned video fragment offset")
            if len(chunk) != expected_chunk_len:
                raise ProtocolError("invalid video fragment length")
            if offset + len(chunk) > frame_len:
                raise ProtocolError("video fragment exceeds frame length")
        except ProtocolError as exc:
            logger.warning("dropping bad udp video packet from %s: %s", addr, exc)
            await self.manager.broadcast(
                {"kind": "device_error", "channel": "video_udp", "error": str(exc)}
            )
            return

        complete_payload: bytes | None = None
        now = time.monotonic()
        async with self.lock:
            if self.active_session_id is None:
                self.active_session_id = session_id
            elif session_id != self.active_session_id:
                if session_id in self.retired_session_ids:
                    return
                self.retired_session_ids.add(self.active_session_id)
                self.active_session_id = session_id
                self.last_completed_frame_id = -1
                self.frames.clear()
            self._drop_expired(now)
            frame = self.frames.get(frame_id)
            if frame is None:
                frame = {
                    "created_at": now,
                    "frame_len": frame_len,
                    "chunk_count": chunk_count,
                    "buffer": bytearray(frame_len),
                    "seen": set(),
                    "received": 0,
                    "timestamp_ms": packet.timestamp_ms,
                }
                self.frames[frame_id] = frame
            if frame["frame_len"] != frame_len or frame["chunk_count"] != chunk_count:
                self.frames.pop(frame_id, None)
                logger.warning("dropping inconsistent udp video frame id=%s from %s", frame_id, addr)
                return
            seen: set[int] = frame["seen"]
            if chunk_index not in seen:
                buffer: bytearray = frame["buffer"]
                buffer[offset : offset + len(chunk)] = chunk
                seen.add(chunk_index)
                frame["received"] += len(chunk)
            if len(seen) == chunk_count and frame["received"] >= frame_len:
                self.frames.pop(frame_id, None)
                if frame_id > self.last_completed_frame_id:
                    complete_payload = bytes(frame["buffer"])
                    self.last_completed_frame_id = frame_id

        if complete_payload is not None:
            self.manager.mark_video_udp_seen()
            await self.manager.handle_video_packet(
                Packet(PacketType.VIDEO_JPEG, frame_id, packet.timestamp_ms, complete_payload)
            )

    def _drop_expired(self, now: float) -> None:
        timeout_ms = max(
            self.config.device.transport.video_frame_timeout_ms,
            self.config.device.transport.video_reorder_window_ms,
        )
        timeout_s = timeout_ms / 1000
        expired = [
            frame_id
            for frame_id, frame in self.frames.items()
            if now - frame["created_at"] > timeout_s
        ]
        for frame_id in expired:
            self.frames.pop(frame_id, None)


class UdpVideoProtocol(asyncio.DatagramProtocol):
    def __init__(self, reassembler: UdpVideoReassembler):
        self.reassembler = reassembler

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        sockname = transport.get_extra_info("sockname")
        logger.info("device udp video listening on %s", sockname)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        asyncio.create_task(self.reassembler.handle_datagram(data, addr))

    def error_received(self, exc: Exception) -> None:
        logger.warning("device udp video error: %s", exc)


def validate_speech_config(config: AppConfig) -> None:
    if not config.speech.enabled:
        return
    if config.speech.mode not in {"ui", "device"}:
        raise ValueError("speech.mode must be either 'ui' or 'device'")
    if config.speech.mode == "device" and not config.device.audio_down.enabled:
        raise ValueError("speech.mode='device' requires device.audio_down.enabled=true")
    if config.speech.mode == "device" and config.speech.provider not in {"dashscope", "local"}:
        raise ValueError("speech.provider must be either 'dashscope' or 'local'")


def create_app(config: AppConfig) -> FastAPI:
    validate_speech_config(config)
    speech = SpeechHub()
    vision = VisionPipeline(config)
    navigation = NavigationStateMachine(tuning=vision.tuning)
    manager = DeviceManager(
        vision,
        navigation,
        speech,
        target_video_fps=clamp_target_video_fps(config.device.capture.video_fps),
        device_jpeg_quality=config.device.capture.jpeg_quality,
        device_camera_profile=config.device.capture.camera_profile,
    )
    speech.add_sink(UiSpeechSink(manager.broadcast))
    if config.speech.enabled and config.speech.mode == "device":
        if config.speech.provider == "dashscope":
            tts_sink = DashscopeTtsSpeechSink(
                config.speech,
                api_key=config.asr.dashscope_api_key,
                send_pcm16=manager.send_speech_pcm16,
                sample_rate=config.device.capture.audio_sample_rate,
                broadcast=manager.broadcast,
                websocket_base_url=config.asr.websocket_base_url,
                http_base_url=config.asr.http_base_url,
            )
        else:
            tts_sink = LocalTtsSpeechSink(
                config.speech,
                send_pcm16=manager.send_speech_pcm16,
                sample_rate=config.device.capture.audio_sample_rate,
                broadcast=manager.broadcast,
            )
        speech.add_sink(tts_sink)
    asr = AsrService(config.asr, lambda text: manager.handle_command_text(text, source="asr"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        udp_video_transport: asyncio.DatagramTransport | None = None
        try:
            benchmark = await asyncio.to_thread(manager.benchmark_processing_capacity)
            logger.info("backend processing benchmark: %s", benchmark)
        except Exception as exc:
            logger.exception("backend processing benchmark failed")
            manager.backend_benchmark = {
                "status": "failed",
                "error": str(exc),
            }
        await asr.start()
        if config.device.transport.video == "udp":
            loop = asyncio.get_running_loop()
            udp_video_transport, _ = await loop.create_datagram_endpoint(
                lambda: UdpVideoProtocol(UdpVideoReassembler(manager, config)),
                local_addr=(config.server.host, config.server.port),
            )
        try:
            yield
        finally:
            if udp_video_transport is not None:
                udp_video_transport.close()
            await manager.stop_recording()
            await asr.stop()

    app = FastAPI(title=config.web.title, lifespan=lifespan)
    web_dir = Path(__file__).resolve().parent
    static_dir = web_dir / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.state.config = config
    app.state.manager = manager
    app.state.asr = asr

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (web_dir / "templates" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"ok": True, "state": manager.snapshot(), "asr": asr.status}

    @app.get("/api/v1/state")
    async def state() -> dict:
        return manager.snapshot()

    @app.get("/api/v1/frame")
    async def frame() -> JSONResponse:
        if manager.last_frame_jpeg is None:
            return JSONResponse({"frame": None, "frame_count": manager.frame_count})
        encoded = base64.b64encode(manager.last_frame_jpeg).decode("ascii")
        return JSONResponse(
            {
                "frame": f"data:image/jpeg;base64,{encoded}",
                "frame_count": manager.frame_count,
            }
        )

    @app.get("/api/v1/frame.jpg")
    async def frame_jpeg() -> Response:
        if manager.last_frame_jpeg is None:
            return Response(status_code=204, headers={"cache-control": "no-store"})
        return Response(
            manager.last_frame_jpeg,
            media_type="image/jpeg",
            headers={
                "cache-control": "no-store",
                "x-frame-count": str(manager.frame_count),
            },
        )

    @app.post("/api/v1/commands")
    async def command(payload: Annotated[dict, Body()]) -> dict:
        text = str(payload.get("text", ""))
        return await manager.handle_command_text(text, source="debug")

    @app.get("/api/v1/device/config")
    async def device_config() -> dict:
        return manager.device_config_payload()

    @app.get("/api/v1/recording/status")
    async def recording_status() -> dict:
        return {"recording": manager.recording_status()}

    @app.post("/api/v1/recording/start")
    async def start_recording() -> dict:
        return {"recording": await manager.start_recording()}

    @app.post("/api/v1/recording/stop")
    async def stop_recording() -> dict:
        return {"recording": await manager.stop_recording()}

    @app.post("/api/v1/device/config")
    async def update_device_config(payload: Annotated[dict, Body()]) -> dict:
        values: dict[str, Any] = {}
        int_fields = {
            "target_fps",
            "video_fps",
            "jpeg_quality",
            "ae_level",
            "saturation",
            "contrast",
            "sharpness",
            "gainceiling",
        }
        for field in int_fields:
            if field not in payload:
                continue
            try:
                values[field] = int(payload[field])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"{field} must be an integer") from exc
        video_fps = values.pop("video_fps", None)
        if video_fps is not None and "target_fps" not in values:
            values["target_fps"] = video_fps
        if "camera_profile" in payload:
            values["camera_profile"] = str(payload["camera_profile"])
        if not values:
            raise HTTPException(status_code=400, detail="no device config fields provided")
        return await manager.update_device_config(**values)

    @app.get("/api/v1/debug/tuning")
    async def debug_tuning() -> dict:
        return {"tuning": manager.vision.tuning.to_dict()}

    @app.post("/api/v1/debug/tuning")
    async def update_debug_tuning(payload: Annotated[dict, Body()]) -> dict:
        try:
            tuning = manager.vision.tuning.updated(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        manager.vision.tuning = tuning
        manager.navigation.tuning = tuning
        await manager.broadcast({"kind": "tuning", "tuning": tuning.to_dict()})
        return {"tuning": tuning.to_dict()}

    @app.post("/api/v1/device/disconnect")
    async def disconnect_device() -> dict:
        return await manager.disconnect_device()

    @app.websocket("/ws/ui")
    async def ws_ui(ws: WebSocket) -> None:
        await ws.accept()
        await manager.add_ui(ws)
        await ws.send_json({"kind": "asr", "status": asr.status})
        try:
            while True:
                message = await ws.receive_text()
                await manager.handle_command_text(message, source="ui")
        except WebSocketDisconnect:
            manager.remove_ui(ws)

    @app.websocket("/ws/device/control")
    async def ws_device_control(ws: WebSocket) -> None:
        try:
            await ws.accept()
            generation = await manager.replace_device_ws("control", ws)
            if not await manager.device_ws_is_current("control", ws):
                return
            if generation is not None:
                await manager.broadcast(
                    manager.device_connection_event(
                        channel="control",
                        connected=True,
                        generation=generation,
                    )
                )
            if not await manager.sync_device_config():
                return
            while True:
                if not await manager.device_ws_is_current("control", ws):
                    return
                packet = await _receive_current_device_packet(
                    ws,
                    manager=manager,
                    channel="control",
                    max_payload_bytes=CONTROL_MAX_PAYLOAD_BYTES,
                )
                if packet is None:
                    continue
                await manager.handle_control_packet(packet)
        except WebSocketDisconnect:
            logger.info("device control websocket disconnected")
        except RuntimeError as exc:
            if not _is_websocket_state_error(exc):
                logger.exception("device control websocket failed")
                await manager.broadcast(
                    {"kind": "device_error", "channel": "control", "error": str(exc)}
                )
            else:
                logger.info("device control websocket disconnected: %s", exc)
        except Exception as exc:
            logger.exception("device control websocket failed")
            await manager.broadcast({"kind": "device_error", "channel": "control", "error": str(exc)})
        finally:
            generation = await manager.clear_device_ws("control", ws)
            if generation is not None:
                await manager.broadcast(
                    manager.device_connection_event(
                        channel="control",
                        connected=False,
                        generation=generation,
                    )
                )

    @app.websocket("/ws/device/video")
    async def ws_device_video(ws: WebSocket) -> None:
        try:
            await ws.accept()
            generation = await manager.replace_device_ws("video", ws)
            if not await manager.device_ws_is_current("video", ws):
                return
            if generation is not None:
                await manager.broadcast(
                    manager.device_connection_event(
                        channel="video",
                        connected=True,
                        generation=generation,
                    )
                )
            while True:
                if not await manager.device_ws_is_current("video", ws):
                    return
                packet = await _receive_current_device_packet(
                    ws,
                    manager=manager,
                    channel="video",
                    max_payload_bytes=VIDEO_MAX_PAYLOAD_BYTES,
                )
                if packet is None:
                    continue
                await manager.handle_video_packet(packet)
        except WebSocketDisconnect:
            logger.info("device video websocket disconnected")
        except RuntimeError as exc:
            if not _is_websocket_state_error(exc):
                logger.exception("device video websocket failed")
                await manager.broadcast(
                    {"kind": "device_error", "channel": "video", "error": str(exc)}
                )
            else:
                logger.info("device video websocket disconnected: %s", exc)
        except Exception as exc:
            logger.exception("device video websocket failed")
            await manager.broadcast({"kind": "device_error", "channel": "video", "error": str(exc)})
        finally:
            generation = await manager.clear_device_ws("video", ws)
            if generation is not None:
                await manager.broadcast(
                    manager.device_connection_event(
                        channel="video",
                        connected=False,
                        generation=generation,
                    )
                )

    @app.websocket("/ws/device/audio-up")
    async def ws_device_audio(ws: WebSocket) -> None:
        try:
            await ws.accept()
            generation = await manager.replace_device_ws("audio", ws)
            if not await manager.device_ws_is_current("audio", ws):
                return
            if generation is not None:
                await manager.broadcast(
                    manager.device_connection_event(
                        channel="audio",
                        connected=True,
                        generation=generation,
                    )
                )
            while True:
                if not await manager.device_ws_is_current("audio", ws):
                    return
                packet = await _receive_current_device_packet(
                    ws,
                    manager=manager,
                    channel="audio",
                    max_payload_bytes=AUDIO_MAX_PAYLOAD_BYTES,
                )
                if packet is None:
                    continue
                await manager.handle_audio_packet(packet)
                await asr.push_pcm16(packet.payload)
        except WebSocketDisconnect:
            logger.info("device audio websocket disconnected")
        except RuntimeError as exc:
            if not _is_websocket_state_error(exc):
                logger.exception("device audio websocket failed")
                await manager.broadcast(
                    {"kind": "device_error", "channel": "audio", "error": str(exc)}
                )
            else:
                logger.info("device audio websocket disconnected: %s", exc)
        except Exception as exc:
            logger.exception("device audio websocket failed")
            await manager.broadcast({"kind": "device_error", "channel": "audio", "error": str(exc)})
        finally:
            generation = await manager.clear_device_ws("audio", ws)
            if generation is not None:
                await manager.broadcast(
                    manager.device_connection_event(
                        channel="audio",
                        connected=False,
                        generation=generation,
                    )
                )

    return app
