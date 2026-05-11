from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from aiglasses.asr import AsrService
from aiglasses.config import AppConfig
from aiglasses.device import DeviceManager, clamp_target_video_fps
from aiglasses.navigation import NavigationStateMachine
from aiglasses.protocol import MAX_PAYLOAD_BYTES, Packet, ProtocolError
from aiglasses.speech import DashscopeTtsSpeechSink, LocalTtsSpeechSink, SpeechHub, UiSpeechSink
from aiglasses.vision import VisionPipeline


logger = logging.getLogger("aiglasses.web")

CONTROL_MAX_PAYLOAD_BYTES = 8 * 1024
AUDIO_MAX_PAYLOAD_BYTES = 64 * 1024
VIDEO_MAX_PAYLOAD_BYTES = MAX_PAYLOAD_BYTES


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
        try:
            yield
        finally:
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
            await manager.replace_device_ws("control", ws)
            if not await manager.device_ws_is_current("control", ws):
                return
            await manager.broadcast({"kind": "device", "channel": "control", "connected": True})
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
            if await manager.clear_device_ws("control", ws):
                await manager.broadcast(
                    {"kind": "device", "channel": "control", "connected": False}
                )

    @app.websocket("/ws/device/video")
    async def ws_device_video(ws: WebSocket) -> None:
        try:
            await ws.accept()
            await manager.replace_device_ws("video", ws)
            if not await manager.device_ws_is_current("video", ws):
                return
            await manager.broadcast({"kind": "device", "channel": "video", "connected": True})
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
            if await manager.clear_device_ws("video", ws):
                await manager.broadcast(
                    {"kind": "device", "channel": "video", "connected": False}
                )

    @app.websocket("/ws/device/audio-up")
    async def ws_device_audio(ws: WebSocket) -> None:
        try:
            await ws.accept()
            await manager.replace_device_ws("audio", ws)
            if not await manager.device_ws_is_current("audio", ws):
                return
            await manager.broadcast({"kind": "device", "channel": "audio", "connected": True})
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
            if await manager.clear_device_ws("audio", ws):
                await manager.broadcast(
                    {"kind": "device", "channel": "audio", "connected": False}
                )

    return app
