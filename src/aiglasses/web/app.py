from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aiglasses.asr import AsrService
from aiglasses.config import AppConfig
from aiglasses.device import DeviceManager
from aiglasses.navigation import NavigationStateMachine
from aiglasses.protocol import Packet
from aiglasses.speech import SpeechHub, UiSpeechSink
from aiglasses.vision import VisionPipeline


logger = logging.getLogger("aiglasses.web")


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title=config.web.title)
    web_dir = Path(__file__).resolve().parent
    static_dir = web_dir / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    speech = SpeechHub()
    navigation = NavigationStateMachine()
    manager = DeviceManager(VisionPipeline(config), navigation, speech)
    speech.add_sink(UiSpeechSink(manager.broadcast))
    asr = AsrService(config.asr, lambda text: manager.handle_command_text(text, source="asr"))

    app.state.config = config
    app.state.manager = manager
    app.state.asr = asr

    @app.on_event("startup")
    async def startup() -> None:
        await asr.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await asr.stop()

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

    @app.post("/api/v1/commands")
    async def command(payload: Annotated[dict, Body()]) -> dict:
        text = str(payload.get("text", ""))
        return await manager.handle_command_text(text, source="debug")

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
        await ws.accept()
        manager.control_ws = ws
        await manager.broadcast({"kind": "device", "channel": "control", "connected": True})
        try:
            while True:
                data = await ws.receive_bytes()
                await manager.handle_control_packet(Packet.unpack(data))
        except WebSocketDisconnect:
            logger.info("device control websocket disconnected")
        except Exception as exc:
            logger.exception("device control websocket failed")
            await manager.broadcast({"kind": "device_error", "channel": "control", "error": str(exc)})
        finally:
            if manager.control_ws is ws:
                manager.control_ws = None
                await manager.broadcast(
                    {"kind": "device", "channel": "control", "connected": False}
                )

    @app.websocket("/ws/device/video")
    async def ws_device_video(ws: WebSocket) -> None:
        await ws.accept()
        manager.video_ws = ws
        await manager.broadcast({"kind": "device", "channel": "video", "connected": True})
        try:
            while True:
                data = await ws.receive_bytes()
                try:
                    packet = Packet.unpack(data)
                except Exception as exc:
                    logger.warning("dropping bad video packet: %s", exc)
                    await manager.broadcast(
                        {"kind": "device_error", "channel": "video", "error": str(exc)}
                    )
                    continue
                await manager.handle_video_packet(packet)
        except WebSocketDisconnect:
            logger.info("device video websocket disconnected")
        except Exception as exc:
            logger.exception("device video websocket failed")
            await manager.broadcast({"kind": "device_error", "channel": "video", "error": str(exc)})
        finally:
            if manager.video_ws is ws:
                manager.video_ws = None
                await manager.broadcast(
                    {"kind": "device", "channel": "video", "connected": False}
                )

    @app.websocket("/ws/device/audio-up")
    async def ws_device_audio(ws: WebSocket) -> None:
        await ws.accept()
        manager.audio_ws = ws
        await manager.broadcast({"kind": "device", "channel": "audio", "connected": True})
        try:
            while True:
                data = await ws.receive_bytes()
                packet = Packet.unpack(data)
                await manager.handle_audio_packet(packet)
                await asr.push_pcm16(packet.payload)
        except WebSocketDisconnect:
            logger.info("device audio websocket disconnected")
        except Exception as exc:
            logger.exception("device audio websocket failed")
            await manager.broadcast({"kind": "device_error", "channel": "audio", "error": str(exc)})
        finally:
            if manager.audio_ws is ws:
                manager.audio_ws = None
                await manager.broadcast(
                    {"kind": "device", "channel": "audio", "connected": False}
                )

    return app
