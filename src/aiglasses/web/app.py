from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import hmac
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
RTP_HEADER = struct.Struct("!BBHII")
RTP_JPEG_HEADER = struct.Struct("!B3sBBBB")
RTP_JPEG_QTABLE_HEADER = struct.Struct("!BBH")
RTP_VERSION = 2
RTP_JPEG_PAYLOAD_TYPE = 26
RTP_JPEG_DYNAMIC_Q = 255
RTP_TIMESTAMP_HALF_RANGE = 1 << 31
RTP_EXTENSION_HEADER = struct.Struct("!HH")
RTP_AUTH_EXTENSION_PROFILE = 0xA147
RTP_AUTH_EXTENSION_MAGIC = b"AGLA"
RTP_AUTH_TAG_BYTES = 16
RTP_AUTH_EXTENSION_BYTES = len(RTP_AUTH_EXTENSION_MAGIC) + RTP_AUTH_TAG_BYTES
RTP_AUTH_EXTENSION_WORDS = RTP_AUTH_EXTENSION_BYTES // 4

JPEG_STD_DHT = bytes.fromhex(
    "ffc401a2"
    "0000010501010101010100000000000000000102030405060708090a0b"
    "100002010303020403050504040000017d0102030004110512213141061351610722"
    "7114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a"
    "3435363738393a434445464748494a535455565758595a636465666768696a7374"
    "75767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aa"
    "b2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3"
    "e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "0100030101010101010101010000000000000102030405060708090a0b"
    "110002010204040304070504040001027700010203110405213106124151076171"
    "1322328108144291a1b1c109233352f0156272d10a162434e125f11718191a26"
    "2728292a35363738393a434445464748494a535455565758595a636465666768"
    "696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5"
    "a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8"
    "d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9fa"
)


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
    # RTP/JPEG over RTP. This accepts the RFC 2435 variants emitted by the ESP32 firmware:
    # baseline JPEG type 0/1 with dynamic 8-bit quantization tables.
    def __init__(self, manager: DeviceManager, config: AppConfig):
        self.manager = manager
        self.config = config
        self.frames: dict[int, dict[str, Any]] = {}
        self.active_session_id: int | None = None
        self.retired_session_ids: set[int] = set()
        self.last_completed_timestamp: int | None = None
        self.video_auth_key = bytes.fromhex(config.device.transport.video_auth_key_hex)
        self.lock = asyncio.Lock()

    async def handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            if len(data) < RTP_HEADER.size + RTP_EXTENSION_HEADER.size + RTP_AUTH_EXTENSION_BYTES:
                raise ProtocolError("rtp/jpeg packet too short")
            first, second, _sequence, timestamp, ssrc = RTP_HEADER.unpack(data[: RTP_HEADER.size])
            version = first >> 6
            has_padding = bool(first & 0x20)
            has_extension = bool(first & 0x10)
            csrc_count = first & 0x0F
            marker = bool(second & 0x80)
            payload_type = second & 0x7F
            if version != RTP_VERSION:
                raise ProtocolError(f"unsupported rtp version: {version}")
            if has_padding or csrc_count:
                raise ProtocolError("unsupported rtp padding/csrc")
            if payload_type != RTP_JPEG_PAYLOAD_TYPE:
                raise ProtocolError(f"unexpected rtp payload type: {payload_type}")
            jpeg_start = self._authenticated_payload_start(data, has_extension)
            if len(data) < jpeg_start + RTP_JPEG_HEADER.size:
                raise ProtocolError("rtp/jpeg header truncated")
            (
                _type_specific,
                fragment_offset_bytes,
                jpeg_type,
                quality,
                width_blocks,
                height_blocks,
            ) = RTP_JPEG_HEADER.unpack(data[jpeg_start : jpeg_start + RTP_JPEG_HEADER.size])
            if jpeg_type not in {0, 1}:
                raise ProtocolError(f"unsupported rtp/jpeg type: {jpeg_type}")
            if quality != RTP_JPEG_DYNAMIC_Q:
                raise ProtocolError("only rtp/jpeg dynamic quantization tables are supported")
            if width_blocks == 0 or height_blocks == 0:
                raise ProtocolError("invalid rtp/jpeg dimensions")
            fragment_offset = int.from_bytes(fragment_offset_bytes, "big")
            payload_start = jpeg_start + RTP_JPEG_HEADER.size
            quant_tables: bytes | None = None
            if fragment_offset == 0:
                if len(data) < payload_start + RTP_JPEG_QTABLE_HEADER.size:
                    raise ProtocolError("rtp/jpeg quantization header missing")
                _mbz, precision, qtable_len = RTP_JPEG_QTABLE_HEADER.unpack(
                    data[payload_start : payload_start + RTP_JPEG_QTABLE_HEADER.size]
                )
                if precision != 0:
                    raise ProtocolError("rtp/jpeg 16-bit quantization tables are not supported")
                if qtable_len != 128:
                    raise ProtocolError("rtp/jpeg quantization table length must be 128")
                qtable_start = payload_start + RTP_JPEG_QTABLE_HEADER.size
                qtable_end = qtable_start + qtable_len
                if len(data) < qtable_end:
                    raise ProtocolError("rtp/jpeg quantization table truncated")
                quant_tables = data[qtable_start:qtable_end]
                payload_start = qtable_end
            chunk = data[payload_start:]
            chunk_bytes = self.config.device.transport.video_payload_bytes
            if not chunk:
                raise ProtocolError("empty rtp/jpeg fragment")
            if len(chunk) > chunk_bytes:
                raise ProtocolError("video fragment exceeds configured payload size")
            fragment_end = fragment_offset + len(chunk)
            if fragment_end > VIDEO_MAX_PAYLOAD_BYTES:
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
                self.active_session_id = ssrc
            elif ssrc != self.active_session_id:
                if ssrc in self.retired_session_ids:
                    return
                self.retired_session_ids.add(self.active_session_id)
                self.active_session_id = ssrc
                self.last_completed_timestamp = None
                self.frames.clear()
            self._drop_expired(now)
            if self.last_completed_timestamp is not None and not self._rtp_timestamp_newer(
                timestamp, self.last_completed_timestamp
            ):
                return
            frame = self.frames.get(timestamp)
            if frame is None:
                frame = {
                    "created_at": now,
                    "fragments": {},
                    "received": 0,
                    "final_len": None,
                    "jpeg_type": jpeg_type,
                    "width_blocks": width_blocks,
                    "height_blocks": height_blocks,
                    "quant_tables": None,
                }
                self.frames[timestamp] = frame
            if (
                frame["jpeg_type"] != jpeg_type
                or frame["width_blocks"] != width_blocks
                or frame["height_blocks"] != height_blocks
            ):
                self.frames.pop(timestamp, None)
                logger.warning("dropping inconsistent rtp/jpeg frame ts=%s from %s", timestamp, addr)
                return
            fragments: dict[int, bytes] = frame["fragments"]
            existing = fragments.get(fragment_offset)
            if existing is not None:
                if existing != chunk or (
                    quant_tables is not None
                    and frame["quant_tables"] is not None
                    and frame["quant_tables"] != quant_tables
                ):
                    self.frames.pop(timestamp, None)
                    logger.warning(
                        "dropping inconsistent duplicate rtp/jpeg fragment ts=%s from %s",
                        timestamp,
                        addr,
                    )
                return
            if quant_tables is not None:
                if frame["quant_tables"] is not None and frame["quant_tables"] != quant_tables:
                    self.frames.pop(timestamp, None)
                    logger.warning(
                        "dropping inconsistent rtp/jpeg quantization tables ts=%s from %s",
                        timestamp,
                        addr,
                    )
                    return
                frame["quant_tables"] = quant_tables
            if any(
                fragment_offset < offset + len(payload) and fragment_end > offset
                for offset, payload in fragments.items()
            ):
                self.frames.pop(timestamp, None)
                logger.warning("dropping overlapping rtp/jpeg frame ts=%s from %s", timestamp, addr)
                return
            fragments[fragment_offset] = chunk
            frame["received"] += len(chunk)
            if marker:
                frame["final_len"] = fragment_end
            final_len: int | None = frame["final_len"]
            if final_len is not None and frame["received"] >= final_len:
                complete_payload = self._assemble_frame(fragments, final_len)
                self.frames.pop(timestamp, None)
                if complete_payload is not None:
                    quant_tables = frame["quant_tables"]
                    if quant_tables is None:
                        return
                    complete_payload = self._build_jpeg(
                        complete_payload,
                        jpeg_type=frame["jpeg_type"],
                        width_blocks=frame["width_blocks"],
                        height_blocks=frame["height_blocks"],
                        quant_tables=quant_tables,
                    )
                    self.last_completed_timestamp = timestamp

        if complete_payload is not None:
            self.manager.mark_video_udp_seen()
            await self.manager.handle_video_packet(
                Packet(PacketType.VIDEO_JPEG, timestamp, int(time.time() * 1000), complete_payload)
            )

    def _assemble_frame(self, fragments: dict[int, bytes], final_len: int) -> bytes | None:
        offset = 0
        parts: list[bytes] = []
        for fragment_offset in sorted(fragments):
            if fragment_offset != offset:
                return None
            payload = fragments[fragment_offset]
            parts.append(payload)
            offset += len(payload)
        if offset != final_len:
            return None
        return b"".join(parts)

    @staticmethod
    def _authentication_data(data: bytes, extension_start: int, payload_start: int) -> bytes:
        tag_start = extension_start + RTP_EXTENSION_HEADER.size + len(RTP_AUTH_EXTENSION_MAGIC)
        return b"".join(
            [
                data[:extension_start],
                data[extension_start:tag_start],
                data[payload_start:],
            ]
        )

    def _authenticated_payload_start(self, data: bytes, has_extension: bool) -> int:
        if not has_extension:
            raise ProtocolError("rtp authentication extension missing")
        extension_start = RTP_HEADER.size
        extension_header_end = extension_start + RTP_EXTENSION_HEADER.size
        profile, extension_words = RTP_EXTENSION_HEADER.unpack(
            data[extension_start:extension_header_end]
        )
        extension_len = extension_words * 4
        extension_end = extension_header_end + extension_len
        if len(data) < extension_end:
            raise ProtocolError("rtp authentication extension truncated")
        if profile != RTP_AUTH_EXTENSION_PROFILE:
            raise ProtocolError("unexpected rtp authentication extension profile")
        if extension_words != RTP_AUTH_EXTENSION_WORDS:
            raise ProtocolError("invalid rtp authentication extension length")
        extension = data[extension_header_end:extension_end]
        if not extension.startswith(RTP_AUTH_EXTENSION_MAGIC):
            raise ProtocolError("invalid rtp authentication extension magic")
        received_tag = extension[len(RTP_AUTH_EXTENSION_MAGIC) :]
        auth_data = self._authentication_data(data, extension_start, extension_end)
        expected_tag = hmac.new(self.video_auth_key, auth_data, hashlib.sha256).digest()[
            :RTP_AUTH_TAG_BYTES
        ]
        if not hmac.compare_digest(received_tag, expected_tag):
            raise ProtocolError("invalid rtp authentication tag")
        return extension_end

    @staticmethod
    def _rtp_timestamp_newer(timestamp: int, previous: int) -> bool:
        delta = (timestamp - previous) & 0xFFFFFFFF
        return 0 < delta < RTP_TIMESTAMP_HALF_RANGE

    def _build_jpeg(
        self,
        scan_data: bytes,
        *,
        jpeg_type: int,
        width_blocks: int,
        height_blocks: int,
        quant_tables: bytes,
    ) -> bytes:
        width = width_blocks * 8
        height = height_blocks * 8
        luma_sampling = 0x21 if jpeg_type == 0 else 0x22
        return b"".join(
            [
                b"\xff\xd8",
                self._dqt_segment(0, quant_tables[:64]),
                self._dqt_segment(1, quant_tables[64:128]),
                JPEG_STD_DHT,
                self._sof0_segment(width, height, luma_sampling),
                self._sos_segment(),
                scan_data,
                b"\xff\xd9",
            ]
        )

    @staticmethod
    def _dqt_segment(table_id: int, table: bytes) -> bytes:
        return b"\xff\xdb" + struct.pack("!H", 67) + bytes([table_id]) + table

    @staticmethod
    def _sof0_segment(width: int, height: int, luma_sampling: int) -> bytes:
        return (
            b"\xff\xc0"
            + struct.pack("!H", 17)
            + bytes(
                [
                    8,
                    (height >> 8) & 0xFF,
                    height & 0xFF,
                    (width >> 8) & 0xFF,
                    width & 0xFF,
                    3,
                    1,
                    luma_sampling,
                    0,
                    2,
                    0x11,
                    1,
                    3,
                    0x11,
                    1,
                ]
            )
        )

    @staticmethod
    def _sos_segment() -> bytes:
        return b"\xff\xda" + struct.pack("!H", 12) + bytes(
            [3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0]
        )

    def _drop_expired(self, now: float) -> None:
        timeout_ms = max(
            self.config.device.transport.video_frame_timeout_ms,
            self.config.device.transport.video_reorder_window_ms,
        )
        timeout_s = timeout_ms / 1000
        expired = [
            timestamp
            for timestamp, frame in self.frames.items()
            if now - frame["created_at"] > timeout_s
        ]
        for timestamp in expired:
            self.frames.pop(timestamp, None)


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
