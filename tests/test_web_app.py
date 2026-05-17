import asyncio
import hashlib
import hmac
import io
import struct
import unittest
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from aiglasses.config import AppConfig, SpeechConfig
from aiglasses.config.settings import AsrConfig, DeviceCaptureConfig
from aiglasses.config.settings import DeviceAudioDownConfig, DeviceConfig
from aiglasses.config.settings import DeviceTransportConfig
from aiglasses.protocol import Packet, PacketType
from aiglasses.web.app import (
    CONTROL_MAX_PAYLOAD_BYTES,
    RTP_AUTH_EXTENSION_MAGIC,
    RTP_AUTH_EXTENSION_PROFILE,
    RTP_AUTH_EXTENSION_WORDS,
    RTP_AUTH_TAG_BYTES,
    RTP_EXTENSION_HEADER,
    RTP_HEADER,
    RTP_JPEG_DYNAMIC_Q,
    RTP_JPEG_HEADER,
    RTP_JPEG_PAYLOAD_TYPE,
    UdpVideoReassembler,
    _receive_current_device_packet,
    _unpack_device_packet,
    validate_speech_config,
)

TEST_VIDEO_AUTH_KEY_HEX = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


class FakeBroadcastManager:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.video_udp_seen_count = 0

    async def broadcast(self, message: dict) -> None:
        self.messages.append(message)

    def mark_video_udp_seen(self) -> None:
        self.video_udp_seen_count += 1

    async def handle_video_packet(self, packet: Packet, **kwargs: object) -> None:
        self.messages.append(
            {
                "kind": "video",
                "seq": packet.seq,
                "timestamp_ms": packet.timestamp_ms,
                "payload": packet.payload,
                "video_session": kwargs.get("video_session"),
            }
        )


class FakeDeviceWebSocket:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.receive_started = asyncio.Event()
        self.allow_receive = asyncio.Event()

    async def receive_bytes(self) -> bytes:
        self.receive_started.set()
        await self.allow_receive.wait()
        return self.data


class FakeCurrentManager:
    def __init__(self, current: bool = True) -> None:
        self.current = current
        self.messages: list[dict] = []

    async def device_ws_is_current(self, channel: str, ws: object) -> bool:
        return self.current

    async def broadcast(self, message: dict) -> None:
        self.messages.append(message)


def rtp_jpeg_packet(
    *,
    ssrc: int = 1,
    timestamp: int,
    offset: int,
    sequence: int,
    marker: bool,
    payload: bytes,
    version: int = 2,
    payload_type: int = RTP_JPEG_PAYLOAD_TYPE,
    first_byte_flags: int = 0,
    jpeg_type: int = 1,
    quality: int = RTP_JPEG_DYNAMIC_Q,
    width_blocks: int = 100,
    height_blocks: int = 75,
    quant_tables: bytes | None = None,
    auth_key_hex: str = TEST_VIDEO_AUTH_KEY_HEX,
    authenticated: bool = True,
) -> bytes:
    first = ((version & 0x03) << 6) | first_byte_flags
    if authenticated:
        first |= 0x10
    rtp_header = RTP_HEADER.pack(
        first,
        (0x80 if marker else 0x00) | payload_type,
        sequence,
        timestamp,
        ssrc,
    )
    jpeg_header = RTP_JPEG_HEADER.pack(
        0,
        offset.to_bytes(3, "big"),
        jpeg_type,
        quality,
        width_blocks,
        height_blocks,
    )
    quant_header = b""
    if offset == 0 and quality == RTP_JPEG_DYNAMIC_Q:
        tables = bytes(range(128)) if quant_tables is None else quant_tables
        quant_header = struct.pack("!BBH", 0, 0, len(tables)) + tables
    payload_bytes = jpeg_header + quant_header + payload
    if not authenticated:
        return rtp_header + payload_bytes
    return authenticated_rtp_packet(rtp_header, payload_bytes, auth_key_hex=auth_key_hex)


def authenticated_rtp_packet(
    rtp_header: bytes,
    payload_bytes: bytes,
    *,
    auth_key_hex: str = TEST_VIDEO_AUTH_KEY_HEX,
) -> bytes:
    extension_header = RTP_EXTENSION_HEADER.pack(
        RTP_AUTH_EXTENSION_PROFILE,
        RTP_AUTH_EXTENSION_WORDS,
    )
    tag = hmac.new(
        bytes.fromhex(auth_key_hex),
        rtp_header + extension_header + RTP_AUTH_EXTENSION_MAGIC + payload_bytes,
        hashlib.sha256,
    ).digest()[:RTP_AUTH_TAG_BYTES]
    return rtp_header + extension_header + RTP_AUTH_EXTENSION_MAGIC + tag + payload_bytes


def app_config_with_video_auth(**transport_overrides: object) -> AppConfig:
    transport = DeviceTransportConfig(
        video_auth_key_hex=TEST_VIDEO_AUTH_KEY_HEX,
        **transport_overrides,
    )
    return AppConfig(path=Path("config.toml"), device=DeviceConfig(transport=transport))


def rtp_jpeg_parts_from_jpeg(jpeg: bytes) -> tuple[int, int, int, bytes, bytes]:
    quant_tables: dict[int, bytes] = {}
    jpeg_type: int | None = None
    width_blocks: int | None = None
    height_blocks: int | None = None
    pos = 2
    if not jpeg.startswith(b"\xff\xd8"):
        raise AssertionError("fixture is not a JPEG")
    while pos + 4 <= len(jpeg):
        while pos < len(jpeg) and jpeg[pos] != 0xFF:
            pos += 1
        while pos + 1 < len(jpeg) and jpeg[pos + 1] == 0xFF:
            pos += 1
        marker = jpeg[pos + 1]
        pos += 2
        if marker == 0xDA:
            segment_len = int.from_bytes(jpeg[pos : pos + 2], "big")
            scan_start = pos + segment_len
            scan_end = len(jpeg) - 2 if jpeg.endswith(b"\xff\xd9") else len(jpeg)
            return (
                jpeg_type if jpeg_type is not None else 1,
                width_blocks if width_blocks is not None else 1,
                height_blocks if height_blocks is not None else 1,
                quant_tables[0] + quant_tables[1],
                jpeg[scan_start:scan_end],
            )
        segment_len = int.from_bytes(jpeg[pos : pos + 2], "big")
        segment = jpeg[pos + 2 : pos + segment_len]
        if marker == 0xDB:
            table_pos = 0
            while table_pos + 65 <= len(segment):
                info = segment[table_pos]
                table_pos += 1
                precision = info >> 4
                table_id = info & 0x0F
                if precision != 0:
                    raise AssertionError("fixture uses non-8-bit quantization tables")
                quant_tables[table_id] = segment[table_pos : table_pos + 64]
                table_pos += 64
        elif marker == 0xC0:
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            sampling = segment[7]
            if sampling == 0x21:
                jpeg_type = 0
            elif sampling == 0x22:
                jpeg_type = 1
            else:
                raise AssertionError(f"unsupported fixture sampling: {sampling:#x}")
            width_blocks = (width + 7) // 8
            height_blocks = (height + 7) // 8
        pos += segment_len
    raise AssertionError("fixture has no SOS segment")


class WebAppTests(unittest.TestCase):
    def test_web_app_imports(self) -> None:
        from aiglasses.web.app import create_app

        self.assertTrue(callable(create_app))

    def test_frame_jpeg_endpoint_returns_latest_frame_bytes(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}
        app.state.manager.last_frame_jpeg = b"\xff\xd8jpeg\xff\xd9"
        app.state.manager.frame_count = 12

        with TestClient(app) as client:
            response = client.get("/api/v1/frame.jpg")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-frame-count"], "12")
        self.assertEqual(response.content, b"\xff\xd8jpeg\xff\xd9")

    def test_disconnect_device_endpoint_returns_disconnect_result(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        async def disconnect_device() -> dict:
            return {"disconnected": ["control"], "state": {}}

        app.state.manager.disconnect_device = disconnect_device

        with TestClient(app) as client:
            response = client.post("/api/v1/device/disconnect")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"disconnected": ["control"], "state": {}})

    def test_create_app_initializes_device_config_from_app_config(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(
            AppConfig(
                path=Path("config.toml"),
                asr=AsrConfig(enabled=False),
                device=DeviceConfig(
                    capture=DeviceCaptureConfig(
                        video_fps=9,
                        jpeg_quality=20,
                        camera_profile="default",
                    )
                ),
            )
        )

        self.assertEqual(
            app.state.manager.device_config_payload(),
            {
                "kind": "config",
                "target_fps": 9,
                "jpeg_quality": 20,
                "camera_profile": "default",
                "ae_level": -1,
                "saturation": 1,
                "contrast": 1,
                "sharpness": 1,
                "gainceiling": 4,
            },
        )

    def test_device_config_accepts_target_fps_and_legacy_video_fps_together(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/device/config",
                json={"target_fps": 7, "video_fps": 3},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["config"]["target_fps"], 7)

    def test_tuning_update_rejects_invalid_payload_without_partial_mutation(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}
        original = app.state.manager.vision.tuning.to_dict()

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={"traffic_go_min_conf": 0.2, "crossing_green_required_frames": "bad"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(app.state.manager.vision.tuning.to_dict(), original)
        self.assertIs(app.state.manager.navigation.tuning, app.state.manager.vision.tuning)

    def test_tuning_update_rejects_invalid_boolean_without_partial_mutation(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}
        original = app.state.manager.vision.tuning.to_dict()

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={"traffic_go_min_conf": 0.2, "traffic_filter_enabled": "maybe"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(app.state.manager.vision.tuning.to_dict(), original)

    def test_tuning_includes_crossing_obstacles_default_false(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.get("/api/v1/debug/tuning")

        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json()["tuning"]["crossing_obstacles_enabled"], False)
        self.assertEqual(response.json()["tuning"]["traffic_light_conf"], 0.2)
        self.assertEqual(response.json()["tuning"]["traffic_go_min_conf"], 0.2)
        self.assertEqual(response.json()["tuning"]["traffic_stop_min_conf"], 0.2)
        self.assertEqual(response.json()["tuning"]["traffic_yellow_min_conf"], 0.9)
        self.assertEqual(response.json()["tuning"]["crossing_alignment_offset_max"], 0.15)
        self.assertEqual(response.json()["tuning"]["crossing_start_bottom_min"], 0.60)
        self.assertEqual(response.json()["tuning"]["crossing_mid_bottom_min"], 0.45)
        self.assertEqual(response.json()["tuning"]["crossing_completion_bottom_max"], 0.35)
        self.assertEqual(response.json()["tuning"]["crossing_completion_min_active_frames"], 4)
        self.assertEqual(response.json()["tuning"]["crossing_completion_min_active_seconds"], 3.0)
        self.assertEqual(response.json()["tuning"]["crossing_completion_lost_frames"], 10)
        self.assertEqual(response.json()["tuning"]["crossing_completion_required_frames"], 3)
        self.assertEqual(response.json()["tuning"]["crossing_wait_signal_suppress_frames"], 3)
        self.assertEqual(response.json()["tuning"]["crossing_obstacle_suppress_frames"], 3)
        self.assertEqual(response.json()["tuning"]["crossing_active_timeout_seconds"], 45.0)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_conf"], 0.2)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_min_area_ratio"], 0.005)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_x_min"], 0.05)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_x_max"], 0.95)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_alert_bottom_min"], 0.25)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_stop_bottom_min"], 0.55)

    def test_tuning_update_accepts_crossing_obstacles_enabled(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={"crossing_obstacles_enabled": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json()["tuning"]["crossing_obstacles_enabled"], True)
        self.assertIs(app.state.manager.vision.tuning.crossing_obstacles_enabled, True)
        self.assertIs(app.state.manager.navigation.tuning, app.state.manager.vision.tuning)

    def test_tuning_update_accepts_crosswalk_detection_thresholds(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={
                    "crosswalk_detection_conf": 0.40,
                    "crosswalk_detection_min_area_ratio": 0.010,
                    "crosswalk_detection_x_min": 0.10,
                    "crosswalk_detection_x_max": 0.90,
                    "crosswalk_detection_alert_bottom_min": 0.30,
                    "crosswalk_detection_stop_bottom_min": 0.60,
                    "crossing_alignment_offset_max": 0.20,
                    "crossing_start_bottom_min": 0.65,
                    "crossing_mid_bottom_min": 0.50,
                    "crossing_completion_bottom_max": 0.30,
                    "crossing_completion_min_active_frames": 6,
                    "crossing_completion_min_active_seconds": 4.5,
                    "crossing_completion_lost_frames": 12,
                    "crossing_completion_required_frames": 4,
                    "crossing_wait_signal_suppress_frames": 5,
                    "crossing_obstacle_suppress_frames": 6,
                    "crossing_active_timeout_seconds": 50.0,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_conf"], 0.40)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_min_area_ratio"], 0.010)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_x_min"], 0.10)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_x_max"], 0.90)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_alert_bottom_min"], 0.30)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_stop_bottom_min"], 0.60)
        self.assertEqual(response.json()["tuning"]["crossing_alignment_offset_max"], 0.20)
        self.assertEqual(response.json()["tuning"]["crossing_start_bottom_min"], 0.65)
        self.assertEqual(response.json()["tuning"]["crossing_mid_bottom_min"], 0.50)
        self.assertEqual(response.json()["tuning"]["crossing_completion_bottom_max"], 0.30)
        self.assertEqual(response.json()["tuning"]["crossing_completion_min_active_frames"], 6)
        self.assertEqual(response.json()["tuning"]["crossing_completion_min_active_seconds"], 4.5)
        self.assertEqual(response.json()["tuning"]["crossing_completion_lost_frames"], 12)
        self.assertEqual(response.json()["tuning"]["crossing_completion_required_frames"], 4)
        self.assertEqual(response.json()["tuning"]["crossing_wait_signal_suppress_frames"], 5)
        self.assertEqual(response.json()["tuning"]["crossing_obstacle_suppress_frames"], 6)
        self.assertEqual(response.json()["tuning"]["crossing_active_timeout_seconds"], 50.0)
        self.assertIs(app.state.manager.navigation.tuning, app.state.manager.vision.tuning)

    def test_tuning_update_swaps_crosswalk_detection_x_range(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={"crosswalk_detection_x_min": 0.90, "crosswalk_detection_x_max": 0.10},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_x_min"], 0.10)
        self.assertEqual(response.json()["tuning"]["crosswalk_detection_x_max"], 0.90)

    def test_tuning_update_orders_crossing_bottom_thresholds(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={
                    "crossing_completion_bottom_max": 0.80,
                    "crossing_mid_bottom_min": 0.20,
                    "crosswalk_detection_stop_bottom_min": 0.70,
                    "crossing_start_bottom_min": 0.40,
                },
            )

        tuning = response.json()["tuning"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tuning["crossing_completion_bottom_max"], 0.20)
        self.assertEqual(tuning["crossing_mid_bottom_min"], 0.40)
        self.assertEqual(tuning["crosswalk_detection_stop_bottom_min"], 0.70)
        self.assertEqual(tuning["crossing_start_bottom_min"], 0.80)

    def test_tuning_update_clamps_single_crossing_bottom_field_without_drifting(
        self,
    ) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={"crossing_completion_bottom_max": 0.50},
            )

        tuning = response.json()["tuning"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tuning["crossing_completion_bottom_max"], 0.45)
        self.assertEqual(tuning["crossing_mid_bottom_min"], 0.45)
        self.assertEqual(tuning["crosswalk_detection_stop_bottom_min"], 0.55)
        self.assertEqual(tuning["crossing_start_bottom_min"], 0.60)

    def test_tuning_update_clamps_crosswalk_alert_not_above_stop(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/debug/tuning",
                json={
                    "crosswalk_detection_alert_bottom_min": 0.80,
                    "crosswalk_detection_stop_bottom_min": 0.55,
                },
            )

        tuning = response.json()["tuning"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tuning["crosswalk_detection_alert_bottom_min"], 0.55)
        self.assertEqual(tuning["crosswalk_detection_stop_bottom_min"], 0.55)

    def test_recording_endpoints_delegate_to_manager(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}
        app.state.manager.recording_status = lambda: {"active": False, "frame_count": 0}

        async def start_recording() -> dict:
            return {"active": True, "frame_count": 0}

        async def stop_recording() -> dict:
            return {"active": False, "frame_count": 4}

        app.state.manager.start_recording = start_recording
        app.state.manager.stop_recording = stop_recording

        with TestClient(app) as client:
            status_response = client.get("/api/v1/recording/status")
            start_response = client.post("/api/v1/recording/start")
            stop_response = client.post("/api/v1/recording/stop")

        self.assertEqual(status_response.json(), {"recording": {"active": False, "frame_count": 0}})
        self.assertEqual(start_response.json(), {"recording": {"active": True, "frame_count": 0}})
        self.assertEqual(stop_response.json(), {"recording": {"active": False, "frame_count": 4}})

    def test_receive_current_device_packet_drops_packet_after_socket_is_superseded(self) -> None:
        raw = Packet(PacketType.VIDEO_JPEG, seq=1, timestamp_ms=2, payload=b"jpeg").pack()
        ws = FakeDeviceWebSocket(raw)
        manager = FakeCurrentManager(current=True)

        async def receive_after_supersede() -> object:
            receive_task = asyncio.create_task(
                _receive_current_device_packet(
                    ws,
                    manager=manager,
                    channel="video",
                    max_payload_bytes=CONTROL_MAX_PAYLOAD_BYTES,
                )
            )
            await ws.receive_started.wait()
            manager.current = False
            ws.allow_receive.set()
            return await receive_task

        packet = asyncio.run(receive_after_supersede())

        self.assertIsNone(packet)
        self.assertEqual(manager.messages, [])

    def test_receive_current_device_packet_returns_packet_for_current_socket(self) -> None:
        raw = Packet(PacketType.CONTROL_JSON, seq=1, timestamp_ms=2, payload=b"{}").pack()
        ws = FakeDeviceWebSocket(raw)
        manager = FakeCurrentManager(current=True)

        async def receive() -> object:
            receive_task = asyncio.create_task(
                _receive_current_device_packet(
                    ws,
                    manager=manager,
                    channel="control",
                    max_payload_bytes=CONTROL_MAX_PAYLOAD_BYTES,
                )
            )
            await ws.receive_started.wait()
            ws.allow_receive.set()
            return await receive_task

        packet = asyncio.run(receive())

        self.assertIsNotNone(packet)
        self.assertEqual(packet.payload, b"{}")
        self.assertEqual(manager.messages, [])

    def test_device_connection_event_includes_generation(self) -> None:
        from aiglasses.web.app import create_app

        app = create_app(AppConfig(path=Path("config.toml"), asr=AsrConfig(enabled=False)))
        app.state.manager.benchmark_processing_capacity = lambda: {"status": "ready"}

        with TestClient(app) as client:
            with client.websocket_connect("/ws/device/video"):
                pass

        self.assertGreaterEqual(app.state.manager.device_ws_generations["video"], 1)

    def test_device_speech_requires_audio_down(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=False)),
        )

        with self.assertRaisesRegex(ValueError, "audio_down"):
            validate_speech_config(config)

    def test_speech_mode_rejects_both(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="both"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        with self.assertRaisesRegex(ValueError, "speech.mode"):
            validate_speech_config(config)

    def test_device_speech_allows_audio_down(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        validate_speech_config(config)

    def test_device_speech_allows_local_provider(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device", provider="local"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        validate_speech_config(config)

    def test_device_speech_rejects_unknown_provider(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device", provider="unknown"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        with self.assertRaisesRegex(ValueError, "speech.provider"):
            validate_speech_config(config)

    def test_unpack_device_packet_broadcasts_and_drops_oversized_packet(self) -> None:
        manager = FakeBroadcastManager()
        raw = Packet(PacketType.CONTROL_JSON, seq=1, timestamp_ms=2, payload=b"{}").pack()

        packet = asyncio.run(
            _unpack_device_packet(
                raw,
                channel="control",
                max_payload_bytes=1,
                manager=manager,
            )
        )

        self.assertIsNone(packet)
        self.assertEqual(len(manager.messages), 1)
        self.assertEqual(manager.messages[0]["kind"], "device_error")
        self.assertEqual(manager.messages[0]["channel"], "control")
        self.assertIn("payload too large", manager.messages[0]["error"])

    def test_unpack_device_packet_returns_valid_packet(self) -> None:
        manager = FakeBroadcastManager()
        raw = Packet(PacketType.CONTROL_JSON, seq=1, timestamp_ms=2, payload=b"{}").pack()

        packet = asyncio.run(
            _unpack_device_packet(
                raw,
                channel="control",
                max_payload_bytes=CONTROL_MAX_PAYLOAD_BYTES,
                manager=manager,
            )
        )

        self.assertIsNotNone(packet)
        self.assertEqual(packet.payload, b"{}")
        self.assertEqual(manager.messages, [])

    def test_udp_video_reassembler_evicts_expired_frames(self) -> None:
        config = app_config_with_video_auth(
            video_frame_timeout_ms=10,
            video_reorder_window_ms=0,
        )
        reassembler = UdpVideoReassembler(FakeBroadcastManager(), config)
        reassembler.frames[1] = {"created_at": 0.0}

        reassembler._drop_expired(now=1.0)

        self.assertEqual(reassembler.frames, {})

    def test_udp_video_reassembler_completes_out_of_order_fragments(self) -> None:
        config = app_config_with_video_auth(video_payload_bytes=4)
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def run() -> None:
            fragments = [
                (4, 2, True, b"ef"),
                (0, 1, False, b"abcd"),
            ]
            for offset, sequence, marker, chunk in fragments:
                raw = rtp_jpeg_packet(
                    timestamp=90_000,
                    offset=offset,
                    sequence=sequence,
                    marker=marker,
                    payload=chunk,
                )
                await reassembler.handle_datagram(raw, ("127.0.0.1", 12345))

        asyncio.run(run())

        video_messages = [message for message in manager.messages if message["kind"] == "video"]
        self.assertEqual(len(video_messages), 1)
        self.assertTrue(video_messages[0]["payload"].startswith(b"\xff\xd8"))
        self.assertIn(b"abcdef", video_messages[0]["payload"])
        self.assertTrue(video_messages[0]["payload"].endswith(b"\xff\xd9"))
        self.assertEqual(manager.video_udp_seen_count, 1)

    def test_udp_video_reassembler_handles_duplicate_fragments(self) -> None:
        config = app_config_with_video_auth(video_payload_bytes=4)
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def send(offset: int, sequence: int, marker: bool, chunk: bytes) -> None:
            raw = rtp_jpeg_packet(
                timestamp=180_000,
                offset=offset,
                sequence=sequence,
                marker=marker,
                payload=chunk,
            )
            await reassembler.handle_datagram(raw, ("127.0.0.1", 12345))

        async def run() -> None:
            await send(0, 1, False, b"abcd")
            await send(0, 1, False, b"abcd")
            await send(4, 2, True, b"ef")

        asyncio.run(run())

        video_messages = [message for message in manager.messages if message["kind"] == "video"]
        self.assertEqual(len(video_messages), 1)
        self.assertIn(b"abcdef", video_messages[0]["payload"])

    def test_udp_video_reassembler_rejects_duplicate_first_fragment_with_changed_qtables(
        self,
    ) -> None:
        config = app_config_with_video_auth(video_payload_bytes=4)
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def run() -> None:
            await reassembler.handle_datagram(
                rtp_jpeg_packet(
                    timestamp=225_000,
                    offset=0,
                    sequence=1,
                    marker=False,
                    payload=b"abcd",
                    quant_tables=bytes(range(128)),
                ),
                ("127.0.0.1", 12345),
            )
            await reassembler.handle_datagram(
                rtp_jpeg_packet(
                    timestamp=225_000,
                    offset=0,
                    sequence=2,
                    marker=False,
                    payload=b"abcd",
                    quant_tables=bytes(reversed(range(128))),
                ),
                ("127.0.0.1", 12345),
            )

        asyncio.run(run())

        self.assertEqual(manager.messages, [])
        self.assertEqual(reassembler.frames, {})
        self.assertEqual(manager.video_udp_seen_count, 0)

    def test_udp_video_reassembler_rebuilds_decodable_jpeg(self) -> None:
        for subsampling, expected_type in ((1, 0), (2, 1)):
            with self.subTest(subsampling=subsampling):
                config = app_config_with_video_auth()
                manager = FakeBroadcastManager()
                reassembler = UdpVideoReassembler(manager, config)
                image = Image.new("RGB", (16, 16), (32, 96, 160))
                jpeg_file = io.BytesIO()
                image.save(jpeg_file, format="JPEG", quality=75, subsampling=subsampling)
                (
                    jpeg_type,
                    width_blocks,
                    height_blocks,
                    quant_tables,
                    scan_data,
                ) = rtp_jpeg_parts_from_jpeg(jpeg_file.getvalue())
                self.assertEqual(jpeg_type, expected_type)

                async def run() -> None:
                    await reassembler.handle_datagram(
                        rtp_jpeg_packet(
                            timestamp=240_000 + subsampling,
                            offset=0,
                            sequence=subsampling,
                            marker=True,
                            payload=scan_data,
                            jpeg_type=jpeg_type,
                            width_blocks=width_blocks,
                            height_blocks=height_blocks,
                            quant_tables=quant_tables,
                        ),
                        ("127.0.0.1", 12345),
                    )

                asyncio.run(run())

                video_messages = [
                    message for message in manager.messages if message["kind"] == "video"
                ]
                self.assertEqual(len(video_messages), 1)
                payload = video_messages[0]["payload"]
                decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(decoded)
                self.assertEqual(decoded.shape[:2], (height_blocks * 8, width_blocks * 8))

    def test_udp_video_reassembler_rejects_overlapping_fragment_offset(self) -> None:
        config = app_config_with_video_auth(video_payload_bytes=4)
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def run() -> None:
            await reassembler.handle_datagram(
                rtp_jpeg_packet(
                    timestamp=270_000,
                    offset=0,
                    sequence=1,
                    marker=False,
                    payload=b"abcd",
                ),
                ("127.0.0.1", 12345),
            )
            await reassembler.handle_datagram(
                rtp_jpeg_packet(
                    timestamp=270_000,
                    offset=2,
                    sequence=2,
                    marker=True,
                    payload=b"cdef",
                ),
                ("127.0.0.1", 12345),
            )

        asyncio.run(run())

        self.assertEqual([message["kind"] for message in manager.messages], [])
        self.assertEqual(reassembler.frames, {})
        self.assertEqual(manager.video_udp_seen_count, 0)

    def test_udp_video_reassembler_rejects_malformed_rtp_header(self) -> None:
        cases = [
            ("unsupported rtp version", {"version": 1}),
            ("unexpected rtp payload type", {"payload_type": 25}),
            ("unsupported rtp padding/csrc", {"first_byte_flags": 0x20}),
            ("unsupported rtp padding/csrc", {"first_byte_flags": 0x01}),
        ]
        for expected_error, overrides in cases:
            with self.subTest(expected_error=expected_error):
                config = app_config_with_video_auth()
                manager = FakeBroadcastManager()
                reassembler = UdpVideoReassembler(manager, config)
                raw = rtp_jpeg_packet(
                    timestamp=360_000,
                    offset=0,
                    sequence=1,
                    marker=True,
                    payload=b"abcd",
                    **overrides,
                )

                asyncio.run(reassembler.handle_datagram(raw, ("127.0.0.1", 12345)))

                self.assertEqual([message["kind"] for message in manager.messages], ["device_error"])
                self.assertIn(expected_error, manager.messages[0]["error"])
                self.assertEqual(manager.video_udp_seen_count, 0)

    def test_udp_video_reassembler_rejects_unauthenticated_rtp(self) -> None:
        cases = {
            "rtp authentication extension missing": {"authenticated": False},
            "invalid rtp authentication tag": {"auth_key_hex": "f" * 64},
        }
        for expected_error, overrides in cases.items():
            with self.subTest(expected_error=expected_error):
                config = app_config_with_video_auth()
                manager = FakeBroadcastManager()
                reassembler = UdpVideoReassembler(manager, config)
                raw = rtp_jpeg_packet(
                    timestamp=390_000,
                    offset=0,
                    sequence=1,
                    marker=True,
                    payload=b"abcd",
                    **overrides,
                )

                asyncio.run(reassembler.handle_datagram(raw, ("127.0.0.1", 12345)))

                self.assertEqual([message["kind"] for message in manager.messages], ["device_error"])
                self.assertIn(expected_error, manager.messages[0]["error"])
                self.assertEqual(manager.video_udp_seen_count, 0)

    def test_udp_video_reassembler_rejects_authenticated_truncated_jpeg_header(self) -> None:
        for payload in (b"", b"\x00" * 7):
            with self.subTest(payload_len=len(payload)):
                config = app_config_with_video_auth()
                manager = FakeBroadcastManager()
                reassembler = UdpVideoReassembler(manager, config)
                rtp_header = RTP_HEADER.pack(0x90, 0x80 | RTP_JPEG_PAYLOAD_TYPE, 1, 400_000, 1)
                raw = authenticated_rtp_packet(rtp_header, payload)

                asyncio.run(reassembler.handle_datagram(raw, ("127.0.0.1", 12345)))

                self.assertEqual([message["kind"] for message in manager.messages], ["device_error"])
                self.assertIn("rtp/jpeg header truncated", manager.messages[0]["error"])
                self.assertEqual(manager.video_udp_seen_count, 0)

    def test_udp_video_reassembler_rejects_malformed_rtp_jpeg_header(self) -> None:
        cases = {
            "unsupported rtp/jpeg type": {"jpeg_type": 2},
            "only rtp/jpeg dynamic quantization tables are supported": {"quality": 128},
            "invalid rtp/jpeg dimensions": {"width_blocks": 0},
        }
        for expected_error, overrides in cases.items():
            with self.subTest(expected_error=expected_error):
                config = app_config_with_video_auth()
                manager = FakeBroadcastManager()
                reassembler = UdpVideoReassembler(manager, config)
                raw = rtp_jpeg_packet(
                    timestamp=360_000,
                    offset=0,
                    sequence=1,
                    marker=True,
                    payload=b"abcd",
                    **overrides,
                )

                asyncio.run(reassembler.handle_datagram(raw, ("127.0.0.1", 12345)))

                self.assertEqual([message["kind"] for message in manager.messages], ["device_error"])
                self.assertIn(expected_error, manager.messages[0]["error"])
                self.assertEqual(manager.video_udp_seen_count, 0)

    def test_udp_video_reassembler_marks_seen_only_after_complete_frame(self) -> None:
        config = app_config_with_video_auth(video_payload_bytes=4)
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)
        raw = rtp_jpeg_packet(
            timestamp=360_000,
            offset=0,
            sequence=1,
            marker=False,
            payload=b"abcd",
        )

        asyncio.run(reassembler.handle_datagram(raw, ("127.0.0.1", 12345)))

        self.assertEqual(manager.video_udp_seen_count, 0)
        self.assertEqual(manager.messages, [])

    def test_udp_video_reassembler_drops_late_completed_older_frame(self) -> None:
        config = app_config_with_video_auth()
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def send_fragment(timestamp: int, sequence: int, payload: bytes) -> None:
            raw = rtp_jpeg_packet(
                timestamp=timestamp,
                offset=0,
                sequence=sequence,
                marker=True,
                payload=payload,
            )
            await reassembler.handle_datagram(raw, ("127.0.0.1", 12345))

        async def run() -> None:
            await send_fragment(90_000, 1, b"newer")
            await send_fragment(45_000, 2, b"older")

        asyncio.run(run())

        self.assertEqual(
            [message["seq"] for message in manager.messages if message["kind"] == "video"],
            [90_000],
        )

    def test_udp_video_reassembler_accepts_timestamp_wraparound(self) -> None:
        config = app_config_with_video_auth()
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def run() -> None:
            await reassembler.handle_datagram(
                rtp_jpeg_packet(
                    timestamp=0xFFFF_F000,
                    offset=0,
                    sequence=1,
                    marker=True,
                    payload=b"before-wrap",
                ),
                ("127.0.0.1", 12345),
            )
            await reassembler.handle_datagram(
                rtp_jpeg_packet(
                    timestamp=0x0000_1000,
                    offset=0,
                    sequence=2,
                    marker=True,
                    payload=b"after-wrap",
                ),
                ("127.0.0.1", 12345),
            )

        asyncio.run(run())

        video_payloads = [message["payload"] for message in manager.messages if message["kind"] == "video"]
        self.assertEqual(len(video_payloads), 2)
        self.assertIn(b"before-wrap", video_payloads[0])
        self.assertIn(b"after-wrap", video_payloads[1])

    def test_udp_video_reassembler_accepts_timestamp_reset_for_new_ssrc(self) -> None:
        config = app_config_with_video_auth()
        manager = FakeBroadcastManager()
        reassembler = UdpVideoReassembler(manager, config)

        async def send_fragment(ssrc: int, timestamp: int, sequence: int, payload: bytes) -> None:
            raw = rtp_jpeg_packet(
                ssrc=ssrc,
                timestamp=timestamp,
                offset=0,
                sequence=sequence,
                marker=True,
                payload=payload,
            )
            await reassembler.handle_datagram(raw, ("127.0.0.1", 12345))

        async def run() -> None:
            await send_fragment(100, 90_000, 1, b"before-reboot")
            await send_fragment(200, 1_000, 1, b"after-reboot")
            await send_fragment(100, 180_000, 2, b"late-old-session")

        asyncio.run(run())

        video_payloads = [message["payload"] for message in manager.messages if message["kind"] == "video"]
        self.assertEqual(len(video_payloads), 2)
        self.assertIn(b"before-reboot", video_payloads[0])
        self.assertIn(b"after-reboot", video_payloads[1])
        video_sessions = [
            message["video_session"] for message in manager.messages if message["kind"] == "video"
        ]
        self.assertEqual(video_sessions, [("udp", 100), ("udp", 200)])


if __name__ == "__main__":
    unittest.main()
