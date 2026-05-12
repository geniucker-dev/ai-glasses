import asyncio
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from aiglasses.config import AppConfig, SpeechConfig
from aiglasses.config.settings import AsrConfig, DeviceCaptureConfig
from aiglasses.config.settings import DeviceAudioDownConfig, DeviceConfig
from aiglasses.protocol import Packet, PacketType
from aiglasses.web.app import (
    CONTROL_MAX_PAYLOAD_BYTES,
    _receive_current_device_packet,
    _unpack_device_packet,
    validate_speech_config,
)


class FakeBroadcastManager:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast(self, message: dict) -> None:
        self.messages.append(message)


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


if __name__ == "__main__":
    unittest.main()
