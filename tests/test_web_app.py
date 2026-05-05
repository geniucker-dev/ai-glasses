import asyncio
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from aiglasses.config import AppConfig, SpeechConfig
from aiglasses.config.settings import AsrConfig
from aiglasses.config.settings import DeviceAudioDownConfig, DeviceConfig
from aiglasses.protocol import Packet, PacketType
from aiglasses.web.app import CONTROL_MAX_PAYLOAD_BYTES, _unpack_device_packet, validate_speech_config


class FakeBroadcastManager:
    def __init__(self) -> None:
        self.messages: list[dict] = []

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
