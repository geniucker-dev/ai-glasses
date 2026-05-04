import asyncio
from dataclasses import dataclass
import json
import unittest

from aiglasses.device import DeviceManager
from aiglasses.navigation import NavigationStateMachine
from aiglasses.protocol import Packet, PacketType
from aiglasses.speech import SpeechHub
from aiglasses.vision.types import FrameAnalysis
from starlette.websockets import WebSocketState


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.bytes_messages: list[bytes] = []

    async def send_text(self, text: str) -> None:
        self.messages.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.bytes_messages.append(data)


class FailingWebSocket:
    async def send_text(self, text: str) -> None:
        raise RuntimeError("closed")


class FakeUiWebSocket(FakeWebSocket):
    client_state = WebSocketState.CONNECTED


@dataclass(frozen=True)
class FakeModelsConfig:
    image_width: int = 16
    image_height: int = 12


@dataclass(frozen=True)
class FakeConfig:
    models: FakeModelsConfig = FakeModelsConfig()


class FakeVision:
    config = FakeConfig()
    model_status: dict[str, str] = {}

    def analyze_jpeg(self, payload: bytes) -> FrameAnalysis:
        return FrameAnalysis(model_status={"fake": "ready"})


class DeviceConfigTests(unittest.TestCase):
    def test_update_device_config_pushes_target_fps(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )
        ws = FakeWebSocket()
        manager.control_ws = ws

        result = asyncio.run(manager.update_device_config(target_fps=12))

        self.assertEqual(result, {"config": {"kind": "config", "target_fps": 12}, "sent": True})
        self.assertEqual(json.loads(ws.messages[-1]), {"kind": "config", "target_fps": 12})

    def test_update_device_config_clamps_minimum(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )

        result = asyncio.run(manager.update_device_config(target_fps=0))

        self.assertEqual(result, {"config": {"kind": "config", "target_fps": 1}, "sent": False})

    def test_update_device_config_clamps_maximum(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )

        result = asyncio.run(manager.update_device_config(target_fps=5000))

        self.assertEqual(result, {"config": {"kind": "config", "target_fps": 1000}, "sent": False})

    def test_failed_config_push_broadcasts_control_disconnect(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )
        ws = FailingWebSocket()
        ui_ws = FakeUiWebSocket()
        manager.control_ws = ws
        manager.ui_clients.add(ui_ws)

        sent = asyncio.run(manager.send_device_config())

        self.assertFalse(sent)
        self.assertIsNone(manager.control_ws)
        self.assertEqual(
            json.loads(ui_ws.messages[-1]),
            {"kind": "device", "channel": "control", "connected": False},
        )

    def test_sync_device_config_broadcasts_successful_delivery(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )
        ws = FakeWebSocket()
        ui_ws = FakeUiWebSocket()
        manager.control_ws = ws
        manager.ui_clients.add(ui_ws)

        sent = asyncio.run(manager.sync_device_config())

        self.assertTrue(sent)
        self.assertEqual(json.loads(ws.messages[-1]), {"kind": "config", "target_fps": 6})
        self.assertEqual(
            json.loads(ui_ws.messages[-1]),
            {"kind": "device_config", "config": {"kind": "config", "target_fps": 6}, "sent": True},
        )

    def test_send_speech_pcm16_packets_control_websocket(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ws = FakeWebSocket()
        manager.control_ws = ws

        sent = asyncio.run(manager.send_speech_pcm16(b"abcdef", chunk_bytes=4))

        self.assertTrue(sent)
        self.assertEqual(len(ws.bytes_messages), 2)
        first = Packet.unpack(ws.bytes_messages[0])
        second = Packet.unpack(ws.bytes_messages[1])
        self.assertEqual(first.packet_type, PacketType.SPEECH_PCM16)
        self.assertEqual(first.seq, 0)
        self.assertEqual(first.payload, b"abcd")
        self.assertEqual(second.packet_type, PacketType.SPEECH_PCM16)
        self.assertEqual(second.seq, 1)
        self.assertEqual(second.payload, b"ef")

    def test_video_packet_broadcasts_processed_frame_bytes_to_ui(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)

        asyncio.run(
            manager.handle_video_packet(
                Packet(PacketType.VIDEO_JPEG, 10, 1234, b"\xff\xd8jpeg\xff\xd9")
            )
        )

        frame_packet = Packet.unpack(ui_ws.bytes_messages[-1])
        self.assertEqual(frame_packet.packet_type, PacketType.VIDEO_JPEG)
        self.assertEqual(frame_packet.seq, 1)
        self.assertEqual(frame_packet.payload, b"\xff\xd8jpeg\xff\xd9")
        message = json.loads(ui_ws.messages[-1])
        self.assertEqual(message["kind"], "frame")
        self.assertEqual(message["frame_count"], 1)
        self.assertIn("received_fps_3s", message["video_stats"])

    def test_processing_benchmark_uses_analyze_jpeg_path(self) -> None:
        vision = FakeVision()
        manager = DeviceManager(
            vision=vision,
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )

        result = manager.benchmark_processing_capacity(warmup_runs=2, measured_runs=3)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["warmup_runs"], 2)
        self.assertEqual(result["measured_runs"], 3)
        self.assertEqual(result["image_width"], 16)
        self.assertEqual(result["image_height"], 12)
        self.assertGreater(result["fps_p50"], 0)
        self.assertEqual(manager.backend_benchmark, result)


if __name__ == "__main__":
    unittest.main()
