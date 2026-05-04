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
        self.closed: bool = False
        self.close_code: int | None = None

    async def send_text(self, text: str) -> None:
        self.messages.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.bytes_messages.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        self.close_code = code


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

    def test_replace_device_ws_closes_previous_channel_connection(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        old_ws = FakeWebSocket()
        new_ws = FakeWebSocket()
        manager.control_ws = old_ws

        asyncio.run(manager.replace_device_ws("control", new_ws))

        self.assertIs(manager.control_ws, new_ws)
        self.assertTrue(old_ws.closed)
        self.assertEqual(old_ws.close_code, 1012)

    def test_disconnect_device_closes_all_device_websockets(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        control_ws = FakeWebSocket()
        video_ws = FakeWebSocket()
        audio_ws = FakeWebSocket()
        manager.control_ws = control_ws
        manager.video_ws = video_ws
        manager.audio_ws = audio_ws

        result = asyncio.run(manager.disconnect_device())

        self.assertEqual(result["disconnected"], ["control", "video", "audio"])
        self.assertIsNone(manager.control_ws)
        self.assertIsNone(manager.video_ws)
        self.assertIsNone(manager.audio_ws)
        self.assertTrue(control_ws.closed)
        self.assertTrue(video_ws.closed)
        self.assertTrue(audio_ws.closed)

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

    def test_snapshot_includes_vision_frame_size(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )

        snapshot = manager.snapshot()

        self.assertEqual(snapshot["vision"], {"image_width": 16, "image_height": 12})


if __name__ == "__main__":
    unittest.main()
