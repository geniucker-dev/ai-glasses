import asyncio
from dataclasses import dataclass
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

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


class FailingBytesWebSocket(FakeWebSocket):
    async def send_bytes(self, data: bytes) -> None:
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


class BlockingVision(FakeVision):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def analyze_jpeg(self, payload: bytes) -> FrameAnalysis:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocked vision analysis")
        return FrameAnalysis(model_status={"fake": payload.decode("ascii", errors="ignore")})


class DeviceConfigTests(unittest.TestCase):
    def _expected_device_config(self, target_fps: int) -> dict[str, object]:
        return {
            "kind": "config",
            "target_fps": target_fps,
            "jpeg_quality": 12,
            "camera_profile": "traffic_signal",
            "ae_level": -1,
            "saturation": 1,
            "contrast": 1,
            "sharpness": 1,
            "gainceiling": 4,
        }

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

        self.assertEqual(result, {"config": self._expected_device_config(12), "sent": True})
        self.assertEqual(json.loads(ws.messages[-1]), self._expected_device_config(12))

    def test_update_device_config_clamps_minimum(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )

        result = asyncio.run(manager.update_device_config(target_fps=0))

        self.assertEqual(result, {"config": self._expected_device_config(1), "sent": False})

    def test_update_device_config_clamps_maximum(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
            target_video_fps=6,
        )

        result = asyncio.run(manager.update_device_config(target_fps=5000))

        self.assertEqual(result, {"config": self._expected_device_config(1000), "sent": False})

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
            {
                "kind": "device",
                "channel": "control",
                "connected": False,
                "generation": 1,
            },
        )

    def test_failed_speech_send_broadcasts_generated_control_disconnect(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ws = FailingBytesWebSocket()
        ui_ws = FakeUiWebSocket()
        manager.control_ws = ws
        manager.device_ws_generations["control"] = 3
        manager.ui_clients.add(ui_ws)

        sent = asyncio.run(manager.send_speech_pcm16(b"abcdef", chunk_bytes=4))

        self.assertFalse(sent)
        self.assertIsNone(manager.control_ws)
        self.assertEqual(
            json.loads(ui_ws.messages[-1]),
            {
                "kind": "device",
                "channel": "control",
                "connected": False,
                "generation": 4,
            },
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
        self.assertEqual(json.loads(ws.messages[-1]), self._expected_device_config(6))
        self.assertEqual(
            json.loads(ui_ws.messages[-1]),
            {"kind": "device_config", "config": self._expected_device_config(6), "sent": True},
        )

    def test_replace_device_ws_schedules_previous_close_without_waiting(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        old_ws = FakeWebSocket()
        new_ws = FakeWebSocket()
        manager.control_ws = old_ws
        close_started = asyncio.Event()
        allow_close = asyncio.Event()
        close_task: asyncio.Task[None] | None = None

        async def close_previous(ws) -> None:
            close_started.set()
            await allow_close.wait()
            await ws.close(code=1012)

        async def run_scenario() -> None:
            nonlocal close_task
            with patch.object(manager, "_close_ws", side_effect=close_previous):
                await manager.replace_device_ws("control", new_ws)
                await close_started.wait()
                close_tasks = [
                    task for task in asyncio.all_tasks() if task is not asyncio.current_task()
                ]
                self.assertEqual(len(close_tasks), 1)
                close_task = close_tasks[0]
                self.assertFalse(close_task.done())
                self.assertIs(manager.control_ws, new_ws)
                allow_close.set()
                await close_task

        asyncio.run(run_scenario())

        self.assertIsNotNone(close_task)
        self.assertIs(manager.control_ws, new_ws)
        self.assertTrue(old_ws.closed)
        self.assertEqual(old_ws.close_code, 1012)

    def test_disconnect_waits_for_superseded_close_task_to_finish(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        old_ws = FakeWebSocket()
        new_ws = FakeWebSocket()
        manager.control_ws = old_ws
        close_started = asyncio.Event()
        close_old = asyncio.Event()
        close_new = asyncio.Event()

        async def close_ws(ws) -> None:
            if ws is old_ws:
                close_started.set()
                await close_old.wait()
            else:
                await close_new.wait()
            await ws.close(code=1012)

        async def run_scenario() -> dict[str, object]:
            with patch.object(manager, "_close_ws", side_effect=close_ws):
                await manager.replace_device_ws("control", new_ws)
                await close_started.wait()
                disconnect_task = asyncio.create_task(manager.disconnect_device())
                await asyncio.sleep(0)
                self.assertFalse(disconnect_task.done())
                close_new.set()
                await asyncio.sleep(0)
                self.assertFalse(disconnect_task.done())
                close_old.set()
                return await disconnect_task

        result = asyncio.run(run_scenario())

        self.assertEqual(result["disconnected"], ["control"])
        self.assertIsNone(manager.control_ws)
        self.assertTrue(old_ws.closed)
        self.assertTrue(new_ws.closed)
        self.assertEqual(new_ws.close_code, 1012)
        self.assertEqual(result["state"]["device"]["generation"]["control"], 2)

    def test_device_connection_events_include_increasing_generations(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        first_ws = FakeWebSocket()
        second_ws = FakeWebSocket()

        first_generation = asyncio.run(manager.replace_device_ws("control", first_ws))
        cleared_generation = asyncio.run(manager.clear_device_ws("control", first_ws))
        second_generation = asyncio.run(manager.replace_device_ws("control", second_ws))

        first_event = manager.device_connection_event(
            channel="control",
            connected=True,
            generation=first_generation,
        )
        cleared_event = manager.device_connection_event(
            channel="control",
            connected=False,
            generation=cleared_generation,
        )
        second_event = manager.device_connection_event(
            channel="control",
            connected=True,
            generation=second_generation,
        )

        self.assertEqual(first_event["generation"], 1)
        self.assertEqual(cleared_event["generation"], 2)
        self.assertEqual(second_event["generation"], 3)
        self.assertTrue(first_event["connected"])
        self.assertFalse(cleared_event["connected"])
        self.assertTrue(second_event["connected"])

    def test_disconnect_does_not_broadcast_stale_disconnect_after_reconnect(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        old_ws = FakeWebSocket()
        reconnected_ws = FakeWebSocket()
        ui_ws = FakeUiWebSocket()
        manager.control_ws = old_ws
        manager.ui_clients.add(ui_ws)
        close_started = asyncio.Event()
        allow_close = asyncio.Event()

        async def close_current(ws) -> None:
            close_started.set()
            await allow_close.wait()
            await ws.close(code=1012)

        async def run_scenario() -> dict[str, object]:
            with patch.object(manager, "_close_ws", side_effect=close_current):
                disconnect_task = asyncio.create_task(manager.disconnect_device())
                await close_started.wait()
                await manager.replace_device_ws("control", reconnected_ws)
                allow_close.set()
                return await disconnect_task

        result = asyncio.run(run_scenario())

        self.assertEqual(result["disconnected"], ["control"])
        self.assertIs(manager.control_ws, reconnected_ws)
        self.assertTrue(old_ws.closed)
        self.assertFalse(reconnected_ws.closed)
        self.assertIn(
            {"kind": "device", "channel": "control", "connected": False},
            [
                {key: value for key, value in json.loads(message).items() if key != "generation"}
                for message in ui_ws.messages
            ],
        )

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
        self.assertIn("processed_fps_3s", message["video_stats"])

    def test_video_packet_processing_is_serialized(self) -> None:
        vision = BlockingVision()
        manager = DeviceManager(
            vision=vision,
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)

        async def run() -> None:
            first = asyncio.create_task(
                manager.handle_video_packet(Packet(PacketType.VIDEO_JPEG, 10, 1234, b"first"))
            )
            analysis_started = await asyncio.to_thread(vision.started.wait, 2)
            self.assertTrue(analysis_started)

            second = asyncio.create_task(
                manager.handle_video_packet(Packet(PacketType.VIDEO_JPEG, 11, 1235, b"second"))
            )
            await asyncio.sleep(0.05)

            self.assertEqual(manager.frame_count, 1)
            self.assertEqual(len(manager.frame_received_at), 2)
            self.assertEqual(manager.frame_processed_at, [])
            self.assertEqual(ui_ws.bytes_messages, [])
            self.assertFalse(second.done())

            vision.release.set()
            await asyncio.gather(first, second)

        asyncio.run(run())

        frame_packets = [Packet.unpack(message) for message in ui_ws.bytes_messages]
        self.assertEqual([packet.seq for packet in frame_packets], [1, 2])
        self.assertEqual([packet.payload for packet in frame_packets], [b"first", b"second"])
        frame_events = [
            json.loads(message)["frame_count"]
            for message in ui_ws.messages
            if json.loads(message).get("kind") == "frame"
        ]
        self.assertEqual(frame_events, [1, 2])

    def test_video_packet_drops_stale_frame_after_newer_processed(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)

        async def run() -> None:
            await manager.handle_video_packet(Packet(PacketType.VIDEO_JPEG, 20, 1234, b"newer"))
            await manager.handle_video_packet(Packet(PacketType.VIDEO_JPEG, 10, 1235, b"older"))

        asyncio.run(run())

        frame_packets = [Packet.unpack(message) for message in ui_ws.bytes_messages]
        self.assertEqual([packet.seq for packet in frame_packets], [1])
        self.assertEqual([packet.payload for packet in frame_packets], [b"newer"])
        self.assertEqual(len(manager.frame_received_at), 2)
        self.assertEqual(len(manager.frame_processed_at), 1)
        self.assertEqual(manager.frame_count, 1)
        self.assertEqual(manager.last_processed_video_seq, 20)

    def test_video_packet_allows_sequence_reset_for_new_session(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)

        async def run() -> None:
            await manager.handle_video_packet(
                Packet(PacketType.VIDEO_JPEG, 20, 1234, b"before-reconnect"),
                video_session=("ws", 1),
            )
            await manager.handle_video_packet(
                Packet(PacketType.VIDEO_JPEG, 1, 1235, b"after-reconnect"),
                video_session=("ws", 2),
            )

        asyncio.run(run())

        frame_packets = [Packet.unpack(message) for message in ui_ws.bytes_messages]
        self.assertEqual([packet.seq for packet in frame_packets], [1, 2])
        self.assertEqual(
            [packet.payload for packet in frame_packets],
            [b"before-reconnect", b"after-reconnect"],
        )
        self.assertEqual(manager.last_processed_video_session, ("ws", 2))
        self.assertEqual(manager.last_processed_video_seq, 1)

    def test_recording_writes_raw_camera_jpeg_and_metadata(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)
        camera_payload = b"\xff\xd8raw-camera-payload\xff\xd9"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("aiglasses.device.manager.RECORDINGS_DIR", Path(tmpdir)):
                start_status = asyncio.run(manager.start_recording())
                asyncio.run(
                    manager.handle_video_packet(
                        Packet(PacketType.VIDEO_JPEG, 10, 1234, camera_payload)
                    )
                )
                stop_status = asyncio.run(manager.stop_recording())

                recording_dir = Path(start_status["recording_dir"])
                frame_bytes = (recording_dir / "frames" / "00000001.jpg").read_bytes()
                metadata_text = (recording_dir / "metadata.jsonl").read_text(encoding="utf-8")
                session = json.loads((recording_dir / "session.json").read_text(encoding="utf-8"))

        self.assertFalse(stop_status["active"])
        self.assertEqual(frame_bytes, camera_payload)
        metadata = json.loads(metadata_text.strip())
        self.assertEqual(metadata["recording_frame_index"], 1)
        self.assertEqual(metadata["global_frame_count"], 1)
        self.assertEqual(metadata["frame_file"], "frames/00000001.jpg")
        self.assertEqual(metadata["jpeg_bytes"], len(camera_payload))
        self.assertEqual(metadata["analysis"]["model_status"], {"fake": "ready"})
        self.assertIn("navigation", metadata)
        self.assertEqual(session["frame_count"], 1)
        self.assertIsNotNone(session["stopped_at"])
        recording_events = [json.loads(message) for message in ui_ws.messages if "recording" in message]
        self.assertEqual(recording_events[0]["kind"], "recording")
        self.assertTrue(recording_events[0]["recording"]["active"])
        self.assertFalse(recording_events[-1]["recording"]["active"])

    def test_recording_start_stop_are_idempotent(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("aiglasses.device.manager.RECORDINGS_DIR", Path(tmpdir)):
                first = asyncio.run(manager.start_recording())
                second = asyncio.run(manager.start_recording())
                stopped = asyncio.run(manager.stop_recording())
                stopped_again = asyncio.run(manager.stop_recording())

        self.assertEqual(first["session_id"], second["session_id"])
        self.assertTrue(first["active"])
        self.assertTrue(second["active"])
        self.assertFalse(stopped["active"])
        self.assertFalse(stopped_again["active"])

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

    def test_video_stats_counts_single_recent_frame_in_window(self) -> None:
        fps = DeviceManager._fps_from_timestamps([9.0], now=10.0, window_s=3.0)

        self.assertAlmostEqual(fps, 1 / 3)

    def test_snapshot_includes_vision_frame_size(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )

        snapshot = manager.snapshot()

        self.assertEqual(snapshot["vision"], {"image_width": 16, "image_height": 12})

    def test_malformed_control_json_broadcasts_error_and_keeps_running(self) -> None:
        manager = DeviceManager(
            vision=FakeVision(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)

        asyncio.run(
            manager.handle_control_packet(Packet(PacketType.CONTROL_JSON, 1, 1234, b'{"bad"'))
        )
        asyncio.run(
            manager.handle_control_packet(
                Packet(PacketType.CONTROL_JSON, 2, 1235, b'{"kind":"hello"}')
            )
        )

        error = json.loads(ui_ws.messages[-2])
        self.assertEqual(error["kind"], "device_error")
        self.assertEqual(error["channel"], "control")
        self.assertEqual(error["packet_type"], "CONTROL_JSON")
        self.assertIn("malformed device JSON", error["error"])
        device = json.loads(ui_ws.messages[-1])
        self.assertEqual(device["kind"], "device")
        self.assertEqual(device["data"], {"kind": "hello"})

    def test_invalid_utf8_imu_json_broadcasts_error(self) -> None:
        manager = DeviceManager(
            vision=object(),
            navigation=NavigationStateMachine(),
            speech=SpeechHub(),
        )
        ui_ws = FakeUiWebSocket()
        manager.ui_clients.add(ui_ws)

        asyncio.run(manager.handle_control_packet(Packet(PacketType.IMU_JSON, 1, 1234, b"\xff")))

        error = json.loads(ui_ws.messages[-1])
        self.assertEqual(error["kind"], "device_error")
        self.assertEqual(error["packet_type"], "IMU_JSON")
        self.assertIsNone(manager.last_imu)


if __name__ == "__main__":
    unittest.main()
