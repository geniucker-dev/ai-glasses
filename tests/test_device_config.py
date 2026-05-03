import asyncio
import json
import unittest

from aiglasses.device import DeviceManager
from aiglasses.navigation import NavigationStateMachine
from aiglasses.speech import SpeechHub
from starlette.websockets import WebSocketState


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, text: str) -> None:
        self.messages.append(text)


class FailingWebSocket:
    async def send_text(self, text: str) -> None:
        raise RuntimeError("closed")


class FakeUiWebSocket(FakeWebSocket):
    client_state = WebSocketState.CONNECTED


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


if __name__ == "__main__":
    unittest.main()
