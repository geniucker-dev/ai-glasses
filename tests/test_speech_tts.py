import asyncio
import time
import unittest
from unittest.mock import patch

from aiglasses.config import SpeechConfig
from aiglasses.speech import DashscopeTtsSpeechSink, SpeechEvent


class QueuedTtsSink(DashscopeTtsSpeechSink):
    def _synthesize_pcm16(self, text: str) -> bytes:
        if text == "first":
            time.sleep(0.03)
        return text.encode("utf-8")


class FakeTtsResult:
    def get_audio_data(self) -> bytes:
        return b"pcm"

    def get_response(self) -> None:
        return None


class SpeechTtsTests(unittest.IsolatedAsyncioTestCase):
    async def test_device_tts_serializes_speech_events(self) -> None:
        sent: list[bytes] = []

        async def send_pcm16(pcm16: bytes) -> bool:
            sent.append(pcm16)
            return True

        sink = QueuedTtsSink(
            SpeechConfig(enabled=True),
            api_key="test-key",
            send_pcm16=send_pcm16,
            sample_rate=16000,
        )

        await sink.emit(SpeechEvent("first"))
        await sink.emit(SpeechEvent("second"))
        await asyncio.wait_for(sink._queue.join(), timeout=1)

        self.assertEqual(sent, [b"first", b"second"])

    def test_device_tts_uses_firmware_sample_rate(self) -> None:
        sink = DashscopeTtsSpeechSink(
            SpeechConfig(enabled=True, sample_rate=24000),
            api_key="test-key",
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )

        with patch(
            "dashscope.audio.tts.SpeechSynthesizer.call",
            return_value=FakeTtsResult(),
        ) as call:
            self.assertEqual(sink._synthesize_pcm16("hello"), b"pcm")

        self.assertEqual(call.call_args.kwargs["sample_rate"], 16000)


if __name__ == "__main__":
    unittest.main()
