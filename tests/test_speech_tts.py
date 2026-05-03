import asyncio
import io
from types import SimpleNamespace
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import wave

from aiglasses.config import SpeechConfig
from aiglasses.speech import DashscopeTtsSpeechSink, LocalTtsSpeechSink, SpeechEvent


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


class FakeLocalTtsSink(LocalTtsSpeechSink):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.synthesized: list[tuple[str, Path]] = []

    def _synthesize_wav(self, text: str, voice_path: Path) -> bytes:
        self.synthesized.append((text, voice_path))
        return _wav_bytes(b"\x01\x00\x02\x00", sample_rate=8000)


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

    def test_local_tts_segments_chinese_and_english(self) -> None:
        sink = FakeLocalTtsSink(
            SpeechConfig(
                enabled=True,
                provider="local",
                language="auto",
                piper_model_dir="voice",
                piper_voice_zh="zh_CN-huayan-medium",
                piper_voice_en="en_US-lessac-medium",
            ),
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )

        pcm = sink._synthesize_pcm16("开始 navigation")

        self.assertGreater(len(pcm), 0)
        self.assertEqual(
            sink.synthesized,
            [
                ("开始 ", Path("voice/zh_CN-huayan-medium.onnx")),
                ("navigation", Path("voice/en_US-lessac-medium.onnx")),
            ],
        )

    def test_local_tts_keeps_punctuation_with_surrounding_text(self) -> None:
        sink = FakeLocalTtsSink(
            SpeechConfig(enabled=True, provider="local", language="auto"),
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )

        pcm = sink._synthesize_pcm16("前方有车，停一下。")

        self.assertGreater(len(pcm), 0)
        self.assertEqual(
            sink.synthesized,
            [("前方有车，停一下。", Path("voice/zh_CN-huayan-medium.onnx"))],
        )

    def test_local_tts_ignores_punctuation_only_segments(self) -> None:
        sink = LocalTtsSpeechSink(
            SpeechConfig(enabled=True, provider="local", language="auto"),
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )

        self.assertEqual(sink._segments("。，."), [])

    def test_local_tts_accepts_explicit_piper_voice_path(self) -> None:
        sink = LocalTtsSpeechSink(
            SpeechConfig(enabled=True, provider="local", piper_voice_zh="/tmp/zh.onnx"),
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )

        self.assertEqual(sink._voice_path_for_language("zh"), Path("/tmp/zh.onnx"))

    def test_local_tts_writes_valid_empty_wav(self) -> None:
        class EmptyVoice:
            config = SimpleNamespace(sample_rate=22050)

            def synthesize_wav(self, *args, **kwargs) -> None:
                return None

        sink = LocalTtsSpeechSink(
            SpeechConfig(enabled=True, provider="local"),
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )
        sink._load_voice = lambda _: EmptyVoice()

        pcm = sink._wave_to_pcm16(sink._synthesize_wav(".", Path("voice/en_US-lessac-medium.onnx")))

        self.assertEqual(pcm, b"")

    def test_local_tts_resamples_wav_to_device_rate(self) -> None:
        sink = LocalTtsSpeechSink(
            SpeechConfig(enabled=True, provider="local"),
            send_pcm16=lambda _: asyncio.sleep(0, result=True),
            sample_rate=16000,
        )

        pcm = sink._wave_to_pcm16(_wav_bytes(b"\x01\x00\x02\x00\x03\x00\x04\x00", sample_rate=8000))

        self.assertGreater(len(pcm), 8)


def _wav_bytes(pcm16: bytes, *, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
