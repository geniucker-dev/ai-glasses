from __future__ import annotations

import asyncio
import audioop
import io
import logging
from pathlib import Path
from typing import Awaitable, Callable
import wave

from aiglasses.config import SpeechConfig

from .events import SpeechEvent


logger = logging.getLogger("aiglasses.speech")


class QueuedPcm16SpeechSink:
    def __init__(
        self,
        config: SpeechConfig,
        *,
        send_pcm16: Callable[[bytes], Awaitable[bool]],
        sample_rate: int,
        broadcast: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.send_pcm16 = send_pcm16
        self.sample_rate = sample_rate
        self.broadcast = broadcast
        self._queue: asyncio.Queue[SpeechEvent] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def emit(self, event: SpeechEvent) -> None:
        if not self.config.enabled or not event.text.strip():
            return
        await self._queue.put(event)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def _run_worker(self) -> None:
        while True:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await self._speak(event)
            finally:
                self._queue.task_done()

    async def _speak(self, event: SpeechEvent) -> None:
        try:
            pcm16 = await asyncio.to_thread(self._synthesize_pcm16, event.text)
            sent = await self.send_pcm16(pcm16)
            if self.broadcast is not None:
                await self.broadcast(
                    {
                        "kind": "speech_audio",
                        "source": event.source,
                        "bytes": len(pcm16),
                        "sent": sent,
                    }
                )
        except Exception as exc:
            logger.exception("speech synthesis failed")
            if self.broadcast is not None:
                await self.broadcast({"kind": "speech_error", "source": event.source, "error": str(exc)})

    def _synthesize_pcm16(self, text: str) -> bytes:
        raise NotImplementedError


class DashscopeTtsSpeechSink(QueuedPcm16SpeechSink):
    def __init__(
        self,
        config: SpeechConfig,
        *,
        api_key: str,
        send_pcm16: Callable[[bytes], Awaitable[bool]],
        sample_rate: int,
        broadcast: Callable[[dict], Awaitable[None]] | None = None,
        websocket_base_url: str = "",
        http_base_url: str = "",
    ) -> None:
        super().__init__(
            config,
            send_pcm16=send_pcm16,
            sample_rate=sample_rate,
            broadcast=broadcast,
        )
        self.api_key = api_key
        self.websocket_base_url = websocket_base_url
        self.http_base_url = http_base_url

    def _synthesize_pcm16(self, text: str) -> bytes:
        if self.config.provider != "dashscope":
            raise RuntimeError(f"unsupported speech provider: {self.config.provider}")
        if not self.api_key or self.api_key.startswith("replace-"):
            raise RuntimeError("missing DashScope API key for speech synthesis")
        if self.config.audio_format != "pcm":
            raise RuntimeError(f"unsupported speech audio format: {self.config.audio_format}")

        import dashscope
        from dashscope.audio.tts import SpeechSynthesizer

        dashscope.api_key = self.api_key
        if self.http_base_url:
            dashscope.base_http_api_url = self.http_base_url
        if self.websocket_base_url:
            dashscope.base_websocket_api_url = self.websocket_base_url

        result = SpeechSynthesizer.call(
            model=self.config.model,
            text=text,
            format=self.config.audio_format,
            sample_rate=self.sample_rate,
            volume=self.config.volume,
            rate=self.config.rate,
            pitch=self.config.pitch,
        )
        data = result.get_audio_data() or b""
        if not data:
            response = result.get_response()
            message = getattr(response, "message", None) or "empty TTS response"
            raise RuntimeError(message)
        return data


class LocalTtsSpeechSink(QueuedPcm16SpeechSink):
    def __init__(
        self,
        config: SpeechConfig,
        *,
        send_pcm16: Callable[[bytes], Awaitable[bool]],
        sample_rate: int,
        broadcast: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(
            config,
            send_pcm16=send_pcm16,
            sample_rate=sample_rate,
            broadcast=broadcast,
        )
        self._voices: dict[Path, object] = {}

    def _synthesize_pcm16(self, text: str) -> bytes:
        if self.config.provider != "local":
            raise RuntimeError(f"unsupported speech provider: {self.config.provider}")
        return b"".join(
            self._wave_to_pcm16(self._synthesize_wav(segment, self._voice_path_for_language(language)))
            for language, segment in self._segments(text)
        )

    def _segments(self, text: str) -> list[tuple[str, str]]:
        language = self.config.language.lower()
        if language.startswith(("zh", "cmn")):
            return [("zh", text)]
        if language.startswith("en"):
            return [("en", text)]

        segments: list[tuple[str, str]] = []
        current_language = ""
        current_text: list[str] = []
        for char in text:
            char_language = _char_language(char)
            if not char_language:
                current_text.append(char)
                continue
            if current_language and char_language != current_language:
                segments.append((current_language, "".join(current_text)))
                current_text = [char]
            else:
                current_text.append(char)
            current_language = char_language
        if current_text:
            segments.append((current_language or "en", "".join(current_text)))
        return [
            (language, segment)
            for language, segment in segments
            if segment.strip() and _has_speech_chars(segment)
        ]

    def _voice_path_for_language(self, language: str) -> Path:
        if language == "zh":
            return self._voice_path(self.config.piper_voice_zh)
        return self._voice_path(self.config.piper_voice_en)

    def _voice_path(self, voice: str) -> Path:
        path = Path(voice)
        if not path.is_absolute() and len(path.parts) == 1:
            path = Path(self.config.piper_model_dir) / path
        if path.suffix != ".onnx":
            path = path.with_suffix(".onnx")
        return path

    def _synthesize_wav(self, text: str, voice_path: Path) -> bytes:
        voice = self._load_voice(voice_path)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(voice.config.sample_rate)
            voice.synthesize_wav(
                text,
                wav,
                syn_config=self._synthesis_config(),
                set_wav_format=False,
            )
        return buffer.getvalue()

    def _load_voice(self, voice_path: Path) -> object:
        voice_path = voice_path.expanduser().resolve()
        cached = self._voices.get(voice_path)
        if cached is not None:
            return cached

        config_path = Path(f"{voice_path}.json")
        if not voice_path.exists() or not config_path.exists():
            raise RuntimeError(
                f"Piper voice files not found: {voice_path} and {config_path}. "
                "Download them with `python -m piper.download_voices --download-dir "
                f"{self.config.piper_model_dir} {voice_path.stem}`."
            )

        from piper import PiperVoice

        voice = PiperVoice.load(
            str(voice_path),
            config_path=str(config_path),
            use_cuda=self.config.piper_use_cuda,
        )
        self._voices[voice_path] = voice
        return voice

    def _synthesis_config(self) -> object:
        try:
            from piper import SynthesisConfig
        except ImportError:
            from piper.voices import SynthesisConfig

        rate = max(0.25, float(self.config.rate))
        return SynthesisConfig(
            volume=max(0.0, float(self.config.volume) / 50.0),
            length_scale=1.0 / rate,
        )

    def _wave_to_pcm16(self, wav_data: bytes) -> bytes:
        with wave.open(io.BytesIO(wav_data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            source_rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())

        if channels > 1:
            pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
            channels = 1
        if sample_width != 2:
            pcm = audioop.lin2lin(pcm, sample_width, 2)
            sample_width = 2
        if source_rate != self.sample_rate:
            pcm, _ = audioop.ratecv(pcm, sample_width, channels, source_rate, self.sample_rate, None)
        return pcm


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _char_language(char: str) -> str:
    if _is_cjk(char):
        return "zh"
    if char.isascii() and char.isalpha():
        return "en"
    return ""


def _has_speech_chars(text: str) -> bool:
    return any(_is_cjk(char) or (char.isascii() and char.isalnum()) for char in text)
