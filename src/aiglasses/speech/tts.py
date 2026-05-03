from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from aiglasses.config import SpeechConfig

from .events import SpeechEvent


logger = logging.getLogger("aiglasses.speech")


class DashscopeTtsSpeechSink:
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
        self.config = config
        self.api_key = api_key
        self.send_pcm16 = send_pcm16
        self.sample_rate = sample_rate
        self.broadcast = broadcast
        self.websocket_base_url = websocket_base_url
        self.http_base_url = http_base_url
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
