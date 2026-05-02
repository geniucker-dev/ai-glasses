from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import contextlib

from aiglasses.config import AsrConfig


CommandCallback = Callable[[str], Awaitable[None]]


@dataclass
class AsrService:
    config: AsrConfig
    on_final_text: CommandCallback

    def __post_init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._task: asyncio.Task | None = None
        self._recognition = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.status = "disabled" if not self.config.enabled else "configured"

    async def start(self) -> None:
        if not self.config.enabled or self._task:
            return
        if self.config.provider != "dashscope":
            self.status = f"unsupported_provider:{self.config.provider}"
            return
        if not self.config.dashscope_api_key or self.config.dashscope_api_key.startswith("replace-"):
            self.status = "missing_dashscope_api_key"
            return
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def push_pcm16(self, chunk: bytes) -> None:
        if not self.config.enabled:
            return
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            _ = self._queue.get_nowait()
            self._queue.put_nowait(chunk)

    async def inject_text(self, text: str) -> None:
        await self.on_final_text(text)

    async def _run(self) -> None:
        try:
            import dashscope
            from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
        except Exception as exc:
            self.status = f"dashscope_import_failed:{exc}"
            return

        dashscope.api_key = self.config.dashscope_api_key
        if self.config.http_base_url:
            dashscope.base_http_api_url = self.config.http_base_url
        if self.config.websocket_base_url:
            dashscope.base_websocket_api_url = self.config.websocket_base_url
        service = self

        class Callback(RecognitionCallback):
            def on_open(self) -> None:
                service.status = "stream_open"

            def on_complete(self) -> None:
                service.status = "complete"

            def on_error(self, result: RecognitionResult) -> None:
                service.status = f"error:{getattr(result, 'message', result)}"

            def on_close(self) -> None:
                if service.status != "complete":
                    service.status = "closed"

            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                sentences = sentence if isinstance(sentence, list) else [sentence]
                for item in sentences:
                    if not item:
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text or not RecognitionResult.is_sentence_end(item):
                        continue
                    if service._loop:
                        asyncio.run_coroutine_threadsafe(service.on_final_text(text), service._loop)

        while True:
            self._recognition = Recognition(
                model=self.config.model,
                callback=Callback(),
                format="pcm",
                sample_rate=self.config.sample_rate,
            )
            try:
                await asyncio.to_thread(self._recognition.start)
                self.status = "running"
                while True:
                    chunk = await self._queue.get()
                    await asyncio.to_thread(self._recognition.send_audio_frame, chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status = f"restarting:{exc}"
                await asyncio.sleep(1.0)
            finally:
                if self._recognition is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(self._recognition.stop)
                self._recognition = None
