from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Awaitable, Callable, Protocol


@dataclass(frozen=True)
class SpeechEvent:
    text: str
    priority: int = 50
    source: str = "navigation"
    created_at: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kind"] = "speech"
        return data


class SpeechSink(Protocol):
    async def emit(self, event: SpeechEvent) -> None:
        ...


class UiSpeechSink:
    def __init__(self, broadcast: Callable[[dict], Awaitable[None]]) -> None:
        self._broadcast = broadcast

    async def emit(self, event: SpeechEvent) -> None:
        await self._broadcast(event.to_dict())


class SpeechHub:
    def __init__(self) -> None:
        self._sinks: list[SpeechSink] = []

    def add_sink(self, sink: SpeechSink) -> None:
        self._sinks.append(sink)

    async def say(self, text: str, *, priority: int = 50, source: str = "navigation") -> None:
        event = SpeechEvent(text=text, priority=priority, source=source, created_at=time.time())
        for sink in list(self._sinks):
            await sink.emit(event)
