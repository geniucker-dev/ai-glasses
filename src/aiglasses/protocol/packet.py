from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
import zlib


class ProtocolError(ValueError):
    pass


class PacketType(IntEnum):
    HELLO = 1
    VIDEO_JPEG = 2
    AUDIO_PCM16 = 3
    IMU_JSON = 4
    CONTROL_JSON = 5
    SPEECH_PCM16 = 6
    PING = 7
    PONG = 8


MAGIC = b"AGL1"
VERSION = 1
HEADER = struct.Struct("<4sBBHQQII")
MAX_PAYLOAD_BYTES = 1_048_576


@dataclass(frozen=True)
class Packet:
    packet_type: PacketType
    seq: int
    timestamp_ms: int
    payload: bytes = b""
    flags: int = 0

    def pack(self) -> bytes:
        crc = zlib.crc32(self.payload) & 0xFFFFFFFF
        header = HEADER.pack(
            MAGIC,
            VERSION,
            int(self.packet_type),
            self.flags,
            self.seq,
            self.timestamp_ms,
            len(self.payload),
            crc,
        )
        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes, *, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> "Packet":
        if len(data) < HEADER.size:
            raise ProtocolError("packet too short")
        magic, version, typ, flags, seq, timestamp_ms, payload_len, crc = HEADER.unpack(
            data[: HEADER.size]
        )
        if magic != MAGIC:
            raise ProtocolError("bad magic")
        if version != VERSION:
            raise ProtocolError(f"unsupported protocol version: {version}")
        actual_payload_len = len(data) - HEADER.size
        if payload_len > max_payload_bytes or actual_payload_len > max_payload_bytes:
            raise ProtocolError(
                f"payload too large: {max(payload_len, actual_payload_len)} bytes exceeds "
                f"{max_payload_bytes}"
            )
        payload = data[HEADER.size :]
        if len(payload) != payload_len:
            raise ProtocolError("payload length mismatch")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise ProtocolError("payload crc mismatch")
        try:
            packet_type = PacketType(typ)
        except ValueError as exc:
            raise ProtocolError(f"unknown packet type: {typ}") from exc
        return cls(packet_type, seq, timestamp_ms, payload, flags)
