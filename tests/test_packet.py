import unittest

from aiglasses.protocol import MAX_PAYLOAD_BYTES, Packet, PacketType, ProtocolError


class PacketTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        packet = Packet(PacketType.VIDEO_JPEG, seq=7, timestamp_ms=1234, payload=b"abc")
        decoded = Packet.unpack(packet.pack())
        self.assertEqual(decoded.packet_type, PacketType.VIDEO_JPEG)
        self.assertEqual(decoded.seq, 7)
        self.assertEqual(decoded.timestamp_ms, 1234)
        self.assertEqual(decoded.payload, b"abc")

    def test_crc_rejects_bad_payload(self) -> None:
        raw = bytearray(Packet(PacketType.AUDIO_PCM16, 1, 2, b"abcd").pack())
        raw[-1] ^= 0xFF
        with self.assertRaises(ProtocolError):
            Packet.unpack(bytes(raw))

    def test_unpack_rejects_payload_over_default_limit(self) -> None:
        packet = Packet(PacketType.CONTROL_JSON, seq=1, timestamp_ms=2, payload=b"{}").pack()
        raw = bytearray(packet)
        raw[24:28] = (MAX_PAYLOAD_BYTES + 1).to_bytes(4, "little")

        with self.assertRaisesRegex(ProtocolError, "payload too large"):
            Packet.unpack(bytes(raw))

    def test_unpack_allows_custom_payload_limit_boundary(self) -> None:
        packet = Packet(PacketType.CONTROL_JSON, seq=1, timestamp_ms=2, payload=b"abcd")

        decoded = Packet.unpack(packet.pack(), max_payload_bytes=4)

        self.assertEqual(decoded.payload, b"abcd")

    def test_unpack_rejects_custom_payload_limit_overflow(self) -> None:
        packet = Packet(PacketType.CONTROL_JSON, seq=1, timestamp_ms=2, payload=b"abcd")

        with self.assertRaisesRegex(ProtocolError, "payload too large"):
            Packet.unpack(packet.pack(), max_payload_bytes=3)


if __name__ == "__main__":
    unittest.main()
