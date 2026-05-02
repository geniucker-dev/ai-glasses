import unittest

from aiglasses.protocol import Packet, PacketType, ProtocolError


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


if __name__ == "__main__":
    unittest.main()
