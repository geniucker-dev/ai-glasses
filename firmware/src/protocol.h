#pragma once

#include <Arduino.h>

enum PacketType : uint8_t {
  PKT_HELLO = 1,
  PKT_VIDEO_JPEG = 2,
  PKT_AUDIO_PCM16 = 3,
  PKT_IMU_JSON = 4,
  PKT_CONTROL_JSON = 5,
  PKT_SPEECH_PCM16 = 6,
  PKT_PING = 7,
  PKT_PONG = 8,
};

struct __attribute__((packed)) PacketHeader {
  char magic[4];
  uint8_t version;
  uint8_t type;
  uint16_t flags;
  uint64_t seq;
  uint64_t timestamp_ms;
  uint32_t payload_len;
  uint32_t crc32;
};

static uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
  crc = ~crc;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int j = 0; j < 8; ++j) {
      crc = (crc >> 1) ^ (0xEDB88320UL & (-(int32_t)(crc & 1)));
    }
  }
  return ~crc;
}

static size_t write_packet_header(uint8_t* out, PacketType type, uint64_t seq, const uint8_t* payload, uint32_t len) {
  PacketHeader h;
  h.magic[0] = 'A'; h.magic[1] = 'G'; h.magic[2] = 'L'; h.magic[3] = '1';
  h.version = 1;
  h.type = (uint8_t)type;
  h.flags = 0;
  h.seq = seq;
  h.timestamp_ms = millis();
  h.payload_len = len;
  h.crc32 = crc32_update(0, payload, len);
  memcpy(out, &h, sizeof(h));
  return sizeof(h);
}
