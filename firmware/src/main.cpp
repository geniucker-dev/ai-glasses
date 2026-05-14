#include <Arduino.h>
#include <ArduinoWebsockets.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ctype.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_wifi.h>
#include <mbedtls/md.h>
#include <SPI.h>
#include "ESP_I2S.h"
#include "camera_pins.h"
#include "generated_config.h"
#include "protocol.h"

using namespace websockets;

static constexpr int STATUS_LED_PIN = 21;
static constexpr int I2S_MIC_CLOCK_PIN = 42;
static constexpr int I2S_MIC_DATA_PIN = 41;
static constexpr int I2S_SPK_BCLK = 7;
static constexpr int I2S_SPK_LRCK = 8;
static constexpr int I2S_SPK_DIN = 9;
static constexpr int IMU_SPI_MISO = 1;
static constexpr int IMU_SPI_MOSI = 2;
static constexpr int IMU_SPI_SCK = 3;
static constexpr int IMU_SPI_CS = 4;
static constexpr int MAX_TARGET_VIDEO_FPS = 1000;

#define AGL_CAMERA_PROFILE_DEFAULT 0
#define AGL_CAMERA_PROFILE_TRAFFIC_SIGNAL 1
#ifndef AGL_CAMERA_PROFILE
#define AGL_CAMERA_PROFILE AGL_CAMERA_PROFILE_TRAFFIC_SIGNAL
#endif

#ifndef AGL_VIDEO_PACKET_CAPACITY
#define AGL_VIDEO_PACKET_CAPACITY (240 * 1024)
#endif
#ifndef AGL_VIDEO_TRANSPORT_UDP
#define AGL_VIDEO_TRANSPORT_UDP 0
#endif
#ifndef AGL_VIDEO_UDP_CHUNK_BYTES
#define AGL_VIDEO_UDP_CHUNK_BYTES 1200
#endif
#ifndef AGL_VIDEO_AUTH_KEY
#define AGL_VIDEO_AUTH_KEY {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#endif

static constexpr int AUDIO_BYTES_PER_CHUNK = AGL_AUDIO_SAMPLE_RATE * AGL_AUDIO_CHUNK_MS / 1000 * 2;
static constexpr size_t PACKET_HEADER_SIZE = sizeof(PacketHeader);
static constexpr size_t CONTROL_PACKET_CAPACITY = 1024;
static constexpr size_t VIDEO_PACKET_CAPACITY = AGL_VIDEO_PACKET_CAPACITY;
static constexpr size_t RTP_HEADER_SIZE = 12;
static constexpr size_t RTP_JPEG_HEADER_SIZE = 8;
static constexpr size_t RTP_JPEG_QTABLE_HEADER_SIZE = 4;
static constexpr size_t RTP_JPEG_QTABLE_BYTES = 128;
static constexpr uint8_t RTP_JPEG_PAYLOAD_TYPE = 26;
static constexpr uint8_t RTP_JPEG_DYNAMIC_Q = 255;
static constexpr uint32_t RTP_VIDEO_CLOCK_HZ = 90000;
static constexpr size_t RTP_EXTENSION_HEADER_SIZE = 4;
static constexpr size_t RTP_AUTH_EXTENSION_BYTES = 20;
static constexpr size_t RTP_AUTH_TAG_BYTES = 16;
static constexpr uint16_t RTP_AUTH_EXTENSION_PROFILE = 0xA147;
static constexpr uint16_t RTP_AUTH_EXTENSION_WORDS = RTP_AUTH_EXTENSION_BYTES / 4;
static constexpr size_t SPEECH_SAMPLES_PER_WRITE = 256;
static constexpr size_t SPEECH_PCM16_MAX_PAYLOAD = 64 * 1024;
static const uint8_t RTP_AUTH_MAGIC[4] = {'A', 'G', 'L', 'A'};
static const uint8_t VIDEO_AUTH_KEY[32] = AGL_VIDEO_AUTH_KEY;
static const uint8_t JPEG_STD_DHT[] = {
  0xFF, 0xC4, 0x01, 0xA2, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01,
  0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x10, 0x00, 0x02,
  0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
  0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31,
  0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91,
  0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33,
  0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26,
  0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43,
  0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
  0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
  0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A,
  0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
  0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
  0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
  0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
  0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0x01, 0x00, 0x03, 0x01,
  0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
  0x0B, 0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05,
  0x04, 0x04, 0x00, 0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04,
  0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22,
  0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33,
  0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25,
  0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36,
  0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
  0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66,
  0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
  0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94,
  0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
  0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
  0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
  0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
  0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
};
static const uint8_t JPEG_STD_SOS[] = {3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0};

WebsocketsClient wsControl;
WebsocketsClient wsVideo;
WebsocketsClient wsAudio;
WiFiUDP udpVideo;
I2SClass i2sIn;
I2SClass i2sOut;
IPAddress udpVideoRemoteIp;

volatile bool wifiReady = false;
volatile bool controlReady = false;
volatile bool videoReady = false;
volatile bool audioReady = false;
volatile bool helloPending = false;
volatile int targetVideoFps = min(MAX_TARGET_VIDEO_FPS, max(1, AGL_VIDEO_FPS));
volatile int runtimeJpegQuality = AGL_JPEG_QUALITY;
volatile int runtimeAeLevel = -1;
volatile int runtimeSaturation = 1;
volatile int runtimeContrast = 1;
volatile int runtimeSharpness = 1;
volatile int runtimeGainceiling = GAINCEILING_4X;
volatile bool runtimeTrafficSignalProfile = AGL_CAMERA_PROFILE == AGL_CAMERA_PROFILE_TRAFFIC_SIGNAL;
uint64_t seqControl = 0;
uint64_t seqVideo = 0;
uint64_t seqAudio = 0;
uint16_t rtpVideoSequence = 0;
uint32_t rtpVideoTimestampBase = 0;
int64_t rtpVideoStartUs = 0;
uint32_t videoSsrc = 0;

static uint8_t audioPacket[PACKET_HEADER_SIZE + AUDIO_BYTES_PER_CHUNK];
static uint8_t audioRaw[AUDIO_BYTES_PER_CHUNK];
static uint8_t controlPacket[CONTROL_PACKET_CAPACITY];
static int32_t speechOut[SPEECH_SAMPLES_PER_WRITE * 2];
static uint8_t* videoPacket = nullptr;
SemaphoreHandle_t controlMutex = nullptr;
SemaphoreHandle_t videoMutex = nullptr;
SemaphoreHandle_t audioMutex = nullptr;

struct RtpJpegFrame {
  const uint8_t* scan = nullptr;
  uint32_t scanLen = 0;
  uint8_t quantTables[RTP_JPEG_QTABLE_BYTES];
  uint8_t jpegType = 1;
  uint8_t widthBlocks = 0;
  uint8_t heightBlocks = 0;
};

bool fill_packet(uint8_t* packet, size_t capacity, PacketType type, uint64_t& seq, const uint8_t* payload, uint32_t len) {
  if (!packet || capacity < PACKET_HEADER_SIZE || len > capacity - PACKET_HEADER_SIZE) return false;
  write_packet_header(packet, type, seq++, payload, len);
  memcpy(packet + PACKET_HEADER_SIZE, payload, len);
  return true;
}

bool send_packet(WebsocketsClient& ws, SemaphoreHandle_t mutex, PacketType type, uint64_t& seq, const uint8_t* payload, uint32_t len) {
  if (!mutex || xSemaphoreTake(mutex, pdMS_TO_TICKS(250)) != pdTRUE) return false;
  bool sent = false;
  if (ws.available()) {
    if (type == PKT_AUDIO_PCM16) {
      sent = fill_packet(audioPacket, sizeof(audioPacket), type, seq, payload, len) &&
        ws.sendBinary((const char*)audioPacket, PACKET_HEADER_SIZE + len);
    } else if (type == PKT_VIDEO_JPEG) {
      if (fill_packet(videoPacket, VIDEO_PACKET_CAPACITY, type, seq, payload, len)) {
        sent = ws.sendBinary((const char*)videoPacket, PACKET_HEADER_SIZE + len);
      } else {
        Serial.printf("[WS video] packet too large: %lu bytes\n", (unsigned long)len);
      }
    } else {
      sent = fill_packet(controlPacket, sizeof(controlPacket), type, seq, payload, len) &&
        ws.sendBinary((const char*)controlPacket, PACKET_HEADER_SIZE + len);
    }
  }
  xSemaphoreGive(mutex);
  return sent;
}

void write_u16_be(uint8_t* out, uint16_t value) {
  out[0] = (uint8_t)((value >> 8) & 0xFF);
  out[1] = (uint8_t)(value & 0xFF);
}

void write_u24_be(uint8_t* out, uint32_t value) {
  out[0] = (uint8_t)((value >> 16) & 0xFF);
  out[1] = (uint8_t)((value >> 8) & 0xFF);
  out[2] = (uint8_t)(value & 0xFF);
}

void write_u32_be(uint8_t* out, uint32_t value) {
  out[0] = (uint8_t)((value >> 24) & 0xFF);
  out[1] = (uint8_t)((value >> 16) & 0xFF);
  out[2] = (uint8_t)((value >> 8) & 0xFF);
  out[3] = (uint8_t)(value & 0xFF);
}

uint16_t read_u16_be(const uint8_t* data) {
  return (uint16_t)((data[0] << 8) | data[1]);
}

uint32_t rtp_video_timestamp_now() {
  int64_t elapsedUs = esp_timer_get_time() - rtpVideoStartUs;
  if (elapsedUs < 0) elapsedUs = 0;
  uint64_t ticks = ((uint64_t)elapsedUs * RTP_VIDEO_CLOCK_HZ) / 1000000ULL;
  return rtpVideoTimestampBase + (uint32_t)ticks;
}

bool write_rtp_auth_tag(uint8_t* packet, uint32_t payloadOffset, uint32_t packetLen) {
  const mbedtls_md_info_t* info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  if (!info) return false;
  mbedtls_md_context_t ctx;
  mbedtls_md_init(&ctx);
  uint8_t digest[32];
  bool ok = mbedtls_md_setup(&ctx, info, 1) == 0 &&
    mbedtls_md_hmac_starts(&ctx, VIDEO_AUTH_KEY, sizeof(VIDEO_AUTH_KEY)) == 0 &&
    mbedtls_md_hmac_update(&ctx, packet, RTP_HEADER_SIZE) == 0 &&
    mbedtls_md_hmac_update(&ctx, packet + RTP_HEADER_SIZE, RTP_EXTENSION_HEADER_SIZE) == 0 &&
    mbedtls_md_hmac_update(&ctx,
                           packet + RTP_HEADER_SIZE + RTP_EXTENSION_HEADER_SIZE,
                           sizeof(RTP_AUTH_MAGIC)) == 0 &&
    mbedtls_md_hmac_update(&ctx, packet + payloadOffset, packetLen - payloadOffset) == 0 &&
    mbedtls_md_hmac_finish(&ctx, digest) == 0;
  mbedtls_md_free(&ctx);
  if (!ok) return false;
  memcpy(packet + RTP_HEADER_SIZE + RTP_EXTENSION_HEADER_SIZE + sizeof(RTP_AUTH_MAGIC),
         digest,
         RTP_AUTH_TAG_BYTES);
  return true;
}

bool sof0_matches_rtp_jpeg_subset(const uint8_t* segment, uint16_t dataLen) {
  if (dataLen < 15 || segment[0] != 8 || segment[5] != 3) return false;
  if (segment[6] != 1 || segment[8] != 0) return false;
  if (segment[9] != 2 || segment[10] != 0x11 || segment[11] != 1) return false;
  if (segment[12] != 3 || segment[13] != 0x11 || segment[14] != 1) return false;
  return segment[7] == 0x21 || segment[7] == 0x22;
}

uint16_t dht_table_len(const uint8_t* table, uint16_t available) {
  if (available < 17) return 0;
  uint16_t symbols = 0;
  for (uint8_t i = 1; i <= 16; ++i) symbols += table[i];
  uint16_t tableLen = 17 + symbols;
  return tableLen <= available ? tableLen : 0;
}

bool dht_table_matches_standard(const uint8_t* table, uint16_t tableLen, uint8_t& tableBit) {
  uint16_t pos = 4;
  uint16_t stdLen = sizeof(JPEG_STD_DHT);
  while (pos < stdLen) {
    uint16_t candidateLen = dht_table_len(JPEG_STD_DHT + pos, stdLen - pos);
    if (candidateLen == 0) return false;
    if (candidateLen == tableLen && memcmp(table, JPEG_STD_DHT + pos, tableLen) == 0) {
      uint8_t info = table[0];
      if (info == 0x00) tableBit = 0x01;
      else if (info == 0x10) tableBit = 0x02;
      else if (info == 0x01) tableBit = 0x04;
      else if (info == 0x11) tableBit = 0x08;
      else return false;
      return true;
    }
    pos += candidateLen;
  }
  return false;
}

bool dht_segment_matches_rtp_jpeg_subset(const uint8_t* segment,
                                         uint16_t dataLen,
                                         uint8_t& tableMask) {
  uint16_t pos = 0;
  while (pos < dataLen) {
    uint16_t tableLen = dht_table_len(segment + pos, dataLen - pos);
    if (tableLen == 0) return false;
    uint8_t tableBit = 0;
    if (!dht_table_matches_standard(segment + pos, tableLen, tableBit)) return false;
    tableMask |= tableBit;
    pos += tableLen;
  }
  return pos == dataLen;
}

bool parse_rtp_jpeg_frame(const uint8_t* jpeg, uint32_t len, RtpJpegFrame& out) {
  // Packetize the camera JPEG into the RFC 2435 baseline JPEG variants understood by the backend.
  if (!jpeg || len < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) return false;
  bool haveQ0 = false;
  bool haveQ1 = false;
  bool haveSof = false;
  uint8_t dhtTableMask = 0;
  uint32_t pos = 2;

  while (pos + 4 <= len) {
    while (pos < len && jpeg[pos] != 0xFF) pos++;
    if (pos + 1 >= len) return false;
    while (pos + 1 < len && jpeg[pos + 1] == 0xFF) pos++;
    uint8_t marker = jpeg[pos + 1];
    pos += 2;

    if (marker == 0xD9) break;
    if (marker == 0xDA) {
      if (pos + 2 > len) return false;
      uint16_t segLen = read_u16_be(jpeg + pos);
      if (segLen < 2 || pos + segLen > len) return false;
      if (segLen != sizeof(JPEG_STD_SOS) + 2 ||
          memcmp(jpeg + pos + 2, JPEG_STD_SOS, sizeof(JPEG_STD_SOS)) != 0) {
        return false;
      }
      out.scan = jpeg + pos + segLen;
      uint32_t scanEnd = len;
      if (len >= 2 && jpeg[len - 2] == 0xFF && jpeg[len - 1] == 0xD9) {
        scanEnd = len - 2;
      }
      if (scanEnd <= pos + segLen) return false;
      out.scanLen = scanEnd - (pos + segLen);
      return haveQ0 && haveQ1 && haveSof;
    }
    if ((marker >= 0xD0 && marker <= 0xD7) || marker == 0x01) continue;
    if (pos + 2 > len) return false;
    uint16_t segLen = read_u16_be(jpeg + pos);
    if (segLen < 2 || pos + segLen > len) return false;
    const uint8_t* segment = jpeg + pos + 2;
    uint16_t dataLen = segLen - 2;

    if (marker == 0xDB) {
      uint16_t tablePos = 0;
      while (tablePos + 65 <= dataLen) {
        uint8_t info = segment[tablePos++];
        uint8_t precision = info >> 4;
        uint8_t tableId = info & 0x0F;
        if (precision != 0) return false;
        if (tableId <= 1) {
          memcpy(out.quantTables + tableId * 64, segment + tablePos, 64);
          if (tableId == 0) haveQ0 = true;
          else haveQ1 = true;
        }
        tablePos += 64;
      }
      if (tablePos != dataLen) return false;
    } else if (marker == 0xC0) {
      if (!sof0_matches_rtp_jpeg_subset(segment, dataLen)) return false;
      uint16_t height = read_u16_be(segment + 1);
      uint16_t width = read_u16_be(segment + 3);
      uint8_t lumaSampling = segment[7];
      if (lumaSampling == 0x21) out.jpegType = 0;
      else if (lumaSampling == 0x22) out.jpegType = 1;
      else return false;
      out.widthBlocks = (uint8_t)min(255, max(1, (int)((width + 7) / 8)));
      out.heightBlocks = (uint8_t)min(255, max(1, (int)((height + 7) / 8)));
      haveSof = true;
    } else if (marker == 0xC4) {
      if (!dht_segment_matches_rtp_jpeg_subset(segment, dataLen, dhtTableMask)) return false;
    } else if (marker == 0xDD) {
      return false;
    }
    pos += segLen;
  }
  return false;
}

bool setup_udp_video() {
  if (!AGL_VIDEO_TRANSPORT_UDP) return true;
  if (WiFi.hostByName(AGL_SERVER_HOST, udpVideoRemoteIp) != 1) {
    Serial.println("[UDP video] DNS failed");
    return false;
  }
  if (udpVideo.begin(0) != 1) {
    Serial.println("[UDP video] begin failed");
    return false;
  }
  videoReady = true;
  Serial.printf("[UDP video] ready remote=%s:%u chunk=%u\n",
                udpVideoRemoteIp.toString().c_str(),
                (unsigned)AGL_SERVER_PORT,
                (unsigned)AGL_VIDEO_UDP_CHUNK_BYTES);
  return true;
}

bool send_udp_video_frame(const uint8_t* jpeg, uint32_t len) {
  if (!videoReady || !videoPacket || AGL_VIDEO_UDP_CHUNK_BYTES == 0) return false;
  RtpJpegFrame frame;
  if (!parse_rtp_jpeg_frame(jpeg, len, frame)) {
    Serial.println("[UDP video] unsupported JPEG for RTP/JPEG packetization");
    return false;
  }
  uint32_t authHeaderBytes = RTP_EXTENSION_HEADER_SIZE + RTP_AUTH_EXTENSION_BYTES;
  uint32_t maxPayloadBytes = min((uint32_t)AGL_VIDEO_UDP_CHUNK_BYTES, (uint32_t)1400);
  if (maxPayloadBytes <= RTP_JPEG_QTABLE_HEADER_SIZE + RTP_JPEG_QTABLE_BYTES) return false;
  uint32_t frameTimestamp = rtp_video_timestamp_now();
  bool ok = true;

  if (xSemaphoreTake(videoMutex, pdMS_TO_TICKS(250)) != pdTRUE) return false;
  uint32_t offset = 0;
  while (offset < frame.scanLen) {
    bool firstPacket = offset == 0;
    uint32_t headerExtra = firstPacket
      ? (RTP_JPEG_QTABLE_HEADER_SIZE + RTP_JPEG_QTABLE_BYTES)
      : 0;
    uint32_t partCapacity = maxPayloadBytes - headerExtra;
    uint32_t partLen = min(partCapacity, frame.scanLen - offset);
    bool marker = offset + partLen >= frame.scanLen;
    uint32_t packetLen = RTP_HEADER_SIZE + authHeaderBytes + RTP_JPEG_HEADER_SIZE +
      headerExtra + partLen;
    if (packetLen > VIDEO_PACKET_CAPACITY) {
      ok = false;
      break;
    }

    videoPacket[0] = 0x90;
    videoPacket[1] = (marker ? 0x80 : 0x00) | RTP_JPEG_PAYLOAD_TYPE;
    write_u16_be(videoPacket + 2, rtpVideoSequence++);
    write_u32_be(videoPacket + 4, frameTimestamp);
    write_u32_be(videoPacket + 8, videoSsrc);

    size_t extensionHeader = RTP_HEADER_SIZE;
    write_u16_be(videoPacket + extensionHeader, RTP_AUTH_EXTENSION_PROFILE);
    write_u16_be(videoPacket + extensionHeader + 2, RTP_AUTH_EXTENSION_WORDS);
    size_t extensionData = extensionHeader + RTP_EXTENSION_HEADER_SIZE;
    memcpy(videoPacket + extensionData, RTP_AUTH_MAGIC, sizeof(RTP_AUTH_MAGIC));
    memset(videoPacket + extensionData + sizeof(RTP_AUTH_MAGIC), 0, RTP_AUTH_TAG_BYTES);

    size_t jpegHeader = RTP_HEADER_SIZE + authHeaderBytes;
    videoPacket[jpegHeader] = 0;
    write_u24_be(videoPacket + jpegHeader + 1, offset);
    videoPacket[jpegHeader + 4] = frame.jpegType;
    videoPacket[jpegHeader + 5] = RTP_JPEG_DYNAMIC_Q;
    videoPacket[jpegHeader + 6] = frame.widthBlocks;
    videoPacket[jpegHeader + 7] = frame.heightBlocks;

    size_t payloadOffset = RTP_HEADER_SIZE + authHeaderBytes + RTP_JPEG_HEADER_SIZE;
    if (firstPacket) {
      videoPacket[payloadOffset] = 0;
      videoPacket[payloadOffset + 1] = 0;
      write_u16_be(videoPacket + payloadOffset + 2, RTP_JPEG_QTABLE_BYTES);
      memcpy(videoPacket + payloadOffset + RTP_JPEG_QTABLE_HEADER_SIZE,
             frame.quantTables,
             RTP_JPEG_QTABLE_BYTES);
      payloadOffset += RTP_JPEG_QTABLE_HEADER_SIZE + RTP_JPEG_QTABLE_BYTES;
    }
    memcpy(videoPacket + payloadOffset, frame.scan + offset, partLen);
    if (!write_rtp_auth_tag(videoPacket, RTP_HEADER_SIZE + authHeaderBytes, packetLen)) {
      ok = false;
      break;
    }
    if (!udpVideo.beginPacket(udpVideoRemoteIp, AGL_SERVER_PORT) ||
        udpVideo.write(videoPacket, packetLen) != packetLen ||
        udpVideo.endPacket() != 1) {
      ok = false;
      break;
    }
    offset += partLen;
  }
  xSemaphoreGive(videoMutex);
  if (!ok) Serial.println("[UDP video] frame send failed");
  return ok;
}

bool send_video_frame(const uint8_t* jpeg, uint32_t len) {
  if (AGL_VIDEO_TRANSPORT_UDP) {
    return send_udp_video_frame(jpeg, len);
  }
  return send_packet(wsVideo, videoMutex, PKT_VIDEO_JPEG, seqVideo, jpeg, len);
}

void apply_camera_profile(sensor_t* s) {
  s->set_hmirror(s, 1);
  s->set_vflip(s, 0);
  s->set_quality(s, runtimeJpegQuality);

  if (runtimeTrafficSignalProfile) {
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_aec2(s, 1);
    s->set_ae_level(s, runtimeAeLevel);
    s->set_gainceiling(s, (gainceiling_t)runtimeGainceiling);
    s->set_saturation(s, runtimeSaturation);
    s->set_contrast(s, runtimeContrast);
    s->set_sharpness(s, runtimeSharpness);
    s->set_denoise(s, 1);
    s->set_bpc(s, 1);
    s->set_wpc(s, 1);
    s->set_lenc(s, 1);
  } else {
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_aec2(s, 0);
    s->set_ae_level(s, 0);
    s->set_gainceiling(s, GAINCEILING_2X);
    s->set_saturation(s, 0);
    s->set_contrast(s, 0);
    s->set_sharpness(s, 0);
    s->set_denoise(s, 0);
    s->set_bpc(s, 0);
    s->set_wpc(s, 1);
    s->set_lenc(s, 1);
  }
}

bool init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = AGL_FRAME_SIZE;
  config.jpeg_quality = AGL_JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init failed: 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    apply_camera_profile(s);
  }
  return true;
}

void init_audio() {
  i2sIn.setPinsPdmRx(I2S_MIC_CLOCK_PIN, I2S_MIC_DATA_PIN);
  if (!i2sIn.begin(I2S_MODE_PDM_RX, AGL_AUDIO_SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("[I2S IN] init failed");
    delay(1500);
    esp_restart();
  }
  i2sOut.setPins(I2S_SPK_BCLK, I2S_SPK_LRCK, I2S_SPK_DIN);
  i2sOut.begin(I2S_MODE_STD, AGL_AUDIO_SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
}

static inline void imu_cs_low() { digitalWrite(IMU_SPI_CS, LOW); }
static inline void imu_cs_high() { digitalWrite(IMU_SPI_CS, HIGH); }

uint8_t imu_read8(uint8_t reg) {
  imu_cs_low();
  SPI.transfer(reg | 0x80);
  uint8_t value = SPI.transfer(0);
  imu_cs_high();
  return value;
}

void imu_write8(uint8_t reg, uint8_t value) {
  imu_cs_low();
  SPI.transfer(reg & 0x7F);
  SPI.transfer(value);
  imu_cs_high();
}

void imu_readn(uint8_t reg, uint8_t* dst, size_t len) {
  imu_cs_low();
  SPI.transfer(reg | 0x80);
  for (size_t i = 0; i < len; ++i) dst[i] = SPI.transfer(0);
  imu_cs_high();
}

bool init_imu() {
  SPI.begin(IMU_SPI_SCK, IMU_SPI_MISO, IMU_SPI_MOSI, IMU_SPI_CS);
  pinMode(IMU_SPI_CS, OUTPUT);
  imu_cs_high();
  delay(5);
  uint8_t who = imu_read8(0x75);
  Serial.printf("[IMU] WHO_AM_I=0x%02X\n", who);
  if (who != 0x47) return false;
  imu_write8(0x4E, 0x0F);
  delay(10);
  return true;
}

bool send_hello() {
  String payload = String("{\"device_id\":\"") + AGL_DEVICE_ID + "\",\"kind\":\"hello\",\"fw\":\"0.1.0\"}";
  return send_packet(wsControl, controlMutex, PKT_HELLO, seqControl, (const uint8_t*)payload.c_str(), payload.length());
}

bool read_json_int(const String& payload, const char* key, int& value) {
  String needle = String("\"") + key + "\"";
  int keyPos = payload.indexOf(needle);
  if (keyPos < 0) return false;
  int colonPos = payload.indexOf(':', keyPos + needle.length());
  if (colonPos < 0) return false;

  int pos = colonPos + 1;
  while (pos < payload.length() && isspace((unsigned char)payload[pos])) pos++;
  bool negative = false;
  if (pos < payload.length() && payload[pos] == '-') {
    negative = true;
    pos++;
  }

  long parsed = 0;
  bool hasDigit = false;
  while (pos < payload.length() && isdigit((unsigned char)payload[pos])) {
    parsed = parsed * 10 + (payload[pos] - '0');
    hasDigit = true;
    pos++;
  }
  if (!hasDigit) return false;
  value = negative ? -parsed : parsed;
  return true;
}

bool read_json_string(const String& payload, const char* key, String& value) {
  String needle = String("\"") + key + "\"";
  int keyPos = payload.indexOf(needle);
  if (keyPos < 0) return false;
  int colonPos = payload.indexOf(':', keyPos + needle.length());
  if (colonPos < 0) return false;
  int pos = colonPos + 1;
  while (pos < payload.length() && isspace((unsigned char)payload[pos])) pos++;
  if (pos >= payload.length() || payload[pos] != '"') return false;
  pos++;
  int end = payload.indexOf('"', pos);
  if (end < 0) return false;
  value = payload.substring(pos, end);
  return true;
}

int clamp_video_fps(int fps) {
  return min(MAX_TARGET_VIDEO_FPS, max(1, fps));
}

int gainceiling_value(int value) {
  if (value <= 2) return GAINCEILING_2X;
  if (value <= 4) return GAINCEILING_4X;
  if (value <= 8) return GAINCEILING_8X;
  if (value <= 16) return GAINCEILING_16X;
  if (value <= 32) return GAINCEILING_32X;
  if (value <= 64) return GAINCEILING_64X;
  return GAINCEILING_128X;
}

void apply_runtime_camera_config() {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  apply_camera_profile(s);
}

void handle_control_message(const String& payload) {
  int value = 0;
  bool cameraChanged = false;
  if (read_json_int(payload, "target_fps", value) || read_json_int(payload, "video_fps", value)) {
    targetVideoFps = clamp_video_fps(value);
    Serial.printf("[CFG] target_fps=%d\n", targetVideoFps);
  }
  if (read_json_int(payload, "jpeg_quality", value)) {
    runtimeJpegQuality = min(63, max(1, value));
    cameraChanged = true;
    Serial.printf("[CFG] jpeg_quality=%d\n", runtimeJpegQuality);
  }
  if (read_json_int(payload, "ae_level", value)) {
    runtimeAeLevel = min(2, max(-2, value));
    cameraChanged = true;
  }
  if (read_json_int(payload, "saturation", value)) {
    runtimeSaturation = min(2, max(-2, value));
    cameraChanged = true;
  }
  if (read_json_int(payload, "contrast", value)) {
    runtimeContrast = min(2, max(-2, value));
    cameraChanged = true;
  }
  if (read_json_int(payload, "sharpness", value)) {
    runtimeSharpness = min(2, max(-2, value));
    cameraChanged = true;
  }
  if (read_json_int(payload, "gainceiling", value)) {
    runtimeGainceiling = gainceiling_value(value);
    cameraChanged = true;
  }
  String profile;
  if (read_json_string(payload, "camera_profile", profile)) {
    runtimeTrafficSignalProfile = profile != "default";
    cameraChanged = true;
  }
  if (cameraChanged) {
    apply_runtime_camera_config();
  }
}

bool parse_packet_header(const uint8_t* data, size_t len, PacketHeader& header) {
  if (len < PACKET_HEADER_SIZE) return false;
  memcpy(&header, data, PACKET_HEADER_SIZE);
  if (memcmp(header.magic, "AGL1", 4) != 0) return false;
  if (header.version != 1) return false;
  size_t payload_len = len - PACKET_HEADER_SIZE;
  if (header.payload_len != payload_len) return false;
  if (header.type == PKT_SPEECH_PCM16 && payload_len > SPEECH_PCM16_MAX_PAYLOAD) return false;
  const uint8_t* payload = data + PACKET_HEADER_SIZE;
  return crc32_update(0, payload, header.payload_len) == header.crc32;
}

void play_speech_pcm16(const uint8_t* payload, uint32_t len) {
#if AGL_AUDIO_DOWN_ENABLED
  size_t offset = 0;
  while (offset + 1 < len) {
    size_t remainingSamples = (size_t)(len - offset) / 2;
    size_t samples = min(SPEECH_SAMPLES_PER_WRITE, remainingSamples);
    for (size_t i = 0; i < samples; ++i) {
      int16_t sample = (int16_t)(payload[offset] | (payload[offset + 1] << 8));
      int32_t expanded = ((int32_t)sample) << 16;
      speechOut[i * 2] = expanded;
      speechOut[i * 2 + 1] = expanded;
      offset += 2;
    }
    i2sOut.write((const uint8_t*)speechOut, samples * 2 * sizeof(int32_t));
  }
#else
  (void)payload;
  (void)len;
#endif
}

void handle_control_binary(const uint8_t* data, size_t len) {
  PacketHeader header;
  if (!parse_packet_header(data, len, header)) {
    Serial.println("[WS control] dropping bad binary packet");
    return;
  }
  const uint8_t* payload = data + PACKET_HEADER_SIZE;
  if (header.type == PKT_SPEECH_PCM16) {
    play_speech_pcm16(payload, header.payload_len);
  }
}

uint32_t current_frame_period_ms() {
  int fps = clamp_video_fps(targetVideoFps);
  uint32_t period = 1000 / fps;
  return period > 0 ? period : 1;
}

void task_camera(void*) {
  uint32_t lastFrame = 0;
  for (;;) {
    uint32_t framePeriod = current_frame_period_ms();
    uint32_t now = millis();
    if (!videoReady || (lastFrame != 0 && now - lastFrame < framePeriod)) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    lastFrame = now;
    if (fb->format == PIXFORMAT_JPEG) {
      send_video_frame(fb->buf, fb->len);
    }
    esp_camera_fb_return(fb);
  }
}

void task_audio(void*) {
  const int samples = AUDIO_BYTES_PER_CHUNK / 2;
  for (;;) {
    if (!audioReady) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    int16_t* out = (int16_t*)audioRaw;
    int filled = 0;
    while (filled < samples) {
      int value = i2sIn.read();
      if (value == -1) {
        delay(1);
      } else {
        out[filled++] = (int16_t)value;
      }
    }
    send_packet(wsAudio, audioMutex, PKT_AUDIO_PCM16, seqAudio, audioRaw, AUDIO_BYTES_PER_CHUNK);
  }
}

void task_imu(void*) {
  bool ready = false;
  const uint32_t period = 1000 / max(1, AGL_IMU_HZ);
  for (;;) {
    if (!ready) {
      ready = init_imu();
      if (!ready) {
        vTaskDelay(pdMS_TO_TICKS(500));
        continue;
      }
    }
    uint8_t raw[14];
    imu_readn(0x1D, raw, sizeof(raw));
    auto s16 = [&](int idx) -> int16_t { return (int16_t)((raw[idx] << 8) | raw[idx + 1]); };
    float temp = (float)s16(0) / 132.48f + 25.0f;
    float ax = (float)s16(2) / 2048.0f * 9.80665f;
    float ay = (float)s16(4) / 2048.0f * 9.80665f;
    float az = (float)s16(6) / 2048.0f * 9.80665f;
    float gx = (float)s16(8) / 16.4f;
    float gy = (float)s16(10) / 16.4f;
    float gz = (float)s16(12) / 16.4f;
    char payload[240];
    int n = snprintf(payload, sizeof(payload),
      "{\"ts\":%lu,\"temp_c\":%.2f,\"accel\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f},\"gyro\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f}}",
      millis(), temp, ax, ay, az, gx, gy, gz);
    if (controlReady && n > 0) {
      send_packet(wsControl, controlMutex, PKT_IMU_JSON, seqControl, (const uint8_t*)payload, n);
    }
    vTaskDelay(pdMS_TO_TICKS(period));
  }
}

void task_led(void*) {
  bool on = false;
  for (;;) {
    if (controlReady && videoReady) {
      digitalWrite(STATUS_LED_PIN, LOW);
    } else if (wifiReady) {
      on = !on;
      digitalWrite(STATUS_LED_PIN, on ? LOW : HIGH);
    } else {
      digitalWrite(STATUS_LED_PIN, HIGH);
    }
    vTaskDelay(pdMS_TO_TICKS(250));
  }
}

void connect_wifi() {
  const unsigned long WIFI_CONNECT_TIMEOUT_MS = 15000;
  const unsigned long WIFI_CONNECT_POLL_MS = 300;

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  WiFi.begin(AGL_WIFI_SSID, AGL_WIFI_PASSWORD);
  Serial.print("[WiFi] connecting");
  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < WIFI_CONNECT_TIMEOUT_MS) {
    Serial.print(".");
    delay(WIFI_CONNECT_POLL_MS);
  }
  if (WiFi.status() != WL_CONNECTED) {
    wifiReady = false;
    Serial.println();
    Serial.printf("[WiFi] connection timed out after %lu ms; restarting\n", WIFI_CONNECT_TIMEOUT_MS);
    delay(1500);
    esp_restart();
  }
  wifiReady = true;
  Serial.println(" " + WiFi.localIP().toString());
}

void setup_ws_handlers() {
  wsControl.onMessage([](WebsocketsMessage message) {
    if (message.isBinary()) {
      const WSString& data = message.rawData();
      handle_control_binary((const uint8_t*)data.data(), data.size());
      return;
    }
    handle_control_message(message.data());
  });
  wsControl.onEvent([](WebsocketsEvent ev, String) {
    if (ev == WebsocketsEvent::ConnectionOpened) {
      controlReady = true;
      Serial.println("[WS control] open");
      helloPending = true;
    } else if (ev == WebsocketsEvent::ConnectionClosed) {
      controlReady = false;
      Serial.println("[WS control] closed");
    }
  });
  wsVideo.onEvent([](WebsocketsEvent ev, String) {
    if (ev == WebsocketsEvent::ConnectionOpened) {
      videoReady = true;
      Serial.println("[WS video] open");
    } else if (ev == WebsocketsEvent::ConnectionClosed) {
      videoReady = false;
      Serial.println("[WS video] closed");
    }
  });
  wsAudio.onEvent([](WebsocketsEvent ev, String) {
    if (ev == WebsocketsEvent::ConnectionOpened) {
      audioReady = true;
      Serial.println("[WS audio] open");
    } else if (ev == WebsocketsEvent::ConnectionClosed) {
      audioReady = false;
      Serial.println("[WS audio] closed");
    }
  });
}

void setup() {
  Serial.begin(115200);
  delay(300);
  controlMutex = xSemaphoreCreateMutex();
  videoMutex = xSemaphoreCreateMutex();
  audioMutex = xSemaphoreCreateMutex();
  if (!controlMutex || !videoMutex || !audioMutex) {
    Serial.println("[MEM] websocket mutex allocation failed");
    delay(1500);
    esp_restart();
  }
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);
  videoPacket = (uint8_t*)heap_caps_malloc(VIDEO_PACKET_CAPACITY, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!videoPacket) videoPacket = (uint8_t*)malloc(VIDEO_PACKET_CAPACITY);
  if (!videoPacket) {
    Serial.println("[MEM] video packet buffer allocation failed");
    delay(1500);
    esp_restart();
  }
  rtpVideoSequence = (uint16_t)(esp_random() & 0xFFFF);
  rtpVideoTimestampBase = esp_random();
  rtpVideoStartUs = esp_timer_get_time();
  videoSsrc = esp_random();
  connect_wifi();
  if (!init_camera()) {
    delay(1500);
    esp_restart();
  }
  init_audio();
  setup_ws_handlers();
  setup_udp_video();
  xTaskCreatePinnedToCore(task_camera, "camera", 8192, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(task_audio, "audio", 4096, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(task_imu, "imu", 4096, nullptr, 1, nullptr, 0);
  xTaskCreatePinnedToCore(task_led, "led", 2048, nullptr, 1, nullptr, 0);
}

void loop() {
  static uint32_t lastRetry = 0;
  if (millis() - lastRetry > 1000) {
    if (xSemaphoreTake(controlMutex, pdMS_TO_TICKS(250)) == pdTRUE) {
      if (!controlReady) wsControl.connect(AGL_SERVER_HOST, AGL_SERVER_PORT, "/ws/device/control");
      xSemaphoreGive(controlMutex);
    }
    if (AGL_VIDEO_TRANSPORT_UDP && !videoReady) {
      setup_udp_video();
    }
    if (!AGL_VIDEO_TRANSPORT_UDP && xSemaphoreTake(videoMutex, pdMS_TO_TICKS(250)) == pdTRUE) {
      if (!videoReady) wsVideo.connect(AGL_SERVER_HOST, AGL_SERVER_PORT, "/ws/device/video");
      xSemaphoreGive(videoMutex);
    }
    if (xSemaphoreTake(audioMutex, pdMS_TO_TICKS(250)) == pdTRUE) {
      if (!audioReady) wsAudio.connect(AGL_SERVER_HOST, AGL_SERVER_PORT, "/ws/device/audio-up");
      xSemaphoreGive(audioMutex);
    }
    lastRetry = millis();
  }
  if (controlReady && xSemaphoreTake(controlMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    wsControl.poll();
    xSemaphoreGive(controlMutex);
  }
  if (!AGL_VIDEO_TRANSPORT_UDP && videoReady && xSemaphoreTake(videoMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    wsVideo.poll();
    xSemaphoreGive(videoMutex);
  }
  if (audioReady && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    wsAudio.poll();
    xSemaphoreGive(audioMutex);
  }
  if (controlReady && helloPending && send_hello()) {
    helloPending = false;
  }
  delay(2);
}
