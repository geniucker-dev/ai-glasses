#include <Arduino.h>
#include <ArduinoWebsockets.h>
#include <WiFi.h>
#include <ctype.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_wifi.h>
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

static constexpr int AUDIO_BYTES_PER_CHUNK = AGL_AUDIO_SAMPLE_RATE * AGL_AUDIO_CHUNK_MS / 1000 * 2;
static constexpr size_t PACKET_HEADER_SIZE = sizeof(PacketHeader);
static constexpr size_t CONTROL_PACKET_CAPACITY = 1024;
static constexpr size_t VIDEO_PACKET_CAPACITY = 240 * 1024;
static constexpr size_t SPEECH_SAMPLES_PER_WRITE = 256;

WebsocketsClient wsControl;
WebsocketsClient wsVideo;
WebsocketsClient wsAudio;
I2SClass i2sIn;
I2SClass i2sOut;

volatile bool wifiReady = false;
volatile bool controlReady = false;
volatile bool videoReady = false;
volatile bool audioReady = false;
volatile bool helloPending = false;
volatile int targetVideoFps = min(MAX_TARGET_VIDEO_FPS, max(1, AGL_VIDEO_FPS));
uint64_t seqControl = 0;
uint64_t seqVideo = 0;
uint64_t seqAudio = 0;

static uint8_t audioPacket[PACKET_HEADER_SIZE + AUDIO_BYTES_PER_CHUNK];
static uint8_t audioRaw[AUDIO_BYTES_PER_CHUNK];
static uint8_t controlPacket[CONTROL_PACKET_CAPACITY];
static int32_t speechOut[SPEECH_SAMPLES_PER_WRITE * 2];
static uint8_t* videoPacket = nullptr;
SemaphoreHandle_t controlMutex = nullptr;
SemaphoreHandle_t videoMutex = nullptr;
SemaphoreHandle_t audioMutex = nullptr;

bool fill_packet(uint8_t* packet, size_t capacity, PacketType type, uint64_t& seq, const uint8_t* payload, uint32_t len) {
  if (!packet || PACKET_HEADER_SIZE + len > capacity) return false;
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
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
    s->set_hmirror(s, 1);
    s->set_vflip(s, 0);
    s->set_quality(s, AGL_JPEG_QUALITY);
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

int clamp_video_fps(int fps) {
  return min(MAX_TARGET_VIDEO_FPS, max(1, fps));
}

void handle_control_message(const String& payload) {
  int fps = 0;
  if (!read_json_int(payload, "target_fps", fps) && !read_json_int(payload, "video_fps", fps)) {
    return;
  }
  targetVideoFps = clamp_video_fps(fps);
  Serial.printf("[CFG] target_fps=%d\n", targetVideoFps);
}

bool parse_packet_header(const uint8_t* data, size_t len, PacketHeader& header) {
  if (len < PACKET_HEADER_SIZE) return false;
  memcpy(&header, data, PACKET_HEADER_SIZE);
  if (memcmp(header.magic, "AGL1", 4) != 0) return false;
  if (header.version != 1) return false;
  if (PACKET_HEADER_SIZE + header.payload_len != len) return false;
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
      send_packet(wsVideo, videoMutex, PKT_VIDEO_JPEG, seqVideo, fb->buf, fb->len);
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
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  WiFi.begin(AGL_WIFI_SSID, AGL_WIFI_PASSWORD);
  Serial.print("[WiFi] connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
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
  connect_wifi();
  if (!init_camera()) {
    delay(1500);
    esp_restart();
  }
  init_audio();
  setup_ws_handlers();
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
    if (xSemaphoreTake(videoMutex, pdMS_TO_TICKS(250)) == pdTRUE) {
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
  if (videoReady && xSemaphoreTake(videoMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
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
