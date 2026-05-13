from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib
from typing import Any


TOP_LEVEL_SECTIONS = {
    "server",
    "device",
    "models",
    "vision",
    "asr",
    "speech",
    "web",
}
AUDIO_MAX_PAYLOAD_BYTES = 64 * 1024
BYTES_PER_PCM16_SAMPLE = 2
DEVICE_ID_MAX_BYTES = 128


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8081
    public_base_url: str = "http://127.0.0.1:8081"


@dataclass(frozen=True)
class DeviceWifiConfig:
    ssid: str = ""
    password: str = ""


@dataclass(frozen=True)
class DeviceCaptureConfig:
    video_fps: int = 6
    jpeg_quality: int = 12
    frame_size: str = "VGA"
    camera_profile: str = "traffic_signal"
    audio_sample_rate: int = 16000
    audio_chunk_ms: int = 100
    imu_hz: int = 50


@dataclass(frozen=True)
class DeviceAudioDownConfig:
    enabled: bool = False


@dataclass(frozen=True)
class DeviceTransportConfig:
    video: str = "ws"
    video_payload_bytes: int = 1200
    video_reorder_window_ms: int = 120
    video_frame_timeout_ms: int = 180
    control: str = "ws"
    audio_up: str = "ws"
    audio_down: str = "ws"


@dataclass(frozen=True)
class DeviceConfig:
    id: str = "xiao-esp32s3-sense-01"
    wifi: DeviceWifiConfig = field(default_factory=DeviceWifiConfig)
    capture: DeviceCaptureConfig = field(default_factory=DeviceCaptureConfig)
    audio_down: DeviceAudioDownConfig = field(default_factory=DeviceAudioDownConfig)
    transport: DeviceTransportConfig = field(default_factory=DeviceTransportConfig)


@dataclass(frozen=True)
class ModelsConfig:
    blind_path: str = "models/yolo-seg.pt"
    obstacle: str = "models/yoloe-11l-seg-obstacle.pt"
    traffic_light: str = "models/trafficlight.pt"
    image_width: int = 640
    image_height: int = 480
    torch_device: str = "cuda:0"
    torch_half: bool = True


@dataclass(frozen=True)
class VisionThresholds:
    blind_path_conf: float = 0.35
    obstacle_conf: float = 0.35
    traffic_light_conf: float = 0.35
    mask_min_area: float = 0.01


@dataclass(frozen=True)
class AsrConfig:
    enabled: bool = True
    provider: str = "dashscope"
    model: str = "paraformer-realtime-v2"
    dashscope_api_key: str = ""
    websocket_base_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    http_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    sample_rate: int = 16000
    language: str = "zh-CN"


@dataclass(frozen=True)
class SpeechConfig:
    enabled: bool = False
    mode: str = "ui"
    provider: str = "dashscope"
    language: str = "auto"
    model: str = "sambert-zhichu-v1"
    audio_format: str = "pcm"
    sample_rate: int = 16000
    volume: int = 50
    rate: float = 1.0
    pitch: float = 1.0
    piper_model_dir: str = "voice"
    piper_voice_zh: str = "zh_CN-huayan-medium"
    piper_voice_en: str = "en_US-lessac-medium"
    piper_use_cuda: bool = False


@dataclass(frozen=True)
class WebConfig:
    title: str = "AI Glasses Console"


@dataclass(frozen=True)
class AppConfig:
    path: Path
    server: ServerConfig = field(default_factory=ServerConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    vision_thresholds: VisionThresholds = field(default_factory=VisionThresholds)
    asr: AsrConfig = field(default_factory=AsrConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    web: WebConfig = field(default_factory=WebConfig)


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table")
    return value


def _build_section(section_name: str, cls: type, data: dict):
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        field_list = ", ".join(unknown)
        raise ValueError(f"Unknown config field in [{section_name}]: {field_list}")
    return cls(**data)


def _validate_int_range(name: str, value: int, *, minimum: int, maximum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Config value {name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"Config value {name} must be >= {minimum}")
        raise ValueError(f"Config value {name} must be between {minimum} and {maximum}")


def _validate_config(config: AppConfig) -> None:
    device_id_bytes = config.device.id.encode("utf-8")
    if not device_id_bytes:
        raise ValueError("Config value device.id must not be empty")
    if len(device_id_bytes) > DEVICE_ID_MAX_BYTES:
        raise ValueError(
            f"Config value device.id must be at most {DEVICE_ID_MAX_BYTES} UTF-8 bytes"
        )
    capture = config.device.capture
    transport = config.device.transport
    _validate_int_range("device.capture.video_fps", capture.video_fps, minimum=1, maximum=1000)
    _validate_int_range("device.capture.jpeg_quality", capture.jpeg_quality, minimum=1, maximum=63)
    _validate_int_range("device.capture.audio_sample_rate", capture.audio_sample_rate, minimum=1)
    _validate_int_range("device.capture.audio_chunk_ms", capture.audio_chunk_ms, minimum=1, maximum=1000)
    _validate_int_range("device.capture.imu_hz", capture.imu_hz, minimum=1, maximum=1000)
    _validate_int_range("models.image_width", config.models.image_width, minimum=1)
    _validate_int_range("models.image_height", config.models.image_height, minimum=1)
    _validate_int_range("asr.sample_rate", config.asr.sample_rate, minimum=1)
    _validate_int_range(
        "device.transport.video_payload_bytes",
        transport.video_payload_bytes,
        minimum=256,
        maximum=1400,
    )
    _validate_int_range(
        "device.transport.video_reorder_window_ms",
        transport.video_reorder_window_ms,
        minimum=0,
        maximum=5000,
    )
    _validate_int_range(
        "device.transport.video_frame_timeout_ms",
        transport.video_frame_timeout_ms,
        minimum=10,
        maximum=5000,
    )
    for field_name in ("video", "control", "audio_up", "audio_down"):
        value = getattr(transport, field_name)
        if value not in {"ws", "udp"}:
            raise ValueError(f"Config value device.transport.{field_name} must be 'ws' or 'udp'")
    if transport.control != "ws" or transport.audio_up != "ws" or transport.audio_down != "ws":
        raise ValueError("Only device.transport.video supports udp; control/audio transports must be 'ws'")
    audio_chunk_bytes = (
        capture.audio_sample_rate * capture.audio_chunk_ms * BYTES_PER_PCM16_SAMPLE // 1000
    )
    if audio_chunk_bytes > AUDIO_MAX_PAYLOAD_BYTES:
        raise ValueError(
            "Config values device.capture.audio_sample_rate and device.capture.audio_chunk_ms "
            f"produce {audio_chunk_bytes} audio bytes per chunk, exceeding "
            f"{AUDIO_MAX_PAYLOAD_BYTES}"
        )
    if config.asr.enabled and config.asr.sample_rate != capture.audio_sample_rate:
        raise ValueError(
            "Config value asr.sample_rate must equal device.capture.audio_sample_rate "
            f"when asr.enabled=true ({config.asr.sample_rate} != {capture.audio_sample_rate})"
        )


def _reject_unknown_top_level_sections(raw: dict[str, Any]) -> None:
    unknown = sorted(set(raw) - TOP_LEVEL_SECTIONS)
    if unknown:
        section_list = ", ".join(unknown)
        raise ValueError(f"Unknown top-level config section: {section_list}")


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Copy config.example.toml to config.toml."
        )

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)
    _reject_unknown_top_level_sections(raw)

    device = _section(raw, "device")
    device_unknown = sorted(set(device) - {"id", "wifi", "capture", "audio_down", "transport"})
    if device_unknown:
        field_list = ", ".join(device_unknown)
        raise ValueError(f"Unknown config field in [device]: {field_list}")
    capture = _section(device, "capture")
    wifi = _section(device, "wifi")
    audio_down = _section(device, "audio_down")
    transport = _section(device, "transport")
    vision = _section(raw, "vision")
    vision_unknown = sorted(set(vision) - {"thresholds"})
    if vision_unknown:
        field_list = ", ".join(vision_unknown)
        raise ValueError(f"Unknown config field in [vision]: {field_list}")

    config = AppConfig(
        path=config_path,
        server=_build_section("server", ServerConfig, _section(raw, "server")),
        device=DeviceConfig(
            id=str(device.get("id", DeviceConfig.id)),
            wifi=_build_section("device.wifi", DeviceWifiConfig, wifi),
            capture=_build_section("device.capture", DeviceCaptureConfig, capture),
            audio_down=_build_section("device.audio_down", DeviceAudioDownConfig, audio_down),
            transport=_build_section("device.transport", DeviceTransportConfig, transport),
        ),
        models=_build_section("models", ModelsConfig, _section(raw, "models")),
        vision_thresholds=_build_section(
            "vision.thresholds", VisionThresholds, _section(vision, "thresholds")
        ),
        asr=_build_section("asr", AsrConfig, _section(raw, "asr")),
        speech=_build_section("speech", SpeechConfig, _section(raw, "speech")),
        web=_build_section("web", WebConfig, _section(raw, "web")),
    )
    _validate_config(config)
    return config
