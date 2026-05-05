from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib


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
class DeviceConfig:
    id: str = "xiao-esp32s3-sense-01"
    wifi: DeviceWifiConfig = field(default_factory=DeviceWifiConfig)
    capture: DeviceCaptureConfig = field(default_factory=DeviceCaptureConfig)
    audio_down: DeviceAudioDownConfig = field(default_factory=DeviceAudioDownConfig)


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


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Copy config.example.toml to config.toml."
        )

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    device = _section(raw, "device")
    device_unknown = sorted(set(device) - {"id", "wifi", "capture", "audio_down"})
    if device_unknown:
        field_list = ", ".join(device_unknown)
        raise ValueError(f"Unknown config field in [device]: {field_list}")
    capture = _section(device, "capture")
    wifi = _section(device, "wifi")
    audio_down = _section(device, "audio_down")
    vision = _section(raw, "vision")
    vision_unknown = sorted(set(vision) - {"thresholds"})
    if vision_unknown:
        field_list = ", ".join(vision_unknown)
        raise ValueError(f"Unknown config field in [vision]: {field_list}")

    return AppConfig(
        path=config_path,
        server=_build_section("server", ServerConfig, _section(raw, "server")),
        device=DeviceConfig(
            id=str(device.get("id", DeviceConfig.id)),
            wifi=_build_section("device.wifi", DeviceWifiConfig, wifi),
            capture=_build_section("device.capture", DeviceCaptureConfig, capture),
            audio_down=_build_section("device.audio_down", DeviceAudioDownConfig, audio_down),
        ),
        models=_build_section("models", ModelsConfig, _section(raw, "models")),
        vision_thresholds=_build_section(
            "vision.thresholds", VisionThresholds, _section(vision, "thresholds")
        ),
        asr=_build_section("asr", AsrConfig, _section(raw, "asr")),
        speech=_build_section("speech", SpeechConfig, _section(raw, "speech")),
        web=_build_section("web", WebConfig, _section(raw, "web")),
    )
