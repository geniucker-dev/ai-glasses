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
    return value if isinstance(value, dict) else {}


def _known_fields(cls: type, data: dict) -> dict:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Copy config.example.toml to config.toml."
        )

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    device = _section(raw, "device")
    capture = _section(device, "capture")
    wifi = _section(device, "wifi")
    audio_down = _section(device, "audio_down")
    vision = _section(raw, "vision")

    return AppConfig(
        path=config_path,
        server=ServerConfig(**_section(raw, "server")),
        device=DeviceConfig(
            id=str(device.get("id", DeviceConfig.id)),
            wifi=DeviceWifiConfig(**wifi),
            capture=DeviceCaptureConfig(**capture),
            audio_down=DeviceAudioDownConfig(**audio_down),
        ),
        models=ModelsConfig(**_known_fields(ModelsConfig, _section(raw, "models"))),
        vision_thresholds=VisionThresholds(**_section(vision, "thresholds")),
        asr=AsrConfig(**_section(raw, "asr")),
        speech=SpeechConfig(**_section(raw, "speech")),
        web=WebConfig(**_section(raw, "web")),
    )
