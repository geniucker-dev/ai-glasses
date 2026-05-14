from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from aiglasses.protocol import MAX_PAYLOAD_BYTES

from .settings import load_config


def _cstr(value: object) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _cbytes_from_hex(value: str, expected_bytes: int) -> str:
    data = bytes.fromhex(value)
    if len(data) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, got {len(data)}")
    return "{" + ", ".join(f"0x{byte:02X}" for byte in data) + "}"


FRAME_SIZE_MACROS = {
    "96X96": "FRAMESIZE_96X96",
    "QQVGA": "FRAMESIZE_QQVGA",
    "QCIF": "FRAMESIZE_QCIF",
    "HQVGA": "FRAMESIZE_HQVGA",
    "240X240": "FRAMESIZE_240X240",
    "QVGA": "FRAMESIZE_QVGA",
    "CIF": "FRAMESIZE_CIF",
    "HVGA": "FRAMESIZE_HVGA",
    "VGA": "FRAMESIZE_VGA",
    "SVGA": "FRAMESIZE_SVGA",
    "XGA": "FRAMESIZE_XGA",
    "HD": "FRAMESIZE_HD",
    "SXGA": "FRAMESIZE_SXGA",
    "UXGA": "FRAMESIZE_UXGA",
}

FRAME_SIZE_VIDEO_PACKET_CAPACITY = {
    "96X96": 64 * 1024,
    "QQVGA": 64 * 1024,
    "QCIF": 64 * 1024,
    "HQVGA": 96 * 1024,
    "240X240": 96 * 1024,
    "QVGA": 128 * 1024,
    "CIF": 160 * 1024,
    "HVGA": 192 * 1024,
    "VGA": 240 * 1024,
    "SVGA": 384 * 1024,
    "XGA": 512 * 1024,
    "HD": 768 * 1024,
    "SXGA": 1024 * 1024,
    "UXGA": MAX_PAYLOAD_BYTES,
}

FRAME_SIZE_DIMENSIONS = {
    "96X96": (96, 96),
    "QQVGA": (160, 120),
    "QCIF": (176, 144),
    "HQVGA": (240, 176),
    "240X240": (240, 240),
    "QVGA": (320, 240),
    "CIF": (400, 296),
    "HVGA": (480, 320),
    "VGA": (640, 480),
    "SVGA": (800, 600),
    "XGA": (1024, 768),
    "HD": (1280, 720),
    "SXGA": (1280, 1024),
    "UXGA": (1600, 1200),
}

CAMERA_PROFILE_MACROS = {
    "DEFAULT": "AGL_CAMERA_PROFILE_DEFAULT",
    "TRAFFIC_SIGNAL": "AGL_CAMERA_PROFILE_TRAFFIC_SIGNAL",
}


def _frame_size_key(value: str) -> str:
    key = value.strip().upper()
    if key not in FRAME_SIZE_MACROS:
        choices = ", ".join(sorted(FRAME_SIZE_MACROS))
        raise ValueError(f"unsupported device.capture.frame_size: {value!r}; choose one of: {choices}")
    return key


def _frame_size_macro(value: str) -> str:
    return FRAME_SIZE_MACROS[_frame_size_key(value)]


def _video_packet_capacity(value: str) -> int:
    return FRAME_SIZE_VIDEO_PACKET_CAPACITY[_frame_size_key(value)]


def _frame_size_dimensions(value: str) -> tuple[int, int]:
    return FRAME_SIZE_DIMENSIONS[_frame_size_key(value)]


def _camera_profile_macro(value: str) -> str:
    key = value.strip().upper()
    try:
        return CAMERA_PROFILE_MACROS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(CAMERA_PROFILE_MACROS))
        raise ValueError(
            f"unsupported device.capture.camera_profile: {value!r}; choose one of: {choices}"
        ) from exc


def _server_endpoint(public_base_url: str) -> tuple[str, int]:
    parsed = urlparse(public_base_url)
    if parsed.scheme != "http":
        raise ValueError("server.public_base_url must use http:// because firmware uses plain WebSocket")
    if not parsed.hostname:
        raise ValueError("server.public_base_url must include a hostname")
    return parsed.hostname, parsed.port or 80


def render_header(config_path: str | Path) -> str:
    config = load_config(config_path)
    server_host, server_port = _server_endpoint(config.server.public_base_url)
    device = config.device
    capture = device.capture
    transport = device.transport
    frame_width, frame_height = _frame_size_dimensions(capture.frame_size)

    return f"""#pragma once

#define AGL_DEVICE_ID {_cstr(device.id)}
#define AGL_WIFI_SSID {_cstr(device.wifi.ssid)}
#define AGL_WIFI_PASSWORD {_cstr(device.wifi.password)}
#define AGL_SERVER_HOST {_cstr(server_host)}
#define AGL_SERVER_PORT {int(server_port)}
#define AGL_VIDEO_FPS {int(capture.video_fps)}
#define AGL_VIDEO_PACKET_CAPACITY {_video_packet_capacity(capture.frame_size)}
#define AGL_JPEG_QUALITY {int(capture.jpeg_quality)}
#define AGL_FRAME_SIZE {_frame_size_macro(capture.frame_size)}
#define AGL_FRAME_WIDTH_PIXELS {frame_width}
#define AGL_FRAME_HEIGHT_PIXELS {frame_height}
#define AGL_CAMERA_PROFILE {_camera_profile_macro(capture.camera_profile)}
#define AGL_AUDIO_SAMPLE_RATE {int(capture.audio_sample_rate)}
#define AGL_AUDIO_CHUNK_MS {int(capture.audio_chunk_ms)}
#define AGL_IMU_HZ {int(capture.imu_hz)}
#define AGL_AUDIO_DOWN_ENABLED {1 if device.audio_down.enabled else 0}
#define AGL_VIDEO_TRANSPORT_UDP {1 if transport.video == "udp" else 0}
#define AGL_VIDEO_UDP_CHUNK_BYTES {int(transport.video_payload_bytes)}
#define AGL_VIDEO_AUTH_KEY {_cbytes_from_hex(transport.video_auth_key_hex or "00" * 32, 32)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate firmware/include/generated_config.h")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", default="firmware/include/generated_config.h")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_header(args.config), encoding="utf-8")
    print(f"Generated {output} from {args.config}")


if __name__ == "__main__":
    main()
