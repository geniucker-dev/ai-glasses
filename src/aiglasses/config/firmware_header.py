from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from .settings import load_config


def _cstr(value: object) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


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

CAMERA_PROFILE_MACROS = {
    "DEFAULT": "AGL_CAMERA_PROFILE_DEFAULT",
    "TRAFFIC_SIGNAL": "AGL_CAMERA_PROFILE_TRAFFIC_SIGNAL",
}


def _frame_size_macro(value: str) -> str:
    key = value.strip().upper()
    try:
        return FRAME_SIZE_MACROS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(FRAME_SIZE_MACROS))
        raise ValueError(f"unsupported device.capture.frame_size: {value!r}; choose one of: {choices}") from exc


def _camera_profile_macro(value: str) -> str:
    key = value.strip().upper()
    try:
        return CAMERA_PROFILE_MACROS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(CAMERA_PROFILE_MACROS))
        raise ValueError(
            f"unsupported device.capture.camera_profile: {value!r}; choose one of: {choices}"
        ) from exc


def render_header(config_path: str | Path) -> str:
    config = load_config(config_path)
    parsed = urlparse(config.server.public_base_url)
    server_host = parsed.hostname or "127.0.0.1"
    server_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    device = config.device
    capture = device.capture

    return f"""#pragma once

#define AGL_DEVICE_ID {_cstr(device.id)}
#define AGL_WIFI_SSID {_cstr(device.wifi.ssid)}
#define AGL_WIFI_PASSWORD {_cstr(device.wifi.password)}
#define AGL_SERVER_HOST {_cstr(server_host)}
#define AGL_SERVER_PORT {int(server_port)}
#define AGL_VIDEO_FPS {int(capture.video_fps)}
#define AGL_JPEG_QUALITY {int(capture.jpeg_quality)}
#define AGL_FRAME_SIZE {_frame_size_macro(capture.frame_size)}
#define AGL_CAMERA_PROFILE {_camera_profile_macro(capture.camera_profile)}
#define AGL_AUDIO_SAMPLE_RATE {int(capture.audio_sample_rate)}
#define AGL_AUDIO_CHUNK_MS {int(capture.audio_chunk_ms)}
#define AGL_IMU_HZ {int(capture.imu_hz)}
#define AGL_AUDIO_DOWN_ENABLED {1 if device.audio_down.enabled else 0}
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
