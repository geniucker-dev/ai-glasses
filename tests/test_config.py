import tempfile
import unittest
from pathlib import Path

from aiglasses.config import load_config
from aiglasses.config.firmware_header import render_header


class ConfigTests(unittest.TestCase):
    def test_example_loads(self) -> None:
        config = load_config("config.example.toml")
        self.assertEqual(config.server.port, 8081)
        self.assertEqual(config.device.capture.camera_profile, "traffic_signal")
        self.assertEqual(config.device.capture.audio_sample_rate, 16000)
        self.assertEqual(config.models.image_width, 800)
        self.assertEqual(config.models.image_height, 608)
        self.assertEqual(config.models.torch_device, "cuda:0")
        self.assertTrue(config.models.torch_half)
        self.assertEqual(config.speech.model, "sambert-zhichu-v1")
        self.assertEqual(config.speech.audio_format, "pcm")
        self.assertEqual(config.speech.sample_rate, 16000)
        self.assertEqual(config.speech.language, "auto")
        self.assertEqual(config.speech.piper_model_dir, "voice")
        self.assertEqual(config.speech.piper_voice_zh, "zh_CN-huayan-medium")
        self.assertEqual(config.speech.piper_voice_en, "en_US-lessac-medium")
        self.assertFalse(config.speech.piper_use_cuda)

    def test_models_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[models]\nruntime = "torch"\ntorch_device = "cpu"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"Unknown config field in \[models\]: runtime"):
                load_config(config_path)

    def test_server_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[server]\nunknown = "value"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"Unknown config field in \[server\]: unknown"):
                load_config(config_path)

    def test_server_rejects_non_table_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('server = "bad"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"Config section \[server\] must be a table"):
                load_config(config_path)

    def test_nested_section_rejects_non_table_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[device]\ncapture = "bad"\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"Config section \[capture\] must be a table",
            ):
                load_config(config_path)

    def test_firmware_header_includes_frame_size(self) -> None:
        header = render_header("config.example.toml")

        self.assertIn("#define AGL_FRAME_SIZE FRAMESIZE_SVGA", header)
        self.assertIn("#define AGL_VIDEO_PACKET_CAPACITY 393216", header)
        self.assertIn("#define AGL_CAMERA_PROFILE AGL_CAMERA_PROFILE_TRAFFIC_SIGNAL", header)
        self.assertIn("#define AGL_FRAME_WIDTH_PIXELS 800", header)
        self.assertIn("#define AGL_FRAME_HEIGHT_PIXELS 600", header)
        self.assertIn("#define AGL_VIDEO_TRANSPORT_UDP 1", header)
        self.assertIn("#define AGL_VIDEO_UDP_CHUNK_BYTES 1200", header)
        self.assertIn("#define AGL_VIDEO_AUTH_KEY {0x01, 0x23, 0x45", header)

    def test_firmware_header_maps_configured_frame_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[device.capture]\nframe_size = "QVGA"\n', encoding="utf-8")

            header = render_header(config_path)

        self.assertIn("#define AGL_FRAME_SIZE FRAMESIZE_QVGA", header)

    def test_firmware_header_disables_udp_video_macro_for_ws_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[device.transport]\nvideo = "ws"\n', encoding="utf-8")

            header = render_header(config_path)

        self.assertIn("#define AGL_VIDEO_TRANSPORT_UDP 0", header)

    def test_firmware_header_caps_large_video_packet_capacity_to_backend_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[device.capture]\nframe_size = "UXGA"\n', encoding="utf-8")

            header = render_header(config_path)

        self.assertIn("#define AGL_FRAME_SIZE FRAMESIZE_UXGA", header)
        self.assertIn("#define AGL_VIDEO_PACKET_CAPACITY 1048576", header)

    def test_firmware_header_maps_configured_camera_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[device.capture]\ncamera_profile = "default"\n',
                encoding="utf-8",
            )

            header = render_header(config_path)

        self.assertIn("#define AGL_CAMERA_PROFILE AGL_CAMERA_PROFILE_DEFAULT", header)

    def test_missing_config_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "config.toml"
            with self.assertRaises(FileNotFoundError) as ctx:
                load_config(missing)
            self.assertIn("Copy config.example.toml", str(ctx.exception))

    def test_top_level_unknown_section_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[spech]\nenabled = true\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"Unknown top-level config section: spech"):
                load_config(config_path)

    def test_asr_sample_rate_must_match_device_audio_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[device.capture]\naudio_sample_rate = 24000\n[asr]\nsample_rate = 16000\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"asr\.sample_rate must equal device\.capture\.audio_sample_rate",
            ):
                load_config(config_path)

    def test_asr_sample_rate_mismatch_is_allowed_when_asr_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[device.capture]\naudio_sample_rate = 24000\n[asr]\nenabled = false\nsample_rate = 16000\n',
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertFalse(config.asr.enabled)
        self.assertEqual(config.device.capture.audio_sample_rate, 24000)
        self.assertEqual(config.asr.sample_rate, 16000)

    def test_invalid_firmware_numeric_config_is_rejected(self) -> None:
        cases = {
            "video_fps": 0,
            "jpeg_quality": 64,
            "audio_sample_rate": 0,
            "audio_chunk_ms": 1001,
            "imu_hz": 0,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    config_path = Path(tmp) / "config.toml"
                    config_path.write_text(
                        f'[device.capture]\n{field} = {value}\n[asr]\nsample_rate = 16000\n',
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, rf"device\.capture\.{field}"):
                        load_config(config_path)

    def test_invalid_transport_config_is_rejected(self) -> None:
        cases = {
            "video_payload_bytes": "200",
            "video_payload_bytes_large": "1401",
            "video_auth_key_hex_empty_for_udp": '""',
            "video_auth_key_hex_short": '"abcd"',
            "video_auth_key_hex_non_hex": '"' + ("0" * 63) + 'z"',
            "video": '"tcp"',
            "control": '"udp"',
            "audio_up": '"udp"',
            "audio_down": '"udp"',
        }
        for field, value in cases.items():
            if field.startswith("video_payload_bytes"):
                field_name = "video_payload_bytes"
            elif field.startswith("video_auth_key_hex"):
                field_name = "video_auth_key_hex"
            else:
                field_name = field
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    config_path = Path(tmp) / "config.toml"
                    key_line = ""
                    if not field.startswith("video_auth_key_hex"):
                        key_line = f'video_auth_key_hex = "{"0" * 64}"\n'
                    video_line = ""
                    if field != "video":
                        video_line = 'video = "udp"\n'
                    config_path.write_text(
                        "[device.transport]\n"
                        f"{video_line}"
                        f"{key_line}"
                        f"{field_name} = {value}\n",
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, r"device\.transport"):
                        load_config(config_path)

    def test_invalid_video_auth_key_is_rejected_for_ws_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[device.transport]\nvideo = "ws"\nvideo_auth_key_hex = "zz"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"video_auth_key_hex"):
                load_config(config_path)

    def test_device_id_is_required_and_length_limited(self) -> None:
        cases = {
            "empty": "",
            "too_long": "x" * 129,
        }
        for name, device_id in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    config_path = Path(tmp) / "config.toml"
                    config_path.write_text(f'[device]\nid = "{device_id}"\n', encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, r"device\.id"):
                        load_config(config_path)

    def test_audio_chunk_larger_than_backend_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                "[device.capture]\n"
                "audio_sample_rate = 48000\n"
                "audio_chunk_ms = 1000\n"
                "[asr]\n"
                "sample_rate = 48000\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"exceeding 65536"):
                load_config(config_path)

    def test_invalid_model_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[models]\nimage_width = 0\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"models\.image_width"):
                load_config(config_path)

    def test_firmware_header_rejects_https_public_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[server]\npublic_base_url = "https://example.com:8443"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"server\.public_base_url must use http://"):
                render_header(config_path)

    def test_firmware_header_rejects_missing_public_url_hostname(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[server]\npublic_base_url = "http:///missing"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"server\.public_base_url must include a hostname"):
                render_header(config_path)

    def test_firmware_header_accepts_http_public_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                '[server]\npublic_base_url = "http://example.com:8081"\n',
                encoding="utf-8",
            )

            header = render_header(config_path)

        self.assertIn('#define AGL_SERVER_HOST "example.com"', header)
        self.assertIn("#define AGL_SERVER_PORT 8081", header)


if __name__ == "__main__":
    unittest.main()
