import tempfile
import unittest
from pathlib import Path

from aiglasses.config import load_config
from aiglasses.config.firmware_header import render_header


class ConfigTests(unittest.TestCase):
    def test_example_loads(self) -> None:
        config = load_config("config.example.toml")
        self.assertEqual(config.server.port, 8081)
        self.assertEqual(config.device.capture.audio_sample_rate, 16000)
        self.assertEqual(config.models.image_width, 640)

    def test_firmware_header_includes_frame_size(self) -> None:
        header = render_header("config.example.toml")

        self.assertIn("#define AGL_FRAME_SIZE FRAMESIZE_VGA", header)

    def test_firmware_header_maps_configured_frame_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[device.capture]\nframe_size = "QVGA"\n', encoding="utf-8")

            header = render_header(config_path)

        self.assertIn("#define AGL_FRAME_SIZE FRAMESIZE_QVGA", header)

    def test_missing_config_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "config.toml"
            with self.assertRaises(FileNotFoundError) as ctx:
                load_config(missing)
            self.assertIn("Copy config.example.toml", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
