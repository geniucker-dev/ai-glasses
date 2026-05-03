import unittest
from pathlib import Path

from aiglasses.config import AppConfig, SpeechConfig
from aiglasses.config.settings import DeviceAudioDownConfig, DeviceConfig
from aiglasses.web.app import validate_speech_config


class WebAppTests(unittest.TestCase):
    def test_web_app_imports(self) -> None:
        from aiglasses.web.app import create_app

        self.assertTrue(callable(create_app))

    def test_device_speech_requires_audio_down(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=False)),
        )

        with self.assertRaisesRegex(ValueError, "audio_down"):
            validate_speech_config(config)

    def test_speech_mode_rejects_both(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="both"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        with self.assertRaisesRegex(ValueError, "speech.mode"):
            validate_speech_config(config)

    def test_device_speech_allows_audio_down(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        validate_speech_config(config)

    def test_device_speech_allows_local_provider(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device", provider="local"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        validate_speech_config(config)

    def test_device_speech_rejects_unknown_provider(self) -> None:
        config = AppConfig(
            path=Path("config.toml"),
            speech=SpeechConfig(enabled=True, mode="device", provider="unknown"),
            device=DeviceConfig(audio_down=DeviceAudioDownConfig(enabled=True)),
        )

        with self.assertRaisesRegex(ValueError, "speech.provider"):
            validate_speech_config(config)


if __name__ == "__main__":
    unittest.main()
