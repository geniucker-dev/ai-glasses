import unittest

from aiglasses.asr import AsrService
from aiglasses.config import AsrConfig


class AsrTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_does_not_start_external_stream(self) -> None:
        seen: list[str] = []

        async def on_text(text: str) -> None:
            seen.append(text)

        service = AsrService(
            AsrConfig(enabled=True, dashscope_api_key="replace-with-your-key"),
            on_text,
        )
        await service.start()
        self.assertEqual(service.status, "missing_dashscope_api_key")
        self.assertIsNone(service._task)
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
