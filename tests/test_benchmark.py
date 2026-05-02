from __future__ import annotations

import unittest

from aiglasses.benchmark import build_parser, format_stats, summarize_samples


class BenchmarkTests(unittest.TestCase):
    def test_summarize_samples(self) -> None:
        stats = summarize_samples([4.0, 1.0, 3.0, 2.0, 5.0])

        self.assertEqual(stats.minimum_ms, 1.0)
        self.assertEqual(stats.p50_ms, 3.0)
        self.assertEqual(stats.mean_ms, 3.0)
        self.assertEqual(stats.p90_ms, 5.0)
        self.assertEqual(stats.p95_ms, 5.0)
        self.assertEqual(stats.maximum_ms, 5.0)

    def test_format_stats(self) -> None:
        stats = summarize_samples([1.0, 2.0, 3.0])

        self.assertEqual(
            format_stats(stats),
            "min=1.00ms p50=2.00ms mean=2.00ms p90=3.00ms p95=3.00ms max=3.00ms",
        )

    def test_parser_accepts_device_index(self) -> None:
        args = build_parser().parse_args(["--device", "vulkan", "--device-index", "1"])

        self.assertEqual(args.device, "vulkan")
        self.assertEqual(args.device_index, 1)


if __name__ == "__main__":
    unittest.main()
