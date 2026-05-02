from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from aiglasses.config import load_config
from aiglasses.config.settings import AppConfig
from aiglasses.vision.ncnn_yolo import NcnnYoloModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    kind: str
    confidence: float


@dataclass(frozen=True)
class TimingStats:
    minimum_ms: float
    p50_ms: float
    mean_ms: float
    p90_ms: float
    p95_ms: float
    maximum_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark configured NCNN models.")
    parser.add_argument("--config", default="config.toml", help="Path to TOML config.")
    parser.add_argument("--runs", type=int, default=100, help="Measured runs per benchmark.")
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=15,
        help="Serial warmup rounds after the initial model load.",
    )
    parser.add_argument("--seed", type=int, default=20260503, help="Random frame seed.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional image file to benchmark instead of a deterministic random frame.",
    )
    parser.add_argument(
        "--device",
        help="Override models.ncnn_device for this benchmark, for example vulkan or cpu.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=("blind_path", "obstacle", "traffic_light"),
        help="Model to include. Repeat to include multiple models. Defaults to all three.",
    )
    return parser


def model_specs(config: AppConfig) -> list[ModelSpec]:
    thresholds = config.vision_thresholds
    return [
        ModelSpec("blind_path", config.models.blind_path, "segment", thresholds.blind_path_conf),
        ModelSpec("obstacle", config.models.obstacle, "segment", thresholds.obstacle_conf),
        ModelSpec(
            "traffic_light",
            config.models.traffic_light,
            "detect",
            thresholds.traffic_light_conf,
        ),
    ]


def summarize_samples(samples: list[float]) -> TimingStats:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(samples)

    def percentile(percent: int) -> float:
        index = round((len(ordered) - 1) * percent / 100)
        return ordered[index]

    return TimingStats(
        minimum_ms=min(samples),
        p50_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        p90_ms=percentile(90),
        p95_ms=percentile(95),
        maximum_ms=max(samples),
    )


def format_stats(stats: TimingStats) -> str:
    return (
        f"min={stats.minimum_ms:.2f}ms "
        f"p50={stats.p50_ms:.2f}ms "
        f"mean={stats.mean_ms:.2f}ms "
        f"p90={stats.p90_ms:.2f}ms "
        f"p95={stats.p95_ms:.2f}ms "
        f"max={stats.maximum_ms:.2f}ms"
    )


def load_frame(config: AppConfig, image_path: Path | None, seed: int) -> np.ndarray:
    width = config.models.image_width
    height = config.models.image_height
    if image_path:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"failed to read image: {image_path}")
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def build_models(
    config: AppConfig,
    selected_names: set[str],
    device_override: str | None,
) -> list[tuple[ModelSpec, NcnnYoloModel]]:
    image_size = (config.models.image_width, config.models.image_height)
    ncnn_device = device_override or config.models.ncnn_device
    specs = [spec for spec in model_specs(config) if spec.name in selected_names]
    return [
        (
            spec,
            NcnnYoloModel(
                spec.path,
                image_size=image_size,
                confidence=spec.confidence,
                kind=spec.kind,
                ncnn_device=ncnn_device,
                min_mask_area=config.vision_thresholds.mask_min_area,
            ),
        )
        for spec in specs
    ]


def time_call(fn) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1000.0, value


def benchmark(config: AppConfig, args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup_rounds < 0:
        raise ValueError("--warmup-rounds cannot be negative")

    selected = set(args.model or ("blind_path", "obstacle", "traffic_light"))
    models = build_models(config, selected, args.device)
    frame = load_frame(config, args.image, args.seed)
    ncnn_device = args.device or config.models.ncnn_device

    print(f"ncnn_device={ncnn_device}")
    print(f"image={config.models.image_width}x{config.models.image_height}")
    print(f"runs={args.runs} warmup_rounds={args.warmup_rounds}")
    print("loading models...")

    for spec, model in models:
        elapsed_ms, output = time_call(lambda model=model: model.predict(frame))
        print(
            f"initial_load {spec.name}: {elapsed_ms:.2f}ms "
            f"detections={len(output.detections)} masks={len(output.masks)}"
        )

    if args.warmup_rounds:
        print("warmup serial...")
        warmup_samples: list[float] = []
        for _ in range(args.warmup_rounds):
            start = time.perf_counter()
            for _, model in models:
                model.predict(frame)
            warmup_samples.append((time.perf_counter() - start) * 1000.0)
        print(f"warmup_serial: {format_stats(summarize_samples(warmup_samples))}")

    print("individual:")
    for spec, model in models:
        samples = [time_call(lambda model=model: model.predict(frame))[0] for _ in range(args.runs)]
        print(f"{spec.name}: {format_stats(summarize_samples(samples))}")

    if len(models) > 1:
        serial_samples: list[float] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            for _, model in models:
                model.predict(frame)
            serial_samples.append((time.perf_counter() - start) * 1000.0)
        stats = summarize_samples(serial_samples)
        print("serial:")
        print(f"all_serial: {format_stats(stats)}")
        print(f"effective_serial_fps_p50={1000.0 / stats.p50_ms:.2f}")


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    benchmark(config, args)


if __name__ == "__main__":
    main()
