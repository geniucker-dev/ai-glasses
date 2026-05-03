from __future__ import annotations

import argparse
from pathlib import Path

from .obstacle_classes import YOLOE_OBSTACLE_CLASS_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the fixed-class YOLOE obstacle model.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/yoloe-11l-seg.pt"),
        help="Source YOLOE .pt model with text-prompt support.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/yoloe-11l-seg-obstacle.pt"),
        help="Output .pt model with the obstacle class prompts embedded.",
    )
    return parser


def export_obstacle_model(source: Path, output: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"source model does not exist: {source}")
    if source.suffix != ".pt" or output.suffix != ".pt":
        raise ValueError("source and output must both be .pt files")

    from ultralytics import YOLOE

    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLOE(str(source))
    names = list(YOLOE_OBSTACLE_CLASS_NAMES)
    current = list((getattr(model.model, "names", {}) or {}).values())
    if current != names:
        text_pe = model.get_text_pe(names)
        model.set_classes(names, text_pe)
    model.save(str(output))


def main() -> None:
    args = build_parser().parse_args()
    export_obstacle_model(args.source, args.output)
    print(f"exported fixed-class obstacle model: {args.output}")


if __name__ == "__main__":
    main()
