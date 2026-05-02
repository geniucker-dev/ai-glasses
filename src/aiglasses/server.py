from __future__ import annotations

import argparse

import uvicorn

from aiglasses.config import load_config
from aiglasses.web import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AI Glasses backend.")
    parser.add_argument("--config", default="config.toml", help="Path to TOML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        ws_ping_interval=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
