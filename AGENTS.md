# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python backend and ESP32 firmware for AI glasses.

- `src/aiglasses/`: Python package. Key areas include `web/`, `vision/`, `device/`, `navigation/`, `asr/`, and `config/`.
- `tests/`: Python unit tests, named `test_*.py`.
- `firmware/`: PlatformIO project for the Seeed XIAO ESP32S3. Firmware source lives in `firmware/src/`; generated headers live in `firmware/include/`.
- `models/`: Local NCNN model directories. Avoid committing large/private artifacts.
- `voice/`: Local speech/audio assets.

## Build, Test, and Development Commands

- `uv sync`: Install Python dependencies.
- `uv run python -m aiglasses.server --config config.toml`: Run the backend and web console.
- `uv run aiglasses-model-benchmark --config config.toml`: Benchmark configured NCNN models.
- `uv run aiglasses-model-benchmark --config config.toml --warmup-rounds 15 --runs 100`: Measure stable latency.
- `uv run python -m aiglasses.config.firmware_header --config config.toml`: Generate `firmware/include/generated_config.h`.
- `uv run pio run -d firmware`: Build ESP32 firmware.
- `uv run python -m unittest discover -s tests`: Run unit tests.
- `uv run ruff check .`: Run Python lint checks.

## Coding Style & Naming Conventions

Python targets 3.11+ and uses Ruff with a 100-character line length. Use 4-space indentation, type annotations for public functions, and clear module-level separation by responsibility.

JavaScript in `src/aiglasses/web/static/` is plain browser JavaScript. Keep state updates explicit.

Firmware is Arduino C++ under PlatformIO. Use existing `AGL_*` generated macros for configuration instead of hardcoding configurable values.

## Testing Guidelines

Use `unittest` for Python tests. Add tests under `tests/` with filenames like `test_config.py` or `test_benchmark.py`. Prefer focused tests that avoid hardware, GPU, network, and external ASR dependencies. For firmware config changes, test generated header text and run `uv run pio run -d firmware`.

## Commit & Pull Request Guidelines

Use Conventional Commits v1.0.0: `<type>[optional scope]: <description>`. Prefer lowercase types such as `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, and `chore`. Examples: `feat(vision): add model benchmark CLI`, `fix(web): preserve ASR status`, `docs: update contributor guide`. Mark breaking changes with `!` or a `BREAKING CHANGE:` footer.

Pull requests should include a concise description, affected areas, verification commands run, and screenshots for visible web UI changes. Mention firmware generation steps when relevant.

## Security & Configuration Tips

Keep `config.toml` local; it may contain WiFi credentials and API keys. Start from `config.example.toml`, regenerate firmware headers after config changes, and avoid logging secrets in backend or firmware output.
