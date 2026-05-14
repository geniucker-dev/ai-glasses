# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python backend and ESP32 firmware for AI glasses.

- `src/aiglasses/`: Python package. Key areas include `web/`, `vision/`, `device/`, `navigation/`, `asr/`, and `config/`.
- `tests/`: Python unit tests, named `test_*.py`.
- `firmware/`: PlatformIO project for the Seeed XIAO ESP32S3. Firmware source lives in `firmware/src/`; generated headers live in `firmware/include/`.
- `models/`: Local model files/directories. Avoid committing large/private artifacts.
- `voice/`: Local speech/audio assets.

## Build, Test, and Development Commands

- `uv sync`: Install Python dependencies.
- `uv pip install --torch-backend rocm6.3 torch torchvision ultralytics`: Install the local Torch runtime on this ROCm host. Use `auto` instead of `rocm6.3` on generic machines.
- `uv run python -m aiglasses.vision.export_yoloe_obstacle --source models/yoloe-11l-seg.pt --output models/yoloe-11l-seg-obstacle.pt`: Precompute and save the fixed 29-class YOLOE obstacle model. Runtime config should point to the exported model, not the raw open-vocabulary YOLOE `.pt`.
- `uv run python -m aiglasses.server --config config.toml`: Run the backend and web console.
- `uv run aiglasses-model-benchmark --config config.toml`: Benchmark configured vision models.
- `uv run aiglasses-model-benchmark --config config.toml --warmup-rounds 15 --runs 100`: Measure stable latency.
- `uv run python -m aiglasses.config.firmware_header --config config.toml`: Generate `firmware/include/generated_config.h`.
- `uv run pio run -d firmware`: Build ESP32 firmware.
- `uv run pio run -d firmware -t upload`: Upload ESP32 firmware through a locally connected serial port.
- If local upload cannot access the ESP32 serial port, use PlatformIO Remote instead: first run `uv run pio remote device list`, pick the ESP32 USB/JTAG port (for example `/dev/ttyACM0`), then run `uv run pio remote run -d firmware -t upload --upload-port <port>`.
- `uv run python -m unittest tests.test_config tests.test_web_app`: Run focused config and web/UDP video tests.
- `uv run python -m unittest discover -s tests`: Run unit tests.
- `uv run ruff check .`: Run Python lint checks.

## Coding Style & Naming Conventions

Python targets 3.12 and uses Ruff with a 100-character line length. Use 4-space indentation, type annotations for public functions, and clear module-level separation by responsibility.

JavaScript in `src/aiglasses/web/static/` is plain browser JavaScript. Keep state updates explicit.

Firmware is Arduino C++ under PlatformIO. Use existing `AGL_*` generated macros for configuration instead of hardcoding configurable values.

## Testing Guidelines

Use `unittest` for Python tests. Add tests under `tests/` with filenames like `test_config.py` or `test_benchmark.py`. Prefer focused tests that avoid hardware, GPU, network, and external ASR dependencies. For firmware config changes, test generated header text and run `uv run pio run -d firmware`.

## Runtime Behavior

Backend startup is expected to block until vision models are warmed up and the startup processing-capacity benchmark completes. This is intentional so the web console reports warm-model capacity and the first live device frames do not pay cold-start inference cost.

## Commit & Pull Request Guidelines

Never create a git commit unless the user explicitly asks to commit using clear wording such as “提交”, “commit”, or “create a commit”. Slash-command routing context or ambiguous phrasing is not sufficient permission; when in doubt, ask before committing.

Use Conventional Commits v1.0.0: `<type>[optional scope]: <description>`. Prefer lowercase types such as `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, and `chore`. Examples: `feat(vision): add model benchmark CLI`, `fix(web): preserve ASR status`, `docs: update contributor guide`. Mark breaking changes with `!` or a `BREAKING CHANGE:` footer.

Pull requests should include a concise description, affected areas, verification commands run, and screenshots for visible web UI changes. Mention firmware generation steps when relevant.

## Security & Configuration Tips

Keep `config.toml` local; it may contain WiFi credentials, API keys, and the UDP video auth key. Start from `config.example.toml`, generate a deployment `device.transport.video_auth_key_hex` with `openssl rand -hex 32`, regenerate firmware headers after config changes, and avoid logging secrets in backend or firmware output.

The default video transport is authenticated RTP/JPEG over UDP. Do not expose the UDP video port to untrusted networks. If `device.transport.video_auth_key_hex` changes or leaks, regenerate `firmware/include/generated_config.h` and re-upload firmware so the backend and ESP32 share the same key.

During early-stage development, do not preserve backward compatibility for renamed or removed configuration fields. Update `config.example.toml`, documentation, code, and tests to the current schema instead of adding compatibility aliases or migration logic.
