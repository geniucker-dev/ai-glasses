# AI Glasses

重写版智能导航眼镜项目。新代码不沿用旧项目架构或协议，只保留已焊接硬件引脚和顶层能力。

## 功能

- Seeed XIAO ESP32S3 Sense 固件，使用 PlatformIO。
- Python 后端使用 `uv` 管理。
- ESP32 通过 WebSocket 上传 JPEG 视频、PCM16 音频、ICM42688 IMU。
- 后端使用 PyTorch/Ultralytics 模型做盲道/斑马线、障碍物、红绿灯推理。
- Web 调试台显示实时画面、检测状态、IMU、ASR/指令和“应播报内容”。
- TTS 支持 UI 文字播报和设备音频下行，可选择 DashScope API 或本地 TTS。

## 配置

复制示例配置，填入本机和设备信息：

```bash
cp config.example.toml config.toml
```

`config.toml` 不应提交。它包含：

- 后端监听地址和端口
- 后端给板子连接的 `public_base_url`
- WiFi SSID/密码
- 采集参数：
  - `device.capture.frame_size`：摄像头分辨率，会写入固件生成头文件，例如 `VGA`、`QVGA`
  - `device.capture.video_fps`、`jpeg_quality`：视频上传帧率和 JPEG 质量
- DashScope ASR 配置：
  - `asr.dashscope_api_key`：DashScope API Key
  - `asr.websocket_base_url`：实时 ASR WebSocket 地址
  - `asr.http_base_url`：DashScope HTTP API 地址
- Torch 模型路径、输入尺寸和推理设备
- TTS/音频下行开关：
  - `speech.mode = "ui"`：只在 Web 调试台显示应播报内容
  - `speech.mode = "device"`：同时把 TTS PCM16 下发到设备，需开启 `device.audio_down.enabled`
  - `speech.provider = "dashscope"`：使用 DashScope API
  - `speech.provider = "local"`：使用 Piper 本地 TTS，支持中英文 voice 自动切换

## TTS 与设备播报

后端始终会把导航或指令产生的播报文字发到 Web 调试台。是否让眼镜设备播放声音由
`speech.enabled`、`speech.mode` 和 `device.audio_down.enabled` 决定：

```toml
[device.audio_down]
enabled = true

[speech]
enabled = true
mode = "device"
```

`speech.mode` 只有两个有效值：

- `ui`：只在 Web 调试台显示播报文字，不合成音频，也不下发到设备。
- `device`：合成 TTS，并通过控制 WebSocket 把 mono PCM16 音频下发给 ESP32 播放。

设备播放需要重新生成固件配置并上传，因为 `device.audio_down.enabled` 会写入
`firmware/include/generated_config.h`：

```bash
uv run python -m aiglasses.config.firmware_header --config config.toml
uv run pio run -d firmware
uv run pio remote run -d firmware -e seeed_xiao_esp32s3 -t upload
```

### DashScope TTS

使用 DashScope API 时：

```toml
[asr]
dashscope_api_key = "your-dashscope-api-key"

[speech]
enabled = true
mode = "device"
provider = "dashscope"
model = "sambert-zhichu-v1"
```

DashScope TTS 输出采样率会跟随 `device.capture.audio_sample_rate`，保持和 ESP32 I2S
播放采样率一致。`speech.sample_rate` 保留在配置里，但设备播放路径不会用它决定下发采样率。

### Piper 本地 TTS

使用 Piper 本地 TTS 时：

```toml
[speech]
enabled = true
mode = "device"
provider = "local"
language = "auto"
piper_model_dir = "voice"
piper_voice_zh = "zh_CN-huayan-medium"
piper_voice_en = "en_US-lessac-medium"
piper_use_cuda = false
```

本地实现使用 `piper-tts` Python 包加载 voice 模型生成 WAV，再由后端转成设备需要的
mono PCM16。`language = "auto"` 会按文本里的中英文字符分段，中文段使用 `piper_voice_zh`，
英文段使用 `piper_voice_en`。如果只想固定一种语言，可设为 `zh-CN` 或 `en-US`。

首次使用前需要安装 Python 依赖并下载 voice 模型：

```bash
uv sync
uv run python -m piper.download_voices --download-dir voice zh_CN-huayan-medium
uv run python -m piper.download_voices --download-dir voice en_US-lessac-medium
```

下载后 `voice/` 目录里每个 voice 应该各有一个 `.onnx` 和一个 `.onnx.json` 文件。
`piper_voice_zh`、`piper_voice_en` 可以写 voice 名称，也可以写 `.onnx` 模型文件路径。

## 运行后端

```bash
uv sync
uv pip install --torch-backend auto torch torchvision ultralytics
uv run python -m aiglasses.server --config config.toml
```

AMD ROCm 环境可显式指定后端，例如本机 MI50/MI60：

```bash
uv sync
uv pip install --torch-backend rocm6.3 torch torchvision ultralytics
```

Torch、TorchVision 和 Ultralytics 不写入 `pyproject.toml`，避免 `uv sync`/lockfile 固定错误的 CUDA 或 ROCm wheel。每次重新 `uv sync` 后，如果 runtime 包被清掉，需要重新执行上面的 `uv pip install --torch-backend ...`。确认 ROCm 可用：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

ROCm 在 PyTorch 里也通过 `torch.cuda` 接口暴露。

默认 Web 控制台：

```text
http://127.0.0.1:8081
```

## 固件

构建：

```bash
uv run python -m aiglasses.config.firmware_header --config config.toml
uv run pio run -d firmware
```

通过 PlatformIO Remote 上传：

```bash
uv run python -m aiglasses.config.firmware_header --config config.toml
uv run pio remote run -d firmware -e seeed_xiao_esp32s3 -t upload
```

远程串口监视：

```bash
uv run pio remote device monitor -b 115200
```

上传/构建前先运行 `aiglasses.config.firmware_header`，它会从 `config.toml` 生成 ignored 的 `firmware/include/generated_config.h`。

如果修改了 WiFi、服务器地址或采集参数，必须重新生成 `generated_config.h` 后再编译/上传固件。

## 模型

视觉推理只使用 Torch/Ultralytics 模型。盲道/斑马线和红绿灯直接使用 `.pt`，
障碍物使用离线固化过类别提示词的 YOLOE `.pt`：

```text
models/yolo-seg.pt
models/yoloe-11l-seg-obstacle.pt
models/trafficlight.pt
```

原始 YOLOE 模型 `models/yoloe-11l-seg.pt` 是开放词表模型，不能直接当作障碍物模型运行。
先用固定的 29 个障碍物类别导出固化模型：

```bash
uv run python -m aiglasses.vision.export_yoloe_obstacle \
  --source models/yoloe-11l-seg.pt \
  --output models/yoloe-11l-seg-obstacle.pt
```

这一步会生成文本 embedding 并写入输出 `.pt`。后端运行 `models/yoloe-11l-seg-obstacle.pt`
时不需要 CLIP/MobileCLIP encoder，也不会再下载 encoder 权重。

`config.toml` 的 `[models]` 决定模型路径、输入尺寸和 Torch 设备：

```toml
[models]
blind_path = "models/yolo-seg.pt"
obstacle = "models/yoloe-11l-seg-obstacle.pt"
traffic_light = "models/trafficlight.pt"
image_width = 640
image_height = 480
torch_device = "cuda:0"
torch_half = true
```

`torch_device = "cuda:0"` 会使用 PyTorch CUDA/ROCm 设备，`torch_half = true` 会使用 FP16 推理。没有可用 GPU 时可临时设为 `torch_device = "cpu"`、`torch_half = false`，但速度会明显下降。ROCm 设备在 PyTorch 中仍通过 `cuda:0`、`cuda:1` 这类名称选择。

### 模型 Benchmark

Benchmark 命令会：

1. 读取 `config.toml` 里的三类模型配置。
2. 加载模型并先跑一次 initial load，这部分不计入稳定耗时。
3. 按串行顺序做 warmup。
4. 分别统计单模型耗时。
5. 如果选择了多个模型，再统计串行跑所有模型的总耗时和等效 FPS。

默认跑全部模型：

```bash
uv run aiglasses-model-benchmark --config config.toml
```

推荐稳定态测试：

```bash
uv run aiglasses-model-benchmark --config config.toml --warmup-rounds 15 --runs 100
```

只测红绿灯模型：

```bash
uv run aiglasses-model-benchmark --config config.toml --model traffic_light
```

使用真实图片而不是固定随机帧：

```bash
uv run aiglasses-model-benchmark --config config.toml --image test-frame.jpg
```

临时切换 Torch 设备或精度：

```bash
uv run aiglasses-model-benchmark --config config.toml --torch-device cuda:0 --torch-half
uv run aiglasses-model-benchmark --config config.toml --torch-device cpu --no-torch-half
```

输出字段示例：

```text
runtime=torch
torch_device=cuda:0
torch_half=True
initial_load traffic_light: 824.81ms detections=0 masks=0
traffic_light: min=12.27ms p50=12.37ms mean=12.40ms p90=12.49ms p95=12.71ms max=12.75ms
all_serial: min=46.47ms p50=46.67ms mean=46.75ms p90=47.07ms p95=47.36ms max=47.43ms
effective_serial_fps_p50=21.43
```

看稳定性能时优先看 `p50`、`p90`、`p95`。`initial_load` 包含模型加载和运行时初始化，不代表持续推理耗时。`all_serial` 接近后端当前串行推理一帧的耗时。

## 测试

无需设备的单元测试：

```bash
uv run python -m unittest discover -s tests
```

Lint：

```bash
uv run ruff check .
```

固件编译检查：

```bash
uv run pio run -d firmware
```

完整链路测试需要：

- 已填写 `config.toml`
- 后端可被 ESP32 访问
- 模型目录已放入 `models/`
- PlatformIO Remote Agent 可看到目标板子

## 贡献

贡献指南见 `AGENTS.md`。提交消息使用 Conventional Commits v1.0.0：

```text
<type>[optional scope]: <description>
```

示例：

```text
feat(vision): add model benchmark CLI
fix(web): preserve ASR status
docs: update contributor guide
```
