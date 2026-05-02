# AI Glasses

重写版智能导航眼镜项目。新代码不沿用旧项目架构或协议，只保留已焊接硬件引脚和顶层能力。

## 功能

- Seeed XIAO ESP32S3 Sense 固件，使用 PlatformIO。
- Python 后端使用 `uv` 管理。
- ESP32 通过 WebSocket 上传 JPEG 视频、PCM16 音频、ICM42688 IMU。
- 后端使用 NCNN 模型目录做盲道/斑马线、障碍物、红绿灯推理。
- Web 调试台显示实时画面、检测状态、IMU、ASR/指令和“应播报内容”。
- TTS 与设备音频下行已预留接口，默认不播放、不下发。

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
- NCNN 模型路径
- TTS/音频下行开关

## 运行后端

```bash
uv sync
uv run python -m aiglasses.server --config config.toml
```

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

运行时只接受 NCNN 模型目录，不加载 `.pt`：

```text
models/yolo-seg_ncnn_model
models/yoloe-11l-seg_ncnn_model
models/trafficlight_ncnn_model
```

每个目录需要包含：

```text
model.ncnn.param
model.ncnn.bin
metadata.yaml
```

`config.toml` 的 `[models]` 决定运行时使用的模型目录、输入尺寸和 NCNN 设备：

```toml
[models]
blind_path = "models/yolo-seg_ncnn_model"
obstacle = "models/yoloe-11l-seg_ncnn_model"
traffic_light = "models/trafficlight_ncnn_model"
image_width = 640
image_height = 480
ncnn_device = "vulkan"
```

`ncnn_device = "vulkan"` 会启用 NCNN Vulkan 推理；当前 NCNN 默认启用 `fp16_storage`、`fp16_packed` 和 `fp16_arithmetic`。如果要对比 CPU/GPU，可用 benchmark 的 `--device` 临时覆盖。

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

临时切换 NCNN 设备：

```bash
uv run aiglasses-model-benchmark --config config.toml --device vulkan
uv run aiglasses-model-benchmark --config config.toml --device cpu
```

输出字段示例：

```text
initial_load traffic_light: 799.74ms detections=0 masks=0
warmup_serial: min=25.45ms p50=25.45ms mean=25.45ms p90=25.45ms p95=25.45ms max=25.45ms
traffic_light: min=24.00ms p50=24.00ms mean=24.00ms p90=24.00ms p95=24.00ms max=24.00ms
all_serial: min=90.07ms p50=91.41ms mean=91.54ms p90=92.19ms p95=92.76ms max=94.82ms
effective_serial_fps_p50=10.94
```

看稳定性能时优先看 `p50`、`p90`、`p95`。`initial_load` 包含模型加载和 Vulkan 初始化，不代表持续推理耗时。`all_serial` 接近后端当前串行推理一帧的耗时。

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
