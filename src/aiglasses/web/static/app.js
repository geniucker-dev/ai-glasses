const stateEl = document.querySelector("#state");
const logEl = document.querySelector("#log");
const frameEl = document.querySelector("#frame");
const emptyEl = document.querySelector("#empty");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const modeEl = document.querySelector("#mode");
const fpsEl = document.querySelector("#fps");
const backendFpsEl = document.querySelector("#backendFps");
const maxBackendFpsEl = document.querySelector("#maxBackendFps");
const uptimeEl = document.querySelector("#uptime");
const lightEl = document.querySelector("#light");
const imuBriefEl = document.querySelector("#imuBrief");
const targetFpsEl = document.querySelector("#targetFps");
const jpegQualityEl = document.querySelector("#jpegQuality");
const cameraProfileEl = document.querySelector("#cameraProfile");
const aeLevelEl = document.querySelector("#aeLevel");
const saturationEl = document.querySelector("#saturation");
const contrastEl = document.querySelector("#contrast");
const sharpnessEl = document.querySelector("#sharpness");
const gainceilingEl = document.querySelector("#gainceiling");
const tuningStatusEl = document.querySelector("#tuningStatus");
const trafficDebugEl = document.querySelector("#trafficDebug");
const webSpeechToggleButton = document.querySelector("#webSpeechToggle");
const webSpeechStatusEl = document.querySelector("#webSpeechStatus");
const recordingToggleButton = document.querySelector("#recordingToggle");
const recordingStatusEl = document.querySelector("#recordingStatus");
const allViewButton = document.querySelector("#allViewButton");
const trafficViewButton = document.querySelector("#trafficViewButton");
const configStatusEl = document.querySelector("#configStatus");
const disconnectDeviceButton = document.querySelector("#disconnectDevice");
const chips = {
  control: document.querySelector("#control"),
  video: document.querySelector("#video"),
  audio: document.querySelector("#audio"),
  asr: document.querySelector("#asr"),
};

let latestState = {};
let latestObservation = null;
let frameCount = 0;
let asrStatus = "unknown";
let audioConnected = false;
let pendingFrameCount = 0;
let queuedFrameCount = 0;
let queuedFrameBlob = null;
let frameLoading = false;
let frameObjectUrl = "";
let lastWsFrameAt = 0;
let trafficOnlyView = false;
let recordingActive = false;
let webSpeechEnabled = window.localStorage.getItem("aiglasses.webSpeechEnabled") !== "false";
const displayedFrameTimes = [];
const displayFpsWindowMs = 3000;
const maxTargetFps = 1000;
const packetHeaderBytes = 32;
const packetTypeVideoJpeg = 2;
const defaultVisionFrame = { width: 640, height: 480 };
const tuningFields = {
  traffic_filter_enabled: { element: document.querySelector("#trafficFilterEnabled"), type: "boolean" },
  traffic_light_conf: { element: document.querySelector("#trafficLightConf"), type: "number" },
  traffic_signal_clear_margin: { element: document.querySelector("#trafficSignalClearMargin"), type: "number" },
  traffic_go_min_conf: { element: document.querySelector("#trafficGoMinConf"), type: "number" },
  traffic_stop_min_conf: { element: document.querySelector("#trafficStopMinConf"), type: "number" },
  traffic_yellow_min_conf: { element: document.querySelector("#trafficYellowMinConf"), type: "number" },
  traffic_conflict_margin: { element: document.querySelector("#trafficConflictMargin"), type: "number" },
  traffic_roi_enabled: { element: document.querySelector("#trafficRoiEnabled"), type: "boolean" },
  traffic_roi_x_min: { element: document.querySelector("#trafficRoiXMin"), type: "number" },
  traffic_roi_x_max: { element: document.querySelector("#trafficRoiXMax"), type: "number" },
  traffic_roi_y_min: { element: document.querySelector("#trafficRoiYMin"), type: "number" },
  traffic_roi_y_max: { element: document.querySelector("#trafficRoiYMax"), type: "number" },
  traffic_min_area_ratio: { element: document.querySelector("#trafficMinAreaRatio"), type: "number" },
  traffic_max_area_ratio: { element: document.querySelector("#trafficMaxAreaRatio"), type: "number" },
  traffic_prefer_center_weight: { element: document.querySelector("#trafficPreferCenterWeight"), type: "number" },
  crossing_green_required_frames: { element: document.querySelector("#crossingGreenRequiredFrames"), type: "integer" },
};

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function mergeState(previous, next) {
  const incoming = next || {};
  const merged = { ...previous, ...incoming };
  if (previous.device || incoming.device) {
    merged.device = { ...(previous.device || {}), ...(incoming.device || {}) };
  }
  if (previous.navigation || incoming.navigation) {
    merged.navigation = { ...(previous.navigation || {}), ...(incoming.navigation || {}) };
  }
  return merged;
}

function setChip(name, live) {
  chips[name]?.classList.toggle("live", Boolean(live));
}

function setAsrStatus(status) {
  asrStatus = status || "unknown";
  latestState = { ...latestState, asr: asrStatus };
  const enabled = isAsrEnabled();
  setChip("asr", enabled);
  updateAudioChip();
  if (chips.asr) chips.asr.title = `ASR: ${asrStatus}`;
}

function isAsrEnabled() {
  return Boolean(asrStatus) && !["disabled", "missing_dashscope_api_key", "unknown"].includes(asrStatus);
}

function updateAudioChip() {
  setChip("audio", audioConnected && isAsrEnabled());
  if (chips.audio) {
    chips.audio.title = `Audio websocket: ${audioConnected ? "connected" : "disconnected"}; ASR: ${asrStatus}`;
  }
}

function addLog(kind, text, source = "") {
  const item = document.createElement("div");
  item.className = `entry ${kind}`;
  const stamp = new Date().toLocaleTimeString();
  item.innerHTML = `<small>${stamp}${source ? ` · ${source}` : ""}</small><div></div>`;
  item.querySelector("div").textContent = text;
  logEl.prepend(item);
  while (logEl.children.length > 80) logEl.lastChild.remove();
}

function setWebSpeechEnabled(enabled) {
  webSpeechEnabled = Boolean(enabled);
  window.localStorage.setItem("aiglasses.webSpeechEnabled", webSpeechEnabled ? "true" : "false");
  if (webSpeechToggleButton) {
    webSpeechToggleButton.textContent = webSpeechEnabled ? "网页播报 开" : "网页播报 关";
    webSpeechToggleButton.classList.toggle("is-muted", !webSpeechEnabled);
    webSpeechToggleButton.setAttribute("aria-pressed", webSpeechEnabled ? "true" : "false");
  }
  if (webSpeechStatusEl) {
    webSpeechStatusEl.textContent = webSpeechEnabled ? "本窗口会播报" : "本窗口静音";
  }
}

function speakInBrowser(text) {
  if (!webSpeechEnabled || !text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "zh-CN";
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

function renderStateJson() {
  stateEl.textContent = JSON.stringify(latestState, null, 2);
}

function renderDeviceConfig(config, sent = null) {
  if (!config) return;
  latestState = mergeState(latestState, { device_config: config });
  if (Number.isFinite(Number(config.target_fps))) {
    targetFpsEl.value = String(config.target_fps);
  }
  if (Number.isFinite(Number(config.jpeg_quality))) jpegQualityEl.value = String(config.jpeg_quality);
  if (config.camera_profile) cameraProfileEl.value = config.camera_profile;
  if (Number.isFinite(Number(config.ae_level))) aeLevelEl.value = String(config.ae_level);
  if (Number.isFinite(Number(config.saturation))) saturationEl.value = String(config.saturation);
  if (Number.isFinite(Number(config.contrast))) contrastEl.value = String(config.contrast);
  if (Number.isFinite(Number(config.sharpness))) sharpnessEl.value = String(config.sharpness);
  if (Number.isFinite(Number(config.gainceiling))) gainceilingEl.value = String(config.gainceiling);
  const suffix = sent === null ? "" : sent ? "sent" : "pending";
  configStatusEl.textContent = suffix ? `target ${config.target_fps} fps · ${suffix}` : `target ${config.target_fps} fps`;
  renderStateJson();
}

function renderTuning(tuning) {
  if (!tuning) return;
  latestState = mergeState(latestState, { tuning });
  Object.entries(tuningFields).forEach(([key, field]) => {
    if (!field.element || !hasOwn(tuning, key)) return;
    if (field.type === "boolean") field.element.checked = Boolean(tuning[key]);
    else field.element.value = String(tuning[key]);
  });
  renderStateJson();
}

function normaliseRecordingStatus(payload) {
  return payload?.recording || payload || {};
}

function renderRecordingStatus(payload) {
  const status = normaliseRecordingStatus(payload);
  recordingActive = Boolean(status.active);
  latestState = mergeState(latestState, { recording: status });
  if (recordingToggleButton) {
    recordingToggleButton.textContent = recordingActive ? "停止录制" : "开始录制";
    recordingToggleButton.classList.toggle("is-recording", recordingActive);
    recordingToggleButton.setAttribute("aria-pressed", recordingActive ? "true" : "false");
  }
  if (recordingStatusEl) {
    const frameCount = Number(status.frame_count || 0);
    recordingStatusEl.textContent = recordingActive
      ? `录制中 · ${frameCount} 帧 · ${status.session_id || "--"}`
      : "未录制";
    recordingStatusEl.title = status.recording_dir || "";
  }
  renderStateJson();
}

function renderTrafficDebug(observation) {
  const debug = observation?.traffic_light_debug || {};
  trafficDebugEl.textContent = JSON.stringify(
    {
      traffic_light: observation?.traffic_light || null,
      selected: debug.selected || null,
      reason: debug.reason || null,
      filter_enabled: debug.filter_enabled,
      candidates: debug.candidates || [],
    },
    null,
    2,
  );
}

function renderBackendFps(stats) {
  const backendFps = Number(stats?.received_fps_3s);
  backendFpsEl.textContent = Number.isFinite(backendFps) ? backendFps.toFixed(1) : "0.0";
}

function renderBackendBenchmark(benchmark) {
  const maxFps = Number(benchmark?.fps_p50);
  maxBackendFpsEl.textContent = Number.isFinite(maxFps) ? maxFps.toFixed(1) : "--";
}

function renderState(snapshot) {
  const incoming = snapshot || {};
  latestState = mergeState(latestState, incoming);
  if (incoming.device_config) renderDeviceConfig(incoming.device_config);
  if (incoming.tuning) renderTuning(incoming.tuning);
  if (incoming.recording) renderRecordingStatus(incoming.recording);
  renderStateJson();
  const device = latestState.device || {};
  setChip("control", device.control);
  setChip("video", device.video);
  audioConnected = Boolean(device.audio);
  updateAudioChip();
  if (hasOwn(incoming, "asr") || hasOwn(latestState, "asr")) {
    setAsrStatus(latestState.asr);
  }
  modeEl.textContent = latestState.navigation?.mode || "idle";
  uptimeEl.textContent = `${latestState.uptime_s || 0}s`;
  if (hasOwn(incoming, "video_stats")) renderBackendFps(incoming.video_stats);
  if (hasOwn(incoming, "backend_benchmark")) {
    renderBackendBenchmark(incoming.backend_benchmark);
  }
  if (latestState.imu?.accel) {
    const a = latestState.imu.accel;
    imuBriefEl.textContent = `${Number(a.x).toFixed(1)}, ${Number(a.y).toFixed(1)}, ${Number(a.z).toFixed(1)}`;
  }
}

function resizeOverlay() {
  const rect = overlay.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.floor(rect.width * scale));
  overlay.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function drawOverlay() {
  resizeOverlay();
  const rect = overlay.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!latestObservation) return;
  const blind = latestObservation.blind_path;
  const crosswalk = latestObservation.crosswalk;
  const obstacles = latestObservation.obstacles || [];
  const traffic = latestObservation.traffic_light_detection;
  const visionFrame = currentVisionFrameSize();

  function colorOverlay(summary, color, label) {
    if (!summary) return;
    const contour = summary.contour || [];
    if (contour.length < 3) return;

    ctx.beginPath();
    contour.forEach(([nx, ny], index) => {
      const x = nx * rect.width;
      const y = ny * rect.height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.28;
    ctx.fill();
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.globalAlpha = 1;

    const x = rect.width * (0.5 + summary.center_offset / 2);
    const y = rect.height * summary.vertical_position;
    ctx.fillStyle = color;
    ctx.font = "700 13px Avenir Next, sans-serif";
    ctx.fillText(label, x + 10, Math.max(16, y - 10));
  }

  colorOverlay(blind, "#2f9c67", "blind path");
  colorOverlay(crosswalk, "#e4572e", "crosswalk");

  const trafficCandidates = latestObservation.traffic_light_candidates || [];
  if (trafficOnlyView) {
    ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
    ctx.fillStyle = "rgba(56, 189, 248, 0.9)";
    ctx.lineWidth = 2;
    trafficCandidates.forEach((candidate) => drawDetectionBox(candidate, "#94a3b8"));
    if (traffic?.box) drawDetectionBox(traffic, "#38bdf8", true);
    drawTrafficRoi(rect);
    return;
  }

  ctx.strokeStyle = "#ffff00";
  ctx.fillStyle = "#ffff00";
  ctx.lineWidth = 3;
  obstacles.slice(0, 6).forEach((obs) => {
    const [x1, y1, x2, y2] = obs.box || [0, 0, 0, 0];
    const sx = rect.width / visionFrame.width;
    const sy = rect.height / visionFrame.height;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillText(obs.label, x1 * sx + 4, y1 * sy + 14);
  });

  if (traffic?.box) {
    drawDetectionBox(traffic, "#38bdf8", true);
  }
}

function drawDetectionBox(detection, color, selected = false) {
  if (!detection?.box) return;
  const rect = overlay.getBoundingClientRect();
  const visionFrame = currentVisionFrameSize();
  const [x1, y1, x2, y2] = detection.box;
  const sx = rect.width / visionFrame.width;
  const sy = rect.height / visionFrame.height;
  const label = `${selected ? "* " : ""}${detection.label} ${Math.round((detection.confidence || 0) * 100)}%`;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = selected ? 4 : 2;
  ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  ctx.fillText(label, x1 * sx + 4, Math.max(14, y1 * sy - 6));
}

function drawTrafficRoi(rect) {
  const tuning = latestState.tuning || latestObservation?.traffic_light_debug?.thresholds;
  if (!tuning?.traffic_roi_enabled) return;
  const x = Number(tuning.traffic_roi_x_min) * rect.width;
  const y = Number(tuning.traffic_roi_y_min) * rect.height;
  const w = (Number(tuning.traffic_roi_x_max) - Number(tuning.traffic_roi_x_min)) * rect.width;
  const h = (Number(tuning.traffic_roi_y_max) - Number(tuning.traffic_roi_y_min)) * rect.height;
  ctx.strokeStyle = "#f97316";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
}

function currentVisionFrameSize() {
  const width = Number(
    latestState.vision?.image_width || latestState.backend_benchmark?.image_width
  );
  const height = Number(
    latestState.vision?.image_height || latestState.backend_benchmark?.image_height
  );
  if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
    return { width, height };
  }
  return defaultVisionFrame;
}

function pruneDisplayedFrames(now = performance.now()) {
  while (displayedFrameTimes.length && now - displayedFrameTimes[0] > displayFpsWindowMs) {
    displayedFrameTimes.shift();
  }
}

function renderDisplayFps(now = performance.now()) {
  pruneDisplayedFrames(now);
  if (displayedFrameTimes.length < 2) {
    fpsEl.textContent = "0.0";
    return;
  }
  const elapsedMs = displayedFrameTimes[displayedFrameTimes.length - 1] - displayedFrameTimes[0];
  const fps = elapsedMs > 0 ? ((displayedFrameTimes.length - 1) * 1000) / elapsedMs : 0;
  fpsEl.textContent = fps.toFixed(1);
}

function recordDisplayedFrame() {
  const loadedFrameCount = Number.parseInt(frameEl.dataset.frameCount || "0", 10);
  if (!Number.isFinite(loadedFrameCount) || loadedFrameCount <= frameCount) {
    frameLoading = false;
    showQueuedFrame();
    return;
  }
  frameCount = loadedFrameCount;
  frameLoading = false;
  const now = performance.now();
  displayedFrameTimes.push(now);
  renderDisplayFps(now);
  showQueuedFrame();
}

function handleFrameError() {
  frameLoading = false;
  pendingFrameCount = frameCount;
  showQueuedFrame();
}

function showQueuedFrame() {
  if (!queuedFrameCount || queuedFrameCount <= frameCount) return;
  if (queuedFrameBlob) {
    const frame = queuedFrameBlob;
    queuedFrameBlob = null;
    showFrameBlob(frame.frameCount, frame.blob);
    return;
  }
  requestFrame(queuedFrameCount);
}

async function loadRecordingStatus() {
  try {
    const res = await fetch("/api/v1/recording/status", { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderRecordingStatus(data);
  } catch (error) {
    if (recordingStatusEl) recordingStatusEl.textContent = error.message || "录制状态不可用";
  }
}

async function toggleRecording() {
  if (recordingToggleButton) recordingToggleButton.disabled = true;
  try {
    const endpoint = recordingActive ? "/api/v1/recording/stop" : "/api/v1/recording/start";
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderRecordingStatus(data);
  } catch (error) {
    if (recordingStatusEl) recordingStatusEl.textContent = error.message || "录制切换失败";
    await loadRecordingStatus();
  } finally {
    if (recordingToggleButton) recordingToggleButton.disabled = false;
  }
}

async function loadTuning() {
  try {
    const res = await fetch("/api/v1/debug/tuning", { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderTuning(data.tuning);
    tuningStatusEl.textContent = "loaded";
  } catch (error) {
    tuningStatusEl.textContent = error.message || "tuning unavailable";
  }
}

async function saveTuning(event) {
  event.preventDefault();
  const payload = {};
  Object.entries(tuningFields).forEach(([key, field]) => {
    if (!field.element) return;
    if (field.type === "boolean") payload[key] = field.element.checked;
    else if (field.type === "integer") payload[key] = Number.parseInt(field.element.value, 10);
    else payload[key] = Number.parseFloat(field.element.value);
  });
  tuningStatusEl.textContent = "updating";
  try {
    const res = await fetch("/api/v1/debug/tuning", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderTuning(data.tuning);
    tuningStatusEl.textContent = "updated";
  } catch (error) {
    tuningStatusEl.textContent = error.message || "update failed";
  }
}

async function loadDeviceConfig() {
  try {
    const res = await fetch("/api/v1/device/config", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderDeviceConfig(await res.json());
  } catch {
    configStatusEl.textContent = "config unavailable";
  }
}

async function saveDeviceConfig(event) {
  event.preventDefault();
  const targetFps = Number.parseInt(targetFpsEl.value, 10);
  if (!Number.isFinite(targetFps) || targetFps < 1) {
    configStatusEl.textContent = "target fps must be >= 1";
    return;
  }
  if (targetFps > maxTargetFps) {
    configStatusEl.textContent = `target fps must be <= ${maxTargetFps}`;
    return;
  }
  const payload = {
    target_fps: targetFps,
    jpeg_quality: Number.parseInt(jpegQualityEl.value, 10),
    camera_profile: cameraProfileEl.value,
    ae_level: Number.parseInt(aeLevelEl.value, 10),
    saturation: Number.parseInt(saturationEl.value, 10),
    contrast: Number.parseInt(contrastEl.value, 10),
    sharpness: Number.parseInt(sharpnessEl.value, 10),
    gainceiling: Number.parseInt(gainceilingEl.value, 10),
  };

  configStatusEl.textContent = "updating";
  try {
    const res = await fetch("/api/v1/device/config", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderDeviceConfig(data.config, data.sent);
  } catch (error) {
    configStatusEl.textContent = error.message || "update failed";
  }
}

async function disconnectDevice() {
  disconnectDeviceButton.disabled = true;
  configStatusEl.textContent = "disconnecting device";
  try {
    const res = await fetch("/api/v1/device/disconnect", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderState(data.state);
    const channels = data.disconnected || [];
    configStatusEl.textContent = channels.length
      ? `disconnected ${channels.join(", ")}`
      : "no device connections";
  } catch (error) {
    configStatusEl.textContent = error.message || "disconnect failed";
  } finally {
    disconnectDeviceButton.disabled = false;
  }
}

function requestFrame(nextFrameCount) {
  const next = Number.parseInt(nextFrameCount, 10);
  if (!Number.isFinite(next) || next <= frameCount || next <= pendingFrameCount) return;
  if (frameLoading) {
    queuedFrameCount = Math.max(queuedFrameCount, next);
    queuedFrameBlob = null;
    return;
  }
  frameLoading = true;
  pendingFrameCount = next;
  queuedFrameCount = 0;
  loadFrameImage(next);
}

function showFrameBlob(nextFrameCount, blob) {
  const next = Number.parseInt(nextFrameCount, 10);
  if (!Number.isFinite(next) || next <= frameCount || next <= pendingFrameCount) return;
  if (frameLoading) {
    if (next > queuedFrameCount) {
      queuedFrameCount = next;
      queuedFrameBlob = { frameCount: next, blob };
    }
    return;
  }
  frameLoading = true;
  pendingFrameCount = next;
  queuedFrameCount = 0;
  const url = URL.createObjectURL(blob);
  if (frameObjectUrl) URL.revokeObjectURL(frameObjectUrl);
  frameObjectUrl = url;
  frameEl.dataset.frameCount = String(next);
  frameEl.src = url;
  emptyEl.style.display = "none";
}

function unpackVideoFrame(data) {
  if (!(data instanceof ArrayBuffer) || data.byteLength < packetHeaderBytes) return null;
  const view = new DataView(data);
  const magic =
    String.fromCharCode(view.getUint8(0)) +
    String.fromCharCode(view.getUint8(1)) +
    String.fromCharCode(view.getUint8(2)) +
    String.fromCharCode(view.getUint8(3));
  if (magic !== "AGL1") return null;
  const packetType = view.getUint8(5);
  if (packetType !== packetTypeVideoJpeg) return null;
  const seq = Number(view.getBigUint64(8, true));
  const payloadLength = view.getUint32(24, true);
  const payloadStart = packetHeaderBytes;
  const payloadEnd = payloadStart + payloadLength;
  if (payloadEnd !== data.byteLength) return null;
  return {
    frameCount: seq,
    blob: new Blob([data.slice(payloadStart, payloadEnd)], { type: "image/jpeg" }),
  };
}

async function loadFrameImage(requestedFrameCount) {
  try {
    const res = await fetch(`/api/v1/frame.jpg?frame_count=${requestedFrameCount}&t=${Date.now()}`, {
      cache: "no-store",
    });
    if (res.status === 204) {
      handleFrameError();
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const responseFrameCount = Number.parseInt(res.headers.get("x-frame-count") || "", 10);
    const loadedFrameCount = Number.isFinite(responseFrameCount)
      ? responseFrameCount
      : requestedFrameCount;
    const blob = await res.blob();
    frameLoading = false;
    pendingFrameCount = frameCount;
    showFrameBlob(loadedFrameCount, blob);
  } catch {
    handleFrameError();
  }
}

async function refreshFrameFallback() {
  try {
    if (performance.now() - lastWsFrameAt < 2500) return;
    const res = await fetch("/api/v1/frame", { cache: "no-store" });
    const data = await res.json();
    if (data.frame) requestFrame(data.frame_count);
  } catch {
    // UI polling should stay quiet during backend restarts.
  } finally {
    renderDisplayFps();
    window.setTimeout(refreshFrameFallback, 1000);
  }
}

function connectUi() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/ui`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      const frame = unpackVideoFrame(event.data);
      if (frame) {
        lastWsFrameAt = performance.now();
        showFrameBlob(frame.frameCount, frame.blob);
      }
      return;
    }
    const msg = JSON.parse(event.data);
    if (msg.kind === "snapshot") renderState(msg.state);
    if (msg.kind === "device_config") renderDeviceConfig(msg.config, msg.sent);
    if (msg.kind === "tuning") renderTuning(msg.tuning);
    if (msg.kind === "recording") renderRecordingStatus(msg.recording);
    if (msg.kind === "asr") setAsrStatus(msg.status);
    if (msg.kind === "frame") {
      latestState = mergeState(latestState, {
        frame_count: msg.frame_count,
        video_stats: msg.video_stats,
      });
      renderBackendFps(msg.video_stats);
      if (msg.recording) renderRecordingStatus(msg.recording);
    }
    if (msg.kind === "speech") {
      addLog("speech", msg.text, msg.source);
      speakInBrowser(msg.text);
    }
    if (msg.kind === "command") addLog("command", msg.text, msg.source);
    if (msg.kind === "analysis") {
      latestObservation = msg.observation;
      renderState({ navigation: msg.navigation, frame_count: msg.frame_count });
      lightEl.textContent = msg.observation?.traffic_light || "--";
      renderTrafficDebug(msg.observation);
      drawOverlay();
    }
    if (msg.kind === "imu") renderState({ imu: msg.data });
    if (msg.kind === "device") {
      if (msg.state) renderState(msg.state);
      if (msg.channel) {
        renderState({ device: { [msg.channel]: Boolean(msg.connected) } });
      }
    }
  };
  ws.onclose = () => window.setTimeout(connectUi, 900);

  document.querySelector("#commandForm").onsubmit = (event) => {
    event.preventDefault();
    const input = document.querySelector("#commandInput");
    const text = input.value.trim();
    if (!text) return;
    ws.send(text);
    input.value = "";
  };
}

window.addEventListener("resize", drawOverlay);
frameEl.addEventListener("load", recordDisplayedFrame);
frameEl.addEventListener("error", handleFrameError);
document.querySelector("#deviceConfigForm").addEventListener("submit", saveDeviceConfig);
document.querySelector("#tuningForm").addEventListener("submit", saveTuning);
allViewButton.addEventListener("click", () => {
  trafficOnlyView = false;
  allViewButton.classList.add("active");
  trafficViewButton.classList.remove("active");
  drawOverlay();
});
trafficViewButton.addEventListener("click", () => {
  trafficOnlyView = true;
  trafficViewButton.classList.add("active");
  allViewButton.classList.remove("active");
  drawOverlay();
});
disconnectDeviceButton.addEventListener("click", disconnectDevice);
recordingToggleButton.addEventListener("click", toggleRecording);
webSpeechToggleButton.addEventListener("click", () => setWebSpeechEnabled(!webSpeechEnabled));
setWebSpeechEnabled(webSpeechEnabled);
connectUi();
loadDeviceConfig();
loadTuning();
loadRecordingStatus();
refreshFrameFallback();
