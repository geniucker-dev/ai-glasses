const stateEl = document.querySelector("#state");
const logEl = document.querySelector("#log");
const frameEl = document.querySelector("#frame");
const emptyEl = document.querySelector("#empty");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const modeEl = document.querySelector("#mode");
const fpsEl = document.querySelector("#fps");
const backendFpsEl = document.querySelector("#backendFps");
const uptimeEl = document.querySelector("#uptime");
const lightEl = document.querySelector("#light");
const imuBriefEl = document.querySelector("#imuBrief");
const targetFpsEl = document.querySelector("#targetFps");
const configStatusEl = document.querySelector("#configStatus");
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
const displayedFrameTimes = [];
const displayFpsWindowMs = 3000;
const maxTargetFps = 1000;
const packetHeaderBytes = 32;
const packetTypeVideoJpeg = 2;

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

function renderStateJson() {
  stateEl.textContent = JSON.stringify(latestState, null, 2);
}

function renderDeviceConfig(config, sent = null) {
  if (!config) return;
  latestState = mergeState(latestState, { device_config: config });
  if (Number.isFinite(Number(config.target_fps))) {
    targetFpsEl.value = String(config.target_fps);
  }
  const suffix = sent === null ? "" : sent ? "sent" : "pending";
  configStatusEl.textContent = suffix ? `target ${config.target_fps} fps · ${suffix}` : `target ${config.target_fps} fps`;
  renderStateJson();
}

function renderBackendFps(stats) {
  const backendFps = Number(stats?.received_fps_3s);
  backendFpsEl.textContent = Number.isFinite(backendFps) ? backendFps.toFixed(1) : "0.0";
}

function renderState(snapshot) {
  const incoming = snapshot || {};
  latestState = mergeState(latestState, incoming);
  if (incoming.device_config) renderDeviceConfig(incoming.device_config);
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

  ctx.strokeStyle = "#ffff00";
  ctx.fillStyle = "#ffff00";
  ctx.lineWidth = 3;
  obstacles.slice(0, 6).forEach((obs) => {
    const [x1, y1, x2, y2] = obs.box || [0, 0, 0, 0];
    const sx = rect.width / 640;
    const sy = rect.height / 480;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillText(obs.label, x1 * sx + 4, y1 * sy + 14);
  });

  if (traffic?.box) {
    const [x1, y1, x2, y2] = traffic.box;
    const sx = rect.width / 640;
    const sy = rect.height / 480;
    const label = `${traffic.label} ${Math.round((traffic.confidence || 0) * 100)}%`;
    ctx.strokeStyle = "#38bdf8";
    ctx.fillStyle = "#38bdf8";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillText(label, x1 * sx + 4, Math.max(14, y1 * sy - 6));
  }
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

  configStatusEl.textContent = "updating";
  try {
    const res = await fetch("/api/v1/device/config", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ target_fps: targetFps }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderDeviceConfig(data.config, data.sent);
  } catch (error) {
    configStatusEl.textContent = error.message || "update failed";
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
    if (msg.kind === "asr") setAsrStatus(msg.status);
    if (msg.kind === "frame") {
      latestState = mergeState(latestState, {
        frame_count: msg.frame_count,
        video_stats: msg.video_stats,
      });
      renderBackendFps(msg.video_stats);
    }
    if (msg.kind === "speech") addLog("speech", msg.text, msg.source);
    if (msg.kind === "command") addLog("command", msg.text, msg.source);
    if (msg.kind === "analysis") {
      latestObservation = msg.observation;
      renderState({ navigation: msg.navigation, frame_count: msg.frame_count });
      lightEl.textContent = msg.observation?.traffic_light || "--";
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
connectUi();
loadDeviceConfig();
refreshFrameFallback();
