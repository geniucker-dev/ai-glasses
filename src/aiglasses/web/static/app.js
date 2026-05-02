const stateEl = document.querySelector("#state");
const logEl = document.querySelector("#log");
const frameEl = document.querySelector("#frame");
const emptyEl = document.querySelector("#empty");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const modeEl = document.querySelector("#mode");
const fpsEl = document.querySelector("#fps");
const uptimeEl = document.querySelector("#uptime");
const lightEl = document.querySelector("#light");
const imuBriefEl = document.querySelector("#imuBrief");
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

function renderState(snapshot) {
  const incoming = snapshot || {};
  latestState = mergeState(latestState, incoming);
  stateEl.textContent = JSON.stringify(latestState, null, 2);
  const device = latestState.device || {};
  setChip("control", device.control);
  setChip("video", device.video);
  audioConnected = Boolean(device.audio);
  updateAudioChip();
  if (hasOwn(incoming, "asr") || hasOwn(latestState, "asr")) {
    setAsrStatus(latestState.asr);
  }
  modeEl.textContent = latestState.navigation?.mode || "idle";
  fpsEl.textContent = String(latestState.frame_count || frameCount || 0);
  uptimeEl.textContent = `${latestState.uptime_s || 0}s`;
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

  ctx.strokeStyle = "#f0b429";
  ctx.fillStyle = "#f0b429";
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

async function refreshFrame() {
  try {
    const res = await fetch("/api/v1/frame", { cache: "no-store" });
    const data = await res.json();
    if (data.frame && data.frame_count !== frameCount) {
      frameCount = data.frame_count;
      frameEl.src = data.frame;
      emptyEl.style.display = "none";
      fpsEl.textContent = String(frameCount);
    }
  } catch {
    // UI polling should stay quiet during backend restarts.
  } finally {
    window.setTimeout(refreshFrame, 160);
  }
}

function connectUi() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/ui`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.kind === "snapshot") renderState(msg.state);
    if (msg.kind === "asr") setAsrStatus(msg.status);
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
connectUi();
refreshFrame();
