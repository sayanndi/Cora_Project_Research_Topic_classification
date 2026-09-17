/* =========================================================
   Cora GCN Explorer — app logic
   Talks to the FastAPI service: /health, /model_info,
   GET /predict, POST /predict/cora_node
   ========================================================= */

const CORA_CLASSES = {
  0: "Case_Based",
  1: "Genetic_Algorithms",
  2: "Neural_Networks",
  3: "Probabilistic_Methods",
  4: "Reinforcement_Learning",
  5: "Rule_Learning",
  6: "Theory",
};

const CLASS_COLORS = {
  0: "#E8A33D", // amber
  1: "#78C46C", // green
  2: "#4FA3E3", // blue
  3: "#B084E8", // violet
  4: "#E85D75", // rose
  5: "#45BFB0", // teal
  6: "#E8D45D", // gold
};

const FEATURE_DIM = 1433;

// ---- DOM refs -------------------------------------------------
const baseUrlInput   = document.getElementById("baseUrlInput");
const connectBtn     = document.getElementById("connectBtn");
const statusPill     = document.getElementById("statusPill");

const statFeatureDim = document.getElementById("statFeatureDim");
const statNumClasses = document.getElementById("statNumClasses");
const statProvider   = document.getElementById("statProvider");
const classLegend    = document.getElementById("classLegend");
const ioSchema       = document.getElementById("ioSchema");
const refreshModelInfoBtn = document.getElementById("refreshModelInfo");

const tabs        = document.querySelectorAll(".tab");
const tabPanels    = document.querySelectorAll(".tab-panel");

const coraIndicesInput = document.getElementById("coraIndicesInput");
const predictCoraBtn   = document.getElementById("predictCoraBtn");

const genNodeCount        = document.getElementById("genNodeCount");
const generateBtn         = document.getElementById("generateBtn");
const customFeaturesInput = document.getElementById("customFeaturesInput");
const customEdgesInput    = document.getElementById("customEdgesInput");
const predictCustomBtn    = document.getElementById("predictCustomBtn");

const resultsEmpty = document.getElementById("resultsEmpty");
const resultsList  = document.getElementById("resultsList");
const resultsMeta  = document.getElementById("resultsMeta");

const footerLog = document.getElementById("footerLog");

const canvas = document.getElementById("graphCanvas");
const ctx = canvas.getContext("2d");

// ---- state ------------------------------------------------------
let graphNodes = [];   // { x, y, vx, vy, classId | null }
let graphEdges = [];   // [i, j]
let animHandle = null;

// =================================================================
// Utilities
// =================================================================
function baseUrl() {
  return baseUrlInput.value.trim().replace(/\/+$/, "");
}

function log(message) {
  const time = new Date().toLocaleTimeString();
  footerLog.textContent = `[${time}] ${message}`;
}

function setStatus(state, text) {
  statusPill.className = `status-pill status-pill--${state}`;
  statusPill.querySelector(".status-text").textContent = text;
}

async function apiGet(path) {
  const res = await fetch(`${baseUrl()}${path}`, { method: "GET" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function apiSend(path, method, payload) {
  const res = await fetch(`${baseUrl()}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ? JSON.stringify(body.detail) : detail;
    } catch (_) {}
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

function randomFeatureVector(dim = FEATURE_DIM) {
  // Cora features are multi-hot bag-of-words: mostly 0s, a few 1s.
  const vec = new Array(dim).fill(0);
  const activeCount = 15 + Math.floor(Math.random() * 20);
  for (let i = 0; i < activeCount; i++) {
    vec[Math.floor(Math.random() * dim)] = 1;
  }
  return vec;
}

// =================================================================
// Connection / health
// =================================================================
async function connect() {
  setStatus("pending", "Connecting\u2026");
  try {
    const health = await apiGet("/health");
    setStatus("ok", `Online \u2014 ${health.providers?.[0] ?? "ready"}`);
    log("Connected to API");
    await loadModelInfo();
  } catch (err) {
    setStatus("error", "Connection failed");
    log(`Health check failed: ${err.message}`);
  }
}

// =================================================================
// Model info
// =================================================================
async function loadModelInfo() {
  try {
    const info = await apiGet("/model_info");

    statFeatureDim.textContent = info.feature_dimension ?? "\u2014";
    statNumClasses.textContent = info.num_classes ?? "\u2014";

    const providerLabel = (info.inputs?.[0]?.type || "onnx").replace("tensor(", "").replace(")", "");
    statProvider.textContent = providerLabel;

    renderLegend(info.class_mapping || CORA_CLASSES);

    ioSchema.textContent = JSON.stringify(
      { inputs: info.inputs, outputs: info.outputs },
      null,
      2
    );
  } catch (err) {
    log(`Could not load model_info: ${err.message}`);
    renderLegend(CORA_CLASSES); // fall back to known mapping so the UI stays usable
  }
}

function renderLegend(mapping) {
  classLegend.innerHTML = "";
  Object.entries(mapping).forEach(([id, name]) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:${classColor(id)}"></span>${prettifyClassName(name)}`;
    classLegend.appendChild(item);
  });
}

function classColor(id) {
  return CLASS_COLORS[Number(id)] || "#8890A0";
}

function prettifyClassName(name) {
  return String(name).replace(/_/g, " ");
}

// =================================================================
// Tabs
// =================================================================
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => {
      t.classList.remove("tab--active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("tab--active");
    tab.setAttribute("aria-selected", "true");

    const target = tab.dataset.tab;
    tabPanels.forEach((p) => {
      p.classList.toggle("tab-panel--active", p.dataset.panel === target);
    });
  });
});

// =================================================================
// Predict: real Cora node
// =================================================================
async function predictCoraNodes() {
  const indices = coraIndicesInput.value
    .split(",")
    .map((v) => v.trim())
    .filter((v) => v.length)
    .map(Number);

  if (!indices.length || indices.some(Number.isNaN)) {
    log("Enter valid, comma-separated node indices.");
    return;
  }

  setButtonBusy(predictCoraBtn, true);
  try {
    const data = await apiSend("/predict/cora_node", "POST", { node_indices: indices });
    renderResults(data, `${data.num_nodes} nodes \u00b7 ${data.num_edges} edges in full graph`);
    buildGraphFromPredictions(data.predictions, { fullGraphHint: true });
    log(`Classified ${data.predictions.length} Cora node(s)`);
  } catch (err) {
    log(`Prediction failed: ${err.message}`);
  } finally {
    setButtonBusy(predictCoraBtn, false);
  }
}

// =================================================================
// Predict: custom graph
// =================================================================
generateBtn.addEventListener("click", () => {
  const n = Math.max(1, Math.min(50, Number(genNodeCount.value) || 4));
  const features = Array.from({ length: n }, () => randomFeatureVector());
  customFeaturesInput.value = JSON.stringify(features);

  // simple ring of edges so the graph is connected, both directions
  const edgesFrom = [], edgesTo = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    edgesFrom.push(i, j);
    edgesTo.push(j, i);
  }
  customEdgesInput.value = n > 1 ? JSON.stringify([edgesFrom, edgesTo]) : "[[0],[0]]";
  log(`Generated a random ${n}-node graph`);
});

async function predictCustomGraph() {
  let node_features, edge_indices;
  try {
    node_features = JSON.parse(customFeaturesInput.value);
  } catch (e) {
    log("node_features is not valid JSON.");
    return;
  }
  if (!Array.isArray(node_features) || !node_features.length) {
    log("node_features must be a non-empty array of arrays.");
    return;
  }
  if (node_features.some((v) => v.length !== FEATURE_DIM)) {
    log(`Every node needs exactly ${FEATURE_DIM} feature values.`);
    return;
  }

  if (customEdgesInput.value.trim()) {
    try {
      edge_indices = JSON.parse(customEdgesInput.value);
    } catch (e) {
      log("edge_indices is not valid JSON.");
      return;
    }
  }

  const payload = { node_features };
  if (edge_indices) payload.edge_indices = edge_indices;

  setButtonBusy(predictCustomBtn, true);
  try {
    // NOTE: the backend defines this route as `@app.get("/predict")`.
    // A GET request cannot carry a JSON body in the browser's fetch API
    // (and most HTTP clients), so this call will fail until that route
    // is changed to `@app.post(...)` on the server. We still send it
    // as specified so this works as soon as that's fixed.
    const data = await apiSend("/predict", "GET", payload);
    renderResults(data, `${data.num_nodes} nodes \u00b7 ${data.num_edges} edges`);
    buildGraphFromPredictions(data.predictions, { fullGraphHint: false });
    log(`Classified ${data.predictions.length} custom node(s)`);
  } catch (err) {
    log(`Prediction failed: ${err.message}`);
    if (String(err.message).toLowerCase().includes("failed to fetch") || err instanceof TypeError) {
      log("This route is defined as GET but needs a JSON body — change it to @app.post on the server.");
    }
  } finally {
    setButtonBusy(predictCustomBtn, false);
  }
}

function setButtonBusy(btn, busy) {
  btn.disabled = busy;
  btn.textContent = busy ? "Running\u2026" : btn.dataset.label || btn.textContent;
  if (!busy) return;
  btn.dataset.label = btn.dataset.label || btn.textContent;
}

// =================================================================
// Results rendering
// =================================================================
function renderResults(data, metaText) {
  resultsMeta.textContent = metaText || "";
  const predictions = data.predictions || [];

  if (!predictions.length) {
    resultsEmpty.style.display = "block";
    resultsList.innerHTML = "";
    return;
  }

  resultsEmpty.style.display = "none";
  resultsList.innerHTML = "";

  predictions.forEach((p) => {
    const card = document.createElement("article");
    card.className = "result-card";

    const top = document.createElement("div");
    top.className = "result-card__top";
    top.innerHTML = `
      <span class="result-card__id">node #${p.node_index}</span>
      <span class="result-card__class">
        <span class="legend-swatch" style="background:${classColor(p.predicted_class_id)}"></span>
        ${prettifyClassName(p.predicted_class_name)}
      </span>
    `;
    card.appendChild(top);

    const bars = document.createElement("div");
    bars.className = "prob-bars";
    p.probabilities.forEach((prob, classId) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      const pct = (prob * 100).toFixed(1);
      row.innerHTML = `
        <span class="prob-row__label">${prettifyClassName(CORA_CLASSES[classId] || classId)}</span>
        <span class="prob-row__track"><span class="prob-row__fill" style="width:${pct}%; background:${classColor(classId)}"></span></span>
        <span class="prob-row__value">${pct}%</span>
      `;
      bars.appendChild(row);
    });
    card.appendChild(bars);
    resultsList.appendChild(card);
  });
}

// =================================================================
// Graph visualization (canvas)
// =================================================================
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function initPlaceholderGraph() {
  const n = 9;
  graphNodes = Array.from({ length: n }, () => ({
    x: Math.random(),
    y: Math.random(),
    vx: (Math.random() - 0.5) * 0.0006,
    vy: (Math.random() - 0.5) * 0.0006,
    classId: null,
  }));
  graphEdges = [];
  for (let i = 0; i < n; i++) {
    graphEdges.push([i, (i + 1) % n]);
    if (i % 2 === 0) graphEdges.push([i, (i + 3) % n]);
  }
}

function buildGraphFromPredictions(predictions, { fullGraphHint }) {
  const n = Math.max(predictions.length, 2);
  graphNodes = predictions.map(() => ({
    x: Math.random(),
    y: Math.random(),
    vx: (Math.random() - 0.5) * 0.0006,
    vy: (Math.random() - 0.5) * 0.0006,
    classId: null,
  }));

  graphEdges = [];
  for (let i = 0; i < predictions.length; i++) {
    graphEdges.push([i, (i + 1) % predictions.length]);
  }

  // fade the colors in rather than snapping, handled in the draw loop
  predictions.forEach((p, i) => {
    graphNodes[i].targetClass = p.predicted_class_id;
    graphNodes[i].colorMix = 0;
  });
}

function drawGraph() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  // integrate simple drifting motion, bounce off edges
  graphNodes.forEach((node) => {
    node.x += node.vx;
    node.y += node.vy;
    if (node.x < 0.06 || node.x > 0.94) node.vx *= -1;
    if (node.y < 0.1 || node.y > 0.9) node.vy *= -1;
    if (node.colorMix !== undefined && node.colorMix < 1) {
      node.colorMix = Math.min(1, node.colorMix + 0.04);
    }
  });

  // edges
  ctx.lineWidth = 1;
  graphEdges.forEach(([a, b]) => {
    const na = graphNodes[a], nb = graphNodes[b];
    if (!na || !nb) return;
    ctx.strokeStyle = "rgba(150,160,180,0.16)";
    ctx.beginPath();
    ctx.moveTo(na.x * w, na.y * h);
    ctx.lineTo(nb.x * w, nb.y * h);
    ctx.stroke();
  });

  // nodes
  graphNodes.forEach((node) => {
    const px = node.x * w;
    const py = node.y * h;
    const baseColor = "#3A4050";
    const targetColor = node.targetClass !== undefined ? classColor(node.targetClass) : null;
    const mix = node.colorMix || 0;

    const fill = targetColor ? mixHexColors(baseColor, targetColor, mix) : baseColor;
    const radius = targetColor ? 6 + mix * 2 : 5;

    if (targetColor && mix > 0.15) {
      ctx.beginPath();
      ctx.fillStyle = hexWithAlpha(targetColor, 0.16 * mix);
      ctx.arc(px, py, radius + 6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.fillStyle = fill;
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(15,17,22,0.9)";
    ctx.stroke();
  });

  animHandle = requestAnimationFrame(drawGraph);
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  const bigint = parseInt(clean, 16);
  return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
}

function mixHexColors(hexA, hexB, t) {
  const a = hexToRgb(hexA), b = hexToRgb(hexB);
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

function hexWithAlpha(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// =================================================================
// Wire up
// =================================================================
connectBtn.addEventListener("click", connect);
refreshModelInfoBtn.addEventListener("click", loadModelInfo);
predictCoraBtn.addEventListener("click", predictCoraNodes);
predictCustomBtn.addEventListener("click", predictCustomGraph);

window.addEventListener("resize", resizeCanvas);

// initial paint
renderLegend(CORA_CLASSES);
resizeCanvas();
initPlaceholderGraph();
drawGraph();
log("Ready. Set the API base URL and click Connect.");
