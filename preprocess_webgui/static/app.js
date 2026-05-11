/* Main UI + polling for the Preprocess web GUI. */

(() => {

const $  = (id) => document.getElementById(id);
const qs = (s)  => document.querySelector(s);
const qsa = (s) => Array.from(document.querySelectorAll(s));

let session = null;
let logPos = 0;
let pipelineRunning = false;
let currentExperimentPath = "";
let currentExperimentVersion = "";
let currentMaskVersion = 0;
let cpZoom = 1;

// Per-experiment cache-buster for frame/thumbnail URLs. Identical experiments
// reuse the browser cache; switching experiments forces a refetch so the user
// never sees the previous experiment's frames.
function frameUrl(idx) {
  return `/api/frame/${idx}.png?e=${currentExperimentVersion}`;
}
function thumbUrl(idx, w = 120) {
  return `/api/thumbnail/${idx}.png?w=${w}&e=${currentExperimentVersion}`;
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

// ── Tabs ────────────────────────────────────────────────────────────────
qsa(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    qsa(".tab-btn").forEach(b => b.classList.remove("active"));
    qsa(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    qsa(`.panel[data-workflow="${btn.dataset.tab}"]`).forEach(p => p.classList.add("active"));
    const panel = $(`panel-${btn.dataset.tab}`);
    if (panel) panel.classList.add("active");
    onTabChange(btn.dataset.tab);
  });
});

$("btn-theme").addEventListener("click", () => {
  const b = document.body;
  b.dataset.theme = b.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("preprocess_theme", b.dataset.theme);
});
const savedTheme = localStorage.getItem("preprocess_theme");
if (savedTheme) document.body.dataset.theme = savedTheme;

function onTabChange(tab) {
  if (tab === "cellpose")  setupCellposeTab();
  if (tab === "tracking")  {
    refreshInterval();
    setupShiftTab();
    setupRoiTab();
  }
  if (tab === "run") {
    refreshSummary();
    refreshValidation();
  }
}

// ── Fetch helpers ───────────────────────────────────────────────────────
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

// ── Session ─────────────────────────────────────────────────────────────
async function loadSession() {
  session = await jget("/api/session");
  refreshRail();
  syncFormsFromSession();
}

function syncFormsFromSession() {
  if (!session) return;
  $("cp-frame").value    = session.frame_idx;
  $("cp-flow").value     = session.flow_threshold;
  $("cp-cellprob").value = session.cellprob_threshold;
  $("cp-niter").value    = session.niter;
  $("cp-diam").value     = session.diameter;
  $("si-value").value    = session.save_interval;
  $("sh-frame").value    = session.shift_frame_idx;
  $("sh-result").textContent = `shift_xy = (${session.shift_xy[0]}, ${session.shift_xy[1]})`;
  $("md-value").value    = session.max_distance.toFixed(1);
  $("roi-radius").value  = session.radius;
  $("roi-y").value       = session.y_shift;
  $("roi-x").value       = session.x_shift;
  $("pa-f0").value       = session.f0_frame;
  $("pa-stim").value     = session.stim_frames || "";
  $("run-mode").value    = session.last_run_mode || "full";
  updateReviewStatus();
  currentExperimentPath  = session.global_dir || "";
  currentExperimentVersion = encodeURIComponent(currentExperimentPath);
  if (currentExperimentPath) $("exp-path").value = currentExperimentPath;
}

async function patchSession(patch) {
  session = await jpost("/api/session", patch);
  refreshRail();
}

function refreshRail() {
  if (!session) return;
  const expName = session.global_dir
    ? session.global_dir.split("/").slice(-2).join("/")
    : "—";
  $("rail-exp").textContent    = expName;
  $("rail-frames").textContent = session.all_frames?.length ?? 0;
  $("rail-masks").textContent  = session.all_masks?.length ?? 0;
  $("rail-cells").textContent  = session.last_roi_count || "—";
  $("rail-params").textContent = [
    `flow        = ${session.flow_threshold}`,
    `cellprob    = ${session.cellprob_threshold}`,
    `niter       = ${session.niter}`,
    `diameter    = ${session.diameter}`,
    `save_intvl  = ${session.save_interval}`,
    `shift_frame = ${session.shift_frame_idx}`,
    `shift_xy    = ${session.shift_xy[0]} ${session.shift_xy[1]}`,
    `max_dist    = ${session.max_distance?.toFixed?.(1)}`,
    `radius      = ${session.radius}`,
    `y_shift     = ${session.y_shift}`,
    `x_shift     = ${session.x_shift}`,
    `grace_per   = ${session.grace_period}`,
    `roi_cells   = ${session.last_roi_count || 0}`,
    `reviewed    = ${session.segmentation_reviewed ? "yes" : "no"}`,
  ].join("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 0 · Experiment
// ═══════════════════════════════════════════════════════════════════════════
async function loadExperimentList() {
  const data = await jget("/api/experiments");
  const el = $("exp-list");
  el.innerHTML = "";
  if (!data.experiments.length) {
    el.innerHTML = `<div class="sub">No experiments found in ${data.root}</div>`;
    return;
  }
  for (const exp of data.experiments) {
    const d = document.createElement("div");
    d.className = "exp-tile" + (exp.has_config ? " has-config" : "");
    d.innerHTML = `
      <div class="name">${exp.rel}</div>
      <div class="meta">
        ${exp.frames} frames · ${exp.masks} masks
        ${exp.has_config ? "· <span style='color:var(--amber)'>prior config</span>" : ""}
      </div>`;
    d.addEventListener("click", () => loadExperiment(exp.path));
    el.appendChild(d);
  }
}

async function loadExperiment(path) {
  $("exp-banner").style.display = "none";
  const res = await jpost("/api/experiment", { path });
  if (!res.ok) {
    alert(res.error || "failed");
    return;
  }
  session = res;
  currentExperimentPath = session.global_dir;
  currentExperimentVersion = encodeURIComponent(session.global_dir || "");
  $("exp-path").value = session.global_dir;
  $("exp-summary").textContent =
    `${session.all_frames.length} frames · ${session.all_masks.length} masks`;

  if (res.resumed_from_config) {
    const b = $("exp-banner");
    b.style.display = "block";
    b.innerHTML = `✓ Resumed parameters from <code>${res.config_txt_path}</code>`;
  }

  syncFormsFromSession();
  refreshRail();
  // Reset thumb + scrubber + image caches for the new experiment
  thumbsLoadedFor = null;
  cpHasMask = false;
  frameImageCache.clear();
  roiCentroidsCache = null;
  refreshValidation();
}

$("btn-exp-load").addEventListener("click", () => {
  const p = $("exp-path").value.trim();
  if (p) loadExperiment(p);
});

// ═══════════════════════════════════════════════════════════════════════════
// Tab 1 · Cellpose (frame scrubber + thumbs + preview)
// ═══════════════════════════════════════════════════════════════════════════
let thumbsLoadedFor = null;
const patchFrameSession = debounce((idx) => patchSession({ frame_idx: idx }), 250);

function setupCellposeTab() {
  if (!session?.all_frames?.length) return;
  const n = session.all_frames.length;
  $("cp-scrub").max = n - 1;
  $("cp-scrub").value = session.frame_idx;
  $("cp-frame").value = session.frame_idx;
  updateScrubLabel();
  loadCellposeFrame(session.frame_idx);
  loadThumbs();
  // If we already have a segmentation, render it
  if (cpHasMask) reloadMaskOverlay();
}

function updateScrubLabel() {
  const n = session.all_frames?.length || 0;
  $("cp-scrub-label").textContent = `${$("cp-scrub").value} / ${n - 1}`;
}

$("cp-scrub").addEventListener("input", (e) => {
  const idx = parseInt(e.target.value, 10);
  $("cp-frame").value = idx;
  updateScrubLabel();
  loadCellposeFrame(idx);
  patchFrameSession(idx);
  highlightThumb(idx);
});
$("cp-frame").addEventListener("change", (e) => {
  const idx = parseInt(e.target.value, 10);
  $("cp-scrub").value = idx;
  updateScrubLabel();
  loadCellposeFrame(idx);
  patchFrameSession(idx);
  highlightThumb(idx);
});

// Cache decoded Image objects across frame switches so the browser doesn't
// rebuild the bitmap each time. Keyed by experiment to avoid leaking data
// across loads.
const frameImageCache = new Map();  // key: `${expVer}|${idx}` → HTMLImageElement
function getFrameImage(idx) {
  const key = `${currentExperimentVersion}|${idx}`;
  let im = frameImageCache.get(key);
  if (!im) {
    im = new Image();
    im.decoding = "async";
    im.src = frameUrl(idx);
    frameImageCache.set(key, im);
  }
  return im;
}

function loadCellposeFrame(idx) {
  const img = $("cp-img-raw");
  const cached = getFrameImage(idx);
  // If the bitmap is already decoded, switch instantly; otherwise let the
  // browser keep showing the previous frame until the new src finishes
  // loading (default <img> behaviour, no white flash).
  img.src = cached.src;
  img.style.display = "block";
  $("cp-placeholder").style.display = "none";
  // Clear mask overlay (it was for a different frame)
  $("cp-img-mask").style.display = "none";
  preloadFrames(idx);
}

function preloadFrames(idx) {
  if (!session?.all_frames?.length) return;
  // Widen the preload window from ±2 to ±5 so casual scrubbing hits the
  // disk-cache → memory-cache pipeline most of the time.
  const radius = 5;
  for (let d = -radius; d <= radius; d++) {
    if (d === 0) continue;
    const i = idx + d;
    if (i < 0 || i >= session.all_frames.length) continue;
    getFrameImage(i);  // populates frameImageCache as a side effect
  }
}

function loadThumbs() {
  if (!session.all_frames?.length) return;
  if (thumbsLoadedFor === session.global_dir) return;
  thumbsLoadedFor = session.global_dir;
  const n = session.all_frames.length;
  const strip = $("cp-thumbs");
  strip.innerHTML = "";
  const stride = Math.max(1, Math.ceil(n / 30));
  for (let i = 0; i < n; i += stride) {
    const im = document.createElement("img");
    im.loading = "lazy";
    im.src = thumbUrl(i, 120);
    im.dataset.idx = i;
    im.title = `frame ${i}`;
    im.addEventListener("click", () => {
      $("cp-scrub").value = i;
      $("cp-frame").value = i;
      updateScrubLabel();
      loadCellposeFrame(i);
      patchFrameSession(i);
      highlightThumb(i);
    });
    strip.appendChild(im);
  }
  highlightThumb(session.frame_idx);
}

function highlightThumb(idx) {
  qsa("#cp-thumbs img").forEach((im) => {
    im.classList.toggle("active", parseInt(im.dataset.idx, 10) === idx);
  });
}

let cpHasMask = false;

$("btn-run-cellpose").addEventListener("click", async () => {
  const body = {
    frame_idx: parseInt($("cp-frame").value, 10),
    flow_threshold: parseFloat($("cp-flow").value),
    cellprob_threshold: parseFloat($("cp-cellprob").value),
    niter: parseInt($("cp-niter").value, 10),
    diameter: parseInt($("cp-diam").value, 10),
  };
  const res = await jpost("/api/cellpose/run", body);
  if (!res.ok) {
    $("cp-status").textContent = res.error;
    return;
  }
  $("btn-run-cellpose").disabled = true;
  $("cp-status").textContent = "Running Cellpose…";
  startCellposeTimer();
});

// ── Live elapsed-time ticker while Cellpose is running ──────────────────
// The model itself is opaque (no progress callback), so showing a ticking
// counter is the cheapest way to make the 2–5s wait feel like work, not
// like a hang.
let cpTimerHandle = null;
let cpTimerStart = 0;
function startCellposeTimer() {
  stopCellposeTimer();
  cpTimerStart = performance.now();
  const tick = () => {
    const dt = (performance.now() - cpTimerStart) / 1000;
    $("cp-status").textContent = `Running Cellpose · ${dt.toFixed(1)}s elapsed`;
    cpTimerHandle = requestAnimationFrame(tick);
  };
  cpTimerHandle = requestAnimationFrame(tick);
}
function stopCellposeTimer() {
  if (cpTimerHandle != null) {
    cancelAnimationFrame(cpTimerHandle);
    cpTimerHandle = null;
  }
}

// ── SSE: cellpose status stream ─────────────────────────────────────────
// Replaces the previous setTimeout polling loop with a single long-lived
// EventSource. Status updates push as soon as the worker thread mutates
// its state — no 250-800ms polling latency between transitions.
let cellposeES = null;
function openCellposeStream() {
  if (cellposeES) return;
  cellposeES = new EventSource("/api/cellpose/stream");
  cellposeES.addEventListener("status", (e) => {
    let st;
    try { st = JSON.parse(e.data); } catch { return; }
    handleCellposeStatus(st);
  });
  cellposeES.onerror = () => {
    // Browser auto-reconnects; nothing to do here.
  };
}

function handleCellposeStatus(st) {
  if (!st) return;
  if (st.state === "running") {
    // Server may report "Loading model..." then "Segmenting..."; keep our
    // local timer running and show the worker's message instead of the
    // generic "running" label.
    if (st.message) $("cp-status").textContent = st.message;
    if (cpTimerHandle == null) startCellposeTimer();
    $("btn-run-cellpose").disabled = true;
    return;
  }
  stopCellposeTimer();
  if (st.state === "done") {
    if (!st.has_mask) {
      // Race: the worker flipped to "done" a hair before the on_done hook
      // committed temp_segmentation. The very next event will arrive
      // within ~150ms and carry has_mask=true — just wait for it.
      $("cp-status").textContent = st.message || "Finishing…";
      return;
    }
    $("btn-run-cellpose").disabled = false;
    $("cp-cells").textContent = st.n_cells;
    $("rail-cells").textContent = st.n_cells;
    $("cp-status").textContent = st.message || `Detected ${st.n_cells} cells`;
    const rt = (st.finished_at && st.started_at)
      ? `${(st.finished_at - st.started_at).toFixed(1)}s` : "—";
    $("cp-runtime").textContent = rt;
    cpHasMask = true;
    if (st.mask_version && st.mask_version !== currentMaskVersion) {
      currentMaskVersion = st.mask_version;
      reloadMaskOverlay();
      // Drop the stale centroids cache; ROI tab will refetch on next open.
      roiCentroidsCache = null;
      // Refresh session so the rail picks up new preview metadata.
      loadSession();
    }
    return;
  }
  if (st.state === "error") {
    $("btn-run-cellpose").disabled = false;
    $("cp-status").textContent = st.message || "Error";
    return;
  }
  // idle / unknown
  $("cp-status").textContent = st.message || "";
  $("btn-run-cellpose").disabled = false;
}

function reloadMaskOverlay() {
  const m = $("cp-img-mask");
  m.onerror = () => {
    m.style.display = "none";
    $("cp-status").textContent = "Segmentation finished, but the overlay image was not ready. Try Run Cellpose again or refresh.";
  };
  m.onload = () => {
    m.style.display = "block";
    applyOverlayControls();
  };
  m.src = `/api/mask/preview.png?v=${currentMaskVersion}`;
}

function updateReviewStatus() {
  if (!$("cp-review-status") || !session) return;
  $("cp-review-status").textContent = session.segmentation_reviewed
    ? "Confirmed. These parameters are ready for the run check."
    : "Not reviewed yet";
  $("cp-review-box")?.classList.toggle("approved", !!session.segmentation_reviewed);
}

async function setSegmentationReviewed(reviewed) {
  session = await jpost("/api/cellpose/review", { reviewed });
  updateReviewStatus();
  refreshRail();
  refreshValidation();
}

function applyOverlayControls() {
  const mask = $("cp-img-mask");
  const raw = $("cp-img-raw");
  const canvas = $("cp-canvas");
  mask.style.opacity = (parseInt($("cp-overlay-opacity").value, 10) / 100).toString();
  mask.classList.toggle("outline-mode", $("cp-outline-mode").checked);
  raw.style.transform = `scale(${cpZoom})`;
  mask.style.transform = `scale(${cpZoom})`;
  canvas.classList.toggle("zoomed", cpZoom > 1);
}

$("btn-cp-approve").addEventListener("click", () => setSegmentationReviewed(true));
$("btn-cp-rerun").addEventListener("click", () => setSegmentationReviewed(false));
$("cp-overlay-opacity").addEventListener("input", applyOverlayControls);
$("cp-outline-mode").addEventListener("change", applyOverlayControls);
$("btn-cp-zoom-in").addEventListener("click", () => { cpZoom = Math.min(4, cpZoom + 0.25); applyOverlayControls(); });
$("btn-cp-zoom-out").addEventListener("click", () => { cpZoom = Math.max(1, cpZoom - 0.25); applyOverlayControls(); });
$("btn-cp-zoom-reset").addEventListener("click", () => { cpZoom = 1; applyOverlayControls(); });

// ═══════════════════════════════════════════════════════════════════════════
// Tab 2 · Save Interval
// ═══════════════════════════════════════════════════════════════════════════
function refreshInterval() {
  const n = session.all_frames?.length || 0;
  const auto = n <= 500 ? 10 : 100;
  $("si-auto").value = `${auto} (${n} frames)`;
  if (!session.save_interval) {
    $("si-value").value = auto;
    patchSession({ save_interval: auto });
  }
  $("si-hint").textContent =
    `With save_interval = ${$("si-value").value} on ${n} frames, ` +
    `trajectories.py will checkpoint ~${Math.ceil(n / Math.max(1, $("si-value").value))} times.`;
}
$("si-value").addEventListener("change", () => {
  patchSession({ save_interval: parseInt($("si-value").value, 10) });
  refreshInterval();
});

// ═══════════════════════════════════════════════════════════════════════════
// Tab 3 · Shift
// ═══════════════════════════════════════════════════════════════════════════
function setupShiftTab() {
  if (!ShiftCanvas._inited) {
    ShiftCanvas.init($("sh-canvas"), (dx, dy) => {
      $("sh-result").textContent = `shift_xy = (${dx}, ${dy})`;
      patchSession({ shift_xy: [dx, dy] });
    });
    ShiftCanvas._inited = true;
  }
  if (session.all_frames?.length) {
    $("sh-frame").max = session.all_frames.length - 1;
  }
}

$("btn-sh-load").addEventListener("click", async () => {
  const idx = parseInt($("sh-frame").value, 10);
  if (idx < 1) return alert("shift_frame must be ≥ 1");
  $("sh-placeholder").style.display = "none";
  await patchSession({ shift_frame_idx: idx });
  try {
    await ShiftCanvas.loadFrames(idx, currentExperimentVersion);
  } catch (e) { alert(`failed: ${e}`); }
});

$("btn-sh-clear").addEventListener("click", () => {
  ShiftCanvas.clearPoints();
  $("sh-result").textContent = "shift_xy = (0, 0)";
  patchSession({ shift_xy: [0, 0] });
});

// ═══════════════════════════════════════════════════════════════════════════
// Tab 4 · Max Distance
// ═══════════════════════════════════════════════════════════════════════════
async function mdCompute(cellId) {
  const res = await jpost("/api/maxdistance/compute",
                          cellId != null ? { cell_id: cellId } : {});
  if (!res.ok) { $("md-info").textContent = res.error; return; }
  $("md-value").value = res.mean_distance.toFixed(1);
  $("md-info").textContent =
    `Cell ${res.chosen_id}: ${res.neighbors.length} neighbours · ` +
    `mean ${res.mean_distance.toFixed(1)} px · ` +
    `distances ${res.neighbors.map(n => n.distance.toFixed(1)).join(", ")}`;
  $("md-stats").value =
    `cell ${res.chosen_id} · ${res.neighbors.length} neighbours · mean ${res.mean_distance.toFixed(1)} px`;
  $("md-visual").src = res.visualization_url;
  $("md-visual").style.display = "block";
  $("md-placeholder").style.display = "none";
  await loadSession();
}

$("btn-md-compute").addEventListener("click", () => mdCompute());
$("btn-md-resample").addEventListener("click", () => mdCompute());
$("md-value").addEventListener("change", () =>
  patchSession({ max_distance: parseFloat($("md-value").value) }));

// ═══════════════════════════════════════════════════════════════════════════
// Tab 5 · ROI
// ═══════════════════════════════════════════════════════════════════════════
let roiInited = false;

// Apply a CSS clip-path to the cached cellpose preview so only the area
// inside the ROI circle is visible. Pure client-side, instant — no server
// roundtrip per move/resize.
function applyRoiClip(radius, xShift, yShift) {
  const mask = $("roi-img-mask");
  const raw = $("roi-img-raw");
  if (!raw.naturalWidth || !raw.naturalHeight) return;
  const rect = raw.getBoundingClientRect();
  if (!rect.width) return;
  const scale = rect.width / raw.naturalWidth;
  const cx = (raw.naturalWidth / 2 + xShift) * scale;
  const cy = (raw.naturalHeight / 2 + yShift) * scale;
  const r = radius * scale;
  mask.style.clipPath = `circle(${r}px at ${cx}px ${cy}px)`;
}

// Centroids of every preview cell, fetched once per mask version. Used to
// count "cells inside the ROI" locally so dragging the circle no longer
// round-trips to the server on every move.
let roiCentroidsCache = null;  // { mask_version, image_size:[w,h], centroids:[{id,cx,cy}] }

async function ensureRoiCentroids() {
  if (roiCentroidsCache && roiCentroidsCache.mask_version === currentMaskVersion) {
    return roiCentroidsCache;
  }
  try {
    const res = await jget("/api/cellpose/centroids");
    if (!res.ok) return null;
    roiCentroidsCache = {
      mask_version: res.mask_version,
      image_size: res.image_size,
      centroids: res.centroids,
    };
    if (res.mask_version && res.mask_version !== currentMaskVersion) {
      currentMaskVersion = res.mask_version;
    }
    return roiCentroidsCache;
  } catch (e) {
    return null;
  }
}

function countCellsInRoiLocal(radius, xShift, yShift) {
  if (!roiCentroidsCache) return null;
  const [w, h] = roiCentroidsCache.image_size;
  const cx = w / 2 + xShift;
  const cy = h / 2 + yShift;
  const r2 = radius * radius;
  let n = 0;
  for (const c of roiCentroidsCache.centroids) {
    const dx = c.cx - cx;
    const dy = c.cy - cy;
    if (dx * dx + dy * dy <= r2) n += 1;
  }
  return n;
}

// Persist ROI params + last_roi_count to the session — debounced so the
// session isn't slammed during a drag. The UI count itself is already
// updated immediately by the local counter.
const persistRoiToSession = debounce((radius, xShift, yShift, nInside) => {
  patchSession({
    radius, y_shift: yShift, x_shift: xShift, last_roi_count: nInside,
  });
}, 250);

function updateRoiCountLive(radius, xShift, yShift) {
  const n = countCellsInRoiLocal(radius, xShift, yShift);
  if (n == null) {
    // Centroids haven't loaded yet — kick off a fetch and show a placeholder.
    $("roi-count").textContent = "…";
    ensureRoiCentroids().then(() => updateRoiCountLive(radius, xShift, yShift));
    return;
  }
  $("roi-count").textContent = n;
  if (session) session.last_roi_count = n;
  $("rail-cells").textContent = n;
  persistRoiToSession(radius, xShift, yShift, n);
}

function setRoiEnabledUi(enabled) {
  const wrap = $("roi-controls-wrap");
  wrap.style.opacity = enabled ? "1" : "0.45";
  wrap.style.pointerEvents = enabled ? "auto" : "none";
  $("roi-circle").style.display = enabled ? "" : "none";
  $("roi-handle").style.display = enabled ? "" : "none";
  if (enabled) {
    const r = parseInt($("roi-radius").value, 10);
    const xs = parseInt($("roi-x").value, 10);
    const ys = parseInt($("roi-y").value, 10);
    applyRoiClip(r, xs, ys);
    updateRoiCountLive(r, xs, ys);
  } else {
    // Show every cell — no clipping
    $("roi-img-mask").style.clipPath = "none";
    $("roi-count").textContent = "all";
  }
}

async function setupRoiTab() {
  if (!session.all_frames?.length) return;
  $("roi-img-raw").src = frameUrl(0);
  await new Promise((res, rej) => {
    const im = $("roi-img-raw");
    if (im.complete && im.naturalWidth) return res();
    im.onload = res; im.onerror = rej;
  });
  // Reuse the already-rendered cellpose preview overlay rather than asking
  // the server to recolor a filtered mask on every interaction.
  const m = $("roi-img-mask");
  m.onerror = () => { m.style.display = "none"; };
  m.onload = () => { m.style.display = "block"; };
  m.src = `/api/mask/preview.png?v=${currentMaskVersion}`;
  const w = $("roi-img-raw").naturalWidth;
  const h = $("roi-img-raw").naturalHeight;
  if (!roiInited) {
    RoiCanvas.init(
      $("roi-canvas-wrap"),
      $("roi-img-raw"),
      $("roi-circle"),
      $("roi-handle"),
      (s) => {
        $("roi-radius").value = s.radius;
        $("roi-y").value = s.y_shift;
        $("roi-x").value = s.x_shift;
        applyRoiClip(s.radius, s.x_shift, s.y_shift);
        debouncedRoiCount(s.radius, s.x_shift, s.y_shift);
      },
    );
    roiInited = true;
  }
  RoiCanvas.setImage(w, h);
  RoiCanvas.setState(session.radius, session.x_shift, session.y_shift);
  $("roi-placeholder").style.display = "none";
  $("roi-enable").checked = session.roi_enabled !== false;
  // Warm the centroids cache so the very first interaction renders an
  // accurate count instantly instead of showing "…" while it fetches.
  await ensureRoiCentroids();
  setRoiEnabledUi($("roi-enable").checked);
}

$("roi-enable").addEventListener("change", async () => {
  const enabled = $("roi-enable").checked;
  setRoiEnabledUi(enabled);
  await patchSession({ roi_enabled: enabled });
  refreshValidation();
});

["roi-radius", "roi-y", "roi-x"].forEach(id => {
  $(id).addEventListener("change", () => {
    if (!$("roi-enable").checked) return;
    const r = parseInt($("roi-radius").value, 10);
    const xs = parseInt($("roi-x").value, 10);
    const ys = parseInt($("roi-y").value, 10);
    RoiCanvas.setState(r, xs, ys);
    applyRoiClip(r, xs, ys);
    updateRoiCountLive(r, xs, ys);
  });
});

window.addEventListener("resize", () => {
  if (!$("roi-enable")?.checked) return;
  const r = parseInt($("roi-radius").value, 10);
  const xs = parseInt($("roi-x").value, 10);
  const ys = parseInt($("roi-y").value, 10);
  applyRoiClip(r, xs, ys);
});

// ═══════════════════════════════════════════════════════════════════════════
// Tab 6 · Duplicate
// ═══════════════════════════════════════════════════════════════════════════
$("btn-dup-run").addEventListener("click", async () => {
  const mode = $("dup-mode").value;
  const source = $("dup-source").value;
  const res = await jpost("/api/duplicate/run", { mode, source });
  if (!res.ok) { alert(res.error); return; }
  pollDupStatus();
});

async function pollDupStatus() {
  const st = await jget("/api/duplicate/status");
  const pct = st.total ? (100 * st.progress / st.total) : 0;
  $("dup-fill").style.width = pct + "%";
  $("dup-label").textContent = `${st.message} · ${st.progress}/${st.total}`;
  if (st.state === "running") setTimeout(pollDupStatus, 500);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 7 · Run & Monitor
// ═══════════════════════════════════════════════════════════════════════════
function refreshSummary() {
  if (!session) return;
  const s = session;
  const mode = $("run-mode").value;
  const lines = [
    `Experiment: ${s.global_dir || "not selected"}`,
    `Run mode: ${mode.replace("_", " ")}`,
    "",
    `Segmentation: flow ${s.flow_threshold}, cell confidence ${s.cellprob_threshold}, niter ${s.niter}, cell size ${s.diameter}`,
    `Tracking: max distance ${s.max_distance?.toFixed?.(1)} px, grace period ${s.grace_period}, checkpoint every ${s.save_interval} frames`,
    `Frame shift: frame ${s.shift_frame_idx}, shift_xy ${s.shift_xy[0]} ${s.shift_xy[1]}`,
    `ROI: radius ${s.radius}, y_shift ${s.y_shift}, x_shift ${s.x_shift}, cells inside ${s.last_roi_count || 0}`,
    `Review: segmentation ${s.segmentation_reviewed ? "confirmed" : "not confirmed"}`,
  ];
  $("run-summary").textContent = lines.join("\n");
}

$("btn-copy-summary").addEventListener("click", () => {
  navigator.clipboard.writeText($("run-summary").textContent);
});
$("btn-preview-script").addEventListener("click", async () => {
  const mode = $("run-mode").value;
  const kind = mode === "post" ? "run_post_processes" : "run_processes";
  const res = await jget(`/api/script/preview?kind=${kind}&run_mode=${mode}`);
  if (!res.ok) { alert(res.error); return; }
  const w = window.open("", "_blank");
  w.document.write(`<pre style="font-family:monospace;padding:20px">${
    res.script.replace(/</g, "&lt;")}</pre>`);
});

$("btn-run-pipeline").addEventListener("click", () => runSelectedMode());
$("run-mode").addEventListener("change", async () => {
  await patchSession({ last_run_mode: $("run-mode").value });
  refreshSummary();
  refreshValidation();
});
$("btn-refresh-validation").addEventListener("click", refreshValidation);

async function refreshValidation() {
  if (!$("validation-list")) return;
  const mode = $("run-mode")?.value || "full";
  try {
    const res = await jget(`/api/validation?mode=${mode}`);
    const list = $("validation-list");
    list.innerHTML = "";
    for (const c of res.checks) {
      const row = document.createElement("div");
      row.className = "validation-row " + (c.ok ? "ok" : "bad");
      row.innerHTML = `<span>${c.ok ? "✓" : "!"}</span><div><strong>${c.label}</strong><small>${c.detail}</small></div>`;
      list.appendChild(row);
    }
    $("validation-hint").textContent = res.ok
      ? "Everything important is ready. You can run the selected mode."
      : "Resolve the highlighted checks before launching the pipeline.";
    $("btn-run-pipeline").disabled = !res.ok || pipelineRunning;
  } catch (e) {
    $("validation-list").innerHTML = "";
    $("validation-hint").textContent = "Validation is unavailable until an experiment is loaded.";
  }
}

function runSelectedMode() {
  const mode = $("run-mode").value;
  if (mode === "preview_only") return runPipeline("run_processes", "preview_only");
  if (mode === "post") return runPipeline("run_post_processes", "post");
  return runPipeline("run_processes", mode);
}

async function runPipeline(kind, runMode) {
  const body = { kind, run_mode: runMode };
  if (kind === "run_post_processes") {
    body.f0_frame = parseInt($("pa-f0").value, 10);
    body.stim_frames = $("pa-stim").value;
  }
  const res = await jpost("/api/pipeline/run", body);
  if (!res.ok) {
    if (res.validation) refreshValidation();
    alert(res.error || "Could not start pipeline");
    return;
  }
  pipelineRunning = true;
  logPos = 0;
  $("btn-stop-pipeline").disabled = false;
  $("btn-run-pipeline").disabled = true;
  $("live-log").textContent = "";
}

$("btn-stop-pipeline").addEventListener("click", async () => {
  await jpost("/api/pipeline/stop");
});

$("btn-clear-log").addEventListener("click", () => {
  $("live-log").textContent = "";
});

// ── SSE: pipeline status/log/progress stream ────────────────────────────
// One persistent EventSource replaces the previous 1s setInterval poll for
// status + log + progress. The server pushes events only when something
// changes, so the UI reacts as fast as the subprocess actually moves.
let pipelineES = null;
let lastPipelinePid = null;
let lastUptimeUpdateAt = 0;
let lastUptimeSec = 0;

function openPipelineStream() {
  if (pipelineES) return;
  pipelineES = new EventSource("/api/pipeline/stream");
  pipelineES.addEventListener("status", (e) => {
    let st; try { st = JSON.parse(e.data); } catch { return; }
    handlePipelineStatus(st);
  });
  pipelineES.addEventListener("log", (e) => {
    let lg; try { lg = JSON.parse(e.data); } catch { return; }
    if (lg.pos != null) logPos = lg.pos;
    if (lg.text) appendLog(lg.text);
  });
  pipelineES.addEventListener("progress", (e) => {
    let pr; try { pr = JSON.parse(e.data); } catch { return; }
    handlePipelineProgress(pr);
  });
  pipelineES.onerror = () => { /* browser auto-reconnects */ };
}

function handlePipelineStatus(st) {
  const pill = $("top-pill");
  const pillText = $("top-pill-text");
  if (st.running) {
    pill.className = "pill running";
    pillText.textContent = `running · ${st.stage ?? "…"}`;
    pipelineRunning = true;
    $("btn-stop-pipeline").disabled = false;
    $("btn-run-pipeline").disabled = true;
  } else {
    pill.className = "pill " + (st.exit_code && st.exit_code !== 0 ? "error" : "stopped");
    pillText.textContent = st.exit_code != null ? `exit ${st.exit_code}` : "idle";
    if (pipelineRunning) {
      pipelineRunning = false;
      $("btn-stop-pipeline").disabled = true;
      $("btn-run-pipeline").disabled = false;
      refreshValidation();
    }
  }
  // If a new pid appeared (new run started), clear stale local log buffer
  // so the streamed-from-zero content replaces it cleanly.
  if (st.pid && st.pid !== lastPipelinePid) {
    if (lastPipelinePid != null) {
      $("live-log").textContent = "";
      logPos = 0;
    }
    lastPipelinePid = st.pid;
  }
  $("rail-status").textContent = st.running ? "running" : "idle";
  $("rail-stage").textContent  = st.stage || "—";
  $("rail-pid").textContent    = st.pid || "—";
  lastUptimeSec = (st.running && st.uptime) ? st.uptime : 0;
  lastUptimeUpdateAt = performance.now();
  $("rail-uptime").textContent = lastUptimeSec ? formatUptime(lastUptimeSec) : "—";
}

function handlePipelineProgress(pr) {
  updateStage("seg", pr.segmentation.pct, `${pr.segmentation.done}/${pr.segmentation.total}`);
  updateStage("traj", pr.trajectories.pct,
              pr.trajectories.pct === 100 ? "complete"
              : pr.trajectories.pct > 0 ? "in progress" : "—");
  updateStage("pre", pr.pre_analysis.pct, pr.pre_analysis.pct === 100 ? "complete" : "—");
  updateStage("post", pr.post_analysis.pct, pr.post_analysis.pct === 100 ? "complete" : "—");
}

// Tick uptime locally — the server only re-emits status on real changes,
// but the user expects the seconds counter to tick continuously.
function tickUptime() {
  if (!pipelineRunning || !lastUptimeUpdateAt) return;
  const delta = (performance.now() - lastUptimeUpdateAt) / 1000;
  $("rail-uptime").textContent = formatUptime(lastUptimeSec + delta);
}

function formatUptime(sec) {
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${ss}s`;
  return `${ss}s`;
}

function updateStage(key, pct, meta) {
  const fill = $(`stage-${key}`);
  const metaEl = $(`stage-${key}-meta`);
  const stage = fill.closest(".stage");
  fill.style.width = `${pct}%`;
  metaEl.textContent = meta;
  stage.classList.toggle("done", pct >= 100);
  stage.classList.toggle("active", pct > 0 && pct < 100);
}

function appendLog(text) {
  const pre = $("live-log");
  pre.textContent += text;
  pre.scrollTop = pre.scrollHeight;
}

// Luminosity plot refresh (slower cadence)
async function pollLuminosity() {
  if (pipelineRunning) {
    $("lumi-img").src = `/api/pipeline/luminosity.png?refresh=${Math.floor(Date.now() / 5000)}`;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────
(async () => {
  await loadSession();
  await loadExperimentList();
  refreshSummary();
  openCellposeStream();
  openPipelineStream();
  setInterval(tickUptime, 1000);
  setInterval(pollLuminosity, 5000);
})();

})();
