/* Main UI + polling for the Preprocess web GUI. */

(() => {

const $  = (id) => document.getElementById(id);
const qs = (s)  => document.querySelector(s);
const qsa = (s) => Array.from(document.querySelectorAll(s));

let session = null;
let logPos = 0;
let pipelineRunning = false;
let currentExperimentPath = "";

// ── Tabs ────────────────────────────────────────────────────────────────
qsa(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    qsa(".tab-btn").forEach(b => b.classList.remove("active"));
    qsa(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`panel-${btn.dataset.tab}`).classList.add("active");
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
  if (tab === "shift")     setupShiftTab();
  if (tab === "maxdist")   renderMdHistogram();
  if (tab === "roi")       setupRoiTab();
  if (tab === "run")       refreshSummary();
  if (tab === "interval")  refreshInterval();
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
  currentExperimentPath  = session.global_dir || "";
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
  // Reset thumb + scrubber caches
  thumbsLoadedFor = null;
}

$("btn-exp-load").addEventListener("click", () => {
  const p = $("exp-path").value.trim();
  if (p) loadExperiment(p);
});

// ═══════════════════════════════════════════════════════════════════════════
// Tab 1 · Cellpose (frame scrubber + thumbs + preview)
// ═══════════════════════════════════════════════════════════════════════════
let thumbsLoadedFor = null;

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
  patchSession({ frame_idx: idx });
  highlightThumb(idx);
});
$("cp-frame").addEventListener("change", (e) => {
  const idx = parseInt(e.target.value, 10);
  $("cp-scrub").value = idx;
  updateScrubLabel();
  loadCellposeFrame(idx);
  patchSession({ frame_idx: idx });
  highlightThumb(idx);
});

function loadCellposeFrame(idx) {
  const img = $("cp-img-raw");
  img.src = `/api/frame/${idx}.png?_t=${Date.now()}`;
  img.style.display = "block";
  $("cp-placeholder").style.display = "none";
  // Clear mask overlay (it was for a different frame)
  $("cp-img-mask").style.display = "none";
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
    im.src = `/api/thumbnail/${i}.png?w=120`;
    im.dataset.idx = i;
    im.title = `frame ${i}`;
    im.addEventListener("click", () => {
      $("cp-scrub").value = i;
      $("cp-frame").value = i;
      updateScrubLabel();
      loadCellposeFrame(i);
      patchSession({ frame_idx: i });
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
  pollCellposeStatus();
});

async function pollCellposeStatus() {
  const st = await jget("/api/cellpose/status");
  $("cp-status").textContent = st.message || "";
  if (st.state === "done") {
    $("btn-run-cellpose").disabled = false;
    $("cp-cells").textContent = st.n_cells;
    const rt = (st.finished_at && st.started_at)
      ? `${(st.finished_at - st.started_at).toFixed(1)}s` : "—";
    $("cp-runtime").textContent = rt;
    cpHasMask = true;
    reloadMaskOverlay();
    $("cp-histogram").src = `/api/cellpose/stats.png?_t=${Date.now()}`;
    $("cp-histogram").style.display = "block";
    await loadSession();
    return;
  }
  if (st.state === "error") {
    $("btn-run-cellpose").disabled = false;
    return;
  }
  setTimeout(pollCellposeStatus, 800);
}

function reloadMaskOverlay() {
  const m = $("cp-img-mask");
  m.src = `/api/mask/preview.png?_t=${Date.now()}`;
  m.style.display = "block";
}

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
    await ShiftCanvas.loadFrames(idx);
  } catch (e) { alert(`failed: ${e}`); }
});

$("btn-sh-auto").addEventListener("click", async () => {
  const idx = parseInt($("sh-frame").value, 10);
  $("sh-result").textContent = "Auto-detecting…";
  const res = await jpost("/api/shift/auto", { idx });
  if (!res.ok) { $("sh-result").textContent = res.error; return; }
  $("sh-result").textContent = `shift_xy = (${res.dx}, ${res.dy})  [auto]`;
  await loadSession();
});

$("btn-sh-clear").addEventListener("click", () => {
  ShiftCanvas.clearPoints();
  $("sh-result").textContent = "shift_xy = (0, 0)";
  patchSession({ shift_xy: [0, 0] });
});

// ═══════════════════════════════════════════════════════════════════════════
// Tab 4 · Max Distance
// ═══════════════════════════════════════════════════════════════════════════
let mdPopulation = [];
let mdMean = 0;

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
    `median ${res.median.toFixed(1)} · 75p ${res.percentile_75.toFixed(1)} · 95p ${res.percentile_95.toFixed(1)}`;
  mdPopulation = res.population;
  mdMean = res.mean_distance;
  renderMdHistogram();
  await loadSession();
}

function renderMdHistogram() {
  const canvas = $("md-histogram");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#0f1419";
  ctx.fillRect(0, 0, w, h);
  if (!mdPopulation.length) return;
  const maxV = Math.max(...mdPopulation);
  const binN = 40;
  const bins = new Array(binN).fill(0);
  mdPopulation.forEach(v => {
    const bi = Math.min(binN - 1, Math.floor(v / maxV * binN));
    bins[bi]++;
  });
  const maxC = Math.max(...bins);
  ctx.fillStyle = "#4f9aa8";
  for (let i = 0; i < binN; i++) {
    const bx = i / binN * w;
    const bw = w / binN - 1;
    const bh = (bins[i] / maxC) * (h - 10);
    ctx.fillRect(bx, h - bh, bw, bh);
  }
  // Mean line
  const mx = mdMean / maxV * w;
  ctx.strokeStyle = "#d8a04a";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(mx, 0); ctx.lineTo(mx, h);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#d8a04a";
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.fillText(`mean = ${mdMean.toFixed(1)}`, mx + 4, 14);
}

$("btn-md-compute").addEventListener("click", () => mdCompute());
$("btn-md-resample").addEventListener("click", () => mdCompute());
$("md-value").addEventListener("change", () =>
  patchSession({ max_distance: parseFloat($("md-value").value) }));

// ═══════════════════════════════════════════════════════════════════════════
// Tab 5 · ROI
// ═══════════════════════════════════════════════════════════════════════════
let roiInited = false;

async function setupRoiTab() {
  if (!session.all_frames?.length) return;
  $("roi-img-raw").src = `/api/frame/0.png?_t=${Date.now()}`;
  await new Promise((res, rej) => {
    const im = $("roi-img-raw");
    if (im.complete && im.naturalWidth) return res();
    im.onload = res; im.onerror = rej;
  });
  const w = $("roi-img-raw").naturalWidth;
  const h = $("roi-img-raw").naturalHeight;
  if (!roiInited) {
    RoiCanvas.init(
      $("roi-canvas-wrap"),
      $("roi-img-raw"),
      $("roi-circle"),
      $("roi-handle"),
      async (s) => {
        $("roi-radius").value = s.radius;
        $("roi-y").value = s.y_shift;
        $("roi-x").value = s.x_shift;
        await roiUpdate(s);
      },
    );
    roiInited = true;
  }
  RoiCanvas.setImage(w, h);
  RoiCanvas.setState(session.radius, session.x_shift, session.y_shift);
  $("roi-placeholder").style.display = "none";
  roiUpdate();
}

async function roiUpdate(override) {
  const body = override || {
    radius: parseInt($("roi-radius").value, 10),
    y_shift: parseInt($("roi-y").value, 10),
    x_shift: parseInt($("roi-x").value, 10),
  };
  const res = await jpost("/api/roi/count", body);
  if (!res.ok) { $("roi-count").textContent = "—"; return; }
  $("roi-count").textContent = res.n_inside;
  $("roi-img-mask").src = `/api/roi/mask.png?_t=${Date.now()}`;
  $("roi-img-mask").style.display = "block";
}

$("btn-roi-show").addEventListener("click", () => setupRoiTab());
["roi-radius", "roi-y", "roi-x"].forEach(id => {
  $(id).addEventListener("change", () => {
    RoiCanvas.setState(
      parseInt($("roi-radius").value, 10),
      parseInt($("roi-x").value, 10),
      parseInt($("roi-y").value, 10),
    );
    roiUpdate();
  });
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
  const lines = [
    "═══════════════════════════════════════════",
    "  Pipeline Parameters",
    "═══════════════════════════════════════════",
    `  GLOBAL_DIR         = ${s.global_dir}`,
    `  flow_threshold     = ${s.flow_threshold}`,
    `  cellprob_threshold = ${s.cellprob_threshold}`,
    `  niter              = ${s.niter}`,
    `  diameter           = ${s.diameter}`,
    `  save_interval      = ${s.save_interval}`,
    `  shift_frame        = ${s.shift_frame_idx}`,
    `  shift_xy           = ${s.shift_xy[0]} ${s.shift_xy[1]}`,
    `  max_distance       = ${s.max_distance?.toFixed?.(1)}`,
    `  radius             = ${s.radius}`,
    `  y_shift            = ${s.y_shift}`,
    `  x_shift            = ${s.x_shift}`,
    `  grace_period       = ${s.grace_period}`,
    "═══════════════════════════════════════════",
  ];
  $("run-summary").textContent = lines.join("\n");
}

$("btn-copy-summary").addEventListener("click", () => {
  navigator.clipboard.writeText($("run-summary").textContent);
});
$("btn-preview-script").addEventListener("click", async () => {
  const res = await jget(`/api/script/preview?kind=run_processes`);
  if (!res.ok) { alert(res.error); return; }
  const w = window.open("", "_blank");
  w.document.write(`<pre style="font-family:monospace;padding:20px">${
    res.script.replace(/</g, "&lt;")}</pre>`);
});

$("btn-run-pipeline").addEventListener("click", () => runPipeline("run_processes"));
$("btn-run-post").addEventListener("click", () => runPipeline("run_post_processes"));

async function runPipeline(kind) {
  const body = { kind };
  if (kind === "run_post_processes") {
    body.f0_frame = parseInt($("pa-f0").value, 10);
    body.stim_frames = $("pa-stim").value;
  }
  const res = await jpost("/api/pipeline/run", body);
  if (!res.ok) { alert(res.error); return; }
  pipelineRunning = true;
  logPos = 0;
  $("btn-stop-pipeline").disabled = false;
  $("btn-run-pipeline").disabled = true;
  $("btn-run-post").disabled = true;
  $("live-log").textContent = "";
}

$("btn-stop-pipeline").addEventListener("click", async () => {
  await jpost("/api/pipeline/stop");
});

$("btn-clear-log").addEventListener("click", () => {
  $("live-log").textContent = "";
});

// ── Pipeline polling (always on, every 1s) ─────────────────────────────
async function pollPipeline() {
  try {
    const st = await jget("/api/pipeline/status");
    const pill = $("top-pill");
    const pillText = $("top-pill-text");
    if (st.running) {
      pill.className = "pill running";
      pillText.textContent = `running · ${st.stage ?? "…"}`;
      pipelineRunning = true;
    } else {
      pill.className = "pill " + (st.exit_code && st.exit_code !== 0 ? "error" : "stopped");
      pillText.textContent = st.exit_code != null
        ? `exit ${st.exit_code}`
        : "idle";
      if (pipelineRunning) {
        // Just finished
        pipelineRunning = false;
        $("btn-stop-pipeline").disabled = true;
        $("btn-run-pipeline").disabled = false;
        $("btn-run-post").disabled = false;
      }
    }
    $("rail-status").textContent = st.running ? "running" : "idle";
    $("rail-stage").textContent  = st.stage || "—";
    $("rail-uptime").textContent = st.running && st.uptime
      ? formatUptime(st.uptime) : "—";
    $("rail-pid").textContent    = st.pid || "—";

    // Log tail
    if (st.running || logPos === 0) {
      const lg = await jget(`/api/pipeline/log?pos=${logPos}`);
      if (lg.exists) {
        logPos = lg.pos;
        if (lg.text) appendLog(lg.text);
      }
    }

    // Progress
    const pr = await jget("/api/pipeline/progress");
    updateStage("seg", pr.segmentation.pct, `${pr.segmentation.done}/${pr.segmentation.total}`);
    updateStage("traj", pr.trajectories.pct, pr.trajectories.pct === 100 ? "complete" : pr.trajectories.pct > 0 ? "in progress" : "—");
    updateStage("pre", pr.pre_analysis.pct, pr.pre_analysis.pct === 100 ? "complete" : "—");
    updateStage("post", pr.post_analysis.pct, pr.post_analysis.pct === 100 ? "complete" : "—");
  } catch (e) {
    // ignore transient errors
  }
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
    $("lumi-img").src = `/api/pipeline/luminosity.png?_t=${Date.now()}`;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────
(async () => {
  await loadSession();
  await loadExperimentList();
  refreshSummary();
  setInterval(pollPipeline, 1000);
  setInterval(pollLuminosity, 5000);
})();

})();
