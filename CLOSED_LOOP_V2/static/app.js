// =============================================================================
// Closed-Loop Bio-Control Hub — web GUI client (V6.1)
// Flask backend endpoints unchanged; this drives the new split-layout UI.
// =============================================================================

const $ = (id) => document.getElementById(id);

// -----------------------------------------------------------------------------
// Tweaks (theme / layout / accent) — persists via Composer host when hosted,
// falls back to localStorage when running under Flask.
// -----------------------------------------------------------------------------
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light",
  "layout": "split",
  "accent": "teal"
}/*EDITMODE-END*/;

function loadTweaks() {
  try {
    const stored = localStorage.getItem('biohub.tweaks');
    if (stored) return { ...TWEAK_DEFAULTS, ...JSON.parse(stored) };
  } catch (e) {}
  return { ...TWEAK_DEFAULTS };
}

const tweakState = loadTweaks();

function persistTweaks() {
  try { localStorage.setItem('biohub.tweaks', JSON.stringify(tweakState)); } catch (e) {}
  try {
    window.parent.postMessage(
      { type: '__edit_mode_set_keys', edits: { ...tweakState } }, '*'
    );
  } catch (e) {}
}

function applyTweaks() {
  document.body.setAttribute('data-theme', tweakState.theme);
  document.body.setAttribute('data-layout', tweakState.layout);
  document.body.setAttribute('data-accent', tweakState.accent);
  document.querySelectorAll('.seg-group').forEach(g => {
    const k = g.dataset.key;
    g.querySelectorAll('button').forEach(b =>
      b.classList.toggle('on', b.dataset.val === tweakState[k]));
  });
  document.querySelectorAll('.swatch-row').forEach(r => {
    const k = r.dataset.key;
    r.querySelectorAll('.swatch').forEach(s =>
      s.classList.toggle('on', s.dataset.val === tweakState[k]));
  });
}

function setTweak(key, val) {
  tweakState[key] = val;
  applyTweaks();
  persistTweaks();
}

document.querySelectorAll('.seg-group').forEach(g => {
  g.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    setTweak(g.dataset.key, b.dataset.val);
  });
});
document.querySelectorAll('.swatch-row').forEach(r => {
  r.addEventListener('click', e => {
    const s = e.target.closest('.swatch'); if (!s) return;
    setTweak(r.dataset.key, s.dataset.val);
  });
});

const tweaksEl = $('tweaks');
$('btn-tweaks').addEventListener('click', () => tweaksEl.classList.toggle('on'));
$('btn-tweaks-close').addEventListener('click', () => tweaksEl.classList.remove('on'));
$('btn-theme').addEventListener('click', () =>
  setTweak('theme', tweakState.theme === 'light' ? 'dark' : 'light'));

window.addEventListener('message', (e) => {
  const d = e.data || {};
  if (d.type === '__activate_edit_mode') tweaksEl.classList.add('on');
  if (d.type === '__deactivate_edit_mode') tweaksEl.classList.remove('on');
});
try { window.parent.postMessage({ type: '__edit_mode_available' }, '*'); } catch (e) {}

applyTweaks();

// -----------------------------------------------------------------------------
// Tab switching
// -----------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('panel-' + btn.dataset.tab).classList.add('active');
  });
});

// -----------------------------------------------------------------------------
// Pipeline diagram: cycle an "active" stage for the aura effect while running
// -----------------------------------------------------------------------------
const stageIds = ['watch', 'segment', 'decide', 'onix', 'media'];
let stageIdx = 0;
let pipelineRunning = false;

function advanceStage() {
  if (!pipelineRunning) {
    document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
    return;
  }
  document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
  const s = document.querySelector(`.stage[data-stage="${stageIds[stageIdx]}"]`);
  if (s) s.classList.add('active');
  stageIdx = (stageIdx + 1) % stageIds.length;
}
setInterval(advanceStage, 1100);

// -----------------------------------------------------------------------------
// Start / Stop buttons + status pill (driven by real pipeline status poll)
// -----------------------------------------------------------------------------
const btnStart = $('btn-start');
const btnStop = $('btn-stop');
const topPill = $('top-pill');
const topPillText = $('top-pill-text');

function setPipelineUi(state) {
  // state: 'running' | 'stopped' | 'exited'
  if (state === 'running') {
    pipelineRunning = true;
    btnStart.classList.add('btn-success');
    btnStart.disabled = true;
    btnStop.classList.add('btn-danger');
    btnStop.disabled = false;
    topPill.classList.remove('stopped');
    topPill.classList.add('running');
  } else {
    pipelineRunning = false;
    btnStart.classList.add('btn-success');
    btnStart.disabled = false;
    btnStop.classList.remove('btn-danger');
    btnStop.disabled = true;
    topPill.classList.remove('running');
    topPill.classList.add('stopped');
  }
}

btnStart.addEventListener('click', async () => {
  // Save config first (matching original app.js behavior)
  try {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectConfig()),
    });
  } catch (e) {}
  try {
    const r = await fetch('/api/pipeline/start', { method: 'POST' });
    const data = await r.json();
    if (!data.ok && data.error !== 'already running') {
      flashStatus($('save-status'), `Start failed: ${data.error}`);
    }
  } catch (e) {
    flashStatus($('save-status'), `Start error: ${e}`);
  }
});

btnStop.addEventListener('click', async () => {
  try { await fetch('/api/pipeline/stop', { method: 'POST' }); } catch (e) {}
});

// -----------------------------------------------------------------------------
// Config form round-trip
// -----------------------------------------------------------------------------
const CONFIG_FIELDS = [
  ['global_path',            'cfg-global-path',      'text'],
  ['num_channels',           'cfg-num-channels',     'int'],
  ['continuous_segmentation','cfg-continuous-seg',   'bool'],
  ['threshold_ratio',        'cfg-threshold-ratio',  'float'],
  ['num_tries',              'cfg-num-tries',        'int'],
  ['sleep_time',             'cfg-sleep-time',       'float'],
  ['run_duration_sec',       'cfg-run-duration',     'int'],
  ['acidic_pulse_sec',       'cfg-acidic-pulse',     'int'],
  ['onix_server_ip',         'cfg-onix-ip',          'text'],
  ['onix_server_port',       'cfg-onix-port',        'int'],
  ['retention_time_hours',   'cfg-retention-hours',  'int'],
  ['cleanup_interval_sec',   'cfg-cleanup-interval', 'int'],
];

let numChannels = 2;      // updated from config
let pulseDuration = 30;   // updated from config

async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const cfg = await r.json();
    for (const [key, id, type] of CONFIG_FIELDS) {
      const el = $(id); if (!el) continue;
      if (type === 'bool') el.checked = !!cfg[key];
      else el.value = cfg[key] != null ? cfg[key] : '';
    }
    numChannels = Number(cfg.num_channels) || 2;
    pulseDuration = Number(cfg.acidic_pulse_sec) || 30;
    ensureChannelsState();
  } catch (e) {
    flashStatus($('save-status'), `Config load failed: ${e}`);
  }
}

function collectConfig() {
  const body = {};
  for (const [key, id, type] of CONFIG_FIELDS) {
    const el = $(id); if (!el) continue;
    if (type === 'bool') body[key] = el.checked;
    else if (type === 'int') body[key] = parseInt(el.value, 10);
    else if (type === 'float') body[key] = parseFloat(el.value);
    else body[key] = el.value;
  }
  return body;
}

$('btn-save').addEventListener('click', async () => {
  try {
    const r = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectConfig()),
    });
    const data = await r.json();
    if (data.ok) {
      flashStatus($('save-status'), `Saved · ${new Date().toLocaleTimeString()}`);
      numChannels = Number(data.config.num_channels) || 2;
      pulseDuration = Number(data.config.acidic_pulse_sec) || 30;
      ensureChannelsState();
      loadSetpoints(); // num_channels may have changed
    } else {
      flashStatus($('save-status'), `Save failed: ${data.error || 'unknown'}`);
    }
  } catch (e) {
    flashStatus($('save-status'), `Save error: ${e}`);
  }
});

function flashStatus(el, text) {
  if (!el) return;
  el.textContent = text;
  clearTimeout(el._flashT);
  el._flashT = setTimeout(() => { if (el.textContent === text) el.textContent = ''; }, 4000);
}

// -----------------------------------------------------------------------------
// Pipeline status polling
// -----------------------------------------------------------------------------
async function pollPipelineStatus() {
  try {
    const r = await fetch('/api/pipeline/status');
    const s = await r.json();
    if (s.running) {
      setPipelineUi('running');
      topPillText.textContent = `Running · pid ${s.pid}`;
    } else if (s.exit_code != null) {
      setPipelineUi('stopped');
      topPillText.textContent = `Exited · code ${s.exit_code}`;
    } else {
      setPipelineUi('stopped');
      topPillText.textContent = 'Not running';
    }
  } catch (e) { /* network blip */ }
}

// -----------------------------------------------------------------------------
// Live log tail (incremental)
// -----------------------------------------------------------------------------
let logPos = 0;
let logCleared = false;
let logLineCount = 0;
let logLineStart = Date.now();

async function pollLog() {
  if (logCleared) return;
  try {
    const r = await fetch(`/api/log/tail?pos=${logPos}`);
    const data = await r.json();
    if (data.pos != null) logPos = data.pos;
    if (data.text) {
      appendLog(data.text);
      const newLines = (data.text.match(/\n/g) || []).length;
      logLineCount += newLines;
      const elapsed = Math.max(1, (Date.now() - logLineStart) / 1000);
      const rate = (logLineCount / elapsed).toFixed(1);
      $('log-rate').textContent = `${rate} lines/s`;
    }
  } catch (e) { /* silent */ }
}

// Colorize log lines client-side: INFO / WARN / ERROR etc.
function colorizeLogLine(line) {
  const m = line.match(/^(\S+\s+\S+)\s+\[(\w+)\]\s+(.*)$/);
  if (!m) return escapeHtml(line);
  const [, ts, lvl, rest] = m;
  const cls = {
    INFO: 'lvl-info', WARNING: 'lvl-warn', WARN: 'lvl-warn',
    ERROR: 'lvl-err', ERR: 'lvl-err', OK: 'lvl-ok', DEBUG: 'lvl-debug',
  }[lvl.toUpperCase()] || 'lvl-info';
  return `<span class="ts">${escapeHtml(ts)}</span> <span class="${cls}">[${escapeHtml(lvl)}]</span> ${escapeHtml(rest)}`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function appendLog(text) {
  const liveBox = $('live-log');
  const railBox = $('rail-log');
  const lines = text.split(/\r?\n/).filter(Boolean);
  const html = lines.map(colorizeLogLine).join('\n') + '\n';
  if (liveBox) {
    liveBox.insertAdjacentHTML('beforeend', html);
    // Cap the live log to last 500 lines to keep the DOM small.
    const maxLines = 500;
    const entries = liveBox.innerHTML.split('\n');
    if (entries.length > maxLines) {
      liveBox.innerHTML = entries.slice(entries.length - maxLines).join('\n');
    }
    liveBox.scrollTop = liveBox.scrollHeight;
  }
  if (railBox) {
    railBox.insertAdjacentHTML('beforeend', html);
    const entries = railBox.innerHTML.split('\n');
    if (entries.length > 80) {
      railBox.innerHTML = entries.slice(entries.length - 80).join('\n');
    }
    railBox.scrollTop = railBox.scrollHeight;
  }
}

$('btn-clear-log').addEventListener('click', () => {
  $('live-log').textContent = '';
  $('rail-log').textContent = '';
  logLineCount = 0;
  logLineStart = Date.now();
});

// -----------------------------------------------------------------------------
// Channels (media state from /api/media-status)
// -----------------------------------------------------------------------------
let channelsState = []; // [{ id, state, pulseStart }]

function ensureChannelsState() {
  // Resize channelsState to match numChannels without losing timing info
  const existing = new Map(channelsState.map(c => [c.id, c]));
  channelsState = [];
  for (let i = 1; i <= numChannels; i++) {
    channelsState.push(existing.get(i) || { id: i, state: 'neutral', pulseStart: null });
  }
  renderChannels();
}

async function pollMediaStatus() {
  try {
    const r = await fetch('/api/media-status');
    const data = await r.json();
    const channels = data.channels || {};
    pulseDuration = data.pulse_duration || pulseDuration;
    const serverTime = data._server_time || (Date.now() / 1000);
    const skew = serverTime - (Date.now() / 1000);
    for (const ch of channelsState) {
      const chData = channels[String(ch.id)];
      if (!chData) {
        ch.state = 'neutral'; ch.pulseStart = null;
        continue;
      }
      ch.state = chData.state || 'neutral';
      // Convert pulse_start (server wall-clock) to local wall-clock
      ch.pulseStart = chData.pulse_start != null
        ? chData.pulse_start - skew
        : null;
    }
    if (data.experiment) {
      const el = $('rail-exp');
      if (el) el.textContent = `exp ${data.experiment}`;
    }
    renderChannels();
  } catch (e) { /* silent */ }
}

function renderChannels() {
  for (const containerId of ['channels-small', 'channels-large']) {
    const wrap = $(containerId);
    if (!wrap) continue;
    wrap.innerHTML = '';
    for (const ch of channelsState) {
      wrap.appendChild(renderChannelCard(ch));
    }
  }
}

function renderChannelCard(ch) {
  const card = document.createElement('div');
  card.className = `channel-card ${ch.state}`;
  let ringPct = 1, primary = 'Neutral', secondary = 'flowing';
  if (ch.state === 'acidic' && ch.pulseStart != null) {
    const elapsed = (Date.now() / 1000) - ch.pulseStart;
    const remaining = Math.max(0, pulseDuration - elapsed);
    ringPct = Math.min(1, elapsed / pulseDuration);
    primary = Math.ceil(remaining) + 's';
    secondary = `pulse · ${Math.ceil(elapsed)}/${pulseDuration}s`;
  } else if (ch.state === 'acidic') {
    primary = 'Acidic';
    secondary = 'pulse · (unknown start)';
  }
  const r = 28, c = 2 * Math.PI * r;
  const dashoffset = c * (1 - ringPct);
  card.innerHTML = `
    <div class="channel-body">
      <div class="ring-wrap">
        <svg class="ring" viewBox="0 0 72 72">
          <circle class="ring-bg" cx="36" cy="36" r="28"/>
          <circle class="ring-fg" cx="36" cy="36" r="28"
            stroke-dasharray="${c}" stroke-dashoffset="${dashoffset}"/>
        </svg>
      </div>
      <div class="channel-text">
        <div class="channel-primary">${primary}</div>
        <div class="channel-secondary">${secondary}</div>
      </div>
    </div>
    <div class="channel-foot">
      <div class="channel-name">CH${ch.id}</div>
      <div class="channel-state">${ch.state === 'acidic' ? 'Acidic' : 'Neutral'}</div>
    </div>
  `;
  return card;
}

// Re-render channels every 500ms so the countdown ticks smoothly between polls.
setInterval(renderChannels, 500);

// -----------------------------------------------------------------------------
// Timeline (recent acidic pulse history)
// -----------------------------------------------------------------------------
// We record channel state transitions from polling and plot them back 30min.
const stateHistory = new Map(); // channel id -> [{ t, state }]

function pushStateHistory(ch) {
  const h = stateHistory.get(ch.id) || [];
  const last = h[h.length - 1];
  if (!last || last.state !== ch.state) {
    h.push({ t: Date.now() / 1000, state: ch.state });
    stateHistory.set(ch.id, h);
  }
  // Prune entries older than 30 minutes
  const cutoff = Date.now() / 1000 - 30 * 60;
  stateHistory.set(ch.id, h.filter(e => e.t >= cutoff));
}

function renderTimeline(containerId, chId) {
  const wrap = $(containerId);
  if (!wrap) return;
  wrap.innerHTML = '';
  const N = 60;
  const now = Date.now() / 1000;
  const start = now - 30 * 60;
  const history = stateHistory.get(chId) || [];
  for (let i = 0; i < N; i++) {
    const bucketT = start + (i / N) * 30 * 60;
    const seg = document.createElement('div');
    seg.className = 'timeline-seg';
    // Find state at bucketT: last transition before or equal to bucketT
    let state = 'idle';
    for (const e of history) {
      if (e.t <= bucketT) state = e.state;
      else break;
    }
    if (state === 'acidic') seg.classList.add('acidic');
    else if (state === 'neutral') { /* default styling */ }
    else seg.classList.add('idle');
    wrap.appendChild(seg);
  }
}

// -----------------------------------------------------------------------------
// Luminosity traces (SVG). Two modes:
//   A) "live svg" — we read recent records from the JSON logs via a tiny endpoint
//      (not present in the backend yet) OR derive from media-status stub.
//   B) "rendered plot" — fall back to the /api/luminosity-plot.png.
// For V6.1 we default to mode B since that endpoint already exists, and offer
// a toggle to show the PNG inside the card. The inline SVGs render a live
// sparkline from recent media-status.
// -----------------------------------------------------------------------------
const recentLumi = new Map(); // ch id -> [value, ...] last N ticks

function pushLumi(ch, v) {
  const arr = recentLumi.get(ch) || [];
  arr.push(v);
  while (arr.length > 80) arr.shift();
  recentLumi.set(ch, arr);
}

function renderTrace(polyId, chId, setpoint) {
  const line = document.getElementById(polyId);
  if (!line) return;
  const arr = recentLumi.get(chId) || [];
  if (arr.length < 2) { line.setAttribute('points', ''); return; }
  const halfRange = 0.010;
  const pts = [];
  for (let i = 0; i < arr.length; i++) {
    const t = i / Math.max(1, arr.length - 1);
    const x = 44 + t * (590 - 44);
    const y = 60 - ((arr[i] - setpoint) / halfRange) * 30;
    pts.push(`${x.toFixed(1)},${Math.max(2, Math.min(118, y)).toFixed(1)}`);
  }
  line.setAttribute('points', pts.join(' '));
}

// Render a fallback image-mode toggle
let lumiImageMode = false;
$('btn-toggle-lumi-mode').addEventListener('click', () => {
  lumiImageMode = !lumiImageMode;
  $('btn-toggle-lumi-mode').textContent = lumiImageMode
    ? 'Use inline trace' : 'Use rendered plot';
  refreshLumi();
});
$('btn-refresh-lumi').addEventListener('click', refreshLumi);

function refreshLumi() {
  const panels = document.querySelectorAll('.lumi-panel');
  // Find or create a single <img id="lumi-image"> that replaces the inline SVGs
  // when in image mode.
  let container = document.getElementById('lumi-image-wrap');
  if (lumiImageMode) {
    panels.forEach(p => p.style.display = 'none');
    if (!container) {
      container = document.createElement('div');
      container.id = 'lumi-image-wrap';
      container.style.cssText = 'display:flex; justify-content:center;';
      container.innerHTML = `<img id="lumi-image" alt="Luminosity plot" style="max-width:100%; border:1px solid var(--border); border-radius: var(--radius); background: #fff;"/>`;
      panels[panels.length - 1].parentNode.appendChild(container);
    }
    container.style.display = 'flex';
    $('lumi-image').src = `/api/luminosity-plot.png?t=${Date.now()}`;
  } else {
    panels.forEach(p => p.style.display = '');
    if (container) container.style.display = 'none';
  }
}

// -----------------------------------------------------------------------------
// Setpoints (dynamic per num_channels)
// -----------------------------------------------------------------------------
async function loadSetpoints() {
  try {
    const r = await fetch('/api/setpoints');
    const data = await r.json();
    if (!data.ok) {
      flashStatus($('setpoint-status'), `Load failed: ${data.error || 'unknown'}`);
      return;
    }
    const rows = $('setpoint-rows');
    rows.innerHTML = '';
    for (const c of data.channels) {
      const field = document.createElement('div');
      field.className = 'field';
      field.innerHTML = `
        <label class="lbl" for="sp-ch-${c.channel}">
          Channel ${c.channel}
          <span class="help" data-tip="Reference luminosity for CH${c.channel}. Decisions fire when live luminosity deviates by more than threshold_ratio from this value.">?</span>
        </label>
        <input type="number" step="any" id="sp-ch-${c.channel}" data-channel="${c.channel}"
          value="${c.value != null ? Number(c.value).toFixed(6) : ''}"
          placeholder="${c.value == null ? 'not yet computed' : ''}">
      `;
      rows.appendChild(field);
    }
    flashStatus($('setpoint-status'),
      data.exists ? `Loaded · ${new Date().toLocaleTimeString()}` : `No file yet`);
  } catch (e) {
    flashStatus($('setpoint-status'), `Load error: ${e}`);
  }
}

$('btn-refresh-setpoints').addEventListener('click', loadSetpoints);
$('btn-save-setpoints').addEventListener('click', async () => {
  const channels = {};
  document.querySelectorAll('#setpoint-rows input').forEach(el => {
    const v = el.value.trim();
    if (v !== '') channels[el.dataset.channel] = parseFloat(v);
  });
  if (Object.keys(channels).length === 0) {
    flashStatus($('setpoint-status'), 'Nothing to save');
    return;
  }
  try {
    const r = await fetch('/api/setpoints', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channels }),
    });
    const data = await r.json();
    flashStatus($('setpoint-status'),
      data.ok ? `Saved · ${new Date().toLocaleTimeString()}`
              : `Save failed: ${data.error || 'unknown'}`);
  } catch (e) {
    flashStatus($('setpoint-status'), `Save error: ${e}`);
  }
});

// -----------------------------------------------------------------------------
// Segmentation tab
// -----------------------------------------------------------------------------
async function pollFrames() {
  // Only when the tab is visible
  if (!$('panel-segmentation').classList.contains('active')) return;
  try {
    const r = await fetch('/api/frames');
    const data = await r.json();
    const sel = $('seg-frame');
    const prev = sel.value;
    const frames = (data.frames || []).map(f => f.frame);
    const desired = frames.join(',');
    if (sel.dataset.last === desired) return;
    sel.dataset.last = desired;
    sel.innerHTML = '';
    for (const fr of frames) {
      const opt = document.createElement('option');
      opt.value = fr;
      opt.textContent = String(fr).padStart(4, '0');
      sel.appendChild(opt);
    }
    if (prev && frames.includes(parseInt(prev, 10))) sel.value = prev;
  } catch (e) { /* silent */ }
}

function setSegStatus(text, kind) {
  const el = $('seg-status');
  if (!el) return;
  el.textContent = text;
  el.className = 'seg-status-line' + (kind ? ' ' + kind : '');
}

$('btn-run-seg').addEventListener('click', async () => {
  const body = {
    frame: parseInt($('seg-frame').value || '0', 10),
    channel: parseInt($('seg-channel').value, 10),
    diameter: parseInt($('seg-diameter').value, 10),
    flow_threshold: parseFloat($('seg-flow').value),
    cellprob_threshold: parseFloat($('seg-cellprob').value),
    niter: parseInt($('seg-niter').value, 10),
  };
  setSegStatus('Segmenting…', 'running');
  try {
    const r = await fetch('/api/segmentation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!data.ok) {
      setSegStatus(data.error || 'Failed to start', 'error');
      return;
    }
    const poll = setInterval(async () => {
      try {
        const sr = await fetch('/api/segmentation/status');
        const s = await sr.json();
        setSegStatus(s.message || s.state, s.state);
        if (s.state === 'done') {
          clearInterval(poll);
          refreshPreview(body.frame, body.channel);
        } else if (s.state === 'error') {
          clearInterval(poll);
        }
      } catch (e) { /* silent */ }
    }, 500);
  } catch (e) {
    setSegStatus(`Run error: ${e}`, 'error');
  }
});

$('btn-load-frame').addEventListener('click', () => {
  const frame = parseInt($('seg-frame').value || '0', 10);
  const channel = parseInt($('seg-channel').value, 10);
  refreshPreview(frame, channel);
  setSegStatus(`Loaded ch${channel} frame ${frame}`, 'done');
});

$('btn-update-masks').addEventListener('click', async () => {
  const body = {
    frame: parseInt($('seg-frame').value || '0', 10),
    channel: parseInt($('seg-channel').value, 10),
  };
  try {
    const r = await fetch('/api/segmentation/update-masks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.ok) {
      setSegStatus(`Mask pushed → ${data.dst.split('/').pop()}`, 'done');
    } else {
      setSegStatus(data.error || 'Push failed', 'error');
    }
  } catch (e) {
    setSegStatus(`Push error: ${e}`, 'error');
  }
});

function refreshPreview(frame, channel) {
  const t = Date.now();
  const raw = $('preview-raw');
  const mask = $('preview-mask');
  const placeholder = $('seg-placeholder');
  const overlay = $('seg-overlay');
  raw.onload = () => { if (placeholder) placeholder.style.display = 'none'; };
  raw.onerror = () => { if (placeholder) placeholder.style.display = ''; };
  raw.src = `/api/frame/${frame}/${channel}.png?t=${t}`;
  mask.onerror = () => { mask.style.display = 'none'; };
  mask.onload = () => { mask.style.display = 'block'; };
  mask.src = `/api/mask/${frame}/${channel}.png?t=${t}`;
  if (overlay) overlay.textContent = `ch${channel} · frame ${String(frame).padStart(4,'0')}`;
}

// -----------------------------------------------------------------------------
// Rail sparkline (last ~60 luminosity samples; averaged across channels)
// -----------------------------------------------------------------------------
function renderRailSpark() {
  const line = $('rail-trace');
  if (!line) return;
  // Average recent luminosity across channels for a single summary spark.
  const chs = [...recentLumi.keys()];
  if (chs.length === 0) { line.setAttribute('points', ''); return; }
  const N = Math.max(...chs.map(c => (recentLumi.get(c) || []).length));
  if (N < 2) { line.setAttribute('points', ''); return; }
  const W = 300, H = 60;
  const pts = [];
  let min = Infinity, max = -Infinity;
  const avgArr = [];
  for (let i = 0; i < N; i++) {
    let sum = 0, n = 0;
    for (const c of chs) {
      const a = recentLumi.get(c);
      if (a && i < a.length) { sum += a[i]; n++; }
    }
    if (n === 0) continue;
    const v = sum / n;
    avgArr.push(v);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  for (let i = 0; i < avgArr.length; i++) {
    const t = i / Math.max(1, avgArr.length - 1);
    const nrm = (avgArr[i] - min) / range;
    pts.push(`${(t * W).toFixed(1)},${(H - nrm * (H - 6) - 3).toFixed(1)}`);
  }
  line.setAttribute('points', pts.join(' '));
}

// -----------------------------------------------------------------------------
// Poll loop
// -----------------------------------------------------------------------------
async function pollAll() {
  await Promise.all([
    pollPipelineStatus(),
    pollLog(),
    pollMediaStatus(),
    pollFrames(),
  ]);
  // Track state history for timeline
  for (const ch of channelsState) pushStateHistory(ch);
  renderTimeline('tl-ch1', 1);
  renderTimeline('tl-ch2', 2);
  renderRailSpark();
  // Setpoint values drive the luminosity trace baselines
  const sp1Input = document.getElementById('sp-ch-1');
  const sp2Input = document.getElementById('sp-ch-2');
  const sp1 = sp1Input ? parseFloat(sp1Input.value) || 0.185 : 0.185;
  const sp2 = sp2Input ? parseFloat(sp2Input.value) || 0.212 : 0.212;
  renderTrace('trace-ch1', 1, sp1);
  renderTrace('trace-ch2', 2, sp2);
}

setInterval(pollAll, 1000);

// -----------------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------------
(async () => {
  await loadConfig();
  await loadSetpoints();
  await pollPipelineStatus();
  await pollMediaStatus();
  await pollFrames();
})();
