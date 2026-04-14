// Closed-Loop Bio-Control Hub — web GUI client
// Polls backend every 1s for pipeline status, log tail, and media status.

const $ = (id) => document.getElementById(id);

// ---------- Tab switching ----------
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ---------- Config form round-trip ----------
const CONFIG_FIELDS = [
  ['global_path', 'cfg-global-path', 'text'],
  ['num_channels', 'cfg-num-channels', 'int'],
  ['continuous_segmentation', 'cfg-continuous-seg', 'bool'],
  ['threshold_ratio', 'cfg-threshold-ratio', 'float'],
  ['num_tries', 'cfg-num-tries', 'int'],
  ['sleep_time', 'cfg-sleep-time', 'float'],
  ['run_duration_sec', 'cfg-run-duration', 'int'],
  ['acidic_pulse_sec', 'cfg-acidic-pulse', 'int'],
  ['onix_server_ip', 'cfg-onix-ip', 'text'],
  ['onix_server_port', 'cfg-onix-port', 'int'],
  ['retention_time_hours', 'cfg-retention-hours', 'int'],
  ['cleanup_interval_sec', 'cfg-cleanup-interval', 'int'],
];

async function loadConfig() {
  const r = await fetch('/api/config');
  const cfg = await r.json();
  for (const [key, id, type] of CONFIG_FIELDS) {
    const el = $(id);
    if (el == null) continue;
    if (type === 'bool') el.checked = !!cfg[key];
    else el.value = cfg[key] != null ? cfg[key] : '';
  }
  // Num channels dictates media labels
  renderMediaLabels(cfg.num_channels || 2);
}

function collectConfig() {
  const body = {};
  for (const [key, id, type] of CONFIG_FIELDS) {
    const el = $(id);
    if (el == null) continue;
    if (type === 'bool') body[key] = el.checked;
    else if (type === 'int') body[key] = parseInt(el.value, 10);
    else if (type === 'float') body[key] = parseFloat(el.value);
    else body[key] = el.value;
  }
  return body;
}

$('btn-save').addEventListener('click', async () => {
  const body = collectConfig();
  const r = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (data.ok) {
    $('config-status').textContent = `Saved to ${data.saved_to}`;
    renderMediaLabels(data.config.num_channels || 2);
  } else {
    $('config-status').textContent = `Save failed: ${data.error || 'unknown'}`;
  }
});

// ---------- Pipeline start/stop ----------
$('btn-start').addEventListener('click', async () => {
  // Save current config first, matching napari behavior
  await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(collectConfig()),
  });
  const r = await fetch('/api/pipeline/start', { method: 'POST' });
  const data = await r.json();
  if (!data.ok && data.error !== 'already running') {
    $('pipeline-status').textContent = `Start failed: ${data.error}`;
  }
});

$('btn-stop').addEventListener('click', async () => {
  await fetch('/api/pipeline/stop', { method: 'POST' });
});

// ---------- 1-second polling loop ----------
let logPos = 0;
let liveLogCleared = false;

async function pollPipelineStatus() {
  try {
    const r = await fetch('/api/pipeline/status');
    const s = await r.json();
    const box = $('pipeline-status');
    const pill = $('pipeline-pill');
    if (s.running) {
      box.textContent = `Pipeline: RUNNING  (pid ${s.pid})`;
      box.className = 'status-box running';
      pill.textContent = `Pipeline: RUNNING (pid ${s.pid})`;
      pill.className = 'pill pill-running';
    } else if (s.exit_code != null) {
      box.textContent = `Pipeline: exited  (code ${s.exit_code})`;
      box.className = 'status-box stopped';
      pill.textContent = `Pipeline: exited (${s.exit_code})`;
      pill.className = 'pill pill-stopped';
    } else {
      box.textContent = 'Pipeline: not running';
      box.className = 'status-box';
      pill.textContent = 'Pipeline: not running';
      pill.className = 'pill pill-idle';
    }
  } catch (e) { /* network blip */ }
}

async function pollLog() {
  if (liveLogCleared) return; // user can reset by reloading page
  try {
    const r = await fetch(`/api/log/tail?pos=${logPos}`);
    const data = await r.json();
    if (data.pos != null) logPos = data.pos;
    if (data.text) {
      const box = $('live-log');
      box.textContent += data.text;
      box.scrollTop = box.scrollHeight;
    }
  } catch (e) { /* network blip */ }
}

function renderMediaLabels(numCh) {
  const wrap = $('media-labels');
  wrap.innerHTML = '';
  for (let ch = 1; ch <= numCh; ch++) {
    const el = document.createElement('div');
    el.className = 'media-label';
    el.id = `media-ch-${ch}`;
    el.textContent = `CH${ch}: IDLE`;
    wrap.appendChild(el);
  }
}

async function pollMediaStatus() {
  try {
    const r = await fetch('/api/media-status');
    const data = await r.json();
    const channels = data.channels || {};
    const pulseDuration = data.pulse_duration || 30;
    const serverTime = data._server_time || (Date.now() / 1000);
    for (const [chStr, chData] of Object.entries(channels)) {
      const el = $(`media-ch-${chStr}`);
      if (!el) continue;
      const state = chData.state || 'neutral';
      if (state === 'acidic' && chData.pulse_start != null) {
        const remaining = Math.max(0, pulseDuration - (serverTime - chData.pulse_start));
        const secs = Math.floor(remaining);
        el.className = 'media-label acidic';
        el.textContent = remaining <= 0
          ? `CH${chStr}: ACIDIC — pulse complete, switching...`
          : `CH${chStr}: ACIDIC PULSE — ${secs}s remaining`;
      } else {
        el.className = 'media-label neutral';
        el.textContent = `CH${chStr}: NEUTRAL`;
      }
    }
    if (data.experiment) {
      $('experiment-label').textContent = `ONIX experiment: ${data.experiment}`;
    }
  } catch (e) { /* silent */ }
}

async function pollFrames() {
  // Only refresh when Segmentation tab is visible to save bandwidth
  if (!$('tab-segmentation').classList.contains('active')) return;
  try {
    const r = await fetch('/api/frames');
    const data = await r.json();
    const sel = $('seg-frame');
    const prev = sel.value;
    const frames = (data.frames || []).map((f) => f.frame);
    const desired = frames.join(',');
    if (sel.dataset.last === desired) return;
    sel.dataset.last = desired;
    sel.innerHTML = '';
    for (const fr of frames) {
      const opt = document.createElement('option');
      opt.value = fr;
      opt.textContent = fr;
      sel.appendChild(opt);
    }
    if (prev && frames.includes(parseInt(prev, 10))) sel.value = prev;
  } catch (e) { /* silent */ }
}

setInterval(() => {
  pollPipelineStatus();
  pollLog();
  pollMediaStatus();
  pollFrames();
}, 1000);

// ---------- Segmentation ----------
$('btn-run-seg').addEventListener('click', async () => {
  const body = {
    frame: parseInt($('seg-frame').value || '0', 10),
    channel: parseInt($('seg-channel').value, 10),
    diameter: parseInt($('seg-diameter').value, 10),
    flow_threshold: parseFloat($('seg-flow').value),
    cellprob_threshold: parseFloat($('seg-cellprob').value),
    niter: parseInt($('seg-niter').value, 10),
  };
  $('seg-status').textContent = 'Segmenting...';
  $('seg-status').className = 'status-box';
  const r = await fetch('/api/segmentation/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!data.ok) {
    $('seg-status').textContent = data.error || 'Failed to start';
    $('seg-status').className = 'status-box error';
    return;
  }
  // Poll segmentation status until done
  const poll = setInterval(async () => {
    const sr = await fetch('/api/segmentation/status');
    const s = await sr.json();
    $('seg-status').textContent = s.message || s.state;
    if (s.state === 'done') {
      $('seg-status').className = 'status-box running';
      clearInterval(poll);
      refreshPreview(body.frame, body.channel);
    } else if (s.state === 'error') {
      $('seg-status').className = 'status-box error';
      clearInterval(poll);
    }
  }, 500);
});

$('btn-load-frame').addEventListener('click', () => {
  const frame = parseInt($('seg-frame').value || '0', 10);
  const channel = parseInt($('seg-channel').value, 10);
  refreshPreview(frame, channel);
  $('seg-status').textContent = `Loaded ch${channel} frame ${frame}`;
});

$('btn-update-masks').addEventListener('click', async () => {
  const body = {
    frame: parseInt($('seg-frame').value || '0', 10),
    channel: parseInt($('seg-channel').value, 10),
  };
  const r = await fetch('/api/segmentation/update-masks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (data.ok) {
    $('seg-status').textContent = `Mask pushed: ${data.dst.split('/').pop()}`;
    $('seg-status').className = 'status-box running';
  } else {
    $('seg-status').textContent = data.error || 'Push failed';
    $('seg-status').className = 'status-box error';
  }
});

function refreshPreview(frame, channel) {
  // Cache-bust so segmentation results appear immediately
  const t = Date.now();
  $('preview-raw').src = `/api/frame/${frame}/${channel}.png?t=${t}`;
  const mask = $('preview-mask');
  mask.onerror = () => { mask.style.display = 'none'; };
  mask.onload = () => { mask.style.display = 'block'; };
  mask.src = `/api/mask/${frame}/${channel}.png?t=${t}`;
}

// ---------- Luminosity plot ----------
$('btn-show-plot').addEventListener('click', () => {
  const img = $('luminosity-plot');
  img.src = `/api/luminosity-plot.png?t=${Date.now()}`;
  img.style.display = 'block';
});

$('btn-hide-plot').addEventListener('click', () => {
  $('luminosity-plot').style.display = 'none';
});

// ---------- Log tab clear button ----------
$('btn-clear-log').addEventListener('click', () => {
  $('live-log').textContent = '';
});

// ---------- Setpoints editor ----------
async function loadSetpoints() {
  try {
    const r = await fetch('/api/setpoints');
    const data = await r.json();
    if (!data.ok) {
      $('setpoint-status').textContent = `Load failed: ${data.error || 'unknown'}`;
      return;
    }
    const rows = $('setpoint-rows');
    rows.innerHTML = '';
    for (const c of data.channels) {
      const row = document.createElement('div');
      row.className = 'field';
      const label = document.createElement('label');
      label.setAttribute('for', `setpoint-ch-${c.channel}`);
      label.textContent = `Channel ${c.channel}`;
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.id = `setpoint-ch-${c.channel}`;
      input.dataset.channel = c.channel;
      if (c.value != null) input.value = Number(c.value).toFixed(6);
      else input.placeholder = 'not yet computed';
      row.appendChild(label);
      row.appendChild(input);
      rows.appendChild(row);
    }
    $('setpoint-status').textContent = data.exists
      ? `Loaded from ${data.path}`
      : `No file yet at ${data.path}`;
  } catch (e) {
    $('setpoint-status').textContent = `Load error: ${e}`;
  }
}

$('btn-refresh-setpoints').addEventListener('click', loadSetpoints);

$('btn-save-setpoints').addEventListener('click', async () => {
  const channels = {};
  document.querySelectorAll('#setpoint-rows input').forEach((el) => {
    const v = el.value.trim();
    if (v !== '') channels[el.dataset.channel] = parseFloat(v);
  });
  if (Object.keys(channels).length === 0) {
    $('setpoint-status').textContent = 'Nothing to save';
    return;
  }
  try {
    const r = await fetch('/api/setpoints', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channels }),
    });
    const data = await r.json();
    if (data.ok) {
      $('setpoint-status').textContent =
        `Saved at ${new Date().toLocaleTimeString()}`;
    } else {
      $('setpoint-status').textContent = `Save failed: ${data.error || 'unknown'}`;
    }
  } catch (e) {
    $('setpoint-status').textContent = `Save error: ${e}`;
  }
});

// ---------- Init ----------
loadConfig();
loadSetpoints();
pollPipelineStatus();
pollFrames();
