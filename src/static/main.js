// ── Globals ────────────────────────────────────────────────
let nvOrig = null;
let nvCrop = null;
let selectedFile = null;
let lastProbs = null;
let lastThreshold = null;
let lastStart = null;
let lastEnd = null;
let currentSliceType = 'axial';

// ── Health check ───────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const dot = document.getElementById('status-dot');
    const lbl = document.getElementById('status-label');
    if (d.model_loaded) {
      dot.className = 'status-dot ok';
      lbl.textContent = 'Model ready';
    } else {
      dot.className = 'status-dot err';
      lbl.textContent = 'Model not loaded';
    }
  } catch {
    document.getElementById('status-dot').className = 'status-dot err';
    document.getElementById('status-label').textContent = 'Server unreachable';
  }
}
checkHealth();

// ── Drag & drop ─────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFileSelect(f);
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
  const name = file.name.toLowerCase();
  if (!name.endsWith('.nii') && !name.endsWith('.nii.gz')) {
    showToast('Please upload a .nii or .nii.gz file.');
    return;
  }
  selectedFile = file;
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('run-btn').disabled = false;

  // Preview original in left viewer
  loadVolumeFromFile(file, 'orig');
}

// ── NiiVue helpers ─────────────────────────────────────────
const SLICE_MAP = {
  axial:       niivue.Niivue.SLICE_TYPE?.AXIAL       ?? 0,
  coronal:     niivue.Niivue.SLICE_TYPE?.CORONAL      ?? 1,
  sagittal:    niivue.Niivue.SLICE_TYPE?.SAGITTAL     ?? 2,
  multiplanar: niivue.Niivue.SLICE_TYPE?.MULTIPLANAR  ?? 3,
};

function makeNV(canvasId) {
  const nv = new niivue.Niivue({
    backColor: [0.02, 0.04, 0.07, 1],
    crosshairColor: [0, 0.78, 1, 0.7],
    show3Dcrosshair: false,
    isColorbar: false,
    isHighResolutionCapable: true,
    logLevel: 'error',
  });
  nv.attachToCanvas(document.getElementById(canvasId));
  return nv;
}

async function loadVolumeFromFile(file, side) {
  const url = URL.createObjectURL(file);
  hideEmpty(side);

  if (side === 'orig') {
    if (nvOrig) { try { nvOrig.volumes = []; } catch {} }
    nvOrig = makeNV('gl-orig');
    await nvOrig.loadVolumes([{ url, name: file.name }]);
    applyColormap(nvOrig);
    applySliceType(nvOrig);
  } else {
    if (nvCrop) { try { nvCrop.volumes = []; } catch {} }
    nvCrop = makeNV('gl-crop');
    await nvCrop.loadVolumes([{ url, name: file.name }]);
    applyColormap(nvCrop);
    applySliceType(nvCrop);
  }
}

async function loadVolumeFromUrl(url, side) {
  hideEmpty(side);
  if (side === 'crop') {
    if (nvCrop) { try { nvCrop.volumes = []; } catch {} }
    nvCrop = makeNV('gl-crop');
    await nvCrop.loadVolumes([{ url }]);
    applyColormap(nvCrop);
    applySliceType(nvCrop);
  }
}

function hideEmpty(side) {
  document.getElementById(`empty-${side}`).style.display = 'none';
}

function applyColormap(nv) {
  const cm = document.getElementById('colormap-sel').value;
  if (nv.volumes.length) {
    nv.setColormap(nv.volumes[0].id, cm);
    nv.drawScene();
  }
}

function applySliceType(nv) {
  const t = SLICE_MAP[currentSliceType] ?? 0;
  nv.setSliceType(t);
  nv.drawScene();
}

// ── Colormap change ────────────────────────────────────────
document.getElementById('colormap-sel').addEventListener('change', () => {
  if (nvOrig) applyColormap(nvOrig);
  if (nvCrop) applyColormap(nvCrop);
});

// ── Slice-type toggles ─────────────────────────────────────
document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSliceType = btn.dataset.slice;
    if (nvOrig) applySliceType(nvOrig);
    if (nvCrop) applySliceType(nvCrop);
  });
});

// ── Run inference ──────────────────────────────────────────
document.getElementById('run-btn').addEventListener('click', runInference);

async function runInference() {
  if (!selectedFile) return;

  setRunning(true);

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const result = await res.json();
    if (result.error) throw new Error(result.error);

    showMetrics(result);
    await loadVolumeFromUrl(`/download/${result.filename}`, 'crop');
    drawProbChart(result.probs, result.threshold, result.start, result.end);

  } catch (err) {
    showToast(err.message || 'Inference failed.');
  } finally {
    setRunning(false);
  }
}



function showMetrics(r) {
  document.getElementById('metrics-section').style.display = '';
  document.getElementById('m-start').textContent  = r.start ?? '—';
  document.getElementById('m-end').textContent    = r.end   ?? '—';
  document.getElementById('m-nslices').textContent = r.nslices ?? '—';
  document.getElementById('m-pos').textContent    = r.pos   ?? '—';
  document.getElementById('m-thresh').textContent = r.threshold != null ? r.threshold.toFixed(3) : '—';

  const dlBtn = document.getElementById('dl-btn');
  dlBtn.href = `/download/${r.filename}`;
  dlBtn.style.display = 'block';

  lastStart     = r.start;
  lastEnd       = r.end;
  lastThreshold = r.threshold;
}

function setRunning(on) {
  const btn  = document.getElementById('run-btn');
  const wrap = document.getElementById('progress-wrap');
  btn.disabled = on;
  btn.textContent = on ? '⏳ RUNNING…' : '▶ RUN INFERENCE';
  if (on) {
    wrap.classList.add('active');
  } else {
    wrap.classList.remove('active');
    btn.textContent = '▶ RUN INFERENCE';
  }
}

// ── Probability chart ──────────────────────────────────────
function drawProbChart(probs, threshold, start, end) {
  document.getElementById('chart-section').style.display = '';
  const canvas = document.getElementById('prob-canvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth;
  const H = 120;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = { top: 8, right: 8, bottom: 20, left: 28 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top  - pad.bottom;
  const n  = probs.length;

  ctx.clearRect(0, 0, W, H);

  // Crop region highlight
  const x1 = pad.left + (start / (n - 1)) * cw;
  const x2 = pad.left + (end   / (n - 1)) * cw;
  ctx.fillStyle = 'rgba(0,255,163,0.08)';
  ctx.fillRect(x1, pad.top, x2 - x1, ch);

  // Threshold line
  const ty = pad.top + ch * (1 - threshold);
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = 'rgba(255,107,53,0.7)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, ty);
  ctx.lineTo(pad.left + cw, ty);
  ctx.stroke();
  ctx.setLineDash([]);

  // Area fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
  grad.addColorStop(0,   'rgba(0,200,255,0.5)');
  grad.addColorStop(1,   'rgba(0,200,255,0.02)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad.left + (i / (n - 1)) * cw;
    const y = pad.top  + ch * (1 - probs[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(pad.left + cw, pad.top + ch);
  ctx.lineTo(pad.left, pad.top + ch);
  ctx.closePath();
  ctx.fill();

  // Line
  ctx.strokeStyle = '#00c8ff';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad.left + (i / (n - 1)) * cw;
    const y = pad.top  + ch * (1 - probs[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Axes
  ctx.strokeStyle = 'rgba(90,122,150,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + ch);
  ctx.moveTo(pad.left, pad.top + ch); ctx.lineTo(pad.left + cw, pad.top + ch);
  ctx.stroke();

  // Y labels
  ctx.fillStyle = '#5a7a96';
  ctx.font = `${9 * dpr / dpr}px 'Space Mono', monospace`;
  ctx.textAlign = 'right';
  ctx.fillText('1.0', pad.left - 4, pad.top + 4);
  ctx.fillText('0.5', pad.left - 4, pad.top + ch * 0.5 + 3);
  ctx.fillText('0.0', pad.left - 4, pad.top + ch + 3);

  // X label
  ctx.textAlign = 'center';
  ctx.fillText('slice index', pad.left + cw / 2, H - 3);
}

// ── Toast ──────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4000);
}
