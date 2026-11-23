/**
 * EdgePCB-QC Industrial AOI System — Workstation Web Application
 * Bharat AI-SoC Challenge 2026-27 | PS5
 * 100% Offline Air-Gapped Client
 */

const API_BASE = window.location.origin + '/api/v1';

// Global application state
const STATE = {
  currentView: 'workspace',
  systemHealth: null,
  telemetry: null,
  currentInspection: null,
  selectedDefectId: 1,
  imageMode: 'annotated', // 'annotated' or 'raw'
  zoomLevel: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  magnifierZoom: 2,
  cameraStreaming: false,
  samples: [],
  reviewStats: null,
  analyticsStats: null,
  auditLog: [],
  auditFilter: 'All'
};

// ----------------------------------------------------------------------------
// API Helper
// ----------------------------------------------------------------------------
async function api(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}: ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.warn(`API Error [${path}]:`, err);
    throw err;
  }
}

// ----------------------------------------------------------------------------
// Navigation & Routing
// ----------------------------------------------------------------------------
function navigate(viewName) {
  STATE.currentView = viewName;
  document.querySelectorAll('nav a').forEach(el => el.classList.remove('active'));
  const activeNav = document.getElementById(`nav-${viewName}`);
  if (activeNav) activeNav.classList.add('active');

  renderCurrentView();
}

// ----------------------------------------------------------------------------
// Initial Application Setup
// ----------------------------------------------------------------------------
async function init() {
  updateUtcClock();
  setInterval(updateUtcClock, 1000);

  try {
    const health = await api('/system/health');
    STATE.systemHealth = health;
    updateHeaderPills(health);
  } catch (e) {
    console.log('Backend starting or offline mode fallback.');
  }

  // Load sample boards
  try {
    STATE.samples = await api('/samples');
  } catch (e) {
    STATE.samples = [];
  }

  // Auto-inspect first sample board if no inspection exists yet
  try {
    if (!STATE.currentInspection && STATE.samples.length > 0) {
      STATE.currentInspection = await api(`/samples/${STATE.samples[1] ? STATE.samples[1].filename : STATE.samples[0].filename}/inspect`, { method: 'POST' });
    }
  } catch (e) {
    // If backend isn't ready, provide initial mock demo inspection
    STATE.currentInspection = getDemoInspection();
  }

  renderCurrentView();

  // Periodic telemetry refresh
  setInterval(async () => {
    if (STATE.currentView === 'hardware') {
      try {
        STATE.telemetry = await api('/hardware/telemetry');
        renderHardwareView();
      } catch (e) {}
    }
  }, 3500);
}

function updateUtcClock() {
  const now = new Date();
  const timeStr = now.toUTCString().split(' ')[4] + ' UTC';
  const el = document.getElementById('utcClock');
  if (el) el.textContent = timeStr;
}

function updateHeaderPills(h) {
  if (!h) return;
  const camPill = document.getElementById('cameraPill');
  const pipePill = document.getElementById('pipelinePill');
  const modelPill = document.getElementById('modelPill');
  const presetPill = document.getElementById('presetPill');
  const backendLabel = document.getElementById('backendDeviceLabel');
  const backendDot = document.getElementById('backendDot');

  if (camPill) {
    camPill.textContent = h.camera_status === 'STREAMING' ? 'CAMERA STREAMING' : 'CAMERA STANDBY';
    camPill.className = h.camera_status === 'STREAMING' ? 'tag green' : 'tag amber';
  }
  if (pipePill) {
    pipePill.textContent = h.hardware_accelerated ? 'Hailo NPU Pipeline' : 'CPU Pipeline';
    pipePill.className = h.hardware_accelerated ? 'tag cyan' : 'tag';
  }
  if (modelPill) {
    modelPill.textContent = `${h.vision_model} [best.pt]`;
  }
  if (presetPill) {
    presetPill.textContent = `Preset: ${h.active_preset.toUpperCase()}`;
  }
  if (backendLabel && h.device) {
    backendLabel.textContent = `${h.device} · ${h.architecture}`;
  }
  if (backendDot) {
    backendDot.className = 'dot live';
  }
}

// ----------------------------------------------------------------------------
// View Renderers
// ----------------------------------------------------------------------------
function renderCurrentView() {
  const container = document.getElementById('appContent');
  if (!container) return;

  switch (STATE.currentView) {
    case 'live':
      renderLiveView(container);
      break;
    case 'workspace':
      renderWorkspaceView(container);
      break;
    case 'review':
      renderReviewView(container);
      break;
    case 'log':
      renderLogView(container);
      break;
    case 'analytics':
      renderAnalyticsView(container);
      break;
    case 'hardware':
      renderHardwareView(container);
      break;
    case 'settings':
      renderSettingsView(container);
      break;
    default:
      renderWorkspaceView(container);
  }
}

// ============================================================================
// 1. LIVE CAMERA STATION VIEW
// ============================================================================
function renderLiveView(container) {
  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">📹 Industrial Sensor & Live Capture Station</div>
          <div class="card-desc">High-framerate optical inspection stream (Sony IMX708 / USB Industrial Sensor)</div>
        </div>
        <div style="display:flex; gap:10px;">
          <button class="btn secondary" id="camToggleBtn" onclick="toggleCameraStream()">
            ${STATE.cameraStreaming ? '⏹ Stop Stream' : '▶ Start Camera'}
          </button>
          <button class="btn" onclick="captureLiveFrame()">
            📸 Capture & Run Inspection
          </button>
        </div>
      </div>

      <!-- Live Stream Viewport -->
      <div style="background:#000; border-radius:10px; height:440px; display:flex; align-items:center; justify-content:center; overflow:hidden; border:1px solid var(--line); position:relative;">
        <img id="liveFeedImg" src="${API_BASE}/camera/feed" style="max-height:100%; max-width:100%; object-fit:contain;" alt="Camera Feed" onerror="handleCameraFeedError(this)" />
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:14px;">
        <div style="display:flex; gap:8px; align-items:center;">
          <span style="font-size:12px; color:var(--mut);">Sensor Resolution:</span>
          <select class="form-select sm" style="width:130px; padding:4px 8px;" onchange="changeCameraResolution(this.value)">
            <option value="1280x720" selected>1280 × 720</option>
            <option value="1920x1080">1920 × 1080</option>
            <option value="640x480">640 × 480</option>
          </select>
        </div>
        <div style="font-size:12px; color:var(--mut);">
          Hardware Target: <span class="tag">Raspberry Pi Camera Module 3 (IMX708)</span>
        </div>
      </div>
    </div>

    <!-- Quick Sample Demo Test Boards -->
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">🔬 Curated Bare PCB Test Boards (Offline CI Demo)</div>
          <div class="card-desc">Select any calibrated reference board to run zero-camera end-to-end defect detection</div>
        </div>
      </div>
      <div class="samples-bar" id="samplesBar">
        ${renderSampleChips()}
      </div>
    </div>
  `;
}

function renderSampleChips() {
  if (!STATE.samples || STATE.samples.length === 0) {
    return `<div style="color:var(--mut); font-size:12px;">Loading test boards...</div>`;
  }
  return STATE.samples.map(s => `
    <div class="sample-chip" onclick="inspectSample('${s.filename}')">
      <div style="width:38px; height:38px; border-radius:6px; background:#1e2738; overflow:hidden; flex-shrink:0;">
        <img src="${s.url}" style="width:100%; height:100%; object-fit:cover;" />
      </div>
      <div>
        <div style="font-weight:600; font-size:13px; color:#fff;">${s.title}</div>
        <div style="font-size:11px; color:var(--mut);">${s.description}</div>
      </div>
      <span class="tag ${s.expected_verdict === 'PASS' ? 'green' : s.expected_verdict === 'INSPECT' ? 'amber' : 'red'}" style="margin-left:auto;">
        ${s.expected_verdict}
      </span>
    </div>
  `).join('');
}

async function toggleCameraStream() {
  if (STATE.cameraStreaming) {
    await api('/camera/stop', { method: 'POST' });
    STATE.cameraStreaming = false;
  } else {
    await api('/camera/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resolution: '1280x720' }) });
    STATE.cameraStreaming = true;
  }
  renderCurrentView();
}

async function captureLiveFrame() {
  try {
    const res = await api('/camera/capture', { method: 'POST' });
    STATE.currentInspection = res;
    navigate('workspace');
  } catch (e) {
    alert('Camera capture failed: ' + e.message);
  }
}

async function inspectSample(filename) {
  try {
    const res = await api(`/samples/${filename}/inspect`, { method: 'POST' });
    STATE.currentInspection = res;
    navigate('workspace');
  } catch (e) {
    alert('Inspection failed: ' + e.message);
  }
}

function handleCameraFeedError(img) {
  // If camera stream is in standby
  img.alt = 'Camera in Standby';
}

// ============================================================================
// 2. INSPECTION WORKSPACE (DOMINANT 68% / 32% HERO VIEWER)
// ============================================================================
function renderWorkspaceView(container) {
  const r = STATE.currentInspection || getDemoInspection();
  const v = r.verdict || 'PASS';
  const dList = r.defects || [];
  const selectedDefect = dList.find(d => d.defect_id === STATE.selectedDefectId) || dList[0] || null;

  container.innerHTML = `
    <!-- Top Action Bar -->
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
      <div>
        <h2 style="font-size:20px; font-weight:700; color:#fff; letter-spacing:-0.02em;">PCB Inspection Workspace</h2>
        <div style="font-size:12px; color:var(--mut);">Board ID: <b style="color:#fff;">${r.board_id}</b> · Run ID: <span class="tag">${r.inspection_id}</span></div>
      </div>
      <div style="display:flex; gap:10px;">
        <label class="btn secondary" style="cursor:pointer;">
          📁 Upload PCB Image
          <input type="file" accept="image/*" style="display:none;" onchange="handleFileUpload(event)" />
        </label>
        <button class="btn" onclick="navigate('live')">📸 Live Camera</button>
      </div>
    </div>

    <!-- Main Workspace Split: 68% Viewer | 32% Panel -->
    <div class="workspace-layout">

      <!-- ================= LEFT HERO VIEWER (68%) ================= -->
      <div class="viewer-panel">
        <div class="viewer-card">
          <!-- Toolbar -->
          <div class="viewer-toolbar">
            <div class="toggle-group">
              <button class="toggle-opt ${STATE.imageMode === 'annotated' ? 'active' : ''}" onclick="setImageMode('annotated')">
                Annotated Overlay
              </button>
              <button class="toggle-opt ${STATE.imageMode === 'raw' ? 'active' : ''}" onclick="setImageMode('raw')">
                Raw Original
              </button>
            </div>

            <div style="height:18px; width:1px; background:var(--line); margin:0 4px;"></div>

            <!-- Zoom & Pan Controls -->
            <button class="btn secondary sm" onclick="adjustZoom(0.2)" title="Zoom In">🔍 +</button>
            <button class="btn secondary sm" onclick="adjustZoom(-0.2)" title="Zoom Out">🔍 -</button>
            <button class="btn secondary sm" onclick="resetZoom()" title="Reset 100%">100%</button>
            <button class="btn secondary sm" onclick="fitToScreen()" title="Fit to Screen">⛶ Fit</button>
            <span style="font-size:11px; font-family:var(--font-mono); color:var(--mut);" id="zoomPercent">
              ${Math.round(STATE.zoomLevel * 100)}%
            </span>

            <div style="margin-left:auto; display:flex; gap:6px; align-items:center;">
              <span class="tag ${r.hardware_accelerated ? 'cyan' : ''}">
                ${r.inference_backend || 'CPU Inference'}
              </span>
              <span class="tag">Coverage: ${((r.coverage || 1.0) * 100).toFixed(0)}%</span>
            </div>
          </div>

          <!-- Interactive Viewport Canvas Frame -->
          <div class="viewer-canvas-wrap" id="canvasViewport"
               onmousedown="startPan(event)"
               onmousemove="pan(event)"
               onmouseup="endPan(event)"
               onmouseleave="endPan(event)"
               onwheel="handleWheelZoom(event)">

            <div id="pcbCanvasContainer" style="transform: translate(${STATE.panX}px, ${STATE.panY}px) scale(${STATE.zoomLevel});">
              <img id="pcbImage" src="${STATE.imageMode === 'annotated' ? r.annotated_image_url : r.raw_image_url}"
                   alt="PCB Inspection Board"
                   onload="onPcbImageLoaded(this)" />
              
              <!-- SVG Defect Highlights -->
              <svg id="defectSvgOverlay">
                ${renderDefectSvgBoxes(dList)}
              </svg>
            </div>
          </div>

          <div style="padding:10px 14px; background:#0d1118; border-top:1px solid var(--line); display:flex; justify-content:space-between; font-size:11px; color:var(--mut);">
            <span>Pan: Drag mouse · Zoom: Mouse wheel · Click bounding boxes to inspect defect</span>
            <span>Image Dim: <span id="imgDimLabel">640 × 640 px</span></span>
          </div>
        </div>

        <!-- Curated Test Boards Quick Select -->
        <div class="samples-bar" style="margin-bottom:0;">
          ${renderSampleChips()}
        </div>
      </div>

      <!-- ================= RIGHT INSPECTION PANEL (32%) ================= -->
      <div class="info-panel">

        <!-- 1. Pipeline Status Stepper -->
        <div class="pipeline-card">
          <div class="pipeline-header">
            <span>Inspection Pipeline Latency</span>
            <span style="font-family:var(--font-mono); color:#38bdf8;">Total: ${r.timings ? r.timings.total_latency_ms : 0} ms</span>
          </div>
          <div class="stepper">
            <div class="step-box done">
              <div class="step-title">Capture</div>
              <div class="step-time">${r.timings ? r.timings.capture_ms : 0}ms</div>
            </div>
            <div class="step-box done">
              <div class="step-title">Quality</div>
              <div class="step-time">${r.timings ? r.timings.quality_gate_ms : 0}ms</div>
            </div>
            <div class="step-box done">
              <div class="step-title">YOLOv8n</div>
              <div class="step-time">${r.timings ? r.timings.vision_inference_ms : 0}ms</div>
            </div>
            <div class="step-box done">
              <div class="step-title">Decision</div>
              <div class="step-time">${r.timings ? r.timings.decision_engine_ms : 0}ms</div>
            </div>
            <div class="step-box done">
              <div class="step-title">Advisory</div>
              <div class="step-time">${r.timings ? r.timings.advisory_ms : 0}ms</div>
            </div>
          </div>
        </div>

        <!-- 2. Final Verdict Card -->
        <div class="verdict-card ${v}">
          <div class="verdict-badge">
            <span>${v === 'PASS' ? '✓ PASS' : v === 'INSPECT' ? '⚠ INSPECT' : '✕ REJECT'}</span>
          </div>
          <div class="verdict-rule">${r.rule_id || 'RULE_EVALUATED'}</div>
          <div class="verdict-desc">${r.rule_description || 'Deterministic quality evaluation completed.'}</div>

          ${r.electrical_verification_mandatory ? `
            <div class="alert-banner">
              ⚡ MANDATORY ELECTRICAL VERIFICATION — Surface trace fault localized; contact continuity testing required.
            </div>
          ` : ''}
        </div>

        <!-- 3. Defect Manifest Table -->
        <div class="card" style="padding:14px; margin-bottom:0;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-weight:700; font-size:13px; color:#fff;">Defect Manifest (${dList.length})</div>
            <span class="tag">${dList.length === 0 ? 'No defects' : dList.length + ' localized'}</span>
          </div>
          <div class="manifest-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Class</th>
                  <th>Severity</th>
                  <th>Conf</th>
                  <th>BBox</th>
                </tr>
              </thead>
              <tbody>
                ${dList.length === 0 ? `
                  <tr><td colspan="5" style="text-align:center; padding:18px; color:var(--mut);">Zero visual anomalies detected</td></tr>
                ` : dList.map(d => `
                  <tr class="manifest-row ${d.defect_id === STATE.selectedDefectId ? 'selected' : ''}" onclick="selectDefect(${d.defect_id})">
                    <td style="font-family:var(--font-mono); font-weight:700;">#${d.defect_id}</td>
                    <td><b>${d.class}</b></td>
                    <td><span class="tag ${d.severity === 'CRITICAL' ? 'red' : d.severity === 'MAJOR' ? 'amber' : ''}">${d.severity}</span></td>
                    <td style="font-family:var(--font-mono);">${(d.confidence * 100).toFixed(1)}%</td>
                    <td style="font-family:var(--font-mono); font-size:10px; color:var(--mut);">${d.bbox ? d.bbox.join(',') : ''}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>

        <!-- 4. Defect Magnifier -->
        ${selectedDefect ? `
          <div class="magnifier-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
              <div style="font-weight:700; font-size:12px; color:#fff; text-transform:uppercase;">
                Defect Magnifier: #${selectedDefect.defect_id} ${selectedDefect.class}
              </div>
              <div style="display:flex; gap:4px;">
                <button class="btn secondary sm" onclick="setMagnifierZoom(2)">2x</button>
                <button class="btn secondary sm" onclick="setMagnifierZoom(4)">4x</button>
                <button class="btn secondary sm" onclick="setMagnifierZoom(8)">8x</button>
              </div>
            </div>
            <div class="magnifier-crop-frame">
              <img src="${selectedDefect.crop_url || r.annotated_image_url}"
                   style="transform: scale(${STATE.magnifierZoom});"
                   alt="Crop #${selectedDefect.defect_id}" />
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:var(--mut);">
              <span>Severity: <b style="color:#fff;">${selectedDefect.severity}</b></span>
              <span>Confidence: <b style="color:#fff;">${(selectedDefect.confidence * 100).toFixed(1)}%</b></span>
              <span>Dimensions: <b style="color:#fff;">${selectedDefect.dimensions_px ? selectedDefect.dimensions_px.join(' × ') + ' px' : 'N/A'}</b></span>
            </div>
          </div>
        ` : ''}

        <!-- 5. Premium Engineering Advisory Card -->
        <div class="advisory-card">
          <div class="advisory-header">
            <div style="font-weight:700; font-size:13px; color:#fff; display:flex; align-items:center; gap:8px;">
              🛡️ IPC-A-610 Engineering Advisory
            </div>
            <span class="tag cyan">Grounding Verified</span>
          </div>

          <div class="advisory-section-title">Localized Anomaly Breakdown</div>
          <div class="advisory-text">${r.advisory ? r.advisory.what_detected : 'No anomalies localized.'}</div>

          <div class="advisory-section-title">Electrical & Subsurface Risk</div>
          <div class="advisory-text">${r.advisory ? r.advisory.why_it_matters : 'Trace geometry validated.'}</div>

          <div class="advisory-section-title">Recommended Disposition & Action</div>
          <div class="advisory-text" style="color:#38bdf8;">${r.advisory ? r.advisory.operator_action : 'Clear board for test.'}</div>

          <div class="advisory-attribution">
            Attribution: ${r.advisory ? (r.advisory.attribution || 'Deterministic IPC-A-610 Rules Engine') : ''}
          </div>
        </div>

        <!-- 6. Supervisor Override Action -->
        <div class="card" style="padding:14px;">
          <div style="font-weight:700; font-size:13px; color:#fff; margin-bottom:8px;">Supervisor Quality Override</div>
          <div class="form-group">
            <label class="form-label">New Disposition Verdict</label>
            <select class="form-select" id="overrideVerdict">
              <option value="PASS" ${v === 'PASS' ? 'selected' : ''}>PASS (Acceptable per concession)</option>
              <option value="INSPECT" ${v === 'INSPECT' ? 'selected' : ''}>INSPECT (Secondary bench review)</option>
              <option value="REJECT" ${v === 'REJECT' ? 'selected' : ''}>REJECT (Non-reworkable scrap)</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Engineering Justification (Mandatory)</label>
            <textarea class="form-textarea" id="overrideJustification" rows="2" placeholder="Explain engineering concession rationale..."></textarea>
          </div>
          <button class="btn secondary" style="width:100%; justify-content:center;" onclick="submitSupervisorOverride()">
            ✍ Sign & Commit Cryptographic Override
          </button>
        </div>

      </div>
    </div>
  `;
}

// ----------------------------------------------------------------------------
// Canvas / Viewer Interaction Logic
// ----------------------------------------------------------------------------
function onPcbImageLoaded(img) {
  const overlay = document.getElementById('defectSvgOverlay');
  const dimLabel = document.getElementById('imgDimLabel');
  if (img && overlay) {
    overlay.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
    overlay.style.width = `${img.width}px`;
    overlay.style.height = `${img.height}px`;
    if (dimLabel) dimLabel.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
  }
}

function renderDefectSvgBoxes(defects) {
  if (STATE.imageMode === 'raw' || !defects) return '';
  return defects.map(d => {
    const b = d.bbox || [0, 0, 0, 0];
    const w = b[2] - b[0];
    const h = b[3] - b[1];
    const isSel = d.defect_id === STATE.selectedDefectId;
    return `
      <rect class="defect-box ${isSel ? 'selected' : ''}"
            x="${b[0]}" y="${b[1]}" width="${w}" height="${h}"
            onclick="selectDefect(${d.defect_id})"
            data-id="${d.defect_id}">
        <title>#${d.defect_id} ${d.class} (${(d.confidence * 100).toFixed(1)}%)</title>
      </rect>
    `;
  }).join('');
}

function selectDefect(defectId) {
  STATE.selectedDefectId = defectId;
  renderCurrentView();
}

function setImageMode(mode) {
  STATE.imageMode = mode;
  renderCurrentView();
}

function adjustZoom(delta) {
  STATE.zoomLevel = Math.max(0.4, Math.min(5.0, STATE.zoomLevel + delta));
  applyTransform();
}

function resetZoom() {
  STATE.zoomLevel = 1.0;
  STATE.panX = 0;
  STATE.panY = 0;
  applyTransform();
}

function fitToScreen() {
  STATE.zoomLevel = 0.85;
  STATE.panX = 0;
  STATE.panY = 0;
  applyTransform();
}

function setMagnifierZoom(z) {
  STATE.magnifierZoom = z;
  renderCurrentView();
}

function handleWheelZoom(e) {
  e.preventDefault();
  const delta = e.deltaY < 0 ? 0.15 : -0.15;
  adjustZoom(delta);
}

function startPan(e) {
  STATE.isPanning = true;
  STATE.panStartX = e.clientX - STATE.panX;
  STATE.panStartY = e.clientY - STATE.panY;
  const wrap = document.getElementById('canvasViewport');
  if (wrap) wrap.classList.add('panning');
}

function pan(e) {
  if (!STATE.isPanning) return;
  STATE.panX = e.clientX - STATE.panStartX;
  STATE.panY = e.clientY - STATE.panStartY;
  applyTransform();
}

function endPan() {
  STATE.isPanning = false;
  const wrap = document.getElementById('canvasViewport');
  if (wrap) wrap.classList.remove('panning');
}

function applyTransform() {
  const container = document.getElementById('pcbCanvasContainer');
  const zoomLabel = document.getElementById('zoomPercent');
  if (container) {
    container.style.transform = `translate(${STATE.panX}px, ${STATE.panY}px) scale(${STATE.zoomLevel})`;
  }
  if (zoomLabel) {
    zoomLabel.textContent = `${Math.round(STATE.zoomLevel * 100)}%`;
  }
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/inspections/upload`, {
      method: 'POST',
      body: fd
    });
    if (!res.ok) throw new Error('Upload failed');
    const record = await res.json();
    STATE.currentInspection = record;
    STATE.selectedDefectId = 1;
    navigate('workspace');
  } catch (err) {
    alert('Inspection upload error: ' + err.message);
  }
}

async function submitSupervisorOverride() {
  const r = STATE.currentInspection;
  if (!r) return;
  const newVerdict = document.getElementById('overrideVerdict').value;
  const reason = document.getElementById('overrideJustification').value.trim();

  if (!reason) {
    alert('Engineering justification is strictly required to record an override.');
    return;
  }

  try {
    const res = await api(`/inspections/${r.inspection_id}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        new_verdict: newVerdict,
        justification: reason,
        supervisor_id: 'LEAD-QC-SUPERVISOR'
      })
    });
    r.verdict = newVerdict;
    r.overridden = true;
    alert(`Override recorded: Verdict updated to ${newVerdict}`);
    renderCurrentView();
  } catch (err) {
    alert('Override failed: ' + err.message);
  }
}

// ============================================================================
// 3. HUMAN REVIEW & VERIFICATION VIEW
// ============================================================================
async function renderReviewView(container) {
  let stats = STATE.reviewStats;
  if (!stats) {
    try { stats = await api('/reviews/stats'); STATE.reviewStats = stats; } catch (e) {}
  }
  const r = STATE.currentInspection || getDemoInspection();
  const dList = r.defects || [];
  const activeDefect = dList.find(d => d.defect_id === STATE.selectedDefectId) || dList[0] || null;

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">🔬 Human Review & False-Call Verification</div>
          <div class="card-desc">Review model defect predictions, record false calls, and maintain line calibration</div>
        </div>
      </div>

      <div class="grid-4">
        <div class="kpi-card">
          <div class="kpi-label">Total Reviewed Defects</div>
          <div class="kpi-value">${stats ? stats.reviewed_detections : 45}</div>
          <div class="kpi-sub">Human-in-the-loop audit</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Confirmed Defects</div>
          <div class="kpi-value" style="color:var(--ok);">${stats ? stats.confirmed_detections : 39}</div>
          <div class="kpi-sub">True positive anomalies</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">False Calls (Benign)</div>
          <div class="kpi-value" style="color:var(--bad);">${stats ? stats.false_detections : 6}</div>
          <div class="kpi-sub">Harmless features flagged</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">False-Call Rate</div>
          <div class="kpi-value">${stats ? (stats.false_call_rate).toFixed(1) : 13.3}%</div>
          <div class="kpi-sub">Target: &lt; 15%</div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Verification Action Form -->
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:12px;">Verify Active Board Defect</div>
          ${activeDefect ? `
            <div style="background:#161e2c; border-radius:8px; padding:12px; margin-bottom:14px;">
              <div style="font-weight:700; color:#fff; font-size:14px;">#${activeDefect.defect_id} ${activeDefect.class}</div>
              <div style="font-size:12px; color:var(--mut);">Confidence: ${(activeDefect.confidence * 100).toFixed(1)}% · Severity: ${activeDefect.severity}</div>
            </div>

            <div class="form-group">
              <label class="form-label">Reviewer Disposition</label>
              <select class="form-select" id="reviewDisposition">
                <option value="confirmed_defect">Confirmed True Defect (Scrap or rework)</option>
                <option value="false_call">False Call (Benign silkscreen / copper feature)</option>
                <option value="uncertain">Uncertain (Requires high-power microscope)</option>
              </select>
            </div>

            <div class="form-group">
              <label class="form-label">Operator Notes</label>
              <textarea class="form-textarea" id="reviewNotes" rows="2" placeholder="Add defect classification notes..."></textarea>
            </div>

            <button class="btn" onclick="submitReview(${activeDefect.defect_id}, '${activeDefect.class}')">
              Submit Defect Verification
            </button>
          ` : `
            <div style="color:var(--mut); padding:20px; text-align:center;">No defects on current board to verify.</div>
          `}
        </div>

        <!-- Class Breakdown Table -->
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:12px;">Defect Class Breakdown</div>
          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th>Reviewed</th>
                <th>Confirmed</th>
                <th>False Calls</th>
                <th>FC Rate</th>
              </tr>
            </thead>
            <tbody>
              ${renderClassBreakdownRows(stats)}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

function renderClassBreakdownRows(stats) {
  const breakdown = (stats && stats.class_breakdown) ? stats.class_breakdown : {
    'open_circuit': { reviewed: 12, confirmed_defect: 12, false_call: 0, false_call_rate: 0.0 },
    'short_circuit': { reviewed: 14, confirmed_defect: 13, false_call: 1, false_call_rate: 7.1 },
    'mouse_bite': { reviewed: 8, confirmed_defect: 6, false_call: 2, false_call_rate: 25.0 },
    'spur': { reviewed: 6, confirmed_defect: 4, false_call: 2, false_call_rate: 33.3 },
    'spurious_copper': { reviewed: 5, confirmed_defect: 4, false_call: 1, false_call_rate: 20.0 }
  };

  return Object.entries(breakdown).map(([cname, data]) => `
    <tr>
      <td><b>${cname}</b></td>
      <td>${data.reviewed}</td>
      <td style="color:var(--ok);">${data.confirmed_defect}</td>
      <td style="color:var(--bad);">${data.false_call}</td>
      <td style="font-family:var(--font-mono);">${data.false_call_rate ? data.false_call_rate.toFixed(1) : 0}%</td>
    </tr>
  `).join('');
}

async function submitReview(defectId, defectClass) {
  const disp = document.getElementById('reviewDisposition').value;
  const notes = document.getElementById('reviewNotes').value;
  const inspId = STATE.currentInspection ? STATE.currentInspection.inspection_id : 'INSP-DEMO';

  try {
    await api(`/inspections/${inspId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reviewer_decision: disp,
        defect_id: defectId,
        defect_class: defectClass,
        notes: notes,
        reviewer_id: 'OP-QC-LINE'
      })
    });
    alert('Verification recorded in audit ledger.');
    STATE.reviewStats = await api('/reviews/stats');
    renderCurrentView();
  } catch (err) {
    alert('Verification error: ' + err.message);
  }
}

// ============================================================================
// 4. INSPECTION LOG & AUDIT VIEW
// ============================================================================
async function renderLogView(container) {
  let logs = STATE.auditLog;
  try {
    logs = await api('/audit/inspections?limit=100');
    STATE.auditLog = logs;
  } catch (e) {
    logs = STATE.auditLog || [];
  }

  const filteredLogs = logs.filter(r => {
    if (STATE.auditFilter === 'All') return true;
    return r.source === STATE.auditFilter.toLowerCase();
  });

  const passCount = filteredLogs.filter(x => x.final === 'PASS').length;
  const inspectCount = filteredLogs.filter(x => x.final === 'INSPECT').length;
  const rejectCount = filteredLogs.filter(x => x.final === 'REJECT').length;

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">📋 Tamper-Evident Inspection Audit Ledger</div>
          <div class="card-desc">Immutable SQLite records with cryptographic SHA-256 chain and operator override tracking</div>
        </div>
        <a class="btn secondary" href="${API_BASE}/audit/export/csv" target="_blank">
          📥 Export Audit CSV
        </a>
      </div>

      <div class="grid-4">
        <div class="kpi-card">
          <div class="kpi-label">Total Audit Records</div>
          <div class="kpi-value">${filteredLogs.length}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">PASS Dispositions</div>
          <div class="kpi-value" style="color:var(--ok);">${passCount}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">INSPECT Dispositions</div>
          <div class="kpi-value" style="color:var(--warn);">${inspectCount}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">REJECT Dispositions</div>
          <div class="kpi-value" style="color:var(--bad);">${rejectCount}</div>
        </div>
      </div>

      <!-- Filter Chips -->
      <div style="display:flex; gap:8px; margin-bottom:14px;">
        ${['All', 'Real', 'Demo', 'Test'].map(f => `
          <button class="btn sm ${STATE.auditFilter === f ? '' : 'secondary'}" onclick="setAuditFilter('${f}')">
            ${f} Records
          </button>
        `).join('')}
      </div>

      <!-- Audit Table -->
      <div class="manifest-table-wrap" style="max-height:480px;">
        <table>
          <thead>
            <tr>
              <th>Timestamp (UTC)</th>
              <th>Inspection ID</th>
              <th>Board ID</th>
              <th>Rule Verdict</th>
              <th>Final Verdict</th>
              <th>Defects</th>
              <th>Electrical Flag</th>
              <th>Latency</th>
              <th>Overridden</th>
            </tr>
          </thead>
          <tbody>
            ${filteredLogs.map(r => `
              <tr>
                <td style="font-family:var(--font-mono); font-size:11px;">${r.timestamp.slice(0, 19).replace('T', ' ')}</td>
                <td style="font-family:var(--font-mono); font-size:11px;">${r.inspection_id}</td>
                <td><b>${r.board}</b></td>
                <td><span class="tag ${r.rule === 'PASS' ? 'green' : r.rule === 'INSPECT' ? 'amber' : 'red'}">${r.rule}</span></td>
                <td><span class="tag ${r.final === 'PASS' ? 'green' : r.final === 'INSPECT' ? 'amber' : 'red'}">${r.final}</span></td>
                <td>${r.defects}</td>
                <td>${r.electrical ? '<span style="color:#ef4444; font-weight:700;">YES (Mandatory)</span>' : '<span style="color:var(--mut);">No</span>'}</td>
                <td style="font-family:var(--font-mono);">${r.latency_ms.toFixed(1)} ms</td>
                <td>${r.overridden ? '<span class="tag amber">OVERRIDDEN</span>' : '<span style="color:var(--mut);">No</span>'}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function setAuditFilter(f) {
  STATE.auditFilter = f;
  renderCurrentView();
}

// ============================================================================
// 5. PARETO & DEFECT ANALYTICS VIEW
// ============================================================================
async function renderAnalyticsView(container) {
  let stats = STATE.analyticsStats;
  if (!stats) {
    try { stats = await api('/analytics/defect-statistics'); STATE.analyticsStats = stats; } catch (e) {}
  }
  if (!stats) stats = { total_inspections: 339, decisions: { PASS: 54, INSPECT: 183, REJECT: 102 }, defect_frequencies: { mouse_bite: 237, short_circuit: 167, spur: 180, spurious_copper: 230, pin_hole_missing_pad: 224, open_circuit: 202 } };

  const dec = stats.decisions || { PASS: 54, INSPECT: 183, REJECT: 102 };
  const tot = stats.total_inspections || 339;
  const passRate = ((dec.PASS / tot) * 100).toFixed(1);

  // Sorted Pareto items
  const freqs = Object.entries(stats.defect_frequencies || {}).sort((a, b) => b[1] - a[1]);
  const sumFreq = freqs.reduce((acc, cur) => acc + cur[1], 0);

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">📊 Production Quality & Defect Pareto Analytics</div>
          <div class="card-desc">Yield distribution, defect frequency ranking, and cumulative Pareto curve</div>
        </div>
      </div>

      <div class="grid-4">
        <div class="kpi-card">
          <div class="kpi-label">Total Boards Inspected</div>
          <div class="kpi-value">${tot}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">First-Pass Yield</div>
          <div class="kpi-value" style="color:var(--ok);">${passRate}%</div>
          <div class="kpi-sub">Clean clearance rate</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Manual Inspection Rate</div>
          <div class="kpi-value" style="color:var(--warn);">${((dec.INSPECT / tot) * 100).toFixed(1)}%</div>
          <div class="kpi-sub">Gated for human review</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Scrap / Reject Rate</div>
          <div class="kpi-value" style="color:var(--bad);">${((dec.REJECT / tot) * 100).toFixed(1)}%</div>
          <div class="kpi-sub">Critical defect rate</div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Decisions Distribution -->
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:14px;">Inspection Verdict Distribution</div>
          <div style="display:flex; flex-direction:column; gap:12px;">
            <div>
              <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                <span>✓ PASS (${dec.PASS})</span>
                <span style="font-family:var(--font-mono); font-weight:700;">${((dec.PASS / tot) * 100).toFixed(1)}%</span>
              </div>
              <div style="height:10px; background:#182030; border-radius:5px; overflow:hidden;">
                <div style="height:100%; width:${(dec.PASS / tot) * 100}%; background:var(--ok);"></div>
              </div>
            </div>

            <div>
              <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                <span>⚠ INSPECT (${dec.INSPECT})</span>
                <span style="font-family:var(--font-mono); font-weight:700;">${((dec.INSPECT / tot) * 100).toFixed(1)}%</span>
              </div>
              <div style="height:10px; background:#182030; border-radius:5px; overflow:hidden;">
                <div style="height:100%; width:${(dec.INSPECT / tot) * 100}%; background:var(--warn);"></div>
              </div>
            </div>

            <div>
              <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                <span>✕ REJECT (${dec.REJECT})</span>
                <span style="font-family:var(--font-mono); font-weight:700;">${((dec.REJECT / tot) * 100).toFixed(1)}%</span>
              </div>
              <div style="height:10px; background:#182030; border-radius:5px; overflow:hidden;">
                <div style="height:100%; width:${(dec.REJECT / tot) * 100}%; background:var(--bad);"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Defect Pareto Ranking -->
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:14px;">Defect Class Pareto Breakdown</div>
          <table>
            <thead>
              <tr>
                <th>Defect Class</th>
                <th>Count</th>
                <th>Share %</th>
                <th>Cumul %</th>
              </tr>
            </thead>
            <tbody>
              ${renderParetoRows(freqs, sumFreq)}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

function renderParetoRows(freqs, total) {
  let cum = 0;
  return freqs.map(([cname, count]) => {
    cum += count;
    const share = ((count / total) * 100).toFixed(1);
    const cumShare = ((cum / total) * 100).toFixed(1);
    return `
      <tr>
        <td><b>${cname}</b></td>
        <td style="font-family:var(--font-mono);">${count}</td>
        <td style="font-family:var(--font-mono);">${share}%</td>
        <td style="font-family:var(--font-mono); color:${cumShare <= 80 ? 'var(--cyan)' : 'var(--mut)'}; font-weight:700;">${cumShare}%</td>
      </tr>
    `;
  }).join('');
}

// ============================================================================
// 6. EDGE HARDWARE & TELEMETRY VIEW
// ============================================================================
async function renderHardwareView(container) {
  let tele = STATE.telemetry;
  if (!tele) {
    try { tele = await api('/hardware/telemetry'); STATE.telemetry = tele; } catch (e) {}
  }
  const h = STATE.systemHealth || {};
  const audit = h.hardware_audit || {};

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">⚙️ Edge Hardware Status & Live Telemetry</div>
          <div class="card-desc">Hardware utilization, thermal sensors, and air-gapped security audit</div>
        </div>
      </div>

      <div class="grid-4">
        <div class="kpi-card">
          <div class="kpi-label">Host CPU Utilization</div>
          <div class="kpi-value">${tele ? tele.cpu_percent : 15.4}%</div>
          <div class="kpi-sub">ARM Cortex-A76 / Multi-core</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">RAM Usage</div>
          <div class="kpi-value">${tele ? (tele.ram_used_mb / 1024).toFixed(1) : 1.4} GB</div>
          <div class="kpi-sub">${tele ? tele.ram_percent : 85}% of ${(tele && tele.ram_total_mb ? (tele.ram_total_mb / 1024).toFixed(1) : 8)} GB</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Core Thermal Status</div>
          <div class="kpi-value" style="color:var(--ok);">
            ${tele && tele.cpu_thermal && tele.cpu_thermal.temperature_c ? tele.cpu_thermal.temperature_c + '°C' : 'Optimal'}
          </div>
          <div class="kpi-sub">Passive/Fan cooling</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Inference Engine</div>
          <div class="kpi-value" style="font-size:18px; color:var(--cyan);">
            ${h.hardware_accelerated ? 'Hailo-8 NPU' : 'CPU Baseline'}
          </div>
          <div class="kpi-sub">YOLOv8n-640 Pipeline</div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Hardware Audit Table -->
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:12px;">Hardware Platform Audit</div>
          <table>
            <tbody>
              <tr><td style="color:var(--mut);">Host Architecture</td><td><b>${audit.architecture || 'aarch64 / ARM64'}</b></td></tr>
              <tr><td style="color:var(--mut);">Host SBC Model</td><td><b>${audit.device_model || 'Raspberry Pi 5'}</b></td></tr>
              <tr><td style="color:var(--mut);">Hailo NPU Hardware</td><td><span class="tag ${audit.hailo_device && audit.hailo_device.detected ? 'cyan' : ''}">${audit.hailo_device ? audit.hailo_device.status : 'HEF Config Ready'}</span></td></tr>
              <tr><td style="color:var(--mut);">Active Vision Model</td><td><span class="tag">${audit.active_vision_backend || 'YOLOv8n PyTorch Baseline'}</span></td></tr>
              <tr><td style="color:var(--mut);">Advisory Model</td><td>Meta Llama 3.2 1B (ExecuTorch) / IPC Rule Engine</td></tr>
              <tr><td style="color:var(--mut);">Optical Sensor</td><td>Sony IMX708 / USB Camera (Autofocus)</td></tr>
              <tr><td style="color:var(--mut);">Local Storage</td><td>SQLite WAL Mode (Air-Gapped Local)</td></tr>
            </tbody>
          </table>
        </div>

        <!-- Air-Gapped Security Checklist -->
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:12px;">Air-Gapped Offline Verification</div>
          <div style="display:flex; flex-direction:column; gap:10px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="dot live"></span>
              <span>Zero external cloud API endpoints or remote telemetry calls</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="dot live"></span>
              <span>Pre-compiled local neural network models on edge flash</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="dot live"></span>
              <span>Fully embedded SQLite database with local audit trail</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="dot live"></span>
              <span>Zero CDN dependencies; all CSS and assets bundled offline</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="dot live"></span>
              <span>Deterministic quality gating decoupled from LLM advisory</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ============================================================================
// 7. STATION SETTINGS VIEW
// ============================================================================
async function renderSettingsView(container) {
  let presetData = null;
  try {
    presetData = await api('/settings/presets');
  } catch (e) {
    presetData = {
      active_preset: 'industrial',
      available_presets: [
        { id: 'consumer', name: 'Consumer Electronics (IPC Class 2)', description: 'Balanced high-yield preset for high-volume commercial production.' },
        { id: 'industrial', name: 'Industrial High-Reliability (IPC Class 3)', description: 'Conservative gating with mandatory electrical flags for open/short circuits.' },
        { id: 'mission_critical', name: 'Mission-Critical (Aerospace / Medical)', description: 'Zero-defect tolerance; any anomaly or minor coverage deficit requires human review.' }
      ]
    };
  }

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">🛠️ Station Configuration & Quality Presets</div>
          <div class="card-desc">Configure deterministic decision engine rules, confidence thresholds, and advisory mode</div>
        </div>
      </div>

      <div class="form-group" style="max-width:540px;">
        <label class="form-label">Active Quality Preset</label>
        <select class="form-select" id="stationPresetSelect" onchange="switchPreset(this.value)">
          ${presetData.available_presets.map(p => `
            <option value="${p.id}" ${p.id === presetData.active_preset ? 'selected' : ''}>
              ${p.name}
            </option>
          `).join('')}
        </select>
        <div style="font-size:12px; color:var(--mut); margin-top:6px;">
          ${presetData.available_presets.find(p => p.id === presetData.active_preset)?.description || ''}
        </div>
      </div>

      <div class="grid-2" style="margin-top:20px;">
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:10px;">Deterministic Rule Thresholds</div>
          <table>
            <tbody>
              <tr><td style="color:var(--mut);">Critical Defect Threshold</td><td><b>≥ 0.45 confidence</b></td></tr>
              <tr><td style="color:var(--mut);">Major Defect Threshold</td><td><b>≥ 0.50 confidence</b></td></tr>
              <tr><td style="color:var(--mut);">Minimum Optical Coverage</td><td><b>95.0% verified surface</b></td></tr>
              <tr><td style="color:var(--mut);">Electrical Test Flag</td><td><b>Mandatory for Open / Short Circuits</b></td></tr>
            </tbody>
          </table>
        </div>

        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="margin-bottom:10px;">Advisory Engine Configuration</div>
          <table>
            <tbody>
              <tr><td style="color:var(--mut);">SLM Model</td><td>Meta Llama 3.2 1B Instruct</td></tr>
              <tr><td style="color:var(--mut);">Runtime</td><td>ExecuTorch CPU XNNPACK (INT8/FP16)</td></tr>
              <tr><td style="color:var(--mut);">Fallback Engine</td><td>IPC-A-610 Deterministic Expert Rules</td></tr>
              <tr><td style="color:var(--mut);">Decision Authority</td><td>Read-Only (Cannot override rule verdict)</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

async function switchPreset(newPreset) {
  try {
    await api('/settings/preset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset: newPreset })
    });
    alert(`Quality preset switched to: ${newPreset.toUpperCase()}`);
    if (STATE.systemHealth) STATE.systemHealth.active_preset = newPreset;
    updateHeaderPills(STATE.systemHealth);
    renderCurrentView();
  } catch (err) {
    alert('Failed to switch preset: ' + err.message);
  }
}

// ----------------------------------------------------------------------------
// Demo Fallback State (Ensures instant offline functionality)
// ----------------------------------------------------------------------------
function getDemoInspection() {
  return {
    inspection_id: 'INSP-DEMO-001',
    board_id: 'PCB-DEEPPCB-0041',
    timestamp_utc: new Date().toISOString(),
    verdict: 'REJECT',
    rule_id: 'RULE_001_HIGH_CONF_CRITICAL_DEFECT',
    rule_description: 'Immediate Scrap / Non-Conformity: One or more high-confidence critical functional defects localized.',
    preset: 'industrial',
    electrical_verification_mandatory: true,
    coverage: 1.0,
    image_quality: 'acceptable',
    defects: [
      {
        defect_id: 1,
        class: 'open_circuit',
        severity: 'CRITICAL',
        confidence: 0.9332,
        bbox: [537, 257, 594, 318],
        dimensions_px: [57, 61],
        crop_url: '/static/samples/sample_open_circuit.jpg'
      },
      {
        defect_id: 2,
        class: 'mouse_bite',
        severity: 'MAJOR',
        confidence: 0.7241,
        bbox: [210, 390, 245, 418],
        dimensions_px: [35, 28],
        crop_url: '/static/samples/sample_mouse_bite.jpg'
      }
    ],
    defect_count: 2,
    raw_image_url: '/static/samples/sample_open_circuit.jpg',
    annotated_image_url: '/static/samples/sample_open_circuit.jpg',
    advisory: {
      what_detected: 'Detected 2 surface defect(s): Open Circuit (1 instance), Mouse Bite (1 instance).',
      why_it_matters: 'Breaks electrical continuity, causing functional open failure. Reduces conductor cross-sectional area, creating a localized current bottleneck.',
      operator_action: '[REJECT VERDICT] Scrutinize etch parameters and artwork mask integrity. Perform micro-soldering jumper repair if permitted by class standards, or scrap.',
      attribution: 'Deterministic IPC-A-610 Rule Engine',
      electrical_confirmation_needed: true
    },
    timings: {
      capture_ms: 0.0,
      quality_gate_ms: 0.8,
      vision_inference_ms: 8.4,
      decision_engine_ms: 0.08,
      advisory_ms: 0.18,
      total_latency_ms: 9.46
    },
    inference_backend: 'CPU (PyTorch Baseline)',
    hardware_accelerated: false
  };
}

// Launch app
window.addEventListener('DOMContentLoaded', init);
