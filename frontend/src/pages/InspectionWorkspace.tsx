import React, { useState, useEffect } from 'react';
import { api, InspectionRecord, SampleBoard } from '../services/api';
import { DefectMagnifier } from '../components/DefectMagnifier';

interface InspectionWorkspaceProps {
  currentInspection: InspectionRecord | null;
  onSetInspection: (rec: InspectionRecord) => void;
}

export const InspectionWorkspace: React.FC<InspectionWorkspaceProps> = ({
  currentInspection,
  onSetInspection,
}) => {
  const [samples, setSamples] = useState<SampleBoard[]>([]);
  const [selectedDefectIdx, setSelectedDefectIdx] = useState<number>(0);
  const [imageTab, setImageTab] = useState<'annotated' | 'raw'>('annotated');
  const [loading, setLoading] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<string>('');
  
  // Supervisor override state
  const [overrideVerdict, setOverrideVerdict] = useState<string>('PASS');
  const [supervisorId, setSupervisorId] = useState<string>('LEAD-INSPECTOR-01');
  const [overrideReason, setOverrideReason] = useState<string>('');
  const [overrideSuccess, setOverrideSuccess] = useState<string>('');

  useEffect(() => {
    api.getSamples().then((res) => {
      if (res && res.samples) {
        setSamples(res.samples);
      }
    }).catch(() => {});
  }, []);

  const handleInspectSample = async (filename: string) => {
    try {
      setLoading(true);
      setStatusMsg(`Inspecting ${filename}…`);
      const rec = await api.inspectSample(filename);
      onSetInspection(rec);
      setSelectedDefectIdx(0);
      setStatusMsg('');
    } catch (err: any) {
      alert(`Inspection failed: ${err.message}`);
      setStatusMsg('');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setLoading(true);
      setStatusMsg(`Uploading and analyzing ${file.name}…`);
      const rec = await api.uploadAndInspect(file);
      onSetInspection(rec);
      setSelectedDefectIdx(0);
      setStatusMsg('');
    } catch (err: any) {
      alert(`Upload inspection failed: ${err.message}`);
      setStatusMsg('');
    } finally {
      setLoading(false);
    }
  };

  const handleOverrideSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentInspection) return;
    if (!overrideReason.trim()) {
      alert('A justification rationale is mandatory to record an engineering override.');
      return;
    }

    try {
      await api.overrideInspection(
        currentInspection.inspection_id,
        overrideVerdict,
        overrideReason,
        supervisorId
      );
      setOverrideSuccess(`Override recorded: Verdict updated to ${overrideVerdict}`);
      const updated = await api.getInspection(currentInspection.inspection_id);
      onSetInspection(updated);
      setTimeout(() => setOverrideSuccess(''), 4000);
    } catch (err: any) {
      alert(`Override failed: ${err.message}`);
    }
  };

  const r = currentInspection;

  return (
    <div>
      <div className="page-title-row">
        <h1 className="page-title">Captured PCB Inspection</h1>
        <p className="page-subtitle">
          Hardware AOI workspace with high-precision YOLO defect localization, advisory & decision audit
        </p>
      </div>

      {/* Curated Test Boards Strip */}
      <div className="test-boards-strip">
        <span className="test-boards-label">Or run a curated test board:</span>
        {samples.map((s) => (
          <button
            key={s.filename}
            className={`chip-btn ${r?.board_id === s.board_id ? 'active' : ''}`}
            disabled={loading}
            onClick={() => handleInspectSample(s.filename)}
          >
            {s.title}
          </button>
        ))}
      </div>

      {/* File Upload Dropzone */}
      <div className="card" style={{ marginBottom: 20 }}>
        <label className="dropzone" htmlFor="pcb-file-input">
          <input
            id="pcb-file-input"
            type="file"
            accept="image/png,image/jpeg,image/bmp"
            style={{ display: 'none' }}
            disabled={loading}
            onChange={handleFileUpload}
          />
          <div className="dropzone-title">
            {loading ? statusMsg : 'Drop a bare PCB image or click to browse'}
          </div>
          <div className="dropzone-subtitle">
            Accepts PNG, JPG, or BMP images. YOLOv8n inference executes on-device with zero cloud telemetry.
          </div>
        </label>
      </div>

      {/* Initial Empty State: When app is started, no board is displayed until started */}
      {!r ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--mut)' }}>
          <div style={{ fontSize: 42, marginBottom: 12 }}>🔍</div>
          <h2 style={{ color: '#fff', marginBottom: 6 }}>No PCB Image Inspected Yet</h2>
          <p style={{ maxWidth: 520, margin: '0 auto 18px', fontSize: 13 }}>
            Upload a board image above, capture from the Live Camera station, or click one of the curated test boards to run the inspection pipeline.
          </p>
        </div>
      ) : (
        <div>
          {/* Verdict Banner */}
          <div className={`verdict-box verdict-${r.verdict}`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
              <div>
                <div className="verdict-label">Final Decision · {r.board_id} · {r.inspection_id}</div>
                <div className="verdict-title">{r.verdict}</div>
                <div className="verdict-rule-desc">
                  <b>{r.rule_id}:</b> {r.rule_description}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <span className="pill pill-minor" style={{ fontSize: 13, padding: '4px 12px' }}>
                  {r.defects.length} defect(s) detected
                </span>
                <span className="pill pill-minor" style={{ fontSize: 13, padding: '4px 12px' }}>
                  {r.timings.total_latency_ms.toFixed(1)} ms total
                </span>
              </div>
            </div>
          </div>

          {/* Electrical Test Alert */}
          {r.requires_electrical_test ? (
            <div className="alert-box alert-danger">
              <span style={{ fontSize: 18 }}>⚡</span>
              <div>
                <b>ELECTRICAL VERIFICATION MANDATORY.</b> Critical copper defects detected. Run a flying-probe or DMM continuity test before rework.
              </div>
            </div>
          ) : (
            <div className="alert-box alert-info">
              <span style={{ fontSize: 16 }}>ℹ️</span>
              <div>
                Optical AOI verifies surface geometry. Standard continuity testing protocol still applies.
              </div>
            </div>
          )}

          {/* Workspace Main Grid */}
          <div className="workspace-grid">
            {/* Left Column: Direct Fitted PCB Viewer */}
            <div>
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                  <div className="tab-toggle-group">
                    <button
                      className={`tab-btn ${imageTab === 'annotated' ? 'active' : ''}`}
                      onClick={() => setImageTab('annotated')}
                    >
                      Annotated
                    </button>
                    <button
                      className={`tab-btn ${imageTab === 'raw' ? 'active' : ''}`}
                      onClick={() => setImageTab('raw')}
                    >
                      Raw Image
                    </button>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--mut)', fontFamily: 'monospace' }}>
                    {r.image_dimensions ? `${r.image_dimensions[0]} × ${r.image_dimensions[1]} px` : '640 × 640 px'}
                  </div>
                </div>

                {/* Direct Image Fit: Clean layout, no mousewheel zoom, no gaps */}
                <div className="pcb-view-container">
                  <img
                    src={imageTab === 'annotated' ? r.annotated_image_url : r.raw_image_url}
                    alt="Inspected PCB Board"
                    className="pcb-view-img"
                  />
                </div>
              </div>

              {/* Engineering Advisory 3 Cards */}
              <div className="card" style={{ marginTop: 20 }}>
                <h2>Engineering Advisory</h2>
                <p className="card-desc">Deterministic IPC-A-610 decision rationale & repair protocol</p>
                
                <div className="advisory-grid">
                  <div className="advisory-card">
                    <h4>🔍 What Was Detected</h4>
                    <p>{r.advisory.what_detected}</p>
                  </div>
                  <div className="advisory-card">
                    <h4>⚠️ Why It Matters</h4>
                    <p>{r.advisory.why_it_matters}</p>
                  </div>
                  <div className="advisory-card">
                    <h4>🛠️ Operator Action</h4>
                    <p>{r.advisory.operator_action}</p>
                  </div>
                </div>

                <div style={{ fontSize: 11.5, color: 'var(--mut)', background: '#101625', padding: '10px 14px', borderRadius: 8 }}>
                  <b>Advisory Caveat:</b> {r.advisory.confidence_caveat}
                </div>
              </div>

              {/* Defect Manifest Table */}
              <div className="card" style={{ marginTop: 20 }}>
                <h2>Defect Manifest</h2>
                <p className="card-desc">Complete localized flaw inventory with bounding coordinates</p>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Class</th>
                        <th>Severity</th>
                        <th>Confidence</th>
                        <th>Bounding Box [x1, y1, x2, y2]</th>
                        <th>Size (px)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.defects.length === 0 ? (
                        <tr>
                          <td colSpan={6} style={{ textAlign: 'center', color: 'var(--mut)', padding: 20 }}>
                            No defects recorded on this board.
                          </td>
                        </tr>
                      ) : (
                        r.defects.map((d, idx) => (
                          <tr
                            key={d.defect_id}
                            style={{
                              cursor: 'pointer',
                              backgroundColor: idx === selectedDefectIdx ? 'rgba(56, 189, 248, 0.08)' : 'transparent',
                            }}
                            onClick={() => setSelectedDefectIdx(idx)}
                          >
                            <td style={{ fontWeight: 700, color: 'var(--cyan)' }}>#{d.defect_id}</td>
                            <td style={{ textTransform: 'capitalize' }}>{d.class.replace('_', ' ')}</td>
                            <td>
                              <span
                                className={`pill ${
                                  d.severity === 'CRITICAL'
                                    ? 'pill-critical'
                                    : d.severity === 'MAJOR'
                                    ? 'pill-major'
                                    : 'pill-minor'
                                }`}
                              >
                                {d.severity}
                              </span>
                            </td>
                            <td style={{ fontFamily: 'monospace' }}>{(d.confidence * 100).toFixed(1)}%</td>
                            <td style={{ fontFamily: 'monospace' }}>[{d.bbox.join(', ')}]</td>
                            <td style={{ fontFamily: 'monospace' }}>
                              {d.dimensions_px[0]} × {d.dimensions_px[1]}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Right Column: Pipeline Execution, Defect Magnifier & Supervisor Override */}
            <div>
              {/* Pipeline Execution */}
              <div className="card" style={{ marginBottom: 20 }}>
                <h2>Pipeline Execution</h2>
                <p className="card-desc">
                  {r.board_id} · Total Latency: {r.timings.total_latency_ms.toFixed(2)} ms
                </p>
                <div className="pipeline-steps-grid">
                  <div className="pipeline-step-item">
                    <div className="pipeline-step-check">✓</div>
                    <div className="pipeline-step-label">Capture</div>
                    <div className="pipeline-step-ms">{r.timings.capture_ms.toFixed(1)} ms</div>
                  </div>
                  <div className="pipeline-step-item">
                    <div className="pipeline-step-check">✓</div>
                    <div className="pipeline-step-label">Quality</div>
                    <div className="pipeline-step-ms">{r.timings.quality_gate_ms.toFixed(1)} ms</div>
                  </div>
                  <div className="pipeline-step-item">
                    <div className="pipeline-step-check">✓</div>
                    <div className="pipeline-step-label">Inference</div>
                    <div className="pipeline-step-ms">{r.timings.vision_inference_ms.toFixed(1)} ms</div>
                  </div>
                  <div className="pipeline-step-item">
                    <div className="pipeline-step-check">✓</div>
                    <div className="pipeline-step-label">Decision</div>
                    <div className="pipeline-step-ms">{r.timings.decision_engine_ms.toFixed(2)} ms</div>
                  </div>
                  <div className="pipeline-step-item">
                    <div className="pipeline-step-check">✓</div>
                    <div className="pipeline-step-label">Advisory</div>
                    <div className="pipeline-step-ms">{r.timings.advisory_ms.toFixed(2)} ms</div>
                  </div>
                </div>
              </div>

              {/* Defect Magnifier Component */}
              <DefectMagnifier
                defects={r.defects}
                selectedIndex={selectedDefectIdx}
                onSelectDefect={(idx) => setSelectedDefectIdx(idx)}
                inspectionId={r.inspection_id}
              />

              {/* Supervisor Override */}
              <div className="card" style={{ marginTop: 20 }}>
                <h2>Supervisor Override</h2>
                <p className="card-desc">Concession sign-off requiring mandatory rationale</p>

                {overrideSuccess && (
                  <div className="alert-box alert-info" style={{ color: 'var(--ok)', borderColor: 'rgba(16,185,129,0.3)' }}>
                    {overrideSuccess}
                  </div>
                )}

                <form onSubmit={handleOverrideSubmit}>
                  <div className="form-group">
                    <label className="form-label" htmlFor="override-verdict">New Verdict</label>
                    <select
                      id="override-verdict"
                      className="form-select"
                      value={overrideVerdict}
                      onChange={(e) => setOverrideVerdict(e.target.value)}
                    >
                      <option value="PASS">PASS (Accept board)</option>
                      <option value="INSPECT">INSPECT (Rework / Re-screen)</option>
                      <option value="REJECT">REJECT (Scrap / Quarantine)</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="override-supervisor">Supervisor / Operator ID</label>
                    <input
                      id="override-supervisor"
                      type="text"
                      className="form-input"
                      value={supervisorId}
                      onChange={(e) => setSupervisorId(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="override-reason">Justification (Required)</label>
                    <textarea
                      id="override-reason"
                      className="form-textarea"
                      rows={2}
                      placeholder="Engineering concession justification…"
                      value={overrideReason}
                      onChange={(e) => setOverrideReason(e.target.value)}
                    />
                  </div>

                  <button type="submit" className="btn btn-secondary" style={{ width: '100%' }}>
                    Record Override
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
