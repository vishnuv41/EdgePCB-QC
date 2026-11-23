import React, { useState } from 'react';
import { api, SystemHealth, InspectionRecord, SampleBoard } from '../services/api';

interface LiveStationProps {
  health: SystemHealth | null;
  cameraActive: boolean;
  onToggleCamera: (start: boolean, resolution?: string) => Promise<void>;
  onInspectionComplete: (rec: InspectionRecord) => void;
  samples: SampleBoard[];
}

export const LiveStation: React.FC<LiveStationProps> = ({
  health,
  cameraActive,
  onToggleCamera,
  onInspectionComplete,
  samples,
}) => {
  const [resolution, setResolution] = useState<string>('1280x720');
  const [busy, setBusy] = useState<boolean>(false);
  const [camLoading, setCamLoading] = useState<boolean>(false);

  const handleStartCam = async () => {
    try {
      setCamLoading(true);
      await onToggleCamera(true, resolution);
    } catch (err: any) {
      alert(`Could not start camera: ${err.message}`);
    } finally {
      setCamLoading(false);
    }
  };

  const handleStopCam = async () => {
    try {
      setCamLoading(true);
      await onToggleCamera(false);
    } catch (err: any) {
      alert(`Could not stop camera: ${err.message}`);
    } finally {
      setCamLoading(false);
    }
  };

  const handleCapture = async () => {
    try {
      setBusy(true);
      const rec = await api.captureCamera();
      onInspectionComplete(rec);
    } catch (err: any) {
      alert(`Capture failed: ${err.message}. If camera is in standby or disconnected, you can select one of the test boards below.`);
    } finally {
      setBusy(false);
    }
  };

  const handleRunSample = async (filename: string) => {
    try {
      setBusy(true);
      const rec = await api.inspectSample(filename);
      onInspectionComplete(rec);
    } catch (err: any) {
      alert(`Sample run failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="page-title-row">
        <h1 className="page-title">Live Camera Inspection</h1>
        <p className="page-subtitle">
          Benchtop optical station with hardware capture control and on-device YOLO defect screening
        </p>
      </div>

      {/* KPI Tiles */}
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Camera Status</span>
            <div className="kpi-icon-box">📹</div>
          </div>
          <div className="kpi-value" style={{ fontSize: 22, color: cameraActive ? 'var(--ok)' : 'var(--warn)' }}>
            {cameraActive ? 'STREAMING' : 'STANDBY'}
          </div>
          <div className="kpi-subtext">
            {health?.camera_info?.status || 'Controlled via button'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Target Framerate</span>
            <div className="kpi-icon-box">⚡</div>
          </div>
          <div className="kpi-value">
            {cameraActive ? '15 FPS' : '0 FPS'}
          </div>
          <div className="kpi-subtext">Optimized for Raspberry Pi 5</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Vision Backend</span>
            <div className="kpi-icon-box">🧠</div>
          </div>
          <div className="kpi-value" style={{ fontSize: 20 }}>
            {health?.active_backend?.toUpperCase() || 'CPU'}
          </div>
          <div className="kpi-subtext">YOLOv8n [best.pt]</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Network Isolation</span>
            <div className="kpi-icon-box">🔒</div>
          </div>
          <div className="kpi-value" style={{ fontSize: 20, color: 'var(--cyan)' }}>
            AIR-GAPPED
          </div>
          <div className="kpi-subtext">Zero cloud dependencies</div>
        </div>
      </div>

      {/* Main Viewport & Controls */}
      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(300px, 1fr)', gap: 20 }}>
        <div className="card">
          <h2>Optical Viewport</h2>
          <p className="card-desc">Live preview from bench camera</p>

          <div className="live-viewport-box">
            {cameraActive ? (
              <img
                src={api.getCameraFeedUrl()}
                alt="Live Camera Feed"
                className="live-feed-img"
              />
            ) : (
              <div className="standby-placeholder">
                <div style={{ fontSize: 44, marginBottom: 12 }}>📷</div>
                <div style={{ fontSize: 16, fontWeight: 600, color: '#f1f5f9', marginBottom: 6 }}>
                  Camera is in Standby
                </div>
                <div style={{ fontSize: 13, maxWidth: 360, margin: '0 auto' }}>
                  Click "Start Camera" below to initiate the video stream at your chosen resolution.
                </div>
              </div>
            )}
          </div>

          {/* Controls Bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
            {!cameraActive ? (
              <button
                className="btn btn-primary"
                onClick={handleStartCam}
                disabled={camLoading}
              >
                {camLoading ? 'Starting…' : '▶ Start Camera'}
              </button>
            ) : (
              <button
                className="btn btn-secondary"
                onClick={handleStopCam}
                disabled={camLoading}
              >
                {camLoading ? 'Stopping…' : '⏹ Stop Camera'}
              </button>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label htmlFor="cam-res-select" style={{ fontSize: 12, color: 'var(--mut)', fontWeight: 600 }}>
                Resolution:
              </label>
              <select
                id="cam-res-select"
                className="form-select"
                style={{ width: 'auto' }}
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                disabled={cameraActive}
              >
                <option value="640x640">640 × 640 (Square Fast)</option>
                <option value="1280x720">1280 × 720 (720p Standard)</option>
                <option value="1920x1080">1920 × 1080 (1080p High Res)</option>
              </select>
            </div>

            <button
              className="btn btn-primary"
              style={{ marginLeft: 'auto' }}
              onClick={handleCapture}
              disabled={busy || !cameraActive}
            >
              {busy ? 'Inspecting Board…' : '⚡ Capture & Run Inspection'}
            </button>
          </div>
        </div>

        {/* Offline Demo Boards Sidebar Card */}
        <div className="card">
          <h2>Offline Curated Test Boards</h2>
          <p className="card-desc">Execute pipeline without physical camera attached</p>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {samples.map((s) => (
              <button
                key={s.filename}
                className="btn btn-secondary"
                style={{ justifyContent: 'flex-start', textAlign: 'left', padding: '10px 14px' }}
                disabled={busy}
                onClick={() => handleRunSample(s.filename)}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#fff' }}>{s.title}</div>
                  <div style={{ fontSize: 11, color: 'var(--mut)' }}>{s.board_id}</div>
                </div>
              </button>
            ))}
          </div>

          <div style={{ marginTop: 20, padding: 12, background: '#101625', borderRadius: 8, fontSize: 12, color: 'var(--mut)' }}>
            <b>Operator Tip:</b> When bench lighting changes or board moves, capture a test board to verify coordinate alignment before production.
          </div>
        </div>
      </div>
    </div>
  );
};
