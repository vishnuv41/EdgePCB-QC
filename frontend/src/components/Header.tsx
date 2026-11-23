import React, { useState, useEffect } from 'react';
import { SystemHealth } from '../services/api';

interface HeaderProps {
  health: SystemHealth | null;
  cameraActive: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, cameraActive }) => {
  const [timeUtc, setTimeUtc] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0] + ' UTC';
      setTimeUtc(timeStr);
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const stationLabel = health?.station_label || 'EdgePCB-QC Industrial Inspection';
  const stationId = health?.station_id ? health.station_id.replace('-AOI', '').replace('-', ' ') : 'STATION 01';
  const backend = health?.active_backend ? `${health.active_backend.toUpperCase()} Pipeline` : 'CPU Pipeline';
  const preset = health?.active_preset ? health.active_preset.charAt(0).toUpperCase() + health.active_preset.slice(1) : 'Industrial';

  return (
    <header className="top-header">
      <div className="header-left">
        <span className="station-pill">{stationId}</span>
        <span className="header-station-name">{stationLabel}</span>

        <span className={`header-tag-pill ${cameraActive ? 'cam-active' : 'cam-standby'}`}>
          <span
            style={{
              display: 'inline-block',
              width: 7,
              height: 7,
              borderRadius: '50%',
              backgroundColor: cameraActive ? '#10b981' : '#f59e0b',
              boxShadow: cameraActive ? '0 0 6px #10b981' : '0 0 6px #f59e0b',
            }}
          />
          {cameraActive ? 'CAMERA ACTIVE' : 'CAMERA STANDBY'}
        </span>

        <span className="header-tag-pill">{backend}</span>
        <span className="header-tag-pill">YOLOv8n [best.pt]</span>
        <span className="header-tag-pill">Preset: {preset}</span>
      </div>

      <div className="header-right">
        <span className="header-tag-pill" style={{ color: '#38bdf8', borderColor: 'rgba(56,189,248,0.3)' }}>
          <span className="status-dot-indicator" />
          Air-Gapped Offline
        </span>
        <span className="header-clock">{timeUtc}</span>
      </div>
    </header>
  );
};
