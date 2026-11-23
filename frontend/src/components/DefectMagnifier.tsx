import React from 'react';
import { Defect } from '../services/api';

interface DefectMagnifierProps {
  defects: Defect[];
  selectedIndex: number;
  onSelectDefect: (index: number) => void;
  inspectionId: string;
}

export const DefectMagnifier: React.FC<DefectMagnifierProps> = ({
  defects,
  selectedIndex,
  onSelectDefect,
  inspectionId,
}) => {
  if (!defects || defects.length === 0) {
    return (
      <div className="card">
        <h2>Defect Magnifier</h2>
        <p className="card-desc">Detailed defect crop and coordinates</p>
        <div className="magnifier-view">
          <div className="magnifier-placeholder">
            No defects detected on this board. Optical inspection clear.
          </div>
        </div>
      </div>
    );
  }

  const selectedDefect = defects[selectedIndex] || defects[0];
  const defectId = selectedDefect.defect_id;
  const cropUrl = `/api/v1/inspections/${inspectionId}/defects/${defectId}/crop`;

  const getSeverityClass = (sev: string) => {
    switch (sev) {
      case 'CRITICAL': return 'pill pill-critical';
      case 'MAJOR': return 'pill pill-major';
      default: return 'pill pill-minor';
    }
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <h2>Defect Magnifier</h2>
        <span className="pill pill-minor">
          Defect #{defectId} of {defects.length}
        </span>
      </div>
      <p className="card-desc" style={{ marginBottom: 10 }}>
        Centered context crop for optical verification
      </p>

      {/* Prominent Defect Crop Viewport */}
      <div className="magnifier-view">
        <img
          src={cropUrl}
          alt={`Defect ${defectId}: ${selectedDefect.class}`}
          className="magnifier-crop-img"
          onError={(e) => {
            // Fallback if crop endpoint takes a moment
            (e.target as HTMLElement).style.display = 'none';
          }}
        />
      </div>

      {/* Technical Defect Manifest */}
      <dl className="kv-table">
        <dt>Class</dt>
        <dd style={{ textTransform: 'capitalize' }}>{selectedDefect.class.replace('_', ' ')}</dd>

        <dt>Severity</dt>
        <dd>
          <span className={getSeverityClass(selectedDefect.severity)}>
            {selectedDefect.severity}
          </span>
        </dd>

        <dt>Confidence</dt>
        <dd>{(selectedDefect.confidence * 100).toFixed(1)}%</dd>

        <dt>Size</dt>
        <dd>
          {selectedDefect.dimensions_px[0]} × {selectedDefect.dimensions_px[1]} px
        </dd>

        <dt>Coordinates</dt>
        <dd>[{selectedDefect.bbox.join(', ')}]</dd>
      </dl>

      <div style={{ marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--mut)', marginBottom: 2 }}>
          <span>Detection Confidence</span>
          <span>{(selectedDefect.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="progress-bar-wrap">
          <div
            className="progress-bar-fill"
            style={{
              width: `${Math.min(100, selectedDefect.confidence * 100)}%`,
              backgroundColor: selectedDefect.severity === 'CRITICAL' ? 'var(--bad)' : 'var(--warn)',
            }}
          />
        </div>
      </div>

      {/* Defect Quick Selector */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#e2e8f0', marginBottom: 6 }}>
          All Detected Defects ({defects.length})
        </div>
        <div className="defect-list">
          {defects.map((d, idx) => {
            const isSel = idx === selectedIndex;
            return (
              <button
                key={d.defect_id}
                className={`defect-item-btn ${isSel ? 'selected' : ''}`}
                onClick={() => onSelectDefect(idx)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ color: 'var(--cyan)', fontWeight: 700, fontSize: 12 }}>#{d.defect_id}</span>
                  <span className="defect-item-name">{d.class.replace('_', ' ')}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className={getSeverityClass(d.severity)}>{d.severity}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--mut)' }}>
                    {(d.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
