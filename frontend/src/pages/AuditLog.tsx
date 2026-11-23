import React, { useState, useEffect } from 'react';
import { api, AuditLogItem } from '../services/api';

export const AuditLog: React.FC = () => {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [filter, setFilter] = useState<string>('All');
  const [loading, setLoading] = useState<boolean>(true);

  const fetchLogs = async (sourceFilter?: string) => {
    try {
      setLoading(true);
      const sf = sourceFilter === 'All' ? undefined : sourceFilter?.toLowerCase();
      const res = await api.getAuditLogs(100, sf);
      setLogs(res || []);
    } catch (e) {
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs(filter);
  }, [filter]);

  const passCount = logs.filter((x) => x.final === 'PASS').length;
  const inspectCount = logs.filter((x) => x.final === 'INSPECT').length;
  const rejectCount = logs.filter((x) => x.final === 'REJECT').length;

  return (
    <div>
      <div className="page-title-row">
        <h1 className="page-title">Inspection Log & Tamper-Evident Audit</h1>
        <p className="page-subtitle">
          Append-only SHA-256 SQLite records complying with ISO 9001 / IPC trace standards
        </p>
      </div>

      {/* KPI Tiles */}
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Total Records</span>
            <div className="kpi-icon-box">📋</div>
          </div>
          <div className="kpi-value">{logs.length}</div>
          <div className="kpi-subtext">Filtered count</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">PASS Yield</span>
            <div className="kpi-icon-box" style={{ color: 'var(--ok)' }}>✓</div>
          </div>
          <div className="kpi-value" style={{ color: 'var(--ok)' }}>{passCount}</div>
          <div className="kpi-subtext">
            {logs.length > 0 ? `${((passCount / logs.length) * 100).toFixed(1)}% yield` : '—'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">INSPECT Flagged</span>
            <div className="kpi-icon-box" style={{ color: 'var(--warn)' }}>⚠️</div>
          </div>
          <div className="kpi-value" style={{ color: 'var(--warn)' }}>{inspectCount}</div>
          <div className="kpi-subtext">Requires review</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">REJECT Scrapped</span>
            <div className="kpi-icon-box" style={{ color: 'var(--bad)' }}>✕</div>
          </div>
          <div className="kpi-value" style={{ color: 'var(--bad)' }}>{rejectCount}</div>
          <div className="kpi-subtext">Quarantined flaws</div>
        </div>
      </div>

      {/* Audit Log Table Card */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 14 }}>
          {/* Source Filter Chips */}
          <div style={{ display: 'flex', gap: 8 }}>
            {['All', 'Real', 'Demo', 'Test'].map((f) => (
              <button
                key={f}
                className={`chip-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f} Source
              </button>
            ))}
          </div>

          <a
            className="btn btn-secondary btn-sm"
            href={api.getAuditCsvUrl(filter === 'All' ? undefined : filter.toLowerCase())}
            download="inspection_audit_export.csv"
          >
            📥 Export Audit CSV
          </a>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp (UTC)</th>
                <th>Board ID</th>
                <th>Rule Decision</th>
                <th>Final Verdict</th>
                <th>Defects</th>
                <th>Electrical Test</th>
                <th>Total Latency</th>
                <th>Override</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--mut)', padding: 30 }}>
                    Loading audit trail…
                  </td>
                </tr>
              ) : logs.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--mut)', padding: 30 }}>
                    No inspection records found for this filter.
                  </td>
                </tr>
              ) : (
                logs.map((row) => (
                  <tr key={row.inspection_id}>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{row.timestamp}</td>
                    <td style={{ fontWeight: 600, color: '#f1f5f9' }}>{row.board}</td>
                    <td>
                      <span className="pill pill-minor">{row.rule}</span>
                    </td>
                    <td>
                      <span
                        className={`pill ${
                          row.final === 'PASS'
                            ? 'pill-pass'
                            : row.final === 'INSPECT'
                            ? 'pill-inspect'
                            : 'pill-reject'
                        }`}
                      >
                        {row.final}
                      </span>
                    </td>
                    <td>{row.defects}</td>
                    <td>
                      {row.electrical ? (
                        <span className="pill pill-reject">PROBE MANDATORY</span>
                      ) : (
                        <span style={{ color: 'var(--mut)' }}>—</span>
                      )}
                    </td>
                    <td style={{ fontFamily: 'monospace' }}>{row.latency_ms.toFixed(1)} ms</td>
                    <td>
                      {row.overridden ? (
                        <span className="pill pill-major">OVERRIDDEN</span>
                      ) : (
                        <span style={{ color: 'var(--mut)' }}>—</span>
                      )}
                    </td>
                    <td style={{ fontSize: 11, color: 'var(--mut)', textTransform: 'uppercase' }}>
                      {row.source}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
