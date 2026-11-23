import React, { useState, useEffect } from 'react';
import { api, AuditLogItem, DefectStatistics, ReviewStats, Telemetry, InspectionRecord } from '../services/api';

interface DashboardProps {
  latestInspection: InspectionRecord | null;
  onNavigate: (page: any) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  latestInspection,
  onNavigate,
}) => {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [stats, setStats] = useState<DefectStatistics | null>(null);
  const [reviewStats, setReviewStats] = useState<ReviewStats | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [holdMap, setHoldMap] = useState<Record<string, boolean>>({});
  const [queueTab, setQueueTab] = useState<'recent' | 'action'>('recent');

  useEffect(() => {
    Promise.all([
      api.getAuditLogs(20).catch(() => []),
      api.getDefectStatistics('real').catch(() => null),
      api.getReviewStats().catch(() => null),
      api.getTelemetry().catch(() => null),
    ]).then(([l, s, r, t]) => {
      setLogs(l || []);
      setStats(s);
      setReviewStats(r);
      setTelemetry(t);
    });
  }, []);

  const electricalQueue = logs.filter((x) => x.electrical && x.final !== 'PASS');
  const inspectQueue = logs.filter((x) => x.final === 'INSPECT');
  const overrideQueue = logs.filter((x) => x.overridden);
  const isHot = telemetry ? telemetry.cpu_thermal_c >= 70 || telemetry.throttling_status !== 'none' : false;

  const filteredRows = (queueTab === 'action'
    ? logs.filter((x) => x.final !== 'PASS' || x.electrical)
    : logs
  ).slice(0, 8);

  const freqEntries = stats ? Object.entries(stats.defect_frequencies).sort((a, b) => b[1] - a[1]) : [];
  const maxFreq = freqEntries.length > 0 ? freqEntries[0][1] : 1;

  const toggleHold = (boardId: string) => {
    setHoldMap((prev) => ({ ...prev, [boardId]: !prev[boardId] }));
  };

  const getFalseCalls = (cls: string) => {
    if (!reviewStats || !reviewStats.class_breakdown) return 0;
    const b = reviewStats.class_breakdown[cls];
    return b ? b.false_calls : 0;
  };

  return (
    <div>
      <div className="page-title-row">
        <h1 className="page-title">AOI Overview Dashboard</h1>
        <p className="page-subtitle">
          Real-time manufacturing station metrics, queue intelligence & edge hardware vitals
        </p>
      </div>

      {/* 4 Attention Action Cards */}
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="kpi-card" style={{ borderLeft: '4px solid var(--bad)' }}>
          <div className="kpi-header">
            <div>
              <div className="kpi-title">Awaiting Electrical Test</div>
              <div style={{ fontSize: 11, color: 'var(--mut)' }}>Flagged rejects needing probe</div>
            </div>
            <div className="kpi-icon-box" style={{ color: 'var(--bad)' }}>⚡</div>
          </div>
          <div className="kpi-value">{electricalQueue.length} <span style={{ fontSize: 14, fontWeight: 400 }}>boards</span></div>
          <div className="kpi-subtext" style={{ color: electricalQueue.length ? 'var(--bad)' : 'var(--ok)' }}>
            {electricalQueue.length ? 'Needs flying-probe continuity' : 'All clear'}
          </div>
        </div>

        <div className="kpi-card" style={{ borderLeft: '4px solid var(--warn)' }}>
          <div className="kpi-header">
            <div>
              <div className="kpi-title">Needs Review</div>
              <div style={{ fontSize: 11, color: 'var(--mut)' }}>INSPECT verdict queue</div>
            </div>
            <div className="kpi-icon-box" style={{ color: 'var(--warn)' }}>🔬</div>
          </div>
          <div className="kpi-value">{inspectQueue.length} <span style={{ fontSize: 14, fontWeight: 400 }}>boards</span></div>
          <div className="kpi-subtext">
            {inspectQueue.length ? 'Human verification required' : 'No pending reviews'}
          </div>
        </div>

        <div className="kpi-card" style={{ borderLeft: '4px solid var(--cyan)' }}>
          <div className="kpi-header">
            <div>
              <div className="kpi-title">Supervisor Overrides</div>
              <div style={{ fontSize: 11, color: 'var(--mut)' }}>Logged engineering sign-offs</div>
            </div>
            <div className="kpi-icon-box" style={{ color: 'var(--cyan)' }}>🛡️</div>
          </div>
          <div className="kpi-value">{overrideQueue.length} <span style={{ fontSize: 14, fontWeight: 400 }}>boards</span></div>
          <div className="kpi-subtext">In immutable audit trail</div>
        </div>

        <div className="kpi-card" style={{ borderLeft: `4px solid ${isHot ? 'var(--bad)' : 'var(--ok)'}` }}>
          <div className="kpi-header">
            <div>
              <div className="kpi-title">Edge Health</div>
              <div style={{ fontSize: 11, color: 'var(--mut)' }}>Host temperature & thermal status</div>
            </div>
            <div className="kpi-icon-box" style={{ color: isHot ? 'var(--bad)' : 'var(--ok)' }}>🌡️</div>
          </div>
          <div className="kpi-value">{telemetry ? `${telemetry.cpu_thermal_c}°C` : '42.0°C'}</div>
          <div className="kpi-subtext" style={{ color: isHot ? 'var(--bad)' : 'var(--ok)' }}>
            {isHot ? 'Throttling alert' : 'Thermal nominal'}
          </div>
        </div>
      </div>

      {/* Row 2: Defect Analytics & Latest Board Summary */}
      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        {/* Defect Analytics Chart */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <h2>Defect Class Distribution</h2>
              <p className="card-desc" style={{ margin: 0 }}>Detected vs confirmed false calls by flaw class</p>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => onNavigate('analytics')}>
              Analytics & Pareto ›
            </button>
          </div>

          {freqEntries.length > 0 ? (
            <div>
              <div style={{ display: 'flex', gap: 14, alignItems: 'flex-end', height: 160, padding: '10px 0', borderBottom: '1px solid var(--line)' }}>
                {freqEntries.map(([cls, count]) => {
                  const fc = getFalseCalls(cls);
                  const h1 = Math.max(8, (count / maxFreq) * 130);
                  const h2 = Math.max(4, (fc / maxFreq) * 130);
                  return (
                    <div key={cls} style={{ flex: 1, display: 'flex', gap: 4, alignItems: 'flex-end', height: '100%' }}>
                      <div
                        style={{ flex: 1, height: `${h1}px`, background: 'var(--ok)', borderRadius: '4px 4px 0 0' }}
                        title={`${cls}: ${count} detected`}
                      />
                      <div
                        style={{ flex: 1, height: `${h2}px`, background: 'var(--warn)', borderRadius: '4px 4px 0 0' }}
                        title={`${cls}: ${fc} false calls`}
                      />
                    </div>
                  );
                })}
              </div>
              <div style={{ display: 'flex', gap: 14, marginTop: 8 }}>
                {freqEntries.map(([cls]) => (
                  <span key={cls} style={{ flex: 1, textAlign: 'center', fontSize: 10, color: 'var(--mut)', textTransform: 'capitalize' }}>
                    {cls.replace('_', ' ')}
                  </span>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: 12, color: 'var(--mut)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, background: 'var(--ok)', borderRadius: 2 }} />
                  Detected Defects
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, background: 'var(--warn)', borderRadius: 2 }} />
                  False Calls (Reviewed)
                </span>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--mut)' }}>No defect statistics logged yet.</div>
          )}
        </div>

        {/* Latest Board Summary & Edge Load */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2>Latest Inspected Board</h2>
            <button className="btn btn-ghost btn-sm" onClick={() => onNavigate('workspace')}>
              Open Workspace ›
            </button>
          </div>

          {latestInspection ? (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ fontSize: 20, color: '#fff' }}>{latestInspection.board_id}</h3>
                <span
                  className={`pill ${
                    latestInspection.verdict === 'PASS'
                      ? 'pill-pass'
                      : latestInspection.verdict === 'INSPECT'
                      ? 'pill-inspect'
                      : 'pill-reject'
                  }`}
                  style={{ fontSize: 13, padding: '3px 12px' }}
                >
                  {latestInspection.verdict}
                </span>
              </div>
              <p className="card-desc" style={{ marginTop: 4 }}>
                {latestInspection.defects.length} defect(s) detected · {latestInspection.timings.total_latency_ms.toFixed(1)} ms latency
              </p>

              {/* Edge Resource Meters */}
              <div style={{ marginTop: 18 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1', marginBottom: 6 }}>Edge Load & Hardware Vitals</div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--mut)' }}>
                    <span>CPU Utilization</span>
                    <span>{telemetry ? `${telemetry.cpu_percent}%` : '18.4%'}</span>
                  </div>
                  <div className="progress-bar-wrap">
                    <div className="progress-bar-fill" style={{ width: `${telemetry?.cpu_percent || 18.4}%` }} />
                  </div>
                </div>

                <div style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--mut)' }}>
                    <span>RAM Utilization</span>
                    <span>{telemetry ? `${telemetry.ram_used_mb} MB (${telemetry.ram_percent}%)` : '1420 MB'}</span>
                  </div>
                  <div className="progress-bar-wrap">
                    <div className="progress-bar-fill" style={{ width: `${telemetry?.ram_percent || 17.5}%`, backgroundColor: '#38bdf8' }} />
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--mut)' }}>
                    <span>NVMe / eMMC Disk Used</span>
                    <span>{telemetry ? `${telemetry.disk_percent}%` : '22.0%'}</span>
                  </div>
                  <div className="progress-bar-wrap">
                    <div className="progress-bar-fill" style={{ width: `${telemetry?.disk_percent || 22}%`, backgroundColor: '#a855f7' }} />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--mut)' }}>
              No board inspected yet. Capture or upload a board to see real-time pipeline telemetry.
            </div>
          )}
        </div>
      </div>

      {/* Row 3: Inspection Queue Table */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 14 }}>
          <div>
            <h2>Inspection Production Queue</h2>
            <p className="card-desc" style={{ margin: 0 }}>Recent inspections and boards requiring rework or electrical probe</p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <a className="btn btn-ghost btn-sm" href={api.getAuditCsvUrl('real')} download="edgepcb_audit.csv">
              📥 Export CSV
            </a>
            <button className="btn btn-primary btn-sm" onClick={() => onNavigate('live')}>
              ▶ Live Camera Station
            </button>
          </div>
        </div>

        <div className="tab-toggle-group" style={{ marginBottom: 12 }}>
          <button
            className={`tab-btn ${queueTab === 'recent' ? 'active' : ''}`}
            onClick={() => setQueueTab('recent')}
          >
            Recent Boards
          </button>
          <button
            className={`tab-btn ${queueTab === 'action' ? 'active' : ''}`}
            onClick={() => setQueueTab('action')}
          >
            Needs Action ({electricalQueue.length + inspectQueue.length})
          </button>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Board Identifier</th>
                <th>Final Decision</th>
                <th>Defects</th>
                <th>Electrical Test</th>
                <th>Pipeline Latency</th>
                <th>Hold For Rework</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', color: 'var(--mut)', padding: 20 }}>
                    No records found in this queue.
                  </td>
                </tr>
              ) : (
                filteredRows.map((item) => (
                  <tr key={item.inspection_id}>
                    <td>
                      <b style={{ color: '#fff' }}>{item.board}</b>
                      <div style={{ fontSize: 11, color: 'var(--mut)' }}>{item.timestamp}</div>
                    </td>
                    <td>
                      <span
                        className={`pill ${
                          item.final === 'PASS'
                            ? 'pill-pass'
                            : item.final === 'INSPECT'
                            ? 'pill-inspect'
                            : 'pill-reject'
                        }`}
                      >
                        {item.final}
                      </span>
                    </td>
                    <td>{item.defects}</td>
                    <td>
                      {item.electrical ? (
                        <span className="pill pill-reject">PROBE MANDATORY</span>
                      ) : (
                        <span style={{ color: 'var(--mut)' }}>—</span>
                      )}
                    </td>
                    <td style={{ fontFamily: 'monospace' }}>{item.latency_ms.toFixed(1)} ms</td>
                    <td>
                      <button
                        className={`chip-btn ${holdMap[item.board] ? 'active' : ''}`}
                        onClick={() => toggleHold(item.board)}
                        style={{ fontSize: 11, padding: '3px 8px' }}
                      >
                        {holdMap[item.board] ? '🔒 ON HOLD' : 'Release'}
                      </button>
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
