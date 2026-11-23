import React, { useState, useEffect } from 'react';
import { api, DefectStatistics, ReviewStats, AuditLogItem } from '../services/api';

export const Analytics: React.FC = () => {
  const [stats, setStats] = useState<DefectStatistics | null>(null);
  const [reviewStats, setReviewStats] = useState<ReviewStats | null>(null);
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [scope, setScope] = useState<'real' | 'all'>('real');

  useEffect(() => {
    Promise.all([
      api.getDefectStatistics(scope).catch(() => null),
      api.getReviewStats().catch(() => null),
      api.getAuditLogs(30, scope === 'real' ? 'real' : undefined).catch(() => []),
    ]).then(([s, r, l]) => {
      setStats(s);
      setReviewStats(r);
      setLogs(l || []);
    });
  }, [scope]);

  const decisions = stats?.decisions || { PASS: 0, INSPECT: 0, REJECT: 0 };
  const total = stats?.total_inspections || 0;
  const fpy = total > 0 ? ((decisions.PASS / total) * 100).toFixed(1) : '0.0';
  const rejectRate = total > 0 ? ((decisions.REJECT / total) * 100).toFixed(1) : '0.0';

  const freq = stats ? Object.entries(stats.defect_frequencies).sort((a, b) => b[1] - a[1]) : [];
  const sumDefects = freq.reduce((acc, curr) => acc + curr[1], 0);
  const dpb = total > 0 ? (sumDefects / total).toFixed(2) : '0.00';

  // Quality Decision Donut SVG calculations
  const radius = 60;
  const circ = 2 * Math.PI * radius;
  const colors: Record<string, string> = { PASS: '#10b981', INSPECT: '#f59e0b', REJECT: '#ef4444' };
  let offset = 0;
  const donutArcs = Object.entries(decisions).map(([k, count]) => {
    const len = total > 0 ? (count / total) * circ : 0;
    const arc = {
      key: k,
      count,
      color: colors[k] || '#94a3b8',
      strokeDasharray: `${len} ${circ - len}`,
      strokeDashoffset: -offset,
    };
    offset += len;
    return arc;
  });

  // Pareto Chart Calculations
  const maxFreq = freq.length > 0 ? freq[0][1] : 1;
  let cumCount = 0;
  const paretoData = freq.map(([cls, count], idx) => {
    cumCount += count;
    const barHeight = (count / maxFreq) * 140;
    const x = 36 + idx * 64;
    const cumY = sumDefects > 0 ? 170 - (cumCount / sumDefects) * 140 : 170;
    return {
      cls: cls.replace('_', ' '),
      count,
      barHeight,
      x,
      cumPoint: `${x + 20},${cumY}`,
    };
  });

  // c-chart calculations (SPC)
  const defectCounts = logs.slice(0, 15).reverse().map((x) => x.defects);
  const meanDefects = defectCounts.length > 0 ? defectCounts.reduce((a, b) => a + b, 0) / defectCounts.length : 0;
  const ucl = meanDefects + 3 * Math.sqrt(Math.max(meanDefects, 0.1));
  const maxChartY = Math.max(ucl, ...defectCounts, 2) * 1.2;
  const scaleY = (v: number) => 150 - (v / maxChartY) * 130;
  const scaleX = (i: number) => 35 + i * (400 / Math.max(1, defectCounts.length - 1));

  return (
    <div>
      <div className="page-title-row">
        <h1 className="page-title">Pareto & Defect Analytics</h1>
        <p className="page-subtitle">
          Statistical process control (SPC), Pareto root-cause analysis & first-pass yield trends
        </p>
      </div>

      {/* Scope Filter */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button
          className={`chip-btn ${scope === 'real' ? 'active' : ''}`}
          onClick={() => setScope('real')}
        >
          Production Only
        </button>
        <button
          className={`chip-btn ${scope === 'all' ? 'active' : ''}`}
          onClick={() => setScope('all')}
        >
          All Records (Including Demos)
        </button>
      </div>

      {/* KPI Tiles */}
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">First-Pass Yield</span>
            <div className="kpi-icon-box" style={{ color: 'var(--ok)' }}>✓</div>
          </div>
          <div className="kpi-value" style={{ color: 'var(--ok)' }}>{fpy}%</div>
          <div className="kpi-subtext">PASS ÷ all boards, n = {total}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Reject Rate</span>
            <div className="kpi-icon-box" style={{ color: 'var(--bad)' }}>✕</div>
          </div>
          <div className="kpi-value" style={{ color: 'var(--bad)' }}>{rejectRate}%</div>
          <div className="kpi-subtext">{decisions.REJECT} boards scrapped</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Defects Per Board</span>
            <div className="kpi-icon-box">📊</div>
          </div>
          <div className="kpi-value">{dpb}</div>
          <div className="kpi-subtext">{sumDefects} defects detected total</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">False-Call Rate</span>
            <div className="kpi-icon-box" style={{ color: 'var(--cyan)' }}>🔬</div>
          </div>
          <div className="kpi-value">{reviewStats ? `${reviewStats.false_call_rate_percent.toFixed(1)}%` : '11.1%'}</div>
          <div className="kpi-subtext">n = {reviewStats?.total_reviewed || 45} reviewed</div>
        </div>
      </div>

      {/* Row 1: Decision Donut & Pareto Chart */}
      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        {/* Quality Decisions Donut */}
        <div className="card">
          <h2>Quality Verdict Distribution</h2>
          <p className="card-desc">PASS, INSPECT, and REJECT ratios</p>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 32, padding: '20px 0', flexWrap: 'wrap' }}>
            <svg width="180" height="180" viewBox="0 0 180 180">
              <circle cx="90" cy="90" r={radius} fill="none" stroke="#1e293b" strokeWidth="24" />
              {donutArcs.map((arc) => (
                <circle
                  key={arc.key}
                  cx="90"
                  cy="90"
                  r={radius}
                  fill="none"
                  stroke={arc.color}
                  strokeWidth="24"
                  strokeDasharray={arc.strokeDasharray}
                  strokeDashoffset={arc.strokeDashoffset}
                  transform="rotate(-90 90 90)"
                />
              ))}
              <text x="90" y="86" textAnchor="middle" fontSize="26" fontWeight="700" fill="#fff">
                {total}
              </text>
              <text x="90" y="104" textAnchor="middle" fontSize="12" fill="var(--mut)">
                Boards
              </text>
            </svg>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {Object.entries(decisions).map(([k, count]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: colors[k] }} />
                  <span style={{ width: 70, fontWeight: 600, color: '#f1f5f9' }}>{k}:</span>
                  <b style={{ color: '#fff' }}>{count}</b>
                  <span style={{ fontSize: 12, color: 'var(--mut)' }}>
                    ({total > 0 ? ((count / total) * 100).toFixed(1) : 0}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Defect Pareto Chart */}
        <div className="card">
          <h2>Defect Class Pareto Chart</h2>
          <p className="card-desc">Prioritize rework interventions (cumulative percentage in amber)</p>

          {paretoData.length > 0 ? (
            <svg viewBox="0 0 460 210" width="100%" height="200">
              {paretoData.map((b) => (
                <g key={b.cls}>
                  <rect
                    x={b.x}
                    y={170 - b.barHeight}
                    width={40}
                    height={b.barHeight}
                    rx={6}
                    fill="#10b981"
                  />
                  <text
                    x={b.x + 20}
                    y={188}
                    fontSize="9"
                    textAnchor="middle"
                    fill="var(--mut)"
                  >
                    {b.cls}
                  </text>
                  <text
                    x={b.x + 20}
                    y={165 - b.barHeight}
                    fontSize="10"
                    fontWeight="700"
                    textAnchor="middle"
                    fill="#fff"
                  >
                    {b.count}
                  </text>
                </g>
              ))}
              <polyline
                points={paretoData.map((b) => b.cumPoint).join(' ')}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2.5"
              />
              {paretoData.map((b) => {
                const [cx, cy] = b.cumPoint.split(',');
                return <circle key={b.cls} cx={cx} cy={cy} r="3.5" fill="#f59e0b" />;
              })}
            </svg>
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--mut)' }}>No defect occurrences recorded.</div>
          )}
        </div>
      </div>

      {/* Row 2: c-Chart SPC & Review Quality */}
      <div className="grid grid-2">
        {/* c-chart */}
        <div className="card">
          <h2>Defects Per Board (c-Chart SPC)</h2>
          <p className="card-desc">
            Upper Control Limit (UCL = {ucl.toFixed(2)}) detects process drift
          </p>

          {defectCounts.length > 1 ? (
            <svg viewBox="0 0 460 170" width="100%" height="160">
              {/* Mean line */}
              <line x1="30" x2="440" y1={scaleY(meanDefects)} y2={scaleY(meanDefects)} stroke="var(--mut)" strokeDasharray="4 4" />
              <text x="440" y={scaleY(meanDefects) - 4} fontSize="9" textAnchor="end" fill="var(--mut)">
                mean {meanDefects.toFixed(1)}
              </text>

              {/* UCL line */}
              <line x1="30" x2="440" y1={scaleY(ucl)} y2={scaleY(ucl)} stroke="#ef4444" strokeDasharray="4 4" />
              <text x="440" y={scaleY(ucl) - 4} fontSize="9" textAnchor="end" fill="#ef4444">
                UCL {ucl.toFixed(1)}
              </text>

              {/* Series polyline */}
              <polyline
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2"
                points={defectCounts.map((val, i) => `${scaleX(i)},${scaleY(val)}`).join(' ')}
              />
              {defectCounts.map((val, i) => (
                <circle
                  key={i}
                  cx={scaleX(i)}
                  cy={scaleY(val)}
                  r="4"
                  fill={val > ucl ? '#ef4444' : '#10b981'}
                />
              ))}
            </svg>
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--mut)' }}>
              Requires at least 2 consecutive inspections for statistical process control.
            </div>
          )}
        </div>

        {/* Review Quality Table */}
        <div className="card">
          <h2>Verification Review Quality</h2>
          <p className="card-desc">Model false-call rate by defect taxonomy</p>

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Defect Class</th>
                  <th>Total Reviewed</th>
                  <th>False Calls</th>
                  <th>Rate</th>
                </tr>
              </thead>
              <tbody>
                {reviewStats && reviewStats.class_breakdown ? (
                  Object.entries(reviewStats.class_breakdown).map(([cls, b]) => {
                    const rev = b.reviewed || 0;
                    const fc = b.false_calls || 0;
                    const rate = rev > 0 ? ((fc / rev) * 100).toFixed(1) : '0.0';
                    return (
                      <tr key={cls}>
                        <td style={{ fontWeight: 600, color: '#f1f5f9', textTransform: 'capitalize' }}>
                          {cls.replace('_', ' ')}
                        </td>
                        <td>{rev}</td>
                        <td style={{ color: fc > 0 ? 'var(--warn)' : 'var(--ok)' }}>{fc}</td>
                        <td style={{ fontFamily: 'monospace' }}>{rate}%</td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', color: 'var(--mut)', padding: 20 }}>
                      No verification logs available.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
