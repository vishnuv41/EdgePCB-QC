import React from 'react';
import { SystemHealth } from '../services/api';

export type PageId =
  | 'dashboard'
  | 'live'
  | 'workspace'
  | 'review'
  | 'log'
  | 'analytics'
  | 'hardware';

interface SidebarProps {
  currentPage: PageId;
  onSelectPage: (page: PageId) => void;
  health: SystemHealth | null;
}

interface NavItem {
  id: PageId;
  label: string;
  icon: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'dashboard', label: 'AOI Overview Dashboard', icon: '📊' },
  { id: 'live', label: 'Live Camera Inspection', icon: '📹' },
  { id: 'workspace', label: 'Captured PCB Inspection', icon: '🔍' },
  { id: 'review', label: 'Human Review & Verification', icon: '🔬' },
  { id: 'log', label: 'Inspection Log & Audit', icon: '📋' },
  { id: 'analytics', label: 'Pareto & Defect Analytics', icon: '📈' },
  { id: 'hardware', label: 'Hardware Status & Telemetry', icon: '⚙️' },
];

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onSelectPage, health }) => {
  const isOnline = !!health && health.status === 'online';

  return (
    <aside className="sidebar">
      {/* Brand Card */}
      <div className="brand-card">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="brand-logo-icon"
        >
          <polygon points="6 2 18 2 22 6 22 18 18 22 6 22 2 18 2 6 6 2" stroke="#38bdf8" fill="rgba(56,189,248,0.1)" />
          <circle cx="12" cy="12" r="3" fill="#38bdf8" />
          <line x1="12" y1="2" x2="12" y2="22" stroke="#38bdf8" />
          <line x1="2" y1="12" x2="22" y2="12" stroke="#38bdf8" />
        </svg>
        <div>
          <div className="brand-title">EdgePCB-QC</div>
          <div className="brand-subtitle">INDUSTRIAL AOI SYSTEM</div>
        </div>
      </div>

      {/* Navigation List */}
      <div className="nav-section-title">STATION NAVIGATION</div>
      <nav className="nav-list">
        {NAV_ITEMS.map((item) => {
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelectPage(item.id)}
            >
              <span className="nav-radio-bullet">
                {isActive && <span className="nav-radio-bullet-inner" />}
              </span>
              <span className="nav-item-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-spacer" />

      {/* Edge Status Card */}
      <div className="sidebar-edge-card">
        <div className="sidebar-edge-title">
          <span className={`edge-status-dot ${isOnline ? '' : 'offline'}`} />
          <span>{isOnline ? 'Backend online' : 'Backend offline'}</span>
        </div>
        <div className="sidebar-edge-desc">
          {health ? `${health.ai_acceleration} · ${health.mode}` : 'Checking connection to localhost:8000…'}
        </div>
      </div>
    </aside>
  );
};
