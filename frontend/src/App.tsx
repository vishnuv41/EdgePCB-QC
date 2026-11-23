import React, { useState, useEffect } from 'react';
import { api, SystemHealth, InspectionRecord, SampleBoard } from './services/api';
import { Sidebar, PageId } from './components/Sidebar';
import { Header } from './components/Header';
import { Dashboard } from './pages/Dashboard';
import { LiveStation } from './pages/LiveStation';
import { InspectionWorkspace } from './pages/InspectionWorkspace';
import { HumanReview } from './pages/HumanReview';
import { AuditLog } from './pages/AuditLog';
import { Analytics } from './pages/Analytics';
import { EdgeHardware } from './pages/EdgeHardware';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('live');
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [currentInspection, setCurrentInspection] = useState<InspectionRecord | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [samples, setSamples] = useState<SampleBoard[]>([]);

  // Initial health check and samples load
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const [h, s] = await Promise.all([
          api.getHealth(),
          api.getSamples().catch(() => ({ samples: [] })),
        ]);
        setHealth(h);
        if (s && s.samples) {
          setSamples(s.samples);
        }
      } catch (e) {
        // Backend might still be starting
      }
    };
    fetchInitial();

    const interval = setInterval(async () => {
      try {
        const h = await api.getHealth();
        setHealth(h);
      } catch (e) {}
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleToggleCamera = async (start: boolean, resolution = '1280x720') => {
    if (start) {
      const res = await api.startCamera(resolution);
      setCameraActive(res.is_streaming);
    } else {
      await api.stopCamera();
      setCameraActive(false);
    }
  };

  const handleInspectionDone = (rec: InspectionRecord) => {
    setCurrentInspection(rec);
    setCurrentPage('workspace');
  };

  return (
    <div className="app-container">
      <Sidebar
        currentPage={currentPage}
        onSelectPage={(p) => setCurrentPage(p)}
        health={health}
      />

      <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header health={health} cameraActive={cameraActive} />

        <main className="main-content">
          {currentPage === 'dashboard' && (
            <Dashboard
              latestInspection={currentInspection}
              onNavigate={(p) => setCurrentPage(p)}
            />
          )}

          {currentPage === 'live' && (
            <LiveStation
              health={health}
              cameraActive={cameraActive}
              onToggleCamera={handleToggleCamera}
              onInspectionComplete={handleInspectionDone}
              samples={samples}
            />
          )}

          {currentPage === 'workspace' && (
            <InspectionWorkspace
              currentInspection={currentInspection}
              onSetInspection={(rec) => setCurrentInspection(rec)}
            />
          )}

          {currentPage === 'review' && (
            <HumanReview currentInspection={currentInspection} />
          )}

          {currentPage === 'log' && <AuditLog />}

          {currentPage === 'analytics' && <Analytics />}

          {currentPage === 'hardware' && <EdgeHardware health={health} />}
        </main>
      </div>
    </div>
  );
};
