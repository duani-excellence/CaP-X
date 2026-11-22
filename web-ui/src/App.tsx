import { useState, useEffect, useRef, useCallback } from 'react';
import { useTrialState } from './hooks/useTrialState';
import { ConfigStartControl } from './components/ConfigStartControl';
import { ChatPanel } from './components/ChatPanel';
import { VisualizationPanel } from './components/VisualizationPanel';

const FALLBACK_CONFIG = 'env_configs/cube_stack/franka_robosuite_cube_stack_web.yaml';

function App() {
  const trial = useTrialState();
  const [hasCheckedSession, setHasCheckedSession] = useState(false);
  const [splitPercent, setSplitPercent] = useState(60);
  const draggingRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const active = await trial.checkActiveSession();
        if (cancelled) return;
        if (active?.session_id) {
          trial.reconnectToSession(active.session_id, active.config_path || FALLBACK_CONFIG);
        } else {
          await trial.loadConfig(FALLBACK_CONFIG);
        }
      } finally { if (!cancelled) setHasCheckedSession(true); }
    };
    void init();
    return () => { cancelled = true; };
  }, []);
  // Drag handler for resizable split
  const [isDragging, setIsDragging] = useState(false);
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    setIsDragging(true);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current || !containerRef.current) return;
      e.preventDefault();
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setSplitPercent(Math.min(70, Math.max(30, pct)));
    };

    const handleMouseUp = () => {
      draggingRef.current = false;
      setIsDragging(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, []);


  if (!hasCheckedSession) return <div className="p-6 text-text-secondary">正在加载…</div>;
  return (
    <div className="h-full flex flex-col bg-surface">
      <header className="flex-shrink-0 bg-surface-raised border-b border-surface-border header-glow">
        <div className="flex flex-wrap items-center px-5 py-3 gap-4">
          <div className="flex items-center gap-2">
            <img src="/capx_logo.svg" alt="" className="w-7 h-7" />
            <h1 className="font-bold text-text-primary">CaP-X</h1>
          </div>
          <ConfigStartControl state={trial.state} configPath={trial.configPath} error={trial.error}
            loadConfig={trial.loadConfig} startTrial={trial.startTrial} stopTrial={trial.stopTrial} />
          <span className="text-xs text-text-tertiary">其余参数使用默认配置</span>
        </div>
      </header>
      {/* Main Content — Resizable Split */}
      <main ref={containerRef} className="flex-1 flex overflow-hidden relative">
        {/* Overlay to capture mouse during drag */}
        {isDragging && <div className="absolute inset-0 z-20" />}

        {/* Left Panel - Chat */}
        <div
          className="flex flex-col bg-surface overflow-hidden"
          style={{ width: `${splitPercent}%` }}
        >
          <ChatPanel
            messages={trial.messages}
            state={trial.state}
            onSendMessage={trial.injectPrompt}
            onResume={trial.resumeTrial}
            taskPrompt={trial.taskPrompt}
          />
        </div>

        {/* Draggable Divider */}
        <div
          className={`flex-shrink-0 w-0.5 cursor-col-resize relative group z-30 transition-colors duration-150 ${
            isDragging ? 'bg-accent gold-rule' : 'bg-surface-border hover:bg-accent/50'
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Wider invisible hit target */}
          <div className="absolute inset-y-0 -left-2 -right-2" />
          {/* Visual grip indicator */}
          <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-8 rounded-full flex items-center justify-center transition-all ${
            isDragging ? 'bg-accent/20' : 'bg-transparent group-hover:bg-surface-overlay'
          }`}>
            <div className="flex flex-col gap-0.5">
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
            </div>
          </div>
        </div>

        {/* Right Panel - Visualization */}
        <div
          className="flex flex-col bg-surface-raised overflow-hidden min-h-0 h-full"
          style={{ width: `${100 - splitPercent}%` }}
        >
          <VisualizationPanel trialState={trial.state} viserEpoch={trial.viserEpoch} />
        </div>
      </main>
    </div>
  );
}

export default App;
