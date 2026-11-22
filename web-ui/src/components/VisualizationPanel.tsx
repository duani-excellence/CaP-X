import { useState, useEffect, useRef } from 'react';
import type { SessionState } from '../types/messages';

/**
 * Default Viser URL — goes through the FastAPI reverse proxy so only one
 * port (the web-UI port) needs to be forwarded.
 *
 * Important: do NOT remount the iframe whenever the proxy flickers or
 * ``viser_ready`` fires — remounting tears down the Viser WebSocket and
 * leaves a blank/white panel even though the scene exists on :8080.
 */
const DEFAULT_VISER_URL = '/viser-proxy/';

interface VisualizationPanelProps {
  trialState?: SessionState;
  /** Bumped when backend reports Viser ready (used only if still disconnected). */
  viserEpoch?: number;
}

export function VisualizationPanel({ trialState, viserEpoch = 0 }: VisualizationPanelProps) {
  const [viserUrl, setViserUrl] = useState(DEFAULT_VISER_URL);
  const [isEditing, setIsEditing] = useState(false);
  const [tempUrl, setTempUrl] = useState(viserUrl);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [iframeKey, setIframeKey] = useState(0);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const wasConnectedRef = useRef(false);

  const remountViser = (reason: string) => {
    console.debug('[viser] remount:', reason);
    setViserUrl(`${DEFAULT_VISER_URL}?t=${Date.now()}`);
    setIframeKey((k) => k + 1);
    setConnectionStatus('connecting');
  };

  const handleSaveUrl = () => {
    setViserUrl(tempUrl);
    setIsEditing(false);
    remountViser('url-save');
  };

  // New trial: show connecting overlay, but keep existing iframe until proxy is up
  // again (avoids about:blank white flash).
  const prevTrialState = useRef(trialState);
  useEffect(() => {
    if (trialState === 'running' && prevTrialState.current !== 'running') {
      wasConnectedRef.current = false;
      setConnectionStatus('connecting');
    }
    prevTrialState.current = trialState;
  }, [trialState]);

  // Only remount on viser_ready if we are NOT already connected (first attach).
  const prevEpoch = useRef(viserEpoch);
  useEffect(() => {
    if (viserEpoch > 0 && viserEpoch !== prevEpoch.current) {
      if (!wasConnectedRef.current) {
        remountViser(`epoch-${viserEpoch}`);
      }
    }
    prevEpoch.current = viserEpoch;
  }, [viserEpoch]);

  // Poll proxy health — drive Connected/Idle; remount only on first recovery.
  useEffect(() => {
    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const ctrl = new AbortController();
          const tid = setTimeout(() => ctrl.abort(), 3000);
          const resp = await fetch(DEFAULT_VISER_URL, { method: 'HEAD', signal: ctrl.signal });
          clearTimeout(tid);

          if (resp.ok) {
            const recovered = !wasConnectedRef.current;
            wasConnectedRef.current = true;
            setConnectionStatus('connected');
            if (recovered) {
              // First time (or after 503): ensure iframe points at live proxy once.
              setViserUrl(`${DEFAULT_VISER_URL}?t=${Date.now()}`);
              setIframeKey((k) => k + 1);
            }
          } else {
            wasConnectedRef.current = false;
            setConnectionStatus((prev) =>
              prev === 'connected' ? 'disconnected' : prev === 'connecting' ? 'connecting' : 'disconnected'
            );
          }
        } catch {
          wasConnectedRef.current = false;
          setConnectionStatus((prev) =>
            prev === 'connected' ? 'disconnected' : prev === 'connecting' ? 'connecting' : 'disconnected'
          );
        }
        await new Promise((r) => setTimeout(r, wasConnectedRef.current ? 4000 : 800));
      }
    }

    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleManualRefresh = () => {
    wasConnectedRef.current = false;
    remountViser('manual');
  };

  const statusDotClass =
    connectionStatus === 'connected'
      ? 'bg-nv-green'
      : connectionStatus === 'disconnected'
        ? 'bg-text-tertiary'
        : 'bg-accent animate-pulse';

  const statusText =
    connectionStatus === 'connected'
      ? 'Connected'
      : connectionStatus === 'disconnected'
        ? 'Idle'
        : 'Connecting...';

  return (
    <div className="flex-1 flex flex-col min-h-0 h-full">
      <div className="flex-shrink-0 px-5 py-3.5 bg-surface-raised border-b border-surface-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-accent/10 border border-accent/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
            </svg>
          </div>
          <span className="text-sm font-bold font-display tracking-wide uppercase text-text-primary">3D Visualization</span>
          <div className="flex items-center gap-1.5 ml-2">
            <div className={`w-2 h-2 rounded-full ${statusDotClass}`} />
            <span className={`text-xs font-display ${connectionStatus === 'connected' ? 'text-nv-green' : 'text-text-tertiary'}`}>
              {statusText}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isEditing ? (
            <>
              <input
                type="text"
                value={tempUrl}
                onChange={(e) => setTempUrl(e.target.value)}
                className="px-3 py-1.5 text-xs bg-surface-sunken text-text-primary border border-surface-border rounded-md w-full sm:w-64 focus:ring-1 focus:ring-accent/40 focus:border-accent/40"
                onKeyDown={(e) => e.key === 'Enter' && handleSaveUrl()}
                autoFocus
              />
              <button
                onClick={handleSaveUrl}
                className="px-3 py-1.5 text-xs font-display bg-accent text-black rounded-md hover:bg-accent-light transition-colors"
              >
                Save
              </button>
              <button
                onClick={() => {
                  setTempUrl(viserUrl);
                  setIsEditing(false);
                }}
                className="px-3 py-1.5 text-xs bg-surface-overlay text-text-secondary rounded-md hover:bg-surface-border transition-colors"
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              <span className="text-xs text-text-tertiary max-w-[200px] truncate" title={viserUrl}>
                {viserUrl}
              </span>
              <button
                onClick={handleManualRefresh}
                className="p-2 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
                title="Refresh connection"
                aria-label="Refresh connection"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
              <button
                onClick={() => setIsEditing(true)}
                className="p-2 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
                title="Edit URL"
                aria-label="Edit URL"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
              <button
                onClick={() => window.open('http://localhost:8080', '_blank')}
                className="p-2 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
                title="Open Viser directly (:8080)"
                aria-label="Open Viser directly"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>
            </>
          )}
        </div>
      </div>

      <div className="flex-1 relative bg-surface-sunken min-h-0">
        {connectionStatus !== 'connected' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-surface-sunken/95">
            <div className="flex items-center gap-3 text-text-secondary mb-2">
              {connectionStatus === 'connecting' ? (
                <div className="w-6 h-6 border-2 border-accent/20 border-t-accent rounded-full animate-spin" />
              ) : (
                <div className={`w-2 h-2 rounded-full ${statusDotClass}`} />
              )}
              <span className="font-display text-sm">
                {connectionStatus === 'connecting'
                  ? 'Waiting for Viser server...'
                  : 'Viser idle — start a trial to open the 3D view'}
              </span>
            </div>
            <span className="text-xs text-text-tertiary">
              If blank while Connected, open http://localhost:8080 in a new tab
            </span>
          </div>
        )}
        {/* Keep iframe mounted whenever we have a URL; overlay covers when down */}
        <iframe
          ref={iframeRef}
          key={`viser-${iframeKey}`}
          src={viserUrl}
          title="Viser 3D Visualization"
          className="absolute inset-0 w-full h-full border-0 bg-black"
          allow="autoplay; fullscreen"
        />
      </div>
    </div>
  );
}
