import { MultitaskControl } from './MultitaskControl';
import { useState, useEffect } from 'react';
import type { SessionState, StartTrialRequest } from '../types/messages';

type Provider = 'deepseek' | 'openai';
type Mode = 'vision' | 'privileged';
interface Options {
  models: { provider: Provider; model: string; label: string }[];
  observations: { mode: Mode; label: string; config_path: string }[];
}
interface Props {
  state: SessionState;
  configPath: string | null;
  error: string | null;
  loadConfig: (path: string) => Promise<unknown>;
  startTrial: (request: StartTrialRequest) => Promise<boolean>;
  stopTrial: () => void;
}

export function ConfigStartControl({ state, configPath, error, loadConfig, startTrial, stopTrial }: Props) {
  const [options, setOptions] = useState<Options | null>(null);
  const [provider, setProvider] = useState<Provider>('deepseek');
  const [mode, setMode] = useState<Mode>('vision');
  const [busy, setBusy] = useState(false);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const running = ['running', 'awaiting_user_input', 'loading_config'].includes(state);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/trial-options', { signal: controller.signal })
      .then(async r => { if (!r.ok) throw new Error('无法加载选项'); return r.json(); })
      .then(setOptions)
      .catch(e => { if (e.name !== 'AbortError') setSelectionError('无法加载选项，请刷新页面'); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const match = options?.observations.find(x => x.config_path === configPath);
    if (match) setMode(match.mode);
  }, [configPath, options]);

  const changeMode = async (next: Mode) => {
    const choice = options?.observations.find(x => x.mode === next);
    if (!choice) return;
    setBusy(true);
    setSelectionError(null);
    try {
      const loaded = await loadConfig(choice.config_path);
      if (loaded) setMode(next);
      else setSelectionError('观测模式加载失败，请重试');
    } finally { setBusy(false); }
  };
  const start = async () => {
    const choice = options?.observations.find(x => x.mode === mode);
    if (!choice) return;
    setBusy(true);
    try {
      await startTrial({ provider, observation_mode: mode, config_path: choice.config_path });
    } finally { setBusy(false); }
  };
  const disabled = running || busy || !options;
  const selectClass = 'px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:opacity-50';
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-sm text-text-secondary">
        模型
        <select aria-label="模型" value={provider} onChange={e => setProvider(e.target.value as Provider)} disabled={disabled} className={selectClass}>
          {options?.models.map(x => <option key={x.provider} value={x.provider}>{x.label}</option>)}
        </select>
      </label>
      <label className="flex items-center gap-2 text-sm text-text-secondary">
        观测模式
        <select aria-label="观测模式" value={mode} onChange={e => void changeMode(e.target.value as Mode)} disabled={disabled} className={selectClass}>
          {options?.observations.map(x => <option key={x.mode} value={x.mode}>{x.label}</option>)}
        </select>
      </label>
      {running ? (
        <button onClick={stopTrial} className="px-4 py-2 rounded-md bg-red-600 text-white text-sm">停止</button>
      ) : (
        <button onClick={() => void start()} disabled={disabled} className="px-4 py-2 rounded-md bg-accent text-black text-sm font-bold disabled:opacity-40">
          {busy ? '准备中…' : '开始推理'}
        </button>
      )}
      <MultitaskControl />
      {(selectionError || error) && <span role="alert" className="text-sm text-red-400 max-w-sm">{selectionError || error}</span>}
    </div>
  );
}
