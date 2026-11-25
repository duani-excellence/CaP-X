import { useEffect, useMemo, useState, useRef } from 'react';
import './experiments.css';
import { RecoveryTimeline } from './RecoveryTimeline';
import { ForcePanel } from './ForcePanel';

type Json = Record<string, any>;
async function api(path: string, init?: RequestInit) {
  const response = await fetch('/api/experiments' + path, init);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
const query = (path: string, values: Record<string, string | number>) => path + '?' + new URLSearchParams(Object.entries(values).map(([k,v]) => [k,String(v)]));
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
const post = (body?: unknown): RequestInit => ({method:'POST',headers:{'Content-Type':'application/json'},body:body === undefined ? undefined : JSON.stringify(body)});
const color: Record<string,string> = {running:'#22d3ee',returned:'#2dd4bf',done:'#4ade80', invalid:'#fbbf24', failed:'#fb7185', UNKNOWN:'#c4b5fd', pending:'#60a5fa'};

function GraphPanel({path, method}: {path:string, method?:string}) {
  const [data,setData] = useState<Json>({nodes:{},edges:[],events:[],graph_ids:[]});
  const [at,setAt] = useState<number|null>(null);
  const [graph,setGraph] = useState('');
  const [edgeKind,setEdgeKind] = useState('all');
  const [layer,setLayer] = useState('all');
  const [search,setSearch] = useState('');
  const [latest,setLatest] = useState(true);
  const [revision,setRevision] = useState('all');
  const [version,setVersion] = useState('all');
  const [detail,setDetail] = useState<Json|null>(null);
  const [error,setError] = useState('');
  const [playing,setPlaying] = useState(false);
  useEffect(() => {setAt(null);setGraph('');setDetail(null);setPlaying(false);setVersion('all');setRevision('all');}, [path]);
  useEffect(() => {
    const abort = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const params:Record<string,string|number>={path};if(at!==null)params.at=at;if(graph)params.graph_id=graph;
        const value=await api(query('/graph',params),{signal:abort.signal});
        if(!abort.signal.aborted){setData(value);setError('');}
      } catch(e) {if(!abort.signal.aborted)setError(String(e));}
      if(!abort.signal.aborted)timer=setTimeout(load,2500);
    };
    void load();return()=>{abort.abort();clearTimeout(timer);};
  },[path,at,graph]);
  useEffect(()=>{if(!playing)return;const t=setInterval(()=>setAt(v=>{const n=Math.min((v??0)+10,data.total_events??0);if(n>=data.total_events)setPlaying(false);return n;}),500);return()=>clearInterval(t);},[playing,data.total_events]);
  const nodes = useMemo(()=>{
    const all=Object.values(data.nodes) as Json[];const versions=new Map<string,number>();
    all.forEach(n=>{if(n.key)versions.set(n.key,Math.max(versions.get(n.key)??0,n.version??0));});
    const planNodes=new Set([revision,...data.edges.filter((e:Json)=>e.source===revision).map((e:Json)=>e.target)]);
    return all.filter(n=>(revision==='all'||!String(n.kind).startsWith('causal')||planNodes.has(n.id))&&(layer==='all'||n.layer===layer)&&(!latest||!n.key||versions.get(n.key)===n.version)&&(version==='all'||String(n.version)===version)&&(!search||pretty(n).toLowerCase().includes(search.toLowerCase())));
  },[data.nodes,data.edges,layer,latest,search,version,revision]);
  const visible=nodes.slice(-180);
  const positions=new Map<string,{x:number,y:number}>(); const lanes:Record<string,number>={L1:0,L2:1,L3:2};const row=[0,0,0];
  visible.forEach((n,i)=>{const lane=layer==='all'?(lanes[n.layer]??1):i%3;positions.set(n.id,{x:lane*315+20,y:row[lane]++*80+30});});
  const height=Math.max(200,...row.map(x=>x*80+40));
  const shownDetail=detail?.id?(data.nodes[detail.id]??{id:detail.id,note:'该节点在当前时间点尚未出现'}):detail;
  const versions=Array.from(new Set((Object.values(data.nodes) as Json[]).map(n=>n.version).filter(v=>v!=null))).sort((a,b)=>a-b);
  return <section className="exp-card">
    <h2>执行依赖图 {method?.includes('LOGIV')&&<small>共享 Skill 契约记录；官方 LOGIV 调度见方法轨迹</small>}<small>{data.origin==='no_recorded_graph'?'此记录没有图；请查看真实程序和 Trace':'运行时真实节点与依赖'}</small></h2>
    <div className="exp-tools">
      <label>层级<select value={layer} onChange={e=>setLayer(e.target.value)}><option value="all">全部</option><option>L1</option><option>L2</option><option>L3</option></select></label>
      <label>边类型<select value={edgeKind} onChange={e=>setEdgeKind(e.target.value)}>{['all','causal','resource','conflict','precondition','data','attestation'].map(k=><option key={k} value={k}>{k==='all'?'全部边':k}</option>)}</select></label>
      <label>图实例<select value={graph||data.graph_id||''} onChange={e=>setGraph(e.target.value)}>{data.graph_ids.map((x:string)=><option key={x}>{x}</option>)}</select></label>
      <label>因果计划<select value={revision} onChange={e=>setRevision(e.target.value)}><option value="all">全部计划</option>{(Object.values(data.nodes) as Json[]).filter(n=>n.kind==='causal_plan').map(n=><option key={n.id} value={n.id}>{n.id} · {n.data.revision}</option>)}</select></label>
      <label>证据/计划版本<select value={version} onChange={e=>setVersion(e.target.value)}><option value="all">全部版本</option>{versions.map(v=><option key={v}>{v}</option>)}</select></label>
      <label><input type="checkbox" checked={latest} onChange={e=>setLatest(e.target.checked)}/> 每个键仅最新版本</label>
      <input aria-label="搜索图节点" placeholder="搜索条件、对象、代码、失效原因" value={search} onChange={e=>setSearch(e.target.value)}/>
    </div>
    <div className="exp-tools"><button onClick={()=>{setAt(0);setPlaying(true);}}>回放</button><button onClick={()=>setPlaying(false)}>暂停</button><button onClick={()=>{setAt(null);setPlaying(false);}}>跟随实时</button>
      <input aria-label="图事件时间轴" type="range" min={0} max={data.total_events||0} value={at??data.total_events??0} onChange={e=>{setPlaying(false);setAt(Number(e.target.value));}}/>
      <span>事件 {at??data.total_events??0} / {data.total_events??0}</span></div>
    <p className="exp-muted">L1 目标 / Skill / 条件 · L2 程序 / 调用 / 计划 · L3 观测 / 状态 / 证据。状态仅按实际记录展示；done 不等于物理任务成功。</p>
    <div className="exp-legend">{Object.entries(color).map(([name,c])=><span key={name}><i style={{background:c}}/>{name}</span>)}<span>亮边：当前事件影响的节点</span></div>
    {error&&<p role="alert">{error}</p>}
    <div className="exp-graph-layout"><div className="exp-canvas">
      {visible.length===0?<p>没有匹配的真实图节点。</p>:<svg width="965" height={height} aria-label="运行时依赖图">
        <defs><marker id="dependency-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#7185a1"/></marker></defs>{data.edges.filter((e:Json)=>(edgeKind==='all'||e.kind===edgeKind)&&positions.has(e.source)&&positions.has(e.target)).map((e:Json,i:number)=>{
          const a=positions.get(e.source)!,b=positions.get(e.target)!;
          return <path key={i} d={`M ${a.x+135} ${a.y+40} C ${a.x+180} ${a.y+50}, ${b.x+180} ${b.y-20}, ${b.x+135} ${b.y}`} stroke={detail?.source===e.source&&detail?.target===e.target?'#fbbf24':'#44556c'} fill="none" strokeWidth="2" markerEnd="url(#dependency-arrow)" tabIndex={0} role="button" aria-label={`${e.source} → ${e.target} ${e.kind}`} onClick={()=>setDetail(e)} onKeyDown={k=>{if(k.key==='Enter')setDetail(e);}}/>;})}
        {visible.map(n=>{const p=positions.get(n.id)!;const title=n.key||n.data?.id||n.data?.intent||n.kind;return <g key={n.id} role="button" tabIndex={0} aria-label={`${n.id} ${title} ${n.display_state}`} onClick={()=>setDetail(n)} onKeyDown={e=>{if(e.key==='Enter')setDetail(n);}} className="exp-node">
          <rect x={p.x} y={p.y} width={280} height={44} rx={7} fill="#182435" stroke={color[n.display_state]||'#94a3b8'} strokeWidth={(data.changed||[]).includes(n.id)?4:1}/>
          <text x={p.x+10} y={p.y+17} fill="#e2e8f0" fontSize="12">{String(title).slice(0,35)}</text><text x={p.x+10} y={p.y+33} fill={color[n.display_state]||'#94a3b8'} fontSize="10">{n.id} · {n.kind} · v{n.version??'—'} · {n.display_state}</text>
        </g>;})}
      </svg>}
    </div><aside><h3>节点 / 边详情</h3><p className="exp-muted">选择节点查看契约、来源、代码与失效原因；选择连线查看真实边类型。</p><pre>{shownDetail?pretty(shownDetail):'尚未选择'}</pre></aside></div>
    {nodes.length>180&&<p>显示筛选结果的最近 180 / {nodes.length} 个节点，请缩小层级、关键词或版本范围。</p>}
    <details><summary>当前时间点的图变化（原始事件）</summary><pre>{pretty(data.events)}</pre></details>
  </section>;
}

export function ExperimentWorkbench() {
  const [caps,setCaps]=useState<Json|null>(null);const [jobs,setJobs]=useState<Json[]>([]);const [episodes,setEpisodes]=useState<Json[]>([]);
  const [selected,setSelected]=useState('');const [summary,setSummary]=useState<Json|null>(null);const [programs,setPrograms]=useState<Json[]>([]);
  const [trace,setTrace]=useState<Json>({rows:[],total:0});const [offset,setOffset]=useState(0);const [filter,setFilter]=useState('');const [manualPath,setManualPath]=useState('');
  const [compare,setCompare]=useState<string[]>([]);const [comparison,setComparison]=useState<Json[]>([]);const [error,setError]=useState('');const [message,setMessage]=useState('');
  const [task,setTask]=useState('pick_place');const [methods,setMethods]=useState(['G0','G1']);const [seeds,setSeeds]=useState('0');const [skill,setSkill]=useState('auto');const [ablation,setAblation]=useState('none');const [scenario,setScenario]=useState('normal');
  const [forceMode,setForceMode]=useState('off');
  const [harness,setHarness]=useState('legacy');const [recoveryRequests,setRecoveryRequests]=useState(0);
  const [config,setConfig]=useState('');const [ik,setIk]=useState('');const [logiv,setLogiv]=useState('');const [reference,setReference]=useState('');
  const [extra,setExtra]=useState('{"max_requests": 6, "cost_limit_usd": 1.0, "wall_seconds": 1800, "max_tokens": 32768}');
  const defaultsLoaded=useRef(false);
  const [busy,setBusy]=useState(false);const [tick,setTick]=useState(0);const [log,setLog]=useState('');
  useEffect(()=>{const abort=new AbortController();let timer:ReturnType<typeof setTimeout>;
    const load=async()=>{try{const [c,j,e]=await Promise.all([api('/capabilities',{signal:abort.signal}),api('/jobs',{signal:abort.signal}),api('/episodes',{signal:abort.signal})]);if(!abort.signal.aborted){setCaps(c);if(!defaultsLoaded.current){setIk(c.defaults.ik_runtime||'');setLogiv(c.defaults.external_logiv||'');defaultsLoaded.current=true;}setJobs(j);setEpisodes(e);setTick(t=>t+1);}}catch(e){if(!abort.signal.aborted)setError(String(e));}if(!abort.signal.aborted)timer=setTimeout(load,4000);};void load();return()=>{abort.abort();clearTimeout(timer);};},[]);
  useEffect(()=>{setOffset(0);setSummary(null);setPrograms([]);},[selected]);
  useEffect(()=>{if(!selected)return;const abort=new AbortController();
    Promise.all([api(query('/episode',{path:selected}),{signal:abort.signal}),api(query('/programs',{path:selected}),{signal:abort.signal}),api(query('/trace',{path:selected,offset,limit:30}),{signal:abort.signal})]).then(([s,p,t])=>{setSummary(s);setPrograms(p);setTrace(t);}).catch(e=>{if(!abort.signal.aborted)setError(String(e));});return()=>abort.abort();},[selected,offset,tick]);
  useEffect(()=>{const abort=new AbortController();Promise.all(compare.map(path=>api(query('/episode',{path}),{signal:abort.signal}).then(v=>({...v,path})))).then(setComparison).catch(e=>{if(!abort.signal.aborted)setError(String(e));});return()=>abort.abort();},[compare,tick]);
  const makeSpec=()=>({...JSON.parse(extra),task,methods,seeds:seeds.split(/[ ,]+/).filter(Boolean).map(Number),skill,ablation,scenario,force_mode:forceMode,harness,recovery_requests:recoveryRequests,...(config?{config}:{}),...(ik?{ik_runtime:ik}:{}),...(logiv?{external_logiv:logiv}:{}),...(reference?{reference}: {})});
  const submit=async(start:boolean)=>{setBusy(true);setError('');try{const s=makeSpec();const value=await api(start?'/jobs':'/validate',post(s));setMessage(start?`已启动 ${value.id}（将产生 API 费用）`:pretty(value));}catch(e){setError(String(e));}finally{setBusy(false);}};
  const mutate=async(id:string,action:string)=>{try{setMessage(pretty(await api(`/jobs/${id}/${action}`,post())));}catch(e){setError(String(e));}};
  const filtered=episodes.filter(e=>e.path.toLowerCase().includes(filter.toLowerCase()));
  const paired=comparison.length>1&&comparison.every(x=>x.snapshot_hash&&x.snapshot_hash===comparison[0].snapshot_hash&&x.initial_code_hash&&x.initial_code_hash===comparison[0].initial_code_hash);
  return <div className="exp-workbench">
    <header className="exp-title"><div><h1>实验与图 · V33</h1><p>统一入口 / 真实记录 / 配对诊断</p></div><code>scripts/local/infer.sh run --methods G0 G1</code></header>
    {error&&<div role="alert" className="exp-error">{error}<button onClick={()=>setError('')}>关闭</button></div>}
    <section className="exp-card"><h2>推理配置 <small>与 CLI 共用 RunSpec；不支持的组合在启动前拒绝</small></h2>
      <div className="exp-form"><label>任务<select value={task} onChange={e=>{setTask(e.target.value);setMethods(v=>v.filter(m=>m!=='B0'&&m!=='LOGIV'));setScenario('normal');setSkill('auto');setAblation('none');setForceMode('off');setHarness('legacy');setRecoveryRequests(0);}}>{caps?.tasks.map((x:string)=><option key={x}>{x}</option>)}</select></label>
      <label>Seed<input value={seeds} onChange={e=>setSeeds(e.target.value)} placeholder="0, 21, 7"/></label>
      <label>模型<select defaultValue="deepseek"><option value="deepseek">DeepSeek · deepseek-v4-pro</option>{caps?.providers.filter((x:Json)=>!x.available).map((x:Json)=><option key={x.id} disabled>{x.id} — {x.reason}</option>)}</select></label>
      <label>观测<select defaultValue="vision"><option value="vision">Vision · 真实公开感知</option><option disabled>Privileged — 当前图 Runner 未接入</option></select></label>
      <label>力觉<select value={forceMode} disabled={task!=='pick_place'} onChange={e=>{setForceMode(e.target.value);setSkill('v32');setMethods(v=>v.filter(x=>x==='G0'||x==='G1'));}}><option value="off">Vision Only</option><option value="on">Vision + 腕部 F/T</option><option value="compare">四组对比（A/B/C/D）</option></select></label>
      <label>执行框架<select value={harness} disabled={task!=='pick_place'} onChange={e=>{setHarness(e.target.value);setRecoveryRequests(0);}}><option value="legacy">原有 Runner</option><option value="persistent">Persistent Harness</option></select></label>
      <label>新增恢复请求上限<select value={recoveryRequests} disabled={harness!=='persistent'} onChange={e=>setRecoveryRequests(Number(e.target.value))}>{[0,1,2].map(x=><option key={x} value={x}>{x} 次{x===0?'（不新增恢复费用）':''}</option>)}</select></label>
      <label>Skill<select value={skill} onChange={e=>setSkill(e.target.value)}>{(task==='pick_place'?['auto','v32','v30','v29']:['auto','v23']).map(x=><option key={x}>{x}</option>)}</select></label>
      <label>场景<select value={scenario} onChange={e=>setScenario(e.target.value)}>{(task==='pick_place'?['normal','evidence_expiry','effect_displacement']:['normal','support_refresh','missing_baseline','target_shift']).map(x=><option key={x}>{x}</option>)}</select></label>
      <label>G1 消融<select value={ablation} onChange={e=>setAblation(e.target.value)} disabled={task!=='pick_place'}>{caps?.ablations.map((x:string)=><option key={x}>{x}</option>)}</select></label>
      <label>规划后端<select><option>PyRoKi（沿用配置）</option><option disabled>cuRobo — /plan CUDA804 未验收</option></select></label></div>
      <div className="exp-tools">{caps?.methods.map((m:Json)=><label key={m.id} title={m.reason||m.tasks.join(', ')}><input type="checkbox" disabled={!m.available||!m.tasks.includes(task)} checked={methods.includes(m.id)} onChange={e=>setMethods(v=>e.target.checked?[...v,m.id]:v.filter(x=>x!==m.id))}/>{m.id}{!m.available?'（未接入）':''}</label>)}</div>
      <details><summary>服务配置、Snapshot、重复次数与预算</summary><div className="exp-form">
        <label>Runner 配置（留空自动选）<input value={config} onChange={e=>setConfig(e.target.value)} placeholder="env_configs/graph_feedback/…json"/></label>
        <label>现有 IK runtime JSON<input value={ik} onChange={e=>setIk(e.target.value)} placeholder="outputs/…/ik-runtime.json"/></label>
        <label>LOGIV 官方源码目录<input value={logiv} onChange={e=>setLogiv(e.target.value)}/></label>
        <label>历史初始程序目录（默认零新 LLM；可显式开启恢复请求）<input value={reference} onChange={e=>setReference(e.target.value)}/></label></div>
        <label>其余 RunSpec 参数 JSON（repeats、snapshot、各预算、推理参数）<textarea rows={5} value={extra} onChange={e=>setExtra(e.target.value)}/></label>
        <p className="exp-muted">API Key 只从服务端本地配置加载。自然失败在 normal 中保留。公开恢复协议尚未接入在线策略，不能作为可执行选项。</p></details>
      <div className="exp-tools"><button disabled={busy} onClick={()=>void submit(false)}>校验配置（不调用模型）</button><button disabled={busy||jobs.some(j=>j.running)} onClick={()=>void submit(true)}>启动{reference?'历史重放':'真实推理 / 配对'}</button></div>
      {message&&<details open><summary>配置 / 操作结果</summary><pre>{message}</pre></details>}
    </section>
    <section className="exp-card"><h2>后台实验</h2>{jobs.length===0?<p className="exp-muted">暂无统一入口 Web 作业。CLI 与历史产物见下方。</p>:jobs.map(j=><div className="exp-job" key={j.id}><strong>{j.id.slice(0,8)}</strong><span>{j.status.state||'launching'} · {j.status.live_phase||j.status.stage} · {j.status.method}</span><span>请求 {j.status.requests??0} / {j.status.budget?.max_requests??'—'}</span><span>{j.status.reason}</span>
      <button onClick={()=>{if(j.status.current)setSelected(j.path+'/'+j.status.current);}}>查看当前执行</button><button onClick={()=>void api(`/jobs/${j.id}/log`).then(v=>setLog(v.text)).catch(e=>setError(String(e)))}>日志</button>
      <button disabled={!j.running} onClick={()=>void mutate(j.id,'stop')}>完整配对后停止</button><button disabled={!j.running} onClick={()=>void mutate(j.id,'terminate')}>立即终止（不完整）</button></div>)}{log&&<details open><summary>后台日志</summary><pre>{log}</pre></details>}</section>
    <section className="exp-card"><h2>历史与对比</h2><input aria-label="搜索实验" placeholder="搜索 v30、seed、G1、inference-web…" value={filter} onChange={e=>setFilter(e.target.value)}/>
      <div className="exp-tools"><input aria-label="产物相对路径" placeholder="旧产物不在最近列表？输入 outputs 下相对路径" value={manualPath} onChange={e=>setManualPath(e.target.value)}/><button onClick={()=>setSelected(manualPath.replace(/^outputs\//,''))} disabled={!manualPath}>直接加载历史记录</button></div><div className="exp-history">{filtered.slice(0,70).map(e=><div key={e.path}><input aria-label={`对比 ${e.path}`} type="checkbox" checked={compare.includes(e.path)} onChange={v=>setCompare(c=>v.target.checked?[...c,e.path].slice(-6):c.filter(x=>x!==e.path))}/><button className={selected===e.path?'selected':''} onClick={()=>setSelected(e.path)}>{e.path}</button><span>{e.reward==null?'未完成 / 无评分':`reward ${e.reward}`}</span></div>)}</div>
      {comparison.length>0&&<><p>{paired?'已核对：相同 Snapshot 与初始程序 hash':'未确认共同 Snapshot/初始程序；不能解释为严格配对效果。B0 接口差异需单独记录。'}</p><div className="exp-table"><table><thead><tr>{['方法','物理成功 / reward','子目标','API / capture','SAM3 / GraspNet / IK','恢复成功/尝试','图失效 / patch','语义 / 几何 / 计划复用','HTTP累计秒','LLM / Token','USD 估算','端到端秒','终止原因'].map(x=><th key={x}>{x}</th>)}</tr></thead><tbody>{comparison.map(r=><tr key={r.path}><td title={r.path}>{r.variant}</td><td>{String(r.raw_success??'UNKNOWN')} / {r.reward??'—'}</td><td>{r.native_goals??'—'} / {r.goal_total??'—'}</td><td>{r.api_calls??r.actual_api_calls??'—'} / {r.captures??'—'}</td><td>{r.http_counts.sam3??'—'} / {r.http_counts.graspnet??'—'} / {r.http_counts.pyroki??'—'}</td><td>{r.recovery_successes} / {r.recovery_attempts}</td><td>{r.graph_invalidations} / {r.graph_patches}</td><td>{r.semantic_reuses??'—'} / {r.geometry_reuses??'—'} / {r.plan_reuses??'—'}</td><td>{(Object.values(r.http_seconds) as number[]).reduce((a,b)=>a+b,0).toFixed(1)}</td><td>{r.llm_requests??'—'} / {(r.input_tokens??0)+(r.output_tokens??0)}</td><td>{r.cost_usd?.toFixed(5)??'—'}</td><td>{(r.end_to_end_seconds??r.seconds)?.toFixed(1)??'—'}</td><td>{r.stop_reason||r.error||'—'}</td></tr>)}</tbody></table></div><p className="exp-muted">每个方法的 Token/费用包含其共享初始请求归因，不能直接相加当成唯一付费总额。缺失的历史指标显示 —，不会补造 0。</p></>}
    </section>
    {selected&&<><section className="exp-card"><h2>{selected}</h2><div className="exp-media"><div>{summary?.video?.length>0?<video key={selected} controls preload="metadata" src={'/api/experiments'+query('/video',{path:selected,name:summary?.video[0]})}/>:<img alt="最新归档公开相机画面（按 capture 更新）" src={'/api/experiments'+query('/frame',{path:selected,t:tick})}/>}</div><pre>{pretty(summary && Object.fromEntries(Object.entries(summary).filter(([k])=>!['latest_event','external'].includes(k))))}</pre></div><details><summary>最后事件 / 外部方法真实轨迹</summary><pre>{pretty({latest:summary?.latest_event,external:summary?.external})}</pre></details></section>
      <RecoveryTimeline path={selected} tick={tick}/>
      <ForcePanel path={selected} tick={tick}/>
      <GraphPanel path={selected} method={summary?.method}/>
      <section className="exp-card"><h2>模型程序与恢复代码</h2>{programs.length?programs.map((p,i)=><details key={i} open={i===0}><summary>{p.file} · {p.data?.origin||'归档来源见详情'}</summary><pre>{pretty(p.data)}</pre></details>):<p>没有归档程序。</p>}</section>
      <section className="exp-card"><h2>Trace · 离线诊断</h2><p className="exp-muted">包含独立评估物理记录，仅供用户诊断，不进入在线 Agent。图策略、补证候选与实际派发保留原始字段。</p><div className="exp-tools"><button disabled={offset===0} onClick={()=>setOffset(Math.max(0,offset-30))}>上一页</button><button disabled={offset+30>=trace.total} onClick={()=>setOffset(offset+30)}>下一页</button><button onClick={()=>setOffset(Math.max(0,trace.total-30))}>最新记录</button><span>{offset}–{Math.min(offset+30,trace.total)} / {trace.total}</span></div>{trace.rows.map((r:Json,i:number)=><details key={offset+i}><summary>#{offset+i} · {r.kind} · {r.event||r.name||r.stage||''} {r.reason||r.error||''}</summary><pre>{pretty(r)}</pre></details>)}</section></>}
  </div>;
}
