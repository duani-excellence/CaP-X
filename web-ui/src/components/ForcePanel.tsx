import { useEffect, useMemo, useState } from 'react';

type Sample = { timestamp:number; sequence:number; external_base:number[]; filtered_base:number[]; gripper_width_m:number; tcp:number[]; valid:boolean; quasistatic:boolean };
type ForceArchive = { rows:Sample[]; total:number; offset:number; online_enabled:boolean|null; events:Record<string,any>[]; calibration:Record<string,unknown>|null };
const COLORS = ['#38bdf8','#fbbf24','#f472b6'];

function Plot({rows,fields,label,cursor}: {rows:Sample[];fields:{name:string;value:(r:Sample)=>number}[];label:string;cursor:number}) {
  const geometry=useMemo(()=>{
    const start=rows[0].timestamp,span=Math.max(.001,rows[rows.length-1].timestamp-start);
    let lo=Infinity,hi=-Infinity;
    rows.forEach(r=>fields.forEach(f=>{const v=f.value(r);if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}));
    if(!Number.isFinite(lo)){lo=0;hi=1;}
    const range=Math.max(.01,hi-lo);lo-=range*.08;hi+=range*.08;
    const x=(t:number)=>52+816*(t-start)/span,y=(v:number)=>150-124*(v-lo)/(hi-lo);
    return {lo,hi,x,y,lines:fields.map(f=>rows.filter(r=>Number.isFinite(f.value(r))).map(r=>`${x(r.timestamp).toFixed(2)},${y(f.value(r)).toFixed(2)}`).join(' '))};
  },[rows,fields]);
  return <div className="force-plot"><strong>{label}</strong><span>{fields.map((f,i)=><span key={f.name} style={{color:COLORS[i%3],marginLeft:16}}>{f.name}</span>)}</span>
    <svg viewBox="0 0 900 180" role="img" aria-label={label}>
      <path d="M52 20V150H868" fill="none" stroke="#64748b"/>
      <text x="2" y="28" fill="#a8b5ca" fontSize="12">{geometry.hi.toFixed(2)}</text><text x="2" y="150" fill="#a8b5ca" fontSize="12">{geometry.lo.toFixed(2)}</text>
      {geometry.lines.map((points,i)=><polyline key={fields[i].name} points={points} fill="none" stroke={COLORS[i%3]} strokeWidth="1.2"/>)}
      <line x1={geometry.x(rows[cursor].timestamp)} x2={geometry.x(rows[cursor].timestamp)} y1="20" y2="150" stroke="#e2e8f0" strokeDasharray="4 3"/>
      <text x="52" y="172" fill="#a8b5ca" fontSize="12">{rows[0].timestamp.toFixed(2)} s</text><text x="800" y="172" fill="#a8b5ca" fontSize="12">{rows[rows.length-1].timestamp.toFixed(2)} s</text>
    </svg></div>;
}
const FORCE=['Fx','Fy','Fz'].map((name,i)=>({name,value:(r:Sample)=>r.external_base[i]}));
const TORQUE=['Tx','Ty','Tz'].map((name,i)=>({name,value:(r:Sample)=>r.external_base[i+3]}));
const JAW=[{name:'夹爪开度',value:(r:Sample)=>r.gripper_width_m*1000}];
const TCP=['X','Y','Z'].map((name,i)=>({name,value:(r:Sample)=>r.tcp[i]}));

export function ForcePanel({path,tick}:{path:string;tick:number}) {
  const [data,setData]=useState<ForceArchive|null>(null),[error,setError]=useState('');
  const [index,setIndex]=useState<number|null>(null),[playing,setPlaying]=useState(false);
  const [windowSeconds,setWindowSeconds]=useState(0);
  useEffect(()=>{setData(null);setIndex(null);setPlaying(false);},[path]);
  useEffect(()=>{const abort=new AbortController();
    fetch('/api/experiments/force?'+new URLSearchParams({path}),{signal:abort.signal}).then(async r=>{if(!r.ok)throw new Error(`F/T HTTP ${r.status}`);return r.json();}).then(v=>{if(!abort.signal.aborted){setData(v);setError('');}}).catch(e=>{if(!abort.signal.aborted)setError(String(e));});
    return()=>abort.abort();
  },[path,tick]);
  const rows=data?.rows??[],cursor=Math.min(index??Math.max(0,rows.length-1),Math.max(0,rows.length-1));
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setIndex(v=>{const n=Math.min((v??0)+4,rows.length-1);if(n>=rows.length-1)setPlaying(false);return n;}),200);return()=>clearInterval(timer);},[playing,rows.length]);
  if(!rows.length)return <section className="exp-card"><h2>六维力 / 力矩</h2><p>{error||'此记录暂无原生腕部传感器数据。'}</p></section>;
  const current=rows[cursor];
  const plotRows=windowSeconds?rows.filter(r=>Math.abs(r.timestamp-current.timestamp)<=windowSeconds/2):rows;
  const plotCursor=windowSeconds?plotRows.findIndex(r=>r.sequence===current.sequence):cursor;
  const events=(data?.events??[]).filter(e=>typeof e.timestamp==='number'&&e.timestamp<=current.timestamp).slice(-5);
  return <section className="exp-card"><h2>腕部 F/T · 仿真时间回放</h2>
    <p>{data?.online_enabled===true?'Vision + F/T：已进入在线执行反馈':data?.online_enabled===false?'Vision Only：以下传感器记录仅供离线诊断，未提供给执行器':'在线启用状态未记录'}。机器人基座坐标系、TCP 原点；N / N·m。工具重力已补偿，负载重力保留。</p>
    <p className="exp-muted">{data?.total} 个实际采样；显示索引 {data?.offset} 起。曲线使用未滤波补偿值，不删去峰值。抓持稳定与释放许可还需视觉和运动学验证。</p>
    <div className="exp-tools"><button onClick={()=>{setIndex(0);setPlaying(true);}}>回放力觉</button><button onClick={()=>setPlaying(false)}>暂停力觉</button><button onClick={()=>{setIndex(null);setPlaying(false);}}>跟随最新力觉</button>
      <input aria-label="力觉采样时间轴" type="range" min="0" max={rows.length-1} value={cursor} onChange={e=>{setIndex(Number(e.target.value));setPlaying(false);}}/>
      <span>{current.timestamp.toFixed(3)} s · #{current.sequence} · {current.valid?'有效':'无效'} · {current.quasistatic?'准静态窗口候选':'动态'}</span></div>
    <label>曲线窗口<select value={windowSeconds} onChange={e=>setWindowSeconds(Number(e.target.value))}><option value={0}>完整 episode</option><option value={4}>当前时间前后 2 秒</option><option value={10}>当前时间前后 5 秒</option></select></label>
    {error&&<p role="alert">{error}</p>}
    <div className="force-charts"><Plot rows={plotRows} fields={FORCE} label="外力 / N" cursor={plotCursor}/><Plot rows={plotRows} fields={TORQUE} label="外力矩 / N·m" cursor={plotCursor}/><Plot rows={plotRows} fields={JAW} label="夹爪开度 / mm" cursor={plotCursor}/><Plot rows={plotRows} fields={TCP} label="TCP 位置 / m" cursor={plotCursor}/></div>
    <details open><summary>当前时间前的力觉与恢复事件（来源节点见执行图）</summary><pre>{JSON.stringify(events,null,2)}</pre></details>
    <details><summary>当前样本与标定来源</summary><pre>{JSON.stringify({sample:current,calibration:data?.calibration},null,2)}</pre></details>
  </section>;
}
