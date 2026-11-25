import {useEffect,useState} from 'react';

type EventRow={sequence:number;type:string;unix:number;data:Record<string,unknown>};
type History={available:boolean;total:number;rows:EventRow[];checkpoint?:{checkpoint?:string}};
export function RecoveryTimeline({path,tick}:{path:string;tick:number}) {
  const [history,setHistory]=useState<History>({available:false,total:0,rows:[]});
  const [offset,setOffset]=useState(0);const [error,setError]=useState('');
  useEffect(()=>{setOffset(0);setHistory({available:false,total:0,rows:[]});},[path]);
  useEffect(()=>{const abort=new AbortController();
    const load=async()=>{try{const r=await fetch('/api/experiments/events?'+new URLSearchParams({path,offset:String(offset),limit:'60'}),{signal:abort.signal});
      if(!r.ok)throw new Error(`事件读取失败 ${r.status}`);
      const data=await r.json() as History;if(!abort.signal.aborted){setHistory(data);setError('');}
    }catch(e){if(!abort.signal.aborted)setError(String(e));}};
    void load();return()=>abort.abort();},[path,tick,offset]);
  return <section className="exp-card"><h2>Harness · 执行与恢复时间线</h2>
    {error&&<p role="alert">{error}</p>}
    {!history.available?<p>该实验没有持久化 Harness 记录；不补造恢复事件。</p>:<>
      <p>事件 {history.total} · Checkpoint {history.checkpoint?.checkpoint||'尚未生成'}。Checkpoint 保存执行现场，不回滚物理世界。</p>
      <div className="exp-tools"><button disabled={offset===0} onClick={()=>setOffset(Math.max(0,offset-60))}>前 60 条</button><button disabled={offset+60>=history.total} onClick={()=>setOffset(offset+60)}>后 60 条</button><button onClick={()=>setOffset(Math.max(0,history.total-60))}>最新事件</button><span>{offset}–{Math.min(offset+60,history.total)}</span></div>
      {history.rows.map(row=><details key={row.sequence}><summary>#{row.sequence} · {row.type} · {String(row.data.level||row.data.state||row.data.name||row.data.reason||'')}</summary><pre>{JSON.stringify(row.data,null,2)}</pre></details>)}
    </>}
  </section>;
}
