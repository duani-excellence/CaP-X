"""Read-only, bounded views of real experiment artifacts. Never an actor API."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
from collections import Counter
from functools import lru_cache

SECRET_KEYS={'api_key','authorization','api-key','token','access_token','password','headers','environment','env'}

def redact(value):
    secrets=[v for k,v in os.environ.items() if (k.endswith('_API_KEY') or k.endswith('_TOKEN')) and len(v)>12]
    def visit(v):
        if isinstance(v,dict):return {k:'[REDACTED]' if k.lower() in SECRET_KEYS or k.lower().endswith('_api_key') else visit(x) for k,x in v.items()}
        if isinstance(v,list):return [visit(x) for x in v]
        if isinstance(v,str):
            for secret in secrets:v=v.replace(secret,'[REDACTED]')
        return v
    return visit(value)

def safe_path(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Path must stay inside registered artifact root')
    return path

def read_json(path,default=None):
    if Path(path).is_symlink():return default
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def lines(path):
    if Path(path).is_symlink():return
    try:
        with Path(path).open() as f:
            for line in f:
                try:yield json.loads(line)
                except ValueError:continue  # a live final partial line is not a completed event
    except FileNotFoundError:return

def replay_graph(rows,at=None,graph_id=None):
    selected=rows if at is None else rows[:at];nodes={};edges=[];changed=[]
    if graph_id is None:graph_id=next((r.get('graph_id','g0') for r in selected),'g0')
    for row in selected:
        if row.get('graph_id','g0')!=graph_id:continue
        e=row.get('event',row);kind=e.get('action');changed=[]
        if kind=='add':
            n=copy.deepcopy(e['node']);nodes[n['id']]=n
            edges.extend(dict(source=p['source'],target=n['id'],kind=p['kind']) for p in e.get('parents',[]))
            changed=[n['id']]
            if n.get('kind')=='call' and isinstance(n.get('data'),dict):
                for other in nodes.values():
                    if other.get('kind')=='skill_intent' and (other.get('data') or {}).get('id')==n['data'].get('intent'):
                        other['observed_execution']='returned' if n['data'].get('returned') else 'failed'
                        other['execution_record']=n['id']
        elif kind=='dispatch' and e['node_id'] in nodes:
            nodes[e['node_id']]['observed_execution']='running'
            nodes[e['node_id']]['execution_record']=copy.deepcopy(e)
        elif kind=='invalidate':
            changed=e.get('invalidated',[])
            for key in changed:
                if key in nodes:
                    nodes[key].update(valid=False,state='invalid');nodes[key].setdefault('invalidated_by',[]).append(e.get('reason'))
        elif kind in ('done','returned_call') and e['node_id'] in nodes:
            nodes[e['node_id']].update(state='done',valid=True);changed=[e['node_id']]
        elif kind=='consume_plan' and e['node_id'] in nodes:
            nodes[e['node_id']].setdefault('data',{})['execution']='consumed';changed=[e['node_id']]
    return dict(nodes=nodes,edges=edges,changed=changed,graph_id=graph_id)

def graph_rows(path):
    path=Path(path)
    journal=path/'graph-events.jsonl'
    if journal.exists():return [r if 'event' in r else dict(index=i+1,graph_id='g0',event=r) for i,r in enumerate(lines(journal))]
    saved=read_json(path/'contract-graph.json',{})
    if not saved:
        candidates=list(path.glob('graph-final.json'))+list(path.glob('**/dependency-graph.json'))+list(path.glob('**/execution-graph.json'))
        if candidates:saved=read_json(candidates[0],{})
    return [dict(index=i+1,graph_id='g0',event=e) for i,e in enumerate(saved.get('events',[]))]


def graph_view(path,at=None,graph_id=None):
    rows=graph_rows(path);view=replay_graph(rows,at,graph_id)
    view.update(total_events=len(rows),at=len(rows) if at is None else min(at,len(rows)),graph_ids=list(dict.fromkeys(r.get('graph_id','g0') for r in rows)),events=rows[max(0,(len(rows) if at is None else at)-25):(None if at is None else at)],
        origin='runtime_dependency_graph_events' if rows else 'no_recorded_graph',
        note='Recorded states only. Pending is not proof of running. Failed calls are historical facts, not task success.')
    for n in view['nodes'].values():
        kind=n.get('kind');data=n.get('data') or {}
        n['layer']='L1' if kind in ('task_goal','task_subgoal','repair_goal','expected_effect','evidence_plan','skill_intent','future_plan','causal_plan','capability','skill_contract') or str(kind).startswith('causal') else ('L3' if kind in ('observation','estimate','evidence','state_memory','object_state','geometry','condition','live_causal_support','verification_fact') else 'L2')
        n['display_state']='invalid' if not n.get('valid') else ('UNKNOWN' if data.get('truth')=='UNKNOWN' or data.get('status')=='UNCERTAIN' else ('failed' if data.get('truth')=='FALSE' or data.get('status')=='FAIL' or data.get('returned') is False else n.get('state','pending')))
        if n.get('observed_execution') and n.get('valid'):n['display_state']=n['observed_execution']
    # Expand only explicitly recorded plan edges, mapping to actual action nodes
    # under that same immutable plan. These are not inferred display relations.
    for plan in list(view['nodes'].values()):
        if plan.get('kind') != 'causal_plan':continue
        children={e['target'] for e in view['edges'] if e['source']==plan['id']}
        by_intent={n.get('data',{}).get('intent'):n['id'] for n in view['nodes'].values() if n['id'] in children and n.get('kind')=='causal_action'}
        goal=next((e['source'] for e in view['edges'] if e['target']==plan['id'] and view['nodes'].get(e['source'],{}).get('kind')=='task_goal'),None)
        for edge in (plan.get('data') or {}).get('edges',[]):
            source=by_intent.get(edge['source']);target=goal if edge['target']=='GOAL' else by_intent.get(edge['target'])
            if edge['source']=='START' and target:
                source=next((e['source'] for e in view['edges'] if e['target']==target and view['nodes'].get(e['source'],{}).get('key')==edge.get('condition')),None)
            if source and target:
                view['edges'].append(dict(edge,source=source,target=target,source_record=plan['id'],plan_revision=(plan.get('data') or {}).get('revision'),predicted_only=True))
    return view


def trace_path(path):
    path=Path(path)
    if (path/'trace.jsonl').exists():return path/'trace.jsonl'
    return next(iter(path.glob('trace/*/trace.jsonl')),path/'trace.jsonl')


def episode_summary(path):
    path=Path(path);r=read_json(path/'result.json',{})
    counts=Counter();times=Counter();cfg={}
    for parent in [path,path.parent,path.parent.parent]:
        if (parent/'config.json').exists():cfg=read_json(parent/'config.json',{});break
    http=path.parent/(path.name+'-http.jsonl')
    if not http.exists():http=trace_path(path).parent/'requests.jsonl'
    for e in lines(http):
        role='other'
        for name,url in cfg.get('services',{}).items():
            if str(e.get('url','')).startswith(url.rstrip('/')+'/'):role=name
        if isinstance(e.get('payload'),dict) and 'model' in e['payload']:role='llm'
        counts[role]+=1;times[role]+=e.get('elapsed_seconds',0)
    events=list(lines(trace_path(path)));kinds=Counter(e.get('kind') for e in events)
    policy=[e for e in events if e.get('kind')=='model_contract']
    goals=r.get('native_progress',{}).get('objects_in_bins',[])
    out={k:r.get(k) for k in ('schema','group','method','seed','study','mode','reward','raw_success','status','stop_reason','last_error','error','snapshot_hash','initial_code_hash','api_calls','actual_api_calls','captures','skill_calls','repairs','llm_requests','new_llm_requests','input_tokens','output_tokens','reasoning_tokens','cost_usd','end_to_end_seconds','seconds','request_paths')}
    out.update(variant=path.name,complete=(path/'result.json').exists(),native_progress=r.get('native_progress'),native_goals=sum(float(x)>0 for x in goals) if goals else None,goal_total=len(goals) or None,http_counts=dict(counts),http_seconds=dict(times),trace_events=len(events),
        recovery_attempts=kinds['skill_recovery_stage'],recovery_successes=kinds['skill_recovery_success'],
        graph_invalidations=sum(r.get('event',r).get('action')=='invalidate' for r in graph_rows(path)),
        graph_patches=sum(e.get('event')=='causal_repair_decision' and e.get('decision',{}).get('status')=='ready' for e in policy),
        latest_event=events[-1] if events else None,
        video=[p.name for p in path.glob('*.mp4')],programs=[p.name for p in path.glob('*program.json')],
        external=read_json(path/'external-method.json'),config_budget=cfg.get('budgets',{}))
    if out['raw_success'] is None:out['raw_success']=r.get('final_physics',{}).get('raw_success',r.get('info',{}).get('raw_success'))
    out['force_feedback']=r.get('force_feedback');out['force_decisions']=r.get('force_decisions')
    if out['captures'] is None and (path/'observations').exists():out['captures']=len(list((path/'observations').glob('*.npz')))
    if out['initial_code_hash'] is None:out['initial_code_hash']=read_json(path/'block-0.json',{}).get('code_hash')
    out['programs']+=[p.name for p in path.glob('block-*.json') if '-evaluation' not in p.name]
    out['programs']+= [str(p.relative_to(path)) for p in path.glob('programs/*.json')]
    out['semantic_reuses']=sum(e.get('kind')=='perception_request' and e.get('reused') is True for e in events) if any(e.get('kind')=='perception_request' for e in events) else None
    out['geometry_reuses']=kinds['perception_geometry_reuse'] if out['semantic_reuses'] is not None else None
    out['plan_reuses']=sum(e.get('kind')=='perception_plan_reuse' and e.get('reused') is True for e in events) if out['semantic_reuses'] is not None else None
    out['capture_seconds']=sum(e.get('capture_and_archive_seconds',0.) for e in events if e.get('kind')=='perception_frame') if any('capture_and_archive_seconds' in e for e in events) else None
    return out


def list_episodes(root,limit=300):
    root=Path(root);found=[]
    # Outputs only, prune bulky evidence/source/model trees. Do not scan project files.
    for directory,dirs,files in os.walk(root,followlinks=False):
        rel=Path(directory).relative_to(root)
        dirs[:]=[x for x in dirs if x not in ('source-bytes','upstream','observations','tools','requests','numba-cache','calibration','frames','videos','.git') and not (Path(directory)/x).is_symlink()]
        if len(rel.parts)>6:dirs[:]=[];continue
        if 'result.json' in files or 'contract-graph.json' in files or ('graph-events.jsonl' in files and 'episodes' not in dirs):
            r=read_json(Path(directory)/'result.json',{})
            if not r and 'graph-events.jsonl' not in files:continue
            found.append(dict(path=str(rel),variant=Path(directory).name,seed=r.get('seed'),reward=r.get('reward'),method=r.get('method',r.get('group')),snapshot_hash=r.get('snapshot_hash'),schema=r.get('schema'),mtime=Path(directory).stat().st_mtime))
    return sorted(found,key=lambda x:x['mtime'],reverse=True)[:limit]
