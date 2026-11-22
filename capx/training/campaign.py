"""Resumable, bounded orchestration over the existing single-episode collector."""
import argparse
import ast
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

from .teacher import (collect, prepare, read, dump, digest, training_example,
                      export_data)


def atomic(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def template_hash(code):
    """Heuristic: ignore constants, comments, docstrings and local variable names."""
    tree=ast.parse(code)
    names={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
            names.setdefault(node.id,f'local_{len(names)}')
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):
            node.id=names.get(node.id,node.id);return node
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant(value=type(node.value).__name__),node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
            return self.generic_visit(node)
    return digest(ast.dump(Normalize().visit(tree),include_attributes=False))


def inventory(attempts, max_per_template):
    stats={k:{m:0 for m in ('privileged','vision')} for k in
           ('physical_success','sft_eligible','accepted','native_accepted','transfer_accepted')}
    physical_sources=Counter()
    reasons=Counter();templates=Counter();seen=set();scenes=set();selected=[]
    for attempt in attempts:
        if attempt['state']!='done':continue
        path=Path(attempt['raw'])/'records.jsonl'
        if not path.exists():continue
        for line in path.read_text().splitlines():
            row=json.loads(line);mode=row['mode']
            if row.get('status')=='success':
                stats['physical_success'][mode]+=1
                physical_sources[mode+':'+row['source']]+=1
            sample,reason=training_example(row,mode)
            if reason:reasons[reason]+=1;continue
            stats['sft_eligible'][mode]+=1
            exact=(mode,digest(sample));scene=(mode,row['snapshot_hash'])
            template=(mode,template_hash(sample['messages'][-1]['content']))
            if exact in seen:reasons['duplicate_sample']+=1;continue
            if scene in scenes:reasons['duplicate_scene_mode']+=1;continue
            if templates[template]>=max_per_template:reasons['template_quota']+=1;continue
            seen.add(exact);scenes.add(scene);templates[template]+=1
            selected.append({**row,'id':f"attempt-{attempt['index']}-"+row['id']})
            stats['accepted'][mode]+=1
            stats['transfer_accepted' if row['source']=='transfer' else 'native_accepted'][mode]+=1
    stats['physical_success_by_source']=dict(physical_sources)
    stats['rejections']=dict(reasons)
    stats['templates']={m:sum(1 for key in templates if key[0]==m) for m in ('privileged','vision')}
    stats['task_families']=1
    stats['diversity_note']='Only current cube task; AST quotas are a heuristic, not proof of behavioral diversity.'
    return stats,selected


def add_arguments(parser):
    parser.add_argument('--target-success',type=int,help='Campaign target: unique eligible examples PER target mode')
    parser.add_argument('--target-modes',nargs='+',choices=['privileged','vision'],default=['vision'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start-seed',type=int,default=0)
    parser.add_argument('--max-scenes',type=int,default=10000)
    parser.add_argument('--retries',type=int,default=1,help='Extra attempts only for known task/invalid-program failures')
    parser.add_argument('--request-cost-cap',type=float,default=1.,help='Conservative per-episode reservation in USD')
    parser.add_argument('--max-hours',type=float,default=4.,help='Stop scheduling after this invocation duration')
    parser.add_argument('--min-free-gb',type=float,default=5.)
    parser.add_argument('--max-per-template',type=int,default=20)


# Audited predecessor: only scheduling semantics change; data/execution contracts remain identical.
LEGACY_BATCH_DIGEST = '3e7f0450409e878d79c7a50e2b6b47da869388db497101883476bd82818b417a'


def compatible_config(previous, current):
    if previous == current:return True
    return (previous.get('implementation_digest') == LEGACY_BATCH_DIGEST
            and {k:v for k,v in previous.items() if k!='implementation_digest'}
            == {k:v for k,v in current.items() if k!='implementation_digest'})


def run(args):
    if getattr(args,'seeds',None) is not None:raise ValueError('Campaign uses --start-seed/--max-scenes, not --seeds')
    if not args.output:raise ValueError('Campaign requires --output (use same path with --resume)')
    positive=('target_success','max_scenes','max_requests','max_cost_usd','request_cost_cap',
              'max_hours','max_per_template','worker_timeout','max_tokens')
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in positive):
        raise ValueError('Targets, limits and budgets must be positive and finite')
    if args.retries<0 or args.start_seed<0 or args.start_seed+args.max_scenes>2**32 or args.min_free_gb<0:
        raise ValueError('Invalid retry/seed/disk limit')
    if not 0<=args.validation_fraction<1:raise ValueError('Invalid validation fraction')
    reachable=set(args.modes)|({'vision'} if 'privileged' in args.modes and args.distill_vision else set())
    if not set(args.target_modes)<=reachable:raise ValueError('Target mode cannot be produced by selected modes')
    if len(set(args.modes))!=len(args.modes) or len(set(args.target_modes))!=len(args.target_modes):
        raise ValueError('Duplicate modes are not allowed')
    # Runtime time/disk controls may change; data semantics and cumulative ceilings may not.
    fields=('target_success','target_modes','modes','start_seed','max_scenes','max_requests','max_cost_usd',
            'request_cost_cap','retries','max_per_template','max_tokens','worker_timeout','distill_vision','mock','validation_fraction')
    config={k:getattr(args,k) for k in fields}
    config['implementation_digest']=digest([Path(__file__).read_text(),Path(__file__).with_name('teacher.py').read_text()])
    root=Path(args.output).resolve()
    if not args.resume:root.mkdir(parents=True,exist_ok=False)
    elif not (root/'state.json').exists():raise ValueError('No campaign checkpoint to resume')
    with (root/'campaign.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Campaign already running')
        state_path=root/'state.json'
        if args.resume:
            state=read(state_path)
            if not compatible_config(state['config'],config):
                raise ValueError('Resume configuration differs from checkpoint')
            if state['config']!=config:
                backup=root/'checkpoint-history'/datetime.now().strftime('%Y%m%d_%H%M%S_%f.json')
                dump(backup,state)
                state['config']=config
                atomic(state_path,state)
        else:
            state=dict(version=1,config=config,attempts=[],scene_index=0,mode_index=0,retry=0,
                       scene_errors=[],stop_reason=None)
            atomic(state_path,state)
        os.environ['NUMBA_CACHE_DIR']=str(root/'numba-cache')
        # A crash may occur after the request was sent. Never automatically send it again.
        for attempt in state['attempts']:
            if attempt['state']=='started':
                summary=Path(attempt['raw'])/'summary.json'
                if summary.exists() and (Path(attempt['raw'])/'records.jsonl').exists() and not read(summary).get('aborted'):
                    attempt['summary']=read(summary);attempt['state']='done'
                    advance(state,args,attempt)
                    atomic(state_path,state)
                else:raise RuntimeError(f"uncertain attempt {attempt['index']}; inspect artifacts; automatic resume blocked")
            if attempt.get('summary',{}).get('aborted'):
                raise RuntimeError('Previous system/API error; automatic resume blocked pending audit')
        started=time.monotonic()
        while True:
            stats,selected=inventory(state['attempts'],args.max_per_template)
            state['stats']=stats
            spent=sum(a.get('summary',{}).get('accounting_cost_usd',args.request_cost_cap) for a in state['attempts'])
            state['accounting_cost_usd']=spent
            state['request_slots_used']=len(state['attempts'])
            state['http_attempts']=sum(a.get('summary',{}).get('http_attempts',0) for a in state['attempts'])
            reason=None
            if all(stats['accepted'][m]>=args.target_success for m in args.target_modes):reason='target_reached'
            elif len(state['attempts'])>=args.max_requests:reason='request_limit'
            elif spent+args.request_cost_cap>args.max_cost_usd:reason='cost_limit'
            elif state['scene_index']>=args.max_scenes:reason='scene_limit'
            elif time.monotonic()-started>=args.max_hours*3600:reason='time_limit'
            elif shutil.disk_usage(root).free<args.min_free_gb*1024**3:reason='disk_limit'
            state['stop_reason']=reason;atomic(state_path,state)
            if reason:break
            seed=args.start_seed+state['scene_index'];mode=args.modes[state['mode_index']]
            scene=root/'scenes'/f'seed-{seed}';manifest=scene/'manifest/scenes.json'
            if not manifest.exists():
                if scene.exists():
                    raise RuntimeError(f'Incomplete scene preparation: {scene}; audit before resume')
                scene.mkdir(parents=True)
                try:prepare(manifest,scene,[seed])
                except Exception as exc:
                    state['scene_errors'].append({'seed':seed,'error':type(exc).__name__})
                    state['scene_index']+=1;state['mode_index']=0;state['retry']=0
                    atomic(state_path,state)
                    raise RuntimeError('Scene preparation failed; saved checkpoint, repair service then resume') from exc
            index=len(state['attempts']);raw=root/'attempts'/f'{index:06d}-{mode}-seed-{seed}'
            attempt=dict(index=index,seed=seed,mode=mode,retry=state['retry'],raw=str(raw),state='started',reserved_cost_usd=args.request_cost_cap)
            state['attempts'].append(attempt)
            state['accounting_cost_usd']=spent+args.request_cost_cap
            state['request_slots_used']=len(state['attempts'])
            atomic(state_path,state)
            options=argparse.Namespace(output=str(raw),manifest=str(manifest),seeds=[seed],modes=[mode],
                max_requests=1,max_cost_usd=args.request_cost_cap,max_tokens=args.max_tokens,
                worker_timeout=args.worker_timeout,distill_vision=args.distill_vision,mock=args.mock)
            collect(options)
            attempt['summary']=read(raw/'summary.json')
            if attempt['summary']['aborted']:
                # Upper reservation remains charged even if the underlying collector could not account for a crash.
                attempt['summary']['accounting_cost_usd']=max(args.request_cost_cap,attempt['summary']['accounting_cost_usd'])
                attempt['state']='blocked';state['stop_reason']='system_error';atomic(state_path,state)
                raise RuntimeError(f'Collection aborted: {raw}; automatic retry disabled')
            attempt['state']='done';advance(state,args,attempt);atomic(state_path,state)
            print(json.dumps({'attempts':len(state['attempts']),'seed':seed,'mode':mode,'status':'recorded'}),flush=True)
        # Each completed invocation exports a fresh immutable view; raw attempts and earlier exports remain untouched.
        view=root/'exports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');view.mkdir(parents=True)
        (view/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        for mode in ('privileged','vision'):
            export_data([view],view/mode,mode,args.validation_fraction)
        dump(view/'stats.json',state)
        print(json.dumps({'stop_reason':state['stop_reason'],'accepted':state['stats']['accepted'],
                          'exports':str(view),'request_slots_used':state['request_slots_used'],
                          'http_attempts':state['http_attempts'],'estimated_cost_usd':state['accounting_cost_usd']}),flush=True)
        return state


def advance(state,args,attempt):
    records=Path(attempt['raw'])/'records.jsonl'
    rows=[json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    retryable=any(r.get('source') in ('teacher','mock') and r.get('mode')==attempt['mode']
                  and r.get('status') in ('task_failure','invalid_program') for r in rows)
    if retryable and state['retry']<args.retries:
        state['retry']+=1;return
    state['retry']=0;state['mode_index']+=1
    if state['mode_index']>=len(args.modes):state['mode_index']=0;state['scene_index']+=1
