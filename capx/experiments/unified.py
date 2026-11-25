"""V31 validated orchestration over existing runners; no new robot execution loop."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

ROOT=Path(__file__).resolve().parents[2]
ABLATIONS=('none','no_selective','no_future','no_memory','no_graph_repair','no_cost','no_causal','no_local_repair','no_resume_validation')
UNAVAILABLE={'CaP-Agent0':'Original multimodel/image-differencing policy is not reproduced; B0 is not CaP-Agent0.',
 'ASPIRE':'Perception services reused; complete paper policy is not integrated.',
 'LIBERO-Recover':'Only scene initialization gate exists; policy/control mapping and protocol state layout unresolved.'}

class RunSpec(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    schema_version:Literal[1]=1
    task:str='pick_place'
    methods:list[str]=Field(default_factory=lambda:['G0','G1'])
    environment:str='robosuite'
    seeds:list[int]=Field(default_factory=lambda:[0])
    repeats:int=Field(default=1,ge=1,le=100)
    observation:str='vision'
    force_mode:Literal['off','on','compare']='off'
    provider:str='deepseek'
    model:str='deepseek-v4-pro'
    skill:str='auto'
    planner:str='pyroki'
    scenario:str='normal'
    ablation:str='none'
    config:str|None=None
    ik_runtime:str|None=None
    external_logiv:str|None=None
    snapshot:str|None=None
    reference:str|None=None
    snapshot_group:str='G1'
    harness:Literal['legacy','persistent']='legacy'
    recovery_requests:int=Field(default=0,ge=0,le=2)
    output:str|None=None
    max_requests:int|None=Field(default=None,ge=1)
    cost_limit_usd:float|None=Field(default=None,gt=0)
    wall_seconds:int|None=Field(default=None,ge=10)
    max_tokens:int|None=Field(default=None,ge=64)
    reasoning_effort:Literal['low','medium','high']|None=None
    request_timeout:int|None=Field(default=None,ge=10)
    max_blocks:int|None=Field(default=None,ge=1)
    block_seconds:int|None=Field(default=None,ge=1)
    api_budget:int|None=Field(default=None,ge=1)
    episode_seconds:int|None=Field(default=None,ge=10)
    skill_budget:int|None=Field(default=None,ge=1)
    log_level:Literal['DEBUG','INFO','WARNING','ERROR']='INFO'

def capabilities():
    defaults=RunSpec(**local_defaults()).model_dump()
    return dict(methods=[{'id':m,'available':True,'tasks':['pick_place'] if m=='LOGIV' else (['stack','lift','restack'] if m=='B0' else ['pick_place','stack','lift','restack'])} for m in ('B0','G0','G1','LOGIV')]+
        [dict(id=m,available=False,reason=r,tasks=[]) for m,r in UNAVAILABLE.items()],
        tasks=['pick_place','stack','lift','restack'],providers=[dict(id='deepseek',available=True),dict(id='openai',available=False,reason='Existing graph runner budget/receipt contract is DeepSeek-specific; legacy interactive UI remains available.'),dict(id='local',available=False,reason='No graph-runner request/cost adapter yet.')],
        observations=[dict(id='vision',available=True),dict(id='privileged',available=False,reason='Graph benchmark runners enforce public Vision. Legacy Oracle/Teacher entry remains separate; no silent truth substitution.')],
        force_modes=['off','on','compare'],skills=['auto','v32','v30','v29','v23'],planners=[dict(id='pyroki',available=True),dict(id='curobo',available=False,reason='V30 actual /plan failed CUDA 804; no qualified replacement backend.')],ablations=list(ABLATIONS),defaults=defaults)

def local_defaults():
    path=ROOT/'env_configs/inference/local-v31.json'
    return read(path) if path.exists() else {}

def read(path):return json.loads(Path(path).read_text())

def resolve(spec):
    s=spec.model_copy(deep=True)
    for name,value in local_defaults().items():
        if name not in ('ik_runtime','external_logiv'):raise ValueError('Local defaults may only select service runtime and external source')
        if getattr(s,name) is None:setattr(s,name,value)
    if s.environment!='robosuite':raise ValueError('Only existing Robosuite runners are integrated; LIBERO policy gate unavailable')
    if s.provider!='deepseek' or s.model!='deepseek-v4-pro':raise ValueError('Existing graph budget supports deepseek/deepseek-v4-pro only; no automatic model substitution')
    if s.observation!='vision':raise ValueError('Graph runners support Vision only; privileged requires the legacy Oracle/Teacher runner')
    if s.planner!='pyroki':raise ValueError('Only qualified PyRoKi backend available; cuRobo /plan failed CUDA804')
    if not s.methods or len(set(s.methods))!=len(s.methods):raise ValueError('Select unique methods')
    for method in s.methods:
        if method not in ('B0','G0','G1','LOGIV'):raise ValueError(UNAVAILABLE.get(method,'Unknown method: '+method))
    if not s.seeds or len(set(s.seeds))!=len(s.seeds) or any(type(x)!=int or not 0<=x<2**32 for x in s.seeds):raise ValueError('Unique uint32 Seeds required')
    family='contract' if s.task=='pick_place' else 'grounded'
    if family=='grounded' and s.task not in ('stack','lift','restack'):raise ValueError('Task unavailable in current runners')
    if family=='contract' and 'B0' in s.methods:raise ValueError('PickPlace has no legal B0 raw-API adapter; do not alias G0 as B0')
    if family=='grounded' and 'LOGIV' in s.methods:raise ValueError('LOGIV adapter currently requires Contract Skills/PickPlace')
    if s.ablation not in ABLATIONS:raise ValueError('Unknown ablation')
    if s.ablation!='none' and 'G1' not in s.methods:raise ValueError('Ablations apply to G1 only')
    if family=='grounded' and s.ablation!='none':raise ValueError('Contract ablations are not equivalent to grounded runner switches')
    if family=='contract' and s.scenario not in ('normal','evidence_expiry','effect_displacement'):raise ValueError('Unknown Contract scenario; natural failures are recorded in normal runs')
    if family=='grounded' and s.scenario not in ('normal','support_refresh','missing_baseline','target_shift'):raise ValueError('Unknown grounded scenario')
    if ('LOGIV' in s.methods or 'B0' in s.methods) and s.scenario!='normal':raise ValueError('This external/B0 adapter lacks an equivalent controlled-fault interface')
    if s.force_mode!='off' and (family!='contract' or any(m not in ('G0','G1') for m in s.methods)):raise ValueError('F/T comparison currently requires PickPlace G0/G1')
    if s.harness=='persistent' and (family!='contract' or any(m not in ('G0','G1') for m in s.methods)):raise ValueError('Persistent Harness currently supports Contract G0/G1 only')
    if s.recovery_requests and s.harness!='persistent':raise ValueError('Recovery requests require persistent Harness')
    if s.skill=='auto':s.skill=('v32' if s.force_mode!='off' else 'v30') if family=='contract' else 'v23'
    if s.skill not in (('v29','v30','v32') if family=='contract' else ('v23',)):raise ValueError('Skill version does not match runner')
    if s.harness=='persistent' and s.skill=='v29':raise ValueError('Persistent open recovery requires staged v30/v32 Skills')
    if s.force_mode!='off' and s.skill!='v32':raise ValueError('Force feedback requires shared V32 Skills')
    if s.reference and family!='contract':raise ValueError('Use graph_v2 --initial-manifest for provenance-validated grounded replay; unified replay currently Contract only')
    if s.reference and s.snapshot:raise ValueError('Replay reference already owns its Snapshot; do not override')
    if s.reference and (len(s.seeds)!=1 or s.repeats!=1):raise ValueError('Historical replay is one exact scene, no independent-generation repeats')
    if s.snapshot:
        snap=read(s.snapshot)
        if len(s.seeds)!=1 or snap.get('seed')!=s.seeds[0]:raise ValueError('Snapshot Seed must match the single requested Seed')
    default=('development-v33.json' if s.harness=='persistent' else 'development-v32.json' if s.skill=='v32' else 'development-v30.json') if family=='contract' else 'development-v31-grounded.json'
    s.config=str((ROOT/'env_configs/graph_feedback'/default) if s.config is None else Path(s.config).resolve())
    cfg=read(s.config)
    from capx.experiments.graph_campaign import reject_secrets
    reject_secrets(cfg)
    if family=='contract' and (not isinstance(cfg.get('tasks'),dict) or s.task not in cfg['tasks']):raise ValueError('Config does not contain selected Contract task')
    if family=='grounded' and isinstance(cfg.get('tasks'),dict):raise ValueError('Grounded runner requires its own grounded config')
    if s.ik_runtime:cfg['services']['pyroki']=read(s.ik_runtime)['url']
    for name in ('max_requests','cost_limit_usd','wall_seconds','max_tokens','reasoning_effort','request_timeout','max_blocks','block_seconds'):
        if getattr(s,name) is not None:cfg['budgets'][name]=getattr(s,name)
    limits=cfg['tasks'][s.task] if family=='contract' else cfg['budgets']
    for name in ('api_budget','episode_seconds','skill_budget'):
        if getattr(s,name) is not None:
            if name=='skill_budget' and family!='contract':raise ValueError('Grounded runner has no Contract skill budget')
            limits[name]=getattr(s,name)
    if family=='contract':
        limits['skill_recovery_version']=int(s.skill[1:]);limits['force_feedback']=s.force_mode=='on'
        if s.harness=='persistent':
            limits['harness_version']=33;limits['harness_max_requests']=s.recovery_requests
            if s.reference and s.recovery_requests:cfg['budgets']['max_requests']=min(cfg['budgets']['max_requests'],s.recovery_requests)
        elif limits.get('harness_version'):raise ValueError('Persistent config requires --harness persistent')
    if 'LOGIV' in s.methods:
        if not s.external_logiv:raise ValueError('LOGIV requires --external-logiv pinned official checkout')
        import subprocess
        cfg['external_logiv']=dict(path=str(Path(s.external_logiv).resolve()),revision=subprocess.check_output(['git','-C',s.external_logiv,'rev-parse','HEAD'],text=True).strip())
        from capx.experiments.external_logiv import load_official
        load_official(cfg['external_logiv']['path'],cfg['external_logiv']['revision'])
    return s,cfg,family

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)));temp.replace(path)

class NoNewLLM:
    count=0
    def query(self,*args):raise RuntimeError('Historical replay: new LLM requests disabled')

def execute(spec):
    s,cfg,family=resolve(spec)
    from capx.experiments.graph_v2 import configure_services
    configure_services(cfg)
    from capx.experiments.graph_campaign import PersistentModelBudget,archive_inventory,verify_source_hashes,lock
    from capx.execution.graph_journal import GraphJournal
    out=Path(s.output or ROOT/'outputs/inference'/time.strftime('%Y%m%d-%H%M%S')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    if (out.parent/'stop.pending').exists():(out/'stop.request').touch()
    logging.basicConfig(level=s.log_level)
    atomic(out/'spec.json',s.model_copy(update={'output':str(out)}).model_dump());atomic(out/'config.json',cfg)
    state=dict(state='starting',stage='initializing',started_unix=time.time(),episodes=[],budget=cfg['budgets'],pid=os.getpid())
    def status(**update):state.update(update);atomic(out/'status.json',state)
    hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in cfg.get('sources',[]) if Path(x).is_file()}
    for x in [Path(__file__),ROOT/'capx/execution/graph_journal.py',ROOT/'capx/experiments/artifacts.py',Path(s.config),ROOT/'env_configs/inference/local-v31.json']:hashes[str(x.resolve())]=hashlib.sha256(x.read_bytes()).hexdigest()
    atomic(out/'source-archive.json',archive_inventory(out,hashes))
    budget=NoNewLLM() if s.reference and not s.recovery_requests else PersistentModelBudget(out,cfg['budgets'])
    deadline=time.monotonic()+cfg['budgets']['wall_seconds']
    try:
        # Shared lock prevents two unified workers from competing for the same simulator/services.
        with lock(ROOT/'outputs/inference-worker.lock'):
            for seed in s.seeds:
                for repeat in range(s.repeats):
                    if (out/'stop.request').exists() or time.monotonic()>=deadline:
                        status(state='stopped',stage='pair_boundary',reason='operator_stop' if (out/'stop.request').exists() else 'wall_budget');return out
                    if verify_source_hashes(hashes):raise RuntimeError('Source drift before pair')
                    unit=out/f'seed-{seed}-repeat-{repeat}';unit.mkdir();atomic(unit/'config.json',cfg)
                    status(state='running',stage='paired_execution',seed=seed,repeat=repeat,current=str(unit.relative_to(out)))
                    if family=='contract':
                        from capx.experiments.contract_models import run_episode
                        initial=None;snap=read(s.snapshot) if s.snapshot else None;common=copy.deepcopy(cfg['tasks'][s.task])
                        if s.reference:
                            ref=Path(s.reference);initial=read(ref/'initial.json');initial['historical_replay']=True
                            snap=read(ref/s.snapshot_group/'snapshot.json')
                            if snap['seed']!=seed:raise ValueError('Historical Snapshot Seed mismatch')
                            common=read(ref/'config.json')['tasks'][s.task]
                            atomic(unit/'initial.json',initial)
                        if not s.reference:budget.prefix=f'seed-{seed}-repeat-{repeat}-'
                        arms=[(method,enabled) for enabled in ([False,True] if s.force_mode=='compare' else [s.force_mode=='on']) for method in s.methods]
                        for method,enabled in arms:
                            name=method+('-'+s.ablation if method=='G1' and s.ablation!='none' else '')
                            if s.force_mode=='compare':name=('D-FT-' if method=='G1' else 'C-FT-')+method if enabled else ('B-Vision-' if method=='G1' else 'A-Vision-')+method
                            dest=unit/name
                            status(stage='episode',method=name,current=str(dest.relative_to(out)))
                            with GraphJournal(dest/'graph-events.jsonl') as journal:
                                variant=copy.deepcopy(cfg['tasks'][s.task]);variant['force_feedback']=enabled
                                row,initial,snap=run_episode(dest,variant,seed,'G0' if method=='LOGIV' else method,budget,cfg,initial,snap,study=s.scenario,ablation=s.ablation if method=='G1' else 'none',environment_spec=common,external_method=cfg.get('external_logiv') if method=='LOGIV' else None)
                            row.update(variant=name,task=s.task,repeat=repeat,artifact=str(dest.relative_to(out)),telemetry_error=journal.error)
                            state['episodes'].append(row);status(requests=budget.count)
                            if initial is None:raise RuntimeError('Initial generation failed; paired arms not silently regenerated')
                    else:
                        from capx.experiments.graph_v2 import run_pair
                        budget.prefix=f'seed-{seed}-repeat-{repeat}-'
                        result=run_pair(unit,s.task,seed,repeat,read(s.snapshot) if s.snapshot else None,cfg,budget,tuple(s.methods),'none' if s.scenario=='normal' else s.scenario)
                        state['episodes'].extend(result['episodes']);status(requests=budget.count)
                    if verify_source_hashes(hashes):raise RuntimeError('Source drift during pair; archived code retained')
            status(state='complete',stage='finished',reason='all_requested_pairs_completed')
    except BaseException as exc:
        status(state='stopped' if isinstance(exc,KeyboardInterrupt) else 'error',stage='terminated',reason=type(exc).__name__+': '+str(exc)[:700]);raise
    finally:
        status(finished_unix=time.time(),seconds=time.time()-state['started_unix'],requests=budget.count)
        atomic(out/'summary.json',dict(episodes=state['episodes'],unique_requests=budget.count,estimated_cost_usd=getattr(getattr(budget,'ledger',None),'spent',{}).get('deepseek',0)))
        atomic(out/'source-postcheck.json',dict(changed=verify_source_hashes(hashes)))
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('capabilities',help='Supported combinations and explicit unavailable reasons')
    for action in ('run','replay'):
        a=sub.add_parser(action,help='Shared Snapshot/initial program across methods; seeds × repeats form a batch')
        a.add_argument('--spec',help='Unified JSON configuration; explicit CLI flags override')
        a.add_argument('--dry-run',action='store_true',help='Validate and print resolved nonsecret configuration, no API or simulation')
        for key in RunSpec.model_fields:
            if key=='schema_version':continue
            typ=int if key in ('repeats','max_requests','wall_seconds','max_tokens','request_timeout','max_blocks','block_seconds','api_budget','episode_seconds','skill_budget','recovery_requests') else float if key=='cost_limit_usd' else str
            kw=dict(type=typ,default=argparse.SUPPRESS)
            if key in ('seeds','methods'):kw.update(nargs='+',type=int if key=='seeds' else str)
            a.add_argument('--'+key.replace('_','-'),**kw)
    a=sub.add_parser('inspect');a.add_argument('path')
    a=sub.add_parser('stop');a.add_argument('path',help='Request stop after a complete method pair')
    a=sub.add_parser('serve');a.add_argument('--host',default='127.0.0.1');a.add_argument('--port',type=int,default=8200)
    args=vars(p.parse_args(argv));action=args.pop('action')
    if action=='capabilities':print(json.dumps(capabilities(),ensure_ascii=False,indent=2));return
    if action=='serve':
        from capx.web.server import main as serve,ServerArgs
        serve(ServerArgs(**args));return
    if action=='stop':
        (Path(args['path'])/'stop.request').touch(exist_ok=True);print('Stop requested at complete pair boundary');return
    if action=='inspect':
        from capx.experiments.artifacts import episode_summary
        print(json.dumps(episode_summary(Path(args['path'])),ensure_ascii=False,indent=2));return
    dry=args.pop('dry_run');path=args.pop('spec',None);data=read(path) if path else {};data.update(args)
    try:
        spec=RunSpec(**data)
        if action=='replay' and not spec.reference:raise ValueError('replay requires --reference')
        spec,cfg,family=resolve(spec)
        if dry:print(json.dumps(dict(spec=spec.model_dump(),runner=family,resolved_config=cfg),ensure_ascii=False,indent=2))
        else:print(execute(spec))
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

if __name__=='__main__':main()
