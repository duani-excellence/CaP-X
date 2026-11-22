"""Durable all-outcome B0/G0/G1 evaluation scheduling, freezing and accounting.

Unlike Teacher collection this never selects successful samples or retries a
failed episode. Unknown request/worker completion blocks automatic continuation.
Runtime service imports happen only after the frozen environment is applied.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
import uuid

GROUPS=('B0','G0','G1')
ROOT=Path(__file__).resolve().parents[2]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def json_value(value):
    if isinstance(value,Path):return str(value)
    if hasattr(value,'tolist'):return value.tolist()
    raise TypeError('Not JSON serializable: '+type(value).__name__)


def atomic(path,value):
    # Same durable checkpoint protocol as training.campaign.atomic, without its
    # eager teacher/simulation imports or success-selection semantics.
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2,default=json_value);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


@contextmanager
def lock(path):
    with Path(path).open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Campaign already running') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def make_plan(tasks,development_scenes=20,test_scenes=100,repeats=2,seed=20261120):
    tasks=list(tasks)
    if not tasks or len(set(tasks))!=len(tasks):raise ValueError('Use unique tasks')
    if min(development_scenes,test_scenes,repeats)<1:raise ValueError('Counts must be positive')
    rng=random.Random(seed);used=set();result=dict(schema=1,tasks=tasks,repeats=repeats,allocation_seed=seed,development=[],test=[])
    for split,n in (('development',development_scenes),('test',test_scenes)):
        for task in tasks:
            for index in range(n):
                while True:
                    scene=rng.randrange(1000000,2**31-1)
                    if scene not in used:break
                used.add(scene);result[split].append(dict(task=task,seed=scene,scene_index=index))
    return result


def reject_secrets(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if re.search(r'(^|_)(api_key|authorization|password|access_token|private_key|secret)($|_)',k,re.I):raise ValueError('Configuration must not contain secret fields')
            reject_secrets(v)
    elif isinstance(value,list):
        for item in value:reject_secrets(item)


def source_inventory(config,*,include_run_inputs=True):
    """Full frozen provenance, or business-only dependencies for one unit.

    Run inputs are archived once in the campaign. Unit archives link that
    immutable campaign and retain only the selected initial/Snapshot payload.
    """
    paths=[]
    for field in ('sources','prompt_files','scoring_files'):
        if not config.get(field):raise ValueError('Freeze requires '+field)
        paths.extend(config[field])
    for field in ('analysis_sources','cli_files','asset_files'):paths.extend(config.get(field,[]))
    calibration=config.get('robot_calibration_file')
    if calibration:
        paths.append(str(Path(calibration).resolve()))
        for entry in read(calibration).get('sources',[]):
            path=Path(entry['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('sha256'):raise ValueError('Robot calibration source drift: '+str(path))
            paths.append(str(path.resolve()))
    inventory=config.get('service_inventory')
    if inventory:
        paths.append(str(Path(inventory).resolve()))
        for service in read(inventory).get('services',{}).values():
            for process in service.get('processes',[]):
                for raw,entry in process.get('source_files',{}).items():
                    file=Path(raw)
                    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=entry.get('sha256'):raise ValueError('Service inventory source drift: '+raw)
                    paths.append(raw)
    paths.extend([str(Path(__file__)),str(Path(__file__).with_name('graph_statistics.py'))])
    manifest=config.get('snapshot_manifest') if include_run_inputs else None
    if manifest:
        manifest=Path(manifest).resolve();paths.append(str(manifest))
        for row in read(manifest)['scenes']:
            if isinstance(row.get('snapshot'),str):paths.append(str(manifest.parent/row['snapshot']))
    if include_run_inputs and config.get('initial_manifest'):
        from capx.experiments.shared_initial import reference_files
        paths.extend(reference_files(config['initial_manifest']))
    acceptance=config.get('acceptance_file') if include_run_inputs else None
    if acceptance:
        acceptance=Path(acceptance).resolve();paths.append(str(acceptance))
        for raw in read(acceptance).get('evidence_files',[]):
            path=Path(raw);paths.append(str(path if path.is_absolute() else acceptance.parent/path))
    resolved=[Path(raw) if Path(raw).is_absolute() else ROOT/raw for raw in paths]
    from capx.experiments.graph_preflight import source_closure
    import_roots=[ROOT,ROOT/'capx/third_party/robosuite']+[p.parent for p in resolved if not p.is_relative_to(ROOT)]
    paths=source_closure(resolved,import_roots)+resolved
    result={}
    for raw in paths:
        path=Path(raw);path=path if path.is_absolute() else ROOT/path
        if not path.is_file():raise ValueError('Missing freeze input: '+str(path))
        result[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result



def archive_inventory(root,hashes):
    """Persist exactly the hashed bytes, including external static assets."""
    root=Path(root);manifest={}
    for raw,digest in sorted(hashes.items()):
        path=Path(raw);content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Source changed during archive: '+raw)
        relative=Path('source-bytes')/digest/path.name;dest=root/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.read_bytes()!=content:raise ValueError('Conflicting source archive')
        else:
            with dest.open('xb') as stream:stream.write(content)
        manifest[raw]={'sha256':digest,'archive_file':str(relative)}
    return manifest


def verify_source_hashes(hashes):
    return [raw for raw,digest in hashes.items() if not Path(raw).is_file() or hashlib.sha256(Path(raw).read_bytes()).hexdigest()!=digest]


def declared_sources(config):
    return {str((Path(p) if Path(p).is_absolute() else ROOT/p).resolve())
            for field in ('sources','prompt_files','scoring_files') for p in config[field]}


def development_execution_evidence(dev_root,state,config):
    """Count fixed-scene attempts, including real model generation failures.

    No success/valid-program selection is allowed. Failed attempts remain evidence. A scene whose
    environment never started, a mock or a missing group is not evidence that
    the implementation was exercised. This is provenance validation, not proof
    that a maliciously fabricated artifact is authentic.
    """
    scenes={task:{} for task in config['tasks']};evidence={};rejected=[]
    for unit in state['units']:
        task=unit['task']
        if task not in scenes:continue
        scenes[task].setdefault(unit['seed'],[])
        valid=False;files=[]
        try:
            if unit['state']!='done' or set(unit['groups'])!=set(GROUPS) or unit['fault']!='none':raise ValueError('incomplete normal three-system unit')
            result_path=Path(unit['result']);rows=_validate_result(read(result_path),unit);files.append(result_path)
            if len({r.get('snapshot_hash') for r in rows})!=1 or not rows[0].get('snapshot_hash'):raise ValueError('missing or unpaired Snapshot')
            paired={r['group']:r for r in rows}
            for key in ('code_hash','initial_prompt_hash','api_docs_hash'):
                if not paired['G0'].get(key) or paired['G0'][key]!=paired['G1'].get(key):raise ValueError('unpaired initial '+key)
            for row in rows:
                if row.get('mock') or row.get('online_vision') is not True:raise ValueError('not an online attempt')
                if row.get('status') in ('scene_preparation_error','not_started') or not row.get('stop_reason'):raise ValueError('missing terminal attempt')
                persisted=result_path.parent/'episodes'/row['group']/'result.json'
                if read(persisted)!=row:raise ValueError('result artifact differs from ledger')
                files.append(persisted)
                initial_state=persisted.parent/'initial-evaluation-state.json'
                if not isinstance(read(initial_state),dict) or not read(initial_state):raise ValueError('missing initialized environment state')
                files.append(initial_state)
                trace=Path(row['trace_path']);files.append(trace)
                trace_rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
                if not any(r.get('kind')=='vision_boundary' and r.get('initial_state_hash')==row['snapshot_hash'] and r.get('seed')==unit['seed'] and r.get('snapshot_replay_verified') is True for r in trace_rows):raise ValueError('no verified environment Snapshot initialization')
                if row.get('actual_api_calls',0)>0 and not any(r.get('kind')=='api' and isinstance(r.get('before'),dict) and isinstance(r.get('after'),dict) for r in trace_rows):raise ValueError('no physical before/after trace')
                receipt_found=False
                for request_path in row.get('request_paths',[]):
                    receipt_path=Path(request_path)/'receipt.json';receipt=read(receipt_path);files.append(receipt_path)
                    if receipt.get('http_status')==200 and receipt.get('response_model')==config['model'] and receipt.get('attempts',0)>0 and not receipt.get('mock'):
                        receipt_found=True
                        response_path=Path(request_path)/'response.json'
                        if not isinstance(read(response_path),dict):raise ValueError('missing model response artifact')
                        files.append(response_path)
                if not receipt_found:
                    # Network/model errors are valid attempted outcomes, too. Match
                    # the initial request's exact unit/group label; no re-request.
                    ledger_path=Path(dev_root)/'request-ledger.json'
                    label=unit['id']+f":initial-{task}-seed-{unit['seed']}-repeat-{unit['repeat']}-"+('B0' if row['group']=='B0' else 'shared-G0-G1')
                    ledger=read(ledger_path) if ledger_path.exists() else {}
                    for request in ledger.get('requests',[]):
                        receipt=request.get('receipt',{})
                        if request.get('label')==label and request.get('state')=='failed' and receipt.get('requested_model')==config['model'] and receipt.get('attempts',0)>0 and not receipt.get('mock'):
                            files.append(ledger_path)
                            candidates=list(Path(request['directory']).glob('llm/*/receipt.json'))
                            if len(candidates)!=1 or read(candidates[0])!=receipt:raise ValueError('failed request receipt differs from ledger')
                            files.extend(candidates);receipt_found=True
                if not receipt_found:raise ValueError('no real model attempt receipt')
            valid=True
        except (ValueError,KeyError,TypeError,OSError) as exc:
            rejected.append(dict(task=task,seed=unit['seed'],repeat=unit['repeat'],reason=str(exc)))
        scenes[task][unit['seed']].append(valid)
        if valid:
            evidence.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    counts={task:sum(bool(repeats) and all(repeats) for repeats in seeds.values()) for task,seeds in scenes.items()}
    return counts,evidence,rejected


CAMPAIGN_SCALE_BUDGETS={'max_requests','cost_limit_usd','wall_seconds'}
GRAPH_MECHANISMS={'diagnostic_slice','selective_invalidation','selective_recovery'}


def compatible_development(config,prior):
    """Only declared graph mechanisms and campaign-scale limits may differ."""
    if prior.get('exploratory_only'):raise ValueError('Exploratory-only development cannot support formal freeze')
    budgets=lambda x:{k:v for k,v in x['budgets'].items() if k not in CAMPAIGN_SCALE_BUDGETS}
    if budgets(config)!=budgets(prior):raise ValueError('Development/formal per-episode or model budget differs')
    current=config.get('graph_options');original=prior.get('graph_options');ablation=None
    if current!=original:
        declared=prior.get('allowed_graph_ablations',[])
        if config.get('study')!='mechanism_ablation' or current not in declared:
            raise ValueError('Graph options change is not a predeclared mechanism ablation')
        if not isinstance(current,dict) or set(current)!=GRAPH_MECHANISMS or any(type(v) is not bool for v in current.values()):
            raise ValueError('Invalid predeclared graph options')
        if not isinstance(original,dict) or set(original)!=GRAPH_MECHANISMS or not all(original.values()):
            raise ValueError('Ablation requires full-graph normal development baseline')
        if sum(current[k]!=original[k] for k in GRAPH_MECHANISMS)!=1:
            raise ValueError('First-version mechanism ablation changes exactly one graph option')
        if set(config.get('groups',GROUPS))!={'G0','G1'}:
            raise ValueError('Mechanism ablation compares only G0/G1')
        ablation=dict(predeclared=True,options=current,development_options=original)
    for field in ('model','services','runtime_environment','robot_calibration_file','asset_files','cli_files','allowed_graph_ablations','mechanism_scene_subset'):
        if config.get(field)!=prior.get(field):raise ValueError('Development/formal configuration differs: '+field)
    return ablation


def _acceptance(config,plan,development_root):
    file=config.get('acceptance_file')
    if not file:raise ValueError('Formal freeze requires online task acceptance evidence')
    evidence=read(file);validated=evidence.get('validated_tasks',[])
    allowed={x if isinstance(x,str) else x.get('task') for x in validated}
    if not set(config['tasks'])<=allowed:raise ValueError('Task missing from acceptance validated_tasks')
    references=evidence.get('evidence_files',[])
    if not references:raise ValueError('Task acceptance requires evidence_files, not names alone')
    demonstrated=set()
    for raw in references:
        p=Path(raw);p=p if p.is_absolute() else Path(file).parent/p
        if not p.is_file():raise ValueError('Missing task acceptance evidence: '+str(p))
        body=read(p);rows=body.get('episodes',[body]) if isinstance(body,dict) else body
        for row in rows:
            if row.get('online_vision') is True and row.get('reward')==1 and row.get('status')=='success' and row.get('snapshot_hash') and not row.get('mock'):
                demonstrated.add(row.get('task'))
    if not set(config['tasks'])<=demonstrated:raise ValueError('Task acceptance lacks online physical success evidence')
    if not development_root:raise ValueError('Formal freeze requires development campaign results')
    dev_root=Path(development_root);dev=load_campaign(dev_root);state=read(dev_root/'state.json')
    if dev['phase']!='development':raise ValueError('Expected development campaign')
    if declared_sources(config)!=declared_sources(dev['config']):raise ValueError('Development/formal business source set differs')
    ablation=compatible_development(config,dev['config'])
    if config.get('service_inventory'):
        if not dev['config'].get('service_inventory'):raise ValueError('Development service identity was not recorded')
        prior=read(dev['config']['service_inventory']);current=read(config['service_inventory'])
        if {k:service_identity(v) for k,v in prior.get('services',{}).items()}!={k:service_identity(v) for k,v in current.get('services',{}).items()}:raise ValueError('Development/formal service identity differs')
    if state.get('signature')!=dev['signature']:raise ValueError('Development checkpoint signature drift')
    planned={(x['task'],x['seed'],repeat,fault) for x in dev['plan']['development'] for repeat in range(dev['plan']['repeats']) for fault in dev['config'].get('faults',['none'])}
    actual=[(x['task'],x['seed'],x['repeat'],x['fault']) for x in state['units']]
    if len(actual)!=len(set(actual)) or set(actual)!=planned:raise ValueError('20 completed scenes require intact planned development unit ledger')
    counts,execution_hashes,rejected=development_execution_evidence(dev_root,state,config)
    if any(n<20 for n in counts.values()):raise ValueError('Formal freeze requires at least 20 completed, Snapshot-initialized three-group development attempt scenes per task; validated='+str(counts)+'; rejected='+str(len(rejected)))
    # A successful standalone JSON/Oracle is not model-task acceptance. Bind
    # success to exactly the persisted records whose model, initialization and
    # physical traces were validated above against this unchanged development.
    from capx.experiments.graph_statistics import success
    persisted={str((Path(u['result']).parent/'episodes'/g/'result.json').resolve())
               for u in state['units'] if u.get('result') for g in ('G0','G1')}
    demonstrated=set()
    for raw in references:
        path=Path(raw);path=(path if path.is_absolute() else Path(file).parent/path).resolve()
        if str(path) not in persisted or str(path) not in execution_hashes:
            raise ValueError('Task acceptance must reference a validated development G0/G1 persisted result')
        row=read(path)
        if (row.get('group') in ('G0','G1') and row.get('online_vision') is True
                and row.get('executed') is True and row.get('actual_api_calls',0)>0 and success(row)):
            demonstrated.add(row.get('task'))
    if not set(config['tasks'])<=demonstrated:
        raise ValueError('Task acceptance lacks validated development G0/G1 model policy success')
    dev_seeds={u['seed'] for u in state['units']}
    if dev_seeds&{x['seed'] for x in plan['test']}:raise ValueError('Development/test seed overlap')
    return dict(file=str(Path(file).resolve()),sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest(),
                evidence_files={str((Path(r) if Path(r).is_absolute() else Path(file).parent/r).resolve()):hashlib.sha256((Path(r) if Path(r).is_absolute() else Path(file).parent/r).read_bytes()).hexdigest() for r in references},
                development_root=str(dev_root.resolve()),development_signature=dev['signature'],
                executed_development_scenes=counts,development_evidence_hashes=execution_hashes,rejected_development_units=rejected,graph_ablation=ablation)



def formal_provenance(config,plan):
    """Require immutable physical starts and explicit executable/service inputs."""
    from capx.experiments.graph_v2 import RUNTIME_ENVIRONMENT
    manifest=config.get('snapshot_manifest')
    if not manifest:raise ValueError('Formal freeze requires a complete test Snapshot manifest')
    scenes=read(manifest).get('scenes',[]);indexed={}
    for entry in scenes:
        key=(entry.get('task'),entry['seed'])
        if key in indexed:raise ValueError('Duplicate Snapshot identity')
        snapshot=entry.get('snapshot')
        if isinstance(snapshot,str):snapshot=read(Path(manifest).parent/snapshot)
        if not isinstance(snapshot,dict) or not snapshot.get('state_hash') or not snapshot.get('model_hash'):raise ValueError('Snapshot lacks physical/model hash')
        if snapshot.get('seed')!=entry['seed'] or snapshot.get('task',entry.get('task'))!=entry.get('task'):raise ValueError('Snapshot task/seed mismatch')
        if entry.get('state_hash',snapshot['state_hash'])!=snapshot['state_hash']:raise ValueError('Snapshot manifest state hash mismatch')
        indexed[key]=snapshot['state_hash']
    if not {(x['task'],x['seed']) for x in plan['test']}<=set(indexed):raise ValueError('Test Snapshot manifest incomplete')
    validate_development_services(config,live=False)
    return dict(snapshot_count=len(indexed),service_inventory=str(Path(config['service_inventory']).resolve()),
                service_identity_limit='Local file hashes, process inventory and schema; remote weights and in-memory bytes cannot be guaranteed')


def validate_development_services(config,*,live=True,inspector=None):
    """Validate formal-compatible service provenance before spending on development."""
    from capx.experiments.graph_v2 import RUNTIME_ENVIRONMENT
    if not config.get('service_inventory'):raise ValueError('Development/formal preparation requires service_inventory; create graph_preflight inventory before this campaign, or explicitly use --exploratory')
    runtime=config.get('runtime_environment',{})
    if set(runtime)!=set(RUNTIME_ENVIRONMENT):raise ValueError('Development/formal preparation requires explicit complete runtime_environment')
    if not config.get('cli_files'):raise ValueError('Development/formal preparation requires CLI script provenance')
    inventory=config.get('service_inventory')
    if not inventory:raise ValueError('Development/formal preparation requires service_inventory')
    body=read(inventory)
    if body.get('runtime',{}).get('runtime_environment')!=runtime:raise ValueError('Recorded actual runtime_environment differs from configured runtime')
    for name in ('sam3','graspnet','pyroki'):
        service=body.get('services',{}).get(name,{})
        if service.get('url','').rstrip('/')!=config['services'].get(name,'').rstrip('/') or service.get('openapi',{}).get('http_status')!=200:raise ValueError('Service inventory URL/schema mismatch: '+name)
        if name=='graspnet' and service.get('seed_request_response') is not True:raise ValueError('GraspNet seed schema not verified')
        if not service.get('processes'):raise ValueError('Service loaded source identity unavailable: '+name)
        for process in service['processes']:
            files=process.get('source_files',{})
            if process.get('source_discovery')!='recursive_local_ast_with_initializers':raise ValueError('Service inventory lacks recursive local source closure: '+name)
            if not files or any(row.get('modified_after_process_start') is not False for row in files.values()):raise ValueError('Service source changed after start or identity unknown: '+name)
    return verify_service_inventory(config,inspector=inspector) if live else dict(inventory=str(Path(inventory).resolve()),validated=True)



def prepare_snapshots(root,config,plan,*,split='test',resume=False,max_scenes=None,prepare=None):
    """Prepare fixed initial states without a provider; never select by task success.

    Each physical preparation is isolated in its own new directory. Failed or
    unknown started scenes are retained, not replaced or silently rerun.
    """
    if split not in ('development','test'):raise ValueError('Invalid Snapshot split')
    if config['tasks']!=plan['tasks']:raise ValueError('Task plan/config mismatch')
    if max_scenes is not None and (type(max_scenes) is not int or max_scenes<1):raise ValueError('max_scenes must be positive')
    if {x['seed'] for x in plan['development']}&{x['seed'] for x in plan['test']}:raise ValueError('Development/test seed overlap')
    selected=plan[split];identities=[(x['task'],x['seed']) for x in selected]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate Snapshot identity')
    reject_secrets(config);root=Path(root)
    if root.exists() and not resume:raise FileExistsError(root)
    if not root.exists() and resume:raise FileNotFoundError(root)
    signature=fingerprint(dict(config=config,plan=plan,split=split))
    if not root.exists():
        sources=source_inventory(config);root.mkdir(parents=True)
        atomic(root/'preparation.json',dict(schema=1,signature=signature,split=split,source_hashes=sources,
            source_archive=archive_inventory(root,sources),scenes=[dict(x,state='pending') for x in selected]))
    with lock(root/'preparation.lock'):
        state=read(root/'preparation.json')
        if state.get('signature')!=signature:raise ValueError('Snapshot preparation signature drift')
        if [(x['task'],x['seed']) for x in state['scenes']]!=identities:raise ValueError('Snapshot preparation allocation drift')
        if verify_source_hashes(state['source_hashes']):raise ValueError('Snapshot preparation source drift')
        if any(x['state']=='started' for x in state['scenes']):raise RuntimeError('Unknown started Snapshot preparation; audit before proceeding')
        from capx.experiments.graph_v2 import configure_services
        configure_services(config)
        if prepare is None:
            from capx.experiments.vision_tasks import prepare_scene
            prepare=prepare_scene
        def manifest():
            ready=[x['entry'] for x in state['scenes'] if x['state']=='done']
            for entry in ready:
                path=root/entry['snapshot']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['snapshot_file_sha256']:raise ValueError('Prepared Snapshot file drift')
            failed=[{k:v for k,v in x.items() if k!='entry'} for x in state['scenes'] if x['state']=='failed']
            body=dict(schema=1,split=split,plan_hash=fingerprint(plan),preparation_signature=signature,scenes=ready,
                failed_scenes=failed,complete=len(ready)==len(selected),model_requests=0)
            atomic(root/'scenes.json',body)
            return dict(manifest=str((root/'scenes.json').resolve()),complete=body['complete'],completed=len(ready),failed=len(failed),planned=len(selected),model_requests=0)
        manifest();completed=0
        for scene in state['scenes']:
            if scene['state']!='pending':continue
            if max_scenes is not None and completed>=max_scenes:break
            if verify_source_hashes(state['source_hashes']):raise ValueError('Snapshot preparation source drift')
            scene['state']='started';atomic(root/'preparation.json',state)
            dest=root/'scenes'/f"{scene['task']}-{scene['seed']}"
            try:
                entry,snapshot=prepare(scene['task'],scene['seed'],dest)
                if not entry.get('snapshot_replay_verified') or not entry.get('valid'):raise ValueError('Snapshot was not physically replay verified')
                if snapshot.get('seed')!=scene['seed'] or snapshot.get('task',scene['task'])!=scene['task'] or not snapshot.get('state_hash') or not snapshot.get('model_hash'):
                    raise ValueError('Invalid Snapshot identity or physical hash')
                if verify_source_hashes(state['source_hashes']):raise ValueError('Snapshot preparation source drift')
                path=dest/'prepared-snapshot.json';atomic(path,snapshot)
                scene['entry']=dict(task=scene['task'],seed=scene['seed'],snapshot=str(path.relative_to(root)),state_hash=snapshot['state_hash'],
                    model_hash=snapshot['model_hash'],snapshot_replay_verified=True,valid=True,snapshot_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                scene['state']='done'
            except Exception as exc:
                scene.update(state='failed',error_type=type(exc).__name__,error=str(exc)[:500])
            finally:atomic(root/'preparation.json',state)
            completed+=1;manifest()
        return manifest()


def selected_scenes(config,plan,phase):
    selected=plan['development' if phase=='development' else 'test']
    subset=config.get('mechanism_scene_subset')
    if subset is None:return selected
    if (not isinstance(subset,dict) or set(subset)!={'rule','scenes_per_task'}
            or subset['rule']!='first_scene_index' or type(subset['scenes_per_task']) is not int
            or subset['scenes_per_task']<1):
        raise ValueError('Invalid mechanism scene subset: use first_scene_index and positive scenes_per_task')
    count=subset['scenes_per_task']
    for task in config['tasks']:
        scenes=[s for s in plan['test'] if s['task']==task]
        indices=[s.get('scene_index') for s in scenes]
        if any(type(i) is not int for i in indices) or set(indices)!=set(range(len(scenes))) or len(indices)!=len(set(indices)) or count>len(scenes):
            raise ValueError('Mechanism scene subset requires complete unique scene indices and enough planned test scenes')
    if phase=='formal' and config.get('study')=='mechanism_ablation':
        return [s for s in selected if s['scene_index']<count]
    return selected


def planned_units(config,plan,phase):
    selected=selected_scenes(config,plan,phase);units=[]
    groups=config.get('groups',list(GROUPS))
    if not groups or len(set(groups))!=len(groups) or not set(groups)<=set(GROUPS):raise ValueError('Invalid evaluation groups')
    faults=config.get('faults',['none'])
    if not faults or len(set(faults))!=len(faults):raise ValueError('Faults must be unique')
    if phase=='formal' and config.get('study','normal')=='normal' and faults==['none'] and set(groups)!=set(GROUPS):raise ValueError('Normal formal comparison requires all three systems')
    for scene in selected:
        for repeat in range(plan['repeats']):
            for fault in faults:
                identity=dict(task=scene['task'],seed=scene['seed'],repeat=repeat,fault=fault)
                order=list(groups);random.Random(fingerprint(identity)).shuffle(order)
                units.append(dict(identity,id=fingerprint(identity)[:20],groups=order,state='pending'))
    if len({u['id'] for u in units})!=len(units):raise ValueError('Duplicate planned units')
    return units


def validate_checkpoint(frozen,state):
    if state['signature']!=frozen['signature']:raise ValueError('Checkpoint signature drift')
    expected=planned_units(frozen['config'],frozen['plan'],frozen['phase'])
    fields=('task','seed','repeat','fault','id','groups')
    identity=lambda u:{key:u.get(key) for key in fields}
    if [identity(u) for u in state['units']]!=[identity(u) for u in expected]:
        raise ValueError('Checkpoint planned unit ledger differs from frozen allocation')
    if any(u.get('state') not in ('pending','started','done') for u in state['units']):raise ValueError('Unknown checkpoint unit state')


def request_coverage(config,units):
    blocks=config['budgets']['max_blocks'];limit=config['budgets']['max_requests']
    if type(blocks) is not int or blocks<1 or type(limit) is not int or limit<1:raise ValueError('max_blocks and max_requests must be positive integers')
    per_unit=[]
    for unit in units:
        groups=unit['groups']
        initials=int('B0' in groups)+int(bool(set(groups)&{'G0','G1'}))
        per_unit.append(initials+len(groups)*(blocks-1))
    required=sum(per_unit)
    return dict(units=len(units),max_blocks=blocks,required_requests=required,
        requests_per_unit=per_unit[0] if per_unit and len(set(per_unit))==1 else None,
        configured_max_requests=limit,sufficient=limit>=required,
        scope='Unique initial requests plus every allowed group recovery; no automatic retries; costs and wall time need separate coverage')


def request_stop(root,reason='operator requested boundary stop'):
    root=Path(root);load_campaign(root,check_sources=False)
    if not isinstance(reason,str) or not reason.strip():raise ValueError('Stop reason required')
    dest=root/'stop-requests';dest.mkdir(exist_ok=True)
    path=dest/(str(time.time_ns())+'-'+uuid.uuid4().hex+'.json')
    tmp=path.with_suffix('.tmp')
    try:
        with tmp.open('x') as stream:
            json.dump(dict(reason=reason[:500],requested_unix=time.time(),policy='finish_current_pair_then_stop'),stream)
            stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)  # Publish complete bytes without replacing any existing request.
        fd=os.open(str(dest),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if tmp.exists():tmp.unlink()
    return dict(request=str(path),policy='finish_current_pair_then_stop',does_not_interrupt_inflight=True)


def consume_stop_request(root,state):
    handled=set(state.get('handled_stop_requests',[]))
    pending=[p.name for p in sorted((Path(root)/'stop-requests').glob('*.json')) if p.name not in handled]
    if not pending:return False
    state['handled_stop_requests']=sorted(handled|set(pending));state['stop_reason']='requested_boundary_stop'
    return True


def freeze_campaign(root,config,plan,*,phase='formal',development_root=None):
    root=Path(root);reject_secrets(config)
    from capx.experiments.graph_v2 import runtime_environment
    runtime_environment(config)  # Validate only; runtime values are included in frozen config.
    if root.exists():raise FileExistsError(root)
    if phase not in ('formal','development'):raise ValueError('Unknown phase')
    if phase=='formal' and config.get('study')=='mechanism_ablation' and not config.get('initial_manifest'):
        raise ValueError('Formal mechanism_ablation requires a valid initial_manifest; independent generation is forbidden')
    if list(config['tasks'])!=plan['tasks']:raise ValueError('Task plan/config mismatch')
    if set(x['seed'] for x in plan['development'])&set(x['seed'] for x in plan['test']):raise ValueError('Development/test seed overlap')
    if not config.get('services') or not config.get('model'):raise ValueError('Freeze requires model/services')
    budgets=config['budgets']
    for k in ('max_requests','cost_limit_usd','max_tokens','wall_seconds','max_blocks','api_budget','request_timeout'):
        if not isinstance(budgets.get(k),(int,float)) or not math.isfinite(budgets[k]) or budgets[k]<=0:raise ValueError('Invalid budget '+k)
    sources=source_inventory(config)
    acceptance=_acceptance(config,plan,development_root) if phase=='formal' else None
    provenance=formal_provenance(config,plan) if phase=='formal' else None
    frozen=dict(schema=1,phase=phase,config=config,plan=plan,source_hashes=sources,acceptance=acceptance,provenance=provenance)
    units=planned_units(config,plan,phase)
    if config.get('initial_manifest'):
        from capx.experiments.shared_initial import validate_initial_manifest
        validate_initial_manifest(config['initial_manifest'],config,units,phase=phase,plan=plan)
    coverage=request_coverage(config,units);frozen['request_coverage']=coverage
    if phase=='formal' and config.get('study','normal')=='normal' and config.get('faults',['none'])==['none'] and not coverage['sufficient']:
        raise ValueError('Formal normal max_requests must cover the complete frozen plan: required '+str(coverage['required_requests'])+', configured '+str(coverage['configured_max_requests']))
    root.mkdir(parents=True,exist_ok=False)
    frozen['source_archive']=archive_inventory(root,sources)
    frozen['signature']=fingerprint(frozen)
    atomic(root/'frozen.json',frozen);atomic(root/'plan.json',plan)
    atomic(root/'state.json',dict(schema=1,signature=frozen['signature'],units=units,active_seconds=0.,stop_reason=None))
    return frozen


def load_campaign(root,*,check_sources=True):
    frozen=read(Path(root)/'frozen.json');signature=frozen.pop('signature')
    if fingerprint(frozen)!=signature:raise ValueError('Frozen configuration drift')
    frozen['signature']=signature
    if check_sources:
        for raw,entry in frozen.get('source_archive',{}).items():
            archive=Path(root)/entry['archive_file']
            if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Frozen source archive drift: '+raw)
        for raw,digest in frozen['source_hashes'].items():
            path=Path(raw)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source drift: '+raw)
        for raw,digest in (frozen.get('acceptance') or {}).get('development_evidence_hashes',{}).items():
            path=Path(raw)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen development evidence drift: '+raw)
    return frozen


class PersistentModelBudget:
    """One durable cumulative request/cost ledger around the existing ModelBudget.

    A received response can be reused by its exact label+prompt hash without a
    second charge. An uncertain sent request can never automatically be retried.
    """
    def __init__(self,root,budgets,*,elapsed_seconds=0.,factory=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.options=budgets;self.factory=factory
        self.path=self.root/'request-ledger.json';self.max_requests=int(budgets['max_requests']);self.max_tokens=int(budgets['max_tokens'])
        self.deadline=time.monotonic()+max(0,budgets['wall_seconds']-elapsed_seconds)
        self.state=read(self.path) if self.path.exists() else dict(version=1,requests=[])
        self.prefix='';self._refresh()

    def _refresh(self):
        self.count=len(self.state['requests']);self.rows=self.state['requests']
        spent=sum(r.get('cost_upper_usd',r.get('reserved_usd',0)) for r in self.rows)
        self.ledger=argparse.Namespace(spent={'deepseek':spent})

    def query(self,prompt,label):
        label=self.prefix+label;ph=fingerprint(prompt)
        with lock(self.root/'request-ledger.lock'):
            self.state=read(self.path) if self.path.exists() else self.state;self._refresh()
            pending=[r for r in self.rows if r['state'] in ('started','uncertain')]
            if pending:raise RuntimeError('uncertain LLM request; audit receipt before automatic resume')
            for row in self.rows:
                if row['label']==label:
                    if row['prompt_hash']!=ph:raise ValueError('Existing request prompt changed')
                    if row['state']!='received':raise RuntimeError('Previously failed request is not automatically retried')
                    return read(Path(row['response_directory'])/'response.json'),row['response_directory']
            if self.count>=self.max_requests:raise RuntimeError('Global request budget exhausted')
            remaining=self.options['cost_limit_usd']-self.ledger.spent['deepseek']
            from capx.experiments.pilot import CostLedger
            # Reuse the provider's conservative input overhead and reserve rule.
            reserve=CostLedger(remaining).reserve('deepseek',input_bytes=len(json.dumps(prompt).encode()),max_tokens=self.max_tokens)
            if self.deadline-time.monotonic()<10:raise RuntimeError('Global wall budget exhausted')
            directory=self.root/'requests'/f'{self.count+1:06d}-{fingerprint(label)[:12]}'
            row=dict(label=label,prompt_hash=ph,state='started',reserved_usd=reserve,directory=str(directory.resolve()),started_unix=time.time())
            self.rows.append(row);atomic(self.path,self.state);self._refresh()
            factory=self.factory
            if factory is None:
                from capx.experiments.layered_feedback import ModelBudget
                factory=ModelBudget
            delegate=factory(directory,1,remaining,self.max_tokens,self.deadline,self.options.get('reasoning_effort','low'),self.options.get('request_timeout',400))
            try:
                answer,source=delegate.query(prompt,'request')
                row.update(state='received',response_directory=source)
                return answer,source
            except Exception:
                row['state']='failed' if delegate.rows and delegate.rows[-1].get('attempts',0)>0 else 'uncertain'
                raise
            finally:
                receipt=delegate.rows[-1] if delegate.rows else {}
                row.update(receipt=receipt,cost_upper_usd=receipt.get('cost_upper_usd',reserve),finished_unix=time.time())
                atomic(self.path,self.state);self._refresh()


def _snapshots(config,*,expected_hashes=None):
    def parse(path):
        path=Path(path).resolve();data=path.read_bytes()
        if expected_hashes is not None and hashlib.sha256(data).hexdigest()!=expected_hashes.get(str(path)):
            raise ValueError('Snapshot input drift: '+str(path))
        return json.loads(data)
    manifest=config.get('snapshot_manifest')
    if not manifest:return {}
    body=parse(manifest);out={}
    for row in body['scenes']:
        task=row.get('task',body.get('task'))
        if not task:raise ValueError('Snapshot entry lacks task')
        snapshot=row.get('snapshot')
        if isinstance(snapshot,str):snapshot=parse(Path(manifest).parent/snapshot)
        out[(task,row['seed'])]=snapshot
    return out


def _validate_result(result,unit):
    rows=result['episodes'] if isinstance(result,dict) else result
    if sorted(r['group'] for r in rows)!=sorted(unit['groups']):raise ValueError('Pair runner returned missing/duplicate groups')
    for row in rows:
        for field in ('task','seed','repeat','fault'):
            if row.get(field)!=unit[field]:raise ValueError('Runner identity mismatch '+field)
    return rows



def service_identity(service):
    processes=[]
    for proc in service.get('processes',[]):
        files=proc.get('source_files',{})
        if not files or any(v.get('modified_after_process_start') is not False for v in files.values()):raise ValueError('Service runtime drift: loaded-source identity unknown')
        processes.append(dict(argv_sha256=proc.get('argv_sha256'),executable=proc.get('executable'),
            source_sha256={p:x['sha256'] for p,x in files.items()},dependency_versions=proc.get('dependency_versions',{})))
    if not processes:raise ValueError('Service runtime drift: process identity unavailable')
    return dict(url=service['url'],schema=service.get('openapi',{}).get('body_sha256'),
                status=service.get('openapi',{}).get('http_status'),seed=service.get('seed_request_response'),
                processes=sorted(processes,key=fingerprint))

def verify_service_inventory(config,*,inspector=None):
    """Read-only live identity check at each formal run/resume, before model use."""
    if inspector is None:
        from capx.experiments.graph_preflight import inspect_service
        inspector=inspect_service
    previous=read(config['service_inventory']);current={}
    for name,url in config['services'].items():
        now=inspector(name,url,5)
        if service_identity(now)!=service_identity(previous['services'][name]):raise ValueError('Service runtime drift: '+name)
        current[name]=now
    return dict(checked_unix=time.time(),method='GET schema/health and local process source metadata; no inference',services=current)


def run_campaign(root,*,runner=None,max_units=None):
    if max_units is not None and (type(max_units) is not int or max_units<1):raise ValueError('max_units must be a positive integer')
    root=Path(root);frozen=load_campaign(root);config=frozen['config']
    with lock(root/'campaign.lock'):
        state=read(root/'state.json')
        validate_checkpoint(frozen,state)
        for unit in state['units']:
            if unit['state']=='started':
                receipt=root/'units'/unit['id']/'pair-result.json'
                if not receipt.exists():raise RuntimeError('uncertain started unit; inspect artifacts, never automatically repeat physical execution')
                body=read(receipt);_validate_result(body,unit);unit['state']='done';unit['result']=str(receipt)
                recovered=body.get('campaign_unit_seconds')
                if 'runtime_seconds' not in unit:
                    if isinstance(recovered,(int,float)) and math.isfinite(recovered) and recovered>=0:
                        state['active_seconds']+=recovered;unit['runtime_seconds']=recovered
                    else:unit['recovered_runtime_unknown']=True
        if all(u['state']=='done' for u in state['units']):
            state['stop_reason']='complete';atomic(root/'state.json',state);return status(root)
        if consume_stop_request(root,state):
            atomic(root/'state.json',state);return status(root)
        atomic(root/'state.json',state)
        for key,var in (('sam3','SAM3_SERVICE_URL'),('graspnet','GRASPNET_SERVICE_URL'),('pyroki','PYROKI_SERVICE_URL')):
            if key in config['services']:os.environ[var]=config['services'][key]
        from capx.experiments.graph_v2 import configure_services
        configure_services(config)
        if frozen['phase']=='formal':atomic(root/f'service-check-{time.time_ns()}.json',verify_service_inventory(config))
        budget=PersistentModelBudget(root,config['budgets'],elapsed_seconds=state['active_seconds'])
        snapshots=_snapshots(config,expected_hashes=frozen['source_hashes']);initial_refs=None;initial_validation_cache={}
        # Full source/archive verification happened once at entry (and every
        # resume). Keep business and selected-input checks at execution boundaries.
        runtime_hashes=source_inventory(config,include_run_inputs=False)
        if any(frozen['source_hashes'].get(p)!=h for p,h in runtime_hashes.items()):raise ValueError('Runtime source set differs from frozen campaign')
        frozen_path=(root/'frozen.json').resolve()
        frozen_digest=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        runtime_hashes[str(frozen_path)]=frozen_digest
        campaign_ref=dict(root=str(root.resolve()),signature=frozen['signature'],frozen_sha256=frozen_digest,
            _initial_validation_cache=initial_validation_cache,_runtime_hashes=runtime_hashes)
        if config.get('initial_manifest'):
            from capx.experiments.shared_initial import validate_initial_manifest
            initial_refs=validate_initial_manifest(config['initial_manifest'],config,state['units'],phase=frozen['phase'],plan=frozen['plan'],_cache=initial_validation_cache)
        if runner is None:
            from capx.experiments.graph_v2 import run_pair
            runner=run_pair
        completed=0;state['stop_reason']=None
        for unit in state['units']:
            if unit['state']=='done':continue
            if consume_stop_request(root,state):break
            if any(row['state'] in ('started','uncertain') for row in budget.rows):state['stop_reason']='uncertain_request_requires_audit';break
            if verify_source_hashes(runtime_hashes):state['stop_reason']='source_changed';break
            if max_units is not None and completed>=max_units:state['stop_reason']='invocation_unit_limit';break
            if time.monotonic()>=budget.deadline:state['stop_reason']='global_wall_budget';break
            if budget.count>=budget.max_requests:state['stop_reason']='global_request_budget';break
            if budget.ledger.spent['deepseek']>=config['budgets']['cost_limit_usd']:state['stop_reason']='global_cost_budget';break
            unit['state']='started';unit['started_unix']=time.time();atomic(root/'state.json',state)
            output=root/'units'/unit['id'];output.mkdir(parents=True,exist_ok=False)
            started=time.monotonic();budget.prefix=unit['id']+':';result=None
            try:
                extra={'campaign_ref':campaign_ref}
                if initial_refs is not None:extra['initial_ref']=initial_refs[(unit['task'],unit['seed'],unit['repeat'],unit['fault'])]
                result=runner(output=output,task=unit['task'],seed=unit['seed'],repeat=unit['repeat'],snapshot=snapshots.get((unit['task'],unit['seed'])),config=config,request_budget=budget,groups=tuple(unit['groups']),fault=unit['fault'],**extra)
                rows=_validate_result(result,unit)
                guards=dict(runtime_hashes)
                if initial_refs is not None:guards.update(extra['initial_ref']['files'])
                changed=verify_source_hashes(guards)
                atomic(output/'campaign-source-postcheck.json',dict(unchanged=not changed,changed=changed))
                if changed:
                    # Preserve each original physical reward and the executed
                    # attempt. Never retry; source-integrity failure is not success.
                    rows=[dict(row,status='source_changed',policy_success=False,
                        stop_reason='frozen_source_drift_after_unit',source_changes=changed,
                        pre_source_guard_status=row.get('status')) for row in rows]
                atomic(output/'pair-result.json',dict(episodes=rows,campaign_unit_seconds=time.monotonic()-started))
                unit.update(state='done',result=str((output/'pair-result.json').resolve()));completed+=1
                if changed:state['stop_reason']='source_changed';break
            except BaseException:
                state['stop_reason']='interrupted_unit_requires_audit';unit['interrupted_unix']=time.time()
                raise
            finally:
                elapsed=time.monotonic()-started;state['active_seconds']+=elapsed;unit['runtime_seconds']=elapsed;atomic(root/'state.json',state)
        if all(u['state']=='done' for u in state['units']) and state['stop_reason']!='source_changed':state['stop_reason']='complete'
        atomic(root/'state.json',state)
    return status(root)


def records(root):
    state=read(Path(root)/'state.json');rows=[]
    validate_checkpoint(load_campaign(root,check_sources=False),state)
    for unit in state['units']:
        if unit['state']=='done':rows.extend(_validate_result(read(unit['result']),unit))
    return rows


def status(root):
    root=Path(root);state=read(root/'state.json');ledger=read(root/'request-ledger.json') if (root/'request-ledger.json').exists() else {'requests':[]}
    rows=records(root)
    return dict(planned_units=len(state['units']),complete_units=sum(u['state']=='done' for u in state['units']),
                uncertain_units=sum(u['state']=='started' for u in state['units']),attempted_episodes=len(rows),planned_episodes=sum(len(u['groups']) for u in state['units']),
                stop_reason=state['stop_reason'],active_seconds=state['active_seconds'],requests=len(ledger['requests']),
                uncertain_requests=sum(r['state'] in ('started','uncertain') for r in ledger['requests']),
                pending_units=sum(u['state']=='pending' for u in state['units']),
                runtime_accounting_complete=not any(u.get('recovered_runtime_unknown') for u in state['units']),
                source_integrity_complete=not any(r.get('status')=='source_changed' or r.get('source_changes') or (r.get('source_postcheck') or {}).get('unchanged') is False for r in rows),
                estimated_cost_upper_usd=sum(r.get('cost_upper_usd',r.get('reserved_usd',0)) for r in ledger['requests']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare-development');p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--development-scenes',type=int,default=20);p.add_argument('--test-scenes',type=int,default=100);p.add_argument('--repeats',type=int,default=2);p.add_argument('--allocation-seed',type=int,default=20261120);p.add_argument('--exploratory',action='store_true',help='Allow smoke-only development without formal service provenance; cannot qualify a formal freeze')
    p=sub.add_parser('prepare-snapshots');p.add_argument('--config',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','test'],default='test');p.add_argument('--resume',action='store_true');p.add_argument('--max-scenes',type=int)
    p=sub.add_parser('export-initials');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('freeze');p.add_argument('--config',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--development-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name in ('run','resume'):
        p=sub.add_parser(name);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--max-units',type=int)
    p=sub.add_parser('request-stop');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--reason',default='operator requested boundary stop')
    p=sub.add_parser('status');p.add_argument('--campaign',type=Path,required=True)
    p=sub.add_parser('analyze');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000)
    p=sub.add_parser('contrast');p.add_argument('--normal',type=Path,required=True);p.add_argument('--ablation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=0)
    a=parser.parse_args(argv)
    if a.action=='prepare-development':
        c=read(a.config);plan=make_plan(c['tasks'],a.development_scenes,a.test_scenes,a.repeats,a.allocation_seed)
        if a.exploratory:
            import sys
            c=dict(c,exploratory_only=True);validation=None
            print('Exploratory-only campaign: cannot be used as formal development acceptance.',file=sys.stderr)
        else:validation=validate_development_services(c)
        freeze_campaign(a.output,c,plan,phase='development')
        if validation:atomic(a.output/'development-service-check.json',validation)
        out=status(a.output)
    elif a.action=='prepare-snapshots':out=prepare_snapshots(a.output,read(a.config),read(a.plan),split=a.split,resume=a.resume,max_scenes=a.max_scenes)
    elif a.action=='export-initials':
        from capx.experiments.shared_initial import export_initial_manifest
        out=export_initial_manifest(a.campaign,a.output)
    elif a.action=='freeze':freeze_campaign(a.output,read(a.config),read(a.plan),phase='formal',development_root=a.development_root);out=status(a.output)
    elif a.action in ('run','resume'):out=run_campaign(a.campaign,max_units=a.max_units)
    elif a.action=='status':out=status(a.campaign)
    elif a.action=='request-stop':out=request_stop(a.campaign,a.reason)
    elif a.action=='contrast':
        from capx.experiments.graph_contrast import paired_mechanism_contrast
        out=paired_mechanism_contrast(a.normal,a.ablation,n_bootstrap=a.bootstrap,random_seed=a.seed)
        with a.output.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
        out={'contrast':str(a.output)}
    else:
        from capx.experiments.graph_statistics import analyze
        frozen=load_campaign(a.campaign);report=status(a.campaign)
        out=analyze(records(a.campaign),n_bootstrap=a.bootstrap,expected_repeats=frozen['plan']['repeats']);out['campaign_status']=report
        complete=report['complete_units']==report['planned_units'] and not report['uncertain_requests'] and report.get('source_integrity_complete',True)
        out.update(campaign_complete=complete,campaign_phase=frozen['phase'],
            missing_planned_episodes=report['planned_episodes']-report['attempted_episodes'])
        for stratum in out['strata'].values():
            for name in ('G1_vs_G0','G1_vs_B0','G1_vs_G0_physical','G1_vs_B0_physical','G1_vs_B0_policy'):
                if name not in stratum:continue
                stratum[name]['formal_estimate_ready'] &= complete and frozen['phase']=='formal'
        if not complete or frozen['phase']!='formal':
            out['campaign_inference_warning']='Development or incomplete fixed allocation; descriptive results only, not completed formal evaluation'
        with a.output.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
        out={'analysis':str(a.output)}
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
