"""Serial, single-program paired pilot; existing P0 actor and physical reward only."""
import ast
import copy
import hashlib
import json
import re
import time
from pathlib import Path

MODELS = {'deepseek':'deepseek-v4-pro','openai':'gpt-6-astra'}
SEEDS = [0,21,1,2,3]

def dump(path, value):
    from capx.envs.reproducibility import json_value
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,default=json_value,indent=2,allow_nan=False)

def digest(value):
    from capx.envs.reproducibility import json_value
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=json_value).encode()).hexdigest()

def one_program(content):
    if not isinstance(content,str) or not content.strip():raise ValueError('Empty model content')
    if '```' in content:
        match=re.fullmatch(r'\s*```(?:python)?\s*\n(.*?)\n```\s*',content,re.S)
        if not match or '```' in match[1]:raise ValueError('Exactly one complete Python block required')
        content=match[1]
    ast.parse(content)
    return content.strip()

class CostLedger:
    def __init__(self,limit=5):
        self.limit=limit;self.spent={p:0.0 for p in MODELS}
    def reserve(self,provider,*,input_bytes,max_tokens):
        # Conservative byte/token upper estimate plus formatting overhead, peak rates, two attempts.
        ir,orr=(10.,50.) if provider=='openai' else (1.32,3.96)
        bound=((input_bytes+1024)*ir+max_tokens*orr)/1e6
        if self.spent[provider]+2*bound>self.limit:raise RuntimeError('Provider cost ceiling reached')
        return bound

class SingleProgram:
    def __init__(self,env):self.env=env;self.used=False
    def execute(self,content):
        if self.used:raise RuntimeError('Single-block budget exhausted')
        self.used=True
        return self.env.step(one_program(content))

def environment(mode,root):
    from capx.envs.tasks import get_config,get_exec_env
    source='franka_robosuite_pick_place_'+('vision_' if mode=='vision' else '')+'code_env'
    cfg=copy.deepcopy(get_config(source));cfg.artifact_root=str(root);cfg.enable_render=True;cfg.execution_timeout=90.
    return get_exec_env(source)(cfg)

def check_services(root):
    import urllib.request,subprocess
    rows={}
    for port in [8114,8116,8215]:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/openapi.json',timeout=5) as f:doc=json.load(f)
            if port==8215:
                assert all('seed' in doc['components']['schemas'][k]['properties'] for k in ['PlanRequest','PlanResponse'])
            rows[str(port)]={'ok':True}
        except Exception as exc:rows[str(port)]={'ok':False,'error':type(exc).__name__}
    dump(root/'gpu_logs'/f'services-{time.time_ns()}.json',rows)
    p=root/'gpu_logs'/f'gpu-{time.time_ns()}.txt';p.write_bytes(subprocess.check_output(['nvidia-smi']))
    return rows

def load_scene(manifest,entry):
    p=(manifest.parent/entry['snapshot']).resolve();snapshot=json.loads(p.read_text())
    if digest(snapshot)!=entry['snapshot_digest'] or snapshot['state_hash']!=entry['state_hash']:
        raise ValueError('Manifest Snapshot integrity mismatch')
    return snapshot

def prepare(manifest,root):
    from capx.envs.physical_evaluation import physical_state
    env=environment('privileged',root/'traces'/'scene-preparation')
    rows=[]
    try:
        for seed in SEEDS:
            env.reset(seed=seed);low=env.low_level_env
            del low._physics_observer
            try:
                audit=low.settle_initial();snapshot=low.capture_snapshot();state=physical_state(low)
                target=manifest.parent/'snapshots'/f'seed-{seed}.json'
                dump(target,snapshot)
                record=dict(seed=seed,snapshot=str(target.relative_to(manifest.parent)),snapshot_digest=digest(snapshot),state_hash=snapshot['state_hash'],model_hash=snapshot['model_hash'],configuration_hash=digest({'low_level':env.cfg.low_level,'settle_max':5,'stable_seconds':.5,'linear':.02,'angular':.2}),api_version='P0 a7a0931; settled snapshot v2',initial_state=state,joints=low.robosuite_env.sim.data.qpos.copy(),settle_audit=audit,valid=True)
                # Verify replay immediately, before certifying this scene.
                env.reset(seed=seed,options={'snapshot':snapshot})
                assert low.physical_state_hash()==snapshot['state_hash']
                rows.append(record)
            except Exception as exc:
                dump(root/'manifest'/f'rejected-seed-{seed}.json',{'seed':seed,'error':type(exc).__name__+': '+str(exc)})
                raise
        assert len(rows)==5 and len({r['state_hash'] for r in rows})==5
        assert len({digest(r['initial_state']['objects']) for r in rows})==5
        dump(manifest,{'version':1,'scenes':rows})
    finally:env.close()

def trial_prompt(env,obs):
    public={k:obs[k] for k in ('robot_joint_pos','robot_cartesian_pos') if k in obs}
    prompt=copy.deepcopy(env._full_prompt)
    prompt.append({'role':'user','content':'Single-round experiment: return exactly one complete Python program. No recovery requests. At most 64 API calls and 90 seconds execution. No files, network, private attributes, or hidden state. Public robot observation: '+json.dumps(public,default=lambda v:v.tolist())})
    return prompt,public

def trace_analysis(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    states=[r['state'] for r in rows if r['kind']=='physics'];apis=[r for r in rows if r['kind']=='api']
    # Physical diagnostics do not alter the reward definition.
    start=apis[0]['before']['objects']['primary']['position'][2] if apis else None
    lift=max((r['objects']['primary']['position'][2]-start for r in states),default=0) if start is not None else 0
    releases=[r for r in apis if r['name']=='open_gripper']
    moves=[]
    for r in apis:
        if r['name']=='goto_pose':
            moves.append({'requested_base_tcp':r['args'][0] if r['args'] else r['kwargs'].get('position'),'actual_world_tcp':r['after']['tcp'],'actual_world_ee':r['after']['ee'],'contacts':r['after']['contacts'],'relative_poses':r['after']['relative_poses'],'error':r['error']})
    return dict(max_lift_m=lift,lift_success=lift>.05,grasp_receipts=[r for r in apis if r['name'] in ('get_object_pose','sample_grasp_pose')],moves=moves,releases=[{'before':r['before'],'after':r['after']} for r in releases],notes='Requested TCP is base frame; actual FK is world. Contacts include expected table/gripper contacts, not automatically abnormal collisions.')

def run_trial(root,manifest,entry,mode,provider,ledger):
    from capx.envs.physical_evaluation import physical_state
    from capx.envs.tasks.franka.franka_pick_place import ORACLE_CODE
    from capx.llm.providers import provider_config
    from capx.llm.client import ModelQueryArgs,query_model
    from capx.utils.video_utils import _write_video
    import numpy as np
    trial_id=f'{provider}-{mode}-seed-{entry["seed"]}'
    dest=root/provider/mode/f'seed-{entry["seed"]}';dest.mkdir(parents=True,exist_ok=False)
    result=dict(trial_id=trial_id,provider=provider,mode=mode,seed=entry['seed'],status='system_error',reward=None,cost_usd=0.,api_usage=None,generation_seconds=0.,execution_seconds=0.)
    env=None;start=time.monotonic()
    try:
        env=environment(mode,root/'traces'/trial_id)
        snapshot=load_scene(manifest,entry)
        obs,reset=env.reset(seed=entry['seed'],options={'snapshot':snapshot})
        state=physical_state(env.low_level_env)
        if reset['initial_state_hash']!=entry['state_hash']:raise ValueError('Snapshot hash mismatch')
        if not all(o['speed']<.02 and o['angular_speed']<.2 for o in state['objects'].values()):raise ValueError('Initial scene not stationary')
        result.update(snapshot_hash=reset['initial_state_hash'],trace_path=str(env._trace_path))
        prompt,public=trial_prompt(env,obs);docs=[api.combined_doc() for api in env._apis.values()]
        result.update(prompt_hash=digest(prompt),api_documentation_hash=digest(docs),actor_observation_hash=digest(public))
        dump(dest/'prompt.json',prompt);dump(dest/'api-docs.json',docs);dump(dest/'actor-observation.json',public)
        # Canonical hashes are created by Oracle; models must match their mode/scene exactly.
        canonical=root/'manifest'/f'{mode}-seed-{entry["seed"]}-hashes.json'
        hashes={k:result[k] for k in ['prompt_hash','api_documentation_hash','actor_observation_hash','snapshot_hash']}
        if provider=='oracle':dump(canonical,hashes)
        elif json.loads(canonical.read_text())!=hashes:raise ValueError('Paired input hash mismatch')
        env.enable_video_capture(True,clear=True)
        env.low_level_env._record_frame()
        if provider=='oracle':code=ORACLE_CODE
        else:
            url,model,key=provider_config(provider)
            if model!=MODELS[provider]:raise ValueError('Configured model mismatch; substitution prohibited')
            bound=ledger.reserve(provider,input_bytes=len(json.dumps(prompt).encode()),max_tokens=4096)
            def observer(attempt,payload,response,error):
                body={}
                if response is not None:
                    try:body=response.json()
                    except ValueError:pass
                usage=body.get('usage',{});it=usage.get('prompt_tokens');ot=usage.get('completion_tokens')
                # Conservative peak rate for the ceiling; actual estimate uses off-peak where applicable.
                ir,orr=(10.,50.) if provider=='openai' else (1.32,3.96)
                charge=(it*ir+ot*orr)/1e6 if it is not None and ot is not None else bound
                ledger.spent[provider]+=charge;result['cost_usd']+=charge
                headers=getattr(response,'headers',{})
                dump(dest/f'http-attempt-{attempt}.json',dict(trial_id=trial_id,provider=provider,attempt=attempt,payload=payload,http_status=getattr(response,'status_code',None),request_id=headers.get('x-request-id') or headers.get('x-ds-request-id'),response=body,error=error,accounting_charge_usd=charge,charge_is_upper_bound=it is None or ot is None))
                if ledger.spent[provider]>=ledger.limit:raise RuntimeError('Provider cost ceiling reached')
            args=ModelQueryArgs(model=model,server_url=url,api_key=key,max_tokens=4096,max_attempts=2,request_timeout=200,response_observer=observer)
            from capx.execution.sandbox import callback_deadline
            t=time.monotonic()
            with callback_deadline(t+200):answer=query_model(args,prompt)
            result['generation_seconds']=time.monotonic()-t
            dump(dest/'model-response.json',answer)
            result['api_usage']=answer.get('usage');result['model']=answer['response_model'];result['attempts']=answer['attempts']
            if answer['response_model']!=model:raise ValueError('Response model differs from request')
            code=answer['content']
        (dest/'generated.py').write_text(code if isinstance(code,str) else '')
        try:
            t=time.monotonic();_,reward,_,truncated,info=SingleProgram(env).execute(code)
            result.update(reward=reward,status='success' if reward else 'task_failure',execution_seconds=time.monotonic()-t,info=info,truncated=truncated)
            if info['execution_status'] in ('system_error','timeout'):result['status']=info['execution_status']
            if 'budget exceeded' in info['stderr']:result['status']='budget_exceeded'
            result['analysis']=trace_analysis(env._trace_path)
        except (ValueError,SyntaxError) as exc:
            result.update(status='invalid_program',reward=0.,error=type(exc).__name__+': '+str(exc))
        low=env.low_level_env
        if hasattr(low,'grasp_sample'):
            np.savez_compressed(dest/'grasp-candidates.npz',candidates=low.grasp_sample,scores=low.grasp_scores,contact_points=low.grasp_contact_pts)
    except Exception as exc:
        import requests
        result['error']=type(exc).__name__+': '+str(exc)
        if isinstance(exc,requests.HTTPError):result['status']='api_error';result['http_status']=getattr(exc.response,'status_code',None)
    finally:
        if env is not None:
            try:
                frames=env.get_video_frames()
                if frames:_write_video(frames,str(root/'videos'),suffix=trial_id,timestamps=list(env.low_level_env._frame_timestamps))
                result['video']=str(root/'videos'/f'video_{trial_id}.mp4') if frames else None
            except Exception as exc:result['status']='system_error';result['video_error']=type(exc).__name__+': '+str(exc)
            env.close()
        result['total_seconds']=time.monotonic()-start
        dump(dest/'result.json',result)
        dump(root/'requests'/f'{trial_id}-ledger.json',ledger.spent)
    print(trial_id,result['status'],result['reward'],flush=True)
    return result

FATAL={'system_error','timeout','budget_exceeded','api_error'}

def run(manifest_path,phase,output_dir):
    import os,fcntl
    os.environ['GRASPNET_SERVICE_URL']='http://127.0.0.1:8215'
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.serial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=Path(manifest_path).resolve()
    services=check_services(root)
    if not services['8116']['ok']:raise RuntimeError('IK service unavailable')
    if phase=='prepare':prepare(manifest,root);return
    entries=json.loads(manifest.read_text())['scenes']
    if len(entries)!=5 or len({e['state_hash'] for e in entries})!=5:raise ValueError('Five distinct scenes required')
    ledger=CostLedger();results=[]
    if phase=='oracle':
        for mode in ['privileged','vision']:
            if mode=='vision' and not all(r['ok'] for r in services.values()):continue
            for entry in entries:
                r=run_trial(root,manifest,entry,mode,'oracle',ledger);results.append(r)
                if r['status'] in FATAL:break
        counts={m:sum(r['reward']==1 for r in results if r['mode']==m) for m in ['privileged','vision']}
        dump(root/'manifest'/'gates.json',dict(privileged=counts['privileged']==5,vision=counts['privileged']==5 and counts['vision']>=4 and not any(r['status'] in FATAL for r in results if r['mode']=='vision'),counts=counts,results=[{k:r[k] for k in ['trial_id','status','reward']} for r in results]))
    elif phase=='models':
        gates=json.loads((root/'manifest'/'gates.json').read_text())
        if not gates['privileged']:raise RuntimeError('Gate A failed; no model requests allowed')
        stopped=set()
        for mode in ['privileged','vision']:
            if mode=='vision' and (not gates['vision'] or not all(r['ok'] for r in services.values())):continue
            for index,entry in enumerate(entries):
                for provider in (['deepseek','openai'] if index%2==0 else ['openai','deepseek']):
                    if provider in stopped:continue
                    r=run_trial(root,manifest,entry,mode,provider,ledger);results.append(r)
                    if r['status'] in FATAL:stopped.add(provider)
        dump(root/'manifest'/'model-cost-ledger.json',ledger.spent)
    else:raise ValueError('Unknown phase')
