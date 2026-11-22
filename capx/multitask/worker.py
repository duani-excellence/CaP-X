"""Dedicated main-thread workers. No worker starts a model/service automatically."""
import argparse,hashlib,importlib,json,time
from pathlib import Path
from capx.multitask.runtime import Runtime
from capx.multitask.catalog import TASKS
from capx.training.teacher import dump,read,successful,MODEL
from capx.experiments.pilot import digest,CostLedger,one_program


def oracle(task):
    module=importlib.import_module('capx.envs.tasks.franka.'+TASKS[task]['module'])
    if task=='handover':return module.PRIVILEGED_ORACLE_CODE
    if task=='nut_assembly':
        from .oracles import NUT
        return NUT
    if task=='wipe':
        from .oracles import WIPE
        return WIPE
    code=module.ORACLE_CODE
    if task=='two_arm_lift':
        import ast
        tree=ast.parse(code)
        branch=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp))
        return 'import numpy as np\n'+ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
    if task=='restack':
        code=code.replace('green_ext[2] + red_ext[2]','green_ext[2]/2 + red_ext[2]/2')
    return code


def prepare(task,manifest,root,seeds):
    rows=[]
    for seed in seeds:
        runtime=Runtime(task,seed,Path(root)/f'prepare-{seed}')
        try:
            snapshot=runtime.initial;path=Path(manifest).parent/'snapshots'/f'seed-{seed}.json';dump(path,snapshot)
            rows.append(dict(seed=seed,task=task,snapshot=str(path.relative_to(Path(manifest).parent)),state_hash=snapshot['state_hash'],snapshot_digest=digest(snapshot)))
        finally:runtime.close()
    dump(manifest,dict(version=1,task=task,scenes=rows))


def trial(task,manifest,entry,root,provider,max_tokens=32768,cost_limit=1.,code=None):
    root=Path(root);dest=root/provider/'privileged'/f"seed-{entry['seed']}";dest.mkdir(parents=True,exist_ok=False)
    runtime=None;started=time.monotonic()
    result=dict(task=task,seed=entry['seed'],trial_id=f"{task}-{provider}-{entry['seed']}",mode='privileged',provider=provider,status='system_error',reward=None,cost_usd=0.)
    try:
        snapshot=read(Path(manifest).parent/entry['snapshot'])
        if snapshot['task']!=task or digest(snapshot)!=entry['snapshot_digest']:raise ValueError('Snapshot task/digest mismatch')
        runtime=Runtime(task,entry['seed'],dest/'runtime')
        if runtime.initial!=snapshot:raise ValueError('Snapshot replay mismatch: physics/RNG/model/controller history')
        docs=[f'{name}: {fn.__doc__ or ""}' for name,fn in runtime.functions.items()]
        prompt=[{'role':'system','content':'Generate one complete Python robot program. Only numpy imports and listed APIs. No files, network, env, obs, globals or private attributes. At most 64 API calls; 90 seconds. No FINISH text.'},
                {'role':'user','content':runtime.task_module.PROMPT+'\nAPI documentation:\n'+'\n'.join(docs)}]
        result.update(snapshot_hash=snapshot['state_hash'],prompt_hash=digest(prompt),trace_path=str(runtime.trace),api_documentation_hash=digest(docs))
        dump(dest/'prompt.json',prompt);dump(dest/'api-docs.json',docs)
        if provider=='oracle':code=oracle(task)
        elif provider=='deepseek':
            from capx.llm.providers import provider_config
            from capx.llm.client import ModelQueryArgs,query_model
            url,model,key=provider_config('deepseek')
            if model!=MODEL:raise ValueError('Wrong Teacher model')
            bound=CostLedger(cost_limit).reserve('deepseek',input_bytes=len(json.dumps(prompt).encode()),max_tokens=max_tokens)
            def observer(attempt,payload,response,error):
                body={}
                if response is not None:
                    try:body=response.json()
                    except ValueError:pass
                u=body.get('usage',{});it=u.get('prompt_tokens');ot=u.get('completion_tokens')
                charge=(it*1.32+ot*3.96)/1e6 if it is not None and ot is not None else bound
                result['cost_usd']+=charge
                dump(dest/f'http-attempt-{attempt}.json',dict(http_status=getattr(response,'status_code',None),response=body,error=error,accounting_charge_usd=charge,charge_is_upper_bound=it is None or ot is None))
            answer=query_model(ModelQueryArgs(model=model,server_url=url,api_key=key,max_tokens=max_tokens,max_attempts=1,request_timeout=200,response_observer=observer),prompt)
            dump(dest/'model-response.json',answer);code=answer['content']
        elif provider!='replay':raise ValueError('Unsupported provider')
        (dest/'generated.py').write_text(code if isinstance(code,str) else '')
        try:code=one_program(code)
        except (ValueError,SyntaxError) as exc:
            result.update(status='invalid_program',reward=0.,error=str(exc))
            return result,dest
        runtime.low.enable_video_capture(True,clear=True)
        result.update(runtime.execute(code))
        from capx.utils.video_utils import _write_video
        frames=runtime.low.get_video_frames()
        if frames:
            timestamps=getattr(runtime.low,'_frame_timestamps',None)
            if timestamps is None or len(timestamps)!=len(frames):raise ValueError('Video timestamps unavailable; not accepted')
            _write_video(frames,str(dest/'videos'),suffix='trial',timestamps=timestamps)
            result['video']=str(dest/'videos/video_trial.mp4')
    except Exception as exc:
        result.update(status='system_error',reward=None,error=type(exc).__name__+': '+str(exc))
    finally:
        if runtime:runtime.close()
        result['total_seconds']=time.monotonic()-started;dump(dest/'result.json',result)
    return result,dest


def accept(task,root,seed):
    root=Path(root);manifest=root/'manifest/scenes.json'
    prepare(task,manifest,root/'preparation',[seed]);entry=read(manifest)['scenes'][0]
    # Guard failures must yield zero independent of FINISH claims; no hidden-state access.
    checks={}
    for label,code in [('pass','pass'),('empty',''),('syntax','if :'),('hidden','print(env)'),('finish','print("FINISH")'),('timeout','while True: pass')]:
        runtime=Runtime(task,seed,root/f'guard-{label}')
        try:
            if runtime.initial!=read(manifest.parent/entry['snapshot']):raise ValueError('Snapshot not deterministic')
            r=runtime.execute(code,timeout=.1 if label=='timeout' else 90.);checks[label]=r['reward']==0 and not r['info']['physical_success']
        finally:runtime.close()
    result,dest=trial(task,manifest,entry,root/'oracle','oracle')
    dump(root/'acceptance.json',dict(task=task,seed=seed,guards=checks,snapshot_replay=True,result=str(dest/'result.json'),status='ready' if all(checks.values()) and successful(result) else 'unverified',oracle_status=result['status'],error=result.get('error'),stderr=result.get('info',{}).get('stderr')))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['accept','episode']);p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--root',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--job')
    a=p.parse_args()
    if a.action=='accept':accept(a.task,a.root,a.seed)
    else:
        job=read(a.job);entry=read(job['manifest'])['scenes'][0]
        control,_=trial(a.task,job['manifest'],entry,Path(a.root)/'control','oracle')
        records=[]
        if successful(control):
            r,dest=trial(a.task,job['manifest'],entry,Path(a.root)/'teacher','replay' if job.get('mock') else 'deepseek',job['max_tokens'],job['cost_limit'],code=oracle(a.task) if job.get('mock') else None)
            records=[dict(task=a.task,id=f"{a.task}-seed-{entry['seed']}",mode='privileged',source='mock' if job.get('mock') else 'teacher',artifacts=str(dest.resolve()),snapshot_hash=r.get('snapshot_hash'),seed=entry['seed'],status=r['status'])]
        dump(Path(a.root)/'worker-records.json',records)
        dump(Path(a.root)/'control-result.json',control)
if __name__=='__main__':main()
