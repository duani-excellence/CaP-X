"""Local experiment control and artifact viewing over the unified CLI."""
import asyncio
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from fastapi import APIRouter,HTTPException,Query,Request
from fastapi.responses import FileResponse,Response
from capx.experiments.unified import ROOT,RunSpec,resolve,capabilities,atomic
from capx.experiments.artifacts import safe_path,redact,read_json,lines,graph_view,episode_summary,list_episodes,trace_path

router=APIRouter(prefix='/api/experiments')
ARTIFACTS=ROOT/'outputs';JOBS=ARTIFACTS/'inference-web';processes={};catalog_cache=(0,[])

def checked(path):
    try:return safe_path(ARTIFACTS,path)
    except ValueError as exc:raise HTTPException(400,str(exc))

def jobpath(ident):
    if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):raise HTTPException(400,'Invalid job ID')
    return JOBS/ident

@router.get('/capabilities')
def options():return capabilities()

@router.post('/validate')
def validate(spec:RunSpec):
    try:
        s,c,f=resolve(spec);return redact(dict(spec=s.model_dump(),runner=f,budgets=c['budgets'],services=c.get('services'),task_config=c.get('tasks')))
    except (ValueError,OSError) as exc:raise HTTPException(400,str(exc))

@router.post('/jobs')
def start(spec:RunSpec):
    from capx.web.session_manager import get_session_manager
    from capx.web.multitask import processes as other
    if any(p.poll() is None for p in list(processes.values())+list(other.values())) or any(str(s.get('state')) in ('running','loading_config','awaiting_user_input') for s in get_session_manager().list_sessions()):raise HTTPException(409,'Another Web trial is active')
    if spec.output is not None:raise HTTPException(400,'Web assigns a unique output directory; CLI supports --output')
    try:spec,_,_=resolve(spec)
    except (ValueError,OSError) as exc:raise HTTPException(400,str(exc))
    ident=uuid.uuid4().hex;root=jobpath(ident);root.mkdir(parents=True,exist_ok=False)
    spec=spec.model_copy(update={'output':str(root/'run')});atomic(root/'request.json',spec.model_dump())
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    with (root/'worker.log').open('x') as log:
        p=subprocess.Popen([str(ROOT/'scripts/local/infer.sh'),'replay' if spec.reference else 'run','--spec',str(root/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes[ident]=p
    return dict(id=ident,pid=p.pid,path=str((root/'run').relative_to(ARTIFACTS)))

@router.get('/jobs')
def jobs():
    result=[]
    if JOBS.exists():
        for root in sorted(JOBS.iterdir(),key=lambda x:x.stat().st_mtime,reverse=True)[:30]:
            if not (root/'request.json').exists():continue
            p=processes.get(root.name);status=read_json(root/'run/status.json',{})
            if p is not None and p.poll() is not None and status.get('state') in ('starting','running',None):status.update(state='error',reason='worker_exited_without_final_status',exit_code=p.returncode)
            if p is None and status.get('state') in ('starting','running'):status.update(state='unknown',reason='server_restarted_process_ownership_unknown')
            ledger=read_json(root/'run/request-ledger.json',{}).get('requests',[])
            status['requests']=len(ledger)
            status['cost_reserved_or_spent_usd']=sum(x.get('cost_upper_usd',x.get('reserved_usd',0)) for x in ledger)
            if ledger and ledger[-1].get('state') in ('started','uncertain'):
                status['live_phase']='model_request_'+ledger[-1]['state']
            elif status.get('current'):
                recent=list(lines(trace_path(root/'run'/status['current'])))
                if recent:
                    last=recent[-1];status['live_phase']=last.get('kind');status['live_event']={k:last.get(k) for k in ('kind','event','name','stage','error')}
            result.append(dict(id=root.name,running=p is not None and p.poll() is None,exit_code=p.poll() if p else None,status=status,spec=read_json(root/'request.json'),path=str((root/'run').relative_to(ARTIFACTS))))
    return redact(result)

@router.post('/jobs/{ident}/stop')
def stop(ident:str):
    root=jobpath(ident);p=processes.get(ident)
    if p is None or p.poll() is not None:raise HTTPException(409,'Worker not active or ownership unknown')
    target=root/'run/stop.request' if (root/'run').exists() else root/'stop.pending'
    target.touch()
    return dict(stopping=True,boundary='complete_method_pair')

@router.post('/jobs/{ident}/terminate')
async def terminate(ident:str):
    root=jobpath(ident);p=processes.get(ident)
    if p is None or p.poll() is not None:raise HTTPException(409,'Worker not active')
    os.killpg(p.pid,signal.SIGINT)
    for _ in range(30):
        if p.poll() is not None:break
        await asyncio.sleep(.1)
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGKILL)
        await asyncio.to_thread(p.wait,timeout=3)
    status=read_json(root/'run/status.json',{});status.update(state='stopped',reason='operator_terminated_incomplete_pair',exit_code=p.poll())
    atomic(root/'run/status.json',status)
    return dict(stopped=True,reason='incomplete pair; not a task success')

@router.get('/jobs/{ident}/log')
def log(ident:str):
    path=jobpath(ident)/'worker.log'
    if not path.exists():return {'text':''}
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-20000));text=f.read().decode(errors='replace')
    return redact({'text':text})

@router.get('/episodes')
def episodes():
    global catalog_cache
    if time.monotonic()-catalog_cache[0]>8:catalog_cache=(time.monotonic(),list_episodes(ARTIFACTS))
    return catalog_cache[1]

@router.get('/episode')
def episode(path:str):return redact(episode_summary(checked(path)))

@router.get('/graph')
def graph(path:str,at:int|None=Query(None,ge=0),graph_id:str|None=None):return redact(graph_view(checked(path),at,graph_id))

@router.get('/trace')
def trace(path:str,offset:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=200)):
    rows=list(lines(trace_path(checked(path))))
    return redact(dict(total=len(rows),offset=offset,rows=rows[offset:offset+limit],channel='offline evaluator diagnostics; never supplied to online agent'))

@router.get('/programs')
def programs(path:str):
    root=checked(path);files=list(root.glob('*program.json'))+list(root.glob('programs/*.json'))+[p for p in root.glob('block-*.json') if '-evaluation' not in p.name]
    if (root.parent/'initial.json').exists():files.insert(0,root.parent/'initial.json')
    return redact([dict(file=p.name,data=read_json(p)) for p in files])

@router.get('/force')
def force(path:str,limit:int=Query(20000,ge=1,le=20000)):
    """Operator-only sensor history; absent/disabled is never fabricated data."""
    from collections import deque
    root=checked(path);rows=deque(maxlen=limit);total=0
    for row in lines(root/'force.jsonl'):rows.append(row);total+=1
    events=[];stamp=None
    for e in lines(trace_path(root)):
        if e.get('sim_time') is not None:stamp=e['sim_time']
        if e.get('kind','').startswith(('force_','skill_attachment','skill_recovery')):
            sample=e.get('sample');sample_time=sample.get('timestamp',stamp) if isinstance(sample,dict) else stamp
            events.append(dict(e,timestamp=e.get('timestamp',sample_time)))
    result=read_json(root/'result.json',{});runtime=read_json(root/'runtime-spec.json',{})
    enabled=result.get('force_feedback',runtime.get('force_feedback'))
    return redact(dict(total=total,offset=total-len(rows),rows=list(rows),events=events[-300:],
        calibration=read_json(root/'force-calibration.json'),online_enabled=enabled,
        channel='operator diagnostic archive; Vision-only arms never receive these sensor packets'))

@router.get('/video')
def video(path:str,name:str):
    root=checked(path)
    try:file=safe_path(root,name)
    except ValueError as exc:raise HTTPException(400,str(exc))
    if file.parent!=root or file.suffix!='.mp4' or not file.exists():raise HTTPException(404,'No archived video')
    return FileResponse(file,media_type='video/mp4')

@router.get('/frame')
def frame(path:str):
    root=checked(path);files=sorted((root/'observations').glob('*.npz'))
    if not files:raise HTTPException(404,'No public RGB-D captures yet')
    import numpy as np
    from PIL import Image
    try:
        with np.load(files[-1],allow_pickle=False) as f:
            key=next((k for k in f.files if 'rgb' in k or 'image' in k),None)
            if key is None:raise ValueError('No RGB field')
            arr=f[key]
            if arr.dtype!=np.uint8:arr=np.clip(arr*(255 if arr.max()<=1 else 1),0,255).astype('uint8')
            stream=io.BytesIO();Image.fromarray(arr[:,:,:3]).save(stream,format='JPEG')
        return Response(stream.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'no-store','X-Capture-File':files[-1].name})
    except (ValueError,OSError,KeyError) as exc:raise HTTPException(404,str(exc))


@router.get('/events')
def events(path:str,offset:int=Query(0,ge=0),limit:int=Query(60,ge=1,le=200)):
    root=checked(path)/'harness';rows=list(lines(root/'events.jsonl'))
    latest=read_json(root/'latest.json',{})
    return redact(dict(available=(root/'events.jsonl').exists(),total=len(rows),offset=offset,
        rows=rows[offset:offset+limit],checkpoint=latest,physical_rollback=False))
