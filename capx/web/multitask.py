"""Bounded local multi-task jobs; subprocesses use the same CLI as terminal users."""
import json,os,signal,subprocess,sys,uuid
from pathlib import Path
from typing import Literal
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from capx.multitask.catalog import ROOT,TASKS,catalog,require_ready
router=APIRouter(prefix='/api/multitask')
JOBS=ROOT/'outputs/multitask-web'
processes={}
class JobRequest(BaseModel):
    targets:dict[str,int]
    action:Literal['oracle','infer','collect']='oracle'
    max_requests_per_task:int=Field(default=10,ge=1,le=10000)
    max_cost_usd_per_task:float=Field(default=2.,gt=0,le=10000)
    max_hours:float=Field(default=1.,gt=0,le=72)
    resume_id:str|None=None

def checked_id(value):
    if len(value)!=32 or any(c not in '0123456789abcdef' for c in value):raise HTTPException(400,'Invalid job id')
    return JOBS/value

@router.get('/tasks')
def tasks():return catalog()

@router.post('/jobs')
def start(request:JobRequest):
    if not request.targets or any(t not in TASKS or n<1 or n>10000 for t,n in request.targets.items()):raise HTTPException(400,'请选择有效任务及目标数量')
    for task in request.targets:
        try:require_ready(task)
        except ValueError as exc:raise HTTPException(400,str(exc))
    if any(p.poll() is None for p in processes.values()):raise HTTPException(409,'已有多任务进程运行，请等待或停止')
    from capx.web.session_manager import get_session_manager
    if any(str(r.get('state')) in ('running','awaiting_user_input','loading_config') for r in get_session_manager().list_sessions()):raise HTTPException(409,'请先停止当前交互推理')
    ident=request.resume_id or uuid.uuid4().hex;root=checked_id(ident)
    if request.resume_id:
        if request.action!='collect' or not (root/'request.json').exists():raise HTTPException(400,'仅采集作业可以续采')
        old=json.loads((root/'request.json').read_text());new=request.model_dump();new['resume_id']=None
        old.pop('max_hours',None);new.pop('max_hours',None)
        if old!=new:raise HTTPException(400,'续采配置必须与原作业一致')
    else:
        root.mkdir(parents=True,exist_ok=False);(root/'request.json').write_text(request.model_dump_json())
    command=[sys.executable,'-B','-m','capx.multitask.cli']
    if request.action=='oracle':command+=['oracle','--tasks',*request.targets,'--output',str(root/'run')]
    else:
        targets={t:1 for t in request.targets} if request.action=='infer' else request.targets
        command+=['collect','--targets',*[f'{t}={n}' for t,n in targets.items()],'--output',str(root/'run'),'--max-requests-per-task',str(1 if request.action=='infer' else request.max_requests_per_task),'--max-cost-usd-per-task',str(request.max_cost_usd_per_task),'--max-hours',str(request.max_hours)]
        if request.resume_id:command+=['--resume']
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',NUMBA_CACHE_DIR=str(root/'numba-cache'))
    with (root/f'launch-{uuid.uuid4().hex}.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes[ident]=process
    return {'id':ident,'output':str(root/'run')}

@router.get('/jobs')
def jobs():
    result=[]
    if not JOBS.exists():return result
    for root in sorted(JOBS.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
        if not (root/'request.json').exists():continue
        request=json.loads((root/'request.json').read_text());stats={}
        for task in request['targets']:
            path=root/'run'/task/'state.json';accept=root/'run'/task/'acceptance.json'
            try:
                if path.exists():
                    s=json.loads(path.read_text());stats[task]={k:s.get(k) for k in ('stats','stop_reason','request_slots_used','http_attempts','accounting_cost_usd')}
                elif accept.exists():stats[task]=json.loads(accept.read_text())
            except (OSError,ValueError):pass
        p=processes.get(root.name)
        result.append(dict(id=root.name,request=request,running=p is not None and p.poll() is None,exit=p.poll() if p else None,stats=stats,output=str(root/'run')))
    return result

@router.post('/jobs/{ident}/stop')
def stop(ident:str):
    checked_id(ident);p=processes.get(ident)
    if p is None or p.poll() is not None:raise HTTPException(409,'作业未运行')
    os.killpg(p.pid,signal.SIGINT)
    return {'stopping':True}
