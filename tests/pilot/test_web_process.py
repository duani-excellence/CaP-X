"""Real spawned-process tests without model/network calls."""
import asyncio
import json
import os
import time
import pytest
from types import SimpleNamespace
from pathlib import Path
from capx.web.session_manager import Session


def fake_worker(payload, events, commands):
 events.put(json.dumps({'kind':'event','event':{'type':'state_update','session_id':payload['session_id'],'state':'running'}}))
 events.put(json.dumps({'kind':'event','event':{'type':'code_execution_start','session_id':payload['session_id'],'block_index':0,'code':'pass'}}))
 Path(payload['config']['pid_path']).write_text(str(os.getpid()))
 while True:time.sleep(.1)


def test_watchdog_terminates_worker_and_allows_fresh_process(tmp_path):
 from capx.web.process_runner import run_trial_process
 async def run():
  pids=[]
  for i in range(2):
   s=Session(session_id=str(i),config={'pid_path':str(tmp_path/f'pid{i}')},execution_timeout=1)
   await run_trial_process(s,SimpleNamespace(),_target=fake_worker)
   pid=int((tmp_path/f'pid{i}').read_text());pids.append(pid)
   assert not Path(f'/proc/{pid}').exists()
   assert s.state.value=='error'
   assert any('timeout' in e.lower() for e in s.event_history)
  assert len(set(pids))==2
 asyncio.run(run())


def test_cancel_terminates_worker(tmp_path):
 from capx.web.process_runner import run_trial_process
 async def run():
  s=Session(session_id='cancel',config={'pid_path':str(tmp_path/'pid')})
  task=asyncio.create_task(run_trial_process(s,SimpleNamespace(),_target=fake_worker))
  for _ in range(200):
   if (tmp_path/'pid').exists():break
   await asyncio.sleep(.05)
  assert (tmp_path/'pid').exists()
  pid=int((tmp_path/'pid').read_text());task.cancel()
  try:await task
  except asyncio.CancelledError:pass
  assert not Path(f'/proc/{pid}').exists()
 asyncio.run(run())


def test_finish_text_cannot_mark_success():
 from capx.web.process_runner import physical_task_success
 assert not physical_task_success({'physical_success':False,'sandbox_rc':0},0,False)
 assert not physical_task_success({'physical_success':True,'sandbox_rc':1},1,False)
 assert physical_task_success({'physical_success':True,'sandbox_rc':0},1,False)


def oracle_worker(payload, events, commands):
 from capx.web import async_trial_runner as runner
 from capx.web.process_runner import _worker
 from capx.envs.tasks.franka.franka_pick_place import ORACLE_CODE
 from capx.envs.tasks.audited_stack import AuditedStackEnv
 original=AuditedStackEnv.reset
 def seeded(self,**kwargs):
  kwargs['seed']=0
  return original(self,**kwargs)
 AuditedStackEnv.reset=seeded
 def stream(*args,**kwargs):
  yield {'type':'done','content':payload['config'].get('_test_program',ORACLE_CODE)}
 runner._query_model_streaming=stream
 _worker(payload,events,commands)


@pytest.mark.parametrize("program,expected", [(None,True),("print(env)",False),("pass",False)])
def test_real_web_worker_oracle(tmp_path,program,expected):
 import pytest
 if not os.getenv('CAPX_WEB_INTEGRATION'):
  pytest.skip('requires existing local vision and IK services')
 from capx.envs.launch import LaunchArgs
 from capx.utils.launch_utils import _load_config
 from capx.web.async_trial_runner import LaunchArgsCompat
 from capx.web.process_runner import run_trial_process
 async def run():
  factory,config,_=_load_config(LaunchArgs(config_path='env_configs/cube_stack/franka_robosuite_cube_stack_web.yaml'))
  config['output_dir']=str(tmp_path/'artifacts')
  if program is not None:config['_test_program']=program
  s=Session(session_id='oracle-integration',env_factory=factory,config=config)
  args=LaunchArgsCompat('mock','http://unused.invalid',None,32768,.2,'medium',False,None,None,None)
  task=asyncio.create_task(run_trial_process(s,args,_target=oracle_worker))
  try:
   for _ in range(2400):
    if s.state.value in ('complete','error'):break
    await asyncio.sleep(.1)
   rows=[json.loads(x) for x in s.event_history]
   (tmp_path/'events.json').write_text(json.dumps(rows,indent=2))
   assert s.state.value=='complete',rows[-3:]
   executions=[r for r in rows if r['type']=='code_execution_result']
   assert executions,rows[-3:]
   if program is None:assert executions[-1]['success'],executions
   if program == 'print(env)':assert not executions[-1]['success'],executions
   completion=next(r for r in rows if r['type']=='trial_complete')
   assert completion['success'] is expected,completion
   assert completion['total_reward']==float(expected),completion
   assert 'main thread' not in executions[-1]['stderr']
   assert list((tmp_path/'artifacts').rglob('*.mp4'))
   assert list((tmp_path/'artifacts').rglob('*.timestamps.json'))
  finally:
   task.cancel()
   try:await task
   except asyncio.CancelledError:pass
 asyncio.run(run())
