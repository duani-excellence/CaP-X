import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
from capx.envs.tasks import get_config,get_exec_env
from capx.envs.physical_evaluation import physical_state

def test_settled_snapshot_restores_exact_state(tmp_path):
 cfg=copy.deepcopy(get_config('franka_robosuite_pick_place_code_env'));cfg.enable_render=False;cfg.artifact_root=str(tmp_path)
 env=get_exec_env('franka_robosuite_pick_place_code_env')(cfg)
 try:
  env.reset(seed=0);low=env.low_level_env
  low.settle_initial(max_seconds=5.,stable_seconds=.5)
  snapshot=low.capture_snapshot()
  assert snapshot['version']==2 and snapshot['settle_steps']>0
  assert all(o['speed']<.02 and o['angular_speed']<.2 for o in physical_state(low)['objects'].values())
  _,info=env.reset(seed=0,options={'snapshot':snapshot})
  assert info['initial_state_hash']==snapshot['state_hash']
  assert env.criteria.initial_z==physical_state(low)['objects']['primary']['position'][2]
  assert json.loads(env._trace_path.read_text().splitlines()[0])['snapshot']['version']==2
  bad=copy.deepcopy(snapshot);bad['state_hash']='bad'
  with pytest.raises(ValueError):env.reset(seed=0,options={'snapshot':bad})
 finally:env.close()

def test_settle_timeout_does_not_create_valid_snapshot(tmp_path):
 cfg=copy.deepcopy(get_config('franka_robosuite_pick_place_code_env'));cfg.enable_render=False;cfg.artifact_root=str(tmp_path)
 env=get_exec_env('franka_robosuite_pick_place_code_env')(cfg)
 try:
  env.reset(seed=3)
  with pytest.raises(TimeoutError):env.low_level_env.settle_initial(max_seconds=.1,stable_seconds=.5)
  with pytest.raises(ValueError):env.low_level_env.capture_snapshot()
 finally:env.close()

@pytest.mark.parametrize('status,expected',[(401,1),(403,1),(404,1),(429,2),(503,2)])
def test_http_attempt_budget(monkeypatch,status,expected):
 import capx.llm.client as c
 import requests
 calls=[]
 class Response:
  status_code=status;headers={}
  def raise_for_status(self):raise requests.HTTPError('test')
  def json(self):return {'error':{'code':'test'}}
 monkeypatch.setattr(c,'audited_post',lambda *a,**k:calls.append(k) or Response())
 monkeypatch.setattr(c.time,'sleep',lambda _:None)
 args=c.ModelQueryArgs(model='deepseek-v4-pro',server_url='https://api.deepseek.com/v1/chat/completions',max_attempts=2)
 with pytest.raises(requests.HTTPError):c.query_model(args,[{'role':'user','content':'test'}])
 assert len(calls)==expected
 assert all(x['timeout']==200 for x in calls)

def test_single_block_and_fee_guard():
 from capx.experiments.pilot import one_program,CostLedger
 assert one_program('pass')=='pass'
 assert one_program('```python\npass\n```')=='pass'
 with pytest.raises(ValueError):one_program('```python\npass\n```\n```python\npass\n```')
 ledger=CostLedger(limit=5)
 ledger.spent['openai']=4.99
 with pytest.raises(RuntimeError):ledger.reserve('openai',input_bytes=10000,max_tokens=4096)

def test_single_program_cannot_execute_twice():
 from capx.experiments.pilot import SingleProgram
 calls=[];runner=SingleProgram(SimpleNamespace(step=lambda code:calls.append(code)))
 runner.execute('pass')
 with pytest.raises(RuntimeError):runner.execute('pass')
 assert calls==['pass']

def test_existing_sandbox_enforces_64_api_calls():
 from capx.execution.sandbox import SandboxProcess
 calls=[];actor=SandboxProcess({'move':lambda:calls.append(1)})
 try:
  result=actor.run('for i in range(65):\n    move()')
  assert result['status']=='policy_violation' and len(calls)==64
 finally:actor.close()

def test_manifest_tampering_rejected(tmp_path):
 from capx.experiments.pilot import load_scene,digest
 snap={'state_hash':'initial'};(tmp_path/'s.json').write_text(json.dumps(snap))
 entry={'snapshot':'s.json','state_hash':'initial','snapshot_digest':digest(snap)}
 assert load_scene(tmp_path/'scenes.json',entry)==snap
 (tmp_path/'s.json').write_text(json.dumps({'state_hash':'different'}))
 with pytest.raises(ValueError):load_scene(tmp_path/'scenes.json',entry)

def test_failed_gate_prevents_model_trials(tmp_path,monkeypatch):
 import capx.experiments.pilot as p
 (tmp_path/'manifest').mkdir();manifest=tmp_path/'manifest/scenes.json'
 manifest.write_text(json.dumps({'scenes':[{'state_hash':str(i)} for i in range(5)]}))
 (tmp_path/'manifest/gates.json').write_text(json.dumps({'privileged':False,'vision':False}))
 monkeypatch.setattr(p,'check_services',lambda _: {'8116':{'ok':True}})
 monkeypatch.setattr(p,'run_trial',lambda *a:pytest.fail('Trial must not run'))
 with pytest.raises(RuntimeError,match='Gate A failed'):p.run(manifest,'models',tmp_path)
