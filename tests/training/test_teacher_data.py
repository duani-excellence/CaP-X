import json
from pathlib import Path
import pytest


def fixture_sample(tmp_path,mode='privileged',source='teacher'):
 from capx.experiments.pilot import dump,digest
 d=tmp_path/'sample';d.mkdir()
 prompt=[{'role':'system','content':'Write Python'},{'role':'user','content':[{'type':'text','text':'Public API documentation'}]}]
 code='open_gripper()'
 result={'status':'success','reward':1.,'truncated':False,'snapshot_hash':'state-a','prompt_hash':digest(prompt),
         'info':{'physical_success':True,'sandbox_rc':0,'execution_status':'ok','motion_calls':1}}
 dump(d/'prompt.json',prompt);(d/'generated.py').write_text(code)
 dump(d/'model-response.json',{'content':code,'finish_reason':'stop','response_model':'deepseek-v4-pro'})
 dump(d/'result.json',result)
 return {'id':'one','mode':mode,'source':source,'artifacts':str(d),'snapshot_hash':'state-a'}


def test_privileged_never_exports_as_vision(tmp_path):
 from capx.training.teacher import training_example
 row=fixture_sample(tmp_path)
 assert training_example(row,'vision')[0] is None
 sample,reason=training_example(row,'privileged')
 assert reason is None
 assert sample['messages'][-1]=={'role':'assistant','content':'open_gripper()'}
 assert sample['messages'][1]['content']=='Public API documentation'
 assert set(sample)=={'messages'}

@pytest.mark.parametrize('mutation',[{'status':'system_error'},{'reward':0.},{'truncated':True},{'info':{'physical_success':False,'sandbox_rc':0,'motion_calls':1}},{'info':{'physical_success':True,'sandbox_rc':1,'motion_calls':1}}])
def test_failure_rejected(tmp_path,mutation):
 from capx.training.teacher import training_example
 row=fixture_sample(tmp_path,'vision');p=Path(row['artifacts'])/'result.json'
 data=json.loads(p.read_text());data.update(mutation);p.write_text(json.dumps(data))
 assert training_example(row,'vision')[0] is None

@pytest.mark.parametrize('source',['oracle','mock'])
def test_non_teacher_sources_rejected(tmp_path,source):
 from capx.training.teacher import training_example
 row=fixture_sample(tmp_path,'vision',source)
 assert training_example(row,'vision')[0] is None


def test_truncated_generation_rejected(tmp_path):
 from capx.training.teacher import training_example
 row=fixture_sample(tmp_path,'vision');p=Path(row['artifacts'])/'model-response.json'
 data=json.loads(p.read_text());data['finish_reason']='length';p.write_text(json.dumps(data))
 assert training_example(row,'vision')[0] is None


def test_transfer_requires_source_validation(tmp_path):
 from capx.training.teacher import training_example
 row=fixture_sample(tmp_path,'vision','transfer')
 assert training_example(row,'vision')[0] is None


def test_snapshot_group_split_is_stable():
 from capx.training.teacher import split_group
 assert split_group('same',.2)==split_group('same',.2)
 assert split_group('same',0)=='train'


def test_export_refuses_overwrite(tmp_path):
 from capx.training.teacher import export_data
 source=tmp_path/'source';source.mkdir();(source/'records.jsonl').write_text('')
 out=tmp_path/'out';out.mkdir()
 with pytest.raises(FileExistsError):export_data([source],out,'vision',.2)


def test_transfer_exports_vision_prompt_only(tmp_path):
 from capx.training.teacher import training_example
 from capx.experiments.pilot import digest
 teacher_dir=tmp_path/'teacher';teacher_dir.mkdir()
 vision_dir=tmp_path/'vision';vision_dir.mkdir()
 source=fixture_sample(teacher_dir)
 row=fixture_sample(vision_dir,'vision','transfer');row['teacher_source']=source
 prompt=[{'role':'user','content':'Vision APIs, no teacher truth'}]
 d=Path(row['artifacts']);(d/'prompt.json').write_text(json.dumps(prompt))
 r=json.loads((d/'result.json').read_text());r['prompt_hash']=digest(prompt)
 (d/'result.json').write_text(json.dumps(r))
 sample,reason=training_example(row,'vision')
 assert reason is None
 assert sample['messages'][0]['content']=='Vision APIs, no teacher truth'
 source['snapshot_hash']='other-scene'
 assert training_example(row,'vision')[0] is None


def test_same_snapshot_never_crosses_splits(tmp_path):
 from capx.training.teacher import export_data
 from capx.experiments.pilot import digest
 source=tmp_path/'run';source.mkdir()
 rows=[]
 for i in range(2):
  base=source/str(i);base.mkdir();row=fixture_sample(base,'vision');row['id']=str(i)
  if i:
   d=Path(row['artifacts']);prompt=[{'role':'user','content':'different prompt same physical snapshot'}]
   (d/'prompt.json').write_text(json.dumps(prompt));r=json.loads((d/'result.json').read_text());r['prompt_hash']=digest(prompt);(d/'result.json').write_text(json.dumps(r))
  rows.append(row)
 (source/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 out=tmp_path/'export';counts=export_data([source],out,'vision',.5)
 assert counts['accepted']==2
 index=[json.loads(l) for l in (out/'index.jsonl').read_text().splitlines()]
 assert len({r['split'] for r in index})==1
 assert 'physical_success' not in (out/'train.jsonl').read_text()+(out/'validation.jsonl').read_text()


def test_teacher_http_failure_is_not_retried(monkeypatch):
 import requests
 from capx.llm import client
 calls=[]
 class Response:
  status_code=503;headers={}
  def json(self):return {'error':'unavailable'}
  def raise_for_status(self):raise requests.HTTPError('mock 503',response=self)
 monkeypatch.setattr(client,'audited_post',lambda *a,**k:calls.append(k) or Response())
 args=client.ModelQueryArgs(model='deepseek-v4-pro',server_url='https://api.deepseek.com/v1/chat/completions',max_attempts=1)
 with pytest.raises(requests.HTTPError):client.query_model(args,[{'role':'user','content':'mock'}])
 assert len(calls)==1


def test_worker_sets_single_request_and_preserves_api_error(tmp_path,monkeypatch):
 import capx.training.teacher as t
 calls=[]
 def trial(root,manifest,entry,mode,provider,ledger,**kwargs):
  calls.append((provider,kwargs))
  result={'seed':0,'snapshot_hash':'state','status':'success','reward':1,'truncated':False,'info':{'physical_success':True,'sandbox_rc':0,'execution_status':'ok','motion_calls':1}}
  if provider=='deepseek':result.update(status='api_error',reward=None)
  t.dump(root/provider/mode/'seed-0'/'result.json',result)
  return result
 monkeypatch.setattr(t,'run_trial',trial)
 root=tmp_path/'job';root.mkdir()
 job=dict(id='job',root=str(root),manifest=str(tmp_path/'manifest'),entry={'seed':0},mode='privileged',mock=False,cost_limit=1,max_tokens=32768,distill_vision=True)
 t.dump(tmp_path/'job.json',job);t.worker(tmp_path/'job.json')
 assert [x[0] for x in calls]==['oracle','deepseek']
 assert calls[1][1]['max_attempts']==1
 assert t.read(root/'worker-records.json')[0]['status']=='api_error'


def test_request_limit_checked_before_spawn(tmp_path,monkeypatch):
 import capx.training.teacher as t
 from types import SimpleNamespace
 snapshot={'version':2,'state_hash':'state','seed':0}
 t.dump(tmp_path/'snapshot.json',snapshot)
 t.dump(tmp_path/'scenes.json',{'scenes':[{'seed':0,'snapshot':'snapshot.json','snapshot_digest':t.digest(snapshot),'state_hash':'state'}]})
 monkeypatch.setattr(t,'check_services',lambda root:{str(p):{'ok':True} for p in [8114,8116,8215]})
 monkeypatch.setattr(t.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn'))
 args=SimpleNamespace(output=tmp_path/'run',manifest=tmp_path/'scenes.json',seeds=None,modes=['privileged','vision'],max_requests=1,max_cost_usd=1,max_tokens=32768,worker_timeout=360,distill_vision=False,mock=True)
 with pytest.raises(ValueError,match='exceed'):t.collect(args)


def run_args(tmp_path):
 from argparse import Namespace
 return Namespace(output=str(tmp_path/'run'),seeds=[0],modes=['privileged','vision'],max_requests=2,max_cost_usd=1.,max_tokens=32768,worker_timeout=360,distill_vision=True,mock=True,validation_fraction=.2)


def test_one_command_pipeline(tmp_path,monkeypatch):
 from capx.training import teacher as t
 args=run_args(tmp_path);events=[]
 monkeypatch.setattr(t,'prepare',lambda manifest,root,seeds:events.append(('prepare',seeds)))
 def collect(a):
  events.append(('collect',a.modes,a.distill_vision,a.mock))
  Path(a.output).mkdir();t.dump(Path(a.output)/'summary.json',{'aborted':False})
 monkeypatch.setattr(t,'collect',collect)
 monkeypatch.setattr(t,'export_data',lambda runs,output,target,fraction:events.append(('export',target)) or {'accepted':0})
 t.run_pipeline(args)
 assert events==[('prepare',[0]),('collect',['privileged','vision'],True,True),('export','vision'),('export','privileged')]
 assert (Path(args.output)/'summary.json').exists()
 with pytest.raises(FileExistsError):t.run_pipeline(args)


def test_one_command_aborts_before_export(tmp_path,monkeypatch):
 from capx.training import teacher as t
 args=run_args(tmp_path)
 monkeypatch.setattr(t,'prepare',lambda *a:None)
 def collect(a):
  Path(a.output).mkdir();t.dump(Path(a.output)/'aborted.json',{'reason':'worker_error'})
 monkeypatch.setattr(t,'collect',collect)
 monkeypatch.setattr(t,'export_data',lambda *a:pytest.fail('must not export aborted collection'))
 with pytest.raises(RuntimeError,match='aborted'):t.run_pipeline(args)


def test_one_command_checks_budget_before_prepare(tmp_path,monkeypatch):
 from capx.training import teacher as t
 args=run_args(tmp_path);args.max_requests=1
 monkeypatch.setattr(t,'prepare',lambda *a:pytest.fail('must validate first'))
 with pytest.raises(ValueError):t.run_pipeline(args)
 assert not Path(args.output).exists()
