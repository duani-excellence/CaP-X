import json
from pathlib import Path
from argparse import Namespace
import pytest
from capx.multitask.cli import parse_targets,export


def test_independent_targets():
 assert parse_targets(['stack=2','lift=3'])=={'stack':2,'lift':3}
 for bad in [['stack=0'],['stack=2','stack=3'],['not_task=1']]:
  with pytest.raises(ValueError):parse_targets(bad)


def test_balanced_export_preserves_task_and_source(tmp_path):
 from capx.training.teacher import dump,digest
 root=tmp_path/'run';root.mkdir();dump(root/'multitask.json',{'targets':{'stack':10,'lift':1}})
 for task,count in [('stack',3),('lift',1)]:
  view=root/task/'exports/one';view.mkdir(parents=True)
  rows=[]
  for i in range(count):
   a=root/task/f'raw{i}';a.mkdir();prompt=[{'role':'user','content':task+str(i)}];code='open_gripper()'
   dump(a/'result.json',dict(status='success',reward=1.,truncated=False,snapshot_hash=f'{task}-{i}',prompt_hash=digest(prompt),info=dict(physical_success=True,sandbox_rc=0,execution_status='ok',motion_calls=1)))
   dump(a/'prompt.json',prompt);(a/'generated.py').write_text(code);dump(a/'model-response.json',dict(content=code,response_model='deepseek-v4-pro',finish_reason='stop'))
   rows.append(dict(source='teacher',mode='privileged',snapshot_hash=f'{task}-{i}',artifacts=str(a)))
  (view/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 out=tmp_path/'out';export(root,out)
 m=json.loads((out/'manifest.json').read_text());assert m['balanced_cap']==1
 rows=[json.loads(l) for l in (out/'index.jsonl').read_text().splitlines()]
 assert {r['task'] for r in rows}=={'stack','lift'}
 assert len(rows)==2
 assert all(r['source']=='teacher' for r in rows)
 with pytest.raises(FileExistsError):export(root,out)


def test_web_blocks_unknown_tasks_and_bad_targets():
 from capx.web.multitask import JobRequest,start
 from fastapi import HTTPException
 with pytest.raises(HTTPException):start(JobRequest(targets={'oops':1}))
 with pytest.raises(HTTPException):start(JobRequest(targets={'stack':0}))


def test_oracles_do_not_widen_sandbox():
 from capx.multitask.worker import oracle
 from capx.execution.sandbox import validate
 for task in ['lift','restack','nut_assembly','wipe','handover','two_arm_lift']:
  validate(oracle(task))


def test_runtime_invalidated_after_timeout():
 from capx.multitask.runtime import Runtime
 obj=Runtime.__new__(Runtime);obj.invalidated=True
 with pytest.raises(RuntimeError,match='Fresh'):obj.execute('pass')
