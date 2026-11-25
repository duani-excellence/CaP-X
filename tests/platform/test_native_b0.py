import copy
import json
from types import SimpleNamespace as S
import pytest


def test_minimal_b0_requires_its_own_program_and_exact_scene(tmp_path):
    from capx.experiments.unified import RunSpec,resolve
    p=tmp_path/'snapshot.json';p.write_text(json.dumps({'seed':0,'state_hash':'frozen'}))
    s,c,f=resolve(RunSpec(methods=['B0'],snapshot=str(p)))
    assert f=='native_b0' and s.skill=='raw'
    for kw in ({'methods':['B0','G1']},{'harness':'open'},{'force_mode':'on'},
               {'skill':'v30'},{'reference':str(tmp_path)},{'snapshot':None}):
        with pytest.raises(ValueError):resolve(RunSpec(**({'methods':['B0'],'snapshot':str(p)}|kw)))


def test_adapter_uses_native_score_and_no_truth_rpc(tmp_path,monkeypatch):
    from capx.experiments import native_b0,native_tasks
    calls=[]
    class Low:
        max_steps=100;_sim_step_count=0
        def _step_once(self):self._sim_step_count+=1
    class Actor:
        def close(self):pass
        def run(self,code,timeout):
            if code=='move':boundary.motion+=1
            return dict(ok=code!='invalid',status='ok' if code!='invalid' else 'invalid_program',stdout='',stderr='')
    class Boundary:
        def __init__(self,spec,seed,root,**kw):
            nonlocal boundary
            boundary=self;self.root=root;root.mkdir(parents=True);self.low=Low();self.initial=kw['expected_snapshot']
            self.actor=Actor();self.calls=self.motion=0
            self.api=S(functions=lambda:{'goto_pose':lambda *a:None,'get_object_pose':lambda *a:None},combined_doc=lambda:'public API')
        def goto_pose(self,*a):calls.append(a)
        def call(self,*a):return None
        def capture(self):return {'robot_cartesian_pos':[0]*7}
        def log(self,*a,**kw):pass
        def close(self):pass
    boundary=None
    monkeypatch.setattr(native_tasks,'NativeBoundary',Boundary)
    monkeypatch.setattr(native_tasks,'physical_state',lambda e:dict(raw_success=True,native_progress={'objects_in_bins':[1]*4}))
    snap={'seed':0,'state_hash':'x','spec':{'api_budget':5}}
    env=native_b0.make_environment('pick_place',tmp_path/'episode',0,snap,5)
    env.boundary.low.robosuite_env=S();env._actor=Actor()
    assert not any('oracle' in k or k in ('execute_skill','decision_state') for k in env._functions)
    assert env.step('pass')[1]==0 and env.step('invalid')[1]==0
    assert env.step('move')[1]==1
    env.close()


def test_b0_recovery_context_contains_no_graph_or_private_physics():
    from capx.experiments.graph_v2 import b0_recovery_prompt
    messages=b0_recovery_prompt([{'role':'user','content':'public task'}],
        {'robot_cartesian_pos':[1,2,3],'objects':{'hidden':[9,9,9]}},'print(1)',
        {'stdout':'public','stderr':'bounded IK failed','physics':{'secret':'truth'}})
    text=json.dumps(messages)
    assert 'bounded IK failed' in text
    assert 'hidden' not in text and 'secret' not in text and 'recovery_scope' not in text


@pytest.mark.parametrize('method',['G0','G1','LOGIV'])
def test_open_semantic_configuration_is_not_silently_aliased_to_legacy(method):
    from capx.experiments.unified import RunSpec,resolve
    with pytest.raises(ValueError,match='open-runtime'):
        resolve(RunSpec(methods=[method],config='env_configs/graph_feedback/development-v36.json'))
