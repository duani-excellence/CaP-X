"""Graph failure-boundary regressions: no simulation or model requests."""
import numpy as np
import pytest
from capx.execution.graph_feedback import GraphFeedbackMonitor, GRAPH_APIS
from capx.execution.sandbox import ExecutionInterrupted, SandboxProcess


def make(mode='G1'):
    state={'robot_cartesian_pos':np.array([.5,0,.1,1,0,0,0,.5]),'robot_joint_pos':np.zeros(8)}
    calls=[]
    def raw(name,args,kwargs):
        calls.append(name)
        if name=='get_object_pose':return np.array([.5,0,0]),np.array([1,0,0,0]),np.array([.04,.04,.04])
        if name=='goto_pose':state['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
    monitor=GraphFeedbackMonitor(mode,lambda:state)
    monitor.evidence_fn=lambda _ : {}
    monitor.begin_program('initial()',0)
    red=monitor.invoke('observe_object',['red cube'],{},raw)
    green=monitor.invoke('observe_object',['green cube'],{},raw)
    place=monitor.invoke('plan_motion',['place',[.5,0,.05],[1,0,0,0],[red['id'],green['id']]],{},raw)
    retreat=monitor.invoke('plan_motion',['retreat',[.5,0,.2],[1,0,0,0],[place]],{},raw)
    return monitor,raw,calls,state,(red['id'],green['id'],place,retreat)


def assert_discarded(monitor,ids):
    assert all(not monitor.graph.nodes[node]['valid'] for node in ids)
    assert monitor.interruption_roots
    done=[node for node in monitor.graph.nodes.values() if node['kind']=='call' and node['state']=='done']
    assert len(done)==2 and all(node['valid'] for node in done)
    assert not monitor.summary()['available_results']


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('bad_call',[
    ('plan_motion',['place',[.5,0],[1,0,0,0],[]],{}),
    ('plan_motion',[],{'unexpected_argument':True}),
])
def test_broker_argument_errors_interrupt_and_discard_unknown_suffix(mode,bad_call):
    monitor,raw,calls,state,ids=make(mode)
    before=list(calls)
    with pytest.raises(ExecutionInterrupted,match='api_contract_error'):
        monitor.invoke(*bad_call,fn=raw)
    assert calls==before
    assert_discarded(monitor,ids)
    assert monitor.events[-1]['reason']=='api_contract_error'
    assert monitor.events[-1]['metrics']['error_type'] in ('ValueError','TypeError')


@pytest.mark.parametrize('mode',['G0','G1'])
def test_partial_motion_argument_failure_preserves_fact_without_replay(mode):
    monitor,raw,calls,state,ids=make(mode)
    monitor.closed=monitor.lifted=True
    def partial(name,args,kwargs):
        raw(name,args,kwargs)
        if name=='goto_pose':raise ValueError('controller reported invalid target after partial movement')
    with pytest.raises(ExecutionInterrupted,match='api_contract_error'):
        monitor.invoke('run_plan',[ids[2]],{},partial)
    assert calls.count('goto_pose')==1
    assert state['robot_cartesian_pos'][2]==.05
    assert monitor.inflight['status']=='unknown_partial_effects'
    assert_discarded(monitor,ids)
    assert monitor.graph.nodes[ids[2]]['data']['execution']=='consumed'
    monitor.begin_program('new recovery actor',1)
    with pytest.raises(ExecutionInterrupted):monitor.invoke('run_plan',[ids[2]],{},raw)
    assert calls.count('goto_pose')==1


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('status',['runtime_error','invalid_program','policy_violation'])
def test_program_failure_boundary_discards_suffix_and_recovery_reobserves(mode,status):
    monitor,raw,calls,state,ids=make(mode)
    before=list(calls)
    assert monitor.end_block_failure(status,'ZeroDivisionError: division by zero') is True
    assert_discarded(monitor,ids)
    assert calls==before
    assert monitor.events[-1]['metrics']['execution_status']==status
    monitor.begin_program('fresh actor',1)
    fresh=monitor.invoke('observe_object',['green cube'],{},raw)
    assert monitor.graph.nodes[fresh['id']]['valid']
    assert not monitor.graph.nodes[ids[1]]['valid']
    assert calls==before+['get_object_pose']


@pytest.mark.parametrize('status',['ok','interrupted','timeout','system_error'])
def test_finished_interrupted_and_rebuild_statuses_keep_original_handling(status):
    monitor,raw,calls,state,ids=make()
    before=monitor.graph.snapshot()
    assert monitor.end_block_failure(status,'original handling') is False
    assert monitor.graph.snapshot()==before


@pytest.mark.parametrize('mode',['G0','G1'])
def test_actor_cannot_catch_broker_contract_error_and_continue(mode):
    monitor,raw,calls,state,ids=make(mode)
    code='''try:
    plan_motion("place", [0, 0], [1, 0, 0, 0], [])
except Exception:
    close_gripper()
'''
    monitor.begin_program(code,1)
    actor=SandboxProcess(dict.fromkeys(GRAPH_APIS+['close_gripper']),callback=lambda name,args,kwargs:monitor.invoke(name,args,kwargs,raw))
    try:
        result=actor.run(code,timeout=10)
        assert result['status']=='interrupted',result
        assert 'api_contract_error' in result['stderr']
        assert 'close_gripper' not in calls
        assert actor.pid is None
        # The failed actor is gone. A fresh actor can use a fresh encoder read
        # for the unfinished retreat without replaying earlier physical calls.
        monitor.released=True
        monitor.begin_program('fresh verified retreat',2)
        recovery='''robot = read_robot()
plan = plan_motion("retreat", [0.5, 0, 0.2], [1, 0, 0, 0], [robot["id"]])
run_plan(plan)
'''
        resumed=actor.run(recovery,timeout=10)
        assert resumed['status']=='ok',resumed
        assert monitor.retreat_completed
        assert calls.count('goto_pose')==1 and 'close_gripper' not in calls
    finally:actor.close()


def test_error_feedback_never_contains_raw_exception_payload():
    monitor,raw,calls,state,ids=make()
    secret='Authorization: Bearer PRIVATE_SENTINEL'
    def fail(name,args,kwargs):raise ValueError(secret)
    with pytest.raises(ExecutionInterrupted):monitor.invoke('observe_object',['red cube'],{},fail)
    assert secret not in str(monitor.graph.snapshot()) and secret not in str(monitor.events)
    monitor.end_block_failure('runtime_error',secret)
    assert secret not in str(monitor.graph.snapshot()) and secret not in str(monitor.events)


@pytest.mark.parametrize('failure',['unknown_contract','runtime_error'])
def test_g1_conservative_fallback_reports_entire_discarded_scope(failure):
    monitor,raw,calls,state,ids=make('G1')
    other=monitor.graph.add('estimate',{'object':'unrelated_uncertain'},key='other')
    plan=monitor.graph.add('plan',{'skill':'unrelated'},parents=[other])
    check=monitor.graph.add('check',{'status':'PASS'},parents=[plan])
    if failure=='unknown_contract':
        with pytest.raises(ExecutionInterrupted):monitor.invoke('goto_pose',[[0,0,0],[1,0,0,0]],{},raw)
    else:monitor.end_block_failure('runtime_error','ZeroDivisionError: division by zero')
    feedback=monitor.summary();scope=feedback['recovery_scope']
    assert scope['policy']=='all_unexecuted_suffix' and scope['fallback'] is True
    assert other in scope['reobserve'] and plan in scope['recompute'] and check in scope['revalidate']
    assert ids[2] in scope['recompute'] and ids[3] in scope['recompute']
    assert feedback['diagnostic_source']=='initial()'
    assert not feedback['available_results']
    monitor.begin_program('fresh actor',1)
    assert monitor.conservative_fallback is False
    assert monitor.summary()['recovery_scope']['policy']=='affected_dependencies'


def test_runner_finalizes_ordinary_failure_before_finish_feedback(monkeypatch,tmp_path):
    import time
    from types import SimpleNamespace
    from capx.experiments import graph_feedback as runner
    import capx.envs.physical_evaluation as evaluation
    order=[]
    class FakeEnv:
        def __init__(self):
            self.cfg=SimpleNamespace();self._full_prompt=[{'role':'user','content':'Stack task'}]
            self._apis={'vision':SimpleNamespace(combined_doc=lambda:'vision docs')}
            self._trace_path=tmp_path/'trace.jsonl';self._api_calls=0
            self._request_audit=SimpleNamespace(count=0);self._actor=SimpleNamespace(close=lambda:None)
            data=SimpleNamespace(xpos=np.zeros((1,3)),xmat=np.eye(3).reshape(1,9))
            self.low_level_env=SimpleNamespace(base_link_idx=0,robosuite_env=SimpleNamespace(sim=SimpleNamespace(data=data),cubeA=SimpleNamespace(size=[.02]*3),cubeB=SimpleNamespace(size=[.025]*3)))
        def reset(self,**kwargs):return {'full_prompt':self._full_prompt},{'initial_state_hash':'state'}
        def enable_video_capture(self,*a,**kw):pass
        def step(self,code):return {'full_prompt':self._full_prompt},0.0,False,False,dict(execution_status='runtime_error',api_calls=0,stderr='ZeroDivisionError: division by zero')
        def get_video_frames(self):return []
        def close(self):pass
    class FakeBudget:
        count=0;deadline=time.monotonic()+30;max_requests=1
        def query(self,prompt,label):
            self.count+=1
            return dict(content='```python\n1/0\n```',finish_reason='stop',response_model='deepseek-v4-pro'),'mock-only'
    monitor=SimpleNamespace(begin_program=lambda *a:order.append('begin'),
        end_block_failure=lambda status,detail:order.append(('failure',status,detail)),
        finish_check=lambda:order.append('finish'),graph=SimpleNamespace(snapshot=lambda:{'nodes':{}},events=[]),
        recoveries=[],events=[],fault_fired=False)
    monkeypatch.setattr(runner,'environment',lambda *a:FakeEnv())
    monkeypatch.setattr(runner,'load_scene',lambda *a:{'state_hash':'state'})
    monkeypatch.setattr(runner,'_api_doc',lambda *a:'graph docs')
    monkeypatch.setattr(runner,'instrument_tools',lambda *a:None)
    monkeypatch.setattr(runner,'install_monitor',lambda *a:monitor)
    monkeypatch.setattr(evaluation,'physical_state',lambda *a:{})
    result,_=runner.run_episode(tmp_path,tmp_path/'manifest.json',dict(seed=7,state_hash='state',model_hash='model'),
        'G1',None,FakeBudget(),SimpleNamespace(api_budget=64,max_local=2,max_blocks=1))
    assert result['status']=='runtime_error'
    assert result['stop_reason']=='episode_code_block_budget'
    assert order==['begin',('failure','runtime_error','ZeroDivisionError: division by zero'),'finish']
