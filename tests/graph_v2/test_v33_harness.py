"""Persistence and recovery gates, using public fixtures; not physics success."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

from capx.execution.agent_events import EventStore
from capx.execution.persistent_harness import RecoveryOrchestrator, classify_failure
from capx.execution.open_recovery import admitted_observation_target


def test_committed_action_is_not_replayed_after_reload(tmp_path):
    s=EventStore(tmp_path)
    s.start_action('move-1','move_tcp',{},physical=True)
    s.finish_action('move-1','DONE',{'tcp':[0,0,1]})
    restored=EventStore(tmp_path,resume=True)
    with pytest.raises(ValueError,match='already'):restored.start_action('move-1','move_tcp',{},physical=True)
    assert restored.actions['move-1']['state']=='DONE'


def test_crash_after_dispatch_is_unknown_and_requires_reconciliation(tmp_path):
    s=EventStore(tmp_path);s.start_action('move-2','goto_pose',{},physical=True)
    restored=EventStore(tmp_path,resume=True)
    assert restored.unresolved()==['move-2']
    assert restored.actions['move-2']['state']=='UNKNOWN'
    with pytest.raises(ValueError):restored.start_action('new-id','goto_pose',{},physical=True)


def test_event_corruption_cannot_be_silently_resumed(tmp_path):
    s=EventStore(tmp_path);s.append('TaskStarted',{'task':'public'})
    p=tmp_path/'events.jsonl';p.write_text(p.read_text().replace('public','changed'))
    with pytest.raises(ValueError,match='integrity'):EventStore(tmp_path,resume=True)


@pytest.mark.parametrize('group',['G0','G1'])
def test_no_deterministic_plan_escalates_equally(group):
    r=RecoveryOrchestrator()
    assert r.decide('PERCEPTION_UNCERTAIN',False,requests_left=2,no_progress=0)['level']=='R2'
    assert r.decide('PHYSICAL_EXECUTION_FAILURE',True,requests_left=2,no_progress=0)['level']=='R1'


@pytest.mark.parametrize('failure',['UNKNOWN_PHYSICAL_EFFECT','HARD_SAFETY_VIOLATION','ENVIRONMENT_FAILURE'])
def test_unknown_effect_and_hard_fault_never_blindly_retry(failure):
    assert RecoveryOrchestrator().decide(failure,True,requests_left=2,no_progress=0)['level']=='R4'


def test_recovery_no_progress_and_budget_are_bounded():
    r=RecoveryOrchestrator()
    assert r.decide('PERCEPTION_UNCERTAIN',False,requests_left=0,no_progress=0)['reason']=='recovery_request_budget'
    assert r.decide('PERCEPTION_UNCERTAIN',False,requests_left=2,no_progress=2)['reason']=='no_information_progress'
    assert classify_failure({'status':'invalid_program','stderr':'syntax'},False)=='PROGRAM_ERROR'
    assert classify_failure({'status':'timeout'},True)=='UNKNOWN_PHYSICAL_EFFECT'


def test_checkpoint_is_audit_not_physics_rollback(tmp_path):
    s=EventStore(tmp_path);a=s.checkpoint({'remaining_goals':['place'],'robot':{'tcp':[0,0,1]}})
    b=s.checkpoint({'remaining_goals':['place'],'robot':{'tcp':[0,0,2]}})
    assert a!=b and json.loads(a.read_text())['state']['robot']['tcp'][2]==1
    assert json.loads(b.read_text())['physical_rollback'] is False


def test_open_motion_cannot_bypass_certified_corridor():
    tcp=np.array([.5,.1,.18,0,0,1,0]);origin=np.array([.5,.1,0.])
    p,q=admitted_observation_target(tcp,origin,[.5,.1,.19],[0,0,1,0])
    assert np.allclose(p,[.5,.1,.19])
    for target in [[.6,.1,.19],[.5,.1,.16],[.5,.1,.3],[float('nan'),.1,.19]]:
        with pytest.raises(ValueError):admitted_observation_target(tcp,origin,target,q)
    with pytest.raises(ValueError):admitted_observation_target(tcp,origin,p,[1,0,0,0])


@pytest.mark.parametrize('group',['G0','G1'])
def test_actual_harness_escalates_exhausted_recovery_without_reset(group,tmp_path):
    from capx.execution.persistent_harness import PersistentHarness
    import time
    c=SimpleNamespace(definition={'goals':['placed:novel']},quarantined=False,conditions={})
    skills=SimpleNamespace(scene=SimpleNamespace(source=None),held=None,phase='empty',motion_version=0)
    r=SimpleNamespace(c=c,skills=skills,group=group,emit=lambda *a,**k:None,terminal_reason=None,last_error='missing_bounded_wall_pair',
        broker=SimpleNamespace(calls=0,max_calls=10),state=lambda:dict(phase='empty',attachment=None,remaining_goals=['placed:novel'],conditions={}))
    def no_plan():r.terminal_reason='no_legal_recovery_capability';return False
    r.prepare_recovery=no_plan;r.deterministic_code=lambda:None
    b=SimpleNamespace(root=tmp_path,spec={'episode_seconds':100},started=time.monotonic(),calls=0,initial={'state_hash':'same'},actor=SimpleNamespace(call_site={'line':3}))
    h=PersistentHarness(b,r,SimpleNamespace(max_requests=2,count=0),request_limit=2)
    h.program(0,'print("initial")','historical_initial','old_receipt')
    decision,_=h.next({'status':'interrupted','ok':False})
    assert decision['level']=='R2' and r.agent_recovery and r.terminal_reason is None
    assert h.context({})['unknown_physical_calls']==[]
    assert not hasattr(h.tools,'env')
    h.supervisor.requests=2
    assert h.next({'status':'interrupted'})[0]['reason']=='recovery_request_budget'


def test_monitor_preserves_synchronous_stop_but_quarantines_unknown(tmp_path):
    from capx.execution.persistent_harness import ExecutionMonitor
    from capx.execution.skill_contracts import SkillBoundaryStop
    s=EventStore(tmp_path);b=SimpleNamespace(actor=SimpleNamespace(call_site={'line':4}))
    m=ExecutionMonitor(s,lambda:{'robot':{'tcp':[1,2,3]}},b)
    def controlled():raise SkillBoundaryStop('force guard: controlled fixed target',{})
    with pytest.raises(SkillBoundaryStop):m.invoke('goto_pose',[],{},controlled)
    assert not s.unresolved()
    def uncertain():raise ValueError('driver failed after dispatch')
    with pytest.raises(ValueError):m.invoke('goto_pose',[],{},uncertain)
    assert s.unresolved()
    with pytest.raises(ValueError,match='Unknown physical'):m.invoke('open_gripper',[],{},lambda:None)


def test_monitor_does_not_retry_physical_call(tmp_path):
    from capx.execution.persistent_harness import ExecutionMonitor
    s=EventStore(tmp_path);m=ExecutionMonitor(s,lambda:{},SimpleNamespace(actor=SimpleNamespace(call_site=None)))
    calls=[]
    def transient():calls.append(1);raise ConnectionError('offline')
    with pytest.raises(ConnectionError):m.invoke('goto_pose',[],{},transient)
    assert len(calls)==1 and s.unresolved()


def test_source_ids_and_wrench_timestamps_do_not_count_as_progress(tmp_path):
    from capx.execution.persistent_harness import TaskSupervisor
    import time
    state=dict(phase='held_verified',attachment='new_name',remaining_goals=['placed'],conditions={'target':{'truth':'UNKNOWN','source':'a'}})
    r=SimpleNamespace(state=lambda:state,skills=SimpleNamespace(motion_version=0))
    s=TaskSupervisor(None,r,None,2)
    assert s.progress()['changed']
    state['conditions']['target']['source']='new'
    assert not s.progress()['changed']
    assert s.progress()['no_progress']==2
    r.skills.motion_version=1
    assert not s.progress()['changed']  # A successful no-op is not physical progress.
    r.skills.scene=SimpleNamespace(obs={'robot_cartesian_pos':[.5,.1,.2,0,0,1,0]})
    assert s.progress()['changed']


def test_replay_never_enables_paid_recovery_implicitly():
    from capx.experiments.unified import RunSpec,resolve
    s,c,f=resolve(RunSpec(harness='persistent',reference='not_opened_in_resolve',force_mode='on'))
    assert s.recovery_requests==0 and c['tasks']['pick_place']['harness_max_requests']==0
    s,c,f=resolve(s.model_copy(update={'recovery_requests':2}))
    assert c['budgets']['max_requests']==2
    with pytest.raises(ValueError):resolve(RunSpec(recovery_requests=2))


def test_public_robot_pose_with_gripper_component_uses_only_quaternion():
    tcp=[.5,.1,.18,0,0,1,0,.04]
    p,q=admitted_observation_target(tcp,[.5,.1,0],[.5,.1,.19],[0,0,1,0])
    assert len(q)==4 and p[2]==.19


def test_open_recovery_motion_changes_current_graph_evidence(tmp_path):
    from capx.execution.open_recovery import OpenRecoveryAPIs
    from capx.execution.skill_contracts import ContractGraph
    definition={'schema':'capx.contract_task.v1','objects':['unseen'],'resources':['arm0'],'goals':['placed:unseen'],
      'intents':[{'id':'put','skill':'deposit','requires':['held:unseen','geometry:new'],'effects':['placed:unseen'],'resources':['arm0']}],
      'condition_specs':{'geometry:new':{'kind':'tool_data'}}}
    c=ContractGraph(definition);c.observe('geometry:new','TRUE',{'source':'old'});c.register_program('move_tcp(...)','unit')
    scene=SimpleNamespace(source='a',obs={'robot_cartesian_pos':np.array([.5,.1,.18,0,0,1,0,.04])})
    scene.frame=lambda:None
    moved=[]
    s=SimpleNamespace(scene=scene,phase='held_verified',held={'object':'unseen'},acquisition={'p':np.array([.5,.1,0])},motion_version=0,
      current_held=lambda:None,_corridor=lambda *a,**k:None,reachable=lambda *a:True)
    def move(p,q):moved.append(p);scene.obs['robot_cartesian_pos'][:3]=p;s.motion_version+=1
    s.move=move;r=SimpleNamespace(skills=s,c=c,planner=SimpleNamespace(motion_epoch=lambda x:None))
    a=OpenRecoveryAPIs(r,EventStore(tmp_path));out=a.move_tcp([.5,.1,.19],[0,0,1,0],'new-view')
    assert len(moved)==1 and out['status']=='PASS' and len(out['observation']['robot']['tcp'])==7
    assert c.truth('geometry:new')=='UNKNOWN' and c.truth('held:unseen')=='TRUE'
    with pytest.raises(ValueError,match='already'):a.move_tcp([.5,.1,.2],[0,0,1,0],'new-view')
    checks=[];s.effects={}
    def uncertain_hold():
        checks.append(1)
        if len(checks)==2:raise ValueError('attachment evidence contradicted after movement')
    s.current_held=uncertain_hold
    with pytest.raises(ValueError,match='contradicted'):a.move_tcp([.5,.1,.2],[0,0,1,0],'next-view')
    assert c.truth('held:unseen')=='UNKNOWN' and s.phase=='closed_unverified'
    from capx.experiments.artifacts import replay_graph
    view=replay_graph(c.graph.events)
    states=[n['state'] for n in view['nodes'].values() if n['kind']=='api_call']
    assert states==['done','invalid']


def test_agent_recovery_state_does_not_claim_old_worklist_is_permission():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.experiments.native_vision_skills import region_task
    skills=SimpleNamespace(phase='held_verified',held={'object':'other'},acquire=lambda row:None,deposit=lambda row:None,verify=lambda keys:{})
    r=ModelContractRuntime(skills,region_task({'other':'public'},grounded=True),group='G1',
      limits={'skill_budget':10,'episode_seconds':60,'recovery_version':27},emit=lambda *a,**k:None)
    r.work_scope={'enforced':True,'allowed_intents':[]};r.agent_recovery=True
    s=r.state()
    assert s['recovery_scope']['enforced'] is False
    assert s['graph_repair_enforced'] is False
    assert s['recovery_scope']['role']=='graph_recommendation_only'
    assert r.work_scope['enforced'] is True  # Historical record unchanged.


def test_hard_process_interruption_keeps_dispatched_motion_unknown(tmp_path):
    from capx.execution.persistent_harness import ExecutionMonitor
    s=EventStore(tmp_path);m=ExecutionMonitor(s,lambda:{},SimpleNamespace(actor=SimpleNamespace(call_site=None)))
    def interrupt():raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):m.invoke('goto_pose',[],{},interrupt)
    assert s.unresolved() and list(s.actions.values())[0]['state']=='UNKNOWN'


def test_open_agent_scope_does_not_bypass_live_contract():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.experiments.native_vision_skills import region_task
    from capx.execution.sandbox import ExecutionInterrupted
    calls=[]
    skills=SimpleNamespace(phase='held_verified',held={'object':'unseen'},acquire=lambda row:calls.append(row),deposit=lambda row:calls.append(row),verify=lambda keys:{})
    r=ModelContractRuntime(skills,region_task({'unseen':'public'},grounded=True),group='G1',
      limits={'skill_budget':10,'episode_seconds':60,'recovery_version':27},emit=lambda *a,**k:None)
    r.agent_recovery=True;r.recovering=True;r.work_scope={'enforced':True,'allowed_intents':[]}
    r.broker.verify=lambda keys:{k:dict(truth='TRUE' if k=='held:unseen' else 'UNKNOWN',evidence={'source':'public_fixture'}) for k in keys}
    with pytest.raises(ExecutionInterrupted,match='precondition'):r.execute('deposit-0')
    assert calls==[] and r.broker.calls==0


def test_recorded_program_requires_honest_explicit_provenance(tmp_path):
    from capx.experiments.contract_models import run_episode
    with pytest.raises(ValueError,match='provenance'):
        run_episode(tmp_path/'no-run',{'harness_version':33},0,'G1',None,{},None,
            recorded_recoveries=[{'code':'print(1)','origin':'real_deepseek'}])
    assert not (tmp_path/'no-run').exists()
