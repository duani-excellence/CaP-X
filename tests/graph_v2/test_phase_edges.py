"""Adversarial boundary regressions using public fake observations only."""
import numpy as np
import pytest
from test_grounded_monitor import fixture
from capx.execution.grounded_feedback import APIS
from capx.execution.sandbox import ExecutionInterrupted, SandboxProcess


@pytest.mark.parametrize('mode',['G0','G1'])
def test_nonplan_parent_id_is_uncatchable_contract_interruption(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    robot=m.invoke('read_robot',[],{},raw)
    with pytest.raises(ExecutionInterrupted):
        m.invoke('run_plan',[robot['id']],{},raw)
    assert not calls


@pytest.mark.parametrize('mode',['G0','G1'])
def test_actor_cannot_swallow_wrong_node_kind_then_send_motion(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    robot=m.invoke('read_robot',[],{},raw)
    code=f'try:\n    run_plan("{robot["id"]}")\nexcept Exception:\n    open_gripper()\n'
    m.begin_program(code,1)
    actor=SandboxProcess(dict.fromkeys(APIS+['open_gripper']),
        callback=lambda name,args,kwargs:m.invoke(name,args,kwargs,raw))
    try:
        result=actor.run(code,timeout=10)
        assert result['status']=='interrupted'
        assert not calls
    finally:actor.close()


def _approach(m,raw):
    m.invoke('observe_object',['red cube'],{},raw)
    grasp=m.invoke('observe_grasp',['red cube'],{},raw)
    plan=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{},raw)
    m.invoke('run_plan',[plan],{},raw)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_home_after_approach_cannot_authorize_close_at_different_pose(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    _approach(m,raw)
    def move_home(name,args,kwargs):
        if name=='home_pose':
            calls.append((name,args));state['robot_cartesian_pos'][:3]=[.7,.3,.4]
            return None
        return raw(name,args,kwargs)
    try:m.invoke('home_pose',[],{},move_home)
    except ExecutionInterrupted:
        assert not any(name=='home_pose' for name,_ in calls)
        return
    before=len(calls)
    with pytest.raises(ExecutionInterrupted):m.invoke('close_gripper',[],{},move_home)
    assert len(calls)==before


@pytest.mark.parametrize('mode',['G0','G1'])
def test_failed_active_hold_probe_revokes_old_holding_permission(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='holding_verified';m.closed=True;m.lifted=True;m.held_object='red cube';m.lift_attempts=1
    def robot_without_object(name,args,kwargs):
        if name=='goto_pose':
            calls.append((name,args))
            state['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
            return None
        return raw(name,args,kwargs)
    with pytest.raises(ExecutionInterrupted,match='hold_probe'):
        m.invoke('verify_hold',[],{},robot_without_object)
    assert m.phase=='holding_unverified'
    assert not m.lifted
    m.begin_program('next actor after failed active probe',1)
    assert m.phase=='holding_unverified' and not m.lifted
    assert m.hold_probes==1


@pytest.mark.parametrize('mode',['G0','G1'])
def test_approach_failure_does_not_leave_a_new_object_authorized_for_close(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    _approach(m,raw)
    assert m.approached_object=='red cube'
    m.invoke('observe_object',['green cube'],{},raw)
    grasp=m.invoke('observe_grasp',['green cube'],{},raw)
    plan=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{},raw)
    def stuck(name,args,kwargs):
        if name=='goto_pose':calls.append((name,args));return None
        return raw(name,args,kwargs)
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},stuck)
    assert m.phase!='approached'
    before=len(calls)
    with pytest.raises(ExecutionInterrupted):m.invoke('close_gripper',[],{},raw)
    assert len(calls)==before


def test_graph_diagnosis_contains_actual_lift_uncertainty_check():
    m,state,surfaces,calls,raw=fixture('G1')
    _approach(m,raw)
    m.invoke('close_gripper',[],{},raw)
    m.baseline=None
    robot=m.invoke('read_robot',[],{},raw)
    lift=m.invoke('plan_skill',['lift',{'robot':robot['id']}],{},raw)
    with pytest.raises(ExecutionInterrupted,match='verify_hold'):
        m.invoke('run_plan',[lift],{},raw)
    checks=[n for n in m.graph.nodes.values()
            if n['kind']=='check' and n['data'].get('check')=='lift' and n['data'].get('status')=='UNCERTAIN']
    assert len(checks)==1
    check=checks[0]
    assert check['data']['metrics']['baseline_available'] is False
    feedback=m.summary()
    assert check['id'] in m.graph.ancestors(feedback['diagnosis']['roots'])
    assert check['id'] in {n['id'] for n in feedback['diagnosis']['nodes']}


@pytest.mark.parametrize('name',['not an object','red cube typo',''])
def test_unknown_surface_name_is_controlled_interruption(name):
    m,state,surfaces,calls,raw=fixture()
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',[name],{},raw)
    assert not calls


@pytest.mark.parametrize('mode',['G0','G1'])
def test_horizontal_hold_probe_cannot_replace_the_required_lift(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='holding_unverified';m.closed=True;m.lifted=False;m.held_object='red cube'
    m.lift_attempts=0
    try:m.invoke('verify_hold',[],{},raw)
    except ExecutionInterrupted:pass
    assert not m.lifted and m.phase!='holding_verified'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_failed_lift_attempt_is_not_refunded_when_actor_restarts(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='holding_unverified';m.closed=True;m.lifted=False;m.held_object='red cube'
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['lift',{'robot':robot['id']}],{},raw)
    def stuck(name,args,kwargs):
        if name=='goto_pose':calls.append((name,args));return None
        return raw(name,args,kwargs)
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},stuck)
    assert m.lift_attempts==1
    m.begin_program('recovery after control did not converge',1)
    robot=m.invoke('read_robot',[],{},raw)
    again=m.invoke('plan_skill',['lift',{'robot':robot['id']}],{},raw)
    before=len(calls)
    with pytest.raises(ExecutionInterrupted,match='verify_hold'):
        m.invoke('run_plan',[again],{},stuck)
    assert len(calls)==before and m.lift_attempts==1
