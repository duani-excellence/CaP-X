"""Review-only regression expectations: no service, simulator, or model requests.

These deliberately expose current incomplete physical-effect reconciliation.
A callback changes public robot state and then raises, as a post-effect service
or validation error may do. No hidden object state is used by the monitor.
"""
import pytest
from test_grounded_monitor import fixture
from capx.execution.sandbox import ExecutionInterrupted


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('next_action',['close_gripper','home_pose'])
def test_unknown_close_effect_cannot_authorize_repeat_close_or_home(mode,next_action):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='approached';m.approached_object='red cube'
    def post_effect_error(name,args,kwargs):
        if name=='close_gripper':
            calls.append((name,args));state['robot_cartesian_pos'][7]=.45
            state['robot_gripper_joint_pos'][:]=[.018,-.018]
            raise ValueError('post-effect validation failure')
        return raw(name,args,kwargs)
    with pytest.raises(ExecutionInterrupted,match='api_contract_error'):
        m.invoke('close_gripper',[],{},post_effect_error)
    assert m.inflight['status']=='unknown_partial_effects'
    assert len(calls)==1
    m.begin_program('new actor after partially completed close',1)
    before=len(calls)
    try:m.invoke(next_action,[],{},raw)
    except ExecutionInterrupted:pass
    assert len(calls)==before,'Unknown close effect was not reconciled before a new physical action'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_returned_close_is_not_replayed_if_post_call_observation_fails(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='approached';m.approached_object='red cube'
    original=m.geometry_fn;fail=[False]
    def geometry(obs):
        if fail[0]:fail[0]=False;raise ValueError('temporary RGB-D parsing error after close')
        return original(obs)
    m.geometry_fn=geometry
    def close_then_sensor_error(name,args,kwargs):
        answer=raw(name,args,kwargs)
        if name=='close_gripper':fail[0]=True
        return answer
    with pytest.raises(ExecutionInterrupted,match='api_contract_error'):
        m.invoke('close_gripper',[],{},close_then_sensor_error)
    assert m.inflight['status']=='returned'
    assert len(m.call_history)==1
    m.begin_program('new actor after returned close but missing camera sample',1)
    before=len(calls)
    try:m.invoke('close_gripper',[],{},raw)
    except ExecutionInterrupted:pass
    assert len(calls)==before,'A returned close was repeated because bookkeeping waited for a camera sample'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_unknown_open_effect_cannot_authorize_repeated_release(mode,monkeypatch):
    import capx.execution.grounded_feedback as module
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='holding_verified';m.closed=True;m.lifted=True;m.held_object='red cube'
    support=m.graph.add('estimate',{'object':'green cube'})
    m.last_place={'skill':'place','inputs':{'support_surface':support}}
    monkeypatch.setattr(m,'_held_surface',lambda:surfaces['red'])
    monkeypatch.setattr(m,'_support_surface',lambda *a,**kw:surfaces['green'])
    monkeypatch.setattr(module,'place_metrics',lambda *a:{'status':'PASS','reason':'test_public_geometry_safe'})
    def release_then_error(name,args,kwargs):
        if name=='open_gripper':
            calls.append((name,args));state['robot_cartesian_pos'][7]=1.
            state['robot_gripper_joint_pos'][:]=[.04,-.04]
            raise ValueError('post-release validation failure')
        return raw(name,args,kwargs)
    with pytest.raises(ExecutionInterrupted,match='api_contract_error'):
        m.invoke('open_gripper',[],{},release_then_error)
    assert m.inflight['status']=='unknown_partial_effects'
    m.begin_program('new actor after release uncertainty',1)
    before=len(calls)
    try:m.invoke('open_gripper',[],{},raw)
    except ExecutionInterrupted:pass
    assert len(calls)==before,'Unknown release was executed again without state reconciliation'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_unknown_motion_effect_expires_old_robot_source_before_new_actor(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='holding_verified';m.closed=True;m.lifted=True;m.held_object='red cube'
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['lift',{'robot':robot['id']}],{},raw)
    def move_then_error(name,args,kwargs):
        result=raw(name,args,kwargs)
        if name=='goto_pose':raise ValueError('post-motion response validation failed')
        return result
    with pytest.raises(ExecutionInterrupted,match='api_contract_error'):
        m.invoke('run_plan',[plan],{},move_then_error)
    assert m.inflight['status']=='unknown_partial_effects'
    assert abs(state['robot_cartesian_pos'][2]-robot['position'][2])>.09
    m.begin_program('new actor after possibly completed motion',1)
    with pytest.raises(ExecutionInterrupted):m.invoke('recall_result',[robot['id']],{},raw)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_unknown_effect_stays_terminal_after_successful_public_capture(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='approached';m.approached_object='red cube'
    def error(name,args,kwargs):
        raw(name,args,kwargs)
        raise ValueError('not a completion receipt')
    with pytest.raises(ExecutionInterrupted):m.invoke('close_gripper',[],{},error)
    m.begin_program('fresh actor',1)
    assert m.phase=='reconciliation_required'
    assert m.terminal_reason=='unknown_physical_effect_terminal_stop'
    assert m.reconciliation_required['api']=='close_gripper'
    assert not m.call_history
    assert m.graph.nodes[m.inflight['id']]['state']!='done'
    before=len(calls)
    for name in ('close_gripper','open_gripper','home_pose','goto_pose'):
        with pytest.raises(ExecutionInterrupted,match='unknown_physical_effect_terminal_stop'):
            m.invoke(name,[],{},raw)
    assert len(calls)==before


@pytest.mark.parametrize('mode',['G0','G1'])
def test_returned_close_commits_unverified_holding_before_camera_parsing(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.phase='approached';m.approached_object='red cube'
    geometry=m.geometry_fn;fail=[False]
    def broken(obs):
        if fail[0]:fail[0]=False;raise ValueError('camera parse failed')
        return geometry(obs)
    m.geometry_fn=broken
    def fn(name,args,kwargs):
        result=raw(name,args,kwargs)
        if name=='close_gripper':fail[0]=True
        return result
    with pytest.raises(ExecutionInterrupted):m.invoke('close_gripper',[],{},fn)
    assert m.closed and not m.lifted and m.phase=='holding_unverified'
    assert m.held_object=='red cube' and m.reconciliation_required is None
    assert m.close_tcp_z==state['robot_cartesian_pos'][2]
    assert m.lift_attempts==0 and m.inflight['status']=='returned'
    m.begin_program('recover observation only',1)
    assert m.phase=='holding_unverified' and m.closed


def test_perception_error_does_not_invent_a_partial_physical_effect():
    m,state,surfaces,calls,raw=fixture()
    def error(name,args,kwargs):raise ValueError('bad perception response')
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_object',['red cube'],{},error)
    assert m.reconciliation_required is None and m.terminal_reason is None
    assert m.robot_epoch==0
