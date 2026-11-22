import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def held(mode='G1',phase='holding_verified'):
    m,s,v,c,f=fixture(mode)
    old=m.invoke('observe_object',['red cube'],{},f)
    m.phase=phase;m.closed=True;m.lifted=phase=='holding_verified';m.held_object='red cube'
    m.baseline={'top_z':0.};m.approached_object='red cube'
    m.attachment={'old':True};m.attachment_id=m.graph.add('verification_fact',{'type':'old_attachment'})
    m._phase_guard('lift',True,'fixture_authorization')
    c.clear()
    return m,s,c,f,old


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('phase',['holding_verified','holding_unverified'])
def test_empty_jaws_seen_before_action_block_without_any_motion_or_replay(mode,phase):
    m,s,c,f,old=held(mode,phase);history=list(m.call_history)
    m.local_attempts=3;m.visibility_probes=2
    s['robot_cartesian_pos'][7]=0.
    with pytest.raises(ExecutionInterrupted,match='empty_jaws_before_action'):
        m._move([.5,0,.17],[0,0,1,0],0.,f)
    assert not c and m.call_history==history
    assert not m.closed and m.held_object is None and m.phase=='ready'
    assert not m.graph.nodes[old['id']]['valid']
    assert m.local_attempts==3 and m.visibility_probes==2
    assert all(m.graph.nodes[n]['state']=='done' for n in history)
    events=[e for e in m.graph.events if e['action']=='public_grasp_pre_action_rejection']
    assert len(events)==1 and events[0]['api']=='goto_pose'
    assert events[0]['physical_action_issued'] is False
    m.begin_program('recovery',1)
    assert m.summary()['held_object'] is None


def test_missing_current_gripper_state_cannot_authorize_a_holding_motion():
    m,s,c,f,old=held()
    s['robot_cartesian_pos']=s['robot_cartesian_pos'][:7]
    with pytest.raises(ExecutionInterrupted,match='current_gripper_state_required_before_action'):
        m._move([.5,0,.17],[0,0,1,0],0.,f)
    assert not c


def test_nonfinite_public_state_is_rejected_before_any_motion():
    m,s,c,f,old=held();s['robot_cartesian_pos'][7]=np.nan
    # Existing graph serialization already rejects this malformed input while
    # capturing it, before the action-boundary guard can receive a state.
    with pytest.raises(ValueError,match='JSON compliant'):
        m._move([.5,0,.17],[0,0,1,0],0.,f)
    assert not c


def test_empty_nonholding_gripper_can_still_open_for_a_new_grasp():
    m,s,v,c,f=fixture();s['robot_cartesian_pos'][7]=0.
    m.invoke('open_gripper',[],{},f)
    assert [x[0] for x in c]==['open_gripper']


def test_lost_grasp_during_returned_motion_keeps_the_motion_as_completed_history():
    m,s,c,f,old=held();before=list(m.call_history)
    def slip(name,args,kw):
        result=f(name,args,kw)
        if name=='goto_pose':s['robot_cartesian_pos'][7]=0.
        return result
    with pytest.raises(ExecutionInterrupted,match='lost_grasp_empty_jaws'):
        m._move([.5,0,.17],[0,0,1,0],0.,slip)
    assert [x[0] for x in c]==['goto_pose']
    assert len(m.call_history)==len(before)+1
    assert m.graph.nodes[m.call_history[-1]]['state']=='done'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_empty_hold_before_release_clears_state_before_geometry_check(mode):
    m,s,c,f,old=held(mode);s['robot_cartesian_pos'][7]=0.
    m.last_place={'skill':'place','inputs':{}}
    with pytest.raises(ExecutionInterrupted,match='empty_jaws_before_action'):
        m.invoke('open_gripper',[],{},f)
    assert not c and m.phase=='ready' and m.held_object is None
    assert m.last_grasp_loss['object']=='red cube'
    # An interrupted actor has no permission to finish its old release. A new
    # actor can explicitly open the already-empty jaws and plan a fresh grasp.
    m.begin_program('recover',1)
    m.invoke('open_gripper',[],{},f)
    assert [x[0] for x in c]==['open_gripper']
