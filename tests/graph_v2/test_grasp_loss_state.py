import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture

@pytest.mark.parametrize('mode',['G0','G1'])
def test_public_empty_jaws_clear_current_identity_but_preserve_history_and_budget(mode):
    m,s,v,c,f=fixture(mode)
    old=m.invoke('observe_object',['red cube'],{},f)
    m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    m.approached_object='red cube';m.baseline={'top_z':0.};m.attachment={'old':True}
    m.attachment_id=m.graph.add('verification_fact',{'type':'old_attachment'})
    m.last_place={'skill':'place'};m.local_attempts=3;m.visibility_probes=2
    m._phase_guard('lift',True,'test_authorization')
    def slip_after_motion(name,args,kw):
        result=f(name,args,kw)
        if name=='goto_pose':s['robot_cartesian_pos'][7]=0.
        return result
    with pytest.raises(ExecutionInterrupted,match='lost_grasp_empty_jaws'):
        m._move([.5,0,.17],[0,0,1,0],0.,slip_after_motion)
    assert m.phase=='ready' and not m.closed and not m.lifted and m.held_object is None
    assert m.attachment is None and m.last_place is None and m.baseline is None
    assert m.approached_object is None
    assert not m.graph.nodes[old['id']]['valid']
    completed=list(m.call_history)
    assert completed and [name for name,args in c]==['get_object_pose','goto_pose']
    m.begin_program('recovery',1)
    summary=m.summary()
    assert summary['held_object'] is None and summary['grasp_active'] is False
    assert summary['last_grasp_loss']['object']=='red cube'
    assert summary['recovery_preconditions']['next_action']=='open_empty_gripper_then_fresh_observation_and_regrasp'
    assert m.local_attempts==3 and m.visibility_probes==2 and m.call_history==completed
    assert all(m.graph.nodes[n]['state']=='done' for n in completed)

def test_new_actor_does_not_clear_an_existing_verified_grasp():
    m,s,v,c,f=fixture();m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    m.begin_program('recovery',1)
    summary=m.summary()
    assert summary['held_object']=='red cube' and summary['grasp_active'] is True
    assert summary['lift_verified'] is True and summary['last_grasp_loss'] is None

def test_released_object_is_history_not_reported_as_currently_held():
    m,s,v,c,f=fixture();m.phase='released';m.closed=False;m.lifted=False;m.held_object='red cube'
    summary=m.summary()
    assert summary['held_object'] is None and summary['grasp_active'] is False
    assert summary['last_manipulated_object']=='red cube'

def test_empty_close_cannot_leave_an_approach_or_placement_permission():
    m,s,v,c,f=fixture();m.phase='approached';m.approached_object='red cube'
    s['robot_cartesian_pos'][7]=0.
    with pytest.raises(ExecutionInterrupted,match='empty_grasp'):m.invoke('close_gripper',[],{},f)
    assert m.held_object is None and m.approached_object is None and not m.closed
    assert m.last_grasp_loss['object']=='red cube'
    assert [name for name,args in c]==['close_gripper']
    assert m.call_history and m.graph.nodes[m.call_history[-1]]['state']=='done'

def test_loss_during_precision_place_aborts_before_release_and_preserves_motion_fact():
    m,s,v,c,f=fixture();m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    m._phase_guard('place',True,'test_authorization')
    def slip_after_motion(name,args,kw):
        result=f(name,args,kw)
        if name=='goto_pose':s['robot_cartesian_pos'][7]=0.
        return result
    with pytest.raises(ExecutionInterrupted,match='slipped_or_empty_during_precise_place'):
        m._precise_place([.55,0,.10],[0,0,1,0],slip_after_motion)
    assert not m.closed and m.held_object is None and m.last_place is None
    assert m.last_grasp_loss['reason']=='slipped_or_empty_during_precise_place'
    assert len(c)==1 and c[0][0]=='goto_pose'
