import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture

@pytest.mark.parametrize('mode',['G0','G1'])
def test_missing_support_reports_interruption_contract_and_observation_replanning(mode):
    m,s,v,c,f=fixture(mode);m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    m._support_surface=lambda name:None
    with pytest.raises(ExecutionInterrupted,match='current_object_surface_unavailable'):
        m.invoke('observe_surfaces',['green cube'],{},f)
    hint=m.summary()['recovery_preconditions']
    assert hint['missing_surface_object']=='green cube'
    assert hint['perception_failure_behavior']=='actor_interruption_not_None'
    assert hint['next_action']=='replan_observation_pose_before_retrying_support'
    assert not c and m.closed and m.held_object=='red cube'

def test_missing_held_surface_does_not_recommend_moving_as_verified_support_recovery():
    m,s,v,c,f=fixture();m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    m._held_surface=lambda:None
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['red cube'],{},f)
    hint=m.summary()['recovery_preconditions']
    assert hint['missing_surface_object']=='red cube'
    assert hint['next_action']!='replan_observation_pose_before_retrying_support'
    assert not c
