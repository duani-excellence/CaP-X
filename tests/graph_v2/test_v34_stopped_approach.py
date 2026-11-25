from types import SimpleNamespace
import numpy as np
import pytest


def state(phase='approach_stopped',width=.08,held=None,stalled=True):
    return SimpleNamespace(phase=phase,held=held,stalled={'object':'unseen'} if stalled else None,
        scene=SimpleNamespace(obs={'robot_gripper_joint_pos':np.array([width/2,-width/2])}))


def test_known_empty_approach_stop_may_propose_but_has_no_permission():
    from capx.execution.open_motion import empty_motion_scope
    result=empty_motion_scope(state())
    assert result['may_propose'] and result['permission'] is False
    assert result['reconcile_stalled']


@pytest.mark.parametrize('s',[state(width=.03),state(stalled=False),state(phase='closed_unverified'),state(held={'object':'unseen'})])
def test_uncertain_or_closed_grasp_cannot_use_empty_escape(s):
    from capx.execution.open_motion import empty_motion_scope
    assert not empty_motion_scope(s)['may_propose']


def test_missing_skill_binding_is_a_structured_proposal_rejection():
    from capx.execution.open_recovery import OpenRecoveryAPIs
    r=SimpleNamespace(skills=state(),library=[{'skill':'retreat_empty','object':'unseen'}])
    tool=OpenRecoveryAPIs(r,None)
    p=tool.plan_skill('retreat_empty')
    assert p['status']=='REJECTED' and p['required_params']==['object'] and 'id' not in p
