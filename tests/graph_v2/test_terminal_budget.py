import pytest
from test_grounded_monitor import fixture
from capx.execution.sandbox import ExecutionInterrupted


def test_exhausted_required_precision_correction_is_terminal():
    m,state,surfaces,calls,raw=fixture()
    m.phase='holding_verified';m.closed=True;m.held_object='red cube';m.lifted=True
    m.local_attempts=m.max_episode_local
    m._phase_guard('place',True,'test_current_place_permission')
    def no_convergence(name,args,kwargs):
        calls.append((name,args))
    with pytest.raises(ExecutionInterrupted,match='correction_budget'):
        m._precise_place([.65,0,.1],[0,0,1,0],no_convergence)
    assert m.terminal_reason=='local_recovery_budget_exhausted'
    assert m.summary()['remaining_local_attempts']==0


def test_exhausted_required_ordinary_control_retry_is_terminal():
    m,state,surfaces,calls,raw=fixture()
    m.local_attempts=m.max_episode_local
    m._phase_guard('approach',True,'test_current_approach_permission')
    with pytest.raises(ExecutionInterrupted):m._move([.7,0,.3],[0,0,1,0],0.,lambda *a:None)
    assert m.terminal_reason=='local_recovery_budget_exhausted'


def test_final_inconclusive_hold_probe_does_not_request_third_actor():
    m,state,surfaces,calls,raw=fixture()
    m.phase='holding_unverified';m.closed=True;m.held_object='red cube';m.lift_attempts=1;m.hold_probes=1
    # Robot moves but object does not: no public co-motion evidence.
    def robot_only(name,args,kwargs):
        state['robot_cartesian_pos'][:7]=args[0]+args[1]
    with pytest.raises(ExecutionInterrupted):m.invoke('verify_hold',[],{},robot_only)
    assert m.hold_probes==2
    assert m.terminal_reason=='hold_evidence_unrecoverable'
