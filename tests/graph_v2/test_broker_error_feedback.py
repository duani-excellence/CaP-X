import pytest
from test_grounded_monitor import fixture

@pytest.mark.parametrize('mode',['G0','G1'])
def test_policy_error_rederived_from_source_without_forwarding_private_stderr(mode):
    m,s,v,c,f=fixture(mode=mode)
    m.begin_program('_id=1\nprint(_id)',1)
    m.end_block_failure('policy_violation','Authorization: private-marker-must-not-leak')
    feedback=m.summary()
    error=feedback['recovery_preconditions']['program_error']
    assert error['source']=='sandbox_static_validation'
    assert error['message']=='Forbidden name: _id'
    assert 'private-marker' not in str(feedback)
    assert not c

@pytest.mark.parametrize('mode',['G0','G1'])
def test_valid_source_never_turns_arbitrary_stderr_into_static_diagnosis(mode):
    m,s,v,c,f=fixture(mode=mode);m.begin_program('print(1)',1)
    m.end_block_failure('policy_violation','Forbidden name: hidden_private_marker')
    feedback=m.summary()
    assert not feedback['recovery_preconditions'].get('program_error')
    assert 'hidden_private_marker' not in str(feedback)
    assert not c

from capx.execution.bounded_ik import IKConvergenceError
from capx.execution.sandbox import ExecutionInterrupted

@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('approach',[0.,.1])
def test_typed_broker_ik_cause_proves_no_motion(mode,approach):
    m,s,v,c,f=fixture(mode,task='lift');m.ik_success_count=lambda:7
    m._phase_guard('approach',True,'fixture_authorization')
    def reject(*args):
        try:raise IKConvergenceError('first_segment_rejected',successful_calls_before=7,call_generation=8)
        except IKConvergenceError as error:raise ConnectionError('IKConvergenceError: first_segment_rejected') from error
    with pytest.raises(ExecutionInterrupted,match='pre_motion_rejected'):
        m._tracked(reject)('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':approach})
    assert m.inflight['status']=='not_executed'
    assert not c and not m.call_history and m.reconciliation_required is None and m.terminal_reason is None

@pytest.mark.parametrize('mode',['G0','G1'])
def test_wrapped_second_segment_failure_stays_quarantined(mode):
    m,s,v,c,f=fixture(mode,task='lift');counter=[7];m.ik_success_count=lambda:counter[0]
    m._phase_guard('approach',True,'fixture_authorization')
    def reject(*args):
        counter[0]+=1;c.append('first_segment_returned')
        try:raise IKConvergenceError('second_segment_rejected',successful_calls_before=8,call_generation=9)
        except IKConvergenceError as error:raise ConnectionError('wrapped') from error
    with pytest.raises(ConnectionError):
        m._tracked(reject)('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':.1})
    assert c==['first_segment_returned'] and m.inflight['status']=='unknown_partial_effects'
    assert m.terminal_reason=='unknown_physical_effect_terminal_stop'

@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('cause',['none','unrelated','unproven'])
def test_error_text_cannot_spoof_motion_proof(mode,cause):
    m,s,v,c,f=fixture(mode,task='lift');m.ik_success_count=lambda:7
    m._phase_guard('approach',True,'fixture_authorization')
    def reject(*args):
        error=ConnectionError('IKConvergenceError: first_segment_rejected')
        if cause=='unrelated':error.__cause__=RuntimeError('not a typed solver receipt')
        if cause=='unproven':
            error.__cause__=IKConvergenceError('unproven',successful_calls_before=7)
            error.__cause__.checked_no_segment_execution=False
        raise error
    with pytest.raises(ConnectionError):
        m._tracked(reject)('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':0})
    assert m.inflight['status']=='unknown_partial_effects' and m.terminal_reason

@pytest.mark.parametrize('mode',['G0','G1'])
def test_syntax_feedback_is_static_and_does_not_leak_stderr(mode):
    m,s,v,c,f=fixture(mode);m.begin_program('for x in:',1)
    m.end_block_failure('invalid_program','Authorization: never-share-this')
    feedback=m.summary();error=feedback['recovery_preconditions']['program_error']
    assert error['source']=='sandbox_static_validation' and error['error_type']=='SyntaxError'
    assert error['line']==1 and len(error['message'])<=256
    assert 'never-share-this' not in str(feedback) and not c
