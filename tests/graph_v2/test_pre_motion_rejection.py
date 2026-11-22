import pytest
from capx.execution.graph_feedback import PreMotionRejected
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def test_proven_pre_motion_rejection_is_not_unknown_physical_effect():
    m,state,surfaces,calls,raw=fixture(task='lift')
    obj=m.invoke('observe_object',['red cube'],{},raw)
    grasp=m.invoke('observe_grasp',['red cube'],{},raw)
    plan=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{'params':{'z_approach':0.}},raw)
    def rejected(name,args,kw):
        if name=='goto_pose':raise PreMotionRejected('ik_not_converged_before_segment')
        return raw(name,args,kw)
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},rejected)
    assert m.reconciliation_required is None and m.terminal_reason is None
    assert m.inflight['status']=='not_executed'
    assert not m.graph.nodes[m.inflight['id']]['valid']
    assert m.inflight['id'] not in m.call_history
    assert not m.closed


def test_unknown_callback_remains_quarantined():
    m,state,surfaces,calls,raw=fixture()
    m.phase='approached';m.approached_object='red cube'
    def uncertain(*a,**kw):raise RuntimeError('failure after physical effect')
    with pytest.raises(RuntimeError):m.invoke('close_gripper',[],{},uncertain)
    assert m.reconciliation_required and m.terminal_reason


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('style',['keyword','positional','default'])
def test_grounded_single_segment_ik_rejection_preserves_current_state(mode,style):
    from capx.execution.bounded_ik import IKConvergenceError
    m,state,surfaces,calls,raw=fixture(mode,task='lift')
    m._phase_guard('approach',True,'fixture_current_authorization')
    old_robot_epoch=m.robot_epoch
    args=[[.5,0,.2],[0,0,1,0]];kwargs={}
    if style=='keyword':kwargs['z_approach']=0.
    if style=='positional':args.append(0.)
    def reject(name,args,kw):
        raise IKConvergenceError('ik_unconverged_before_motion',call_generation=3,successful_calls_before=2,iterations=8)
    with pytest.raises(ExecutionInterrupted,match='pre_motion_rejected'):
        m._tracked(reject)('goto_pose',args,kwargs)
    assert m.inflight['status']=='not_executed'
    assert m.robot_epoch==old_robot_epoch and m.phase=='ready'
    assert not m.call_history and m.reconciliation_required is None
    assert m.terminal_reason is None and not calls


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('style',['keyword','positional'])
def test_multi_segment_ik_failure_after_first_move_stays_unknown(mode,style):
    from capx.execution.bounded_ik import IKConvergenceError
    m,state,surfaces,calls,raw=fixture(mode,task='lift')
    m._phase_guard('approach',True,'fixture_current_authorization')
    args=[[.5,0,.2],[0,0,1,0]];kwargs={}
    if style=='keyword':kwargs['z_approach']=.1
    else:args.append(.1)
    def first_segment_then_fail(name,args,kw):
        # Mirrors the existing API's two segment boundary: completed first move,
        # then the second solve rejects before its own motion.
        state['robot_cartesian_pos'][2]=.3;calls.append(('first_segment',[]))
        raise IKConvergenceError('ik_unconverged_before_motion',call_generation=2,successful_calls_before=1,iterations=8)
    with pytest.raises(IKConvergenceError):m._tracked(first_segment_then_fail)('goto_pose',args,kwargs)
    assert len(calls)==1 and state['robot_cartesian_pos'][2]==.3
    assert m.inflight['status']=='unknown_partial_effects'
    assert m.reconciliation_required and m.terminal_reason=='unknown_physical_effect_terminal_stop'
    assert m.phase=='reconciliation_required' and not m.call_history


def test_ik_exception_without_current_segment_proof_is_not_treated_as_zero_effect():
    from capx.execution.bounded_ik import IKConvergenceError
    m,state,surfaces,calls,raw=fixture(task='lift')
    m._phase_guard('approach',True,'fixture_current_authorization')
    error=IKConvergenceError('no_proof');error.checked_no_segment_execution=False
    def reject(*args):raise error
    with pytest.raises(IKConvergenceError):m._tracked(reject)('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':0})
    assert m.inflight['status']=='unknown_partial_effects' and m.terminal_reason


def test_home_pose_cannot_claim_single_segment_goto_proof():
    from capx.execution.bounded_ik import IKConvergenceError
    m,state,surfaces,calls,raw=fixture(task='lift')
    m._phase_guard('home_pose',True,'fixture_current_authorization')
    def reject(*args):raise IKConvergenceError('not_a_goto_segment')
    with pytest.raises(IKConvergenceError):m._tracked(reject)('home_pose',[],{})
    assert m.inflight['status']=='unknown_partial_effects' and m.terminal_reason


@pytest.mark.parametrize('mode',['G0','G1'])
def test_trusted_counter_proves_first_segment_of_multisegment_goto_unexecuted(mode):
 from capx.execution.bounded_ik import IKConvergenceError
 m,state,surfaces,calls,raw=fixture(mode,task='lift');m.ik_success_count=lambda:7
 m._phase_guard('approach',True,'fixture_current_authorization')
 def reject(*args):raise IKConvergenceError('first_segment_rejected',successful_calls_before=7,call_generation=8)
 with pytest.raises(ExecutionInterrupted,match='pre_motion_rejected'):
  m._tracked(reject)('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':.1})
 assert m.inflight['status']=='not_executed' and m.reconciliation_required is None
 assert m.terminal_reason is None and not calls


@pytest.mark.parametrize('mode',['G0','G1'])
def test_counter_detects_returned_first_segment_before_second_ik_failure(mode):
 from capx.execution.bounded_ik import IKConvergenceError
 m,state,surfaces,calls,raw=fixture(mode,task='lift');counter=[7];m.ik_success_count=lambda:counter[0]
 m._phase_guard('approach',True,'fixture_current_authorization')
 def reject(*args):
  counter[0]+=1;calls.append('completed_first_segment')
  raise IKConvergenceError('second_segment_rejected',successful_calls_before=8,call_generation=9)
 with pytest.raises(IKConvergenceError):m._tracked(reject)('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':.1})
 assert calls==['completed_first_segment'] and m.inflight['status']=='unknown_partial_effects'
 assert m.terminal_reason=='unknown_physical_effect_terminal_stop'
