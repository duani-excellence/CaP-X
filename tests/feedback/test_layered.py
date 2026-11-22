import numpy as np
import pytest
from capx.execution.feedback import FeedbackMonitor, release_check, public_observation
from capx.execution.sandbox import SandboxProcess, ExecutionInterrupted


def evidence(dx=0., gap=.006):
    return {'red': {'center': [dx,0,.1], 'lo':[dx-.02,-.02,.05+gap], 'hi':[dx+.02,.02,.09+gap], 'pixels':100},
            'green': {'center':[0,0,.025], 'lo':[-.025,-.025,0], 'hi':[.025,.025,.05], 'pixels':100}}

def obs():
    return {'robot_cartesian_pos':np.array([.5,0,.2,0,0,1,0,.5]), 'robot_joint_pos':np.zeros(8)}

@pytest.mark.parametrize('dx,gap,status',[(0,.006,'PASS'),(.025,.006,'FAIL'),(0,.09,'FAIL')])
def test_release_geometry(dx,gap,status):
    assert release_check(evidence(dx,gap))['status']==status

def test_missing_visibility_uncertain():
    assert release_check({'red':None,'green':None})['status']=='UNCERTAIN'

def test_hidden_fields_removed():
    raw={**obs(),'cube_poses':'SECRET','raw_success':True,'env':'SECRET','full_prompt':'SECRET'}
    assert set(public_observation(raw))=={'robot_joint_pos','robot_cartesian_pos'}

def test_original_mode_has_no_monitoring():
    m=FeedbackMonitor('A',lambda:pytest.fail('must not observe'))
    assert m.invoke('open_gripper',[],{},lambda *a:42)==42
    assert not m.events

def test_initial_open_allowed():
    m=FeedbackMonitor('D',obs,evidence_fn=lambda _:evidence())
    calls=[];m.invoke('open_gripper',[],{},lambda *a:calls.append(a[0]))
    assert calls==['open_gripper']

def test_uncertain_release_cannot_execute():
    m=FeedbackMonitor('D',obs,evidence_fn=lambda _: {})
    m.closed=True;calls=[]
    with pytest.raises(ExecutionInterrupted):
        m.invoke('open_gripper',[],{},lambda *a:calls.append(a[0]))
    assert 'open_gripper' not in calls

def test_bounded_alignment_then_release():
    states=iter([evidence(.025,.08),evidence(0,.006),evidence(0,.006)])
    current=obs()
    m=FeedbackMonitor('D',lambda:current,evidence_fn=lambda _:next(states),max_local=2)
    m.closed=True;calls=[]
    def execute(name,args,kwargs):
        calls.append(name)
        if name=='goto_pose':current['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
    m.invoke('open_gripper',[],{},execute)
    assert calls==['goto_pose','open_gripper']
    assert m.recoveries==1

def test_alignment_exhaustion_blocks_release():
    current=obs()
    m=FeedbackMonitor('D',lambda:current,evidence_fn=lambda _:evidence(.025,.08),max_local=2)
    m.closed=True;calls=[]
    def execute(name,args,kwargs):
        calls.append(name)
        if name=='goto_pose':current['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
    with pytest.raises(ExecutionInterrupted):
        m.invoke('open_gripper',[],{},execute)
    assert calls==['goto_pose','goto_pose']

def test_repeated_estimate_not_independent_pass():
    m=FeedbackMonitor('B',obs,evidence_fn=lambda _:evidence())
    value=(np.array([.2,0,.1]),np.array([1,0,0,0]),np.ones(3)*.04)
    calls=[]
    m.invoke('get_object_pose',['red cube',True],{},lambda *a:(calls.append(1) or value))
    assert len(calls)==2
    assert m.events[-1]['status']=='UNCERTAIN'

def test_control_failure_retry_bounded():
    m=FeedbackMonitor('C',obs,evidence_fn=lambda _:evidence())
    calls=[]
    with pytest.raises(ExecutionInterrupted):
        m.invoke('goto_pose',[[1,0,.2],[0,0,1,0]],{},lambda *a:calls.append(1))
    assert len(calls)==2

def test_actor_cannot_catch_broker_interruption():
    calls=[]
    def callback(name,args,kwargs):
        calls.append(name)
        if name=='open_gripper':raise ExecutionInterrupted('release blocked')
    actor=SandboxProcess({'open_gripper':None,'goto_pose':None},callback=callback)
    try:
        r=actor.run('try:\n    open_gripper()\nexcept Exception:\n    goto_pose()\n',timeout=5)
        assert r['status']=='interrupted'
        assert calls==['open_gripper']
        assert actor.pid is None
        assert actor.run('goto_pose()',timeout=5)['ok']
    finally:actor.close()

def test_structured_feedback_contains_only_public_evidence():
    m=FeedbackMonitor('E',lambda:{**obs(),'cube_poses':'SECRET','raw_success':True},evidence_fn=lambda _:evidence())
    m.invoke('close_gripper',[],{},lambda *a:None)
    assert 'SECRET' not in str(m.summary())
    assert 'raw_success' not in str(m.summary())

def test_empty_jaws_after_lift_stop_before_place_and_allow_regrasp():
    current=obs();current['robot_cartesian_pos'][7]=.01
    m=FeedbackMonitor('D',lambda:current,evidence_fn=lambda _: {})
    m.closed=True;m.close_state=obs();m.close_evidence={}
    calls=[]
    with pytest.raises(ExecutionInterrupted,match='lost_grasp'):
        m.invoke('goto_pose',[current['robot_cartesian_pos'][:3],current['robot_cartesian_pos'][3:7]],{},lambda *a:calls.append(a[0]))
    assert not m.closed
    m.invoke('open_gripper',[],{},lambda *a:calls.append(a[0]))
    assert calls==['goto_pose','open_gripper']

def test_empty_close_allows_next_attempt_to_open():
    current=obs();current['robot_cartesian_pos'][7]=.01
    m=FeedbackMonitor('D',lambda:current,evidence_fn=lambda _:evidence())
    with pytest.raises(ExecutionInterrupted):m.invoke('close_gripper',[],{},lambda *a:None)
    assert not m.closed

def test_feedback_summary_compacts_trace_without_losing_failure():
    m=FeedbackMonitor('E',obs,evidence_fn=lambda _:evidence())
    m.record('L1','tcp_tracking','FAIL','not_converged',{'position_error_m':.04,'joints':list(range(100))})
    summary=m.summary()
    assert summary['checks'][0]['metrics']['position_error_m']==.04
    assert 'joints' not in summary['checks'][0]['metrics']
    assert len(m.events[0]['metrics']['joints'])==100
