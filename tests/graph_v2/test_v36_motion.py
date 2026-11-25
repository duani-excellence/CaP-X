import numpy as np
import pytest
from capx.execution.trajectory_execution import execute_waypoints


def test_executes_every_waypoint_and_checks_fresh_tracking():
    q=np.zeros(7);sent=[];events=[]
    def read():return {'joints':q.copy(),'source':len(events)}
    def send(value):q[:]=value;sent.append(value.copy())
    result=execute_waypoints(np.array([[0]*7,[.01]*7,[.02]*7]),read,lambda *a:{'status':'PASS'},send,lambda kind,**kw:events.append(kw))
    assert len(sent)==2 and np.allclose(sent[0],.01) and np.allclose(sent[1],.02)
    assert result['status']=='EFFECT_VERIFIED'


def test_stale_start_or_unknown_path_never_dispatches():
    sent=[];read=lambda:{'joints':np.zeros(7),'source':1}
    with pytest.raises(ValueError,match='start'):
        execute_waypoints(np.array([[.1]*7,[.2]*7]),read,lambda *a:{'status':'PASS'},sent.append,lambda *a,**kw:None)
    with pytest.raises(ValueError,match='admission'):
        execute_waypoints(np.array([[0]*7,[.01]*7]),read,lambda *a:{'status':'UNKNOWN'},sent.append,lambda *a,**kw:None)
    assert not sent


def test_no_return_equals_effect_verification():
    with pytest.raises(ValueError,match='tracking'):
        execute_waypoints(np.array([[0]*7,[.05]*7]),lambda:{'joints':np.zeros(7),'source':1},
            lambda *a:{'status':'PASS'},lambda q:None,lambda *a,**kw:None)


def test_fixed_assembly_exclusions_follow_joints_not_task_or_home_pose():
    from types import SimpleNamespace as S
    from capx.integrations.motion.collision_audit import fixed_assembly_pairs
    u=S(link_map={x:None for x in ['a','b','c','d']},joint_map={
        'ab':S(type='fixed',parent='a',child='b'),
        'bc':S(type='fixed',parent='b',child='c'),
        'cd':S(type='revolute',parent='c',child='d')})
    assert set(fixed_assembly_pairs(u))=={('a','b'),('a','c'),('b','c')}
