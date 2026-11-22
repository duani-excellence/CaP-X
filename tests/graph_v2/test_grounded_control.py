"""Public-proprioception precision control, without simulator truth or services."""
import math
import numpy as np
import pytest
from capx.execution.grounded_control import precise_goto
from capx.execution.sandbox import ExecutionInterrupted


class Robot:
    def __init__(self, position=(0.,0.,.1), bias=(0.,0.,0.), frozen=False):
        self.tcp=np.r_[position,[1.,0.,0.,0.],.5]
        self.bias=np.asarray(bias);self.frozen=frozen;self.calls=[];self.reservations=0;self.events=[]
    def observe(self):
        return {'robot_cartesian_pos':self.tcp.copy(),'robot_joint_pos':np.zeros(7),
                'forbidden_ground_truth':object()}
    def execute(self,name,args,kwargs):
        self.calls.append((name,args,kwargs))
        if not self.frozen:self.tcp[:7]=np.r_[np.asarray(args[0])+self.bias,args[1]]
    def reserve(self):self.reservations+=1;return True
    def run(self,target=(.3,.2,.1),**kwargs):
        return precise_goto(target,[1,0,0,0],observe=self.observe,execute=self.execute,
            reserve_correction=self.reserve,emit=self.events.append,lift_verified=True,**kwargs)


def test_low_start_ascends_same_xy_then_moves_horizontally_then_descends():
    r=Robot();result=r.run()
    poses=[np.asarray(x[1][0]) for x in r.calls]
    assert result['status']=='PASS' and len(poses)==3
    assert np.allclose(poses[0],[0,0,.16])
    assert np.allclose(poses[1],[.3,.2,.16])
    assert np.allclose(poses[2],[.3,.2,.1])
    assert all(x[0]=='goto_pose' and x[2]=={'z_approach':0.} for x in r.calls)
    assert r.reservations==0
    assert [e['stage'] for e in r.events if e['check']=='precise_tcp_command']==['ascend','above_target','final']


def test_current_safe_height_is_preserved_without_extra_ascent():
    r=Robot(position=(0,0,.3));r.run()
    assert len(r.calls)==2
    assert np.allclose(r.calls[0][1][0],[.3,.2,.3])
    assert np.allclose(r.calls[1][1][0],[.3,.2,.1])


def test_public_residual_compensation_is_measured_not_fixed():
    r=Robot(bias=(.008,-.004,.001))
    result=r.run(approach_height=0)
    assert len(r.calls)==2 and r.reservations==1 and result['corrections']==1
    assert np.allclose(r.calls[0][1][0],[.3,.2,.1])
    assert np.allclose(r.calls[1][1][0],[.292,.204,.099])
    assert result['position_error_m']<=.003
    assert all('forbidden_ground_truth' not in e for e in r.events)


def test_residual_budget_shared_across_stages():
    r=Robot()
    count={'n':0}
    def execute(name,args,kwargs):
        count['n']+=1;r.calls.append((name,args,kwargs))
        # Each new nominal target initially misses; its correction then succeeds.
        bias=np.array([.008 if count['n']<=2 else .016,0,0])
        r.tcp[:7]=np.r_[np.asarray(args[0])+bias,args[1]]
    r.execute=execute
    with pytest.raises(ExecutionInterrupted,match='correction_limit'):
        r.run(max_corrections=1)
    assert r.reservations==1
    assert len(r.calls)==3  # ascend, its correction, above-target; no final descent
    assert not any(e.get('stage')=='final' for e in r.events)


def test_xy_failure_blocks_descent_even_if_motion_callback_returns():
    r=Robot(position=(0,0,.3),frozen=True)
    with pytest.raises(ExecutionInterrupted):r.run(max_corrections=0)
    assert len(r.calls)==1 and r.calls[0][1][0][2]==.3
    assert not any(e.get('stage')=='final' for e in r.events)


def test_budget_denial_prevents_correction_call():
    r=Robot(bias=(.01,0,0));r.reserve=lambda:False
    with pytest.raises(ExecutionInterrupted,match='budget'):
        r.run(approach_height=0)
    assert len(r.calls)==1


def test_cumulative_compensation_bound_stops_stalled_robot():
    r=Robot(frozen=True)
    with pytest.raises(ExecutionInterrupted,match='offset_limit'):
        r.run(approach_height=0,max_command_offset=.02)
    assert len(r.calls)==1 and r.reservations==0


def test_orientation_error_is_corrected_from_public_pose():
    r=Robot()
    def execute(name,args,kwargs):
        r.calls.append((name,args,kwargs));r.tcp[:3]=args[0]
        q=np.asarray(args[1]);b=np.array([math.cos(.1),0,0,math.sin(.1)])
        r.tcp[3:7]=[q[0]*b[0]-q[3]*b[3],0,0,q[0]*b[3]+q[3]*b[0]]
    r.execute=execute
    result=r.run(approach_height=0)
    assert result['orientation_error_deg']<=5 and r.reservations==1
    assert r.calls[1][1][1][3]<0


def test_quaternion_sign_equivalence_does_not_trigger_correction():
    r=Robot(position=(.3,.2,.1));r.tcp[3:7]=[-1,0,0,0]
    result=r.run(approach_height=0)
    assert result['status']=='PASS' and not r.calls


@pytest.mark.parametrize('bad',[None,[0,0,0,0,0,0,0,0],[0,0,float('nan'),1,0,0,0,0]])
def test_invalid_public_tcp_fails_without_motion(bad):
    r=Robot();r.observe=lambda:{'robot_cartesian_pos':bad}
    with pytest.raises(ExecutionInterrupted):r.run()
    assert not r.calls


def test_lift_precondition_mandatory_before_any_motion():
    r=Robot()
    with pytest.raises(ExecutionInterrupted,match='lift_verified'):
        precise_goto([1,0,0],[1,0,0,0],observe=r.observe,execute=r.execute,reserve_correction=r.reserve)
    assert not r.calls


def test_executor_interrupt_is_not_swallowed_or_retried():
    r=Robot()
    def execute(*a):raise ExecutionInterrupted('broker_stop')
    r.execute=execute
    with pytest.raises(ExecutionInterrupted,match='broker_stop'):r.run()
    assert r.reservations==0
