"""IK must satisfy public configured FK before the original API can move."""
import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted

class FakeFK:
 urdf_sha256='static-public-model';joint_names=['x','y','z']
 def validate(self,q):
  q=np.asarray(q,float)
  if q.shape!=(3,) or not np.isfinite(q).all() or np.any(abs(q)>2):raise ValueError('invalid joints')
  return q.copy()
 def pose(self,q):return np.r_[1.,0,0,0,q]

def build(solver,**kw):
 from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
 return BoundedIK(solver,kinematics=FakeFK(),budget=kw.pop('budget',IKSolveBudget(24)),**kw)

def test_only_returns_a_fk_valid_solution_after_in_memory_iterations():
 calls=[];events=[];answers=[np.array([.7,0,0]),np.array([.95,0,0]),np.array([1.,0,0])]
 def solve(target_pose_wxyz_xyz,prev_cfg=None):calls.append((target_pose_wxyz_xyz.copy(),prev_cfg.copy()));return answers[len(calls)-1]
 q0=np.zeros(3);wrapper=build(solve,emit=events.append,observe_joints=lambda:[.1,.2,.3],calibration_hash='cal')
 out=wrapper(np.r_[1,0,0,0,1,0,0],prev_cfg=q0)
 assert out==pytest.approx([1,0,0]) and len(calls)==3
 assert calls[1][1]==pytest.approx(answers[0]) and calls[2][1]==pytest.approx(answers[1])
 assert q0==pytest.approx([0,0,0])
 assert all(e['public_joints']==[.1,.2,.3] and e['calibration_hash']=='cal' for e in events)
 assert events[-1]['status']=='PASS' and events[-1]['position_error_m']<=.001


def test_unconverged_never_returns_and_respects_eight_iteration_cap():
 calls=[]
 def solver(**kw):calls.append(kw);return np.zeros(3)
 fn=build(solver)
 with pytest.raises(ExecutionInterrupted,match='ik_unconverged_before_motion'):fn(np.r_[1,0,0,0,1,0,0])
 assert len(calls)==8


def test_episode_budget_is_shared_across_wrappers_and_checked_before_solver():
 from capx.execution.bounded_ik import IKSolveBudget
 budget=IKSolveBudget(1);calls=[]
 fn=lambda **kw:calls.append(kw) or np.zeros(3)
 first=build(fn,budget=budget);second=build(fn,budget=budget)
 first(np.r_[1,0,0,0,0,0,0])
 with pytest.raises(ExecutionInterrupted,match='ik_solve_budget'):second(np.r_[1,0,0,0,0,0,0])
 assert len(calls)==1 and budget.calls==1

@pytest.mark.parametrize('q',[np.array([np.nan,0,0]),np.array([3.,0,0]),np.zeros(2)])
def test_invalid_or_limit_violating_response_stops_without_retry(q):
 calls=[];fn=build(lambda **kw:calls.append(kw) or q)
 with pytest.raises(ExecutionInterrupted,match='ik_invalid_solution'):fn(np.r_[1,0,0,0,0,0,0])
 assert len(calls)==1


def test_provider_failure_is_not_automatically_retried_by_wrapper():
 calls=[]
 def fail(**kw):calls.append(kw);raise ConnectionError('offline')
 with pytest.raises(ConnectionError):build(fail)(np.r_[1,0,0,0,0,0,0])
 assert len(calls)==1


def test_original_api_cfg_and_motion_are_untouched_on_failure():
 class API:
  cfg=np.zeros(3);moves=[]
  def goto(self):
   self.cfg=self.solver(np.r_[1,0,0,0,1,0,0],prev_cfg=self.cfg)
   self.moves.append(self.cfg.copy())
 api=API();api.solver=build(lambda **kw:np.zeros(3))
 with pytest.raises(ExecutionInterrupted):api.goto()
 assert api.cfg==pytest.approx([0,0,0]) and api.moves==[]

@pytest.mark.parametrize('target',[np.zeros(7),np.r_[1,0,0,0,np.nan,0,0],np.zeros(6)])
def test_bad_targets_do_not_query_solver(target):
 calls=[];fn=build(lambda **kw:calls.append(kw) or np.zeros(3))
 with pytest.raises(ValueError):fn(target)
 assert not calls


def test_orientation_residual_also_blocks_even_when_position_matches():
 class Rotated(FakeFK):
  def pose(self,q):return np.r_[0.,0,0,1,q]
 from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
 fn=BoundedIK(lambda **kw:np.zeros(3),kinematics=Rotated(),budget=IKSolveBudget(8))
 with pytest.raises(ExecutionInterrupted,match='ik_unconverged_before_motion'):fn(np.r_[1,0,0,0,0,0,0])


def test_static_urdf_fk_and_joint_limits_without_model_download(tmp_path):
 from capx.execution.bounded_ik import URDFFK
 p=tmp_path/'simple.urdf';p.write_text('<robot name="test"><link name="panda_link0"/><link name="panda_hand"/><joint name="slide" type="prismatic"><parent link="panda_link0"/><child link="panda_hand"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint></robot>')
 fk=URDFFK(p);q=fk.validate([.5]);assert fk.pose(q)==pytest.approx([1,0,0,0,.5,0,0])
 with pytest.raises(ValueError):fk.validate([2.1])
 with pytest.raises(FileNotFoundError):URDFFK(tmp_path/'missing.urdf')


def test_rejection_scope_is_only_current_segment_and_preserves_prior_success_count():
 from capx.execution.bounded_ik import IKConvergenceError
 fn=build(lambda **kw:np.zeros(3))
 fn(np.r_[1,0,0,0,0,0,0])
 with pytest.raises(IKConvergenceError) as caught:fn(np.r_[1,0,0,0,1,0,0])
 error=caught.value
 assert error.checked_no_segment_execution is True
 assert error.call_generation==2 and error.successful_calls_before==1
 assert error.iterations==8
 assert not hasattr(error,'whole_api_unexecuted')


def test_budget_rejection_retains_current_call_metadata():
 from capx.execution.bounded_ik import IKConvergenceError,IKSolveBudget
 fn=build(lambda **kw:np.zeros(3),budget=IKSolveBudget(1))
 fn(np.r_[1,0,0,0,0,0,0])
 with pytest.raises(IKConvergenceError) as caught:fn(np.r_[1,0,0,0,1,0,0])
 assert caught.value.call_generation==2 and caught.value.successful_calls_before==1
 assert caught.value.iterations==0


def test_antipodal_quaternion_is_same_orientation():
 events=[];fn=build(lambda **kw:np.zeros(3),emit=events.append)
 assert fn(np.r_[-1,0,0,0,0,0,0])==pytest.approx([0,0,0])
 assert events[-1]['orientation_error_deg']==0


def test_stalled_reference_restarts_once_from_static_urdf_midpoint_within_same_budget():
 class RestartFK(FakeFK):
  def neutral_configuration(self):return np.array([.4,0,0])
 from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
 calls=[];events=[]
 def solve(**kw):
  prev=kw['prev_cfg'];calls.append(None if prev is None else prev.copy())
  return np.array([1.,0,0]) if prev is not None and np.allclose(prev,[.4,0,0]) else np.array([.2,0,0])
 budget=IKSolveBudget(8);fn=BoundedIK(solve,kinematics=RestartFK(),budget=budget,emit=events.append)
 assert fn(np.r_[1,0,0,0,1,0,0])==pytest.approx([1,0,0])
 assert len(calls)==3 and budget.calls==3 and calls[2]==pytest.approx([.4,0,0])
 assert events[-1]['reference_strategy']=='urdf_joint_midpoint_restart'
 assert events[-1]['reference_restarts']==1


def test_reference_restart_never_increases_eight_call_cap_or_repeats():
 class RestartFK(FakeFK):
  def neutral_configuration(self):return np.array([.4,0,0])
 from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
 calls=[]
 def solve(**kw):calls.append(kw['prev_cfg']);return np.array([.2,0,0])
 fn=BoundedIK(solve,kinematics=RestartFK(),budget=IKSolveBudget(8))
 with pytest.raises(ExecutionInterrupted):fn(np.r_[1,0,0,0,1,0,0])
 assert len(calls)==8
 assert sum(q is not None and np.allclose(q,[.4,0,0]) for q in calls)==1


def test_urdf_midpoint_reference_uses_only_static_joint_limits(tmp_path):
 from capx.execution.bounded_ik import URDFFK
 p=tmp_path/'simple.urdf';p.write_text('<robot name="test"><link name="panda_link0"/><link name="panda_hand"/><joint name="slide" type="prismatic"><parent link="panda_link0"/><child link="panda_hand"/><axis xyz="1 0 0"/><limit lower="-1" upper="3" effort="1" velocity="1"/></joint></robot>')
 assert URDFFK(p).neutral_configuration()==pytest.approx([1.])
