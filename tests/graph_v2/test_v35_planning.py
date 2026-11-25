"""No simulator truth: V35 planning and decision encoding contracts."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest


def test_ik_preserves_residuals_and_caches_exact_context():
    from capx.execution.preplanning import DiagnosticReachability
    from capx.execution.bounded_ik import IKConvergenceError
    class Solver:
        events=[];kinematics=SimpleNamespace(urdf_sha256='robot',limits=np.array([[-2,2]]*8))
        def __call__(self,**kw):
            self.events.append(dict(status='FAIL',position_error_m=.03,orientation_error_deg=4.,reference=kw['prev_cfg'].tolist()))
            raise IKConvergenceError('ik_unconverged_before_motion')
    solver=Solver();q=np.zeros(8)
    ik=DiagnosticReachability(solver,lambda p,r:np.r_[r,p],lambda:q,calibration='cal')
    first=ik.query([.5,0,.1],[0,0,1,0]);second=ik.query([.5,0,.1],[0,0,1,0])
    assert first['status']=='NO_SOLUTION_FOUND' and first['physical_unreachable'] is False
    assert first['iterations'][0]['position_error_m']==.03
    assert second['cache_hit'] and len(solver.events)==1
    q[0]=.01
    assert not ik.query([.5,0,.1],[0,0,1,0])['cache_hit']


def test_placement_search_varies_yaw_without_changing_attachment_or_margins():
    from capx.execution.preplanning import placement_candidates
    h=dict(tcp=np.eye(4),shape=dict(center=[.0,0,.05],half_extent_xy=[.02,.03],height_m=.1))
    profile=dict(xy_error_m=.003,z_error_m=.003)
    rows=placement_candidates(h,h['shape'],dict(residual_95_m=.002),dict(lo=[.3,.3],hi=[.7,.7],floor_z=0),profile)
    assert len({tuple(r['quaternion_wxyz']) for r in rows})>1
    for r in rows:
        assert r['boundary_margin_m']>=.025+.003+.002-1e-8
        assert np.allclose(r['object_center'],np.array(r['position'])+np.array(r['rotation'])@np.array(r['attachment_local']))
        assert np.allclose(np.array(r['rotation'])[:,2],[0,0,-1])


def test_preplanning_never_calls_motion_and_stale_context_is_not_reused():
    from capx.execution.preplanning import SkillPreplanner
    scene=SimpleNamespace(serial=1,source='x',observation_key='a',obs={'robot_cartesian_pos':[0,0,0,0,0,1,0],'robot_joint_pos':[0]*7,'robot_gripper_joint_pos':[.04,.04]})
    s=SimpleNamespace(scene=scene,motion_version=0,held=None,phase='empty',effects={},plans={},profile={})
    s.motion=lambda *a:pytest.fail('planning moved robot')
    s._acquire_plan=lambda name:({'center':[0,0,0]},np.zeros(3),np.array([0,0,1,0]),{})
    planner=SkillPreplanner(s)
    first=planner.preview({'skill':'acquire','object':'novel'})
    assert first['status']=='KINEMATIC_CANDIDATE' and first['permission'] is False
    assert planner.preview({'skill':'acquire','object':'novel'})['cache_hit']
    s.motion_version+=1
    assert not planner.preview({'skill':'acquire','object':'novel'})['cache_hit']


def test_encoder_is_goal_relevant_bounded_and_never_claims_permission():
    from capx.execution.decision_state import encode_decision_state
    s=SimpleNamespace(held=None,phase='empty',scene=SimpleNamespace(obs={'robot_cartesian_pos':[0,0,0,0,0,1,0]},source='frame'))
    state=dict(remaining_goals=['placed:new'],conditions={'placed:done':{'truth':'TRUE'},'geometry:new':{'truth':'UNKNOWN'}})
    b={'object:new':{'values':{'center':[1,2,3],'points':list(range(10000))},'source':'f','kind':'historical_measurement'}}
    result=encode_decision_state(state,s,b,[],{},[])
    assert 'points' not in str(result) and 'permission' in result and result['permission'] is False
    assert result['goals']['remaining']==['placed:new']
    assert result['objects']['new']['evidence_kind']=='historical_measurement'


def test_trajectory_requires_actual_start_and_attachment_collision_evidence():
    from capx.execution.preplanning import validate_trajectory
    kin=SimpleNamespace(validate=lambda q:np.asarray(q),pose=lambda q:np.r_[1,0,0,0,q[:3]])
    target=[1,0,0,0,.1,0,0];path=np.zeros((3,8));path[:,0]=[0,.05,.1]
    assert validate_trajectory(path,np.zeros(8),target,kin,collision_evidence=None)['status']=='UNKNOWN'
    check={'status':'PASS','attachment_included':True,'complete_public_world':True}
    assert validate_trajectory(path,np.zeros(8),target,kin,collision_evidence=check)['status']=='VALIDATED_CANDIDATE'
    bad=path.copy();bad[0,1]=.8
    assert validate_trajectory(bad,np.zeros(8),target,kin,collision_evidence=check)['status']=='REJECTED'
    assert validate_trajectory(path,np.zeros(8),target,kin,collision_evidence=dict(check,attachment_included=False),held=True)['status']=='UNKNOWN'


def test_tilt_diagnostics_cannot_authorize_original_upright_release():
    from capx.execution.preplanning import placement_candidates
    from scipy.spatial.transform import Rotation
    h=dict(tcp=np.eye(4),shape=dict(center=[0,0,0],half_extent_xy=[.02,.02],height_m=.1))
    args=(h,h['shape'],dict(residual_95_m=.002),dict(lo=[.3,.3],hi=[.7,.7],floor_z=0),dict(xy_error_m=.003))
    q=np.roll((Rotation.from_euler('y',40,degrees=True)*Rotation.from_quat([0,1,0,0])).as_quat(),1)
    with pytest.raises(ValueError,match='upright'):placement_candidates(*args,quaternion=q)
    rows=placement_candidates(*args,diagnostic_orientations=[q])
    assert rows and rows[0]['permission'] is False and rows[0]['diagnostic_only'] is True


def test_instrumented_ik_preserves_diagnostics_without_bypassing_wrapper():
    from capx.execution.preplanning import DiagnosticReachability
    calls=[]
    class Solver:
        events=[];kinematics=SimpleNamespace(urdf_sha256='x',limits=np.array([[-1,1]]*8))
        def __call__(self,**kw):
            self.events.append(dict(status='PASS',position_error_m=0,orientation_error_deg=0));return np.zeros(8)
    solver=Solver()
    def wrapped(**kw):calls.append(kw);return solver(**kw)
    wrapped.__wrapped__=solver
    ik=DiagnosticReachability(wrapped,lambda p,q:np.r_[q,p],lambda:np.zeros(8))
    r=ik.query([0,0,0],[1,0,0,0])
    assert len(calls)==1 and r['iterations'][0]['status']=='PASS'


def test_planning_default_result_is_bounded_detail_stays_queryable():
    from capx.execution.preplanning import planning_summary
    import json
    full=dict(id='search-x',status='NO_SOLUTION_FOUND',skill='deposit',object='new',permission=False,
              candidate_count=48,remaining_candidates=0,tested=[{'checks':[{'iterations':list(range(10000))}]}]*48)
    short=planning_summary(full)
    assert len(json.dumps(short))<4096 and 'tested' not in short and short['id']=='search-x'
    assert len(full['tested'])==48


def test_goal_encoding_does_not_count_true_geometry_as_completed_goal():
    from capx.execution.decision_state import encode_decision_state
    s=SimpleNamespace(held=None,phase='empty',scene=SimpleNamespace(obs={},source='x'))
    state=dict(task_goals=['placed:a','placed:b'],remaining_goals=['placed:b'],conditions={'geometry:b':{'truth':'TRUE'},'placed:a':{'truth':'TRUE'}})
    result=encode_decision_state(state,s,{},[],{},[])
    assert result['goals']['verified']==['placed:a']


def test_full_grasp_transform_preserves_translation_and_maps_closing_axis():
    from capx.execution.grasp_geometry import cgn_panda_pose
    raw=np.eye(4);raw[:3,3]=[.4,.1,.2]
    row=cgn_panda_pose(raw,[0,0,.107])
    assert np.allclose(row['position'],[.4,.1,.307])
    assert np.allclose(np.array(row['rotation'])[:,1],raw[:3,0])
    assert row['permission'] is False


def test_negative_cache_does_not_recharge_original_search_duration():
    from capx.execution.preplanning import SkillPreplanner
    s=SimpleNamespace(scene=SimpleNamespace(observation_key='same',source='s',obs={},serial=1),held=None,motion_version=0,phase='empty',effects={},plans={},profile={})
    planner=SkillPreplanner(s)
    row={'skill':'unsupported','object':'new'}
    first=planner.preview(row);second=planner.preview(row)
    assert second['cache_hit'] and second['new_solver_calls']==0
    assert second['original_compute_seconds']==first['seconds']


def test_spatial_relation_uses_registered_current_shape_and_keeps_provenance():
    from capx.execution.decision_state import encode_decision_state
    s=SimpleNamespace(held=None,phase='empty',scene=SimpleNamespace(obs={},source='f'))
    beliefs={'object:new':{'values':{'center':[0,0,0]},'kind':'historical_measurement','source':'old'}}
    plan=dict(id='p',object='new',target='cell',status='NO_SOLUTION_FOUND',region={'lo':[1,1],'hi':[3,3]},source='new')
    result=encode_decision_state({'remaining_goals':['placed:new']},s,beliefs,[plan],{},[])
    relation=result['spatial_relations'][0]
    assert relation['delta_xy_to_target_center']==[2.,2.]
    assert relation['object_evidence_kind']=='historical_measurement' and relation['permission'] is False
