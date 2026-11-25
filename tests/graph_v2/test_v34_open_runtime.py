"""Public-only counterexamples for V34; these are not physical success claims."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_open_program_analysis_accepts_computation_branch_and_bounded_loop():
    from capx.execution.causal_plan import program_calls
    code="""r = read_robot()
for i in range(3):
    if i == 1:
        s = observe_surfaces('destination')
        p = plan_skill('deposit', {'region': s['id']}, {'object': 'novel', 'target': '0'})
        run_plan(p['id'])
"""
    result=program_calls(code)
    assert {'read_robot','observe_surfaces','plan_skill','run_plan'} <= {x['name'] for x in result}
    assert next(x for x in result if x['name']=='run_plan')['conditional']


def test_information_improvement_counts_but_motion_and_source_ids_do_not():
    from capx.execution.persistent_harness import ProgressTracker
    t=ProgressTracker();state={'remaining_goals':['goal'],'conditions':{'target':{'truth':'UNKNOWN'}},'phase':'held'}
    assert t.update(state,{'target':{'coverage':.2}})['changed']
    assert not t.update(state,{'target':{'coverage':.2,'source':'new'}},robot=[1,0,0])['changed']
    assert t.update(state,{'target':{'coverage':.9}})['information_progress']
    assert not t.update(state,{'target':{'coverage':.9}},robot=[1.001,0,0])['changed']
    assert not t.update(state,{'target':{'coverage':.2}},robot=[2,0,0])['changed']
    assert not t.update(state,{'target':{'coverage':.9}})['changed']


def test_search_progress_is_finite_and_not_new_ids():
    from capx.execution.persistent_harness import ProgressTracker
    t=ProgressTracker(search_limit=2);s={'conditions':{},'remaining_goals':['g']}
    t.update(s,{})
    assert t.update(s,{},search=[{'strategy':'view','parameters':[1,2,3],'outcome':'occluded'}])['search_progress']
    assert not t.update(s,{},search=[{'strategy':'view','parameters':[1,2,3],'outcome':'occluded','id':'new'}])['changed']
    assert t.update(s,{},search=[{'strategy':'view','parameters':[2,2,3],'outcome':'occluded'}])['search_progress']
    assert not t.update(s,{},search=[{'strategy':'view','parameters':[3,2,3],'outcome':'occluded'}])['changed']


def test_interface_return_is_not_effect_verification(tmp_path):
    from capx.execution.agent_events import EventStore
    from capx.execution.persistent_harness import ExecutionMonitor
    store=EventStore(tmp_path)
    b=SimpleNamespace(actor=SimpleNamespace(call_site={'line':2}),spec={'harness_version':34})
    m=ExecutionMonitor(store,lambda:{'source':'old'},b)
    m.invoke('goto_pose',[[0,0,1],[0,0,1,0]],{},lambda *a:None)
    assert store.actions['p0:call1']['state']=='RETURNED'
    m.verify_motion([0,0,1],[0,0,1,0],{'source':'fresh','robot':{'tcp':[0,0,1,0,0,1,0]}},True)
    assert store.actions['p0:call1']['state']=='DONE'
    assert store.actions['p0:call1']['result']['evidence_source']=='fresh'


def test_stale_or_failed_effect_cannot_mark_motion_done(tmp_path):
    from capx.execution.agent_events import EventStore
    from capx.execution.persistent_harness import ExecutionMonitor
    st=EventStore(tmp_path);b=SimpleNamespace(actor=SimpleNamespace(call_site=None),spec={'harness_version':34})
    m=ExecutionMonitor(st,lambda:{'source':'old'},b)
    m.invoke('goto_pose',[[0,0,1],[0,0,1,0]],{},lambda *a:None)
    with pytest.raises(ValueError,match='fresh'):m.verify_motion([0,0,1],[0,0,1,0],{'source':'old'},True)
    m.verify_motion([0,0,1],[0,0,1,0],{'source':'new'},False)
    assert st.actions['p0:call1']['state']=='FAILED'


def test_region_failure_keeps_axis_and_partial_measurements():
    from capx.execution.region_geometry import receptacle_geometry
    x,y=np.meshgrid(np.linspace(0,.4,40),np.linspace(0,.3,40));floor=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    fit=receptacle_geometry(floor,[(0,np.empty((0,3))),(1,np.empty((0,3)))])
    assert fit['status']=='UNKNOWN' and fit['axis']==0
    assert 'floor' in fit and 'wall_diagnostics' in fit


def test_view_history_requires_current_change_detection():
    from capx.execution.region_geometry import fuse_region_measurements
    old={'status':'PASS','floor_z':0.,'lo':[0,0],'hi':[.4,.3],'center':[.2,.15,0.], 'wall_planes':[[0,.4],[0,.3]],'source':'old'}
    partial={'status':'UNKNOWN','reason':'missing_bounded_wall_pair','axis':0,'floor':dict(old),'wall_diagnostics':[[0],[0,.3]]}
    assert fuse_region_measurements(old,partial,change_check={'status':'UNKNOWN'})['status']=='UNKNOWN'
    assert fuse_region_measurements(old,partial,change_check={'status':'FAIL'})['status']=='UNKNOWN'
    row=fuse_region_measurements(old,partial,change_check={'status':'PASS','calibration_unchanged':True,'coverage':.98})
    assert row['status']=='PASS' and row['evidence_kind']=='history_with_current_depth_change_detection'
    partial['wall_diagnostics'][0]=[.02]
    assert fuse_region_measurements(old,partial,change_check={'status':'PASS','calibration_unchanged':True,'coverage':1.})['status']=='UNKNOWN'


def test_motion_trajectory_supports_lateral_and_rotation_but_bounds_scope():
    from capx.execution.open_motion import interpolate_targets
    start=[.5,0,.3,0,0,1,0]
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler('z',10,degrees=True)*Rotation.from_quat([0,1,0,0])
    steps=interpolate_targets(start,[.55,.03,.31],np.roll(q.as_quat(),1))
    assert len(steps)>1 and np.allclose(steps[-1][0],[.55,.03,.31])
    with pytest.raises(ValueError):interpolate_targets(start,[1.5,0,.3],[0,0,1,0])


def test_missing_depth_is_not_free_space_and_obstacle_rejects():
    from capx.execution.open_motion import observed_free
    cam={'intrinsics':np.array([[100,0,10],[0,100,10],[0,0,1]]),'pose_mat':np.eye(4),'images':{'depth':np.ones((21,21))*2}}
    assert observed_free(np.array([[0,0,1.]]),{'cam':cam})['status']=='PASS'
    cam['images']['depth'][:]=.9
    assert observed_free(np.array([[0,0,1.]]),{'cam':cam})['status']!='PASS'
    cam['images']['depth'][:]=0
    assert observed_free(np.array([[0,0,1.]]),{'cam':cam})['status']=='UNKNOWN'


def test_placement_quality_prefers_interior_not_nearest_edge():
    from capx.execution.region_geometry import rank_placements
    region={'lo':[0,0],'hi':[1,1]};shape={'center':[.9,.9,0],'half_extent_xy':[.05,.05]}
    rows=rank_placements([[.9,.9],[.5,.5]],shape,region)
    assert rows[0]['xy']==[.5,.5] and rows[0]['boundary_margin_m']>.4


def test_open_cli_has_finite_multi_round_budget():
    from capx.experiments.unified import RunSpec,resolve
    s,c,f=resolve(RunSpec(harness='open',recovery_requests=6,force_mode='on',methods=['G0']))
    assert c['tasks']['pick_place']['harness_version']==34 and s.recovery_requests==6
    with pytest.raises(ValueError):RunSpec(harness='open',recovery_requests=10000)


def test_execution_policy_does_not_come_from_historical_snapshot_spec(tmp_path):
    from capx.execution.persistent_harness import PersistentHarness
    from capx.execution.skill_contracts import ContractGraph
    from capx.experiments.native_vision_skills import region_task
    import time
    scene=SimpleNamespace(source='initial',obs={})
    s=SimpleNamespace(scene=scene,phase='empty',held=None)
    c=ContractGraph(region_task({'unseen':'public'}));r=SimpleNamespace(c=c,skills=s,group='G0',emit=lambda *a,**k:None,open_runtime=True)
    b=SimpleNamespace(root=tmp_path,spec={'episode_seconds':60},started=time.monotonic(),calls=0,initial={'state_hash':'original'},actor=SimpleNamespace(call_site=None))
    h=PersistentHarness(b,r,SimpleNamespace(max_requests=6,count=0),request_limit=5)
    assert h.supervisor.open_runtime and h.tools.open_runtime and h.monitor.effect_lifecycle
    assert 'plan_motion' in h.tools.functions() and 'harness_version' not in b.spec
    h.monitor.invoke('read_robot',[],{},lambda:{'tcp':[0,0,0]})
    assert any(n['kind']=='api_call' for n in c.graph.nodes.values())
    b.public_capture_sink({'capture':0,'files':{'camera':'public.npz'},'robot':{'robot_cartesian_pos':[0,0,1,0,0,1,0]}})
    assert h.tools.belief.query('robot')['kind']=='direct_measurement'
    r.state=lambda:dict(phase='empty',attachment=None,remaining_goals=['placed:unseen'],conditions={'robot_ready':{'truth':'TRUE','node':'actual'}})
    r.broker=SimpleNamespace(calls=0,max_calls=10)
    assert h.context({})['conditions']['robot_ready']=={'truth':'TRUE'}
