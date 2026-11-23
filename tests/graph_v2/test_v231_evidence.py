"""V23.1 software mechanisms; real experiments are recorded separately."""
import copy
import numpy as np
import pytest
from test_v23_goals_memory import memory


def test_rigid_dimension_does_not_accrue_translation_or_age_error():
    _,m,_=memory()
    x=m.estimate('payload',tcp=[.6,0,.12,1,0,0,0,.5],stamp=2.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected')
    assert x['bounds']['height_m']==m.profile['height_m']
    assert x['bounds']['z_m']>m.profile['z_m']
    assert x['uncertainty']['translation_m']==pytest.approx(.1*m.profile['translation_rate'])
    assert x['uncertainty']['correlation_cancellation_m']==0


def test_forecast_never_refreshes_or_mutates_memory():
    _,m,_=memory();old=copy.deepcopy(m.records)
    x=m.forecast('payload',tcp=[.6,0,.12,1,0,0,0,.5],stamp=2.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected')
    assert m.records==old
    assert x['permission'] is False and x['status']=='PREDICTED'


def test_unclassified_semantic_fringe_is_not_a_confirmed_foreign_obstacle():
    from capx.execution.table_clearance import classify_unresolved_fringe
    mask=np.zeros((10,10),bool);mask[3:7,3:7]=True
    residual=np.zeros_like(mask);residual[2,3:7]=True
    result=classify_unresolved_fringe(residual,mask,np.zeros_like(mask))
    assert result['all_on_semantic_boundary']
    residual[0,0]=True
    assert not classify_unresolved_fringe(residual,mask,np.zeros_like(mask))['all_on_semantic_boundary']
    residual[0,0]=False;other=np.zeros_like(mask);other[2,3]=True
    assert not classify_unresolved_fringe(residual,mask,other)['all_on_semantic_boundary']


def test_future_place_uses_anchor_and_reports_actual_missing_evidence():
    from capx.execution.evidence_planning import placement_obligations
    _,m,_=memory();support=dict(center=[.5,0,.025],hi=[.52,.02,.05])
    args=dict(held_object='payload',support=support,geometry={'height_m':.04},
        tcp=[.5,0,.12,1,0,0,0,.5],target=[.5,0,.076],quaternion=[1,0,0,0],
        stamp=1.,calibration='c1',grasp_epoch=1,intent_id='intent-1')
    d=placement_obligations(m,**args)
    assert d['status']=='PASS' and not d['permission']
    assert d['future_intent_id']=='intent-1' and d['retained_sources']
    m.invalidate('payload','slip')
    d=placement_obligations(m,**args)
    assert d['status']=='UNCERTAIN' and d['missing']==['held_geometry_anchor']


def test_future_plan_retains_unknown_correlation_and_blocks_unsafe_xy():
    from capx.execution.evidence_planning import placement_obligations
    _,m,_=memory()
    d=placement_obligations(m,held_object='payload',support=dict(center=[.5,0,.025],hi=[.52,.02,.05]),
        geometry={'height_m':.04},tcp=[.5,0,.12,1,0,0,0,.5],target=[.507,0,.076],
        quaternion=[1,0,0,0],stamp=1.,calibration='c1',grasp_epoch=1)
    assert d['status']=='UNCERTAIN' and d['next_work']=='recompute_alignment'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_real_monitor_future_gate_blocks_without_anchor_before_descent(mode):
    from test_executive_v22 import positioned
    from capx.execution.state_memory import StateMemory
    from capx.execution.sandbox import ExecutionInterrupted
    from test_v23_goals_memory import profile
    m,state,visible,calls,raw,*_=positioned(mode)
    m.memory=StateMemory(m.graph,profile());m.evidence_planning=True
    plan=next(n for n in reversed(list(m.graph.nodes.values())) if n['kind']=='plan' and n['data']['skill']=='place')
    m.active_plan=plan['id'];m._refresh_memory=lambda:None
    old=len(calls)
    with pytest.raises(ExecutionInterrupted,match='future_evidence_not_ready'):
        m._prepare_descent(plan['data']['position'],plan['data']['quaternion_wxyz'])
    assert len(calls)==old
    assert m.evidence_request['missing']==['held_geometry_anchor']
    assert any(n['kind']=='evidence_plan' for n in m.graph.nodes.values())


def test_production_envelope_permits_short_aligned_propagation_without_tuning():
    import json
    from pathlib import Path
    from capx.execution.state_memory import StateMemory,bounded_place_metrics
    from capx.execution.dependency_graph import DependencyGraph
    p=json.loads((Path(__file__).parents[2]/'env_configs/graph_feedback/calibration/memory-development-v23.json').read_text())
    g=DependencyGraph();m=StateMemory(g,p);o=g.add('observation',{})
    m.observe('payload',dict(center=[.5,0,.12],lo=[.48,-.02,.10],hi=[.52,.02,.14]),
        {'height_m':.04},source=o,calibration='c',stamp=0,evidence_ref='mock',
        tcp=[.5,0,.12,1,0,0,0,.5],attached=True,grasp_epoch=1)
    e=m.estimate('payload',tcp=[.5,0,.076,1,0,0,0,.5],stamp=1,calibration='c',
        grasp_epoch=1,hold_verified=True,occlusion='expected')
    support=dict(center=[.5,0,.025],hi=[.525,.025,.05])
    b={k:p[k] for k in ('xy_m','z_m','height_m')}
    assert bounded_place_metrics(e['surface'],support,{'height_m':.04},e['bounds'],b)['status']=='PASS'


def test_current_semantic_points_may_supply_missing_face_not_a_prior_box(monkeypatch):
    from test_table_clearance import fixture,Robot
    from test_held_mask_boundary import current_prediction
    from capx.execution.table_clearance import current_table_evidence
    import capx.execution.grip_offset as grip
    o,r,p,g,m=fixture();m[56,:]=False;seen=[]
    def check(pred,visible,**kw):
        seen.append(kw['points'].copy())
        return dict(status='PASS',current_surface=current_prediction(pred))
    monkeypatch.setattr(grip,'check_surface',check)
    out=current_table_evidence(o,r,'green cube',p,Robot(),lambda *a,**kw:[{'mask':m,'score':.9}],
        observation_id='current',held_identity_verified=True,nonheld_geometry=g)
    assert seen and out['evidence']['semantic_current_shape']['source']=='current_two_visible_vertical_faces_and_top_with_prior_public_cube_side'
    assert out['masks']['held'][56,56:73].all()


def test_tilt_increases_extent_uncertainty_without_changing_intrinsic_size():
    _,m,_=memory();angle=np.deg2rad(3)
    x=m.estimate('payload',tcp=[.5,0,.12,np.cos(angle/2),np.sin(angle/2),0,0,.5],
        stamp=1,calibration='c1',grasp_epoch=1,hold_verified=True,occlusion='expected')
    assert x['status']=='PREDICTED'
    assert x['uncertainty']['tilted_extent_m']>0
    assert x['bounds']['height_m']==m.profile['height_m']
    angle=np.deg2rad(6)
    assert m.estimate('payload',tcp=[.5,0,.12,np.cos(angle/2),np.sin(angle/2),0,0,.5],
        stamp=1,calibration='c1',grasp_epoch=1,hold_verified=True,occlusion='expected')['status']=='UNKNOWN'


def test_forecast_cannot_renew_expired_evidence_or_cancel_shared_bias_by_frame_id():
    _,m,_=memory();before=len(m.graph.events)
    x=m.forecast('payload',tcp=[.5,0,.12,1,0,0,0,.5],stamp=100.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected')
    assert x['status']=='UNKNOWN' and m.records['payload']['stamp']==0
    assert len(m.graph.events)==before


@pytest.mark.parametrize('mode',['G0','G1'])
def test_unidentified_table_points_select_bounded_view_not_claim_occupied(mode):
    from test_table_boundary import ready
    from capx.execution.sandbox import ExecutionInterrupted
    m,state,calls,raw,plan,table=ready();m.mode=mode
    m._view_budget=lambda:dict(change_view_available=True)
    m.table_evidence_fn=lambda *a,**k:dict(validation=dict(status='FAIL',
        reason='current_nonheld_obstacle_in_table_region',clearance_verified=False),
        evidence=dict(nonheld_current_evidence=[dict(object='red cube',source_id='geometry',
            outside_region_with_margin=True)]))
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},raw)
    assert not calls and not m.rejected_table_targets
    assert m.summary()['recovery_preconditions']['next_action']=='reobserve_table_same_region_then_new_plan_or_stop'


def test_graph_does_not_replace_missing_clearance_with_ineffective_surface_capture():
    from test_table_boundary import ready
    from capx.execution.graph_executive import GraphExecutive
    m,*_=ready();m.mode='G1';m.enable_task_conditions();c=m.conditions;obj=m.held_object
    c.set('holding:'+obj,'TRUE',reason='fixture')
    c.set('aligned:'+obj+':table','UNKNOWN',reason='fixture')
    c.history['positioned:'+obj+':table']={'fixture':True}
    c.set('clearance:'+obj+':table','UNKNOWN',reason='unresolved_actual_depth')
    assert GraphExecutive().decide(m)['kind']=='replace_placement'


def test_legacy_centroid_does_not_inherit_identifiable_face_error_envelope():
    from test_executive_v22 import positioned
    from capx.execution.state_memory import StateMemory
    from test_v23_goals_memory import profile
    m,*_=positioned();m.memory=StateMemory(m.graph,profile())
    assert m._support_surface('green cube',allow_retained=False) is not None
    assert m._calibrated_direct_surface('green cube') is None


@pytest.mark.parametrize('mode',['G0','G1'])
def test_predicted_release_without_calibrated_support_never_dispatches(mode,monkeypatch):
    import time
    import capx.execution.state_memory as sm
    from test_executive_v22 import positioned
    from test_v23_goals_memory import profile
    from capx.execution.sandbox import ExecutionInterrupted
    m,state,visible,calls,raw,*_=positioned(mode)
    m.memory=sm.StateMemory(m.graph,profile());m._memory_calibration=lambda:'c1'
    m.memory.observe(m.held_object,copy.deepcopy(visible['red']),dict(height_m=.04),
        source=m.last_observation,calibration='c1',stamp=time.monotonic(),evidence_ref='mock-held.npz',
        tcp=state['robot_cartesian_pos'],attached=True,grasp_epoch=m.grasp_epoch)
    monkeypatch.setattr(sm,'depth_occlusion',lambda *a:'expected')
    m._refresh_memory=lambda:None;m._held_surface=lambda:m._predicted_surface(m.held_object)
    before=len(calls)
    with pytest.raises(ExecutionInterrupted):m.invoke('open_gripper',[],{},raw)
    assert not any(name=='open_gripper' for name,_ in calls[before:])


def test_propagated_shape_cannot_masquerade_as_current_semantic_face():
    from capx.execution.table_clearance import _current_cube_shape
    shape=dict(source='current_two_visible_vertical_faces_and_top_with_prior_public_cube_side',
        face_point_counts=[60,70],evidence_kind='predicted',current_visual_detection=False)
    assert not _current_cube_shape(shape)


def test_place_api_uses_same_current_calibrated_support_as_future_verifier():
    import time
    from test_executive_v22 import positioned
    from capx.execution.state_memory import StateMemory
    from test_v23_goals_memory import profile
    m,state,visible,*_=positioned();m.memory=StateMemory(m.graph,profile());m._memory_calibration=lambda:'c1'
    fitted=copy.deepcopy(visible['green']);fitted['center'][0]+=.001
    m.memory.observe('green cube',fitted,dict(height_m=.05),source=m.last_observation,
        calibration='c1',stamp=time.monotonic(),evidence_ref='mock-fitted.npz',tcp=state['robot_cartesian_pos'])
    actual=m._current_state_surface('green cube')
    assert actual['center']==fitted['center'] and actual['evidence_kind']=='direct_calibrated_face'
