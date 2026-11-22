"""Active-view resources persist across actors and follow returned translation."""
import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import capx.execution.grounded_feedback as module
from capx.execution.sandbox import ExecutionInterrupted
from test_table_reobservation import ready


def prepared(monkeypatch,mode='G1',limit=8):
    m,s,v,c,f,*_=ready(monkeypatch,mode)
    m.max_episode_views=limit
    m.max_episode_local=24
    return m,s,v,c,f


def view(m,f):
    return m._recover_held_visibility(f,single_step=True)


def move(m,s,f,x):
    p=s['robot_cartesian_pos'][:3].copy();p[0]=x
    m._move(p.tolist(),s['robot_cartesian_pos'][3:7].tolist(),0.,f)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_new_returned_location_can_use_two_new_views_without_actor_reset(monkeypatch,mode):
    m,s,v,c,f=prepared(monkeypatch,mode)
    assert view(m,f) is not None and view(m,f) is not None
    m.begin_program('new_actor',1)
    move(m,s,f,.58)
    assert view(m,f) is not None and view(m,f) is not None
    assert m.visibility_probes==4 and m.local_attempts==4
    assert m.summary()['remaining_visibility_probes']==4
    assert len(c)==5  # one translation plus four explicitly bounded view actions
    assert m.summary()['visibility_context']['available_yaw_degrees']==[]


@pytest.mark.parametrize('refresh',['capture','actor','drift'])
def test_no_candidate_recharge_from_capture_actor_or_small_pose_drift(monkeypatch,refresh):
    m,s,v,c,f=prepared(monkeypatch)
    view(m,f);view(m,f)
    if refresh=='capture':m.capture()
    elif refresh=='actor':m.begin_program('new_actor',1)
    else:
        s['robot_cartesian_pos'][0]+=.004
        m.capture()
    before=len(c)
    assert view(m,f) is None and len(c)==before
    summary=m.summary()
    assert summary['visibility_context']['available_yaw_degrees']==[]
    assert summary['remaining_visibility_probes']==6
    assert summary['visibility_policy']['min_translation_m']==.03


def test_return_to_old_location_reuses_exhausted_candidates(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    view(m,f);view(m,f)
    first=m.summary()['visibility_context']['context_id']
    move(m,s,f,.58);view(m,f)
    assert m.summary()['visibility_context']['context_id']!=first
    move(m,s,f,.5)
    before=len(c)
    assert view(m,f) is None and len(c)==before
    assert m.summary()['visibility_context']['context_id']==first
    assert m.visibility_probes==3


def test_each_context_uses_plus_and_minus_about_its_own_reference(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    reference=s['robot_cartesian_pos'][3:7].copy()
    view(m,f);view(m,f)
    move(m,s,f,.58)
    reference2=s['robot_cartesian_pos'][3:7].copy()
    view(m,f);view(m,f)
    for command,ref,angle in [(c[0],reference,20),(c[1],reference,-20),(c[3],reference2,20),(c[4],reference2,-20)]:
        delta=Rotation.from_quat(np.roll(command[1][1],-1))*Rotation.from_quat(np.roll(ref,-1)).inv()
        np.testing.assert_allclose(delta.as_rotvec(),[0,0,np.deg2rad(angle)],atol=1e-8)


def test_episode_global_limit_and_local_limit_do_not_reset_at_new_position(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch,limit=3)
    view(m,f);view(m,f);move(m,s,f,.58);view(m,f)
    m.begin_program('new_actor',1)
    before=len(c)
    assert view(m,f) is None and len(c)==before
    assert m.visibility_probes==3 and m.summary()['remaining_visibility_probes']==0
    assert m.terminal_reason=='visibility_probe_budget_exhausted'


def test_no_safe_clearance_does_not_consume_candidate(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    original=module.predict_attachment
    monkeypatch.setattr(module,'predict_attachment',lambda *args:{'center':[.5,0,.03],'bottom_z':.005,'orientation_verified':True})
    assert view(m,f) is None and c==[] and m.visibility_probes==0
    monkeypatch.setattr(module,'predict_attachment',original)
    assert view(m,f) is not None and m.visibility_probes==1
    assert m.summary()['visibility_context']['tried_yaw_degrees']==[20]


def test_local_budget_denies_without_consuming_view_candidate(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch);m.local_attempts=m.max_episode_local
    assert view(m,f) is None and not c and m.visibility_probes==0
    assert m.terminal_reason=='local_recovery_budget_exhausted'
    assert m.summary()['visibility_context']['tried_yaw_degrees']==[]


def test_failed_returned_probe_is_charged_and_not_repeated(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    def fail(*args,**kwargs):raise ExecutionInterrupted('injected_motion_stop')
    m._move=fail
    with pytest.raises(ExecutionInterrupted):view(m,f)
    assert m.visibility_probes==1 and m.local_attempts==1
    m.begin_program('retry_actor',1)
    assert m.summary()['visibility_context']['tried_yaw_degrees']==[20]


def test_automatic_refresh_stops_after_two_current_context_candidates(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    m._held_surface=lambda:None
    assert m._recover_held_visibility(f) is None
    assert len(c)==2 and m.visibility_probes==2
    assert m.terminal_reason=='held_surface_evidence_unrecoverable'
    assert m.summary()['remaining_visibility_probes']==6


def test_new_grasp_has_new_reference_but_not_new_episode_budget(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    view(m,f);view(m,f)
    old=m.summary()['visibility_context']['context_id']
    m.closed=False;m.lifted=False;m.phase='approached';m.approached_object='red cube'
    m.invoke('close_gripper',[],{},f)
    assert m.visibility_probes==2
    m.phase='holding_verified';m.lifted=True;m.attachment={'public_only':True}
    m.attachment_id=m.graph.add('verification_fact',{'type':'public_rigid_grasp_attachment'})
    assert view(m,f) is not None
    assert m.summary()['visibility_context']['context_id']!=old and m.visibility_probes==3


def test_constructor_preserves_legacy_two_views_and_accepts_explicit_limit():
    obs=lambda:{'robot_cartesian_pos':np.array([.5,0,.14,0,0,1,0,.5]),'robot_joint_pos':np.zeros(8)}
    assert module.GroundedMonitor('G1','lift',obs).max_episode_views==2
    assert module.GroundedMonitor('G1','lift',obs,max_episode_views=8).max_episode_views==8
    for bad in (-1,True,1.5):
        with pytest.raises(ValueError):module.GroundedMonitor('G1','lift',obs,max_episode_views=bad)


def test_precise_returned_motion_also_establishes_a_new_view_context(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    view(m,f);view(m,f)
    old=m.summary()['visibility_context']['context_id']
    def precise(p,q,**kw):
        kw['execute']('goto_pose',[p,q],{'z_approach':0.})
        return {'corrections':0}
    monkeypatch.setattr(module,'precise_goto',precise)
    m._precise_place([.58,0,.14],s['robot_cartesian_pos'][3:7].tolist(),f)
    assert m.summary()['visibility_context']['context_id']!=old
    assert view(m,f) is not None and m.visibility_probes==3


def test_exhausted_explicit_same_region_view_has_no_false_recovery_option(monkeypatch):
    m,s,v,c,f,old,*_=ready(monkeypatch)
    m.max_episode_views=8;m.max_episode_local=24
    current=m.invoke('reobserve_table',[old['id']],{},f)
    current=m.invoke('reobserve_table',[current['id']],{},f)
    count=len(c)
    m.begin_program('another_actor',1)
    with pytest.raises(ExecutionInterrupted,match='table_active_view_unavailable'):
        m.invoke('reobserve_table',[current['id']],{},f)
    assert len(c)==count and m.visibility_probes==2
    assert m.summary()['remaining_visibility_probes']==6
    assert m.summary()['change_view_available'] is False
    events=[n['data'] for n in m.graph.nodes.values() if n['kind']=='check']
    assert any(e.get('metrics',{}).get('recoverable') is False for e in events)


def test_policy_context_is_historical_not_a_geometry_permission(monkeypatch):
    m,s,v,c,f=prepared(monkeypatch)
    view(m,f)
    context=m.summary()['visibility_context'];node=m.graph.nodes[context['source_id']]
    assert node['data']['historical_only'] and not node['data']['permission']
    reserve=[e for e in m.graph.events if e.get('action')=='visibility_candidate_reserved']
    assert len(reserve)==1 and reserve[0]['context_id']==context['context_id']
    assert reserve[0]['yaw_degrees']==20
