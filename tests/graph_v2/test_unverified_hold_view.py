import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from capx.execution.grip_offset import calibrate
from test_grounded_monitor import fixture

def occluded(mode='G1'):
    m,s,v,c,f=fixture(mode)
    for k in ('center','lo','hi'):v['red'][k][2]+=.10
    v['red']['top_xy_bounds']=[v['red']['lo'][:2],v['red']['hi'][:2]]
    m.phase='holding_unverified';m.closed=True;m.lifted=False;m.held_object='red cube'
    m.lift_attempts=1;m.close_tcp_z=.02
    m.geometry['red cube']={'height_m':.04,'frame':'robot_base','shape':'cube','orientation_verified':True}
    m.attachment=calibrate(v['red'],m.geometry['red cube'],s['robot_cartesian_pos'],source_id='source',tcp_id='tcp')
    m.attachment_id=m.graph.add('verification_fact',{'type':'public_rigid_grasp_attachment'})
    base_geometry=m.geometry_fn;base_points=m.color_points_fn
    visible=lambda:any(name=='goto_pose' for name,args in c)
    m.geometry_fn=lambda state:({**v,'red':v['red'] if visible() else None},0.,np.empty((0,3)))
    m.color_points_fn=lambda state,name:base_points(state,name) if name!='red cube' or visible() else np.empty((0,3))
    return m,s,v,c,f

@pytest.mark.parametrize('mode',['G0','G1'])
def test_occluded_unverified_hold_can_reveal_view_then_check_comotion(mode):
    m,s,v,c,f=occluded(mode)
    out=m.invoke('verify_hold',[],{},f)
    assert out['status']=='PASS' and m.phase=='holding_verified'
    assert m.visibility_probes==1 and m.local_attempts==1 and m.hold_probes==1
    assert [name for name,args in c]==['goto_pose']*3
    # View motion rotates in place. Only the separate co-motion probe may
    # establish a hold; no repeated vertical lift or close is issued.
    np.testing.assert_allclose(c[0][1][0],[.5,0,.14])
    assert np.linalg.norm(np.array(c[0][1][1])-[0,0,1,0])>.01
    assert all(abs(args[0][2]-.14)<1e-8 for name,args in c)
    assert any(e['check']=='hold_probe' and e['status']=='PASS' for e in m.events)

@pytest.mark.parametrize('case',['missing_attachment','low_robot_rise','low_object_clearance','missing_neighbor','budget'])
def test_unverified_visibility_search_needs_existing_public_safety_evidence(case):
    m,s,v,c,f=occluded()
    if case=='missing_attachment':m.attachment=None
    if case=='low_robot_rise':m.close_tcp_z=.12
    if case=='low_object_clearance':m.attachment['offset_tcp']=[0,0,.12]
    if case=='missing_neighbor':v['green']=None;m.color_points_fn=lambda *args:np.empty((0,3))
    if case=='budget':m.local_attempts=m.max_episode_local
    with pytest.raises(ExecutionInterrupted):m.invoke('verify_hold',[],{},f)
    assert not c and not m.lifted and m.phase=='holding_unverified'

def test_new_view_is_not_itself_a_verified_grasp_or_replenished_budget():
    m,s,v,c,f=occluded()
    # Exercise only the parent-owned perception helper; no co-motion proof.
    m._recover_held_visibility(f,single_step=True,verification_probe=True)
    assert m.phase=='holding_unverified' and not m.lifted
    assert m.visibility_probes==1 and m.local_attempts==1
    m.begin_program('new_actor',1)
    assert m.visibility_probes==1 and m.local_attempts==1
