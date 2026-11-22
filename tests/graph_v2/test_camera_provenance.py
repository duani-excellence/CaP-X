"""Camera convention is a versioned public dependency, not scene-fitted geometry."""
import copy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from capx.execution.grounded_feedback import GroundedMonitor
from capx.experiments import graph_v2 as runner


def monitor():
    m=GroundedMonitor('G1','stack',lambda:{'robot_joint_pos':np.zeros(7),'robot_cartesian_pos':np.array([0,0,0,1,0,0,0])})
    m.geometry_fn=lambda state:({},None,np.empty((0,3)))
    m.color_points_fn=lambda state,name:np.empty((0,3))
    return m


def test_camera_version_is_actual_observation_ancestor_and_changed_projection_invalidates_plan():
    m=monitor();metadata={'camera':{'input_intrinsics':[[1,0,2],[0,1,2],[0,0,1]],'output_intrinsics':[[1,0,1.5],[0,1,1.5],[0,0,1]]}}
    ids=m.register_camera_calibration(metadata);m.capture();obs=m.last_observation
    unrelated=m.graph.add('estimate',{'public':'unrelated'})
    p=m.graph.add('plan',{'skill':'place'},parents=[obs])
    assert any(e['source']==ids[0] and e['target']==obs for e in m.graph.edges)
    assert m.register_camera_calibration(copy.deepcopy(metadata))==ids
    assert m.graph.nodes[p]['valid']
    changed=copy.deepcopy(metadata);changed['camera']['output_intrinsics'][0][0]=1.1
    new=m.register_camera_calibration(changed)
    assert new!=ids and not m.graph.nodes[p]['valid']
    assert m.graph.nodes[unrelated]['valid']
    m.capture();assert m.graph.nodes[m.last_observation]['valid']


def test_g0_conservative_invalidation_refreshes_calibration_without_changing_metadata():
    m=monitor();data={'camera':{'source':'static_projection'}}
    old=m.register_camera_calibration(data)
    m.graph.invalidate(old,'conservative_recovery',conservative=True)
    new=m.register_camera_calibration(data)
    assert new!=old and m.graph.nodes[new[0]]['valid']


def test_installed_monitor_archives_exact_corrected_camera_and_keeps_legacy_input(tmp_path,monkeypatch):
    from capx.execution import bounded_ik
    asset=tmp_path/'robot.urdf';asset.write_text('<robot/>')
    calibration={'calibration_hash':'fixture','sources':[{'role':'ik_urdf','path':str(asset),'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}]}
    raw={'robot_joint_pos':np.zeros(7),'robot_cartesian_pos':np.array([0,0,0,1,0,0,0]),'hidden_object_truth':[1,2,3],
        'camera':{'intrinsics':np.array([[100,0,2],[0,100,2],[0,0,1.]]),'pose_mat':np.eye(4),
        'images':{'rgb':np.zeros((4,4,3),np.uint8),'depth':np.ones((4,4))}}}
    from capx.execution.depth_sampling import resolve_profile
    positions=[[.375,.125],[.875,.375],[.125,.625],[.625,.875]]
    trials=[]
    for i in range(4):
        depths=np.roll([.2,.4,.6,.8],i).tolist();trials.append({'assigned_sample_depths':depths,'resolved_depths':[depths[0]]*3})
    profile=resolve_profile(positions,trials,sample_count=4)
    actor=SimpleNamespace(apis={},call_site={})
    env=SimpleNamespace(public_robot_calibration=lambda:calibration,public_depth_sampling=lambda:{'camera':profile},_functions={},_actor=actor,
        _apis={'FrankaControlApi':SimpleNamespace(ik_solve_fn=lambda:None)},_get_observation=lambda:raw,
        _write=lambda *a,**kw:None,evaluation_state=lambda:{})
    monkeypatch.setattr(bounded_ik,'BoundedIK',lambda *a,**kw:SimpleNamespace(successful_calls=0))
    dest=tmp_path/'episode';dest.mkdir()
    m=runner.install_monitor(env,'G1','stack',dest,{'api_budget':10,'max_local':2,'max_episode_local':4},'none')
    m.geometry_fn=lambda state:({},None,np.empty((0,3)));m.color_points_fn=lambda state,name:np.empty((0,3))
    state,_=m.capture()
    assert state['camera']['intrinsics'][0,2]==1.625
    assert state['camera']['intrinsics'][1,2]==1.125
    assert raw['camera']['intrinsics'][0,2]==2
    assert 'hidden_object_truth' not in state
    with np.load(dest/'observations'/'00001.npz') as saved:
        np.testing.assert_array_equal(saved['camera_intrinsics'],state['camera']['intrinsics'])
        assert 'hidden_object_truth' not in saved
    meta=json.loads((dest/'observations'/'00001.camera.json').read_text())
    assert meta['calibration']['camera']['principal_point_delta_pixels']==[-.375,-.875]
    assert meta['calibration']['camera']['rgb_principal_point_delta_pixels']==[-.5,-.5]
    assert meta['calibration']['camera']['depth_sampling']['profile_sha256']==profile['profile_sha256']
    assert meta['graph_source_ids']==m.observation_calibration_sources
    assert all(m.graph.nodes[node]['kind']=='estimate' for node in meta['graph_source_ids'])
    assert any(e['source'] in meta['graph_source_ids'] and e['target']==m.last_observation for e in m.graph.edges)


def test_disappearing_camera_invalidates_old_visual_plans_instead_of_retaining_permission():
    m=monitor();ids=m.register_camera_calibration({'camera':{'source':'projection'}})
    m.capture();plan=m.graph.add('plan',{'skill':'place'},parents=[m.last_observation])
    unrelated=m.graph.add('estimate',{'public':'robot_only'})
    assert m.register_camera_calibration({})==[]
    assert not m.graph.nodes[plan]['valid'] and not m.graph.nodes[ids[0]]['valid']
    assert m.graph.nodes[unrelated]['valid']


def test_installed_monitor_asks_current_renderer_profile_after_frame(tmp_path,monkeypatch):
    from capx.execution import bounded_ik,public_camera
    asset=tmp_path/'robot.urdf';asset.write_text('<robot/>')
    calibration={'calibration_hash':'fixture','sources':[{'role':'ik_urdf','path':str(asset),'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}]}
    order=[];profile={'camera':{'status':'PASS','fixture':'current-context'}}
    raw={'robot_joint_pos':np.zeros(7),'robot_cartesian_pos':np.array([0,0,0,1,0,0,0])}
    def frame():order.append('render');return raw
    def sampling():order.append('sample-current-context');return profile
    def convert(observation,*,depth_sampling=None,require_depth_sampling=False):
        order.append('convert')
        assert depth_sampling is profile and require_depth_sampling is True
        return observation,{}
    actor=SimpleNamespace(apis={},call_site={})
    env=SimpleNamespace(public_robot_calibration=lambda:calibration,public_depth_sampling=sampling,
        _functions={},_actor=actor,_apis={'FrankaControlApi':SimpleNamespace(ik_solve_fn=lambda:None)},
        _get_observation=frame,_write=lambda *a,**kw:None,evaluation_state=lambda:{})
    monkeypatch.setattr(bounded_ik,'BoundedIK',lambda *a,**kw:SimpleNamespace(successful_calls=0))
    monkeypatch.setattr(public_camera,'robosuite_pixel_centers',convert)
    dest=tmp_path/'episode';dest.mkdir()
    m=runner.install_monitor(env,'G1','stack',dest,{'api_budget':10,'max_local':2,'max_episode_local':4},'none')
    m.geometry_fn=lambda state:({},None,np.empty((0,3)));m.color_points_fn=lambda state,name:np.empty((0,3))
    m.capture()
    assert order==['render','sample-current-context','convert']
