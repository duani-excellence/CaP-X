import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from capx.execution.robot_self_geometry import classify_mesh_depth, PandaSelfGeometry


def box():
    result=trimesh.creation.box(extents=[.1,.1,.1]);result.apply_translation([0,0,.2]);return result


def test_surface_depth_match_only():
    p=np.array([[0,0,.15],[0,0,.10],[0,0,.20],[.08,0,.15]])
    r=classify_mesh_depth(p,[0,0,0],box())
    assert r['robot_mask'].tolist()==[True,False,False,False]
    assert r['evidence']['kind']=='known_robot_mesh_first_ray_depth_agreement'
    assert r['evidence']['independent_sensor'] is False


def test_nearest_surface_inside_or_back_face_is_not_enough():
    p=np.array([[0,0,.25],[.049,0,.20]])
    assert not classify_mesh_depth(p,[0,0,0],box())['robot_mask'].any()


def test_protected_named_object_never_removed():
    r=classify_mesh_depth([[0,0,.15],[.01,0,.15]],[0,0,0],box(),protected_mask=[True,False])
    assert r['robot_mask'].tolist()==[False,True]
    assert r['evidence']['protected_points']==1


def test_geometry_tolerance_is_small_and_explicit():
    r=classify_mesh_depth([[0,0,.1515],[0,0,.1535]],[0,0,0],box(),tolerance_m=.002)
    assert r['robot_mask'].tolist()==[True,False]
    assert r['evidence']['surface_tolerance_m']==.002


@pytest.mark.parametrize('tol',[0,-1,.005,float('nan')])
def test_unsafe_tolerance_rejected(tol):
    with pytest.raises(ValueError):classify_mesh_depth([[0,0,.15]],[0,0,0],box(),tolerance_m=tol)


def test_invalid_points_do_not_gain_robot_identity():
    r=classify_mesh_depth([[0,0,.15],[float('nan'),0,0],[0,0,0]],[0,0,0],box())
    assert r['robot_mask'].tolist()==[True,False,False]


def test_empty_mesh_is_unknown():
    r=classify_mesh_depth([[0,0,.15]],[0,0,0],trimesh.Trimesh())
    assert not r['robot_mask'].any()


def test_protected_mask_must_align_exactly():
    with pytest.raises(ValueError):classify_mesh_depth([[0,0,.15]],[0,0,0],box(),protected_mask=[])


def test_local_missing_urdf_rejected_no_network(tmp_path):
    with pytest.raises(FileNotFoundError):PandaSelfGeometry(tmp_path/'missing.urdf',{})


def fake_urdf(tmp_path):
    p=tmp_path/'mini.urdf'
    links=['panda_link'+str(i) for i in range(8)]+['panda_hand','panda_leftfinger','panda_rightfinger']
    text='<robot name="panda">'+''.join('<link name="'+x+'"><visual><geometry><box size=".02 .02 .02"/></geometry></visual></link>' for x in links)
    for i in range(1,8):text+=f'<joint name="panda_joint{i}" type="revolute"><parent link="panda_link{i-1}"/><child link="panda_link{i}"/><origin xyz="0 0 .02"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
    text+='<joint name="hand" type="fixed"><parent link="panda_link7"/><child link="panda_hand"/></joint>'
    for i,name in [(1,'panda_leftfinger'),(2,'panda_rightfinger')]:
        text+=f'<joint name="panda_finger_joint{i}" type="prismatic"><parent link="panda_hand"/><child link="{name}"/><origin xyz="0 0 .02"/><axis xyz="0 {1 if i==1 else -1} 0"/><limit lower="0" upper=".04" effort="1" velocity="1"/>'+ ('<mimic joint="panda_finger_joint1"/>' if i==2 else '')+'</joint>'
    p.write_text(text+'</robot>')
    c={'schema':'capx.public_robot_calibration.v1','T_ik_base_public_base':np.eye(4).tolist(),'T_actual_hand_ik_hand':np.eye(4).tolist(),'actual_tcp_offset_in_hand_m':[0,0,.107],'ik_urdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    c['calibration_hash']=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    return p,c


def test_unknown_fingers_excluded_and_asymmetric_encoders_supported(tmp_path):
    p,c=fake_urdf(tmp_path);g=PandaSelfGeometry(p,c)
    _,meta=g.mesh(np.zeros(7))
    assert set(meta['excluded_links'])=={'panda_leftfinger','panda_rightfinger'}
    _,meta=g.mesh(np.zeros(7),finger_positions_m=[.01,.03])
    assert meta['excluded_links']==[]
    assert meta['finger_positions_m']==[.01,.03]
    left=g.urdf.get_transform('panda_leftfinger')[:3,3];right=g.urdf.get_transform('panda_rightfinger')[:3,3]
    assert left[1]==pytest.approx(.01)
    assert right[1]==pytest.approx(-.03)


def test_calibration_and_urdf_hash_must_match(tmp_path):
    p,c=fake_urdf(tmp_path);bad=copy.deepcopy(c);bad['T_ik_base_public_base'][0][3]=.1
    with pytest.raises(ValueError):PandaSelfGeometry(p,bad)
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):PandaSelfGeometry(p,c)


def test_fk_public_tcp_inconsistency_revokes_robot_mask(tmp_path):
    p,c=fake_urdf(tmp_path);g=PandaSelfGeometry(p,c)
    r=g.classify([[0,0,.01]],[0,0,-1],np.zeros(7),observed_tcp=[.5,0,.247,1,0,0,0])
    assert not r['robot_mask'].any()
    assert r['evidence']['status']=='UNCERTAIN'
    assert r['evidence']['reason']=='public_joint_fk_tcp_inconsistent'


@pytest.mark.parametrize('q',[np.zeros(8),[0]*6+[float('nan')]])
def test_joint_shape_finite_required(tmp_path,q):
    p,c=fake_urdf(tmp_path);g=PandaSelfGeometry(p,c)
    with pytest.raises(ValueError):g.mesh(q)


def test_missing_public_tcp_is_not_silent_calibration(tmp_path):
    p,c=fake_urdf(tmp_path);g=PandaSelfGeometry(p,c)
    r=g.classify([[0,0,.01]],[0,0,-1],np.zeros(7))
    assert not r['robot_mask'].any()
    assert r['evidence']['reason']=='public_tcp_required_for_current_fk_validation'


def test_visual_mesh_bytes_are_in_asset_closure(tmp_path):
    p,c=fake_urdf(tmp_path);mesh=tmp_path/'robot-piece.stl';trimesh.creation.box(extents=[.02,.02,.02]).export(mesh)
    p.write_text(p.read_text().replace('<box size=".02 .02 .02"/>','<mesh filename="robot-piece.stl"/>',1))
    c['ik_urdf_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();c.pop('calibration_hash');c['calibration_hash']=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    first=PandaSelfGeometry(p,c);sources=first.sources
    match=next(s for s in sources if s['path']==str(mesh.resolve()))
    assert match['sha256']==hashlib.sha256(mesh.read_bytes()).hexdigest()
    _,meta=first.mesh(np.zeros(7))
    assert meta['geometry_asset_hash']==first.metadata['geometry_asset_hash']
    trimesh.creation.box(extents=[.03,.02,.02]).export(mesh)
    second=PandaSelfGeometry(p,c)
    assert second.urdf_hash==first.urdf_hash
    assert second.metadata['geometry_asset_hash']!=first.metadata['geometry_asset_hash']
