import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from capx.execution.robot_self_geometry import PandaSelfGeometry
from test_configured_gripper import configured
from test_robot_self_geometry import fake_urdf


def test_public_tcp_anchors_only_configured_gripper_not_arm(tmp_path):
    p,c,_,_=configured(tmp_path);g=PandaSelfGeometry(p,c);j=np.zeros(7)
    original,m0=g.mesh(j,finger_positions_m=[.01,.03])
    fk=np.array(m0['public_fk_tcp']);tcp=np.r_[fk[:3,3]+[.00002,0,0],[1.,0,0,0]]
    shifted,m1=g.mesh(j,finger_positions_m=[.01,.03],observed_tcp=tcp)
    # Last 3 pieces are actual configured hand/fingers; all arm vertices remain identical.
    count=sum(len(m.vertices) for m in g.gripper_visual.parts(np.eye(4),np.array([.01,.03])))
    assert np.allclose(shifted.vertices[:-count],original.vertices[:-count])
    assert np.allclose(shifted.vertices[-count:]-original.vertices[-count:],[.00002,0,0])
    assert m1['gripper_pose_source']=='current_public_tcp_plus_static_hand_offset'
    assert np.allclose(m1['public_fk_tcp'],fk) # independent FK guard reference is preserved


def test_public_tcp_rotation_applies_to_static_offset_and_fingers(tmp_path):
    p,c,_,_=configured(tmp_path);g=PandaSelfGeometry(p,c)
    r=Rotation.from_euler('z',.01);q=np.roll(r.as_quat(),1);tcp=np.r_[[.00001,0,.247],q]
    mesh,meta=g.mesh(np.zeros(7),finger_positions_m=[.01,.03],observed_tcp=tcp)
    hand=np.array(meta['public_gripper_hand']);assert np.allclose(hand[:3,:3],r.as_matrix())
    assert np.allclose(hand[:3,3],tcp[:3]-r.as_matrix()@np.array([0,0,.107]))


def test_current_tcp_does_not_remove_independent_fk_consistency_guard(tmp_path):
    p,c,_,_=configured(tmp_path);g=PandaSelfGeometry(p,c)
    result=g.classify([[0,0,.01]],[0,0,-1],np.zeros(7),finger_positions_m=[.01,.03],observed_tcp=[.1,0,.247,1,0,0,0])
    assert not result['robot_mask'].any()
    assert result['evidence']['reason']=='public_joint_fk_tcp_inconsistent'


def test_unknown_second_finger_still_not_invented_with_tcp(tmp_path):
    p,c,_,_=configured(tmp_path);g=PandaSelfGeometry(p,c)
    _,meta=g.mesh(np.zeros(7),observed_tcp=[0,0,.247,1,0,0,0])
    assert set(meta['excluded_links'])=={'panda_leftfinger','panda_rightfinger'}


def test_legacy_geometry_does_not_gain_configured_tcp_claim(tmp_path):
    p,c=fake_urdf(tmp_path);g=PandaSelfGeometry(p,c)
    a,_=g.mesh(np.zeros(7));b,meta=g.mesh(np.zeros(7),observed_tcp=[.00002,0,.247,1,0,0,0])
    assert np.allclose(a.vertices,b.vertices)
    assert meta['gripper_pose_source']=='joint_fk_legacy_geometry'
