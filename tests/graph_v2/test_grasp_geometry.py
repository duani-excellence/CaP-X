import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from capx.execution.grasp_geometry import cgn_upright_panda, panda_open_sweep


def rot(q):
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()


def quat(m):
    return np.roll(Rotation.from_matrix(m).as_quat(), 1)


@pytest.mark.parametrize('angle', [0., .45, 1.4, -2.8])
def test_cgn_baseline_is_panda_hand_y_and_tool_z_is_down(angle):
    z = np.array([0., 0., -1.]); x = np.array([np.cos(angle), np.sin(angle), 0.])
    raw = np.column_stack([x, np.cross(z, x), z])
    result = cgn_upright_panda(quat(raw))
    r = rot(result['quaternion_wxyz'])
    assert np.allclose(r[:, 1], x)
    assert np.allclose(r[:, 2], z)
    assert np.isclose(np.linalg.det(r), 1.)
    assert result['source_axis'] == 'contact_graspnet_x'
    assert result['target_axis'] == 'panda_tcp_y'


def test_tilted_candidate_projects_closing_axis_not_its_orthogonal_axis():
    raw = Rotation.from_euler('xyz', [172., 7., 28.], degrees=True).as_matrix()
    answer = cgn_upright_panda(quat(raw)); r = rot(answer['quaternion_wxyz'])
    expected = np.r_[raw[:2, 0] / np.linalg.norm(raw[:2, 0]), 0.]
    assert np.allclose(r[:, 1], expected)
    assert answer['raw_tool_tilt_deg'] > 0
    assert np.allclose(r[:, 2], [0, 0, -1])


@pytest.mark.parametrize('q', [[0,0,0,0], [1,2,3], [np.nan,0,0,1], [2,0,0,0]])
def test_bad_quaternion_rejected(q):
    with pytest.raises(ValueError): cgn_upright_panda(q)


def test_vertical_cgn_baseline_cannot_supply_yaw():
    with pytest.raises(ValueError, match='horizontal'):
        cgn_upright_panda(np.roll(Rotation.from_euler('y', 90, degrees=True).as_quat(), 1))


def cloud(center, n=64):
    # Deterministic visible patch, no simulator or private object data.
    x, y = np.meshgrid(np.linspace(-.001, .001, 8), np.linspace(-.001, .001, 8))
    return np.asarray(center) + np.c_[x.ravel(), y.ravel(), np.zeros(n)]


def test_visible_neighbour_in_finger_body_is_blocked():
    q=[0,1,0,0]
    points=cloud([0, -.060, .020])
    result=panda_open_sweep([0,0,0],q,{'support':points},opening_m=.08,z_approach=.1,padding_m=0)
    assert result['status']=='FAIL'
    assert result['collisions']['support']['finger1_body'] > 0
    assert result['proof']=='visible_points_only_not_collision_free_certificate'


def test_corrected_cgn_mapping_avoids_wrong_axis_neighbour_patch():
    raw=[0,0,1,0]
    old=[0,1,0,0]
    points=cloud([0, -.060, .020])
    correct=cgn_upright_panda(raw)['quaternion_wxyz']
    assert panda_open_sweep([0,0,0],old,{'support':points},padding_m=0)['status']=='FAIL'
    assert panda_open_sweep([0,0,0],correct,{'support':points},padding_m=0)['status']=='PASS'


def test_vertical_sweep_catches_point_above_final_finger_volume():
    p=cloud([0, -.06, .16]);q=[0,1,0,0]  # Above palm as well as fingers at final pose.
    assert panda_open_sweep([0,0,0],q,{'support':p},z_approach=0,padding_m=0)['status']=='PASS'
    assert panda_open_sweep([0,0,0],q,{'support':p},z_approach=.1,padding_m=0)['status']=='FAIL'


def test_world_rigid_translation_and_yaw_preserve_collision():
    q=np.array([0,1,0,0]);p=cloud([0,-.06,.02]);r=Rotation.from_euler('z', .7).as_matrix();t=np.array([.5,-.2,.1])
    a=panda_open_sweep([0,0,0],q,{'support':p})
    b=panda_open_sweep(t,quat(r@rot(q)),{'support':p@r.T+t})
    assert a['status']==b['status']=='FAIL'
    assert a['collisions']==b['collisions']


def test_missing_or_sparse_known_neighbor_is_uncertain_not_safe():
    assert panda_open_sweep([0,0,0],[0,1,0,0],{'support':np.empty((0,3))})['status']=='UNCERTAIN'
    assert panda_open_sweep([0,0,0],[0,1,0,0],{'support':np.ones((2,3))})['status']=='UNCERTAIN'


def test_malformed_points_rejected_not_silently_dropped():
    with pytest.raises(ValueError):panda_open_sweep([0,0,0],[0,1,0,0],{'support':np.ones((4,4))})
    with pytest.raises(ValueError):panda_open_sweep([0,0,0],[0,1,0,0],{'support':np.full((40,3),np.nan)})


def test_sweep_rejects_non_upright_or_nonphysical_opening():
    with pytest.raises(ValueError):panda_open_sweep([0,0,0],[1,0,0,0],{})
    with pytest.raises(ValueError):panda_open_sweep([0,0,0],[0,1,0,0],{},opening_m=.2)
    with pytest.raises(ValueError):panda_open_sweep([0,0,0],[0,1,0,0],{},z_approach=-.1)


def test_clear_neighbor_pass_is_explicitly_conditional():
    result=panda_open_sweep([0,0,0],[0,1,0,0],{'support':cloud([.2,.2,.02])})
    assert result['status']=='PASS'
    assert result['motion_scope']=='fixed_upright_vertical_approach_only'
    assert result['calibration']['opening_axis']=='panda_tcp_y'
    assert result['calibration']['source'].endswith('panda_gripper.xml + collision STL bounds')
