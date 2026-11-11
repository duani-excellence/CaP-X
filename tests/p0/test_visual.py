from pathlib import Path
import numpy as np
import pytest
from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel


def test_xyzw_pose_matches_mujoco():
    import viser.transforms as vtf
    e = FrankaRobosuiteCubesLowLevel(privileged=True, enable_render=False)
    try:
        e.reset(seed=71)
        pose = e.get_observation()['cube_poses']['primary']
        base = vtf.SE3(e.base_link_wxyz_xyz)
        actual = base @ vtf.SE3.from_rotation_and_translation(vtf.SO3(pose[3:]), pose[:3])
        wanted = e.robosuite_env.sim.data.xmat[e.robosuite_env.cubeA_body_id].reshape(3,3)
        np.testing.assert_allclose(actual.rotation().as_matrix(), wanted, atol=1e-6)
    finally:
        e.close()


def test_no_depth_root_overwrite():
    assert '.save("depth_image.jpg")' not in Path('capx/integrations/franka/control.py').read_text()


def test_candidate_filter():
    from capx.integrations.franka.geometry import select_grasp
    bad = np.eye(4); bad[0,0] = 4
    good = np.eye(4)
    T, idx, score = select_grasp(np.stack([bad,good,good]), np.array([100,2,2]))
    assert idx == 1 and score == 2
    with pytest.raises(ValueError):
        select_grasp([], [])
    with pytest.raises(ValueError):
        select_grasp([good], [np.nan])


def test_tcp_and_base_vertical_approach():
    from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
    from types import SimpleNamespace
    api = object.__new__(FrankaControlPrivilegedApi)
    targets=[]
    api.cfg=None
    api.ik_solve_fn=lambda **kw: targets.append(kw['target_pose_wxyz_xyz'].copy()) or np.zeros(8)
    api._env=SimpleNamespace(move_to_joints_blocking=lambda x:None)
    api.goto_pose(np.array([.3,.1,.4]), np.array([1.,0,0,0]), z_approach=.1)
    np.testing.assert_allclose(targets[0][-3:]-targets[1][-3:], [0,0,.1])
    np.testing.assert_allclose(targets[1][-3:], [.3,.1,.293])

def test_observed_tcp_matches_command_offset():
    import viser.transforms as vtf
    e=FrankaRobosuiteCubesLowLevel(privileged=True,enable_render=False)
    try:
        obs,_=e.reset(seed=10)
        sim=e.robosuite_env.sim
        i=sim.model.body_name2id('robot0_right_hand')
        hand=vtf.SE3.from_rotation_and_translation(vtf.SO3(sim.data.xquat[i]),sim.data.xpos[i])
        expected=vtf.SE3(e.base_link_wxyz_xyz).inverse() @ hand @ vtf.SE3.from_translation(np.array([0.,0.,.107]))
        np.testing.assert_allclose(obs['robot_cartesian_pos'][:3],expected.translation(),atol=1e-7)
        np.testing.assert_allclose(vtf.SO3(obs['robot_cartesian_pos'][3:7]).as_matrix(),expected.rotation().as_matrix(),atol=1e-7)
    finally:
        e.close()
