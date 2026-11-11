import numpy as np
import pytest

from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel


@pytest.fixture
def env():
    obj = FrankaRobosuiteCubesLowLevel(privileged=True, enable_render=False)
    yield obj
    obj.close()


def test_same_seed_resets_objects_and_robot(env):
    env.reset(seed=101)
    first = env.robosuite_env.sim.data.qpos.copy()
    env.reset(seed=202)
    env.reset(seed=101)
    np.testing.assert_allclose(env.robosuite_env.sim.data.qpos, first, atol=1e-12, rtol=0)


def test_snapshot_replay_and_initial_hash(env):
    env.reset(seed=101)
    snapshot = env.capture_snapshot()
    initial = env.initial_state_hash
    hashes = []
    for _ in range(3):
        env.restore_snapshot(snapshot)
        assert env.initial_state_hash == initial
        for _ in range(8):
            env._set_gripper(1.0)
            env._step_once()
        hashes.append(env.physical_state_hash())
    assert len(set(hashes)) == 1


def test_different_seeds_different_initial_hash(env):
    env.reset(seed=101)
    one = env.initial_state_hash
    env.reset(seed=102)
    assert one != env.initial_state_hash


def test_api_sampling_seed_replayed(env):
    env.reset(seed=101)
    first = [env.next_grasp_seed() for _ in range(3)]
    env.reset(seed=101)
    assert first == [env.next_grasp_seed() for _ in range(3)]
