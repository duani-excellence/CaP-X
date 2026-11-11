import copy
import json
import pytest
from capx.envs.tasks import get_config, get_exec_env

@pytest.fixture
def env(tmp_path):
    cfg = copy.deepcopy(get_config('franka_robosuite_pick_place_code_env'))
    cfg.privileged = True
    cfg.enable_render = False
    cfg.artifact_root = str(tmp_path)
    cfg.execution_timeout = 0.4
    e = get_exec_env('franka_robosuite_pick_place_code_env')(cfg)
    yield e
    e.close()

def test_observation_and_globals(env):
    obs, info = env.reset(seed=42)
    assert 'cube_poses' not in obs and 'object-state' not in obs
    for code in ['print(env)', 'print(obs)', 'print(globals())', 'print(get_object_pose.__globals__)']:
        _, r, _, _, info = env.step(code)
        assert r == 0 and info['execution_status'] == 'policy_violation'

def test_timeout_rebuild_and_noop(env):
    env.reset(seed=42)
    old = env.low_level_env
    _, r, _, truncated, info = env.step('while True: pass')
    assert truncated and info['execution_status'] == 'timeout' and r == 0
    env.reset(seed=42)
    assert env.low_level_env is not old
    assert env.step('pass')[1] == 0

def test_trace(env):
    env.reset(seed=42)
    env.cfg.execution_timeout = 10
    _, r, _, _, info = env.step('open_gripper()')
    assert r == 0 and info['api_calls'] == 1 and info['code_blocks'] == 1
    rows = [json.loads(x) for x in open(info['trace_path'])]
    call = next(x for x in rows if x['kind'] == 'api')
    for key in ['objects', 'ee', 'gripper_opening', 'contacts', 'relative_poses', 'sim_time']:
        assert key in call['before'] and key in call['after']
    assert call['after']['sim_time'] > call['before']['sim_time']

def test_student_cannot_bind_privileged_api(env):
    cfg = copy.copy(env.cfg)
    cfg.privileged = False
    with pytest.raises(ValueError, match='privileged'):
        get_exec_env('franka_robosuite_pick_place_code_env')(cfg)

def test_parent_service_failure_rebuilds(env):
    from capx.utils.serve_utils import RobotAPIConnectionError
    env.reset(seed=42)
    env.cfg.execution_timeout=10
    old=env.low_level_env
    env._functions['open_gripper']=lambda:(_ for _ in ()).throw(RobotAPIConnectionError('offline'))
    _,reward,_,truncated,info=env.step('open_gripper()')
    assert reward==0 and truncated and info['execution_status']=='system_error'
    env.reset(seed=42)
    assert env.low_level_env is not old
