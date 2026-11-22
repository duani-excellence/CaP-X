"""Real simulation at broker boundary; no cloud or perception calls."""
import copy
import pytest
from capx.envs.tasks import get_config,get_exec_env
from capx.execution.feedback import FeedbackMonitor

@pytest.fixture
def env(tmp_path):
    cfg=copy.deepcopy(get_config('franka_robosuite_pick_place_code_env'))
    cfg.enable_render=False;cfg.artifact_root=str(tmp_path);cfg.execution_timeout=5
    obj=get_exec_env('franka_robosuite_pick_place_code_env')(cfg)
    obj.reset(seed=19)
    yield obj
    obj.close()

def test_episode_budget_includes_broker_recovery(env):
    env.cfg.feedback_api_budget=1
    low=env.low_level_env
    _,reward,done,truncated,info=env.step('open_gripper()\nclose_gripper()')
    assert info['api_calls']==1 and info['execution_status']=='interrupted'
    assert reward==0 and not done and not truncated
    assert env.low_level_env is low
    assert not env._needs_rebuild

def test_reset_removes_old_monitor_state(env):
    env._feedback_monitor=FeedbackMonitor('D',env._get_observation)
    env._feedback_monitor.closed=True
    env.reset(seed=19)
    assert not hasattr(env,'_feedback_monitor')
