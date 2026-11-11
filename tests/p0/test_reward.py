"""Reward contract, without importing or starting simulation/services."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def reward():
    tasks = types.ModuleType('capx.envs.tasks')
    tasks.get_config = lambda source: types.SimpleNamespace(privileged=True)
    tasks.get_exec_env = lambda source: lambda cfg: FakeEnv()
    path = Path(__file__).parents[2] / 'verl_agent_reward/capx_franka_reward.py'
    spec = importlib.util.spec_from_file_location('reward_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'capx.envs.tasks': tasks}):
        spec.loader.exec_module(mod)
    return mod


class FakeEnv:
    closed = False
    def reset(self, *, seed):
        self.seed = seed
    def step(self, code):
        if code == 'syntax_error':
            return {}, .9, False, False, {'sandbox_rc': 1, 'task_completed': False, 'physical_success': False, 'motion_calls': 0}
        if code == 'timeout':
            raise TimeoutError('deadline')
        if code == 'system_error':
            raise ConnectionError('IK unavailable')
        success = code == 'move()'
        return {}, 1.0 if success else .7, success, False, {
            'sandbox_rc': 0, 'task_completed': success, 'physical_success': success,
            'motion_calls': int(success), 'initial_state_hash': 'initial',
        }
    def close(self):
        self.closed = True


@pytest.mark.parametrize('code', ['pass', '', '   ', '# comment', 'FINISH', 'print("FINISH")', 'syntax_error'])
def test_non_success_never_gets_positive_reward(reward, code):
    out = reward.compute_score('fake', code, {}, {'seed': 7})
    assert out['score'] == 0
    assert out['won'] is False


def test_physical_success_only(reward):
    out = reward.compute_score('fake', 'move()', {}, {'seed': 7})
    assert out['score'] == 1 and out['won'] is True
    assert out['task_completed'] is True
    assert out['initial_state_hash'] == 'initial'


def test_timeout_evicts_environment(reward):
    env = FakeEnv()
    reward.initialized_envs['fake'] = env
    out = reward.compute_score('fake', 'timeout', {}, {})
    assert out['status'] == 'timeout' and out['score'] == 0
    assert env.closed and 'fake' not in reward.initialized_envs


def test_system_error_is_distinct(reward):
    out = reward.compute_score('fake', 'system_error', {}, {})
    assert out['status'] == 'system_error'
    assert out['valid_for_training'] is False
    assert 'fake' not in reward.initialized_envs


def test_constructor_error_is_captured(reward):
    reward.get_config = lambda name: (_ for _ in ()).throw(KeyError('not registered'))
    out = reward.compute_score('missing', 'move()', {}, {})
    assert out['status'] == 'system_error'
    assert out['won'] is False


def test_multiple_fences_rejected(reward):
    out = reward.compute_score('fake', '```python\nmove()\n```\n```python\npass\n```', {}, {})
    assert out['score'] == 0 and out['status'] == 'invalid_program'


def test_single_python_fence_supported(reward):
    assert reward.compute_score('fake', '```python\nmove()\n```', {}, {})['score'] == 1
