"""Terminal reward contract. Only the trusted evaluator can report success.

Infrastructure failures are NOT valid negative training examples. Callers must
stop/retry a batch when valid_for_training is false.
"""
from __future__ import annotations

import ast
import copy
import os
import re
import time
from typing import Any

os.environ.setdefault('MUJOCO_GL', 'egl')
from capx.envs.tasks import get_config, get_exec_env

initialized_envs: dict[str, Any] = {}


def _extract_code(content: str) -> str:
    text = content.strip()
    if '```' in text:
        match = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
        if match is None or '```' in match.group(1):
            raise ValueError('Expected exactly one Python block without surrounding text')
        text = match.group(1).strip()
    tree = ast.parse(text)
    if not tree.body:
        raise ValueError('Empty program')
    if text == 'FINISH':
        raise ValueError('FINISH is not a physical action')
    return text


def _evict(source: str) -> None:
    env = initialized_envs.pop(source, None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


def compute_score(data_source: str, solution_str: str, ground_truth: dict | str,
                  extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(score=0.0, won=False, task_completed=False, physical_success=False,
               sandbox_rc=-1, terminated=False, truncated=False, error='',
               status='task_failure', valid_for_training=True, elapsed_seconds=0.0,
               initial_state_hash=None, motion_calls=0)
    try:
        code = _extract_code(solution_str)
    except (SyntaxError, ValueError, TypeError) as exc:
        out.update(status='invalid_program', error=str(exc), sandbox_rc=1)
        return out
    started = time.monotonic()
    try:
        extra = extra_info or {}
        seed = int(extra.get('seed', 0))
        if data_source not in initialized_envs:
            cfg = copy.deepcopy(get_config(data_source))
            if cfg.privileged:
                cfg.enable_render = False
            initialized_envs[data_source] = get_exec_env(data_source)(cfg)
        env = initialized_envs[data_source]
        env.reset(seed=seed)
        _, raw_reward, terminated, truncated, info = env.step(code)
        # info is built by the parent evaluator, never accepted from actor RPC.
        physical = bool(info.get('physical_success', False))
        motion_calls = int(info.get('motion_calls', 0))
        status = info.get('execution_status', 'ok' if info['sandbox_rc'] == 0 else 'runtime_error')
        valid = status == 'ok' and info['sandbox_rc'] == 0 and not truncated
        won = physical and valid and motion_calls > 0
        out.update(score=float(won), won=won, physical_success=physical,
                   task_completed=bool(info.get('task_completed', False)),
                   sandbox_rc=int(info['sandbox_rc']), terminated=bool(terminated),
                   truncated=bool(truncated), raw_reward=float(raw_reward),
                   initial_state_hash=info.get('initial_state_hash'), motion_calls=motion_calls,
                   status='success' if won else ('task_failure' if status == 'ok' else status),
                   error=info.get('stderr', ''), seed=seed, trace_path=info.get('trace_path'),
                   api_calls=info.get('api_calls', 0), code_blocks=info.get('code_blocks', 0),
                   http_requests=info.get('http_requests', 0), llm_requests=info.get('llm_requests', 0),
                   valid_for_training=status not in ('system_error', 'sandbox_unavailable', 'timeout'))
        if status in ('timeout', 'system_error', 'sandbox_unavailable'):
            _evict(data_source)
    except TimeoutError as exc:
        out.update(status='timeout', truncated=True, valid_for_training=False, error=repr(exc))
        _evict(data_source)
    except Exception as exc:
        out.update(status='system_error', valid_for_training=False, error=repr(exc))
        _evict(data_source)
    finally:
        out['elapsed_seconds'] = time.monotonic() - started
    return out
