"""P0 isolated Stack runner. Other tasks retain their legacy runner until audited."""
import json
from pathlib import Path
import time
import uuid

from capx.execution.sandbox import SandboxProcess
from capx.envs.physical_evaluation import PhysicalCriteria, physical_state
from capx.envs.reproducibility import json_value
from capx.envs.tasks.base import CodeExecutionEnvBase


class AuditedStackEnv(CodeExecutionEnvBase):
    def __init__(self, cfg):
        if not cfg.privileged and any('Privileged' in a for a in cfg.apis):
            raise ValueError('Student may not bind privileged APIs')
        if cfg.low_level != 'franka_robosuite_cubes_low_level':
            raise ValueError('P0 runner supports only the simulated Stack environment')
        from capx.envs.base import register_env
        from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel
        register_env('franka_robosuite_cubes_low_level', FrankaRobosuiteCubesLowLevel)
        self._needs_rebuild = False
        self._trace_path = None
        self._request_audit = None
        super().__init__(cfg)

    def _init_exec_globals(self):
        if hasattr(self, '_actor'):
            self._actor.close()
        self._exec_globals = {}
        self._functions = {k: v for api in self._apis.values() for k,v in api.functions().items()}
        self._actor = SandboxProcess(self._functions, callback=self._call_api)

    def _write(self, kind, **row):
        if self._trace_path is not None:
            with self._trace_path.open('a') as f:
                f.write(json.dumps(dict(kind=kind, **row), default=json_value, allow_nan=False)+'\n')

    def _get_observation(self):
        raw = self.low_level_env.get_observation()
        obs = {k: raw[k] for k in ('robot_joint_pos', 'robot_cartesian_pos') if k in raw}
        for name in self.low_level_env.render_camera_names:
            camera = raw.get(name, {})
            obs[name] = {k: camera[k] for k in ('pose', 'pose_mat', 'intrinsics') if k in camera}
            obs[name]['images'] = {k:v for k,v in camera.get('images', {}).items() if k in ('rgb','depth')}
        obs['full_prompt'] = self._full_prompt
        return obs

    def reset(self, *, seed=None, options=None):
        # Monitors are episode-scoped and explicitly installed after reset.
        if hasattr(self, '_feedback_monitor'):
            del self._feedback_monitor
        self._actor.close()
        if self._request_audit is not None:
            self._request_audit.__exit__()
            self._request_audit = None
        if hasattr(self.low_level_env, "_physics_observer"):
            del self.low_level_env._physics_observer
        if self._needs_rebuild:
            self.low_level_env.close()
            self.low_level_env = self._build_low_level(self.cfg.low_level, self.cfg.privileged, self.cfg.enable_render, self.cfg.viser_debug)
            from capx.integrations.base_api import get_api
            self._apis = {n:get_api(n)(self.low_level_env) for n in self.cfg.apis}
            self._needs_rebuild = False
        snapshot = (options or {}).get('snapshot')
        if snapshot is not None:
            if seed is not None and seed != snapshot['seed']:
                raise ValueError('Snapshot seed mismatch')
            _, info = self.low_level_env.restore_snapshot(snapshot)
            if info['initial_state_hash'] != snapshot['state_hash']:
                raise ValueError('Restored physical hash mismatch')
        else:
            _, info = self.low_level_env.reset(seed=seed, options=options)
        for api in self._apis.values():
            if hasattr(api, 'cfg'):
                api.cfg = None
        self._init_exec_globals()
        self._step_count = self._api_calls = self._motion_calls = 0
        root = Path(getattr(self.cfg, 'artifact_root', None) or 'runs/p0')
        run = root / uuid.uuid4().hex
        run.mkdir(parents=True, exist_ok=False)
        self._trace_path = run / 'trace.jsonl'
        from capx.utils.request_audit import RequestAudit
        self._request_audit = RequestAudit(run / 'requests.jsonl').__enter__()
        self.criteria = PhysicalCriteria(physical_state(self.low_level_env))
        self.low_level_env._physics_observer = self._observe_physics
        self._write('reset', seed=info['seed'], initial_state_hash=info['initial_state_hash'], mode='teacher_privileged' if self.cfg.privileged else 'student', snapshot=self.low_level_env.capture_snapshot())
        return self._get_observation(), info

    def _observe_physics(self):
        state = physical_state(self.low_level_env)
        self.criteria.update(state)
        self._write('physics', state=state)

    def _call_api(self, name, args, kwargs):
        monitor = getattr(self, '_feedback_monitor', None)
        if monitor is not None:
            return monitor.invoke(name, args, kwargs, self._execute_api)
        return self._execute_api(name, args, kwargs)

    def _execute_api(self, name, args, kwargs):
        limit = getattr(self.cfg, 'feedback_api_budget', None)
        if limit is not None and self._api_calls >= limit:
            from capx.execution.sandbox import ExecutionInterrupted
            raise ExecutionInterrupted('Episode API budget exhausted (including recovery)')
        before = physical_state(self.low_level_env)
        self._api_calls += 1
        if name in ('goto_pose','open_gripper','close_gripper','home_pose'):
            self._motion_calls += 1
        error = None
        response = None
        try:
            response = self._functions[name](*args, **kwargs)
            return response
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            # Transport/service failures must not become training negatives.
            from capx.utils.serve_utils import RobotAPIConnectionError
            if not isinstance(exc, (ValueError, TypeError, TimeoutError)):
                raise ConnectionError(error) from exc
            raise
        finally:
            after = physical_state(self.low_level_env)
            self._write('api', name=name, args=args, kwargs=kwargs, before=before, after=after, response=response, error=error)

    def step(self, action):
        if self._needs_rebuild:
            raise RuntimeError('reset() required after worker failure')
        if self._trace_path is None:
            raise RuntimeError('reset() required before execution')
        start_motion_calls = self._motion_calls
        self._step_count += 1
        self._write('code', code=action, block=self._step_count)
        result = self._actor.run(action, timeout=getattr(self.cfg, 'execution_timeout', 90.0))
        self._needs_rebuild = result['status'] in ('timeout', 'system_error')
        if result['ok'] and self._motion_calls:
            # Evaluate a released stack over a physical hold window, independent of code claims.
            for _ in range(12):
                self.low_level_env._step_once()
        state = physical_state(self.low_level_env)
        self.criteria.update(state)
        physical_success = bool(self.criteria.stable_placement)
        truncated = self._needs_rebuild or self.low_level_env._sim_step_count >= self.low_level_env.max_steps
        won = physical_success and result['ok'] and self._motion_calls > start_motion_calls and not truncated
        info = dict(sandbox_rc=0 if result['ok'] else 1, stdout=result['stdout'], stderr=result['stderr'],
                    execution_status=result['status'], task_completed=physical_success, physical_success=physical_success,
                    raw_success=state['raw_success'], grasp_success=self.criteria.grasp_success, slipped=self.criteria.slipped,
                    http_requests=self._request_audit.count, llm_requests=self._request_audit.llm_count,
                    motion_calls=self._motion_calls, api_calls=self._api_calls, code_blocks=self._step_count,
                    initial_state_hash=self.low_level_env.initial_state_hash, trace_path=str(self._trace_path),
                    task_prompt=self._task_prompt, frame_timestamps=list(getattr(self.low_level_env, '_frame_timestamps', [])),
                    wrist_frame_timestamps=list(getattr(self.low_level_env, '_wrist_frame_timestamps', [])))
        self._write('result', reward=float(won), **info)
        return self._get_observation(), float(won), bool(won), bool(truncated), info

    def close(self):
        if getattr(self, "_request_audit", None) is not None:
            self._request_audit.__exit__()
            self._request_audit = None
        if hasattr(self, '_actor'):
            self._actor.close()
        super().close()
