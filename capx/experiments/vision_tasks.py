"""Trusted Vision adapters for Stack, Lift and Restack.

The existing task Runtime owns physical state, scoring and deterministic resets.
Only RGB-D/calibration and robot proprioception reach FrankaControlApi and actors.
The simulator's ``privileged=True, render=True`` constructor selects offscreen
rendering; its privileged object API is replaced before any actor can execute.
Initial snapshots replay reset/controller history; in-flight restore is unsupported.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import numpy as np

TASKS = ('stack', 'lift', 'restack')


def public_depth_sampling(env):
    """Current renderer calibration only; caller must render the public frame first.

    No object state, task score, model geometry or private simulator arrays are
    accessed. A context recreation/resize is re-measured by the per-env probe.
    """
    from capx.execution.depth_sampling import DepthSamplingProbe
    if not hasattr(env, '_depth_sampling_probe'):
        env._depth_sampling_probe = DepthSamplingProbe()
    low = env.low_level_env
    context = low.robosuite_env.sim._render_context_offscreen
    profile = env._depth_sampling_probe.measure(context)
    return {name: copy.deepcopy(profile) for name in low.render_camera_names}


def public_robot_calibration(env):
    """Static robot calibration only, suitable for a parent motion-transform hook.

    Uses robot-model MJCF and the already installed IK URDF, never sim.data,
    object state, contacts, observations or evaluation results. No model download
    fallback is allowed. Returning this metadata does not alter the legacy API.
    The caller must explicitly enable transform_target for its experimental arm.
    """
    import hashlib
    from capx.execution.control_frames import calibration_from_xml
    from capx.utils.robot_urdf import resolve_robot_urdf_path
    robot = env.low_level_env.robosuite_env.robots[0]
    model = robot.robot_model
    gripper = robot.gripper.get('right')
    if gripper is None or not getattr(gripper, 'file', None):
        raise ValueError('Configured right gripper static model is required')
    api = env._apis['FrankaControlApi']
    urdf = Path(resolve_robot_urdf_path('panda_description')).expanduser()
    if not urdf.is_file():
        raise FileNotFoundError('Configured local Panda IK URDF is unavailable; no download attempted')
    sources = []
    for role, path in [('robot_mjcf', Path(model.file)),
                       ('mount_mjcf', Path(model.base.file)), ('ik_urdf', urdf),
                       ('gripper_mjcf', Path(gripper.file))]:
        path = path.resolve()
        sources.append(dict(role=role, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return calibration_from_xml(
        model.get_xml(), urdf.read_text(), public_base=model.base.root_body,
        actual_base=model.naming_prefix + 'link0', actual_parent=model.naming_prefix + 'link7',
        actual_hand=model.naming_prefix + 'right_hand',
        # This is the explicitly defined virtual public TCP, matching _robot_tcp
        # and AuditedStack; it is not inferred from any simulator object state.
        actual_tcp_offset=[0., 0., .107], api_tcp_to_hand_offset=api._TCP_OFFSET,
        sources=sources)


def public_sensors(raw, camera_names):
    """Copy a strict deployment sensor whitelist, including camera calibration."""
    result = {key: np.asarray(raw[key]).copy()
              for key in ('robot_joint_pos', 'robot_cartesian_pos', 'robot_gripper_joint_pos') if key in raw}
    for name in camera_names:
        camera = raw.get(name, {})
        if not isinstance(camera, dict):
            continue
        result[name] = {key: np.asarray(camera[key]).copy()
                        for key in ('pose', 'pose_mat', 'intrinsics') if key in camera}
        result[name]['images'] = {key: np.asarray(value).copy()
                                  for key, value in camera.get('images', {}).items()
                                  if key in ('rgb', 'depth')}
    return result



def stack_public_observation(env):
    """Adapter-only Stack observation: add both real finger encoders, no truth.

    AuditedStack's original implementation, physical evaluator and API callback
    remain unchanged. Read once so RGB-D and proprioception share the same step;
    never infer the second finger from the legacy normalized first-finger value.
    """
    raw = env.low_level_env.get_observation()
    result = public_sensors(raw, env.low_level_env.render_camera_names)
    if 'robot0_gripper_qpos' in raw:
        result['robot_gripper_joint_pos'] = np.asarray(raw['robot0_gripper_qpos'], float).copy()
    result['full_prompt'] = copy.deepcopy(env._full_prompt)
    return result


def _robot_tcp(low, raw):
    """Robot FK only: same hand-to-TCP +.107 m convention as audited Stack.

    This replaces stale legacy gripper-site caches, never object poses/contact or
    success flags. A physical robot provides the equivalent encoder/FK signal.
    """
    import viser.transforms as vtf
    sim = low.robosuite_env.sim
    hand_id = sim.model.body_name2id('robot0_right_hand')
    hand = vtf.SE3.from_rotation_and_translation(
        vtf.SO3(sim.data.xquat[hand_id]), sim.data.xpos[hand_id])
    tcp = vtf.SE3(low.base_link_wxyz_xyz).inverse() @ hand @ vtf.SE3.from_translation(
        np.array([0., 0., .107]))
    return np.concatenate([tcp.translation(), tcp.rotation().wxyz,
                            [raw['robot0_gripper_qpos'][0] / low.gripper_metric_length]])


def _task_objects(low, task):
    env = low.robosuite_env
    if task == 'lift':
        return {'red': env.cube}
    if task in ('stack', 'restack'):
        return {'red': env.cubeA, 'green': env.cubeB}
    raise ValueError('Unsupported private evaluation task')


def _table_geometry(low):
    """Private axis-aligned collision-box support plane; unknown stays None."""
    unknown = dict(table_top_world_z=None, table_center_world=None,
                   table_half_extent_xy=None, table_geometry_source='unavailable')
    try:
        sim = low.robosuite_env.sim
        gid = sim.model.geom_name2id('table_collision')
        if gid < 0 or int(sim.model.geom_type[gid]) != 6:  # MuJoCo BOX
            return unknown
        rotation = np.asarray(sim.data.geom_xmat[gid]).reshape(3, 3)
        # Axis-aligned extents must not over-certify a rotated tabletop's corners.
        absolute = np.abs(rotation)
        if not (np.allclose(absolute @ absolute.T, np.eye(3), atol=1e-8)
                and np.all(np.isclose(absolute, 0, atol=1e-8) | np.isclose(absolute, 1, atol=1e-8))):
            return unknown
        half = absolute @ np.asarray(sim.model.geom_size[gid])
        center = np.asarray(sim.data.geom_xpos[gid]).copy()
        if not np.isfinite(np.r_[half, center]).all() or np.any(half <= 0):
            return unknown
        return dict(table_top_world_z=float(center[2] + half[2]),
                    table_center_world=center, table_half_extent_xy=half[:2].copy(),
                    table_geometry_source='simulator_axis_aligned_table_collision_box_offline_only')
    except (AttributeError, KeyError, ValueError, IndexError):
        return unknown


def _evaluation_geometry(low, task):
    """Private simulator geometry for offline diagnostics, never online feedback."""
    sim = low.robosuite_env.sim
    d = sim.data
    base = low.base_link_idx
    return dict(schema='capx.private_evaluation_geometry.v1', task=task,
                **_table_geometry(low), objects={color: dict(label=color + ' cube', object_name=obj.name,
                                     root_body=obj.root_body, half_size=np.asarray(obj.size).copy())
                         for color, obj in _task_objects(low, task).items()},
                base_world=dict(position=d.xpos[base].copy(), quaternion_wxyz=d.xquat[base].copy(),
                                rotation=d.xmat[base].reshape(3, 3).copy()),
                tcp_offset_in_hand_m=[0., 0., .107], quaternion_order='wxyz', units='m,s',
                visibility='parent_only_offline_diagnostic_not_monitor_or_actor')


def _canonical_evaluation(low, task):
    """Fresh private FK and per-object bilateral contact, separate from scoring.

    Restack can hold green first; the original Runtime.grasp tests red only.
    This supplemental diagnostic checks both named objects independently without
    changing Runtime.qualifies, PhysicalCriteria, rewards or actor observations.
    """
    env = low.robosuite_env
    sim = env.sim
    d, m = sim.data, sim.model
    geometry = _evaluation_geometry(low, task)
    robot = env.robots[0]
    gripper = robot.gripper[robot.arms[0]] if isinstance(robot.gripper, dict) else robot.gripper
    left = set(gripper.important_geoms['left_fingerpad'])
    right = set(gripper.important_geoms['right_fingerpad'])
    contacts = [dict(geom1=m.geom_id2name(c.geom1), geom2=m.geom_id2name(c.geom2),
                     distance=float(c.dist), position=np.asarray(c.pos).copy())
                for c in d.contact[:d.ncon]]
    objects = {}
    for color, obj in _task_objects(low, task).items():
        bid = m.body_name2id(obj.root_body)
        object_geoms = set(obj.contact_geoms)
        pairs = [c for c in contacts if c['geom1'] in object_geoms or c['geom2'] in object_geoms]
        def touching(fingers):
            return any((c['geom1'] in object_geoms and c['geom2'] in fingers) or
                       (c['geom2'] in object_geoms and c['geom1'] in fingers) for c in pairs)
        left_contact, right_contact = touching(left), touching(right)
        objects[color] = dict(**geometry['objects'][color],
                              position_world=d.xpos[bid].copy(), quaternion_wxyz=d.xquat[bid].copy(),
                              rotation_world=d.xmat[bid].reshape(3, 3).copy(),
                              speed=float(np.linalg.norm(d.get_body_xvelp(obj.root_body))),
                              angular_speed=float(np.linalg.norm(d.get_body_xvelr(obj.root_body))),
                              contact_left_fingerpad=left_contact, contact_right_fingerpad=right_contact,
                              bilateral_contact=left_contact and right_contact, contacts=pairs)
    hand = m.body_name2id('robot0_right_hand')
    ee = low.gripper_link_idx
    obs = env._get_observations(force_update=True)
    return dict(schema='capx.private_evaluation.v1', task=task, sim_time=float(d.time),
                frame='world', quaternion_order='wxyz', units='m,s',
                **{key: geometry[key] for key in ('table_top_world_z', 'table_center_world', 'table_half_extent_xy', 'table_geometry_source')},
                base_world=geometry['base_world'], objects=objects,
                tcp_world=dict(position=d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ np.array([0., 0., .107]),
                               quaternion_wxyz=d.xquat[hand].copy()),
                ee_world=dict(position=d.xpos[ee].copy(), quaternion_wxyz=d.xquat[ee].copy()),
                gripper_opening_m=float(np.abs(obs['robot0_gripper_qpos']).sum()),
                visibility='parent_only_offline_diagnostic_not_monitor_or_actor')


def _with_canonical(state, low, task):
    return dict(state, canonical=_canonical_evaluation(low, task))


class PublicLowLevel:
    """Capability proxy for trusted perception/control code, with no truth passthrough.

    Perception diagnostics are stored on this proxy, not exposed as simulator state.
    The untrusted actor only sees RPC function names, never this object.
    """
    def __init__(self, low, seed):
        self.__low = low
        self.__rng = np.random.default_rng(np.random.SeedSequence([seed, 831]))
        self.render_camera_names = tuple(low.render_camera_names)

    def get_observation(self):
        raw = self.__low.get_observation()
        result = public_sensors(raw, self.render_camera_names)
        result['robot_cartesian_pos'] = _robot_tcp(self.__low, raw)
        # Both finger encoders are public proprioception. Do not infer the
        # second finger through a mimic assumption when contact is asymmetric.
        if 'robot0_gripper_qpos' in raw:
            result['robot_gripper_joint_pos'] = np.asarray(raw['robot0_gripper_qpos'], float).copy()
        return result

    def next_grasp_seed(self):
        return int(self.__rng.integers(0, 2**31 - 1))

    def move_to_joints_blocking(self, joints, **kwargs):
        return self.__low.move_to_joints_blocking(joints, **kwargs)

    def _set_gripper(self, value):
        return self.__low._set_gripper(value)

    def _step_once(self):
        return self.__low._step_once()


def _runtime_factory(*args, **kwargs):
    from capx.multitask.runtime import Runtime
    return Runtime(*args, **kwargs)


def _vision_api_factory(low):
    from capx.integrations.franka.control import FrankaControlApi
    return FrankaControlApi(low)


def _validate(task, seed, api_budget):
    if task not in TASKS:
        raise ValueError('Unsupported trusted Vision task: ' + str(task))
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32)')
    if type(api_budget) is not int or api_budget <= 0:
        raise ValueError('API budget must be a positive integer')


class VisionTaskEnvironment:
    """AuditedStack-compatible broker around unchanged task-specific Runtime."""
    def __init__(self, task, artifact_root, seed, snapshot=None, api_budget=128):
        _validate(task, seed, api_budget)
        self.task = task
        self._artifact_root = Path(artifact_root)
        self.cfg = SimpleNamespace(privileged=False, enable_render=True,
                                   feedback_api_budget=api_budget, execution_timeout=90.)
        self._runtime = None
        self._request_audit = None
        self._trace_path = None
        self.reset(seed=seed, options={'snapshot': snapshot} if snapshot is not None else None)

    @property
    def _api_calls(self):
        return self._runtime.calls

    @property
    def _motion_calls(self):
        return self._runtime.motion

    def _write(self, kind, **row):
        from capx.envs.reproducibility import json_value
        if self._trace_path is not None:
            with self._trace_path.open('a') as stream:
                stream.write(json.dumps(dict(kind=kind, **row), default=json_value,
                                        allow_nan=False) + '\n')

    def reset(self, *, seed=None, options=None):
        snapshot = (options or {}).get('snapshot')
        if seed is None:
            seed = snapshot['seed'] if snapshot is not None else self.seed
        _validate(self.task, seed, self.cfg.feedback_api_budget)
        if snapshot is not None and (snapshot.get('seed') != seed or snapshot.get('task') != self.task):
            raise ValueError('Snapshot task/seed mismatch')
        self.close()
        if hasattr(self, '_feedback_monitor'):
            del self._feedback_monitor
        self.seed = seed
        self._step_count = 0
        root = self._artifact_root / uuid.uuid4().hex
        try:
            runtime = self._runtime = _runtime_factory(self.task, seed, root, render=True)
            # Exact comparison includes model, physics, controller warmstart and RNG.
            if snapshot is not None and runtime.initial != snapshot:
                raise ValueError('Snapshot replay mismatch: physics/RNG/model/controller history')
            self.low_level_env = runtime.low
            self._public_low = PublicLowLevel(runtime.low, seed)
            api = _vision_api_factory(self._public_low)
            runtime.actor.close()
            runtime.api = api
            self._apis = {'FrankaControlApi': api}
            self._functions = runtime.functions = api.functions()
            from capx.execution.sandbox import SandboxProcess
            self._actor = runtime.actor = SandboxProcess(self._functions, callback=self._call_api)
            self._task_prompt = runtime.task_module.PROMPT
            self._full_prompt = [
                {'role': 'system', 'content': 'Generate a complete Python robot program using only the listed APIs and numpy. No files, network, env, obs, globals or private attributes. Physical success is independently evaluated.'},
                {'role': 'user', 'content': self._task_prompt + '\nAPI documentation:\n' + api.combined_doc()},
            ]
            self._trace_path = runtime.trace
            from capx.utils.request_audit import RequestAudit
            self._request_audit = RequestAudit(root / 'requests.jsonl').__enter__()
            self.reset_info = dict(seed=seed, initial_state_hash=runtime.initial['state_hash'],
                                   task=self.task, mode='vision', snapshot_replay_verified=snapshot is not None,
                                   grasp_seed_scheme='SeedSequence([episode_seed,831])',
                                   snapshot_scope='initial_reset_and_settle_history')
            self._write('vision_boundary', **self.reset_info, public_api_class=type(api).__name__)
            return self._get_observation(), self.reset_info
        except BaseException:
            self.close()
            raise

    def _get_observation(self):
        result = self._public_low.get_observation()
        result['full_prompt'] = copy.deepcopy(self._full_prompt)
        return result

    def _call_api(self, name, args, kwargs):
        monitor = getattr(self, '_feedback_monitor', None)
        return (monitor.invoke(name, args, kwargs, self._execute_api) if monitor is not None
                else self._execute_api(name, args, kwargs))

    def _execute_api(self, name, args, kwargs):
        if self._api_calls >= self.cfg.feedback_api_budget:
            from capx.execution.sandbox import ExecutionInterrupted
            raise ExecutionInterrupted('Episode API budget exhausted (including recovery)')
        # Keep original Runtime physical trace and exception policy. No physical
        # state returned to the model/monitor beyond this public API's response.
        return self._runtime.call(name, args, kwargs)

    def step(self, code):
        self._step_count += 1
        result = self._runtime.execute(code, timeout=self.cfg.execution_timeout)
        info = result['info']
        info.update(code_blocks=self._step_count, task_completed=info['physical_success'],
                    http_requests=self._request_audit.count, llm_requests=self._request_audit.llm_count,
                    initial_state_hash=self.reset_info['initial_state_hash'], trace_path=str(self._trace_path),
                    task_prompt=self._task_prompt,
                    frame_timestamps=list(self.low_level_env._frame_timestamps),
                    wrist_frame_timestamps=list(getattr(self.low_level_env, '_wrist_frame_timestamps', [])))
        truncated = bool(result['truncated'] or info['execution_status'] in ('timeout', 'system_error'))
        return self._get_observation(), result['reward'], bool(result['reward']), truncated, info

    def enable_video_capture(self, *args, **kwargs):
        return self.low_level_env.enable_video_capture(*args, **kwargs)

    def get_video_frames(self, **kwargs):
        return self.low_level_env.get_video_frames(**kwargs)

    def capture_snapshot(self):
        return copy.deepcopy(self._runtime.initial)

    def public_depth_sampling(self):
        return public_depth_sampling(self)

    def public_robot_calibration(self):
        return public_robot_calibration(self)

    def evaluation_state(self):
        """Parent-only independent evaluator; must never enter actor/monitor inputs."""
        return _with_canonical(self._runtime.state(), self.low_level_env, self.task)

    def evaluation_geometry(self):
        """Private named dimensions/base calibration for offline evaluation only."""
        return _evaluation_geometry(self.low_level_env, self.task)

    def close(self):
        if self._request_audit is not None:
            self._request_audit.__exit__()
            self._request_audit = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


def make_environment(task, artifact_root, seed, snapshot=None, api_budget=128):
    """Return an already-reset environment; reset_info records verified start hash."""
    _validate(task, seed, api_budget)
    if task == 'stack':
        from capx.experiments.pilot import environment
        env = environment('vision', Path(artifact_root))
        try:
            env.cfg.feedback_api_budget = api_budget
            _, env.reset_info = env.reset(seed=seed, options={'snapshot': snapshot} if snapshot is not None else None)
            env._get_observation = lambda: stack_public_observation(env)
            env.capture_snapshot = lambda: env.low_level_env.capture_snapshot()
            env.public_robot_calibration = lambda: public_robot_calibration(env)
            env.public_depth_sampling = lambda: public_depth_sampling(env)
            from capx.envs.physical_evaluation import physical_state
            env.evaluation_state = lambda: _with_canonical(physical_state(env.low_level_env), env.low_level_env, 'stack')
            env.evaluation_geometry = lambda: _evaluation_geometry(env.low_level_env, 'stack')
            return env
        except BaseException:
            env.close()
            raise
    return VisionTaskEnvironment(task, artifact_root, seed, snapshot, api_budget)


def prepare_scene(task, seed, dest):
    """Create and immediately replay one exclusive scene, returning (entry,snapshot).

    Paths in entry are relative to dest/scenes.json, matching pilot.load_scene.
    """
    _validate(task, seed, 128)
    from capx.experiments.pilot import prepare, dump, digest, load_scene
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dest / 'scenes.json'
    if task == 'stack':
        prepare(manifest, dest, [seed])
        entry = json.loads(manifest.read_text())['scenes'][0]
        return dict(entry, task=task, snapshot_replay_verified=True), load_scene(manifest, entry)
    env = make_environment(task, dest / 'prepare', seed)
    try:
        snapshot = env.capture_snapshot()
    finally:
        env.close()
    replay = make_environment(task, dest / 'replay-check', seed, snapshot)
    replay.close()
    path = dest / 'snapshots' / f'seed-{seed}.json'
    dump(path, snapshot)
    entry = dict(task=task, seed=seed, snapshot=str(path.relative_to(dest)),
                 state_hash=snapshot['state_hash'], model_hash=snapshot['model_hash'],
                 snapshot_digest=digest(snapshot), snapshot_replay_verified=True,
                 snapshot_scope='initial_reset_and_settle_history',
                 grasp_seed_scheme='SeedSequence([episode_seed,831])', valid=True)
    dump(manifest, dict(version=1, task=task, scenes=[entry]))
    return entry, snapshot
