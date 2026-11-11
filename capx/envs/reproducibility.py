"""Trusted initial-state snapshots. Never exported to actor observation/RPC."""
from __future__ import annotations
import copy
import hashlib
import json

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'),
                                    default=json_value, allow_nan=False).encode()).hexdigest()


class InitialSnapshotMixin:
    """v1 snapshots intentionally cover episode starts, not arbitrary controllers."""
    def _seed_episode(self, seed):
        if seed is None:
            seed = int(np.random.SeedSequence().entropy) % (2**32)
        self.episode_seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._grasp_rng = np.random.default_rng(np.random.SeedSequence([seed, 831]))
        env = self.robosuite_env
        env.seed = int(seed)
        env.rng = np.random.default_rng(seed)
        def bind(sampler):
            if sampler is None:
                return
            sampler.rng = env.rng
            for child in getattr(sampler, 'samplers', {}).values():
                bind(child)
        bind(getattr(env, 'placement_initializer', None))
        self._gripper_fraction = 1.0

    def next_grasp_seed(self):
        return int(self._grasp_rng.integers(0, 2**31 - 1))

    def _physical_arrays(self):
        sim = self.robosuite_env.sim
        return {'sim_state': sim.get_state().flatten().tolist(),
                'ctrl': sim.data.ctrl.tolist(),
                'qacc_warmstart': sim.data.qacc_warmstart.tolist()}

    def physical_state_hash(self):
        # Exact floats, not rounded coordinates: detects even small state drift.
        return state_hash(self._physical_arrays())

    def _snapshot_model_hash(self):
        return hashlib.sha256(self.robosuite_env.model.get_xml().encode()).hexdigest()

    def capture_snapshot(self):
        if self._sim_step_count != 0:
            raise ValueError('v1 only supports settled initial-state snapshots')
        return dict(version=1, seed=self.episode_seed, model_hash=self._snapshot_model_hash(),
                    physics=self._physical_arrays(), state_hash=self.physical_state_hash(),
                    rng_state=copy.deepcopy(self.robosuite_env.rng.bit_generator.state),
                    grasp_rng_state=copy.deepcopy(self._grasp_rng.bit_generator.state),
                    current_joints=self._current_joints.tolist(), gripper_fraction=self._gripper_fraction)

    def restore_snapshot(self, snapshot):
        if snapshot.get('version') != 1 or state_hash(snapshot['physics']) != snapshot['state_hash']:
            raise ValueError('Invalid snapshot version or state hash')
        self.reset(seed=snapshot['seed'])
        if self._snapshot_model_hash() != snapshot['model_hash']:
            raise ValueError('Snapshot model does not match this environment')
        # Initial controller state is recreated by reset; arbitrary mid-episode
        # controller snapshots are deliberately unsupported.
        if self.physical_state_hash() != snapshot['state_hash']:
            raise ValueError('Initial state no longer reproducible in this runtime')
        sim = self.robosuite_env.sim
        sim.set_state_from_flattened(np.asarray(snapshot['physics']['sim_state']))
        sim.data.ctrl[:] = snapshot['physics']['ctrl']
        sim.data.qacc_warmstart[:] = snapshot['physics']['qacc_warmstart']
        sim.forward()
        self.robosuite_env.rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])
        self._grasp_rng.bit_generator.state = copy.deepcopy(snapshot['grasp_rng_state'])
        self._current_joints = np.asarray(snapshot['current_joints'])
        self._gripper_fraction = snapshot['gripper_fraction']
        self.initial_state_hash = self.physical_state_hash()
        return self.get_observation(), {'initial_state_hash': self.initial_state_hash, 'seed': self.episode_seed}
