"""Read-only arithmetic audit of the upstream failure-scene perturbation.

Executes the exact selected upstream functions and MjSimState class on sentinel
arrays. This is a protocol counterexample, NOT a rollout or a patched protocol.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def selected(path, names):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError('Upstream definitions changed')
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def audit(protocol, binding):
    helpers = selected(protocol, {'build_object_xy_qpos_adrs', 'perturb_object_xy'})
    cls = selected(binding, {'MjSimState'})['MjSimState']
    model = SimpleNamespace(body_jntadr=[0], jnt_type=[0], jnt_qposadr=[2], nq=9, nv=8, na=0)
    sim = SimpleNamespace(model=model)
    env = SimpleNamespace(env=SimpleNamespace(sim=sim, obj_body_id={'sentinel': 0}))
    initial = cls(time=3.0, qpos=np.arange(9, dtype=float), qvel=np.zeros(8))
    flat = initial.flatten()
    updated, deltas = helpers['perturb_object_xy'](
        flat, helpers['build_object_xy_qpos_adrs'](env), .02, np.random.RandomState(7))
    restored = cls.from_flattened(updated, sim)
    delta = restored.qpos - initial.qpos
    dx, dy = deltas['sentinel']
    return dict(mode='synthetic_protocol_arithmetic_counterexample_NOT_physics',
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),
        qpos_address=2, intended_xy_delta=[dx, dy], actual_xy_delta=delta[2:4].tolist(),
        changed_qpos_indices=np.flatnonzero(delta).tolist(),
        previous_qpos_delta=float(delta[1]), original_array_preserved=bool(np.array_equal(flat, initial.flatten())),
        time_prefix_offset_confirmed=bool(np.isclose(delta[1], dx) and np.isclose(delta[2], dy) and delta[3] == 0),
        official_code_modified=False, corrected_protocol_executed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(audit(a.protocol, a.binding), f, indent=2)
