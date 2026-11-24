"""Evaluator-only physical perturbation. Never imported by online skills/planners.

An external wrench is a stress intervention, not an actor action, snapshot reset,
or substitute for recovery. Whether it actually broke an effect is independently
reported, never inferred from the intervention label.
"""
import numpy as np


def disturbance_wrench(spec, mass, gravity):
    """Evaluator-only, mass-normalized impulse; never an online recovery action."""
    if 'impulse_velocity_world' not in spec:
        return np.asarray(spec['wrench_world'], dtype=float)
    velocity=np.asarray(spec['impulse_velocity_world'], dtype=float)
    duration=float(spec['duration_seconds'])
    if velocity.shape!=(3,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity)>1:
        raise ValueError('Bounded finite impulse velocity required')
    if not np.isfinite(mass) or mass<=0 or not 0<duration<=1:
        raise ValueError('Positive body mass and bounded duration required')
    acceleration=velocity/duration
    if spec.get('compensate_gravity',False):acceleration-=np.asarray(gravity)
    return np.r_[mass*acceleration,np.zeros(3)]


def perturb_completed_effect(boundary, object_name, spec):
    from capx.experiments.native_tasks import physical_state
    env = boundary.low.robosuite_env
    body = next(o.root_body for o in env.model.mujoco_objects if o.name == object_name)
    bid = env.sim.model.body_name2id(body)
    mass=float(env.sim.model.body_mass[bid])
    wrench = disturbance_wrench(spec,mass,env.sim.model.opt.gravity)
    if wrench.shape != (6,) or not np.isfinite(wrench).all():
        raise ValueError('Finite six-dimensional evaluator wrench required')
    duration = float(spec['duration_seconds'])
    settle = float(spec['settle_seconds'])
    if not 0 < duration <= 1 or not 0 <= settle <= 2:
        raise ValueError('Bounded disturbance timing required')
    before = physical_state(env)
    try:
        until = env.sim.data.time + duration
        while env.sim.data.time < until:
            env.sim.data.xfrc_applied[bid] = wrench
            boundary.low._step_once()
    finally:
        env.sim.data.xfrc_applied[bid] = 0
    until = env.sim.data.time + settle
    while env.sim.data.time < until:
        boundary.low._step_once()
    after = physical_state(env)
    boundary.log('controlled_physical_disturbance', object=object_name, settings=spec,
                 actual_wrench_world=wrench.tolist(),body_mass=mass,
                 private_before=before, private_after=after, actor_received_private_state=False)
    return dict(before=before, after=after)
