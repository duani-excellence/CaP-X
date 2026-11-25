"""Optional mature PyRoKi adapter. CPU supported; no environment or robot handles.

Caller supplies public obstacle geometry in the configured IK frame and the
measured start configuration. This does not infer unseen free space, invent an
attached object model, or dispatch a returned trajectory.
"""
import json
from pathlib import Path
import numpy as np
from capx.execution.bounded_ik import URDFFK
from capx.execution.preplanning import validate_trajectory


def plan_pyroki(urdf_path,sphere_path,start,target,obstacles,*,timesteps=12,dt=.2,held=False):
    from capx.utils.robot_urdf import load_urdf_for_robot
    from capx.serving.launch_pyroki_server import _build_world_coll,set_min_distance_from_limits
    from .pyroki_snippets._trajopt import solve_trajopt
    import pyroki as pk
    fk=URDFFK(urdf_path);start=fk.validate(start);start_pose=fk.pose(start);target=np.asarray(target,float)
    urdf=set_min_distance_from_limits(load_urdf_for_robot(urdf_path))
    robot=pk.Robot.from_urdf(urdf)
    from .collision_audit import fixed_assembly_pairs,refine_sphere_contacts
    rigid_pairs=fixed_assembly_pairs(urdf)
    coll=pk.collision.RobotCollision.from_sphere_decomposition(json.loads(Path(sphere_path).read_text()),urdf,user_ignore_pairs=rigid_pairs)
    world=_build_world_coll(obstacles)
    fixed=tuple(i for i,name in enumerate(fk.joint_names) if 'finger' in name)
    path=solve_trajopt(robot,coll,world,fk.target_link,start_pose[4:],start_pose[:4],target[4:],target[:4],timesteps,dt,start_joint_position=start,fixed_joint_indices=fixed)
    import jax.numpy as jnp
    samples=[]
    for a,b in zip(np.asarray(path)[:-1],np.asarray(path)[1:]):
        count=max(1,int(np.ceil(np.max(abs(b-a))/.01)))
        if len(samples)+count>2048:raise ValueError('Self collision sample budget exhausted')
        samples.extend(a*(1-t)+b*t for t in np.linspace(0,1,count+1))
    distances=np.asarray(coll.compute_self_collision_distance(robot,jnp.asarray(samples)))
    minimum=float(np.min(distances))
    self_check=dict(status='PASS' if np.isfinite(minimum) and minimum>=0 else 'FAIL',
        minimum_signed_distance_m=minimum,samples=len(samples),scope='configured_sphere_pairs_sampled_not_continuous_proof')
    if minimum<.004:
        self_check['narrow_phase']=refine_sphere_contacts(urdf_path,coll,np.array(samples),distances)
        self_check['status']=self_check['narrow_phase']['status']
    self_check['fixed_assembly_exclusions']=rigid_pairs
    # Upstream objective penalizes collisions; it does not certify complete world
    # coverage or attached-body safety. Never report that evidence as PASS.
    validation=validate_trajectory(path,start,target,fk,held=held)
    if fixed and np.max(abs(np.asarray(path)[:,fixed]-start[list(fixed)]))>.0001:
        validation.update(status='REJECTED',reason='noncommanded_finger_changed')
    return dict(backend='existing_pyroki_trajopt',trajectory=np.asarray(path).tolist(),validation=validation,
        self_collision=self_check,
        permission=False,attachment_included=False,world_source='caller_public_obstacles',
        limitations=['partial world is not free-space proof','attached shape collision model not yet integrated','optimizer soft costs require independent path validation'])
