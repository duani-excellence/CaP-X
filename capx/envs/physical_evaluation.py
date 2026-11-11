"""Parent-only physical evaluation for the Robosuite Stack task (SI units)."""
import numpy as np


class PhysicalCriteria:
    def __init__(self, initial):
        self.initial_z = initial['objects']['primary']['position'][2]
        self.grasp_since = self.placement_since = None
        self.grasp_success = self.slipped = self.stable_placement = False

    def update(self, state):
        t = state['sim_time']
        obj = state['objects']['primary']
        held = state['grasp_contact'] and obj['position'][2] > self.initial_z + .05
        self.grasp_since = (t if self.grasp_since is None else self.grasp_since) if held else None
        if self.grasp_since is not None and t - self.grasp_since >= .5:
            self.grasp_success = True
        if self.grasp_success and not state['grasp_contact'] and state.get('gripper_command', 0.) < .5 and state['gripper_opening'] < .05:
            self.slipped = True
        if self.grasp_success and state.get('gripper_command', 0.) < .5 and obj['position'][2] < self.initial_z + .02 and not state['raw_success']:
            self.slipped = True
        still = all(o['speed'] < .02 and o.get('angular_speed', 0.) < .2 for o in state['objects'].values())
        placed = still and state['raw_success'] and not state['grasp_contact'] and state['gripper_opening'] > .05 and obj['speed'] < .02
        self.placement_since = (t if self.placement_since is None else self.placement_since) if placed else None
        self.stable_placement = self.placement_since is not None and t - self.placement_since >= .5


def physical_state(low):
    env = low.robosuite_env
    sim = env.sim
    d, m = sim.data, sim.model
    ee_id = low.gripper_link_idx
    ee_p = d.xpos[ee_id].copy()
    ee_R = d.xmat[ee_id].reshape(3, 3).copy()
    objects, relative = {}, {}
    for name, body_id in [('primary', env.cubeA_body_id), ('secondary', env.cubeB_body_id)]:
        pos, quat = d.xpos[body_id].copy(), d.xquat[body_id].copy()
        body_name = m.body_id2name(body_id)
        objects[name] = dict(position=pos, quaternion_wxyz=quat,
                             speed=float(np.linalg.norm(d.get_body_xvelp(body_name))),
                             angular_speed=float(np.linalg.norm(d.get_body_xvelr(body_name))))
        relative[name] = dict(position=ee_R.T @ (pos-ee_p), rotation=ee_R.T @ d.xmat[body_id].reshape(3,3))
    contacts = [dict(geom1=m.geom_id2name(c.geom1), geom2=m.geom_id2name(c.geom2),
                     distance=float(c.dist), position=c.pos.copy()) for c in d.contact[:d.ncon]]
    robot = env.robots[0]
    gripper = robot.gripper['right'] if isinstance(robot.gripper, dict) else robot.gripper
    grasp = bool(env._check_grasp(gripper=gripper, object_geoms=env.cubeA.contact_geoms))
    obs = env._get_observations(force_update=True)
    hand_id = m.body_name2id('robot0_right_hand')
    tcp_position = d.xpos[hand_id] + d.xmat[hand_id].reshape(3,3) @ np.array([0.,0.,.107])
    return dict(tcp=dict(position=tcp_position, quaternion_wxyz=d.xquat[hand_id].copy()), sim_time=float(d.time), objects=objects,
                ee=dict(position=ee_p, quaternion_wxyz=d.xquat[ee_id].copy()),
                gripper_opening=float(np.sum(np.abs(obs['robot0_gripper_qpos']))),
                contacts=contacts, relative_poses=relative, grasp_contact=grasp, gripper_command=low._gripper_fraction,
                raw_success=bool(env._check_success()), frame='world', units='m,s', quaternion='WXYZ')
