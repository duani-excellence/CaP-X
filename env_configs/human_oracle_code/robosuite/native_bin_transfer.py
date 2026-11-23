"""Gate 2 reference only: native bin centers and object geometry are privileged.
The same parameterized transfer is used for every object; this is not an agent.
"""
import numpy as np

for index, name in oracle_bindings():
    obj = oracle_object(name)
    pos = obj['position']
    quat = obj['grasp_quaternion']
    up = np.array([0., 0., .20])
    open_gripper()
    goto_pose(pos + up, quat)
    goto_pose(pos, quat)
    close_gripper()
    goto_pose(pos + up, quat)
    region = oracle_region(index)
    target = region['center'] - obj['bottom_offset']
    # Choose a nearby interior point, with clearance from the public shape
    # radius. Goal regions do not require their exact center coordinates.
    margin = obj['horizontal_radius'] + .01
    target[:2] = np.clip(pos[:2], region['center'][:2]-region['half_extent_xy']+margin,
                        region['center'][:2]+region['half_extent_xy']-margin)
    # A bin goal constrains position, not yaw. Use the same canonical downward
    # placement orientation for every object, rather than preserving arbitrary
    # source yaw at the edge of the robot workspace.
    place_quat = np.array([0., 0., 1., 0.])
    goto_pose(target + up, place_quat)
    goto_pose(target + np.array([0., 0., .01]), place_quat)
    open_gripper()
    goto_pose(target + up, place_quat)
