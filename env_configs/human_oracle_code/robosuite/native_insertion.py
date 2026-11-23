"""Reference only. Same feature-relative insertion for every declared pair."""
import numpy as np
for index in oracle_insertion_bindings():
    # Previous physical operations can move another object: only the identity
    # binding is retained, and feature geometry is refreshed at this boundary.
    pair = oracle_insertion(index)
    handle = pair['handle']
    quat = pair['grasp_quaternion']
    up = np.array([0.,0.,.15])
    open_gripper()
    goto_pose(handle + up, quat)
    goto_pose(handle, quat)
    close_gripper()
    goto_pose(handle + up, quat)
    target = pair['target']
    target_quat = pair['target_quaternion']
    goto_pose(target + up, target_quat)
    goto_pose(target, target_quat)
    goto_pose(target - np.array([0.,0.,.02]), target_quat)
    open_gripper()
    goto_pose(target + up, target_quat)
