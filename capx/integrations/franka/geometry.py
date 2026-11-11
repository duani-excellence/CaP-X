"""Conventions: camera optical +X right/+Y down/+Z forward; poses in robot base.
All public pose quaternions are WXYZ. TCP->IK hand offset is local (0,0,-.107)m.
Contact-GraspNet candidate->command TCP offset is local (0,0,+.12)m, retained
from the legacy baseline (not a new empirical calibration); these are distinct transforms, not cancellations.
"""
import numpy as np

GRASP_TO_TCP_METERS = np.array([0., 0., .12])
TCP_TO_HAND_METERS = np.array([0., 0., -.107])

def select_grasp(poses, scores):
    poses, scores = np.asarray(poses), np.asarray(scores).reshape(-1)
    if poses.shape != (len(scores),4,4) or len(scores) == 0:
        raise ValueError('No valid grasp candidates')
    valid = np.isfinite(poses).all(axis=(1,2)) & np.isfinite(scores)
    for i, T in enumerate(poses):
        valid[i] &= np.allclose(T[3], [0,0,0,1], atol=1e-5) and np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-4) and np.isclose(np.linalg.det(T[:3,:3]), 1, atol=1e-4)
    if not valid.any():
        raise ValueError('No finite rigid grasp candidates')
    idx = int(np.argmax(np.where(valid, scores, -np.inf)))
    return poses[idx], idx, float(scores[idx])
