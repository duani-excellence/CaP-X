"""Public, robot-calibrated grasp geometry; no simulator, actor state or evaluator.

Contact-GraspNet's matrix X is its predicted baseline/closing direction.
Panda hand/TCP Y is the actual closing direction in this project's MJCF:
right_gripper Rz(-pi/2) times finger Rz(pi/2) leaves the joint's +Y axis.
The existing vision client only changes the camera frame and translates the
candidate origin. It does NOT apply this gripper-axis mapping.

The swept boxes below conservatively bound collision meshes, not free space.
A PASS only says supplied visible neighbor points miss those boxes. Occlusion,
unknown objects and the joint-space transit to the above-target pose still need
separate handling. These are not calibrated perception uncertainty estimates.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.) > 1e-3:
        raise ValueError('finite unit WXYZ quaternion required')
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()


def cgn_upright_panda(quaternion_wxyz):
    """Project CGN baseline to horizontal, map to Panda TCP Y, point tool Z down.

    Input is the unremapped CGN candidate quaternion already in robot-base frame.
    Do not call twice or pass an already-Panda-frame quaternion.
    Position/TCP origin calibration is deliberately left to the parent producer.
    """
    raw = _rotation(quaternion_wxyz)
    horizontal = raw[:2, 0]
    strength = float(np.linalg.norm(horizontal))
    if strength < .25:
        raise ValueError('CGN baseline horizontal projection too small to define yaw')
    y = np.r_[horizontal / strength, 0.]
    z = np.array([0., 0., -1.])
    x = np.cross(y, z)
    rotation = np.column_stack([x, y, z])
    q = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
    if q[np.flatnonzero(abs(q) > 1e-12)[0]] < 0: q = -q
    return {'quaternion_wxyz': q.tolist(), 'source_axis': 'contact_graspnet_x',
            'target_axis': 'panda_tcp_y', 'raw_baseline': raw[:, 0].tolist(),
            'public_closing_axis': y.tolist(), 'horizontal_projection': strength,
            'raw_tool_tilt_deg': float(np.degrees(np.arccos(np.clip(-raw[2, 2], -1, 1)))),
            'policy': 'upright_projection_after_explicit_gripper_frame_conversion',
            'limitation': 'upright projection changes the candidate; collision and grasp success remain unverified'}


def _panda_boxes(opening):
    """Bounds in hand/TCP frame, rounded outwards from the checked robot assets.

    TCP is hand +107 mm. Finger body origin is hand +52.4 mm.
    Mesh finger.stl bounds are X[-.01047945,.01049482],
    Y[-.00013264,.02640338], Z[.00013170,.05384903] m.
    Pads use the XML's exact box dimensions/offset. Palm collision STL already
    has its XML two opposite Z rotations cancelled in the hand frame.
    opening is symmetric joint separation, not clearance between pad faces.
    """
    q = opening / 2.
    return {
        'finger1_body': ([-.01050, q-.00014, -.05447], [.01050, q+.02641, -.00075]),
        'finger2_body': ([-.01050, -q-.02641, -.05447], [.01050, -q+.00014, -.00075]),
        'finger1_pad': ([-.008, q-.0005, -.0216], [.008, q+.0075, -.0056]),
        'finger2_pad': ([-.008, -q-.0075, -.0216], [.008, -q+.0005, -.0056]),
        'palm': ([-.03164, -.10399, -.13293], [.03162, .10043, -.04103]),
    }


def panda_open_sweep(position, quaternion_wxyz, neighbor_points, *, opening_m=.08,
                     z_approach=.1, padding_m=.004):
    """Check visible non-target RGB-D points against an open Panda descent sweep.

    neighbor_points is {known_neighbor_name: Nx3 robot-base points}. An empty or
    sparse cloud for a known neighbor gives UNCERTAIN; do not omit invisible
    known neighbors to manufacture PASS. The parent owns identity/freshness.
    The nominal approach is fixed orientation, vertical, from target+Z dz.
    It does not model a curved joint-space trajectory or approve transit.
    """
    p = np.asarray(position, float); r = _rotation(quaternion_wxyz)
    if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('finite XYZ target required')
    if not isinstance(neighbor_points, dict): raise ValueError('named public neighbor point clouds required')
    if not np.isfinite([opening_m, z_approach, padding_m]).all(): raise ValueError('finite sweep parameters required')
    if not 0 <= opening_m <= .0801 or not 0 <= z_approach <= .30 or not 0 <= padding_m <= .02:
        raise ValueError('sweep parameters exceed calibrated robot range')
    if not np.allclose(r[:, 2], [0., 0., -1.], atol=1e-6):
        raise ValueError('sweep requires exactly upright Panda orientation')
    boxes = _panda_boxes(float(opening_m)); collisions = {}; sparse = []; counts = {}
    for name, value in neighbor_points.items():
        pts = np.asarray(value, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError('finite Nx3 public points required')
        counts[name] = len(pts)
        if len(pts) < 30: sparse.append(name)
        local = (pts - p) @ r
        hits = {}
        for label, (low, high) in boxes.items():
            lo = np.asarray(low) - padding_m; hi = np.asarray(high) + padding_m
            lo[2] -= z_approach  # world +Z equals local -Z under upright orientation
            count = int(np.count_nonzero(np.all((local >= lo) & (local <= hi), axis=1)))
            if count: hits[label] = count
        if hits: collisions[name] = hits
    status = 'FAIL' if collisions else ('UNCERTAIN' if sparse else 'PASS')
    return {'status': status,
            'reason': 'visible_neighbor_intersects_gripper_sweep' if collisions else
                      ('known_neighbor_insufficiently_visible' if sparse else 'no_visible_neighbor_point_in_sweep'),
            'collisions': collisions, 'insufficient_neighbors': sparse, 'point_counts': counts,
            'proof': 'visible_points_only_not_collision_free_certificate',
            'motion_scope': 'fixed_upright_vertical_approach_only',
            'calibration': {'source': 'robosuite Panda panda_gripper.xml + collision STL bounds',
                            'opening_axis': 'panda_tcp_y', 'tcp_offset_hand_z_m': .107,
                            'opening_m': float(opening_m), 'padding_m': float(padding_m),
                            'z_approach_m': float(z_approach)},
            'unverified': ['occluded_volume', 'unknown_objects', 'joint_space_transit', 'grasp_success']}


def cgn_panda_pose(transform,hand_to_tcp):
    """Unprojected candidate under explicit CGN-hand-origin correspondence.

    CGN build_6d_grasp subtracts its gripper_depth (0.1034 m) from the predicted
    contact midpoint. That origin must not be passed as a TCP position. The
    configured robot hand-to-TCP translation is applied AFTER the axis mapping.
    This is an auditable candidate conversion, not a grasp/motion certificate;
    upstream morphology correspondence still needs physical calibration.
    """
    t=np.asarray(transform,float);offset=np.asarray(hand_to_tcp,float)
    if (t.shape!=(4,4) or offset.shape!=(3,) or not np.isfinite(t).all() or not np.isfinite(offset).all()
        or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
        or np.linalg.det(t[:3,:3])<0):raise ValueError('Rigid grasp transform and configured TCP translation required')
    r=np.column_stack([-t[:3,1],t[:3,0],t[:3,2]])
    return dict(position=(t[:3,3]+r@offset).tolist(),quaternion_wxyz=np.roll(Rotation.from_matrix(r).as_quat(),1).tolist(),
        rotation=r.tolist(),permission=False,assumption='CGN frame origin corresponds to configured Panda hand; not physically calibrated here',
        source='CGN build_6d_grasp and explicit configured hand-to-TCP translation')
