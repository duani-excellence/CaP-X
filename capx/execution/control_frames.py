"""Static robot-frame calibration for parent-broker motion targets.

No simulator state or object geometry is read. Coordinates are transformed from
Robosuite's public mount base / virtual TCP into the existing IK client's root /
hand convention. This is a calibration adapter, not residual-motion correction.
The legacy API and B0 remain unchanged unless a caller explicitly applies it.

T_command = T_ikbase_publicbase @ T_public_tcp @ T_tcp_actualhand
            @ T_actualhand_ikhand @ inverse(T_api_tcp_to_hand).
The final inverse cancels goto_pose's existing *negative* .107 m local offset;
applying another negative offset here would double-count TCP compensation.
"""
from __future__ import annotations
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np


def _vector(value, length):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'Expected finite vector of length {length}')
    return result


def _rotation(wxyz):
    q = _vector(wxyz, 4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('Quaternion must be nonzero')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _quaternion(r):
    # Symmetric eigenproblem is stable at 180 degrees, unlike trace-only formulas.
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
                  [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
                  [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def _translation(vector):
    value = np.eye(4)
    value[:3, 3] = _vector(vector, 3)
    return value


def _se3(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4,4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0,0,0,1], atol=1e-10)
            or not np.allclose(a[:3,:3].T @ a[:3,:3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(a[:3,:3]), 1., atol=1e-8)):
        raise ValueError('Expected a finite rigid SE3 transform')
    return a


def _numbers(raw, length):
    return _vector([float(x) for x in raw.split()], length)


def _mjcf_fixed(root, ancestor, descendant):
    bodies = {node.get('name'): node for node in root.iter('body')}
    if ancestor not in bodies or descendant not in bodies:
        raise ValueError('Missing MJCF calibration body')
    parents = {child: node for node in root.iter() for child in node}
    def ancestors(node):
        chain = []
        while node is not None:
            chain.append(node)
            node = parents.get(node)
        return chain
    source_chain, target_chain = ancestors(bodies[ancestor]), ancestors(bodies[descendant])
    common = next((node for node in source_chain if node in target_chain), None)
    if common is None or common.tag != 'body':
        raise ValueError('Calibration bodies lack a common fixed robot body')
    def relative(chain):
        result = np.eye(4)
        for node in reversed(chain[:chain.index(common)]):
            if node.tag != 'body' or node.find('joint') is not None or node.find('freejoint') is not None:
                raise ValueError('Calibration requires a fixed MJCF body chain')
            if any(key in node.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis', 'childclass')):
                raise ValueError('Unsupported MJCF calibration orientation/defaults')
            part = _translation(_numbers(node.get('pos', '0 0 0'), 3))
            part[:3,:3] = _rotation(_numbers(node.get('quat', '1 0 0 0'), 4))
            result = result @ part
        return result
    return np.linalg.inv(relative(source_chain)) @ relative(target_chain)


def _urdf_fixed(root, ancestor, descendant):
    links = {node.get('name') for node in root.findall('link')}
    if ancestor not in links or descendant not in links:
        raise ValueError('Missing URDF calibration link')
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    chain, current = [], descendant
    while current != ancestor:
        if current not in joints or len(chain) > len(joints):
            raise ValueError('Missing or cyclic URDF calibration chain')
        joint = joints[current]
        if joint.get('type') != 'fixed':
            raise ValueError('Calibration requires fixed URDF joints')
        origin = joint.find('origin')
        attrs = origin.attrib if origin is not None else {}
        part = _translation(_numbers(attrs.get('xyz', '0 0 0'), 3))
        roll, pitch, yaw = _numbers(attrs.get('rpy', '0 0 0'), 3)
        cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        part[:3,:3] = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                [-sp, cp*sr, cp*cr]])
        chain.append(part)
        current = joint.find('parent').get('link')
    result = np.eye(4)
    for part in reversed(chain):
        result = result @ part
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def calibration_from_xml(robot_xml, urdf_xml, *, public_base, actual_base,
                         actual_parent, actual_hand, actual_tcp_offset,
                         api_tcp_to_hand_offset, ik_base='panda_link0',
                         ik_parent='panda_link7', ik_hand='panda_hand', sources=None):
    """Extract fixed calibration from the configured robot model and local URDF.

    The two models describe the same Panda link0/link7 kinematic frames. Moving
    arm joints are deliberately not evaluated; only fixed mount/hand chains enter
    the calibration. Source content hashes make that assumption auditable.
    """
    robot = ET.fromstring(robot_xml)
    urdf = ET.fromstring(urdf_xml)
    if ik_base not in {node.get('name') for node in urdf.findall('link')}:
        raise ValueError('Missing IK root link')
    mount = _mjcf_fixed(robot, actual_base, public_base)
    actual_hand_pose = _mjcf_fixed(robot, actual_parent, actual_hand)
    ik_hand_pose = _urdf_fixed(urdf, ik_parent, ik_hand)
    result = dict(schema='capx.public_robot_calibration.v1',
                  frames=dict(public_base=public_base, actual_base=actual_base,
                              actual_parent=actual_parent, actual_hand=actual_hand,
                              ik_base=ik_base, ik_parent=ik_parent, ik_hand=ik_hand),
                  T_ik_base_public_base=_se3(mount).tolist(),
                  T_actual_hand_ik_hand=_se3(np.linalg.inv(actual_hand_pose) @ ik_hand_pose).tolist(),
                  actual_tcp_offset_in_hand_m=_vector(actual_tcp_offset, 3).tolist(),
                  api_tcp_to_hand_offset_m=_vector(api_tcp_to_hand_offset, 3).tolist(),
                  model_xml_sha256=hashlib.sha256(robot_xml.encode()).hexdigest(),
                  ik_urdf_sha256=hashlib.sha256(urdf_xml.encode()).hexdigest(),
                  sources=list(sources or []), units='m', quaternion_order='wxyz',
                  scope='public_static_robot_calibration_no_scene_state',
                  assumption='Corresponding Panda link0/link7 kinematic frames; fixed mount and hand calibration only')
    result['calibration_hash'] = _digest(result)
    return result


def transform_target(calibration, position, quaternion_wxyz):
    """Return original-goto command (XYZ list, WXYZ list), without calling motion."""
    raw = dict(calibration)
    expected = raw.pop('calibration_hash', None)
    if expected != _digest(raw):
        raise ValueError('Robot calibration hash mismatch')
    if raw.get('schema') != 'capx.public_robot_calibration.v1':
        raise ValueError('Unsupported robot calibration schema')
    target = _translation(position)
    target[:3,:3] = _rotation(quaternion_wxyz)
    mapped = (_se3(raw['T_ik_base_public_base']) @ target
              @ _translation(-_vector(raw['actual_tcp_offset_in_hand_m'], 3))
              @ _se3(raw['T_actual_hand_ik_hand'])
              @ _translation(-_vector(raw['api_tcp_to_hand_offset_m'], 3)))
    return mapped[:3,3].tolist(), _quaternion(mapped[:3,:3]).tolist()
