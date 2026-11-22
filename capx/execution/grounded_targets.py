"""Trusted numerical skill recipes for a parent-owned API broker.

The caller supplies *validated parent nodes*, never actor-supplied node data.
This module binds targets to specified source fields; the monitor still owns ID,
version, observation freshness, held-object identity and skill permission checks.
No arbitrary XYZ/quaternion target or arbitrary expression is accepted as params.
Surface geometry is public perception, not proof of contact or task success.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real


_ROLES = {
    'approach': {'grasp'},
    'lift': {'robot'},
    'retreat': {'robot'},
    'place': {'robot', 'held_surface', 'support_surface'},
    'table_place': {'robot', 'held_surface', 'table_region'},
}
_PARAMS = {
    'approach': {'z_approach'},
    'lift': {'dz'},
    'retreat': {'dz'},
    'place': {'gap', 'z_approach'},
    'table_place': {'gap', 'z_approach'},
}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)


def _vector(value, size, label):
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(label + ' must be a vector')
    try:
        parts = list(value)
    except TypeError as exc:
        raise ValueError(label + ' must be a vector') from exc
    if len(parts) != size:
        raise ValueError(label + ' has the wrong shape')
    return [_number(v, label) for v in parts]


def _bounded(params, key, default, low, high):
    value = _number(params.get(key, default), key)
    if not low <= value <= high:
        raise ValueError(key + ' is outside the skill bounds')
    return value


def _source(inputs, role):
    source = inputs[role]
    if not isinstance(source, Mapping) or not isinstance(source.get('id'), str) or not source['id']:
        raise ValueError(role + ' requires a parent node ID')
    if source.get('valid') is False or source.get('state') == 'invalid':
        raise ValueError(role + ' is invalid')
    data = source.get('data')
    if not isinstance(data, Mapping) or data.get('frame') != 'robot_base':
        raise ValueError(role + ' must use the robot_base frame')
    return source, data


def _pose(data, role):
    if data.get('quaternion_order') != 'wxyz':
        raise ValueError(role + ' must declare quaternion_order=wxyz')
    position = _vector(data.get('position'), 3, role + '.position')
    quaternion = _vector(data.get('quaternion_wxyz'), 4, role + '.quaternion_wxyz')
    norm = math.sqrt(sum(v * v for v in quaternion))
    if abs(norm - 1.) > .02:
        raise ValueError(role + ' quaternion must be near unit length')
    return position, [v / norm for v in quaternion]


def _surface(data, role):
    center = _vector(data.get('center'), 3, role + '.center')
    lo = _vector(data['lo'], 3, role + '.lo') if 'lo' in data else None
    hi = _vector(data['hi'], 3, role + '.hi') if 'hi' in data else None
    if (lo is None) != (hi is None):
        raise ValueError(role + ' requires both lo and hi when bounds are provided')
    if lo is not None:
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError(role + ' surface bounds are inverted')
        if any(c < a - 1e-9 or c > b + 1e-9 for a, c, b in zip(lo, center, hi)):
            raise ValueError(role + ' center lies outside its surface bounds')
    return center, lo, hi


def _reference(role, source, fields):
    data = source['data']
    # Only the recipe's actual numerical inputs enter this fingerprint.
    values = {field: data[field] for field in fields}
    encoded = json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False,
                         default=lambda value: value.tolist()).encode()
    return dict(role=role, id=source['id'], fields=list(fields),
                value_digest=hashlib.sha256(encoded).hexdigest())


def _base_vector_to_tcp(vector, quaternion):
    """R(WXYZ).T times a vector; this is an offset, never a world position."""
    w, x, y, z = quaternion
    rotation = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )
    return [sum(rotation[row][column] * vector[row] for row in range(3))
            for column in range(3)]


def compile_target(skill, inputs, params=None):
    """Compile one trusted recipe from parent node data into a TCP target.

    ``inputs`` maps each required role to ``{'id': ..., 'data': ...}`` (full
    graph nodes are also accepted). Every data mapping declares ``frame``.
    Pose inputs use ``position``, ``quaternion_wxyz`` and ``quaternion_order``.
    Surface inputs use ``center, lo, hi``. A measured held ``bottom_z`` is used
    preferentially; falling back to visible ``lo[2]`` is explicitly uncertain.
    A table region requires the parent's exact boolean ``clearance_verified``
    and either ``top_z`` or bounds. It is never selected from actor XYZ params.
    """
    if not isinstance(skill, str) or skill not in _ROLES:
        raise ValueError('Unknown grounded skill recipe')
    if not isinstance(inputs, Mapping) or set(inputs) != _ROLES[skill]:
        raise ValueError('Recipe source roles do not match the skill')
    params = {} if params is None else params
    if not isinstance(params, Mapping) or not set(params) <= _PARAMS[skill]:
        raise ValueError('Unsupported recipe parameter; arbitrary targets are forbidden')
    sources = {role: _source(inputs, role) for role in sorted(inputs)}
    recipe = dict(name='grounded_' + skill, version=1, skill=skill,
                  frame='robot_base', quaternion_order='wxyz', parameters={})
    uncertainty = []
    used_sources = []

    if skill == 'approach':
        source, data = sources['grasp']
        position, quaternion = _pose(data, 'grasp')
        z_approach = _bounded(params, 'z_approach', .1, 0., .15)
        recipe['parameters']['z_approach'] = z_approach
        used_sources.append(_reference('grasp', source, ['position', 'quaternion_wxyz']))
    else:
        source, data = sources['robot']
        tcp, quaternion = _pose(data, 'robot')
        used_sources.append(_reference('robot', source, ['position', 'quaternion_wxyz']))
        if skill in ('lift', 'retreat'):
            dz = _bounded(params, 'dz', .1, .05 if skill == 'lift' else .08, .25)
            position = [tcp[0], tcp[1], tcp[2] + dz]
            z_approach = 0.
            recipe['parameters']['dz'] = dz
        else:
            held_source, held_data = sources['held_surface']
            held, lo, hi = _surface(held_data, 'held_surface')
            if 'bottom_z' in held_data:
                bottom = _number(held_data['bottom_z'], 'held_surface.bottom_z')
                if hi is not None and bottom > hi[2] + 1e-9:
                    raise ValueError('Measured held bottom is above the visible top')
                held_fields = ['center', 'bottom_z']
                recipe['bottom_evidence'] = 'measured_bottom'
            else:
                if lo is None:
                    raise ValueError('Held surface needs measured bottom_z or visible bounds')
                bottom = lo[2]
                held_fields = ['center', 'lo']
                recipe['bottom_evidence'] = 'visible_lower_bound'
                uncertainty.append('held_bottom_uses_visible_surface_lower_bound')
            support_role = 'table_region' if skill == 'table_place' else 'support_surface'
            support_source, support_data = sources[support_role]
            support, _, support_hi = _surface(support_data, support_role)
            support_fields = ['center']
            if 'top_z' in support_data:
                top = _number(support_data['top_z'], support_role + '.top_z')
                support_fields.append('top_z')
            else:
                if support_hi is None:
                    raise ValueError('Support surface needs top_z or visible bounds')
                top = support_hi[2]
                support_fields.append('hi')
            if skill == 'table_place':
                if support_data.get('clearance_verified') is not True:
                    raise ValueError('Table region clearance is not verified by the parent')
                support_fields.append('clearance_verified')
            gap = _bounded(params, 'gap', .006, -.004, .016)
            z_approach = _bounded(params, 'z_approach', 0., 0., .15)
            offset = [held[0]-tcp[0], held[1]-tcp[1], bottom-tcp[2]]
            desired = [support[0], support[1], top+gap]
            position = [a-b for a, b in zip(desired, offset)]
            recipe.update(support_role=support_role, held_offset_base=offset,
                          held_offset_tcp=_base_vector_to_tcp(offset, quaternion))
            recipe['parameters'].update(gap=gap, z_approach=z_approach)
            used_sources.extend((_reference('held_surface', held_source, held_fields),
                                 _reference(support_role, support_source, support_fields)))
    return dict(position=position, quaternion_wxyz=quaternion, z_approach=z_approach,
                recipe=recipe, used_sources=used_sources, uncertainty=uncertainty)
