"""Validate soft IK solutions against configured public robot kinematics.

This wrapper never moves the robot or changes the caller's joint configuration.
It iterates the EXISTING instrumented IK callback in memory, with an episode-wide
solver budget. It adds no network retries. The legacy HTTP client's own retry
policy still applies and must be accounted for by its request audit.

A rejection proves only that this solver segment returned no executable solution.
A multi-segment goto may already have moved during an earlier successful segment;
callers must not interpret ``checked_no_segment_execution`` as an entire-API
rollback or a guarantee that no preceding physical action happened.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np

from .sandbox import ExecutionInterrupted


class IKConvergenceError(ExecutionInterrupted):
    """Trusted pre-return rejection of the current IK segment, not whole goto."""

    def __init__(self, reason, *, call_generation=0, successful_calls_before=0,
                 iterations=0):
        super().__init__(reason)
        self.reason = reason
        self.checked_no_segment_execution = True
        self.call_generation = call_generation
        self.successful_calls_before = successful_calls_before
        self.iterations = iterations


class IKSolveBudget:
    """Share one finite solver-call allowance across all episode API instances."""

    def __init__(self, max_calls):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError('max_calls must be a positive integer')
        self.max_calls = max_calls
        self.calls = 0

    def reserve(self):
        if self.calls >= self.max_calls:
            raise IKConvergenceError('ik_solve_budget_exhausted')
        self.calls += 1


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('IK target must be finite WXYZXYZ')
    norm = float(np.linalg.norm(pose[:4]))
    if norm < 1e-9 or abs(norm - 1.) > .001:
        raise ValueError('IK target quaternion must be unit WXYZ')
    pose = pose.copy()
    pose[:4] /= norm
    return pose


class URDFFK:
    """Static local URDF FK and limits; no meshes, simulator, network or download."""

    def __init__(self, urdf_path, target_link='panda_hand'):
        from yourdfpy import URDF
        path = Path(urdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.urdf_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.urdf_path = str(path.resolve())
        self.model = URDF.load(str(path), load_meshes=False)
        if target_link not in self.model.link_map:
            raise ValueError('IK verification target link missing in configured URDF')
        self.target_link = target_link
        self.joint_names = list(self.model.actuated_joint_names)
        self.limits = []
        for joint in self.model.actuated_joints:
            if joint.type == 'continuous':
                self.limits.append((-math.inf, math.inf))
            elif joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
                self.limits.append((float(joint.limit.lower), float(joint.limit.upper)))
            else:
                raise ValueError('Actuated joint has no configured position limits: ' + joint.name)
        self.limits = np.asarray(self.limits, dtype=float)
        if self.limits.shape != (len(self.joint_names), 2) or np.any(self.limits[:, 0] > self.limits[:, 1]):
            raise ValueError('Invalid configured URDF joint limits')

    def validate(self, value):
        q = np.asarray(value, dtype=float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError('Invalid IK joint vector')
        # Float32 service output can differ from a decimal URDF limit by <1e-6.
        if np.any(q < self.limits[:, 0] - 1e-6) or np.any(q > self.limits[:, 1] + 1e-6):
            raise ValueError('IK joint solution outside configured URDF limits')
        return q.copy()

    def neutral_configuration(self):
        """A static bounded reference, not a physical waypoint or home command."""
        lo, hi = self.limits.T
        midpoint = np.zeros(len(self.joint_names), dtype=float)
        finite = np.isfinite(lo) & np.isfinite(hi)
        midpoint[finite] = (lo[finite] + hi[finite]) / 2
        return self.validate(midpoint)

    def pose(self, joints):
        from scipy.spatial.transform import Rotation
        q = self.validate(joints)
        self.model.update_cfg(dict(zip(self.joint_names, q)))
        matrix = self.model.get_transform(self.target_link)
        xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return np.r_[xyzw[[3, 0, 1, 2]], matrix[:3, 3]]


class BoundedIK:
    """Drop-in ``ik_solve_fn(target_pose_wxyz_xyz, prev_cfg=None)`` verifier.

    Construct after existing tool instrumentation; every in-memory iteration then
    retains the original HTTP and payload evidence. FK target and output use the
    configured IK base and hand link, not the public TCP frame. The separate
    control-frame mapping remains responsible for those frame transformations.
    The public-joint observer is diagnostic only; it never supplies object truth.
    If residuals stall, at most one alternate static URDF midpoint reference may
    be supplied as prev_cfg within the SAME eight-call allowance. The existing
    service exposes a velocity-cost reference, not optimizer initial values; this
    is a bounded reference restart, not a new optimizer or random multistart.
    """

    def __init__(self, solve, *, budget, urdf_path=None, kinematics=None,
                 max_iterations=8, position_tolerance=.001,
                 orientation_tolerance_deg=1., observe_joints=None,
                 calibration_hash=None, emit=None):
        if not callable(solve) or not isinstance(budget, IKSolveBudget):
            raise ValueError('A solver and shared IKSolveBudget are required')
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 8:
            raise ValueError('max_iterations must be between 1 and 8')
        tolerances = np.asarray([position_tolerance, orientation_tolerance_deg], dtype=float)
        if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
            raise ValueError('IK tolerances must be finite and positive')
        if (kinematics is None) == (urdf_path is None):
            raise ValueError('Specify exactly one local URDF or kinematics object')
        self.solve = solve
        self.budget = budget
        self.kinematics = kinematics if kinematics is not None else URDFFK(urdf_path)
        self.max_iterations = max_iterations
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance_deg = float(orientation_tolerance_deg)
        self.observe_joints = observe_joints
        self.calibration_hash = calibration_hash
        self.emit = emit or (lambda event: None)
        self.events = []
        self.call_generation = 0
        self.successful_calls = 0

    def __call__(self, target_pose_wxyz_xyz, prev_cfg=None):
        self.call_generation += 1
        target = _pose(target_pose_wxyz_xyz)
        previous = None if prev_cfg is None else self.kinematics.validate(prev_cfg)
        public = None
        if self.observe_joints is not None:
            public = np.asarray(self.observe_joints(), dtype=float)
            if public.ndim != 1 or not np.isfinite(public).all():
                raise ValueError('Invalid public robot joint observation')
            public = public.tolist()
        started = time.monotonic()
        reference_restarts = 0
        reference_strategy = 'caller_reference' if previous is not None else 'existing_solver_default'
        previous_score = None
        branch_iterations = 0
        base = {'schema': 'capx.bounded_ik.v1', 'layer': 'L1', 'check': 'ik_fk_convergence',
                'call_generation': self.call_generation, 'target_wxyz_xyz': target.tolist(),
                'public_joints': public, 'calibration_hash': self.calibration_hash,
                'urdf_sha256': self.kinematics.urdf_sha256,
                'joint_names': list(self.kinematics.joint_names),
                'position_tolerance_m': self.position_tolerance,
                'orientation_tolerance_deg': self.orientation_tolerance_deg,
                'successful_calls_before': self.successful_calls,
                'current_segment_executed': False}

        def record(status, reason, iteration, **fields):
            row = {**base, 'status': status, 'reason': reason, 'iteration': iteration,
                   'solver_calls_used': self.budget.calls, 'solver_calls_limit': self.budget.max_calls,
                   'elapsed_seconds': time.monotonic() - started,
                   'reference_strategy': reference_strategy,
                   'reference_restarts': reference_restarts, **fields}
            self.events.append(row)
            self.emit(row)

        def reject(reason, iterations):
            raise IKConvergenceError(reason, call_generation=self.call_generation,
                                     successful_calls_before=self.successful_calls,
                                     iterations=iterations)

        for iteration in range(1, self.max_iterations + 1):
            try:
                self.budget.reserve()
            except IKConvergenceError:
                record('FAIL', 'ik_solve_budget_exhausted', iteration - 1)
                reject('ik_solve_budget_exhausted', iteration - 1)
            # No transport retry here: the existing callback owns that policy.
            try:
                answer = self.solve(target_pose_wxyz_xyz=target.copy(),
                                    prev_cfg=None if previous is None else previous.copy())
            except Exception:
                record('FAIL', 'ik_provider_error', iteration)
                raise
            try:
                joints = self.kinematics.validate(answer)
                actual = _pose(self.kinematics.pose(joints))
            except (ValueError, TypeError, IndexError, KeyError):
                record('FAIL', 'ik_invalid_solution', iteration)
                reject('ik_invalid_solution_before_motion', iteration)
            pos = float(np.linalg.norm(actual[4:] - target[4:]))
            dot = float(np.clip(abs(np.dot(actual[:4], target[:4])), 0., 1.))
            angle = math.degrees(2 * math.acos(dot))
            passed = pos <= self.position_tolerance and angle <= self.orientation_tolerance_deg
            record('PASS' if passed else 'FAIL', 'ik_converged' if passed else 'ik_residual', iteration,
                   fk_wxyz_xyz=actual.tolist(), joint_solution=joints.tolist(),
                   position_error_m=pos, orientation_error_deg=angle)
            if passed:
                self.successful_calls += 1
                return joints.copy()
            branch_iterations += 1
            score = max(pos / self.position_tolerance, angle / self.orientation_tolerance_deg)
            stalled = (previous_score is not None and score >= previous_score * .99)
            # Keep normal residual convergence intact. A stuck branch gets one
            # alternate static reference, still counted against all same budgets.
            restart = (reference_restarts == 0 and iteration < self.max_iterations
                       and (stalled or branch_iterations >= 4)
                       and callable(getattr(self.kinematics, 'neutral_configuration', None)))
            if restart:
                previous = self.kinematics.validate(self.kinematics.neutral_configuration())
                reference_restarts += 1
                reference_strategy = 'urdf_joint_midpoint_restart'
                previous_score = None
                branch_iterations = 0
            else:
                previous = joints.copy()
                previous_score = score
                reference_strategy = 'continued_valid_joint_solution'
        reject('ik_unconverged_before_motion', self.max_iterations)
