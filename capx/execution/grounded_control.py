"""Bounded TCP correction using deployment-available robot proprioception only.

Blocking goto_pose calls cannot be cancelled in flight. Intervention and checks
happen at API boundaries. Vertical-then-horizontal staging avoids a known diagonal
insertion path; it is NOT collision-free trajectory planning. The caller must
confirm a currently held object has already passed its lift verification, and
must route every execute call through the existing API budget/trace broker.
"""
from __future__ import annotations
import math
import numpy as np
from .sandbox import ExecutionInterrupted


def _quaternion(value):
    q=np.asarray(value,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
        raise ValueError('invalid_wxyz_quaternion')
    return q/np.linalg.norm(q)


def _mul(a,b):
    w,x,y,z=a;v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def _inverse(q):
    return q*np.array([1.,-1.,-1.,-1.])


def _angle(a,b):
    return math.degrees(2*math.acos(float(np.clip(abs(np.dot(a,b)),0.,1.))))


def precise_goto(target_position,target_quaternion,*,observe,execute,reserve_correction,
                 emit=None,lift_verified=False,approach_height=.06,
                 position_tolerance=.003,orientation_tolerance_deg=5.,
                 max_corrections=3,max_command_offset=.03,
                 max_orientation_offset_deg=20.):
    """Reach a base-frame TCP pose with measured, bounded residual corrections.

    ``observe()`` returns only public robot observations (TCP XYZ/WXYZ at
    ``robot_cartesian_pos``). ``execute(name,args,kwargs)`` is the existing budgeted
    broker. It is called for every staging/correction motion with z_approach=0.
    ``reserve_correction()`` consumes one episode-local recovery allowance before
    each additional correction; returning False or raising aborts before motion.
    Normal ascend/horizontal/descend staging does not consume recovery allowance.

    The three correction allowance is shared across all stages. Compensation is
    estimated from actual target error, retained across stages, and limited in
    cumulative position/orientation. It contains no task- or seed-specific bias.
    Setting approach_height=0 disables staging; use only where the caller's skill
    contract permits a direct motion. A successful result is TCP convergence,
    never proof of grasp stability, placement safety, or physical task success.
    """
    emit=emit or (lambda event:None)
    history=[];calls=0;corrections=0
    offset=np.zeros(3);rotation_offset=np.array([1.,0.,0.,0.])

    def record(check,status,reason,**fields):
        row={'schema':'capx.public_tcp_control.v1','layer':'L1','check':check,
             'status':status,'reason':reason,**fields}
        history.append(row);emit(row)
        return row

    def abort(reason,**fields):
        record('precise_tcp','FAIL',reason,**fields)
        raise ExecutionInterrupted('precise_tcp_'+reason)

    if lift_verified is not True:
        abort('lift_verified_required')
    try:
        target=np.asarray(target_position,dtype=float)
        qtarget=_quaternion(target_quaternion)
        limits=[approach_height,position_tolerance,orientation_tolerance_deg,
                max_command_offset,max_orientation_offset_deg]
        if target.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(limits).all():
            raise ValueError('invalid_target_or_limits')
        if approach_height<0 or min(limits[1:])<=0:
            raise ValueError('invalid_limits')
        if isinstance(max_corrections,bool) or not isinstance(max_corrections,int) or max_corrections<0:
            raise ValueError('invalid_correction_count')
    except (ValueError,TypeError):
        abort('invalid_target_or_limits')

    def read_robot():
        try:
            public=observe()
            tcp=np.asarray(public['robot_cartesian_pos'],dtype=float)
            if tcp.ndim!=1 or len(tcp)<7 or not np.isfinite(tcp).all():
                raise ValueError('invalid_public_tcp')
            return tcp[:3].copy(),_quaternion(tcp[3:7])
        except (KeyError,ValueError,TypeError,IndexError):
            abort('invalid_public_tcp')

    def errors(p,q,actual_p,actual_q):
        return {'position_error_m':float(np.linalg.norm(p-actual_p)),
                'xy_error_m':float(np.linalg.norm(p[:2]-actual_p[:2])),
                'orientation_error_deg':_angle(q,actual_q),
                'target_tcp':np.r_[p,q].tolist(),'actual_tcp':np.r_[actual_p,actual_q].tolist()}

    def converged(metrics):
        return metrics['position_error_m']<=position_tolerance and metrics['orientation_error_deg']<=orientation_tolerance_deg

    def reach(stage,p,q):
        nonlocal calls,corrections,offset,rotation_offset
        actual_p,actual_q=read_robot()
        metrics=errors(p,q,actual_p,actual_q)
        if converged(metrics):
            record('precise_tcp','PASS','already_at_target',stage=stage,correction=False,**metrics)
            return metrics
        correction=False
        while True:
            command=p+offset;command_q=_quaternion(_mul(rotation_offset,q))
            record('precise_tcp_command','UNCERTAIN','bounded_residual_correction' if correction else 'staging_motion',
                   stage=stage,correction=correction,command_tcp=np.r_[command,command_q].tolist(),
                   target_tcp=np.r_[p,q].tolist(),corrections=corrections)
            calls+=1
            try:
                execute('goto_pose',[command.tolist(),command_q.tolist()],{'z_approach':0.})
            except ExecutionInterrupted:
                record('precise_tcp','FAIL','broker_interrupted',stage=stage,correction=correction)
                raise
            except Exception as exc:
                abort('motion_callback_error',stage=stage,error_type=type(exc).__name__)
            actual_p,actual_q=read_robot()
            metrics=errors(p,q,actual_p,actual_q)
            good=converged(metrics)
            record('precise_tcp','PASS' if good else 'FAIL','tcp_converged' if good else 'tcp_residual',
                   stage=stage,correction=correction,**metrics)
            if good:
                return metrics
            if corrections>=max_corrections:
                abort('correction_limit',stage=stage,corrections=corrections,**metrics)
            next_offset=offset+p-actual_p
            next_rotation=_quaternion(_mul(_mul(q,_inverse(actual_q)),rotation_offset))
            if np.linalg.norm(next_offset)>max_command_offset+1e-12 or _angle(next_rotation,np.array([1.,0.,0.,0.]))>max_orientation_offset_deg+1e-9:
                abort('offset_limit',stage=stage,requested_offset_m=next_offset.tolist(),**metrics)
            if reserve_correction() is False:
                abort('correction_budget',stage=stage,**metrics)
            corrections+=1;correction=True
            offset=next_offset;rotation_offset=next_rotation

    initial_p,initial_q=read_robot()
    if approach_height>0:
        safe_z=max(float(initial_p[2]),float(target[2]+approach_height))
        if initial_p[2]<safe_z-position_tolerance:
            # Keep XY and orientation unchanged until clear of the low start.
            reach('ascend',np.array([initial_p[0],initial_p[1],safe_z]),initial_q)
        reach('above_target',np.array([target[0],target[1],safe_z]),qtarget)
        # Fresh public pose must still be aligned immediately before descent.
        actual_p,actual_q=read_robot()
        aligned=errors(np.array([target[0],target[1],safe_z]),qtarget,actual_p,actual_q)
        if not converged(aligned):
            abort('pre_descent_alignment_changed',stage='pre_descent',**aligned)
        record('precise_tcp_pre_descent','PASS','fresh_xy_and_orientation_verified',stage='pre_descent',**aligned)
    final=reach('final',target,qtarget)
    return {'status':'PASS','corrections':corrections,'api_calls':calls,
            'position_offset_m':offset.tolist(),'orientation_offset_wxyz':rotation_offset.tolist(),
            'history':history,**final}
