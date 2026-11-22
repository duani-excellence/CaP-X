"""Deployment-observation-only Stack feedback. No simulator/evaluator imports.

RGB colour surfaces are a second estimator, not an independent sensor. They are
specific to the existing red-on-green Stack task. Occlusion stays UNCERTAIN.
All positions are robot-base metres; TCP quaternion is WXYZ.
"""
from __future__ import annotations
import time
import numpy as np
from .sandbox import ExecutionInterrupted

LEVELS = {'A': -1, 'B': 0, 'C': 1, 'D': 2, 'E': 3}


def public_observation(raw):
    out = {k: np.asarray(raw[k]).copy() for k in ('robot_joint_pos','robot_cartesian_pos','robot_gripper_joint_pos') if k in raw}
    for name, camera in raw.items():
        if not isinstance(camera, dict) or 'images' not in camera:
            continue
        if not all(k in camera for k in ('intrinsics','pose_mat')):
            continue
        out[name] = {k:np.asarray(camera[k]).copy() for k in ('intrinsics','pose_mat')}
        out[name]['images'] = {k:np.asarray(camera['images'][k]).copy() for k in ('rgb','depth') if k in camera['images']}
    return out


def surface_evidence(obs):
    """Fresh calibrated RGB-D, independent of SAM3/OBB pose estimates.

    Visible surface bounds are NOT full object extents. Top-plane centre avoids
    the known side-surface OBB bias; visible lower surface gives only a gap cue.
    """
    cameras=[v for v in obs.values() if isinstance(v,dict) and 'images' in v]
    if not cameras:return {}
    cam=cameras[0];rgb=cam['images']['rgb'].astype(float);depth=cam['images']['depth'].squeeze()
    k=cam['intrinsics'];t=cam['pose_mat']
    yy,xx=np.indices(depth.shape)
    points=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],axis=-1)
    points=points @ t[:3,:3].T+t[:3,3]
    valid=np.isfinite(points).all(axis=-1)&(depth>0)
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    masks={'red':(r>50)&(r>1.5*g)&(r>1.5*b),'green':(g>45)&(g>1.4*r)&(g>1.3*b)}
    out={}
    for name,mask in masks.items():
        p=points[mask&valid]
        if len(p)<30:out[name]=None;continue
        lo,hi=np.quantile(p,[.01,.99],axis=0)
        top=p[p[:,2]>=hi[2]-.003]
        if len(top)<15:out[name]=None;continue
        xylo,xyhi=np.quantile(top[:,:2],[.02,.98],axis=0)
        if np.min(xyhi-xylo)<.005:out[name]=None;continue
        center=np.r_[(xylo+xyhi)/2,(lo[2]+hi[2])/2]
        out[name]={'center':center.tolist(),'lo':lo.tolist(),'hi':hi.tolist(),
                   'top_xy_bounds':[xylo.tolist(),xyhi.tolist()], 'pixels':len(p),'top_pixels':len(top)}
    return out


def release_check(evidence):
    red,green=evidence.get('red'),evidence.get('green')
    if not red or not green:
        return dict(status='UNCERTAIN',reason='insufficient_visible_surfaces',metrics={},action='reperceive_or_replan')
    delta=np.asarray(green['center'])[:2]-np.asarray(red['center'])[:2]
    gap=float(red['lo'][2]-green['hi'][2])
    xy=float(np.linalg.norm(delta))
    good=xy<=.008 and -.004<=gap<=.016
    return dict(status='PASS' if good else 'FAIL',reason='visible_support_alignment' if good else 'release_geometry_risk',
                metrics={'xy_error_m':xy,'visible_gap_m':gap,'delta_xy_m':delta.tolist(),
                         'evidence':'fresh_colour_RGBD_surface_not_SAM3_OBB',
                         'limitation':'shared_camera_and_visible_surfaces_only_not_contact_confirmation'},
                action='release' if good else 'local_align_or_replan')


class FeedbackMonitor:
    def __init__(self,group,observe,*,emit=None,evidence_fn=surface_evidence,max_local=2):
        self.level=LEVELS[group];self.group=group;self.observe=observe;self.emit=emit or (lambda row:None)
        self.evidence_fn=evidence_fn;self.max_local=max_local;self.events=[];self.recoveries=0
        self.closed=False;self.lifted=False;self.close_state=None;self.close_evidence=None
        self.last_public={};self.last_evidence={};self.started=time.monotonic()

    def record(self,layer,check,status,reason,metrics=None,action='continue',**extra):
        row=dict(layer=layer,check=check,status=status,reason=reason,metrics=metrics or {},action=action,
                 elapsed_seconds=time.monotonic()-self.started,**extra)
        self.events.append(row);self.emit(row);return row

    def capture(self):
        state=public_observation(self.observe())
        self.last_public={k:np.asarray(v).tolist() for k,v in state.items() if not isinstance(v,dict)}
        self.last_evidence=self.evidence_fn(state)
        return state,self.last_evidence

    def interrupt(self,reason):
        self.record('L3','handoff','FAIL',reason,action='replan' if self.level>=3 else 'stop',
                    replanning_enabled=self.level>=3)
        raise ExecutionInterrupted(reason)

    def recovery(self,kind,fn,name,args,kwargs):
        self.recoveries+=1
        self.record('L0' if kind=='reperceive' else 'L2' if kind=='align' else 'L1',
                    'recovery','UNCERTAIN',kind,{'api':name,'args':args,'kwargs':kwargs},action=kind)
        return fn(name,args,kwargs)

    def pose_check(self,name,args,kwargs,response,evidence):
        try:
            p,q=np.asarray(response[0],float),np.asarray(response[1],float)
            valid=p.shape==(3,) and q.shape==(4,) and np.isfinite(p).all() and np.isfinite(q).all() and abs(np.linalg.norm(q)-1)<.02
            if len(response)>2 and response[2] is not None:
                ext=np.asarray(response[2],float);valid=valid and ext.shape==(3,) and np.isfinite(ext).all() and bool(np.all(ext>0))
            if not valid:return 'FAIL','invalid_geometry',{}
        except (ValueError,TypeError,IndexError):return 'FAIL','invalid_geometry',{}
        label=str(args[0] if args else kwargs.get('object_name',''))
        surface=evidence.get('red' if 'red' in label else 'green' if 'green' in label else '')
        if name=='sample_grasp_pose':return 'UNCERTAIN','candidate_finite_contact_unverified',{'quaternion_norm':float(np.linalg.norm(q))}
        if not surface:return 'UNCERTAIN','object_occluded_or_surface_unavailable',{}
        xy=float(np.linalg.norm(p[:2]-np.asarray(surface['center'])[:2]))
        return ('PASS','geometry_consistent',{'surface_xy_disagreement_m':xy}) if xy<=.025 else ('UNCERTAIN','estimators_disagree',{'surface_xy_disagreement_m':xy})

    def control_check(self,args,kwargs,state):
        target=np.asarray(args[0] if args else kwargs['position'],float)
        quat=np.asarray(args[1] if len(args)>1 else kwargs['quaternion_wxyz'],float)
        actual=np.asarray(state['robot_cartesian_pos'],float)
        err=float(np.linalg.norm(target-actual[:3]))
        dot=abs(float(np.dot(quat/np.linalg.norm(quat),actual[3:7]/np.linalg.norm(actual[3:7]))))
        angle=float(2*np.arccos(np.clip(dot,0,1)))
        status='PASS' if np.isfinite([err,angle]).all() and err<=.025 and angle<=np.deg2rad(20) else 'FAIL'
        return status,dict(target_tcp=np.r_[target,quat].tolist(),actual_tcp=actual.tolist(),position_error_m=err,orientation_error_rad=angle,joints=state['robot_joint_pos'].tolist(),contact_sensor='unavailable')

    def before_release(self,fn):
        for attempt in range(self.max_local+1):
            state,evidence=self.capture();check=release_check(evidence)
            self.record('L2','pre_release',**check,evidence=evidence,attempt=attempt)
            if check['status']=='PASS':return
            if check['status']=='UNCERTAIN' or attempt==self.max_local:
                self.interrupt('pre_release_'+check['reason'])
            # Bound correction from fresh visible surfaces, never simulator object poses.
            tcp=np.asarray(state['robot_cartesian_pos'],float)
            delta=np.r_[check['metrics']['delta_xy_m'],.006-check['metrics']['visible_gap_m']]
            norm=np.linalg.norm(delta[:2])
            if norm>.035:delta[:2]*=.035/norm
            delta[2]=np.clip(delta[2],-.04,.02)
            self.recovery('align',fn,'goto_pose',[(tcp[:3]+delta).tolist(),tcp[3:7].tolist()],{})
            after,_=self.capture();status,metrics=self.control_check([(tcp[:3]+delta),tcp[3:7]],{},after)
            self.record('L1','local_alignment_control',status,'tcp_tracking',metrics)
            if status=='FAIL':self.interrupt('local_alignment_control_failed')

    def invoke(self,name,args,kwargs,fn):
        if self.level<0:return fn(name,args,kwargs)
        if self.level>=2 and name=='open_gripper' and self.closed:self.before_release(fn)
        response=fn(name,args,kwargs)
        if name in ('get_object_pose','sample_grasp_pose'):
            _,evidence=self.capture();status,reason,metrics=self.pose_check(name,args,kwargs,response,evidence)
            self.record('L0',name,status,reason,metrics)
            if status=='FAIL' or reason=='estimators_disagree':
                response=self.recovery('reperceive',fn,name,args,kwargs)
                _,evidence=self.capture();status,reason,metrics=self.pose_check(name,args,kwargs,response,evidence)
                self.record('L0',name,status,reason,metrics,repeated_same_estimator=True)
                if status=='FAIL':self.interrupt('invalid_perception_after_retry')
        if self.level>=1 and name=='goto_pose':
            state,evidence=self.capture();status,metrics=self.control_check(args,kwargs,state)
            self.record('L1','tcp_tracking',status,'tcp_tracking',metrics)
            if status=='FAIL':
                response=self.recovery('control_retry',fn,name,args,kwargs)
                state,evidence=self.capture();status,metrics=self.control_check(args,kwargs,state)
                self.record('L1','tcp_tracking',status,'tcp_tracking_after_retry',metrics)
                if status=='FAIL':self.interrupt('control_not_converged')
            if self.level>=2 and self.closed and float(state['robot_cartesian_pos'][7])<.05:
                self.closed=False;self.lifted=False
                self.record('L2','slip','FAIL','empty_jaws_after_close',
                            {'gripper_fraction':float(state['robot_cartesian_pos'][7]),'contact_sensor':'unavailable'},action='regrasp')
                self.interrupt('lost_grasp_empty_jaws')
            if self.level>=2 and self.closed and self.close_state is not None:
                tcp_delta=float(state['robot_cartesian_pos'][2]-self.close_state['robot_cartesian_pos'][2])
                before=self.close_evidence.get('red');now=evidence.get('red')
                if tcp_delta>.05:
                    rise=now['hi'][2]-before['hi'][2] if before and now else None
                    status='UNCERTAIN' if rise is None else 'PASS' if rise>.035 else 'FAIL'
                    self.record('L2','lift',status,'visual_rise_after_close',{'tcp_rise_m':tcp_delta,'red_surface_rise_m':rise})
                    if status=='PASS':self.lifted=True
                    if status=='FAIL':self.interrupt('object_did_not_follow_lift')
        if self.level>=2 and name=='close_gripper':
            self.closed=True;self.lifted=False
            self.close_state,self.close_evidence=self.capture()
            opening=float(self.close_state['robot_cartesian_pos'][7])
            status='FAIL' if opening<.05 else 'UNCERTAIN'
            self.record('L2','grasp',status,'empty_close' if status=='FAIL' else 'jaw_obstruction_requires_lift_verification',{'gripper_fraction':opening,'contact_sensor':'unavailable'})
            if status=='FAIL':
                self.closed=False
                self.interrupt('empty_grasp')
        if self.level>=2 and name=='open_gripper':
            self.closed=False;self.lifted=False
        return response

    def finish_check(self):
        if self.level<2:return None
        _,evidence=self.capture();check=release_check(evidence)
        # Visual support is not proof of physical stable placement.
        return self.record('L2','place',check['status'],check['reason'],check['metrics'],
                           action='finish_candidate' if check['status']=='PASS' and not self.closed else 'replan',
                           gripper_commanded_closed=self.closed)

    def summary(self):
        keys={'xy_error_m','visible_gap_m','delta_xy_m','position_error_m','orientation_error_rad',
              'surface_xy_disagreement_m','gripper_fraction','tcp_rise_m','red_surface_rise_m','api','args','kwargs'}
        checks=[dict(layer=e['layer'],check=e['check'],status=e['status'],reason=e['reason'],action=e['action'],
                     metrics={k:v for k,v in e['metrics'].items() if k in keys}) for e in self.events[-10:]]
        return dict(schema='capx.feedback.v1',status=self.events[-1]['status'] if self.events else 'UNCERTAIN',
                    public_robot=self.last_public,surfaces=self.last_evidence,checks=checks,
                    recoveries=self.recoveries,gripper_commanded_closed=self.closed,
                    warning='No tactile sensor. Shared-camera surface estimates may be biased or occluded. FINISH is not physical success.')
