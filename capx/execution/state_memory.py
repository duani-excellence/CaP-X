"""Parent-owned, indexed public evidence and conditional SE(3) propagation.

No simulator/evaluator inputs. Bounds are an offline model artifact, not an
online correction. Missing calibration or violated model assumptions => UNKNOWN.
An empirical envelope is conditional, never a distribution-free safety proof.
"""
import copy
import math
import numpy as np
from .grip_offset import rotation
from .public_geometry import place_metrics


class StateMemory:
    def __init__(self,graph,profile):
        p=copy.deepcopy(profile)
        fields=('xy_m','z_m','height_m','translation_rate','rotation_rate','age_rate',
                'static_rate','max_age_seconds','max_travel_m','max_rotation_rad')
        if (p.get('schema')!='capx.memory_bounds.v1' or p.get('validated') is not True
            or p.get('frame')!='robot_base' or not p.get('evidence') or not p.get('calibration_id')
            or any(type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) or p[k]<0 for k in fields)):
            raise ValueError('Validated finite offline error envelope required')
        self.graph=graph;self.profile=p;self.records={}
        self.model=graph.add('capability',dict(type='conditional_memory_error_model',profile=p))

    def observe(self,obj,surface,geometry,*,source,calibration,stamp,evidence_ref,tcp,
                attached=False,grasp_epoch=None):
        tcp=np.asarray(tcp,float);R=rotation(tcp[3:7]);center=np.asarray(surface['center'],float)
        if not np.isfinite(center).all() or not calibration:return None
        old=self.records.get(obj)
        if old and attached and old.get('grasp_epoch')==grasp_epoch and (
            old['status']=='CONTRADICTED' or old.get('invalid_reason') in
            ('slip','grasp_lost','unknown_effect','calibration_changed','touch')):
            # A subsequent same-view fit must not silently reset a slipped grip.
            return None
        if old:self.graph.invalidate([old['node']],'memory_direct_estimate_replaced')
        data=dict(object=obj,surface=copy.deepcopy(surface),geometry=copy.deepcopy(geometry),
            calibration=calibration,stamp=float(stamp),last_stamp=float(stamp),evidence_ref=evidence_ref,
            offset=(R.T@(center-tcp[:3])).tolist(),rotation=R.tolist(),tcp=tcp.tolist(),
            path_m=0.,angle_rad=0.,attached=attached,grasp_epoch=grasp_epoch,
            status='DIRECT',current_status='DIRECT',current_visual_detection=True)
        node=self.graph.add('state_memory',dict(data,source_id=source,frame='robot_base'),parents=[source,self.model])
        self.records[obj]=dict(data,node=node)
        return node

    def invalidate(self,obj,reason):
        r=self.records.get(obj)
        if r:
            self.graph.invalidate([r['node']],'state_memory_'+reason)
            r.update(status='UNKNOWN',current_status='UNKNOWN',invalid_reason=reason)
            self.graph._event('memory_invalidated',object=obj,source_id=r['node'],reason=reason)

    def contradict(self,obj,surface,*,source):
        self.invalidate(obj,'contradictory_current_evidence')
        if obj in self.records:self.records[obj].update(status='CONTRADICTED',current_status='CONTRADICTED')
        self.graph.add('state_conflict',dict(object=obj,current=surface,permission=False),parents=[source])

    def partial_conflict(self,obj,points,tcp):
        """Reject gross contradictory pixels; containment never verifies pose."""
        r=self.records.get(obj)
        if not r:return False
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<30:return False
        center=np.asarray(r['surface']['center'])
        if r['attached']:
            center=np.asarray(tcp[:3])+rotation(tcp[3:7])@np.asarray(r['offset'])
        # Any upright cube yaw fits this conservative enclosing prism. This
        # check is one-sided refutation, not a source of narrower bounds.
        h=r['geometry']['height_m'];b=self.profile
        span=np.array([h/np.sqrt(2)+2*b['xy_m']]*2+[h/2+2*b['z_m']+b['height_m']])
        outside=(np.abs(p-center)>span).any(axis=1)
        return bool(outside.sum()>=30 and outside.mean()>.2)

    def estimate(self,obj,*,tcp,stamp,calibration,grasp_epoch=None,hold_verified=False,
                 occlusion='unexplained',static_clear=False):
        r=self.records.get(obj);p=self.profile
        def unknown(reason):
            if r:r['current_status']='UNKNOWN'
            return dict(status='UNKNOWN',reason=reason,current_visual_detection=False)
        if not r or r['status'] in ('UNKNOWN','CONTRADICTED') or not self.graph.nodes[r['node']]['valid']:
            return unknown((r or {}).get('invalid_reason','no_valid_memory'))
        if r['calibration']!=calibration:
            self.invalidate(obj,'calibration_changed');return unknown('calibration_changed')
        age=float(stamp)-r['stamp']
        if age<0 or age>p['max_age_seconds']:return unknown('memory_age_limit')
        tcp=np.asarray(tcp,float)
        if tcp.shape!=(8,) or not np.isfinite(tcp).all():return unknown('invalid_proprioception')
        surface=copy.deepcopy(r['surface']);path=r['path_m'];angle=r['angle_rad'];tilt=0.
        if r['attached']:
            if not hold_verified or grasp_epoch!=r['grasp_epoch'] or tcp[7]<.05:
                self.invalidate(obj,'attachment_precondition_lost');return unknown('attachment_precondition_lost')
            R=rotation(tcp[3:7]);last=np.asarray(r['tcp']);last_R=rotation(last[3:7])
            path+=float(np.linalg.norm(tcp[:3]-last[:3]))
            angle+=float(np.arccos(np.clip((np.trace(R@last_R.T)-1)/2,-1,1)))
            r.update(path_m=path,angle_rad=angle,tcp=tcp.tolist(),last_stamp=float(stamp))
            if path>p['max_travel_m'] or angle>p['max_rotation_rad']:return unknown('motion_envelope_exhausted')
            initial=np.asarray(r['rotation'])
            tilt=float(np.arccos(np.clip(R[:,2]@initial[:,2],-1,1)))
            if tilt>np.deg2rad(5):return unknown('upright_attachment_invalid')
            center=tcp[:3]+R@np.asarray(r['offset']);delta=center-np.asarray(surface['center'])
            for key in ('center','lo','hi'):surface[key]=(np.asarray(surface[key])+delta).tolist()
            if 'bottom_z' in surface:surface['bottom_z']+=float(delta[2])
        elif not static_clear:return unknown('static_support_motion_not_excluded')
        if occlusion!='expected':return unknown('unexplained_missing_observation')
        uncertainty=dict(initial_xy_m=p['xy_m'],initial_top_z_m=p['z_m'],
            intrinsic_height_m=p['height_m'],translation_m=path*p['translation_rate'],
            rotation_m=angle*p['rotation_rate'],age_m=age*p['age_rate'],
            static_m=0. if r['attached'] else age*p['static_rate'],
            # A rigid side length does not change when translated. Tilt changes
            # its vertical extent instead: bound any corner's displacement by
            # 2*r*sin(theta/2), without assuming an unmeasured object yaw.
            tilted_extent_m=r['geometry']['height_m']*math.sqrt(3)*math.sin(tilt/2),
            correlation_cancellation_m=0.,
            initial_pose_attachment_calibration='lumped_empirical_envelope_not_separately_identified')
        growth=sum(uncertainty[k] for k in ('translation_m','rotation_m','age_m','static_m'))
        bounds=dict(xy_m=p['xy_m']+growth,z_m=p['z_m']+growth+uncertainty['tilted_extent_m'],
            height_m=p['height_m'])
        data=dict(status='PREDICTED',object=obj,surface=surface,bounds=bounds,
            source_id=r['node'],evidence_ref=r['evidence_ref'],calibration=calibration,
            age_seconds=age,travel_m=path,rotation_rad=angle,current_visual_detection=False,
            uncertainty=uncertainty,propagation_model='rigid_dimension_separate_from_pose_v23.1',
            assumptions=['bounded calibration/geometry error','bounded unobserved disturbance',
                'verified rigid attachment' if r['attached'] else 'bounded static support during certified noncontact motion'])
        r['current_status']='PREDICTED'
        self.graph._event('memory_prediction',**data)
        return data

    def forecast(self,obj,**kwargs):
        """Read-only future estimate for evidence planning, NEVER permission.

        Forecasting must not spend travelled distance, renew a timestamp, emit
        a current observation or invalidate live nodes before the motion occurs.
        """
        class ReadOnlyGraph:
            nodes=self.graph.nodes
            def invalidate(self,*a,**k):pass
            def _event(self,*a,**k):pass
        model=copy.copy(self);model.records=copy.deepcopy(self.records);model.graph=ReadOnlyGraph()
        return dict(model.estimate(obj,**kwargs),permission=False,forecast=True)


def bounded_place_metrics(held,support,geometry,held_bounds,support_bounds):
    """Universal interval inclusion within the UNCHANGED release thresholds."""
    result=place_metrics(held,support,geometry)
    if result['status']!='PASS':return result
    if held_bounds is None or support_bounds is None:
        return dict(result,status='UNCERTAIN',reason='missing_validated_error_bounds')
    xy=held_bounds['xy_m']+support_bounds['xy_m']
    z=held_bounds['z_m']+held_bounds['height_m']+support_bounds['z_m']
    gap=result['gap_m'];safe=result['xy_error_m']+xy<=.008 and gap-z>=-.004 and gap+z<=.016
    return dict(result,status='PASS' if safe else 'UNCERTAIN',
        reason='bounded_predicted_alignment' if safe else 'prediction_interval_exceeds_action_contract',
        evidence='conditional_current_state_prediction_not_current_visual_detection',
        xy_error_upper_m=result['xy_error_m']+xy,gap_interval_m=[gap-z,gap+z],
        held_bounds=held_bounds,support_bounds=support_bounds)


def depth_occlusion(public,surface,height):
    """Current depth ahead of predicted geometry witnesses occlusion, not absence.

    Requires calibrated projection and most in-frame samples occluded. Empty
    depth, out of frame and merely missing segmentation cannot enable memory.
    This establishes line-of-sight blockage only, not attachment or no motion.
    """
    cams=[c for c in public.values() if isinstance(c,dict) and 'images' in c]
    if not cams:return 'unexplained'
    c=cams[0]
    try:
        k=np.asarray(c['intrinsics']);T=np.asarray(c['pose_mat']);d=np.asarray(c['images']['depth']).squeeze()
        center=np.asarray(surface['center']);grid=np.array([[x,y,z] for x in (-.3,0,.3) for y in (-.3,0,.3) for z in (0,.4)])*height+center
        q=(grid-T[:3,3])@T[:3,:3];uv=q@k.T;uv=uv[:,:2]/uv[:,2,None]
        valid=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<d.shape[1])&(uv[:,1]<d.shape[0])
        if valid.mean()<.8:return 'unexplained'
        ij=np.rint(uv[valid]).astype(int);ij[:,0]=np.clip(ij[:,0],0,d.shape[1]-1);ij[:,1]=np.clip(ij[:,1],0,d.shape[0]-1)
        observed=d[ij[:,1],ij[:,0]];occluded=(observed>0)&np.isfinite(observed)&(observed<q[valid,2]-.003)
        return 'expected' if occluded.mean()>=.7 else 'unexplained'
    except (KeyError,TypeError,ValueError,IndexError):return 'unexplained'
