"""Public pregrasp-to-close rigid attachment estimate, never simulator truth.

This is a conditional estimate, not a contact sensor. Current visible surfaces
must remain consistent, and large rotation/occlusion revokes permission. The
under-gripper visible top centroid is not used to silently recalibrate grasp.
"""
import numpy as np


def rotation(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.02:raise ValueError('Invalid WXYZ')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def calibrate(surface,geometry,close_tcp,*,source_id,tcp_id):
    if not surface or not geometry or not source_id or not tcp_id:return None
    tcp=np.asarray(close_tcp,float);height=float(geometry['height_m'])
    if not np.isfinite(tcp).all() or not .015<=height<=.09:return None
    center=np.r_[np.asarray(surface['center'])[:2],surface['hi'][2]-height/2]
    R=rotation(tcp[3:7]);offset=R.T@(center-tcp[:3])
    if np.linalg.norm(offset)>.06:return None
    return dict(offset_tcp=offset.tolist(),close_rotation=R.tolist(),height_m=height,
                sources=[source_id,tcp_id],frame='robot_base',
                assumption='rigid_attachment_since_observed_pregrasp_and_close',
                limitation='public geometric estimate, not independent tactile proof; slip/tilt may violate attachment')


def predict(state,tcp):
    tcp=np.asarray(tcp,float);R=rotation(tcp[3:7]);initial=np.asarray(state['close_rotation'])
    angle=float(np.arccos(np.clip((np.trace(R@initial.T)-1)/2,-1,1)))
    tilt=float(np.arccos(np.clip(R[:,2]@initial[:,2],-1,1)))
    upright=max(float(np.arccos(np.clip(abs(R[2,2]),-1,1))),float(np.arccos(np.clip(abs(initial[2,2]),-1,1))))
    center=tcp[:3]+R@np.asarray(state['offset_tcp']);h=state['height_m']
    return dict(center=center.tolist(),lo=(center-h/2).tolist(),hi=(center+h/2).tolist(),bottom_z=float(center[2]-h/2),
                height_m=h,orientation_verified=bool(tilt<=np.deg2rad(5) and upright<=np.deg2rad(5)),
                rotation_since_close_rad=angle,tilt_since_close_rad=tilt,
                orientation_assumption='rigid grasp preserves upright cube; robot yaw allowed, object tilt not independently measured',
                source='public_pregrasp_center_and_current_robot_rigid_attachment',sources=state['sources'])



def _cloud(value):
    try:p=np.asarray(value,float)
    except (TypeError,ValueError):return np.empty((0,3))
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():return np.empty((0,3))
    return p if len(p)<=4096 else p[np.linspace(0,len(p)-1,4096,dtype=int)]


def _top(points,h):
    if len(points)<80:return None
    z=float(np.quantile(points[:,2],.99));q=points[abs(points[:,2]-z)<=min(.0015,.04*h)]
    if len(q)<30:return None
    center=q.mean(0);_,s,vh=np.linalg.svd(q-center,full_matrices=False)
    if s[1]/np.sqrt(len(q))<.002 or abs(vh[-1,2])<np.cos(np.deg2rad(7)):return None
    if np.quantile(abs((q-center)@vh[-1]),.9)>.0012:return None
    return float(np.median(q[:,2])),q


def _top_coverage(q):
    lo,hi=np.quantile(q[:,:2],[.01,.99],axis=0)
    if np.any(hi-lo<.005):return 0.
    ij=np.floor((q[:,:2]-lo)/(hi-lo)*6).astype(int);ij=np.clip(ij,0,5)
    counts=np.zeros((6,6),int);np.add.at(counts,(ij[:,0],ij[:,1]),1)
    return float(np.mean(counts>=2))


def _two_faces(points,h,top_z):
    """Locate an upright cube from TWO current vertical faces, not a prior box.

    A half-degree yaw search uses the existing support geometry's resolution.
    No proposed centre is ranked by agreement with the attachment prediction.
    Both faces need 2-D extent and independent normal constraints; containment
    merely rejects contradictory points and cannot itself establish a centre.
    """
    side=points[points[:,2]<top_z-.003];tol=min(.0015,.04*h)
    if len(side)<70:return None
    candidates=[]
    for angle in np.arange(180)*np.pi/360:
        u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-np.sin(angle),np.cos(angle)])
        faces=[]
        for axis,tangent in ((u,v),(v,u)):
            found=[];low,high=np.quantile(side[:,:2]@axis,[.02,.98])
            for boundary,sign in ((low,-1),(high,1)):
                q=side[abs(side[:,:2]@axis-boundary)<=tol]
                if len(q)<30:continue
                plane=float(np.median(q[:,:2]@axis));q=side[abs(side[:,:2]@axis-plane)<=tol]
                span=np.ptp(np.quantile(np.c_[q[:,:2]@tangent,q[:,2]],[.05,.95],axis=0),axis=0)
                if np.any(span<.35*h) or np.any(span>1.2*h):continue
                center=q.mean(0);_,singular,vh=np.linalg.svd(q-center,full_matrices=False);normal=vh[-1]
                residual=float(np.quantile(abs((q-center)@normal),.9))
                if singular[1]/np.sqrt(len(q))<.003 or abs(normal[2])>np.sin(np.deg2rad(7)) or residual>.0012:continue
                normal=normal if normal[:2]@axis>=0 else -normal
                found.append(dict(center_projection=plane-sign*h/2,normal=normal,residual=residual,points=len(q),span=span.tolist(),sample=q,sign=sign))
            faces.append(found)
        for first in faces[0]:
            for second in faces[1]:
                if abs(first['normal']@second['normal'])>.14:continue
                # The search strips discover faces; their coarse axes are not
                # measured orientation. Refine with the actual paired PCA normals
                # and fit plane offsets again in one orthogonal upright basis.
                first_axis=first['normal'][:2]/np.linalg.norm(first['normal'][:2])
                second_axis=second['normal'][:2]/np.linalg.norm(second['normal'][:2])
                measured_u=first_axis+np.array([second_axis[1],-second_axis[0]])
                if np.linalg.norm(measured_u)<1e-8:continue
                measured_u/=np.linalg.norm(measured_u);measured_v=np.array([-measured_u[1],measured_u[0]])
                offset_u=float(np.median(first['sample'][:,:2]@measured_u))-first['sign']*h/2
                offset_v=float(np.median(second['sample'][:,:2]@measured_v))-second['sign']*h/2
                xy=measured_u*offset_u+measured_v*offset_v
                measured_angle=float(np.arctan2(measured_u[1],measured_u[0]))
                local=np.c_[(points[:,:2]-xy)@measured_u,(points[:,:2]-xy)@measured_v,points[:,2]-(top_z-h)]
                fraction=float(np.mean(np.all(abs(local[:,:2])<=h/2+.004,axis=1)&(local[:,2]>=-.004)&(local[:,2]<=h+.004)))
                if fraction<.95:continue
                candidates.append(dict(center=xy,angle=measured_angle,search_angle=float(angle),contained_fraction=fraction,
                    residual=first['residual']+second['residual'],faces=[first,second]))
    if not candidates:return None
    spread=np.ptp(np.array([f['center'] for f in candidates]),axis=0)
    if np.max(spread)>.006:return None
    best=min(candidates,key=lambda f:(f['residual'],-sum(v['points'] for v in f['faces']),f['angle']))
    xy=best['center'];a=best['angle'];u=np.array([np.cos(a),np.sin(a)]);v=np.array([-np.sin(a),np.cos(a)]);half=(abs(u)+abs(v))*h/2
    return dict(center=[*xy.tolist(),top_z-h/2],lo=[*(xy-half).tolist(),top_z-h],hi=[*(xy+half).tolist(),top_z],
        bottom_z=top_z-h,height_m=h,frame='robot_base',yaw_rad=a,orientation_source='paired_current_PCA_plane_normals',search_strip_yaw_rad=best['search_angle'],
        source='current_two_visible_vertical_faces_and_top_with_prior_public_cube_side',
        face_point_counts=[v['points'] for v in best['faces']],
        face_residual_p90_m=[v['residual'] for v in best['faces']],
        face_spans_m=[v['span'] for v in best['faces']],center_candidate_spread_xy_m=spread.tolist(),
        candidate_count=len(candidates),contained_fraction=best['contained_fraction'])


def _complete_held_face(points,top_z,geometry,camera_origin):
    """Reuse the existing complete-face fitter with a measured held bottom.

    The bottom reference comes from CURRENT top minus PRIOR PUBLIC side, and
    the existing fitter still requires current pixels along that bottom edge.
    It is not a table plane or contact observation. A partial face cannot pass.
    """
    if camera_origin is None:return None
    from .support_geometry import current_support
    h=float(geometry['height_m'])
    fit=current_support(points,top_z-h,geometry,camera_origin=camera_origin)
    if not fit or fit.get('source')!='current_RGBD_complete_vertical_face_camera_side_and_public_cube_side':return None
    evidence=dict(fit['fit_evidence']);reference=evidence.pop('table_z')
    normal=np.asarray(evidence['outward_normal_xy'])
    evidence.update(observed_bottom_edge=True,bottom_reference_z_m=reference,
        bottom_reference_source='current_top_minus_prior_public_side_with_observed_bottom_pixels')
    fit.update(source='current_complete_vertical_face_and_observed_held_bottom',
        height_m=h,yaw_rad=float(np.arctan2(normal[1],normal[0])),fit_evidence=evidence,
        assumptions=['upright rigid cube','prior public side length remains valid',
            'current full face horizontal and bottom edges are observed',
            'calibrated camera selects exterior face; no table contact inference'],
        limitation='same_camera_complete_face_inference_not_independent_hold_or_contact_proof')
    return fit


def check_surface(predicted,visible,*,points=None,geometry=None,camera_origin=None):
    """Visibility-sensitive same-camera consistency, NEVER independent hold proof.

    A complete-top screen keeps the original 15 mm XY / 10 mm Z limits. A partial
    centroid is diagnostic only. Partial observations need two currently visible
    vertical faces plus top, or a complete face with observed bottom edges and
    public camera side, a previously PUBLIC cube side, and conditional upright
    grasp evidence. A partial single face, interior points, missing top or
    ambiguous fits remain UNCERTAIN. No prior offset is recalibrated here.
    """
    from .support_geometry import reliable_top
    result=dict(status='UNCERTAIN',reason='held_surface_occluded',visibility='unavailable',
        center_observable=False,conditional=True,independent_verification=False,
        xy_threshold_m=.015,top_threshold_m=.010,
        limitation='same-camera geometry with prior public cube side and conditional rigid upright grasp; not independent contact/slip proof or calibrated uncertainty')
    if not visible:return result
    try:
        center=np.asarray(predicted['center'],float);visible_center=np.asarray(visible['center'],float)
        hi=np.asarray(predicted['hi'],float);vhi=np.asarray(visible['hi'],float)
        h=float((geometry or {}).get('height_m',predicted['height_m']))
        if any(a.shape!=(3,) or not np.isfinite(a).all() for a in (center,visible_center,hi,vhi)) or not .015<=h<=.09:return result
        if geometry and (geometry.get('frame','robot_base')!='robot_base' or geometry.get('shape','cube')!='cube'):return result
    except (KeyError,TypeError,ValueError):return result
    raw_xy=float(np.linalg.norm(center[:2]-visible_center[:2]));raw_top=abs(float(hi[2]-vhi[2]))
    result.update(xy_disagreement_m=raw_xy,top_disagreement_m=raw_top,
                  centroid_is_full_center=False,visibility='partial_top')
    if not predicted.get('orientation_verified'):
        result.update(status='FAIL',reason='conditional_upright_attachment_orientation_invalid');return result
    g=dict(geometry or {},frame='robot_base',height_m=h)
    p=_cloud(points);top=_top(p,h) if len(p) else None
    full=reliable_top(visible,g)
    if points is not None:
        coverage=_top_coverage(top[1]) if top else 0.
        result.update(top_grid_coverage=coverage,point_count=len(p))
        full=full and top is not None and coverage>=.7
    if full:
        good=raw_xy<=.015 and raw_top<=.010
        fitted=None
        if top is not None:
            fitted=_two_faces(p,h,top[0])
            if fitted is None:fitted=_complete_held_face(p,top[0],g,camera_origin)
        if fitted is not None:
            delta=float(np.linalg.norm(np.asarray(fitted['center'])[:2]-center[:2]));zerror=abs(float(fitted['hi'][2])-hi[2])
            good=good and delta<=.015 and zerror<=.010
            result.update(current_surface=fitted,current_shape_center_disagreement_m=delta,
                current_shape_top_disagreement_m=zerror)
        result.update(status='PASS' if good else 'FAIL',visibility='full_top',center_observable=True,
            centroid_is_full_center=True,reason='full_top_consistent_with_attachment' if good else 'full_top_attachment_disagreement',
            localized_center_disagreement_m=raw_xy,evidence='current_complete_top_heuristic_screen')
        return result
    if top is None:
        result['reason']='partial_surface_has_no_observable_top_and_center';return result
    fitted=_two_faces(p,h,top[0])
    if fitted is None:fitted=_complete_held_face(p,top[0],g,camera_origin)
    if fitted is None:
        result['reason']='partial_surface_center_unobservable_requires_new_view';return result
    delta=float(np.linalg.norm(np.asarray(fitted['center'])[:2]-center[:2]));zerror=abs(top[0]-hi[2])
    good=delta<=.015 and zerror<=.010
    result.update(status='PASS' if good else 'FAIL',center_observable=True,
        reason='current_visible_faces_consistent_with_attachment' if good else 'current_visible_faces_attachment_disagreement',
        localized_center_disagreement_m=delta,localized_top_disagreement_m=zerror,
        evidence=('current_complete_face_camera_side_and_observed_bottom_plus_prior_public_side'
            if fitted.get('source')=='current_complete_vertical_face_and_observed_held_bottom'
            else 'two_current_vertical_faces_and_horizontal_top_plus_prior_public_cube_side'),
        current_surface=fitted)
    return result
