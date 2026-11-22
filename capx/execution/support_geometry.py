"""Conservative same-camera support geometry, never simulator truth.

A size prior is a previously PUBLICLY measured upright cube side, not a hidden
mesh extent. Two current vertical faces and visible table-level bottom edges can
constrain a current centre. A complete single face additionally requires camera
side and full horizontal/bottom edges; a partial single face cannot. All thresholds are geometric
heuristics, not calibrated uncertainty or contact proof.

``retained_consistent`` ONLY rejects contradictions in currently visible points.
True does not locate an object, prove no motion, or renew a historical version.
The monitor must separately require unchanged object epoch/no possible contact.
"""
from __future__ import annotations
import numpy as np


def _points(value):
    try:p=np.asarray(value,dtype=float)
    except (TypeError,ValueError):return np.empty((0,3))
    if p.ndim!=2 or p.shape[1]!=3:return np.empty((0,3))
    return p[np.isfinite(p).all(axis=1)]


def _side(geometry):
    if not isinstance(geometry,dict) or geometry.get('frame')!='robot_base':return None
    if geometry.get('shape','cube')!='cube':return None
    try:h=float(geometry['height_m'])
    except (KeyError,TypeError,ValueError):return None
    if not np.isfinite(h) or not .015<=h<=.09:return None
    return h


def color_points(public_obs,name):
    """Current calibrated RGB-D colour pixels, XYZ in robot-base metres."""
    if name not in ('red','red cube','green','green cube'):raise ValueError('Unknown cube colour')
    cameras=[v for v in public_obs.values() if isinstance(v,dict) and 'images' in v]
    if not cameras:return np.empty((0,3))
    cam=cameras[0]
    try:
        rgb=np.asarray(cam['images']['rgb'],float);d=np.asarray(cam['images']['depth'],float).squeeze()
        k=np.asarray(cam['intrinsics'],float);t=np.asarray(cam['pose_mat'],float)
    except (KeyError,TypeError,ValueError):return np.empty((0,3))
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or d.shape!=rgb.shape[:2] or k.shape!=(3,3) or t.shape!=(4,4):return np.empty((0,3))
    if not np.isfinite(k).all() or not np.isfinite(t).all() or k[0,0]<=0 or k[1,1]<=0:return np.empty((0,3))
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    mask=(r>50)&(r>1.5*g)&(r>1.5*b) if name.startswith('red') else (g>45)&(g>1.4*r)&(g>1.3*b)
    yy,xx=np.indices(d.shape)
    p=np.stack([(xx-k[0,2])*d/k[0,0],(yy-k[1,2])*d/k[1,1],d],axis=-1)@t[:3,:3].T+t[:3,3]
    return p[mask&(d>0)&np.isfinite(p).all(axis=-1)]


def reliable_top(surface,geometry):
    """Heuristic full-footprint screen; not proof of an unoccluded full plane.

    For an upright square, both axis-aligned spans are approximately equal and
    lie between one side and sqrt(2) sides. Tiny/one-sided remnants are rejected.
    It is still the same RGB-D sensor and can have correlated systematic bias.
    """
    h=_side(geometry)
    if not isinstance(surface,dict) or h is None:return False
    try:
        bounds=np.asarray(surface['top_xy_bounds'],float)
        pixels=int(surface['top_pixels'])
    except (KeyError,TypeError,ValueError,OverflowError):return False
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or pixels<80:return False
    span=bounds[1]-bounds[0]
    return bool(np.min(span)>=.85*h and np.max(span)<=1.5*h and np.max(span)<=1.2*np.min(span))


def retained_consistent(points,prior_surface,geometry):
    """Only visible-point noncontradiction, NEVER full current localization.

    The caller must gate retention on unchanged object epoch and no possible
    contact. Occluded portions are untested, and a hidden motion can pass this
    check. Do not relabel the prior as a new observation or use this after touch.
    """
    p=_points(points);h=_side(geometry)
    if h is None or len(p)<30 or not isinstance(prior_surface,dict):return False
    if prior_surface.get('frame','robot_base')!='robot_base':return False
    try:lo=np.asarray(prior_surface['lo'],float);hi=np.asarray(prior_surface['hi'],float)
    except (KeyError,TypeError,ValueError):return False
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite(np.r_[lo,hi]).all() or np.any(lo>hi):return False
    tolerance=min(.004,.1*h)
    # The prior footprint must itself have passed the complete-top screen.
    if not reliable_top(prior_surface,geometry):return False
    low=lo.copy();low[2]=hi[2]-h
    inside=np.all((p>=low-tolerance)&(p<=hi+tolerance),axis=1)
    return bool(np.mean(inside)>=.95)


def _face(p,axis,tangent,boundary,sign,h,table_z):
    # At most 1.5mm strip; enough 2-D span is required to reject a thin edge
    # of one face being mistaken for a second perpendicular face.
    distance=p[:,:2]@axis-boundary
    q=p[np.abs(distance)<=min(.0015,.04*h)]
    if len(q)<35:return None
    plane=float(np.median(q[:,:2]@axis))
    q=p[np.abs(p[:,:2]@axis-plane)<=min(.0015,.04*h)]
    uv=q[:,:2]@tangent;zz=q[:,2]
    spans=np.quantile(np.c_[uv,zz],[.05,.95],axis=0)
    span=spans[1]-spans[0]
    if np.any(span<.60*h) or np.any(span>1.15*h):return None
    if np.quantile(zz,.05)>table_z+.004 or np.quantile(zz,.01)<table_z-.004:return None
    # PCA uses only this putative side. An inclined plane or line stays unknown.
    center=q.mean(axis=0);_,singular,vh=np.linalg.svd(q-center,full_matrices=False)
    if singular[1]/np.sqrt(len(q))<.006:return None
    normal=vh[-1]
    if abs(normal[2])>np.sin(np.deg2rad(5)):return None
    residual=np.abs((q-center)@normal)
    if np.quantile(residual,.9)>.0012:return None
    return dict(points=q,axis=axis,tangent=tangent,plane=plane,sign=sign,
                center_projection=plane-sign*h/2,normal=normal,
                residual_p90=float(np.quantile(residual,.9)),span=span)


def current_support(points,table_z,geometry,*,camera_origin=None):
    """Infer an upright cube from TWO current adjacent vertical faces or None.

    Uses a deterministic half-degree orientation search, current bottom-edge
    evidence and an earlier public cube-side measurement. Never fills a missing
    second face using a stale centre. Tilt, airborne objects and weak coverage
    remain unknown. The output is an inference under stated cube assumptions.
    """
    p=_points(points);h=_side(geometry)
    if h is None or len(p)<80 or isinstance(table_z,bool):return None
    try:table=float(table_z)
    except (TypeError,ValueError):return None
    if not np.isfinite(table):return None
    if len(p)>4096:p=p[np.linspace(0,len(p)-1,4096,dtype=int)]
    zlo,zhi=np.quantile(p[:,2],[.01,.99])
    if zlo<table-.004 or zlo>table+.004 or zhi>table+h+.004:return None
    side=p[(p[:,2]>=table-.003)&(p[:,2]<=table+h-.003)]
    if len(side)<70:return None
    best=None;single_faces=[]
    for angle in np.arange(180)*np.pi/360:
        u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-np.sin(angle),np.cos(angle)])
        candidates=[]
        for axis,tangent in ((u,v),(v,u)):
            low,high=np.quantile(side[:,:2]@axis,[.02,.98])
            candidates.append([f for bound,sign in ((low,-1),(high,1))
                               if (f:=_face(side,axis,tangent,bound,sign,h,table)) is not None])
        single_faces.extend(candidates[0]+candidates[1])
        for first in candidates[0]:
            for second in candidates[1]:
                if abs(float(np.dot(first['normal'],second['normal'])))>.14:continue
                center_xy=u*first['center_projection']+v*second['center_projection']
                fits=[];okay=True
                for face,tangent in ((first,v),(second,u)):
                    q=face['points'];horizontal=q[:,:2]@tangent-center_xy@tangent
                    z=q[:,2]-table
                    idx=np.floor(np.c_[(horizontal+h/2)/h,z/h]*4).astype(int)
                    valid=np.all((idx>=0)&(idx<4),axis=1);grid=np.zeros((4,4),bool)
                    grid[idx[valid,0],idx[valid,1]]=True
                    coverage=float(grid.mean())
                    if coverage<.625 or np.count_nonzero(grid.any(axis=0))<3 or np.count_nonzero(grid.any(axis=1))<3:
                        okay=False;break
                    fits.append(dict(points=len(q),normal=face['normal'].tolist(),
                        residual_p90_m=face['residual_p90'],span_m=face['span'].tolist(),
                        grid_coverage=coverage,bottom_z_p05=float(np.quantile(q[:,2],.05))))
                if not okay:continue
                local=np.c_[(p[:,:2]-center_xy)@u,(p[:,:2]-center_xy)@v,p[:,2]-table]
                contained=np.all((np.abs(local[:,:2])<=h/2+.003),axis=1)&(local[:,2]>=-.004)&(local[:,2]<=h+.004)
                if np.mean(contained)<.95:continue
                score=(first['residual_p90']+second['residual_p90'],-sum(f['points'] for f in fits),float(angle))
                if best is None or score<best[0]:best=(score,center_xy,u,v,fits,float(np.mean(contained)))
    if best is None:return _single_complete_face(p,table,h,single_faces,camera_origin)
    _,xy,u,v,fits,contained=best
    half=(np.abs(u)+np.abs(v))*h/2;center=[*xy,table+h/2]
    return dict(frame='robot_base',center=center,lo=[*(xy-half),table],hi=[*(xy+half),table+h],
        top_z=table+h,bottom_z=table,top_xy_bounds=[(xy-half).tolist(),(xy+half).tolist()],
        top_pixels=0,pixels=len(p),source='current_RGBD_two_vertical_faces_with_public_cube_side',
        assumptions=['rigid cube','upright on currently visible table bottom edge','prior public side length remains valid'],
        fit_evidence=dict(current_vertical_faces=2,faces=fits,contained_fraction=contained,
            measured_side_m=h,table_z=table,independent_sensor=False,
            upright_normal_tolerance_deg=5.,grid_orientation_resolution_deg=.5,
            uncertainty_kind='heuristic_uncalibrated',complete_current_localization_under_assumptions=True),
        limitation='same_camera_inference_not_contact_proof_or_calibrated_uncertainty')


def _single_complete_face(p,table,h,faces,camera_origin):
    """A complete current square face plus camera side resolves cube centre.

    Unlike a partial plane, BOTH current horizontal edges and a table-level
    bottom edge must span the prior side length. Camera pose selects the exterior
    normal; no old centre, unseen second face, or evaluator state is used.
    """
    if camera_origin is None:return None
    camera=np.asarray(camera_origin,float)
    if camera.shape!=(3,) or not np.isfinite(camera).all():return None
    best=None
    for face in faces:
        q=face['points'];normal=face['normal'][:2].copy();normal/=np.linalg.norm(normal)
        mid=q[:,:2].mean(axis=0);toward=camera[:2]-mid
        if np.dot(normal,toward)<0:normal=-normal
        if np.dot(normal,toward)<.25*np.linalg.norm(toward) or np.dot(normal,toward)<h:continue
        tangent=np.array([-normal[1],normal[0]])
        along=q[:,:2]@tangent;edges=np.quantile(along,[.01,.99]);width=edges[1]-edges[0]
        if not .90*h<=width<=1.06*h:continue
        bottom=q[q[:,2]<=table+.004]
        if len(bottom)<15 or np.ptp(bottom[:,:2]@tangent)<.90*h:continue
        if np.quantile(q[:,2],.95)-np.quantile(q[:,2],.05)<.80*h:continue
        plane=float(np.median(q[:,:2]@normal));xy=normal*(plane-h/2)+tangent*float(np.mean(edges))
        uv=np.c_[(q[:,:2]-xy)@tangent/h+.5,(q[:,2]-table)/h]
        idx=np.floor(uv*6).astype(int);valid=np.all((idx>=0)&(idx<6),axis=1);grid=np.zeros((6,6),bool);grid[idx[valid,0],idx[valid,1]]=True
        if grid.mean()<.80 or not grid[:,0].all():continue
        local=np.c_[(p[:,:2]-xy)@normal,(p[:,:2]-xy)@tangent,p[:,2]-table]
        inside=(np.abs(local[:,:2])<=h/2+.003).all(axis=1)&(local[:,2]>=-.004)&(local[:,2]<=h+.004)
        contained=float(np.mean(inside))
        if contained<.95:continue
        score=(face['residual_p90'],-len(q),plane)
        if best is None or score<best[0]:best=(score,xy,normal,tangent,q,width,float(grid.mean()),contained)
    if best is None:return None
    _,xy,normal,tangent,q,width,coverage,contained=best;half=(np.abs(normal)+np.abs(tangent))*h/2
    return dict(frame='robot_base',center=[*xy,table+h/2],lo=[*(xy-half),table],hi=[*(xy+half),table+h],
        top_z=table+h,bottom_z=table,top_xy_bounds=[(xy-half).tolist(),(xy+half).tolist()],top_pixels=0,pixels=len(p),
        source='current_RGBD_complete_vertical_face_camera_side_and_public_cube_side',
        assumptions=['rigid cube','upright on current visible table bottom edge','complete horizontal face edges span prior public side length','opaque exterior face faces calibrated camera'],
        fit_evidence=dict(current_vertical_faces=1,camera_resolved_outward_normal=True,camera_origin_base=camera.tolist(),
            outward_normal_xy=normal.tolist(),visible_face_width_m=width,face_grid_coverage=coverage,
            face_points=len(q),face_residual_p90_m=best[0][0],contained_fraction=contained,measured_side_m=h,table_z=table,
            independent_sensor=False,upright_normal_tolerance_deg=5.,uncertainty_kind='heuristic_uncalibrated',
            complete_current_localization_under_assumptions=True),
        limitation='same_camera_complete_face_inference_not_contact_proof_or_calibrated_uncertainty')
