"""Public RGB-D geometry used by the shared G0/G1 substrate; no simulator state.

A visible top and a previously measured rigid height estimate define a virtual
bottom. This reduces side-surface truncation bias, but is still same-camera
inference, not tactile proof. Unknown or occluded measurements stay unknown.
"""
from __future__ import annotations
import copy
import numpy as np
from .feedback import surface_evidence


def point_cloud(obs):
    cams=[v for v in obs.values() if isinstance(v,dict) and 'images' in v]
    if not cams:return np.empty((0,3))
    c=cams[0];d=np.asarray(c['images']['depth']).squeeze();k=np.asarray(c['intrinsics']);t=np.asarray(c['pose_mat'])
    yy,xx=np.indices(d.shape)
    p=np.stack([(xx-k[0,2])*d/k[0,0],(yy-k[1,2])*d/k[1,1],d],axis=-1)
    p=p@t[:3,:3].T+t[:3,3]
    return p[(d>0)&np.isfinite(p).all(axis=-1)]


def table_plane(points):
    p=np.asarray(points)
    # Geometric workspace, not an object location or Seed-specific correction.
    p=p[(p[:,0]>.15)&(p[:,0]<1)&(np.abs(p[:,1])<.6)&(p[:,2]>-.5)&(p[:,2]<.25)]
    if len(p)<200:return None
    bins=np.round(p[:,2]/.002).astype(int);u,n=np.unique(bins,return_counts=True)
    for b in u[np.argsort(n)[::-1]][:6]:
        q=p[np.abs(p[:,2]-b*.002)<.003]
        if len(q)>200 and np.all(np.ptp(q[:,:2],axis=0)>.20):
            return float(np.median(q[:,2]))
    return None


def capture_geometry(obs):
    surfaces=surface_evidence(obs)
    points=point_cloud(obs)
    return surfaces,table_plane(points),points


def object_geometry(name,surface,table_z,surfaces,previous=None):
    if previous is not None:
        if previous.get('object')!=name or previous.get('frame')!='robot_base':return None
        return copy.deepcopy(previous)
    if not surface or table_z is None:return None
    support_z=table_z;support_object=None
    source='visible_top_minus_observed_support_plane'
    overlapping_lower=[]
    for other,value in surfaces.items():
        if other in name or not value:continue
        if (np.linalg.norm(np.asarray(surface['center'])[:2]-np.asarray(value['center'])[:2])<.045
            and value['hi'][2]<surface['hi'][2]-.012):
            overlapping_lower.append((other,value))
    if overlapping_lower:
        # An occluded lower side must not turn total stack height into the
        # carried object's height by silently falling back to the table.
        supported=[(other,value) for other,value in overlapping_lower
                   if abs(surface['lo'][2]-value['hi'][2])<.02]
        if len(supported)!=1:return None
        support_object,value=supported[0];support_z=float(value['hi'][2])
        source='visible_top_minus_observed_object_support'
    elif abs(surface['lo'][2]-table_z)>.02:
        # Elevated/side-truncated object without a visible support is unknown.
        return None
    height=float(surface['hi'][2]-support_z)
    if not .015<=height<=.09:return None
    return dict(frame='robot_base',object=name,height_m=height,source=source,
                top_z=float(surface['hi'][2]),support_z=support_z,support_object=support_object,
                orientation_policy='upright_only',orientation_verified=False,
                orientation_evidence='height_estimator_does_not_observe_object_tilt',
                uncertainty_m=None,heuristic_uncertainty_m=.004,
                uncertainty_kind='heuristic_uncalibrated',
                limitation='same_camera_upright_vertical_span_not_rotation_invariant_rigid_size')


def _orientation_usable(geometry):
    # The monitor must supply current public orientation/skill evidence for
    # newly measured geometry. Legacy diagnostic height-only inputs retain
    # their explicit historical semantics, not a new observation guarantee.
    return geometry.get('orientation_policy')!='upright_only' or geometry.get('orientation_verified') is True


def measured_surface(surface,geometry):
    if not surface:return None
    value={**surface,'frame':'robot_base'}
    if geometry and _orientation_usable(geometry):
        value['bottom_z']=float(surface['hi'][2]-geometry['height_m'])
        value['height_m']=geometry['height_m']
        value['bottom_source']='visible_top_minus_prior_public_upright_height'
        value['orientation_policy']=geometry.get('orientation_policy')
    elif geometry:
        value['height_status']='UNCERTAIN'
        value['height_reason']='upright_height_orientation_unverified'
    return value


def place_metrics(held,support,geometry):
    if not held or not support or not geometry:
        return dict(status='UNCERTAIN',reason='missing_current_surface_or_rigid_height',xy_error_m=None,gap_m=None)
    if not _orientation_usable(geometry):
        return dict(status='UNCERTAIN',reason='upright_height_orientation_unverified',xy_error_m=None,gap_m=None)
    delta=np.asarray(support['center'])[:2]-np.asarray(held['center'])[:2]
    gap=float(held['hi'][2]-geometry['height_m']-support['hi'][2])
    xy=float(np.linalg.norm(delta))
    good=xy<=.008 and -.004<=gap<=.016
    return dict(status='PASS' if good else 'FAIL',reason='measured_support_alignment' if good else 'release_geometry_risk',
                xy_error_m=xy,gap_m=gap,delta_xy_m=delta.tolist(),
                evidence='public_RGBD_top_minus_prior_upright_height',uncertainty_m=geometry.get('uncertainty_m'),
                heuristic_uncertainty_m=geometry.get('heuristic_uncertainty_m',.004),
                uncertainty_kind='heuristic_uncalibrated')


def _footprint_grid(plane, xy, half, *, minimum_points=1):
    """Same six-by-six visibility discretization for selection and revalidation."""
    counts=np.zeros((6,6),dtype=int)
    if len(plane):
        bins=np.floor((np.asarray(plane)[:,:2]-(xy-half))/(2*half)*6).astype(int)
        bins=np.clip(bins,0,5)
        np.add.at(counts,(bins[:,0],bins[:,1]),1)
    visible=counts>=minimum_points
    return visible,float(np.mean(visible)),bool(np.all(visible[2:4,2:4]))


def validate_table_region(points,table_z,region,held_points=None,context=None):
    """Revalidate a historical table footprint using CURRENT public RGB-D only.

    Current held colors may remove only one-to-one nearest full-cloud samples
    within 1e-7 m, not every point in a held-object bounding box. The parent owns
    point identity, frame and observation freshness; this function never reads
    a simulator or infers unobserved clearance from a historic flag.

    A partially hidden region can receive an explicitly conditional PASS only
    with parent context: camera_origin_base, current_observation_id,
    held_identity_verified, all_known_nonheld_visible_outside,
    static_scene_assumption. The parent must establish these contracts afresh,
    using its task family/phase and referenced public geometry. Prior complete
    clearance, substantial current visible table and current matched-held ray
    shadows must together explain every missing grid cell. This is same-camera
    evidence under a declared static-scene assumption, not independent proof.
    """
    result=dict(status='UNCERTAIN',reason='invalid_current_table_or_region',clearance_verified=False,
                conditional_static_scene=False,held_points_removed=0,held_match_tolerance_m=1e-7,
                obstacle_points=0,visible_plane_points=0,footprint_coverage_fraction=0.,
                center_coverage_verified=False,footprint_grid_shape=[6,6],
                limitation='visible_footprint_only_unobserved_volume_not_certified')
    try:
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():return result
        if table_z is None or not np.isfinite(table_z) or not isinstance(region,dict):return result
        xy=np.asarray(region['center'],float)[:2];lo=np.asarray(region['lo'],float);hi=np.asarray(region['hi'],float)
        if region.get('frame')!='robot_base' or xy.shape!=(2,) or lo.shape!=(3,) or hi.shape!=(3,):return result
        half=(hi[:2]-lo[:2])/2
        if not np.isfinite(np.r_[xy,lo,hi]).all() or np.any(half<=0) or np.any(half>.12):return result
        if not np.allclose(xy,(hi[:2]+lo[:2])/2,atol=1e-7):return result
        if 'footprint_half_extent_xy' in region and not np.allclose(region['footprint_half_extent_xy'],half,atol=1e-7):return result
        if abs(float(region.get('top_z',region['center'][2]))-table_z)>.004:
            result['reason']='current_table_plane_disagrees_with_region';return result
    except (KeyError,TypeError,ValueError,IndexError):return result
    held_current=np.empty((0,3))
    if held_points is not None:
        try:held=np.asarray(held_points,float)
        except (TypeError,ValueError):return result
        if held.ndim!=2 or held.shape[1]!=3 or not np.isfinite(held).all():return result
        if len(p) and len(held):
            from scipy.spatial import cKDTree
            distances,indices=cKDTree(p).query(held,k=1,distance_upper_bound=1e-7)
            indices=np.unique(indices[np.isfinite(distances)])
            keep=np.ones(len(p),dtype=bool);keep[indices]=False
            held_current=p[indices].copy();p=p[keep]
            result['held_points_removed']=len(indices)
    near=p[np.all(np.abs(p[:,:2]-xy)<=half+1e-12,axis=1)]
    plane=near[np.abs(near[:,2]-table_z)<.004]
    obstacles=near[near[:,2]>table_z+.012]
    # Points in the gap between the plane band and obstacle threshold are
    # unresolved elevated geometry, not permission to declare a clear surface.
    ambiguous=near[(near[:,2]>=table_z+.004)&(near[:,2]<=table_z+.012)]
    visible,coverage,center_visible=_footprint_grid(plane,xy,half)
    result.update(obstacle_points=len(obstacles),ambiguous_elevated_points=len(ambiguous),
                  visible_plane_points=len(plane),footprint_coverage_fraction=coverage,
                  center_coverage_verified=center_visible)
    if len(obstacles)>5:
        result.update(status='FAIL',reason='current_nonheld_obstacle_in_table_region');return result
    if len(obstacles) or len(ambiguous):
        result['reason']='unresolved_nonheld_elevated_points';return result
    if len(plane)>=60 and coverage>=.9 and center_visible:
        result.update(status='PASS',reason='current_visible_clear_table_patch',clearance_verified=True);return result
    result['reason']='current_table_region_occluded_or_insufficiently_visible'
    if not isinstance(context,dict):return result
    required=('held_identity_verified','all_known_nonheld_visible_outside','static_scene_assumption')
    if any(context.get(key) is not True for key in required) or not context.get('current_observation_id'):return result
    try:
        camera=np.asarray(context['camera_origin_base'],float)
        if camera.shape!=(3,) or not np.isfinite(camera).all() or camera[2]<=table_z+.05:return result
    except (KeyError,TypeError,ValueError):return result
    prior_ok=(region.get('clearance_verified') is True and region.get('footprint_coverage_fraction',0)>=.9
              and region.get('center_coverage_verified') is True and region.get('visible_plane_points',0)>=60)
    if not prior_ok or len(plane)<40 or coverage<.4 or len(held_current)<30:return result
    rays=held_current-camera
    with np.errstate(divide='ignore',invalid='ignore'):
        scale=(table_z-camera[2])/rays[:,2]
    usable=np.isfinite(scale)&(scale>=1.)&(held_current[:,2]>table_z+.012)
    shadow=camera+rays[usable]*scale[usable,None]
    shadow=shadow[np.all(np.abs(shadow[:,:2]-xy)<=half+1e-12,axis=1)]
    shadow_visible,shadow_fraction,_=_footprint_grid(shadow,xy,half,minimum_points=3)
    explained=visible|shadow_visible
    result.update(held_shadow_coverage_fraction=shadow_fraction,combined_coverage_fraction=float(np.mean(explained)),
                  unexplained_grid_cells=int(np.count_nonzero(~explained)),
                  observation_id=context['current_observation_id'],
                  nonheld_geometry_source_ids=copy.deepcopy(context.get('nonheld_geometry_source_ids',[])))
    if np.all(explained):
        result.update(status='PASS',reason='current_table_and_held_shadow_under_static_scene_assumption',
                      clearance_verified=True,conditional_static_scene=True,
                      limitation='same_camera_static_scene_condition_not_independent_clearance_proof; parent must link current identity and nonheld exclusion evidence')
    else:result['reason']='held_shadow_does_not_explain_missing_table_region'
    return result


def choose_table_region(points,table_z,surfaces,object_name,*,excluded_regions=()):
    if table_z is None or len(points)<200:return None
    current=surfaces.get('red' if 'red' in object_name else 'green')
    if not current:return None
    anchor=np.asarray(current['center'])[:2];p=np.asarray(points)
    # The footprint is measured from public visible bounds, with a 1cm margin.
    # Point count and span alone cannot certify an unseen hole under the object.
    half=np.maximum((np.asarray(current['hi'])[:2]-np.asarray(current['lo'])[:2])/2+.01,.03)
    if not np.isfinite(half).all() or np.any(half>.12):return None
    others=[(np.asarray(s['center'])[:2],(np.asarray(s['hi'])[:2]-np.asarray(s['lo'])[:2])/2)
            for s in surfaces.values() if s]
    # Historical rejected targets are search exclusions, never current free-space
    # or global reachability claims. Every alternative still needs fresh RGB-D.
    exclusions=[]
    for region in excluded_regions:
        c=np.asarray(region['center'][:2],float);h=np.asarray(region['footprint_half_extent_xy'],float)
        if c.shape!=(2,) or h.shape!=(2,) or not np.isfinite(np.r_[c,h]).all() or np.any(h<=0):
            raise ValueError('Invalid historical table search exclusion')
        exclusions.append((c,h,region.get('id')))
    for radius in (.14,.19,.24):
        for angle in (np.pi,0,np.pi/2,-np.pi/2,np.pi*.75,-np.pi*.75):
            xy=anchor+radius*np.array([np.cos(angle),np.sin(angle)])
            if any(np.all(np.abs(xy-c)<=h) for c,h,_ in exclusions):continue
            if any(np.all(np.abs(xy-c)<half+h+.01) for c,h in others):continue
            near=p[np.all(np.abs(p[:,:2]-xy)<=half,axis=1)]
            plane=near[np.abs(near[:,2]-table_z)<.004]
            obstacles=near[near[:,2]>table_z+.012]
            if len(plane)<60 or len(obstacles)>5:continue
            _,coverage,center_visible=_footprint_grid(plane,xy,half)
            if coverage<.9 or not center_visible:continue
            return dict(frame='robot_base',center=[*xy,float(table_z)],top_z=float(table_z),
                        lo=[*(xy-half),float(table_z)],hi=[*(xy+half),float(table_z)],
                        clearance_verified=True,visible_plane_points=len(plane),
                        footprint_half_extent_xy=half.tolist(),footprint_grid_shape=[6,6],
                        footprint_coverage_fraction=coverage,center_coverage_verified=center_visible,
                        source='RGBD_visible_clear_table_patch',
                        excluded_intent_ids=[i for _,_,i in exclusions if i is not None],
                        limitation='visible_footprint_only_unobserved_volume_not_certified')
    return None
