"""Current PUBLIC table-clearance evidence for the static cube task family.

The parent must supply current phase/attachment identity and historical public
size provenance. Each call uses ONE current frame, a current instrumented SAM3
instance request, public encoders/FK, and current nonheld geometric observations.
No scene truth, stale mask, arbitrary bbox erasure or independent-sensor claim.
"""
from __future__ import annotations
import numpy as np
from .public_geometry import capture_geometry,validate_table_region
from .support_geometry import current_support,reliable_top


def _colour(rgb,name):
    r,g,b=np.asarray(rgb,float).transpose(2,0,1)
    if name=='red cube':return (r>50)&(r>1.5*g)&(r>1.5*b)
    if name=='green cube':return (g>45)&(g>1.4*r)&(g>1.3*b)
    raise ValueError('Only declared red/green static cube family is supported')


def _current_cube_shape(surface):
    if not isinstance(surface,dict):return False
    if surface.get('source')=='current_two_visible_vertical_faces_and_top_with_prior_public_cube_side':
        return len(surface.get('face_point_counts',[]))==2
    if surface.get('source')=='current_complete_vertical_face_and_observed_held_bottom':
        proof=surface.get('fit_evidence',{})
        return (proof.get('current_vertical_faces')==1 and proof.get('camera_resolved_outward_normal') is True
            and proof.get('observed_bottom_edge') is True
            and proof.get('complete_current_localization_under_assumptions') is True
            and proof.get('bottom_reference_source')=='current_top_minus_prior_public_side_with_observed_bottom_pixels')
    return False


def current_table_evidence(public_obs,region,held_name,held_prediction,robot_geometry,segment_fn,*,
                           observation_id,held_identity_verified,nonheld_geometry,
                           static_scene_assumption=False,artifact_callback=None,
                           reserve_semantic_refinement=None):
    """One full-frame estimate, optionally one budgeted semantic crop refresh.

    The crop contributes a NEW SAM3 estimate from the same current image. It is
    not an independent sensor, shape dilation, or a wider geometric tolerance.
    All mapped candidates pass the unchanged identity, current shape, first-ray
    and table checks. A parent callback must reserve the shared recovery budget.
    """
    inputs=dict(observation_id=observation_id,held_identity_verified=held_identity_verified,
                nonheld_geometry=nonheld_geometry,static_scene_assumption=static_scene_assumption)
    result=_single_table_evidence(public_obs,region,held_name,held_prediction,robot_geometry,segment_fn,**inputs)
    validation=result['validation'];semantic=result['masks']['held_semantic']
    eligible=(validation.get('status')=='UNCERTAIN'
        and validation.get('reason')=='unresolved_nonheld_elevated_points'
        and isinstance(held_prediction,dict)
        and _current_cube_shape(held_prediction)
        and not result['evidence'].get('held_evidence_failure') and semantic.any())
    if eligible and reserve_semantic_refinement is not None:
        camera=next(c for c in public_obs.values() if isinstance(c,dict) and 'images' in c)
        rgb=np.asarray(camera['images']['rgb']);yy,xx=np.where(semantic)
        pad=max(int(np.ptp(xx)+1),int(np.ptp(yy)+1))
        x0=max(0,int(xx.min())-pad);x1=min(rgb.shape[1],int(xx.max())+1+pad)
        y0=max(0,int(yy.min())-pad);y1=min(rgb.shape[0],int(yy.max())+1+pad)
        crop=rgb[y0:y1,x0:x1].copy()
        detail={'attempted':False,'independent_sensor':False,'current_observation_id':observation_id,
                'initial_validation':dict(validation),'crop_xyxy':[x0,y0,x1,y1],
                'method':'one_semantic_extent_context_crop_no_resize','new_llm_requests':0,
                'limitation':'same-image semantic re-estimation; no independent geometric or contact proof'}
        if reserve_semantic_refinement():
            detail['attempted']=True;original=result;mapped=[]
            try:
                proposals=segment_fn(crop,text_prompt=held_name)
                if isinstance(proposals,(list,tuple)):
                    for candidate in proposals:
                        if not isinstance(candidate,dict):continue
                        mask=np.asarray(candidate.get('mask'))
                        if mask.dtype.kind!='b' or mask.shape!=crop.shape[:2]:continue
                        full=np.zeros(rgb.shape[:2],bool);full[y0:y1,x0:x1]=mask
                        mapped.append({**candidate,'mask':full})
            except Exception as exc:detail['error_type']=type(exc).__name__
            if mapped:
                refined=_single_table_evidence(public_obs,region,held_name,held_prediction,robot_geometry,
                    lambda *args,**kwargs:mapped,**inputs)
                detail['refined_validation']=dict(refined['validation'])
                if not refined['evidence'].get('held_evidence_failure'):
                    result=refined
                    result['masks']['original_held']=original['masks']['held']
                    result['masks']['original_held_semantic']=original['masks']['held_semantic']
            detail['mask_replaced']=result is not original
            # The mapped candidate evaluation is local; the two actual callable
            # invocations are the original full image and the single crop.
            result['evidence']['sam3_requests']=original['evidence']['sam3_requests']+1
        else:detail['reason']='shared_local_recovery_budget_exhausted'
        result['evidence']['semantic_refinement']=detail
    result['evidence']['mask_artifact']=artifact_callback(result['masks']) if artifact_callback else None
    return result


def _single_table_evidence(public_obs,region,held_name,held_prediction,robot_geometry,segment_fn,*,
                           observation_id,held_identity_verified,nonheld_geometry,
                           static_scene_assumption=False,artifact_callback=None):
    """Return current filtered/held points, context, validation and aligned masks.

    `segment_fn` MUST be the existing parent-instrumented SAM3 callable so request
    telemetry is preserved. `artifact_callback(arrays)` may persist returned
    Boolean image masks in an external NPZ and return its serializable reference.
    Only the parent supplies identity/geometry; this is not an actor-facing API.
    """
    if not isinstance(observation_id,str) or not observation_id:raise ValueError('Current public observation ID is required')
    if held_name not in (None,'red cube','green cube'):raise ValueError('Unknown held identity')
    cameras=[c for c in public_obs.values() if isinstance(c,dict) and 'images' in c]
    if not cameras:raise ValueError('Current calibrated RGB-D observation is required')
    cam=cameras[0];rgb=np.asarray(cam['images']['rgb']);depth=np.asarray(cam['images']['depth'],float).squeeze();k=np.asarray(cam['intrinsics'],float);t=np.asarray(cam['pose_mat'],float)
    if rgb.shape!=(*depth.shape,3) or depth.ndim!=2 or k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all():raise ValueError('Invalid current RGB-D calibration')
    yy,xx=np.indices(depth.shape);p=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],axis=-1)@t[:3,:3].T+t[:3,3]
    valid=(depth>0)&np.isfinite(p).all(axis=-1);points=p[valid]
    surfaces,table,captured=capture_geometry(public_obs)
    if len(points)!=len(captured) or not np.allclose(points,captured,equal_nan=True):raise ValueError('RGB-D point-to-pixel mapping mismatch')
    colours={n:_colour(rgb,n)&valid for n in ('red cube','green cube')};protected=colours['red cube']|colours['green cube'];held_mask=np.zeros(depth.shape,bool)
    evidence={'current_observation_id':observation_id,'frame':'robot_base','new_llm_requests':0,'sam3_requests':0,
              'segmentation_source':'provided_instrumented_sam3_callable','independent_sensor':False,
              'sam3_requests_scope':'instrumented callable invocations; transport attempts are owned by existing client telemetry','held_mask_candidates':[],
              'static_scene_assumption':bool(static_scene_assumption),'geometry_source_ids':[],
              'limitation':'same-camera conditional geometric/identity evidence; unexplained points are retained, not independent contact or free-space proof'}
    need_held=bool(held_identity_verified and held_name)
    pred_valid=False
    if need_held and isinstance(held_prediction,dict):
        center=np.asarray(held_prediction.get('center',[]),float)
        height=float(held_prediction.get('height_m',float('nan')))
        pred_valid=(center.shape==(3,) and np.isfinite(center).all() and np.isfinite(height) and .015<=height<=.09 and held_prediction.get('orientation_verified') is True)
    semantic_mask=held_mask.copy()
    segmentation_ok=not need_held
    if need_held and pred_valid:
        evidence['sam3_requests']=1
        try:proposals=segment_fn(rgb,text_prompt=held_name)
        except Exception as exc:
            proposals=[];evidence['segmentation_error_type']=type(exc).__name__
        if not isinstance(proposals,(list,tuple)):proposals=[]
        selected=[];own=colours[held_name];other=colours['red cube' if held_name=='green cube' else 'green cube']
        for index,candidate in enumerate(proposals):
            try:mask=np.asarray(candidate['mask'],bool);score=float(candidate['score'])
            except (KeyError,TypeError,ValueError):continue
            row={'index':index,'score':score if np.isfinite(score) else None}
            evidence['held_mask_candidates'].append(row)
            if mask.shape!=depth.shape or not np.isfinite(score) or score<.25:
                row['reason']='invalid_shape_or_low_score';continue
            mask=mask&valid&~other
            row.update(points=int(mask.sum()),named_colour_coverage=float(np.count_nonzero(own&mask)/max(1,np.count_nonzero(own))))
            if np.count_nonzero(mask)<30 or row['named_colour_coverage']<.5:
                row['reason']='insufficient_named_object_evidence';continue
            sample=p[mask]
            # Public upright-cube circumsphere in XY admits yaw without assuming
            # an unobserved yaw. This ONLY bounds a current semantic mask; it is
            # never an arbitrary volume deletion or a stand-alone held detector.
            xy_ok=np.linalg.norm(sample[:,:2]-center[:2],axis=1)<=height/np.sqrt(2)+.004
            z_ok=np.abs(sample[:,2]-center[2])<=height/2+.004
            compatible=xy_ok&z_ok;fraction=float(np.mean(compatible))
            row.update(compatible_fraction=fraction,xy_compatible_fraction=float(np.mean(xy_ok)),z_compatible_fraction=float(np.mean(z_ok)))
            if fraction<.9:
                row['reason']='current_mask_geometry_inconsistent';continue
            row['reason']='compatible_candidate'
            actual=np.zeros(depth.shape,bool);ids=np.flatnonzero(mask);actual.flat[ids[compatible]]=True
            selected.append((score,float(np.mean(mask[own])) if own.any() else 0.,-index,actual,fraction))
        if selected:
            chosen=max(selected,key=lambda x:x[:3]);held_mask=chosen[3];semantic_mask=held_mask.copy();segmentation_ok=True
            held_mask,boundary=_complete_held_boundary(p,held_mask,other,held_prediction,t[:3,3],camera_intrinsics=k,camera_rotation=t[:3,:3])
            evidence.update(held_mask_score=chosen[0],held_mask_geometry_compatible_fraction=chosen[4],held_mask_pixels=int(held_mask.sum()),held_mask_boundary_refinement=boundary)
    if need_held and not segmentation_ok:evidence['held_evidence_failure']='current_instance_mask_or_rigid_prediction_unavailable'
    held_points=p[held_mask];protected|=held_mask
    q=np.asarray(public_obs.get('robot_joint_pos',[]),float);tcp=np.asarray(public_obs.get('robot_cartesian_pos',[]),float)
    fingers=public_obs.get('robot_gripper_joint_pos')
    if fingers is not None:
        fingers=np.asarray(fingers,float)
        if fingers.shape!=(2,) or not np.isfinite(fingers).all():fingers=None
        else:fingers=np.abs(fingers).tolist()  # Robosuite second slide coordinate is negative.
    robot_mask=np.zeros(depth.shape,bool);robot_evidence={'status':'UNCERTAIN','reason':'current_public_robot_state_unavailable'}
    if q.ndim==1 and len(q)>=7 and tcp.ndim==1 and len(tcp)>=7 and table is not None:
        # Only possible obstacles in the region need rays; table samples and
        # every point elsewhere remain untouched and retain original indexing.
        xy=np.asarray(region['center'][:2],float);half=np.asarray(region['footprint_half_extent_xy'],float)
        candidate=np.all(np.abs(points[:,:2]-xy)<=half,axis=1)&(points[:,2]>table+.004)
        ids=np.flatnonzero(candidate)
        result=robot_geometry.classify(points[ids],t[:3,3],q[:7],finger_positions_m=fingers,observed_tcp=tcp,protected_mask=protected[valid][ids])
        robot_evidence=result['evidence'];selected=np.asarray(result['robot_mask'])
        if selected.shape!=(len(ids),) or selected.dtype.kind!='b':raise ValueError('Robot mask must align with queried public points')
        robot_flat=np.zeros(len(points),bool);robot_flat[ids]=selected&~protected[valid][ids]
        robot_mask[valid]=robot_flat
    evidence['robot_self_segmentation']=robot_evidence
    filtered=points[~robot_mask[valid]]
    nonheld=['red cube' if held_name=='green cube' else 'green cube'] if held_name else ['red cube','green cube']
    outside=True;nonheld_details=[];geometry_sources=[]
    for name in nonheld:
        geometry=nonheld_geometry.get(name) if isinstance(nonheld_geometry,dict) else None
        source_id=geometry.get('source_id') if isinstance(geometry,dict) else None
        surface=surfaces.get(name.split()[0]);method='current_complete_top'
        if not source_id or not geometry or table is None:surface=None
        elif not reliable_top(surface,geometry):
            surface=current_support(p[colours[name]],table,geometry,camera_origin=t[:3,3]);method='current_visible_faces_with_public_side'
        excluded=False
        if surface:
            lo=np.asarray(surface['lo'][:2],float)-.01;hi=np.asarray(surface['hi'][:2],float)+.01
            xy=np.asarray(region['center'][:2],float);half=np.asarray(region['footprint_half_extent_xy'],float)
            excluded=bool(np.any((lo>xy+half)|(hi<xy-half)))
        outside &= excluded
        nonheld_details.append({'object':name,'current_localization':method if surface else 'unavailable','outside_region_with_margin':excluded,'margin_m':.01,'source_id':source_id,'surface':surface})
        if source_id:geometry_sources.append(source_id)
    context={'camera_origin_base':t[:3,3].tolist(),'current_observation_id':observation_id,
             'held_identity_verified':bool(need_held and segmentation_ok and pred_valid),
             'all_known_nonheld_visible_outside':bool(outside),'static_scene_assumption':bool(static_scene_assumption),
             'nonheld_geometry_source_ids':geometry_sources}
    evidence['nonheld_current_evidence']=nonheld_details;evidence['geometry_source_ids']=geometry_sources
    evidence['held_points_removed']=len(held_points);evidence['robot_points_removed']=int(robot_mask.sum());evidence['nonheld_color_points_removed']=0
    validation=validate_table_region(filtered,table,region,held_points=held_points,context=context)
    if need_held and not segmentation_ok:
        validation={**validation,'status':'UNCERTAIN','reason':'current_held_mask_or_prediction_unavailable','clearance_verified':False}
    masks={'held':held_mask,'held_semantic':semantic_mask,'robot':robot_mask,'valid_depth':valid,'protected_named_color':colours['red cube']|colours['green cube']}
    evidence['mask_artifact']=artifact_callback(masks) if artifact_callback else None
    return {'filtered_points':filtered,'held_points':held_points,'context':context,'validation':validation,'evidence':evidence,'masks':masks}


def _complete_held_boundary(points_image,mask,other_mask,surface,camera_origin,*,camera_intrinsics=None,camera_rotation=None):
    """One-pixel semantic boundary completion with CURRENT fitted face geometry.

    A SAM3 mask can omit antialiased object-edge pixels. No arbitrary neighbour
    or bounding volume is erased: only its immediate 8-connected image boundary
    whose DEPTH matches the first ray hit on the current fitted cube (two faces, or a complete face with observed bottom and camera side).
    Prior rigid attachment alone cannot authorize this. Named nonheld colour
    always wins. This remains conditional same-camera identity evidence.
    """
    evidence={'added_pixels':0,'boundary_width_pixels':1,'surface_tolerance_m':.002,
        'independent_sensor':False,'reason':'current_observed_cube_shape_required',
        'limitation':'conditional current semantic boundary and fitted-surface agreement; subpixel thin foreign material at matching depth is unresolved, not independent object identity proof'}
    if not _current_cube_shape(surface):return mask,evidence
    try:angle=float(surface['yaw_rad']);height=float(surface['height_m']);center=np.asarray(surface['center'],float)
    except (KeyError,TypeError,ValueError):return mask,evidence
    if center.shape!=(3,) or not np.isfinite(np.r_[center,angle,height]).all() or not .015<=height<=.09:return mask,evidence
    from scipy.ndimage import binary_dilation
    candidate=binary_dilation(mask,structure=np.ones((3,3),bool))&~mask&~other_mask
    ids=np.flatnonzero(candidate);p=np.asarray(points_image,float).reshape(-1,3)[ids]
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    camera=np.asarray(camera_origin,float);origin=(camera-center)@rotation
    rays=[p-camera]
    if camera_intrinsics is not None and camera_rotation is not None:
        k=np.asarray(camera_intrinsics,float);camera_r=np.asarray(camera_rotation,float)
        if k.shape!=(3,3) or camera_r.shape!=(3,3) or not np.isfinite(k).all() or not np.isfinite(camera_r).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid current camera footprint calibration')
        camera_points=(p-camera)@camera_r
        # A raster sample occupies one pixel cell. Its four corner rays bound
        # subpixel edge coverage; this is an angular bound from K, not a larger
        # 3-D object box or a wider surface-distance threshold.
        for dx,dy in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
            corner=camera_points.copy();corner[:,0]+=dx*corner[:,2]/k[0,0];corner[:,1]+=dy*corner[:,2]/k[1,1]
            rays.append(corner@camera_r.T)
        evidence['ray_footprint_pixels']=.5
    error=np.full(len(p),np.inf)
    for ray in rays:
        direction=ray@rotation
        near=np.full(direction.shape,-np.inf);far=np.full(direction.shape,np.inf);nonzero=np.abs(direction)>1e-12
        np.divide(-height/2-origin,direction,out=near,where=nonzero)
        np.divide(height/2-origin,direction,out=far,where=nonzero)
        enter=np.max(np.minimum(near,far),axis=1);leave=np.min(np.maximum(near,far),axis=1)
        parallel_outside=np.any((~nonzero)&(np.abs(origin)>height/2),axis=1)
        first_surface=camera+enter[:,None]*ray
        distance=np.linalg.norm(first_surface-p,axis=1)
        hit=np.isfinite(p).all(axis=1)&np.isfinite(distance)&(enter>0)&(leave>=enter)&~parallel_outside
        error=np.minimum(error,np.where(hit,distance,np.inf))
    if camera_intrinsics is not None and camera_rotation is not None:
        # A very narrow visible corner can fall between the sampled rays.
        # Test the closest point on each *actual* fitted face continuously:
        # it must project inside this same pixel cell and be the first cube
        # surface along its own ray. No mesh inflation or hidden-face match.
        local=(p-center)@rotation
        for axis in range(3):
            for sign in (-1,1):
                face_local=np.clip(local,-height/2,height/2);face_local[:,axis]=sign*height/2
                face_point=face_local@rotation.T+center;ray=face_point-camera
                direction=ray@rotation;nonzero=np.abs(direction)>1e-12
                near=np.full(direction.shape,-np.inf);far=np.full(direction.shape,np.inf)
                np.divide(-height/2-origin,direction,out=near,where=nonzero);np.divide(height/2-origin,direction,out=far,where=nonzero)
                enter=np.max(np.minimum(near,far),axis=1);leave=np.min(np.maximum(near,far),axis=1)
                visible=(enter>0)&(leave>=enter)&(np.abs(enter-1)*np.linalg.norm(ray,axis=1)<1e-7)
                face_camera=ray@camera_r
                projected=(face_camera[:,:2]/face_camera[:,2,None]-camera_points[:,:2]/camera_points[:,2,None])*np.array([k[0,0],k[1,1]])
                inside_cell=(np.abs(projected)<=.5+1e-9).all(axis=1)&(face_camera[:,2]>0)&(camera_points[:,2]>0)
                distance=np.linalg.norm(face_point-p,axis=1)
                error=np.minimum(error,np.where(visible&inside_cell&np.isfinite(distance),distance,np.inf))
        evidence['footprint_method']='continuous_visible_cube_surface_projection'
    good=error<=.002
    result=mask.copy();result.flat[ids[good]]=True
    evidence.update(reason='one_pixel_boundary_depth_matches_current_cube_first_ray',candidate_pixels=len(ids),added_pixels=int(good.sum()),
        source=surface['source'],first_ray_error_p90_m=float(np.quantile(error[good],.9)) if good.any() else None)
    return result,evidence
