"""Explicit Robosuite/OpenGL raster-to-integer-index camera calibration.

Robosuite camera_utils.get_camera_intrinsic_matrix uses (width/2,height/2).
OpenGL pixel centres use half-integer window coordinates. The legacy opt-in
conversion subtracts 0.5; strict mode additionally uses the independently measured
depth-resolve sample offset. RGB sample averaging is recorded separately. This
conversion is opt-in at the G0/G1 parent observation boundary, never inferred from
object positions, fitted to a Seed, or applied to the legacy model/API branch.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np

SOURCE_URL='https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html'
CONVENTION='integer_image_indices_from_opengl_half_integer_pixel_centres'


class DepthCalibrationUnavailable(RuntimeError):
    """Public renderer metadata cannot authorize a geometric measurement."""


def _depth_profile(profiles, name):
    try:
        profile=copy.deepcopy(profiles[name])
        digest=profile.pop('profile_sha256')
        if hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest:
            raise ValueError('profile changed')
        profile['profile_sha256']=digest
        if profile['schema']!='capx.public_depth_sampling.v1' or profile['status']!='PASS':raise ValueError('unknown profile')
        offset=np.asarray(profile['depth_offset_from_pixel_center_xy'],float)
        if offset.shape!=(2,) or not np.isfinite(offset).all() or np.any(np.abs(offset)>.5):raise ValueError('invalid offset')
        if profile['sample_count']==0:
            if profile['resolve_rule']!='single_sample_pixel_center' or np.any(offset!=0):raise ValueError('invalid single sample')
        else:
            positions=np.asarray(profile['sample_positions_bottom_origin_xy'],float)
            index=profile['selected_sample']
            if profile['resolve_rule']!='measured_fixed_depth_sample' or positions.shape!=(profile['sample_count'],2) or not isinstance(index,int):raise ValueError('invalid measured sample')
            expected=np.array([positions[index,0]-.5,.5-positions[index,1]])
            if not np.allclose(offset,expected,rtol=0,atol=1e-10):raise ValueError('inconsistent measured offset')
        return offset,profile
    except (KeyError,TypeError,ValueError,IndexError,OverflowError):
        raise DepthCalibrationUnavailable('Public renderer depth sampling calibration unavailable') from None


def robosuite_pixel_centers(observation, *, depth_sampling=None, require_depth_sampling=False):
    """Return (independent observation dict, JSON calibration metadata).

    RGB-D arrays are copied, never resampled. Only a known Robosuite projection
    with the principal point at width/2,height/2 is accepted. Double conversion
    and custom real-camera intrinsics are rejected rather than silently shifted.
    """
    result=copy.deepcopy(observation);metadata={}
    for name,camera in result.items():
        if not isinstance(camera,dict) or 'images' not in camera:continue
        if camera.get('pixel_center_convention'):raise ValueError('Camera pixel convention already declared')
        depth=np.asarray(camera['images']['depth']).squeeze();k=np.asarray(camera['intrinsics'],float)
        if depth.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid calibrated RGB-D camera')
        height,width=depth.shape
        if not np.allclose([k[0,2],k[1,2]],[width/2,height/2],atol=1e-8,rtol=0):raise ValueError('Not the explicitly supported Robosuite OpenGL principal-point convention')
        offset=np.zeros(2);profile=None
        if require_depth_sampling or depth_sampling is not None:
            offset,profile=_depth_profile(depth_sampling,name)
            descriptor=profile.get('descriptor',{})
            if descriptor and (descriptor.get('width',width)<width or descriptor.get('height',height)<height):
                raise DepthCalibrationUnavailable('Public renderer depth sampling calibration unavailable')
        delta=-.5-offset
        corrected=k.copy();corrected[0,2]+=delta[0];corrected[1,2]+=delta[1]
        camera['intrinsics']=corrected;camera['pixel_center_convention']=CONVENTION
        metadata[name]={'schema':'capx.public_camera_calibration.v2' if profile else 'capx.public_camera_calibration.v1','input_intrinsics':k.tolist(),'output_intrinsics':corrected.tolist(),
            'image_shape':[height,width],'input_coordinate_convention':'opengl_window_pixel_centres_at_index_plus_0.5',
            'output_coordinate_convention':CONVENTION,'principal_point_delta_pixels':delta.tolist(),
            'source_url':SOURCE_URL,'local_sources':['robosuite/utils/camera_utils.py:get_camera_intrinsic_matrix','robosuite/utils/binding_utils.py:MjRenderContext.render/read_pixels'],
            'rgbd_resampled':False,'fitted_to_scene':False,'scope':'G0/G1 monitor public observation copy only'}
        if profile:
            metadata[name].update(depth_sampling=profile,rgb_principal_point_delta_pixels=[-.5,-.5],
                depth_coordinate_convention='integer_image_indices_from_measured_renderer_depth_sample',
                rgb_coordinate_convention=CONVENTION,
                rgb_depth_identity_limit='RGB sample averaging and selected depth sample can disagree at silhouettes; masks remain pixel labels, not independent depth identity')
            camera['pixel_center_convention']='integer_image_indices_from_measured_renderer_depth_sample'
            metadata[name]['output_coordinate_convention']=camera['pixel_center_convention']
    return result,metadata
