import copy
import json
import numpy as np
import pytest
from capx.execution.public_camera import robosuite_pixel_centers, DepthCalibrationUnavailable
from capx.execution.depth_sampling import resolve_profile
from capx.execution.public_geometry import point_cloud
from capx.execution.support_geometry import color_points
from test_public_camera import obs
from test_depth_sampling import POSITIONS,trials

def profile():return resolve_profile(POSITIONS,trials(),sample_count=4)

def test_depth_sample_shift_is_consistent_across_all_public_geometry():
    raw=obs();before=copy.deepcopy(raw)
    current,meta=robosuite_pixel_centers(raw,depth_sampling={'camera':profile()},require_depth_sampling=True)
    assert np.allclose(current['camera']['intrinsics'][:2,2],[1.625,.125])
    assert np.allclose(point_cloud(current),color_points(current,'red cube'))
    assert np.allclose(point_cloud(current)[0],[-.01625,-.00125,1.])
    assert np.array_equal(raw['camera']['intrinsics'],before['camera']['intrinsics'])
    assert np.array_equal(raw['camera']['images']['rgb'],current['camera']['images']['rgb'])
    m=meta['camera'];assert m['principal_point_delta_pixels']==[-.375,-.875]
    assert m['rgb_principal_point_delta_pixels']==[-.5,-.5]
    assert m['depth_sampling']['profile_sha256']==profile()['profile_sha256']
    assert json.loads(json.dumps(meta))==meta

@pytest.mark.parametrize('profiles',[None,{}, {'camera':{'status':'UNCERTAIN','reason':'no renderer'}}, {'camera':{'status':'PASS','depth_offset_from_pixel_center_xy':[9,9]}}])
def test_strict_unknown_profile_never_falls_back(profiles):
    with pytest.raises(DepthCalibrationUnavailable,match='depth sampling calibration unavailable'):
        robosuite_pixel_centers(obs(),depth_sampling=profiles,require_depth_sampling=True)

def test_known_single_sample_profile_retains_center_math():
    p=resolve_profile([],[],sample_count=0)
    current,_=robosuite_pixel_centers(obs(),depth_sampling={'camera':p},require_depth_sampling=True)
    assert np.allclose(current['camera']['intrinsics'][:2,2],[1.5,.5])

def test_changed_calibration_changes_metadata_without_changing_rgbd():
    p=profile();other=resolve_profile(POSITIONS,trials(selected=1),sample_count=4)
    a,ma=robosuite_pixel_centers(obs(),depth_sampling={'camera':p},require_depth_sampling=True)
    b,mb=robosuite_pixel_centers(obs(),depth_sampling={'camera':other},require_depth_sampling=True)
    assert ma!=mb and not np.array_equal(a['camera']['intrinsics'],b['camera']['intrinsics'])
    assert np.array_equal(a['camera']['images']['depth'],b['camera']['images']['depth'])

def test_tampered_profile_hash_is_not_a_valid_calibration():
    p=profile();p['depth_offset_from_pixel_center_xy']=[.2,.2]
    with pytest.raises(DepthCalibrationUnavailable,match='depth sampling calibration unavailable'):
        robosuite_pixel_centers(obs(),depth_sampling={'camera':p},require_depth_sampling=True)


def test_unknown_calibration_is_infrastructure_not_invalid_actor_program():
    from capx.execution.public_camera import DepthCalibrationUnavailable
    assert issubclass(DepthCalibrationUnavailable, RuntimeError)
    assert not issubclass(DepthCalibrationUnavailable, ValueError)
