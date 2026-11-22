import copy,json
import numpy as np
import pytest
from capx.execution.public_camera import robosuite_pixel_centers
from capx.execution.public_geometry import point_cloud
from capx.execution.support_geometry import color_points

def obs():
 return {'camera':{'intrinsics':np.array([[100.,0,2.],[0,100.,1.],[0,0,1.]]),'pose_mat':np.eye(4),'images':{'rgb':np.tile(np.array([200,0,0],np.uint8),(2,4,1)),'depth':np.ones((2,4))}},'robot_joint_pos':np.zeros(8)}

def test_half_pixel_conversion_has_one_consistent_projection_and_no_input_mutation():
 raw=obs();before=copy.deepcopy(raw);current,meta=robosuite_pixel_centers(raw)
 assert np.allclose(current['camera']['intrinsics'][:2,2],[1.5,.5])
 assert np.array_equal(raw['camera']['intrinsics'],before['camera']['intrinsics'])
 assert np.array_equal(raw['camera']['images']['rgb'],current['camera']['images']['rgb'])
 assert np.allclose(point_cloud(current),color_points(current,'red cube'))
 assert np.allclose(point_cloud(current)[0],[-.015,-.005,1.])
 assert json.loads(json.dumps(meta))==meta and meta['camera']['fitted_to_scene'] is False
 assert meta['camera']['principal_point_delta_pixels']==[-.5,-.5]

def test_double_conversion_rejected():
 current,_=robosuite_pixel_centers(obs())
 with pytest.raises(ValueError,match='already declared'):robosuite_pixel_centers(current)

def test_custom_real_camera_intrinsics_not_guessed_or_shifted():
 raw=obs();raw['camera']['intrinsics'][0,2]=1.7
 with pytest.raises(ValueError,match='explicitly supported'):robosuite_pixel_centers(raw)
