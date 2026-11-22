import numpy as np
import pytest
from capx.execution.grip_offset import _two_faces
from test_attachment_visibility import cloud

@pytest.mark.parametrize('yaw',[.17,.37,.79,1.28,1.49])
@pytest.mark.parametrize('noise',[0.,.0001])
def test_output_axes_follow_measured_planes_not_search_strip(yaw,noise):
 center=np.array([.5,.1,.12]);p=cloud()-center
 r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
 p=p@r.T+center
 if noise:p+=np.random.default_rng(17).normal(0,noise,p.shape)
 fit=_two_faces(p,.04,.14)
 assert fit is not None
 angle_error=abs((fit['yaw_rad']-yaw+np.pi/4)%(np.pi/2)-np.pi/4)
 assert angle_error<np.deg2rad(.5)
 assert np.linalg.norm(np.asarray(fit['center'])-center)<.001
 assert fit['orientation_source']=='paired_current_PCA_plane_normals'
