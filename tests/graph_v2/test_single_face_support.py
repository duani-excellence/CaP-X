import numpy as np
import pytest
from capx.execution.support_geometry import current_support
from test_support_geometry import faces,geometry

@pytest.mark.parametrize('yaw',[0.,.3,.8,1.4])
def test_current_complete_single_face_camera_resolves_normal(yaw):
    points=faces(yaw,which=('y-',));normal=np.array([np.sin(yaw),-np.cos(yaw),0.])
    surface=current_support(points,0.,geometry(),camera_origin=np.array([.5,.1,.02])+normal+[0,0,.4])
    assert surface is not None
    assert np.linalg.norm(np.asarray(surface['center'])-[.5,.1,.02])<.001
    assert surface['fit_evidence']['current_vertical_faces']==1
    assert surface['fit_evidence']['camera_resolved_outward_normal'] is True
    assert surface['fit_evidence']['independent_sensor'] is False

@pytest.mark.parametrize('kind',['half_width','missing_bottom','tilted','airborne','no_camera','grazing_camera'])
def test_single_face_missing_constraints_remains_unknown(kind):
    points=faces(0,which=('y-',));camera=np.array([.5,-1.,.4])
    if kind=='half_width':points=points[points[:,0]<.5]
    if kind=='missing_bottom':points=points[points[:,2]>.012]
    if kind=='tilted':points=faces(0,tilt=.25,which=('y-',))
    if kind=='airborne':points+=np.array([0,0,.02])
    if kind=='no_camera':camera=None
    if kind=='grazing_camera':camera=np.array([1.,.08,.4])
    assert current_support(points,0.,geometry(),camera_origin=camera) is None
