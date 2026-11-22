import numpy as np
import pytest
from test_grounded_monitor import fixture
from test_support_geometry import faces,geometry

@pytest.mark.parametrize('mode',['G0','G1'])
def test_release_support_uses_current_camera_side_not_retained_pose(mode):
 m,*_=fixture(mode);pose=np.eye(4);pose[:3,3]=[.5,-1.,.4]
 m.current_public_state={'camera':{'images':{},'pose_mat':pose}}
 m.geometry['green cube']=geometry();m.last_evidence={'green':None};m.table_z=0.;m.colored_points['green cube']=faces(0.,which=('y-',))
 m.support_history['green cube']={'epoch':0,'surface':{'center':[999,999,999]},'source_id':'not_used'}
 current=m._support_surface('green cube',allow_retained=False)
 assert current is not None
 assert np.linalg.norm(np.asarray(current['center'])-[.5,.1,.02])<.001
 assert current['fit_evidence']['camera_resolved_outward_normal'] is True
