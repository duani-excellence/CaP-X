import numpy as np
from capx.execution.grip_offset import calibrate, predict, check_surface


def geometry():return dict(height_m=.04,orientation_verified=True)
def surface(x=.5,y=.1,z=.02):return dict(center=[x,y,z],lo=[x-.02,y-.02,z-.02],hi=[x+.02,y+.02,z+.02])


def test_offset_uses_pregrasp_center_and_actual_close_tcp():
    before=surface();tcp=[.498,.102,.019,0,0,1,0]
    state=calibrate(before,geometry(),tcp,source_id='observation-before',tcp_id='tcp-close')
    p=predict(state,[.698,.202,.119,0,0,1,0])
    np.testing.assert_allclose(p['center'],[.7,.2,.12])
    assert state['sources']==['observation-before','tcp-close']


def test_offset_rotates_with_tcp_not_world_axes():
    state=calibrate(surface(.52,0,.02),geometry(),[.5,0,.02,1,0,0,0],source_id='p',tcp_id='q')
    p=predict(state,[.5,0,.02,2**-.5,0,0,2**-.5])
    np.testing.assert_allclose(p['center'],[.5,.02,.02],atol=1e-10)


def test_missing_or_strongly_disagreeing_surface_never_passes():
    state=calibrate(surface(),geometry(),[.5,.1,.02,0,0,1,0],source_id='p',tcp_id='q')
    p=predict(state,[.5,.1,.12,0,0,1,0])
    assert check_surface(p,None)['status']=='UNCERTAIN'
    full=surface(.55,.1,.12)
    full.update(top_xy_bounds=[full['lo'][:2],full['hi'][:2]],top_pixels=400)
    assert check_surface(p,full)['status']=='FAIL'


def test_partial_top_center_bias_does_not_become_new_rigid_offset():
    state=calibrate(surface(),geometry(),[.5,.1,.02,0,0,1,0],source_id='p',tcp_id='q')
    p=predict(state,[.5,.1,.12,0,0,1,0])
    measured=surface(.508,.097,.12)
    checked=check_surface(p,measured)
    # A summary lacking complete-top evidence must no longer authorize motion.
    # The old unconditional PASS for this partial centroid was the audited bug.
    assert checked['status']=='UNCERTAIN'
    np.testing.assert_allclose(p['center'],[.5,.1,.12])
    assert checked['limitation']


def test_tilt_invalidates_upright_height_instead_of_reusing_it():
    state=calibrate(surface(),geometry(),[.5,.1,.02,1,0,0,0],source_id='p',tcp_id='q')
    p=predict(state,[.5,.1,.02,np.cos(.3),np.sin(.3),0,0])
    assert not p['orientation_verified']


def test_actor_cannot_use_missing_sources_or_nonfinite_geometry():
    assert calibrate(None,geometry(),[.5,.1,.02,0,0,1,0],source_id='p',tcp_id='q') is None
    assert calibrate(surface(),None,[.5,.1,.02,0,0,1,0],source_id='p',tcp_id='q') is None
