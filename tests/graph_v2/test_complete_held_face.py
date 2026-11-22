import numpy as np
import pytest
from capx.execution.grip_offset import check_surface
from test_attachment_visibility import predicted,geometry,partial
from test_held_mask_boundary import run_case
from test_table_clearance import fixture


def full_face_cloud(width=.04,shift=0.):
    x,y=np.meshgrid(np.linspace(-.02,.02,25),np.linspace(-.02,.02,25))
    top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.02)]
    t,z=np.meshgrid(np.linspace(-width/2,width/2,25),np.linspace(-.02,.02,25))
    face=np.c_[np.full(t.size,-.02),t.ravel(),z.ravel()]
    return np.vstack([top,face])+[.5+shift,.1,.12]


def test_full_top_can_use_existing_complete_face_fitter_with_current_camera():
    p=full_face_cloud();r=check_surface(predicted(),partial(p),points=p,geometry=geometry(),camera_origin=[.1,.1,.4])
    assert r['status']=='PASS' and r['current_surface']['source']=='current_complete_vertical_face_and_observed_held_bottom'
    s=r['current_surface'];assert np.linalg.norm(np.array(s['center'])-[.5,.1,.12])<.003
    assert s['fit_evidence']['current_vertical_faces']==1 and s['fit_evidence']['observed_bottom_edge']
    assert 'table_z' not in s['fit_evidence']
    assert s['fit_evidence']['bottom_reference_source']=='current_top_minus_prior_public_side_with_observed_bottom_pixels'
    assert r['independent_verification'] is False


@pytest.mark.parametrize('case',['no_camera','partial_side','shifted'])
def test_incomplete_or_contradictory_face_cannot_add_current_shape_permission(case):
    p=full_face_cloud(width=.02 if case=='partial_side' else .04,shift=.025 if case=='shifted' else 0)
    r=check_surface(predicted(),partial(p),points=p,geometry=geometry(),camera_origin=None if case=='no_camera' else [.1,.1,.4])
    if case=='shifted':assert r['status']=='FAIL'
    else:assert 'current_surface' not in r


def test_camera_on_hidden_side_cannot_approve_wrong_exterior_face():
    p=full_face_cloud();r=check_surface(predicted(),partial(p),points=p,geometry=geometry(),camera_origin=[.8,.1,.4])
    assert r['status']!='PASS' or 'current_surface' not in r


def test_current_complete_face_can_support_existing_one_pixel_completion():
    o,region,p,g,mask=fixture();mask[56,:]=False
    p.update(source='current_complete_vertical_face_and_observed_held_bottom',yaw_rad=0.,
        fit_evidence={'current_vertical_faces':1,'camera_resolved_outward_normal':True,
            'observed_bottom_edge':True,'complete_current_localization_under_assumptions':True,
            'bottom_reference_source':'current_top_minus_prior_public_side_with_observed_bottom_pixels'})
    r=run_case(o,region,p,g,mask)
    assert r['validation']['status']=='PASS' and r['masks']['held'][56,56:73].all()
    assert r['evidence']['held_mask_boundary_refinement']['surface_tolerance_m']==.002
    p['fit_evidence']['observed_bottom_edge']=False
    r=run_case(o,region,p,g,mask)
    assert not r['masks']['held'][56,56:73].any()
