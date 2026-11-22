import numpy as np
import pytest
from capx.execution.public_geometry import object_geometry, place_metrics, choose_table_region


def surf(zlo,zhi,x=.5,y=0):
 return {'center':[x,y,(zlo+zhi)/2],'lo':[x-.02,y-.02,zlo],'hi':[x+.02,y+.02,zhi], 'pixels':100,'top_pixels':40}


def test_pregrasp_geometry_uses_visible_support_plane_not_occluded_close():
 s=surf(.006,.04)
 g=object_geometry('red cube',s,0,{},None)
 assert g['height_m']==pytest.approx(.04)
 assert g['source']=='visible_top_minus_observed_support_plane'
 assert g['frame']=='robot_base'


def test_restack_geometry_uses_visible_lower_support():
 surfaces={'red':surf(.004,.04)}
 g=object_geometry('green cube',surf(.043,.08),0,surfaces,None)
 assert g['height_m']==pytest.approx(.04)


def test_held_bottom_comes_from_earlier_rigid_height():
 held=surf(.13,.16); support=surf(.004,.04)
 d=place_metrics(held,support,{'height_m':.04})
 assert d['gap_m']==pytest.approx(.08)
 assert d['status']=='FAIL'


def test_release_unseen_or_unknown_geometry_is_uncertain():
 assert place_metrics(None,surf(0,.04),None)['status']=='UNCERTAIN'
 assert place_metrics(surf(.04,.08),surf(0,.04),None)['status']=='UNCERTAIN'


def test_clear_table_region_requires_visible_plane_coverage():
 xy=np.array([(x,y) for x in np.linspace(.2,.8,121) for y in np.linspace(-.3,.3,121)])
 points=np.c_[xy,np.zeros(len(xy))]
 region=choose_table_region(points,0,{'green':surf(.04,.08),'red':surf(0,.04)},'green cube')
 assert region['clearance_verified']
 assert np.linalg.norm(np.array(region['center'])[:2]-[.5,0])>.07
 assert region['frame']=='robot_base'
 assert choose_table_region(points[:0],0,{},'green cube') is None


def test_occluded_restack_support_does_not_fall_back_to_table_height():
 # Visible green sides begin too high to establish support contact; .08m is
 # the total stack height, not evidence that green itself is .08m tall.
 assert object_geometry('green cube',surf(.065,.08),0,{'red':surf(.004,.04)},None) is None


def test_elevated_or_truncated_surface_without_support_is_unknown():
 assert object_geometry('green cube',surf(.065,.08),0,{},None) is None


def test_shape_height_records_actual_support_and_uncalibrated_uncertainty():
 g=object_geometry('green cube',surf(.043,.08),0,{'red':surf(.004,.04)},None)
 assert g['support_object']=='red'
 assert g['source']=='visible_top_minus_observed_object_support'
 assert g['orientation_policy']=='upright_only' and g['orientation_verified'] is False
 assert g['uncertainty_kind']=='heuristic_uncalibrated' and g['uncertainty_m'] is None


def test_upright_only_height_requires_monitor_orientation_verification():
 from capx.execution.public_geometry import measured_surface
 g=object_geometry('red cube',surf(.004,.04),0,{},None)
 held=surf(.05,.08);support=surf(.004,.04)
 assert 'bottom_z' not in measured_surface(held,g)
 assert place_metrics(held,support,g)['status']=='UNCERTAIN'
 g['orientation_verified']=True
 assert measured_surface(held,g)['bottom_z']==pytest.approx(.04)
 assert place_metrics(held,support,g)['status']=='PASS'
 g['orientation_verified']=False
 assert 'bottom_z' not in measured_surface(held,g)
 assert place_metrics(held,support,g)['reason']=='upright_height_orientation_unverified'


def test_cached_geometry_cannot_change_object_identity():
 g=object_geometry('red cube',surf(.004,.04),0,{},None)
 assert object_geometry('green cube',surf(.043,.08),0,{},g) is None


def test_clear_table_region_rejects_hollow_ring_and_parallel_stripes():
 a=np.linspace(0,2*np.pi,360,endpoint=False)
 ring=np.c_[.36+.039*np.cos(a),.039*np.sin(a),np.zeros(len(a))]
 surfaces={'green':surf(.04,.08),'red':surf(0,.04)}
 assert choose_table_region(ring,0,surfaces,'green cube') is None
 y=np.linspace(-.04,.04,300)
 stripes=np.r_[np.c_[np.full(len(y),.325),y,np.zeros(len(y))],np.c_[np.full(len(y),.395),y,np.zeros(len(y))]]
 assert choose_table_region(stripes,0,surfaces,'green cube') is None


def test_clear_table_region_reports_footprint_visibility_coverage():
 xy=np.array([(x,y) for x in np.linspace(.2,.8,121) for y in np.linspace(-.3,.3,121)])
 points=np.c_[xy,np.zeros(len(xy))]
 region=choose_table_region(points,0,{'green':surf(.04,.08),'red':surf(0,.04)},'green cube')
 assert region['clearance_verified'] is True
 assert region['footprint_coverage_fraction']>=.9
 assert region['center_coverage_verified'] is True
 assert len(region['footprint_half_extent_xy'])==2


def test_clear_table_region_fails_closed_for_object_larger_than_patch():
 # Current visible object footprint is 14cm wide; a 6cm plane patch is inadequate.
 xy=np.array([(x,y) for x in np.linspace(.33,.39,61) for y in np.linspace(-.03,.03,61)])
 points=np.c_[xy,np.zeros(len(xy))]
 large=surf(.04,.08);large['lo'][:2]=[.43,-.07];large['hi'][:2]=[.57,.07]
 assert choose_table_region(points,0,{'green':large},'green cube') is None
