import copy
import numpy as np
import pytest
from capx.execution.public_geometry import choose_table_region, validate_table_region


def scene():
    xx,yy=np.meshgrid(np.linspace(.2,.8,121),np.linspace(-.3,.3,121))
    points=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    s={'center':[.5,0,.06],'lo':[.48,-.02,.04],'hi':[.52,.02,.08]}
    region=choose_table_region(points,0,{'green':s},'green cube')
    assert region
    return points,region


def patch(region,z=.04):
    center=np.array(region['center']);xx,yy=np.meshgrid(np.linspace(-.009,.009,8),np.linspace(-.009,.009,8))
    return center+np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,z)]


def test_current_full_visible_patch_revalidates_without_trusting_history_flag():
    points,region=scene();region['clearance_verified']=False
    result=validate_table_region(points,0,region)
    assert result['status']=='PASS' and result['clearance_verified'] is True
    assert result['footprint_coverage_fraction']>=.9
    assert result['conditional_static_scene'] is False


def test_previously_clear_region_new_obstacle_fails():
    points,region=scene();result=validate_table_region(np.r_[points,patch(region)],0,region)
    assert result['status']=='FAIL' and result['clearance_verified'] is False
    assert result['obstacle_points']==64


def test_partial_occlusion_does_not_inherit_old_clearance():
    points,region=scene();c=np.array(region['center'])[:2]
    points=points[~np.all(abs(points[:,:2]-c)<.021,axis=1)]
    result=validate_table_region(points,0,region)
    assert result['status']=='UNCERTAIN' and not result['clearance_verified']
    assert not result['center_coverage_verified']


def test_exact_current_held_points_can_be_removed_but_not_nearby_obstacles():
    points,region=scene();held=patch(region)
    result=validate_table_region(np.r_[points,held],0,region,held_points=held+1e-9)
    assert result['status']=='PASS' and result['held_points_removed']==64
    # Nearby but distinct samples must remain even in a nominal held-object box.
    obstacle=held+np.array([1e-5,0,0])
    result=validate_table_region(np.r_[points,held,obstacle],0,region,held_points=held)
    assert result['status']=='FAIL' and result['obstacle_points']==64
    assert result['held_points_removed']==64


def test_duplicate_held_samples_do_not_delete_other_points():
    points,region=scene();held=patch(region)
    result=validate_table_region(np.r_[points,held],0,region,held_points=np.repeat(held,3,axis=0))
    assert result['held_points_removed']==64 and result['status']=='PASS'


def test_removing_held_points_cannot_fill_in_hidden_table():
    points,region=scene();c=np.array(region['center'])[:2]
    visible=points[~np.all(abs(points[:,:2]-c)<.021,axis=1)];held=patch(region)
    result=validate_table_region(np.r_[visible,held],0,region,held_points=held)
    assert result['status']=='UNCERTAIN'
    assert result['held_points_removed']==64


def test_small_unexplained_elevated_cluster_is_uncertain_not_clear():
    points,region=scene();result=validate_table_region(np.r_[points,patch(region)[:3]],0,region)
    assert result['status']=='UNCERTAIN'


@pytest.mark.parametrize('case',['plane_missing','plane_moved','wrong_frame','bad_bounds','sparse_points'])
def test_missing_or_inconsistent_region_evidence_is_uncertain(case):
    points,region=scene();table=0
    if case=='plane_missing':table=None
    if case=='plane_moved':table=.02
    if case=='wrong_frame':region['frame']='world'
    if case=='bad_bounds':region['hi'][0]=region['lo'][0]
    if case=='sparse_points':points=points[:1]
    result=validate_table_region(points,table,region)
    assert result['status']=='UNCERTAIN' and not result['clearance_verified']


def test_does_not_mutate_prior_region_or_point_arrays():
    points,region=scene();before=copy.deepcopy(region);original=points.copy()
    validate_table_region(points,0,region,held_points=patch(region))
    assert region==before
    np.testing.assert_array_equal(points,original)


def occluded_scene():
    _,region=scene();c=np.asarray(region['center']);half=np.asarray(region['footprint_half_extent_xy']);plane=[];held=[]
    for i in range(6):
        for j in range(6):
            for u in (.2,.5,.8):
                for v in (.2,.5,.8):
                    xy=c[:2]-half+2*half*np.array([(i+u)/6,(j+v)/6])
                    if i in (2,3) and j in (2,3):
                        # Current held point ray projects to precisely this missing table sample.
                        held.append([*(c[:2]+.95*(xy-c[:2])),.05])
                    else:plane.append([*xy,0.])
    held=np.array(held);points=np.r_[plane,held]
    context={'camera_origin_base':[*c[:2],1.], 'current_observation_id':'current-obs-9',
             'held_identity_verified':True,'all_known_nonheld_visible_outside':True,
             'static_scene_assumption':True,'nonheld_geometry_source_ids':['current-red-9']}
    return points,region,held,context


def test_current_held_ray_shadow_allows_explicit_conditional_static_scene():
    points,region,held,context=occluded_scene()
    result=validate_table_region(points,0,region,held_points=held,context=context)
    assert result['status']=='PASS' and result['conditional_static_scene'] is True
    assert result['reason']=='current_table_and_held_shadow_under_static_scene_assumption'
    assert result['footprint_coverage_fraction']<.9
    assert result['combined_coverage_fraction']==1.
    assert result['observation_id']=='current-obs-9'
    assert 'not_independent' in result['limitation']


@pytest.mark.parametrize('missing',['held_identity_verified','all_known_nonheld_visible_outside','static_scene_assumption','current_observation_id','camera_origin_base'])
def test_conditional_shadow_requires_each_current_parent_contract(missing):
    points,region,held,context=occluded_scene();context.pop(missing)
    assert validate_table_region(points,0,region,held_points=held,context=context)['status']=='UNCERTAIN'


def test_conditional_shadow_requires_real_prior_full_region_verification():
    points,region,held,context=occluded_scene();region['clearance_verified']=False
    assert validate_table_region(points,0,region,held_points=held,context=context)['status']=='UNCERTAIN'


def test_conditionally_retained_region_still_rejects_current_nonheld_obstacle():
    points,region,held,context=occluded_scene();obstacle=held[:10]+np.array([.00001,0,0])
    assert validate_table_region(np.r_[points,obstacle],0,region,held_points=held,context=context)['status']=='FAIL'


def test_held_points_not_in_current_cloud_cannot_manufacture_shadow():
    points,region,held,context=occluded_scene()
    result=validate_table_region(points[:-len(held)],0,region,held_points=held,context=context)
    assert result['status']=='UNCERTAIN' and result['held_points_removed']==0


def test_shadow_must_explain_missing_region_not_merely_exist_somewhere():
    points,region,held,context=occluded_scene();shift=np.array([.10,0,0]);points[-len(held):]+=shift;held=held+shift
    result=validate_table_region(points,0,region,held_points=held,context=context)
    assert result['status']=='UNCERTAIN'


def test_no_remaining_visible_table_is_not_enough_for_conditional_clearance():
    points,region,held,context=occluded_scene()
    assert validate_table_region(held,0,region,held_points=held,context=context)['status']=='UNCERTAIN'
