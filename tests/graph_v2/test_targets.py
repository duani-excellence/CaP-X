"""Trusted target recipes: numerical provenance, no environment or model calls."""
import copy
import math
import pytest
from capx.execution.grounded_targets import compile_target


def node(node_id, **data):
    return {'id': node_id, 'valid': True, 'data': {'frame': 'robot_base', **data}}


def robot(q=None):
    return node('robot-1', position=[.6, .1, .3], quaternion_wxyz=q or [1, 0, 0, 0], quaternion_order='wxyz')


def grasp():
    return node('grasp-1', position=[.5, -.1, .02], quaternion_wxyz=[0, 0, 1, 0], quaternion_order='wxyz')


def surfaces(bottom=True):
    held = node('held-1', center=[.58, .12, .28], lo=[.56, .10, .27], hi=[.60, .14, .31])
    if bottom: held['data']['bottom_z'] = .25
    support = node('support-1', center=[.7, -.1, .04], lo=[.67, -.13, .01], hi=[.73, -.07, .07])
    return {'robot': robot(), 'held_surface': held, 'support_surface': support}


def used(result, role):
    return next(s for s in result['used_sources'] if s['role'] == role)


def test_approach_binds_exact_grasp_and_wxyz_orientation():
    inputs={'grasp': grasp()}; before=copy.deepcopy(inputs)
    out=compile_target('approach',inputs)
    assert out['position']==[.5,-.1,.02]
    assert out['quaternion_wxyz']==[0,0,1,0]
    assert out['z_approach']==.1
    assert used(out,'grasp')['id']=='grasp-1'
    assert used(out,'grasp')['fields']==['position','quaternion_wxyz']
    assert out['recipe']['frame']=='robot_base' and out['recipe']['quaternion_order']=='wxyz'
    assert inputs==before


@pytest.mark.parametrize('skill,dz', [('lift',.1),('retreat',.08)])
def test_vertical_targets_bind_fresh_tcp_and_preserve_orientation(skill,dz):
    q=[math.sqrt(.5),0,0,math.sqrt(.5)]
    out=compile_target(skill,{'robot':robot(q)},{'dz':dz})
    assert out['position']==pytest.approx([.6,.1,.3+dz])
    assert out['quaternion_wxyz']==pytest.approx(q)
    assert out['z_approach']==0
    assert used(out,'robot')['fields']==['position','quaternion_wxyz']


def test_lift_default_is_positive_and_at_most_15cm():
    out=compile_target('lift',{'robot':robot()})
    assert 0 < out['position'][2]-.3 <= .15


def test_place_uses_measured_bottom_and_nonzero_tcp_object_offset():
    out=compile_target('place',surfaces())
    assert out['position']==pytest.approx([.72,-.12,.126])
    assert out['z_approach']==0
    assert out['quaternion_wxyz']==[1,0,0,0]
    assert 'bottom_z' in used(out,'held_surface')['fields']
    assert 'lo' not in used(out,'held_surface')['fields']
    assert out['recipe']['bottom_evidence']=='measured_bottom'
    assert out['uncertainty']==[]


def test_rotated_tcp_offset_uses_wxyz_rotation_without_frame_swap():
    inputs=surfaces();inputs['robot']=robot([math.sqrt(.5),0,0,math.sqrt(.5)])
    out=compile_target('place',inputs)
    assert out['position']==pytest.approx([.72,-.12,.126])
    assert out['recipe']['held_offset_base']==pytest.approx([-.02,.02,-.05])
    assert out['recipe']['held_offset_tcp']==pytest.approx([.02,.02,-.05])


def test_visible_lower_surface_fallback_is_explicitly_uncertain():
    out=compile_target('place',surfaces(bottom=False))
    assert out['position'][2]==pytest.approx(.106)
    assert 'lo' in used(out,'held_surface')['fields']
    assert out['recipe']['bottom_evidence']=='visible_lower_bound'
    assert out['uncertainty']==['held_bottom_uses_visible_surface_lower_bound']


def test_support_update_changes_target_and_its_source_reference():
    inputs=surfaces();old=compile_target('place',inputs)
    inputs['support_surface']['id']='support-2'
    inputs['support_surface']['data']['center'][0]+=.03
    inputs['support_surface']['data']['hi'][2]+=.02
    new=compile_target('place',inputs)
    assert new['position']==pytest.approx([old['position'][0]+.03,old['position'][1],old['position'][2]+.02])
    assert used(new,'support_surface')['id']=='support-2'
    assert used(new,'support_surface')['value_digest']!=used(old,'support_surface')['value_digest']


def test_table_place_requires_parent_verified_region_and_uses_observed_height():
    inputs=surfaces();inputs.pop('support_surface')
    inputs['table_region']=node('table-1',center=[.4,-.2,-.1],top_z=-.1,clearance_verified=True)
    out=compile_target('table_place',inputs)
    assert out['position']==pytest.approx([.42,-.22,-.044])
    assert 'clearance_verified' in used(out,'table_region')['fields']
    assert 'top_z' in used(out,'table_region')['fields']
    assert out['recipe']['support_role']=='table_region'


@pytest.mark.parametrize('clearance',[False,None,1,'true'])
def test_table_region_without_boolean_clearance_is_rejected(clearance):
    inputs=surfaces();inputs.pop('support_surface')
    inputs['table_region']=node('table-1',center=[.4,-.2,-.1],top_z=-.1,clearance_verified=clearance)
    with pytest.raises(ValueError):compile_target('table_place',inputs)


@pytest.mark.parametrize('skill,dz',[('lift',.049),('lift',.251),('retreat',.079),('retreat',.251),('lift',float('nan')),('lift',True),('lift','0.1')])
def test_vertical_parameters_are_bounded_numbers(skill,dz):
    with pytest.raises(ValueError):compile_target(skill,{'robot':robot()},{'dz':dz})


@pytest.mark.parametrize('params',[{'position':[.1,.2,.3]},{'quaternion_wxyz':[0,0,0,1]},{'target_xyz':[0,0,0]},{'gap':.017},{'gap':-.005},{'z_approach':-.1},{'z_approach':.151}])
def test_actor_geometry_and_unsafe_place_parameters_are_rejected(params):
    with pytest.raises(ValueError):compile_target('place',surfaces(),params)


@pytest.mark.parametrize('mutation',['wrong_frame','wrong_order','invalid','missing_id','zero_quaternion','nan_position','bad_shape','missing_role','extra_role'])
def test_invalid_or_ambiguous_source_data_fails_closed(mutation):
    inputs={'grasp':grasp()};source=inputs['grasp']
    if mutation=='wrong_frame':source['data']['frame']='camera'
    elif mutation=='wrong_order':source['data']['quaternion_order']='xyzw'
    elif mutation=='invalid':source['valid']=False
    elif mutation=='missing_id':source.pop('id')
    elif mutation=='zero_quaternion':source['data']['quaternion_wxyz']=[0,0,0,0]
    elif mutation=='nan_position':source['data']['position'][0]=float('nan')
    elif mutation=='bad_shape':source['data']['position']=[1,2]
    elif mutation=='missing_role':inputs={}
    else:inputs['unrelated']=robot()
    with pytest.raises(ValueError):compile_target('approach',inputs)


def test_changed_actor_copy_cannot_mutate_compiled_source_or_result():
    inputs=surfaces();before=copy.deepcopy(inputs)
    out=compile_target('place',inputs)
    inputs['held_surface']['data']['center'][0]=9
    assert out['position']==pytest.approx([.72,-.12,.126])
    out['position'][0]=8
    assert before['held_surface']['data']['center'][0]==.58


@pytest.mark.parametrize('mutation',['inverted_bounds','bottom_above_top','nonfinite_surface','missing_bottom_and_lo'])
def test_invalid_surface_geometry_fails_closed(mutation):
    inputs=surfaces();h=inputs['held_surface']['data']
    if mutation=='inverted_bounds':h['lo'][0]=h['hi'][0]+1
    elif mutation=='bottom_above_top':h['bottom_z']=h['hi'][2]+1
    elif mutation=='nonfinite_surface':h['center'][0]=float('inf')
    else:h.pop('bottom_z');h.pop('lo')
    with pytest.raises(ValueError):compile_target('place',inputs)


def test_unknown_skill_cannot_fall_back_to_raw_motion():
    with pytest.raises(ValueError):compile_target('raw_goto',{'robot':robot()})
