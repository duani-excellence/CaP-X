"""Offline table decisions need independent labels, not the online mask/reward."""
import json
import numpy as np
import pytest
from capx.experiments.graph_v2_analysis import label_check, analyze_episode
from test_offline_analysis import geometry, state, event, fixture_episode


def nodes(center=(.2, 0., 0.), half=(.04,.04)):
    return {'region':{'kind':'estimate','valid':False,'data':{
        'type':'table_region','frame':'robot_base','center':list(center),
        'footprint_half_extent_xy':list(half)}}}


def private(red=(.4,0,.02),green=(.2,0,.14)):
    return {'state':state(red=red,green=green,green_contact=True),
            'held_object':'green cube','phase':'holding_verified'}


def checkpoint(status='PASS'):
    return event('table_clearance',status,metrics={'region_id':'region',
        'validation':{'status':'PASS','obstacle_points':0,'clearance_verified':True},
        'evidence':{'all_known_nonheld_visible_outside':True}})


@pytest.mark.parametrize('status,expected',[('PASS','true_allow'),('FAIL','false_block'),('UNCERTAIN','false_block')])
def test_clear_table_uses_physical_scene_and_preserves_uncertainty(status,expected):
    out=label_check(checkpoint(status),private(),geometry(),graph_nodes=nodes())
    assert out['classification']==expected
    assert out['label_scope']=='table_footprint_geometric_risk_proxy'
    assert out['uncertain']==(status=='UNCERTAIN')
    assert out['held_object_excluded']=='green'


@pytest.mark.parametrize('status,expected',[('PASS','false_allow'),('FAIL','true_block'),('UNCERTAIN','true_block')])
def test_nonheld_cube_in_region_overrides_online_clear_claim(status,expected):
    out=label_check(checkpoint(status),private(red=(.2,0,.02)),geometry(),graph_nodes=nodes())
    assert out['classification']==expected and out['overlapping_objects']==['red']


def test_table_boundary_corners_not_only_center():
    out=label_check(checkpoint(),private(),geometry(),graph_nodes=nodes(center=(.49,0,0)))
    assert out['classification']=='false_allow' and not out['inside_table']


def test_world_base_translation_and_rotation_are_applied():
    g=geometry();g['base_world']={'position':[.1,0,0],'rotation':[[0,-1,0],[1,0,0],[0,0,1]]}
    out=label_check(checkpoint(),private(red=(.1,.2,.02)),g,graph_nodes=nodes())
    assert out['classification']=='false_allow' and out['overlapping_objects']==['red']


@pytest.mark.parametrize('change',[lambda g,n:n.clear(),lambda g,n:g.update(table_top_world_z=None),lambda g,n:n['region']['data'].update(frame='camera'),lambda g,n:n['region']['data'].update(footprint_half_extent_xy=[float('nan'),.04])])
def test_unknown_region_or_geometry_never_certified(change):
    g=geometry();n=nodes();change(g,n)
    assert label_check(checkpoint(),private(),g,graph_nodes=n)['classification']=='unknown'


def test_analyze_uses_retained_graph_id_even_if_later_invalidated(tmp_path):
    path=fixture_episode(tmp_path);result=json.loads(path.read_text());rows=[
        {'kind':'feedback',**checkpoint('FAIL')},
        {'kind':'feedback_evaluation','event_index':0,**private()}]
    trace=tmp_path/'trace.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'graph-final.json').write_text(json.dumps({'nodes':nodes()}))
    out=analyze_episode(path)
    assert out['detection_by_type']['table']['false_block']==1
    assert out['detection_coverage']['independently_labeled_checkpoints']==1
    assert out['task_success'] is False


def test_table_proxy_does_not_certify_estimated_height():
    out=label_check(checkpoint(),private(),geometry(),graph_nodes=nodes(center=(.2,0,.2)))
    assert out['classification']=='true_allow'
    assert out['estimated_plane_error_m']==pytest.approx(.2)
    assert 'height' in out['limitation']


def test_elevated_nonheld_cube_is_conservative_vertical_footprint_risk():
    out=label_check(checkpoint(),private(red=(.2,0,1.)),geometry(),graph_nodes=nodes())
    assert out['classification']=='false_allow'
    assert out['overlapping_objects']==['red']
    assert 'vertical' in out['limitation']
