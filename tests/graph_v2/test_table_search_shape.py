import copy
import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_table_boundary import ready, checker
from test_table_alternatives import scene


def search_fixture(mode='G1', contaminated=True):
    m,state,calls,raw,plan,table=ready()
    m.mode=mode;m.graph.mode=mode;m.graph_options={k:mode=='G1' for k in m.graph_options}
    points,_=scene();old=m.geometry_fn
    surfaces=copy.deepcopy(old(state)[0])
    if contaminated:
        surfaces['green']['lo'][0]=.1
        surfaces['green']['hi'][0]=.8
    m.geometry_fn=lambda obs:(surfaces,0.,points)
    return m,state,calls,raw,plan,table


@pytest.mark.parametrize('mode',['G0','G1'])
def test_table_search_uses_current_verified_held_shape_not_raw_color_outliers(mode):
    m,state,calls,raw,plan,old=search_fixture(mode)
    region=m.invoke('observe_table',['green cube'],{},raw)
    assert region['clearance_verified'] and max(region['footprint_half_extent_xy'])<.05
    assert not calls and m.phase=='holding_verified'
    source=region['search_shape_source_id']
    fact=m.graph.nodes[source]
    assert fact['kind']=='verification_fact' and fact['data']['historical_only'] is True
    assert fact['data']['permission'] is False and fact['data']['object']=='green cube'
    assert any(e['source']==source and e['target']==region['id'] for e in m.graph.edges)
    assert m.geometry_ids['green cube'] in [e['source'] for e in m.graph.edges if e['target']==source]


@pytest.mark.parametrize('mode',['G0','G1'])
def test_search_shape_never_substitutes_for_current_table_release_check(mode):
    m,state,calls,raw,plan,old=search_fixture(mode)
    region=m.invoke('observe_table',['green cube'],{},raw)
    held=m.invoke('observe_surfaces',['green cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    # Refreshing the current held pose does not erase historical candidate
    # bounds. Actual execution still obtains its own current clearance.
    assert m.graph.nodes[region['id']]['valid']
    fresh=m.invoke('plan_skill',['table_place',{'table_region':region['id'],'held_surface':held['id'],'robot':robot['id']}],{},raw)
    m.table_evidence_fn=checker('FAIL')
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):
        m.invoke('run_plan',[fresh],{},raw)
    assert not calls and m.closed and m.phase=='holding_verified'


def test_missing_current_held_shape_does_not_fall_back_to_raw_or_old_geometry():
    m,state,calls,raw,plan,old=search_fixture(contaminated=False)
    m._held_surface=lambda:None
    with pytest.raises(ExecutionInterrupted,match='current_held_shape_required_for_table_search'):
        m.invoke('observe_table',['green cube'],{},raw)
    assert not calls and m.closed
    hint=m.summary()['recovery_preconditions']
    assert hint['next_action']=='observe_held_surface_before_table_search_or_stop'


def test_current_shape_does_not_certify_unseen_plane_or_erase_obstacle_points():
    m,state,calls,raw,plan,old=search_fixture()
    surfaces=m.geometry_fn(state)[0]
    m.geometry_fn=lambda obs:(surfaces,0.,np.empty((0,3)))
    with pytest.raises(ExecutionInterrupted,match='no_observed_clear_table_region'):
        m.invoke('observe_table',['green cube'],{},raw)
    assert not calls
