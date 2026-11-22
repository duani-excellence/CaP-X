import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_table_boundary import ready
from test_table_alternatives import scene

def blocked_evidence(status='FAIL',reason='current_nonheld_obstacle_in_table_region'):
    return lambda *a,**kw: {'validation':{'status':status,'reason':reason,'clearance_verified':False},'evidence':{'current_observation_id':kw['observation_id']}}

@pytest.mark.parametrize('mode',['G0','G1'])
def test_unclear_obstacle_region_is_search_history_not_motion_or_free_space_permission(mode):
    m,state,calls,raw,plan,table=ready();m.mode=mode;m.graph.mode=mode;m.graph_options={k:mode=='G1' for k in m.graph_options}
    m.table_evidence_fn=blocked_evidence()
    with pytest.raises(ExecutionInterrupted,match='table_clearance_current_nonheld_obstacle'):
        m.invoke('run_plan',[plan],{},raw)
    assert not calls and m.phase=='holding_verified' and m.closed
    assert m.last_place is None and m.terminal_reason is None
    assert not m.graph.nodes[table['id']]['valid'] and not m.graph.nodes[plan]['valid']
    assert len(m.rejected_table_targets)==1
    h=m.rejected_table_targets[0];fact=m.graph.nodes[h['source_id']]
    assert fact['data']['type']=='unverified_table_target'
    assert fact['data']['physical_obstacle_confirmed'] is False
    check=fact['data']['source_check_id']
    assert any(e['source']==check and e['target']==fact['id'] for e in m.graph.edges)
    m.begin_program('print(1)',1)
    hint=m.summary()['recovery_preconditions']
    assert hint['next_action']=='observe_alternative_table_region_then_refresh_surface_robot_and_plan'
    points,_=scene();geom=m.geometry_fn
    m.geometry_fn=lambda s:(geom(s)[0],0.,points)
    fresh=m.invoke('observe_table',['green cube'],{},raw)
    assert m.graph.nodes[fresh['id']]['valid'] and fresh['clearance_verified']
    assert table['id'] in fresh['excluded_intent_ids']
    assert not np.allclose(fresh['center'],table['center'])
    assert not calls and m.phase=='holding_verified'

def test_failed_reobservation_keeps_one_exclusion_without_replaying_actions():
    m,state,calls,raw,plan,table=ready();m.table_evidence_fn=blocked_evidence();m.attachment={'fixture':True}
    for _ in range(2):
        with pytest.raises(ExecutionInterrupted,match='table_clearance'):
            m.invoke('reobserve_table',[table['id']],{'change_view':False},raw)
    assert len(m.rejected_table_targets)==1 and not calls and m.closed

@pytest.mark.parametrize('reason',['current_table_region_occluded_or_insufficiently_visible','unresolved_nonheld_elevated_points','current_held_or_table_provider_unavailable'])
def test_missing_or_sparse_evidence_does_not_exclude_region_as_obstacle(reason):
    m,state,calls,raw,plan,table=ready();m.table_evidence_fn=blocked_evidence('UNCERTAIN',reason)
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):m.invoke('run_plan',[plan],{},raw)
    assert not m.rejected_table_targets and not calls

def test_blocked_release_does_not_open_jaws_or_leave_old_release_permission():
    m,state,calls,raw,plan,table=ready();m.table_evidence_fn=blocked_evidence()
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):m.invoke('open_gripper',[],{},raw)
    assert m.last_place is None and m.closed and not calls
    assert len(m.rejected_table_targets)==1
