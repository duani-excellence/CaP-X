import numpy as np
import pytest
from capx.execution.public_geometry import choose_table_region
from capx.execution.sandbox import ExecutionInterrupted
from test_table_boundary import ready,checker


def scene():
    x,y=np.meshgrid(np.linspace(.1,.9,160),np.linspace(-.4,.4,160))
    points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    surfaces={'green':{'center':[.5,0,.02],'lo':[.48,-.02,0],'hi':[.52,.02,.04]}}
    return points,surfaces


def test_new_region_excludes_failed_target_but_keeps_current_visibility_checks():
    points,surfaces=scene();first=choose_table_region(points,0,surfaces,'green cube')
    second=choose_table_region(points,0,surfaces,'green cube',excluded_regions=[{'id':'failed-region',**first}])
    assert second is not None and not np.allclose(first['center'],second['center'])
    assert second['clearance_verified'] and second['footprint_coverage_fraction']>=.9
    assert second['excluded_intent_ids']==['failed-region']
    assert choose_table_region(points[:0],0,surfaces,'green cube',excluded_regions=[first]) is None


@pytest.mark.parametrize('mode',['G0','G1'])
def test_pre_motion_table_failure_updates_historical_scope_and_does_not_repeat_motion(mode):
    m,state,calls,raw,plan,table=ready();m.mode=mode;m.graph.mode=mode;m.graph_options={k:mode=='G1' for k in m.graph_options}
    m.table_evidence_fn=checker('PASS')
    m._phase_guard('table_place',True,'fixture_authorization')
    def blocked(*args):
        m.inflight={'id':m.graph.add('call',{'api':'goto_pose'}),'api':'goto_pose','status':'not_executed'}
        raise ExecutionInterrupted('pre_motion_rejected')
    m._precise_place=blocked
    with pytest.raises(ExecutionInterrupted,match='pre_motion_table_target_rejected'):
        m.invoke('run_plan',[plan],{},raw)
    assert not calls and m.phase=='holding_verified' and m.terminal_reason is None
    assert len(m.rejected_table_targets)==1
    old=list(m.rejected_table_targets)
    m.begin_program('print(1)',1)
    assert m.rejected_table_targets==old and not m.graph.nodes[table['id']]['valid']
    hint=m.summary()['recovery_preconditions']
    assert hint['next_action']=='observe_alternative_table_region_then_refresh_surface_robot_and_plan'
    assert hint['rejected_region_id']==table['id']
    # The next actor can obtain an actual fresh public alternative. History
    # may be invalid, but it cannot poison a newly observed region's validity.
    points,_=scene();prior_geometry=m.geometry_fn
    m.geometry_fn=lambda state:(prior_geometry(state)[0],0.,points)
    fresh=m.invoke('observe_table',['green cube'],{},raw)
    assert m.graph.nodes[fresh['id']]['valid']
    assert fresh['excluded_intent_ids']==[table['id']]
    assert not np.allclose(fresh['center'],table['center'])
    assert any(e['source']==old[0]['source_id'] and e['target']==fresh['id'] for e in m.graph.edges)
    assert not calls and m.phase=='holding_verified'


def test_unknown_partial_motion_does_not_create_a_recoverable_target_rejection():
    m,state,calls,raw,plan,table=ready();m.table_evidence_fn=checker('PASS')
    def partial(*args):
        m.inflight={'status':'unknown_partial_effects'}
        raise ExecutionInterrupted('pre_motion_rejected')
    m._precise_place=partial
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},raw)
    assert not m.rejected_table_targets
