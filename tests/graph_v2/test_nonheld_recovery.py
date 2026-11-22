import copy
import pytest
from test_grounded_monitor import fixture
from test_table_boundary import ready
from capx.execution.sandbox import ExecutionInterrupted

@pytest.mark.parametrize('mode',['G0','G1'])
def test_nonheld_geometry_can_be_acquired_while_holding(mode):
    m,s,v,c,f=fixture(mode=mode,task='restack')
    m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='green cube'
    m.invoke('observe_object',['red cube'],{},f)
    assert 'red cube' in m.geometry
    node=m.graph.nodes[m.geometry_ids['red cube']]
    assert node['data']['type']=='public_geometry_calibration'
    assert m.last_observation in m.graph.ancestors([node['id']])
    assert m.phase=='holding_verified' and m.held_object=='green cube'
    assert [a[0] for a in c]==['get_object_pose']

@pytest.mark.parametrize('mode',['G0','G1'])
def test_held_geometry_is_not_recalibrated_from_occluded_current_surface(mode):
    m,s,v,c,f=fixture(mode=mode,task='restack')
    m.invoke('observe_object',['green cube'],{},f)
    prior=copy.deepcopy(m.geometry['green cube']);source=m.geometry_ids['green cube']
    m.closed=True;m.held_object='green cube';m.phase='holding_verified'
    v['green']['hi'][2]+=.10
    m.invoke('observe_object',['green cube'],{},f)
    assert m.geometry['green cube']==prior and m.geometry_ids['green cube']==source


def test_missing_nonheld_source_requests_observation_not_motion():
    m,s,c,f,p,t=ready()
    m.table_evidence_fn=lambda *a,**kw:{
        'validation':{'status':'UNCERTAIN','reason':'current_table_region_occluded_or_insufficiently_visible','clearance_verified':False},
        'evidence':{'nonheld_current_evidence':[{'object':'red cube','source_id':None,'current_localization':'unavailable','outside_region_with_margin':False}]}}
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},f)
    gate=next(e for e in reversed(m.events) if e['check']=='dependency_guard')
    assert gate['metrics']['missing_geometry_objects']==['red cube']
    assert gate['metrics']['next_action']=='observe_missing_nonheld_objects_then_reobserve_table_without_motion'
    assert gate['metrics']['recoverable'] is True
    assert c==[] and m.closed and not m.graph.nodes[t['id']]['valid']


def test_missing_geometry_hint_is_visible_to_both_recovery_policies():
    for mode in ('G0','G1'):
        m,s,v,c,f=fixture(mode=mode,task='restack')
        with pytest.raises(ExecutionInterrupted):
            m._stop('table_clearance_current_table_region_occluded_or_insufficiently_visible',metrics={
                'recoverable':True,'missing_geometry_objects':['red cube'],
                'next_action':'observe_missing_nonheld_objects_then_reobserve_table_without_motion',
                'private_unapproved_field':'must not be forwarded'})
        hint=m.summary()['recovery_preconditions']
        assert hint['missing_geometry_objects']==['red cube'] and hint['recoverable']
        assert 'private_unapproved_field' not in hint


def test_real_obstacle_is_not_reclassified_as_missing_observation():
    m,s,c,f,p,t=ready()
    m.table_evidence_fn=lambda *a,**kw:{
        'validation':{'status':'FAIL','reason':'current_nonheld_obstacle_in_table_region','clearance_verified':False},
        'evidence':{'nonheld_current_evidence':[{'object':'red cube','source_id':None}]}}
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},f)
    gate=next(e for e in reversed(m.events) if e['check']=='dependency_guard')
    assert not gate['metrics'].get('missing_geometry_objects')
    assert c==[] and m.closed

@pytest.mark.parametrize('mode',['G0','G1'])
def test_new_actor_can_fill_missing_source_and_revalidate_same_intent(monkeypatch,mode):
    from test_table_reobservation import ready as held_fixture
    m,s,v,c,f,old,plan,completed,checks=held_fixture(monkeypatch,mode)
    def evidence(*args,**kw):
        geometry=kw['nonheld_geometry'];source=geometry.get('red cube',{}).get('source_id')
        return {'validation':{'status':'PASS' if source else 'UNCERTAIN',
            'reason':'current_visible_clear_table_patch' if source else 'current_table_region_occluded_or_insufficiently_visible',
            'clearance_verified':bool(source),'visible_plane_points':100,'footprint_coverage_fraction':1.,'center_coverage_verified':True},
            'evidence':{'geometry_source_ids':[source] if source else [],
                'nonheld_current_evidence':[{'object':'red cube','source_id':source}]}}
    m.table_evidence_fn=evidence
    with pytest.raises(ExecutionInterrupted):m.invoke('reobserve_table',[old['id']],{'change_view':False},f)
    assert m.summary()['recovery_preconditions']['missing_geometry_objects']==['red cube']
    m.begin_program('acquire missing nonheld evidence',1)
    m.invoke('observe_object',['red cube'],{},f)
    result=m.invoke('reobserve_table',[old['id']],{'change_view':False},f)
    assert m.graph.nodes[result['id']]['valid']
    assert not m.graph.nodes[old['id']]['valid'] and not m.graph.nodes[plan]['valid']
    assert m.graph.nodes[completed]['state']=='done'
    assert [x[0] for x in c]==['get_object_pose']
    assert m.phase=='holding_verified' and m.visibility_probes==0
    gate=result['historical_clearance_check_id']
    assert m.geometry_ids['red cube'] in m.graph.ancestors([gate])
