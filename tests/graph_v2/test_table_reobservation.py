"""Failed clearance may revalidate a historical intent, never an old permit."""
import copy
import numpy as np
import pytest
import capx.execution.grounded_feedback as module
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def ready(monkeypatch,mode='G1',status='PASS'):
    m,state,surfaces,calls,raw=fixture(mode=mode,task='restack')
    m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='green cube';m.grasp_epoch=1
    m.attachment={'public_only':True};m.attachment_id=m.graph.add('verification_fact',{'type':'public_rigid_grasp_attachment'})
    m.geometry['green cube']={'height_m':.05,'orientation_verified':True};m.geometry_ids['green cube']=m.graph.add('verification_fact',{'type':'public_size','object':'green cube','height_m':.05})
    held={'center':[.5,0,.2],'lo':[.475,-.025,.175],'hi':[.525,.025,.225],'bottom_z':.175,'height_m':.05,'orientation_verified':True,'frame':'robot_base'}
    m._held_surface=lambda:copy.deepcopy(held)
    monkeypatch.setattr(module,'predict_attachment',lambda *args:copy.deepcopy(held))
    region=m._new('table_region',dict(type='table_region',object='green cube',frame='robot_base',center=[.5,0,0],lo=[.46,-.04,0],hi=[.54,.04,0],top_z=0.,clearance_verified=True,footprint_half_extent_xy=[.04,.04],footprint_coverage_fraction=1.,center_coverage_verified=True,visible_plane_points=100))
    plan=m.graph.add('plan',{'skill':'table_place'},parents=[region['id']])
    completed=m.graph.add('call',{'api':'close_gripper'},state='done')
    m.graph.invalidate([region['id']],'failed_current_table_check')
    checks=[]
    def evidence(obs,value,name,prediction,**kw):
        checks.append((copy.deepcopy(value),kw['observation_id']))
        return {'validation':{'status':status,'reason':'synthetic_current_clearance','clearance_verified':status=='PASS','visible_plane_points':111,'footprint_coverage_fraction':1.,'center_coverage_verified':True},'evidence':{'current_observation_id':kw['observation_id'],'geometry_source_ids':[m.geometry_ids['green cube']]}}
    m.table_evidence_fn=evidence
    return m,state,surfaces,calls,raw,region,plan,completed,checks


@pytest.mark.parametrize('mode',['G0','G1'])
def test_reobserve_same_invalid_intent_creates_current_permit_without_replaying(monkeypatch,mode):
    m,state,surfaces,calls,raw,old,plan,done,checks=ready(monkeypatch,mode)
    m.begin_program('new_actor',1)
    result=m.invoke('reobserve_table',[old['id']],{'change_view':False},raw)
    assert result['id']!=old['id'] and result['center']==old['center'] and result['lo']==old['lo']
    assert result['clearance_verified'] and m.graph.nodes[result['id']]['valid']
    assert not m.graph.nodes[old['id']]['valid'] and not m.graph.nodes[plan]['valid']
    assert m.graph.nodes[done]['state']=='done' and calls==[]
    ancestors=m.graph.ancestors([result['id']]);current=checks[-1][1]
    assert current in ancestors
    gate=result['historical_clearance_check_id']
    assert m.graph.nodes[gate]['data']['check']=='table_clearance'
    assert m.graph.nodes[gate]['data']['status']=='PASS'
    assert gate not in {e['source'] for e in m.graph.edges if e['target']==result['id']}
    assert result['clearance_permission']=='fresh_boundary_check_required'
    assert any(m.graph.nodes[i]['data'].get('type')=='table_region_reobservation_intent' and m.graph.nodes[i]['data']['historical_only'] for i in ancestors)
    assert old['id'] in ancestors  # provenance only, immutable intent explicitly breaks permission semantics
    assert m.phase=='holding_verified' and m.last_place is None


@pytest.mark.parametrize('mode',['G0','G1'])
def test_failed_fresh_check_never_mints_valid_region(monkeypatch,mode):
    m,s,v,c,f,old,plan,done,checks=ready(monkeypatch,mode,status='UNCERTAIN')
    count=sum(n['data'].get('type')=='table_region' for n in m.graph.nodes.values())
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):
        m.invoke('reobserve_table',[old['id']],{'change_view':False},f)
    assert len(checks)==1 and c==[] and m.phase=='holding_verified'
    assert sum(n['data'].get('type')=='table_region' for n in m.graph.nodes.values())==count
    assert not m.graph.nodes[plan]['valid'] and m.graph.nodes[done]['state']=='done'
    assert m.summary()['remaining_visibility_probes']==2


@pytest.mark.parametrize('mode',['G0','G1'])
def test_one_request_makes_at_most_one_view_then_new_actor_can_retry(monkeypatch,mode):
    m,s,v,c,f,old,plan,done,checks=ready(monkeypatch,mode)
    oldrobot=m.invoke('read_robot',[],{},f)
    first=m.invoke('reobserve_table',[old['id']],{'change_view':True},f)
    assert [a[0] for a in c]==['goto_pose'] and m.visibility_probes==1 and m.local_attempts==1
    assert not m.graph.nodes[oldrobot['id']]['valid']
    np.testing.assert_allclose(c[0][1][0],[.5,0,.14])
    assert m.phase=='holding_verified'
    m.graph.invalidate([first['id']],'later_clearance_uncertain');m.begin_program('another_actor',2)
    second=m.invoke('reobserve_table',[first['id']],{'change_view':True},f)
    assert second['id']!=first['id'] and len(c)==2 and m.visibility_probes==2
    m.graph.invalidate([second['id']],'later_clearance_uncertain');m.begin_program('budget_actor',3)
    with pytest.raises(ExecutionInterrupted,match='visibility.*budget|visibility.*exhausted'):
        m.invoke('reobserve_table',[second['id']],{'change_view':True},f)
    assert len(c)==2 and m.summary()['remaining_visibility_probes']==0


@pytest.mark.parametrize('limit',['clearance','neighbor','local_budget'])
def test_no_safe_clearance_or_budget_never_executes_view(monkeypatch,limit):
    m,s,v,c,f,old,plan,done,checks=ready(monkeypatch)
    if limit=='clearance':monkeypatch.setattr(module,'predict_attachment',lambda *a:{'center':[.5,0,.03],'bottom_z':.005,'orientation_verified':True})
    elif limit=='neighbor':m.color_points_fn=lambda *a:np.empty((0,3))
    else:m.local_attempts=m.max_episode_local
    with pytest.raises(ExecutionInterrupted):m.invoke('reobserve_table',[old['id']],{'change_view':True},f)
    assert c==[] and checks==[] and m.phase=='holding_verified' and not m.graph.nodes[plan]['valid']


@pytest.mark.parametrize('bad',['unknown','wrong_kind','wrong_identity'])
def test_only_known_parent_region_with_held_identity_is_accepted(monkeypatch,bad):
    m,s,v,c,f,old,plan,done,checks=ready(monkeypatch)
    ref='unknown' if bad=='unknown' else done if bad=='wrong_kind' else old['id']
    if bad=='wrong_identity':m.held_object='red cube'
    with pytest.raises(ExecutionInterrupted):m.invoke('reobserve_table',[ref],{'change_view':True},f)
    assert c==[] and checks==[]


def test_current_grasp_resets_reference_but_not_episode_view_budget(monkeypatch):
    m,s,v,c,f=fixture(task='restack');m.visibility_probes=1;m.local_attempts=2
    m.visibility_reference_q=np.array([1.,0,0,0]);m.phase='approached';m.approached_object='red cube'
    m.invoke('close_gripper',[],{},f)
    assert m.visibility_probes==1 and m.local_attempts==2
    assert m.visibility_reference_q is None


def test_valid_old_region_also_requires_new_verification_and_old_plan_expires(monkeypatch):
    m,s,v,c,f,old,plan,done,checks=ready(monkeypatch)
    fresh=m._new('table_region',{k:v for k,v in old.items() if k!='id'})
    oldplan=m.graph.add('plan',{'skill':'table_place'},parents=[fresh['id']])
    result=m.invoke('reobserve_table',[fresh['id']],{'change_view':False},f)
    assert result['id']!=fresh['id'] and not m.graph.nodes[oldplan]['valid'] and m.graph.nodes[result['id']]['valid']
    assert len(checks)==1


def test_api_is_documented_and_broker_exported():
    assert 'reobserve_table' in module.APIS
    assert 'reobserve_table(region_id, change_view=True)' in module.API_DOC
    assert 'old plans remain invalid' in module.API_DOC
