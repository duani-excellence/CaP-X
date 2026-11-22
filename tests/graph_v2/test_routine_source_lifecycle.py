"""V8 public read sequence: provenance is not a reusable boundary permit.

Uses real monitor grasp/lift/attachment and table checks with synthetic public
point clouds. No physics truth, LLM, or online vision services are involved.
"""
import copy
import numpy as np
import pytest
from capx.execution.public_geometry import validate_table_region
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def held_table(mode):
    m,state,surfaces,calls,base_raw=fixture(mode,task='restack')
    grid=np.linspace(.2,.85,180)
    xx,yy=np.meshgrid(grid,np.linspace(-.25,.25,140))
    cloud={'points':np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]}
    m.geometry_fn=lambda obs:(surfaces,0.,cloud['points'])
    # The real adapter republishes unchanged calibration on each capture, even
    # after G0 conservatively invalidates all work at a true interruption.
    calibration={'test_camera':{'public_intrinsics_version':1}}
    def observe():
        m.register_camera_calibration(calibration)
        return state
    m.observe=observe
    def raw(name,args,kw):
        value=base_raw(name,args,kw)
        if name=='goto_pose' and m.closed:
            surface=surfaces['green']
            surface['top_xy_bounds']=[surface['lo'][:2].copy(),surface['hi'][:2].copy()]
        return value
    m.invoke('observe_object',['green cube'],{},raw)
    grasp=m.invoke('observe_grasp',['green cube'],{},raw)
    approach=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{},raw)
    m.invoke('run_plan',[approach],{},raw)
    m.invoke('close_gripper',[],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    lift=m.invoke('plan_skill',['lift',{'robot':robot['id']},{'dz':.05}],{},raw)
    m.invoke('run_plan',[lift],{},raw)
    assert m.phase=='holding_verified' and m.attachment is not None
    evaluations=[]
    def evidence(obs,region,name,held,**kw):
        result=validate_table_region(cloud['points'],0.,region)
        evaluations.append({'validation':copy.deepcopy(result),'observation':kw['observation_id']})
        return {'validation':result,'evidence':{
            'current_observation_id':kw['observation_id'],
            'geometry_source_ids':[m.geometry_ids['green cube']]}}
    m.table_evidence_fn=evidence
    return m,state,surfaces,calls,raw,cloud,evaluations


def reobserved_region(m,raw):
    region=m.invoke('observe_table',['green cube'],{},raw)
    return m.invoke('reobserve_table',[region['id']],{'change_view':False},raw)


def plan_table(m,raw,region):
    held=m.invoke('observe_surfaces',['green cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['table_place',{
        'held_surface':held['id'],'table_region':region['id'],'robot':robot['id']}],{},raw)
    return held,robot,plan


@pytest.mark.parametrize('mode',['G0','G1'])
def test_v8_held_robot_table_robot_sequence_preserves_current_siblings(mode):
    m,s,v,c,raw,cloud,checks=held_table(mode)
    held=m.invoke('observe_surfaces',['green cube'],{},raw)
    attachment_check=held['attachment_check_id']
    assert m.graph.nodes[attachment_check]['data']['check']=='attachment_consistency'
    first_robot=m.invoke('read_robot',[],{},raw)
    oldplan=m.invoke('plan_skill',['retreat',{'robot':first_robot['id']}],{},raw)
    region=m.invoke('observe_table',['green cube'],{},raw)
    before=len(c)
    second_robot=m.invoke('read_robot',[],{},raw)
    assert m.graph.nodes[held['id']]['valid']
    assert m.graph.nodes[attachment_check]['valid']
    assert m.graph.nodes[region['id']]['valid']
    assert not m.graph.nodes[first_robot['id']]['valid']
    assert not m.graph.nodes[oldplan]['valid']
    plan=m.invoke('plan_skill',['table_place',{
        'held_surface':held['id'],'table_region':region['id'],'robot':second_robot['id']}],{},raw)
    assert m.graph.nodes[plan]['valid'] and len(c)==before
    consumed=[n for n in m.graph.nodes.values() if n['kind']=='plan' and n['data'].get('execution')=='consumed']
    assert len(consumed)==2  # Real approach/lift ancestry, not invented graph parents.


@pytest.mark.parametrize('mode',['G0','G1'])
def test_reobserved_table_permission_is_historical_but_data_sources_stay_linked(mode):
    m,s,v,c,raw,cloud,checks=held_table(mode)
    previous_context=m.placement
    region=reobserved_region(m,raw)
    gate=region['historical_clearance_check_id']
    assert region['clearance_permission']=='fresh_boundary_check_required'
    assert m.graph.nodes[gate]['data']['check']=='table_clearance'
    assert m.graph.nodes[gate]['data']['status']=='PASS'
    direct={e['source'] for e in m.graph.edges if e['target']==region['id']}
    assert gate not in direct
    assert region['current_observation_id'] in direct
    assert region['reobservation_intent_id'] in direct
    calibration=m.observation_calibration_sources[0]
    assert calibration in m.graph.ancestors([region['id']])
    held,robot,plan=plan_table(m,raw,region)
    assert m.graph.nodes[m.placement]['valid'] and m.graph.nodes[plan]['valid']
    if mode=='G0':
        assert m.placement!=previous_context and not m.graph.nodes[previous_context]['valid']
    newer=m.invoke('read_robot',[],{},raw)
    assert m.graph.nodes[region['id']]['valid'] and m.graph.nodes[held['id']]['valid']
    assert not m.graph.nodes[plan]['valid']
    assert m.graph.nodes[newer['id']]['valid']


@pytest.mark.parametrize('mode',['G0','G1'])
def test_reobserved_table_executes_with_fresh_gate_per_motion_without_replay(mode):
    m,s,v,c,raw,cloud,checks=held_table(mode)
    region=reobserved_region(m,raw)
    held,robot,plan=plan_table(m,raw,region)
    old_retreat=m.invoke('plan_skill',['retreat',{'robot':robot['id']}],{},raw)
    prior={n['id'] for n in m.graph.nodes.values() if n['kind']=='call' and n['state']=='done'}
    boundaries=[];before=len(c)
    def record_boundary(name,args,kw):
        if name=='goto_pose':
            gates=[n for n in m.graph.nodes.values() if n['kind']=='check' and n['data'].get('check')=='table_clearance']
            assert gates[-1]['valid'] and gates[-1]['data']['status']=='PASS'
            boundaries.append(gates[-1]['id'])
        return raw(name,args,kw)
    result=m.invoke('run_plan',[plan],{},record_boundary)
    assert result['status']=='returned' and len(boundaries)==3
    assert len(set(boundaries))==len(boundaries)
    assert all(name=='goto_pose' for name,_ in c[before:])
    assert m.graph.nodes[region['id']]['valid']
    assert not m.graph.nodes[held['id']]['valid'] and not m.graph.nodes[robot['id']]['valid']
    assert not m.graph.nodes[old_retreat]['valid']
    assert all(m.graph.nodes[i]['state']=='done' for i in prior)
    assert sum(e['action']=='returned_call' and e.get('node_id') in prior for e in m.graph.events)==len(prior)
    assert m.last_place['skill']=='table_place'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_obstacle_after_first_stage_blocks_the_next_motion(mode):
    m,s,v,c,raw,cloud,checks=held_table(mode)
    region=reobserved_region(m,raw)
    _,_,plan=plan_table(m,raw,region)
    before=len(c);moved=[]
    def insert_obstacle(name,args,kw):
        value=raw(name,args,kw)
        if name=='goto_pose':
            moved.append(list(args[0]))
            xy=np.asarray(region['center'])[:2]
            points=np.c_[xy[0]+np.linspace(-.004,.004,10),np.full(10,xy[1]),np.full(10,.025)]
            cloud['points']=np.vstack([cloud['points'],points])
        return value
    with pytest.raises(ExecutionInterrupted,match='table_clearance_current_nonheld_obstacle'):
        m.invoke('run_plan',[plan],{},insert_obstacle)
    assert len(moved)==1 and len(c)==before+1
    assert not m.graph.nodes[region['id']]['valid']
    assert m.last_place is None and m.phase=='holding_verified'
    assert checks[-1]['validation']['status']=='FAIL'


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('source',['camera','table_version'])
def test_real_source_changes_invalidate_refreshed_region_plans(mode,source):
    m,s,v,c,raw,cloud,checks=held_table(mode)
    region=reobserved_region(m,raw);_,_,plan=plan_table(m,raw,region)
    before=len(c)
    if source=='camera':
        m.register_camera_calibration({'test_camera':{'public_intrinsics_version':2}})
    else:
        newer=m.invoke('observe_table',['green cube'],{},raw)
        assert newer['id']!=region['id']
    assert not m.graph.nodes[region['id']]['valid'] and not m.graph.nodes[plan]['valid']
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},raw)
    assert len(c)==before


def test_g0_real_interruption_still_discards_all_unexecuted_suffix():
    m,s,v,c,raw,cloud,checks=held_table('G0')
    region=reobserved_region(m,raw);held,robot,plan=plan_table(m,raw,region)
    prior={n['id'] for n in m.graph.nodes.values() if n['kind']=='call' and n['state']=='done'}
    before=len(c)
    m.end_block_failure('runtime_error','ValueError')
    assert not m.graph.nodes[plan]['valid']
    assert not m.graph.nodes[region['id']]['valid'] and not m.graph.nodes[held['id']]['valid']
    assert all(m.graph.nodes[i]['state']=='done' for i in prior)
    assert len(c)==before
