"""Historic clear table flags cannot authorize a later placement or release."""
import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def ready():
    m,state,surfaces,calls,raw=fixture(task='restack')
    m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='green cube'
    m.geometry['green cube']={'height_m':.05,'orientation_verified':True}
    m.geometry_ids['green cube']=m.graph.add('verification_fact',{'public_height':.05})
    held={'center':[.5,0,.031],'lo':[.475,-.025,.006],'hi':[.525,.025,.056],
          'height_m':.05,'bottom_z':.006,'orientation_verified':True,'frame':'robot_base'}
    m._held_surface=lambda:held
    h=m._new('surface:green cube',dict(held,type='surface',object='green cube',object_epoch=0))
    table=m._new('table_region',dict(type='table_region',object='green cube',frame='robot_base',
        center=[.5,0,0],lo=[.46,-.04,0],hi=[.54,.04,0],top_z=0.,clearance_verified=True,
        footprint_half_extent_xy=[.04,.04],footprint_coverage_fraction=1.,center_coverage_verified=True,visible_plane_points=100))
    robot=m.invoke('read_robot',[],{},raw)
    p=m.invoke('plan_skill',['table_place',{'robot':robot['id'],'held_surface':h['id'],'table_region':table['id']}],{},raw)
    m.last_place=m.graph.nodes[p]['data'].copy()
    calls.clear()
    return m,state,calls,raw,p,table


def checker(status):
    return lambda *args,**kwargs:{'validation':{'status':status,'reason':'current_test_geometry','clearance_verified':status=='PASS'},'evidence':{'current_observation_id':kwargs['observation_id']}}


def test_new_obstacle_invalidates_old_region_and_blocks_release():
    m,state,calls,raw,p,table=ready();m.table_evidence_fn=checker('FAIL')
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):m.invoke('open_gripper',[],{},raw)
    assert not calls and m.closed
    assert not m.graph.nodes[table['id']]['valid'] and not m.graph.nodes[p]['valid']


def test_unverified_current_clearance_blocks_before_table_place_motion():
    m,state,calls,raw,p,table=ready();m.table_evidence_fn=checker('UNCERTAIN')
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):m.invoke('run_plan',[p],{},raw)
    assert not calls


def test_fresh_clearance_must_be_checked_again_at_release():
    m,state,calls,raw,p,table=ready();observations=[]
    def check(*a,**kw):
        observations.append(kw['observation_id']);return checker('PASS')(*a,**kw)
    m.table_evidence_fn=check
    m._precise_place=lambda *a:None
    m.invoke('run_plan',[p],{},raw)
    m.invoke('open_gripper',[],{},raw)
    assert len(observations)>=2 and observations[0]!=observations[-1]
    assert [x[0] for x in calls]==['open_gripper']


def test_missing_table_evidence_provider_fails_closed():
    m,state,calls,raw,p,table=ready()
    with pytest.raises(ExecutionInterrupted,match='table_clearance'):m.invoke('open_gripper',[],{},raw)
    assert not calls
