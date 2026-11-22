"""A newly acquired frame must follow invalidation of old physical evidence."""
import numpy as np
import pytest
from test_routine_source_lifecycle import held_table,reobserved_region,plan_table


@pytest.mark.parametrize('mode',['G0','G1'])
@pytest.mark.parametrize('entry',['capture','read_robot'])
def test_table_release_current_frame_and_same_actor_retreat_survive(mode,entry):
    m,state,surfaces,calls,raw,cloud,checks=held_table(mode)
    region=reobserved_region(m,raw)
    held,robot,place=plan_table(m,raw,region)
    old_retreat=m.invoke('plan_skill',['retreat',{'robot':robot['id']}],{},raw)
    m.invoke('run_plan',[place],{},raw)
    m.invoke('open_gripper',[],{},raw)
    assert m.phase=='released' and not m.graph.nodes[old_retreat]['valid']
    prior=set(m.call_history);before=len(calls);block=m.block
    if entry=='capture':
        m.capture()
        assert m.graph.nodes[m.last_observation]['valid']
        fresh=m.invoke('read_robot',[],{},raw)
    else:
        fresh=m.invoke('read_robot',[],{},raw)
    assert m.graph.nodes[m.last_observation]['valid']
    assert m.graph.nodes[fresh['id']]['valid']
    assert all(m.graph.nodes[i]['valid'] for i in m.observation_calibration_sources)
    assert not m.graph.nodes[old_retreat]['valid']
    retreat=m.invoke('plan_skill',['retreat',{'robot':fresh['id']}],{},raw)
    m.invoke('run_plan',[retreat],{},raw)
    assert m.phase=='ready' and m.retreat_completed and m.block==block
    assert [n for n,a in calls[before:]]==['goto_pose']
    assert all(m.graph.nodes[i]['state']=='done' for i in prior)
    assert sum(e['action']=='returned_call' and e.get('node_id') in prior for e in m.graph.events)==len(prior)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_changed_public_support_expires_old_plan_but_registers_valid_current_frame(mode):
    m,state,surfaces,calls,raw,cloud,checks=held_table(mode)
    # The other (red) object is static support while green is held.
    source=m.invoke('observe_surfaces',['red cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    held=m.invoke('observe_surfaces',['green cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['place',{'robot':robot['id'],
        'held_surface':held['id'],'support_surface':source['id']}],{},raw)
    old_frame=m.last_observation;old_cal=list(m.observation_calibration_sources)
    before=len(calls);old_epoch=m.object_epochs['red cube']
    for key in ('center','lo','hi'):surfaces['red'][key][0]+=.012
    surfaces['red']['top_xy_bounds']=[surfaces['red']['lo'][:2].copy(),surfaces['red']['hi'][:2].copy()]
    m.capture()
    assert not m.graph.nodes[plan]['valid'] and not m.graph.nodes[source['id']]['valid']
    assert m.object_epochs['red cube']==old_epoch+1
    assert m.graph.nodes[m.last_observation]['valid'] and m.last_observation!=old_frame
    assert all(m.graph.nodes[i]['valid'] for i in m.observation_calibration_sources)
    if mode=='G0':
        assert not m.graph.nodes[old_frame]['valid']
        assert all(not m.graph.nodes[i]['valid'] for i in old_cal)
    updated=m.invoke('observe_surfaces',['red cube'],{},raw)
    assert m.graph.nodes[updated['id']]['valid']
    np.testing.assert_allclose(updated['center'][:2],surfaces['red']['center'][:2])
    assert len(calls)==before
