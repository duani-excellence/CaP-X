"""Current release/approach evidence must be ancestors of the actual physical call."""
import copy
import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def release_ready(mode='G1'):
    m,state,surfaces,calls,raw=fixture(mode=mode)
    m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    surfaces['red'].update(center=[.65,0,.076],lo=[.63,-.02,.056],hi=[.67,.02,.096],top_xy_bounds=[[.63,-.02],[.67,.02]])
    state['robot_cartesian_pos'][:3]=[.65,0,.076]
    for name,height in [('red cube',.04),('green cube',.05)]:
        m.geometry[name]={'frame':'robot_base','height_m':height,'orientation_verified':True}
        m.geometry_ids[name]=m.graph.add('verification_fact',{'type':'public_size','object':name,'height_m':height})
    m.attachment_id=m.graph.add('verification_fact',{'type':'public_rigid_attachment','object':'red cube'})
    checks=[]
    def held():
        m.record('L0','attachment_consistency','PASS','current_public_fixture',graph_parents=[m.attachment_id,m.geometry_ids['red cube']])
        checks.append(m.last_check)
        return {**copy.deepcopy(surfaces['red']),'frame':'robot_base','height_m':.04,'bottom_z':.056,'orientation_verified':True,'attachment_check_id':m.last_check}
    m._held_surface=held
    support=m.graph.add('estimate',{'type':'surface','object':'green cube',**surfaces['green']})
    m.last_place={'skill':'place','inputs':{'support_surface':support}}
    return m,state,surfaces,calls,raw,checks


@pytest.mark.parametrize('mode',['G0','G1'])
def test_actual_release_includes_this_releases_held_check_and_geometry(mode):
    m,s,v,c,f,checks=release_ready(mode);m.invoke('open_gripper',[],{},f)
    assert [name for name,_ in c]==['open_gripper']
    ancestors=m.graph.ancestors([m.call_history[-1]])
    assert checks[-1] in ancestors
    assert m.geometry_ids['red cube'] in ancestors
    assert m.geometry_ids['green cube'] in ancestors
    pre=next(n['id'] for n in reversed(list(m.graph.nodes.values())) if n['data'].get('check')=='pre_release')
    assert pre in ancestors


def test_current_support_surface_is_explicit_release_evidence():
    m,s,v,c,f,checks=release_ready();m.invoke('open_gripper',[],{},f)
    ancestor=m.graph.ancestors([m.call_history[-1]])
    evidence=[m.graph.nodes[i] for i in ancestor if m.graph.nodes[i]['data'].get('evidence_role')=='pre_release_support']
    assert evidence and evidence[-1]['data']['object']=='green cube'
    assert evidence[-1]['data']['surface']['center']==v['green']['center']
    assert m.geometry_ids['green cube'] in m.graph.ancestors([evidence[-1]['id']])
    assert any(m.graph.nodes[i]['kind']=='observation' for i in m.graph.ancestors([evidence[-1]['id']]))


def test_failed_prerelease_still_never_opens():
    m,s,v,c,f,checks=release_ready();m.max_local=0;v['green']['center'][0]+=.15
    with pytest.raises(ExecutionInterrupted):m.invoke('open_gripper',[],{},f)
    assert not c and m.closed


@pytest.mark.parametrize('mode',['G0','G1'])
def test_current_sweep_check_is_actual_approach_call_ancestor(mode):
    m,s,v,c,f=fixture(mode=mode)
    m.invoke('observe_object',['red cube'],{},f)
    grasp=m.invoke('observe_grasp',['red cube'],{},f)
    plan=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{},f)
    m.invoke('run_plan',[plan],{},f)
    sweep=next(n['id'] for n in reversed(list(m.graph.nodes.values())) if n['data'].get('check')=='gripper_sweep')
    motion=next(i for i in m.call_history if m.graph.nodes[i]['data']['api']=='goto_pose')
    assert sweep in m.graph.ancestors([motion])
