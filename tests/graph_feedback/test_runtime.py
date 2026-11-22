import numpy as np
import pytest
from capx.execution.graph_feedback import GraphFeedbackMonitor
from capx.execution.sandbox import ExecutionInterrupted

def make(mode='G1',fault='none'):
    state={'robot_cartesian_pos':np.array([.5,0,.1,1,0,0,0,.5]),'robot_joint_pos':np.zeros(8)}
    calls=[]
    def raw(name,args,kw):
        calls.append(name)
        if name=='get_object_pose':return np.array([.5,0,0]),np.array([1,0,0,0]),np.array([.04,.04,.04])
        if name=='goto_pose':state['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
    m=GraphFeedbackMonitor(mode,lambda:state,fault=fault)
    m.begin_program('observe_object("red cube")',0)
    m.evidence_fn=lambda _:{}
    return m,raw,calls,state

def plans(m,raw):
    red=m.invoke('observe_object',['red cube'],{},raw)
    green=m.invoke('observe_object',['green cube'],{},raw)
    place=m.invoke('plan_motion',['place',[.5,0,.05],[1,0,0,0],[red['id'],green['id']]],{},raw)
    retreat=m.invoke('plan_motion',['retreat',[.5,0,.2],[1,0,0,0],[place]],{},raw)
    return red,green,place,retreat

def test_support_update_blocks_old_place_preserves_unrelated():
    m,raw,calls,state=make(fault='support_refresh');red,green,p,t=plans(m,raw)
    m.closed=m.lifted=True
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},raw)
    assert 'goto_pose' not in calls
    assert not m.graph.nodes[p]['valid'] and not m.graph.nodes[t]['valid']
    assert m.graph.nodes[red['id']]['valid']
    assert m.summary()['recovery_scope']['policy']=='affected_dependencies'

def test_conservative_discards_unrelated():
    m,raw,calls,state=make('G0',fault='support_refresh');red,green,p,t=plans(m,raw);m.closed=m.lifted=True
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},raw)
    assert not m.graph.nodes[red['id']]['valid']

def test_local_alignment_invalidates_retreat_and_home_guard():
    m,raw,calls,state=make();red,green,p,t=plans(m,raw)
    m.active_plan=p
    m.recovery('align',m._tracked(raw),'goto_pose',[[.51,0,.05],[1,0,0,0]],{})
    m.active_plan=None;m.released=True
    assert not m.graph.nodes[t]['valid']
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[t],{},raw)
    with pytest.raises(ExecutionInterrupted):m.invoke('home_pose',[],{},raw)
    assert calls.count('goto_pose')==1

def test_revalidated_new_retreat_continues_without_replaying_release():
    m,raw,calls,state=make();m.released=True
    s=m.invoke('read_robot',[],{},raw)
    p=m.invoke('plan_motion',['retreat',[.5,0,.2],[1,0,0,0],[s['id']]],{},raw)
    m.invoke('run_plan',[p],{},raw)
    assert m.retreat_completed
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},raw)
    assert calls==['goto_pose']

def test_unknown_dependencies_fail_closed_and_secret_boundary():
    m,raw,calls,state=make();state['raw_success']=True;state['cube_poses']='SECRET'
    with pytest.raises(ExecutionInterrupted):m.invoke('goto_pose',[[.5,0,.2],[1,0,0,0]],{},raw)
    text=str(m.summary())+str(m.graph.snapshot())
    assert 'SECRET' not in text and 'raw_success' not in text
    assert not calls

@pytest.mark.parametrize('mode',['G0','G1'])
def test_control_interruption_invalidates_plan_suffix(mode):
    m,raw,calls,state=make(mode);red,green,p,t=plans(m,raw);m.closed=m.lifted=True
    # Real L1 monitor retries a target the fake controller cannot reach.
    def stuck(name,args,kw):
        if name=='goto_pose':calls.append(name);return
        return raw(name,args,kw)
    with pytest.raises(ExecutionInterrupted,match='control_not_converged'):
        m.invoke('run_plan',[p],{},stuck)
    assert not m.graph.nodes[p]['valid'] and not m.graph.nodes[t]['valid']
    assert m.graph.nodes[red['id']]['valid']==(mode=='G1')
    assert len([n for n in m.graph.nodes.values() if n['kind']=='call' and n['state']=='done'])==4
    m.begin_program('recovery()',1)
    assert m.graph.nodes[m.placement]['valid']


def test_home_cannot_move_held_object_outside_contract():
    m,raw,calls,state=make();m.closed=True
    with pytest.raises(ExecutionInterrupted,match='holding'):
        m.invoke('home_pose',[],{},raw)
    assert not calls


def test_update_evidence_is_present_in_recovery_context():
    m,raw,calls,state=make(fault='support_refresh');red,green,p,t=plans(m,raw);m.closed=m.lifted=True
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},raw)
    context=m.summary()
    assert any(e['action']=='version' and e.get('key')=='object:green cube' for e in context['dependency_changes'])
    assert any(r['id']==m.graph.latest['object:green cube'] for r in context['available_results'])


def test_stale_robot_sample_cannot_drive_a_new_motion():
    m,raw,calls,state=make();first=m.invoke('read_robot',[],{},raw)
    state['robot_cartesian_pos'][0]=.6
    latest=m.invoke('read_robot',[],{},raw)
    with pytest.raises(ExecutionInterrupted):m.invoke('recall_result',[first['id']],{},raw)
    assert not m.graph.nodes[first['id']]['valid']
    assert m.graph.nodes[latest['id']]['valid']
    assert not calls


def test_alignment_replacement_has_fresh_valid_evidence_in_both_groups():
    for mode in ('G0','G1'):
        m,raw,calls,state=make(mode);red,green,p,t=plans(m,raw)
        m.recovery('align',m._tracked(raw),'goto_pose',[[.51,0,.05],[1,0,0,0]],{})
        assert all(m.graph.nodes[i]['valid'] for i in m.change_roots)
        assert m.last_call in m.graph.ancestors(m.change_roots)

@pytest.mark.parametrize('name',['get_object_pose','sample_grasp_pose'])
def test_legacy_perception_cannot_bypass_version_tracking(name):
    m,raw,calls,state=make()
    with pytest.raises(ExecutionInterrupted,match='versioned'):
        m.invoke(name,['green cube'],{},raw)
    assert not calls


def test_release_guard_cannot_be_bypassed_by_mislabelled_approach():
    m,raw,calls,state=make();m.released=True
    grasp=m.graph.add('estimate',{'type':'grasp'})
    p=m.invoke('plan_motion',['approach',[.8,0,.1],[1,0,0,0],[grasp]],{},raw)
    with pytest.raises(ExecutionInterrupted,match='retreat'):
        m.invoke('run_plan',[p],{},raw)
    assert not calls


def test_g0_fresh_recomputation_in_same_actor_can_use_new_context():
    m,raw,calls,state=make('G0');plans(m,raw)
    red=m.invoke('observe_object',['red cube'],{},raw)
    green=m.invoke('observe_object',['green cube'],{},raw)
    p=m.invoke('plan_motion',['place',[.5,0,.05],[1,0,0,0],[red['id'],green['id']]],{},raw)
    assert m.graph.nodes[p]['valid']


def test_post_block_visual_check_does_not_erase_interruption_diagnosis():
    m,raw,calls,state=make(fault='support_refresh');red,green,p,t=plans(m,raw);m.closed=m.lifted=True
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},raw)
    m.finish_check()  # Runner performs a final observational check after env.step.
    context=m.summary()
    assert p in context['diagnosis']['roots']
    assert p in context['recovery_scope']['recompute']
    assert m.graph.latest['object:green cube'] in context['diagnosis']['roots']
    m.begin_program('next_actor()',1)
    assert not m.interruption_roots


def test_held_red_pose_expires_after_local_alignment_without_losing_support():
    m,raw,calls,state=make();red,green,p,t=plans(m,raw);m.closed=True
    m.recovery('align',m._tracked(raw),'goto_pose',[[.51,0,.05],[1,0,0,0]],{})
    assert not m.graph.nodes[red['id']]['valid']
    assert m.graph.nodes[green['id']]['valid']
    assert red['id'] not in [r['id'] for r in m.summary()['available_results']]
    assert m.graph.nodes[m.placement]['valid']
    assert m.graph.nodes[m.last_call]['state']=='done'
    assert calls.count('goto_pose')==1
