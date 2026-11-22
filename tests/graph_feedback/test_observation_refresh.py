"""G0 may discard plans conservatively without deadlocking fresh input collection."""
import numpy as np
import pytest
from capx.execution.graph_feedback import GraphFeedbackMonitor
from capx.execution.sandbox import ExecutionInterrupted


def setup(mode='G0',fault='none'):
    robot={'robot_cartesian_pos':np.array([.5,0,.15,1,0,0,0,.5]),
           'robot_joint_pos':np.zeros(8)}
    calls=[]
    def execute(name,args,kwargs):
        calls.append(name)
        if name=='get_object_pose':
            return np.array([.5,0,.02]),np.array([1,0,0,0]),np.array([.04,.04,.04])
        if name=='sample_grasp_pose':
            return np.array([.5,0,.02]),np.array([1,0,0,0])
        if name=='goto_pose':robot['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
    monitor=GraphFeedbackMonitor(mode,lambda:robot,fault=fault)
    monitor.evidence_fn=lambda _:{}
    monitor.begin_program('test_observation_cycle()',0)
    return monitor,execute,calls


def observe(m,raw,name):return m.invoke('observe_object',[name],{},raw)

def place(m,raw,red,green):
    return m.invoke('plan_motion',['place',[.5,0,.06],[1,0,0,0],[red['id'],green['id']]],{},raw)


@pytest.mark.parametrize('order',[
    ['green cube','red cube','green cube'],
    ['red cube','green cube','red cube'],
])
@pytest.mark.parametrize('mode',['G0','G1'])
def test_non_symmetric_fresh_observations_can_form_new_place_plan(mode,order):
    m,raw,calls=setup(mode)
    observations={}
    for object_name in order:observations[object_name]=observe(m,raw,object_name)
    target=place(m,raw,observations['red cube'],observations['green cube'])
    assert m.graph.nodes[target]['valid']
    assert all(m.graph.nodes[value['id']]['valid'] for value in observations.values())
    assert calls==['get_object_pose']*3


def test_g0_recovery_can_recollect_overlapping_observations_across_rounds():
    m,raw,calls=setup()
    for block in range(3):
        if block:
            m.begin_program('new_actor_recovery()',block)
        green=observe(m,raw,'green cube')
        red=observe(m,raw,'red cube')
        green=observe(m,raw,'green cube')
        plan=place(m,raw,red,green)
        assert m.graph.nodes[plan]['valid']
        # An actual conservative abort discards the suffix; the next actor must
        # still be able to collect a new coherent set of observations.
        with pytest.raises(ExecutionInterrupted):
            m.invoke('goto_pose',[[.6,0,.1],[1,0,0,0]],{},raw)
        assert not m.graph.nodes[red['id']]['valid']
        assert not m.graph.nodes[green['id']]['valid']
    assert calls==['get_object_pose']*9


@pytest.mark.parametrize('mode',['G0','G1'])
def test_g0_normal_refresh_discards_all_plans_but_preserves_unrelated_inputs(mode):
    m,raw,calls=setup(mode)
    green=observe(m,raw,'green cube')
    red=observe(m,raw,'red cube')
    robot=m.invoke('read_robot',[],{},raw)
    independent=m.invoke('plan_motion',['lift',[.5,0,.3],[1,0,0,0],[robot['id']]],{},raw)
    check=m.graph.add('check',{'status':'PASS'},parents=[independent])
    refreshed=observe(m,raw,'green cube')
    assert not m.graph.nodes[green['id']]['valid']
    assert m.graph.nodes[refreshed['id']]['valid']
    assert m.graph.nodes[red['id']]['valid']
    assert m.graph.nodes[robot['id']]['valid']
    assert m.graph.nodes[independent]['valid']==(mode=='G1')
    assert m.graph.nodes[check]['valid']==(mode=='G1')


@pytest.mark.parametrize('mode',['G0','G1'])
def test_support_fault_remains_full_abort_for_g0_selective_for_g1(mode):
    m,raw,calls=setup(mode,'support_refresh')
    grasp=m.invoke('observe_grasp',['red cube'],{},raw)
    red=observe(m,raw,'red cube');green=observe(m,raw,'green cube')
    p=place(m,raw,red,green)
    m.closed=m.lifted=True
    completed=list(m.call_history)
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[p],{},raw)
    assert not m.graph.nodes[p]['valid']
    assert m.graph.nodes[grasp['id']]['valid']==(mode=='G1')
    assert m.graph.nodes[red['id']]['valid']==(mode=='G1')
    assert 'goto_pose' not in calls
    assert all(m.graph.nodes[c]['state']=='done' for c in completed)
    assert calls.count('sample_grasp_pose')==1


@pytest.mark.parametrize('mode',['G0','G1'])
def test_actual_local_alignment_still_discards_g0_inputs_without_replay(mode):
    m,raw,calls=setup(mode)
    grasp=m.invoke('observe_grasp',['red cube'],{},raw)
    red=observe(m,raw,'red cube');green=observe(m,raw,'green cube')
    p=place(m,raw,red,green)
    t=m.invoke('plan_motion',['retreat',[.5,0,.25],[1,0,0,0],[p]],{},raw)
    completed=list(m.call_history)
    m.recovery('align',m._tracked(raw),'goto_pose',[[.51,0,.06],[1,0,0,0]],{})
    assert not m.graph.nodes[t]['valid']
    assert m.graph.nodes[grasp['id']]['valid']==(mode=='G1')
    assert m.graph.nodes[red['id']]['valid']==(mode=='G1')
    assert all(m.graph.nodes[c]['state']=='done' for c in completed)
    assert calls.count('goto_pose')==1 and calls.count('sample_grasp_pose')==1
