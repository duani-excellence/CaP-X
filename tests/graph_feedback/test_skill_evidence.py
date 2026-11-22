"""Skill evidence lineage and current permission; no simulator or model calls."""
import numpy as np
import pytest
from capx.execution.dependency_graph import DependencyGraph
from capx.execution.graph_feedback import GraphFeedbackMonitor, GRAPH_APIS
from capx.execution.sandbox import ExecutionInterrupted, SandboxProcess


def build(mode='G1',baseline_visible=True):
    state={'robot_cartesian_pos':np.array([.5,0,.1,1,0,0,0,.5]),'robot_joint_pos':np.zeros(8)}
    visibility={'red':baseline_visible};calls=[]
    def evidence(_):
        z=float(state['robot_cartesian_pos'][2])
        red=dict(center=[.5,0,z],lo=[.48,-.02,z-.02],hi=[.52,.02,z+.02]) if visibility['red'] else None
        green=dict(center=[.5,0,0],lo=[.475,-.025,-.025],hi=[.525,.025,.025])
        return {'red':red,'green':green}
    def raw(name,args,kwargs):
        calls.append(name)
        if name=='goto_pose':
            state['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]];visibility['red']=True
        if name=='get_object_pose':
            z=state['robot_cartesian_pos'][2] if 'red' in args[0] else 0
            return np.array([.5,0,z]),np.array([1,0,0,0]),np.array([.04,.04,.04])
    monitor=GraphFeedbackMonitor(mode,lambda:state);monitor.evidence_fn=evidence
    monitor.begin_program('initial close lift place',0)
    monitor.invoke('close_gripper',[],{},raw)
    robot=monitor.invoke('read_robot',[],{},raw)
    lift=monitor.invoke('plan_motion',['lift',[.5,0,.2],[1,0,0,0],[robot['id']]],{},raw)
    monitor.invoke('run_plan',[lift],{},raw)
    return monitor,raw,calls,state,visibility


def place(monitor,raw):
    red=monitor.invoke('observe_object',['red cube'],{},raw)
    green=monitor.invoke('observe_object',['green cube'],{},raw)
    return monitor.invoke('plan_motion',['place',[.5,0,.05],[1,0,0,0],[red['id'],green['id']]],{},raw)


def test_historical_verification_is_immutable_but_not_permission():
    graph=DependencyGraph()
    observation=graph.add('observation',{'red':None})
    check=graph.add('check',{'status':'UNCERTAIN'},parents=[observation])
    fact=graph.add('verification_fact',{'historical_only':True,'status':'UNCERTAIN'},parents=[check])
    gate=graph.add('check',{'status':'FAIL'},parents=[(fact,'precondition')])
    graph.invalidate([observation],'updated')
    assert graph.nodes[fact]['valid'] and graph.nodes[fact]['data']['historical_only']
    assert not graph.nodes[check]['valid'] and not graph.nodes[gate]['valid']
    assert observation in graph.ancestors([fact])


def test_missing_close_baseline_reaches_graph_diagnosis_without_unrelated_evidence():
    monitor,raw,calls,state,visibility=build(baseline_visible=False)
    assert not monitor.lifted
    unrelated=monitor.graph.add('check',{'reason':'UNRELATED_CHECK_SENTINEL'})
    target=place(monitor,raw)
    with pytest.raises(ExecutionInterrupted,match='lift_precondition_unverified'):
        monitor.invoke('run_plan',[target],{},raw)
    feedback=monitor.summary()
    evidence=feedback['skill_precondition_evidence']
    lift=next(item for item in evidence if item['check']=='lift')
    assert lift['status']=='UNCERTAIN' and lift['baseline_red_visible'] is False
    baseline=lift['close_baseline_id']
    assert monitor.graph.nodes[baseline]['data']['surfaces']['red'] is None
    assert baseline in monitor.graph.ancestors(feedback['diagnosis']['roots'])
    assert unrelated not in monitor.graph.ancestors(feedback['diagnosis']['roots'])
    assert 'UNRELATED_CHECK_SENTINEL' not in str(feedback['diagnosis'])
    assert calls.count('goto_pose')==1  # Place was blocked before physical dispatch.


@pytest.mark.parametrize('mode',['G0','G1'])
def test_current_precondition_revalidated_after_old_checks_expire(mode):
    monitor,raw,calls,state,visibility=build(mode)
    assert monitor.lifted
    lift_fact=monitor.skill_facts['lift']
    old_lift=monitor.graph.nodes[lift_fact]['data']['source_check_id']
    place(monitor,raw)  # First observation set; then replace it in the next stage.
    target=place(monitor,raw)
    if mode=='G0':assert not monitor.graph.nodes[old_lift]['valid']
    assert monitor.graph.nodes[lift_fact]['valid']
    monitor.begin_program('replanned unfinished place',1)
    monitor.invoke('run_plan',[target],{},raw)
    call=monitor.graph.nodes[monitor.last_call]
    parents=[monitor.graph.nodes[e['source']] for e in monitor.graph.edges if e['target']==call['id'] and e['kind']=='precondition']
    gate=next(node for node in parents if node['kind']=='check' and node['data'].get('check')=='skill_precondition')
    assert gate['data']['status']=='PASS' and gate['data']['metrics']['lift_verified'] is True
    assert lift_fact in monitor.graph.ancestors([gate['id']])
    assert calls.count('goto_pose')==2


def test_new_grasp_forgets_old_lift_pointer_but_keeps_history():
    monitor,raw,calls,state,visibility=build()
    old=monitor.skill_facts['lift'];epoch=monitor.grasp_epoch
    visibility['red']=False
    monitor.invoke('close_gripper',[],{},raw)
    assert monitor.grasp_epoch==epoch+1 and 'lift' not in monitor.skill_facts
    assert monitor.graph.nodes[old]['data']['historical_only'] is True
    target=place(monitor,raw)
    with pytest.raises(ExecutionInterrupted):monitor.invoke('run_plan',[target],{},raw)
    assert all(item['grasp_epoch']==monitor.grasp_epoch for item in monitor.summary()['skill_precondition_evidence'])
    assert old not in monitor.skill_failure_evidence


@pytest.mark.parametrize('mode',['G0','G1'])
def test_real_actor_boundary_preserves_uncertain_skill_evidence(mode):
    monitor,raw,calls,state,visibility=build(mode,baseline_visible=False)
    target=place(monitor,raw)
    code=f'try:\n    run_plan("{target}")\nexcept Exception:\n    close_gripper()\n'
    monitor.begin_program(code,1)
    actor=SandboxProcess(dict.fromkeys(GRAPH_APIS+['close_gripper']),callback=lambda name,args,kwargs:monitor.invoke(name,args,kwargs,raw))
    try:
        result=actor.run(code,timeout=10)
        assert result['status']=='interrupted'
        assert calls.count('close_gripper')==1 and calls.count('goto_pose')==1
        assert any(item['check']=='lift' and item['status']=='UNCERTAIN' for item in monitor.summary()['skill_precondition_evidence'])
    finally:actor.close()
