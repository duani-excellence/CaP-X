import pytest
from test_grounded_monitor import fixture
from capx.execution.grounded_feedback import GroundedMonitor
from capx.execution.sandbox import ExecutionInterrupted


def test_g0_cannot_silently_enable_graph_mechanisms():
    m=GroundedMonitor('G0','stack',lambda:{},graph_options={'diagnostic_slice':True})
    assert m.graph_options==dict(diagnostic_slice=False,selective_invalidation=False,selective_recovery=False)


def test_invalid_ablation_option_rejected():
    with pytest.raises(ValueError):GroundedMonitor('G1','stack',lambda:{},graph_options={'selective_invalidaton':False})


def test_selective_update_preserves_unrelated_plan_only_when_enabled():
    for enabled in (True,False):
        m,s,v,c,f=fixture();m.graph_options['selective_invalidation']=enabled
        a=m.invoke('observe_object',['red cube'],{},f);b=m.invoke('observe_object',['green cube'],{},f)
        plan=m.graph.add('plan',{'irrelevant':True},parents=[b['id']])
        m.invoke('observe_object',['red cube'],{},f)
        assert m.graph.nodes[plan]['valid']==enabled


def test_diagnosis_ablation_retains_same_invalidation_and_restores_whole_source():
    m,s,v,c,f=fixture();m.graph_options['diagnostic_slice']=False
    m.source='complete_actor_source';result=m.summary()
    assert result['diagnostic_source']=='complete_actor_source'
    assert result['graph_options']['selective_invalidation']


def test_conservative_recovery_ablation_discards_pending_sources_not_completed_calls():
    m,s,v,c,f=fixture();m.graph_options['selective_recovery']=False
    old=m.invoke('observe_object',['green cube'],{},f);completed=list(m.call_history)
    with pytest.raises(ExecutionInterrupted):m.invoke('close_gripper',[],{},f)
    assert not m.graph.nodes[old['id']]['valid']
    assert all(m.graph.nodes[x]['state']=='done' for x in completed)
    assert m.summary()['recovery_scope']['policy']=='all_unexecuted_suffix'
