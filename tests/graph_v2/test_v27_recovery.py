import copy
import pytest
from capx.execution.skill_contracts import ContractGraph, ContractBroker, SkillBoundaryStop
from capx.execution.contract_recovery import ContractRecovery, ConditionPlanner


def task():
    return dict(schema='capx.contract_task.v1', objects=['part', 'peer'], resources=['arm'],
        goals=['placed:part', 'placed:peer'], condition_specs={
            k:dict(observable=True) for k in ['ready','geometry','held:part','placed:part','held:peer','placed:peer']},
        intents=[r for name in ['part','peer'] for r in [
            dict(id='get-'+name,skill='get',requires=['ready','geometry'],effects=['held:'+name],resources=['arm']),
            dict(id='put-'+name,skill='put',requires=['held:'+name,'geometry'],effects=['placed:'+name],resources=['arm'],
                 after=['get-'+name],allow_verified_predecessors=True)]])


def fixture():
    c=ContractGraph(task())
    for k in ['ready','geometry']:c.observe(k,'TRUE',{})
    for name in ['part','peer']:
        c.observe('held:'+name,'FALSE',{});c.observe('placed:'+name,'FALSE',{})
    return c


def test_unknown_requires_evidence_not_regrasp():
    c=fixture();c.observe('held:part','UNKNOWN',{});c.observe('ready','FALSE',{})
    p=ConditionPlanner(c)
    d=p.plan(['placed:part'])
    assert d['work']=='evidence' and d['keys']==['held:part']
    p.exhaust(['held:part']);d=p.plan(['placed:part'])
    assert d['work']=='blocked' and not d['allowed_intents']


def test_false_regresses_to_legal_producer_and_preserves_sibling():
    c=fixture();c.observe('placed:peer','TRUE',{})
    d=ConditionPlanner(c).plan()
    assert d['work']=='skill' and d['allowed_intents']==['get-part']
    assert 'placed:peer' in d['preserved']


def test_missing_capability_and_dependency_cycle_terminate():
    c=fixture();c.observe('geometry','FALSE',{})
    d=ConditionPlanner(c).plan()
    assert d['work']=='blocked' and any(x['reason']=='no_producer' for x in d['blocked'])


def test_repaired_work_gets_new_id_not_completed_call_replay():
    c=fixture();t=c.admit('get-part');c.finish(t,{'held:part':'TRUE'})
    c.observe('held:part','FALSE',{})
    p=ConditionPlanner(c);d=p.plan(['placed:part'])
    assert d['allowed_intents']==['get-part'] and d['new_plan_required']==['get-part']
    b=ContractBroker(c,{},lambda k:{})
    new=ContractRecovery(c,b,group='G1').repair('get-part')
    assert new!='get-part' and 'get-part' in c.completed


def test_shared_geometry_failure_is_graph_condition_and_no_fake_progress():
    c=fixture();c.observe('geometry','UNKNOWN',{'frame':1})
    p=ConditionPlanner(c);assert p.plan()['work']=='evidence'
    p.exhaust(['geometry']);c.observe('geometry','UNKNOWN',{'frame':2})
    assert p.plan()['work']=='blocked'
    c.observe('geometry','TRUE',{'frame':3})
    assert p.plan()['work']=='skill'


def test_partial_effects_link_to_failed_call_and_invalidate_lease():
    c=fixture()
    def stop(r):raise SkillBoundaryStop('geometry changed',{'geometry':dict(truth='UNKNOWN',evidence={'source':'live'})})
    b=ContractBroker(c,{'get':stop},lambda ks:{k:dict(truth='TRUE',evidence={}) for k in ks})
    with pytest.raises(SkillBoundaryStop):b.execute('get-part')
    nodes=c.graph.ancestors([c.conditions['geometry']])
    assert any(c.graph.nodes[n]['kind']=='call' for n in nodes)
    assert c.truth('geometry')=='UNKNOWN' and not c.quarantined


def test_task_definition_v27_has_runtime_preconditions_and_no_stale_config():
    from capx.experiments.native_vision_skills import region_task
    d=region_task({'renamed':'descriptor'},grounded=True)
    assert set(d['intents'][0]['requires']) >= {'workspace_geometry','geometry:renamed','grasp_ready:renamed','robot_ready'}
    assert 'target_geometry:0' in d['intents'][1]['requires']
    assert d['condition_specs']['held:renamed']['observable']


@pytest.mark.parametrize('group,ablation,enforced',[('G0','none',False),('G1','none',True),('G1','no_graph_repair',False),('G1','no_selective',True)])
def test_strategies_differ_in_graph_authority_not_skills(group,ablation,enforced):
    from capx.experiments.contract_models import ModelContractRuntime
    class Skills:
        phase='empty';held=None;anchors={'x':True}
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,ks):return {k:dict(truth='TRUE',evidence={}) for k in ks}
    r=ModelContractRuntime(Skills(),task(),group=group,ablation=ablation,
        limits=dict(skill_budget=10,episode_seconds=60,recovery_version=27),emit=lambda *a,**k:None)
    assert r.graph_repair is enforced


def test_rejected_action_is_counted_before_repeated_capture():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.execution.sandbox import ExecutionInterrupted
    class Skills:
        phase='closed_unverified';held={'object':'part'};anchors={'x':True};captures=0
        def acquire(self,r):raise AssertionError('unsafe dispatch')
        def deposit(self,r):raise AssertionError('unsafe dispatch')
        def verify(self,ks):
            self.captures+=1
            return {k:dict(truth='UNKNOWN',evidence={}) for k in ks}
    s=Skills();r=ModelContractRuntime(s,task(),group='G1',limits=dict(skill_budget=8,episode_seconds=60,recovery_version=27),emit=lambda *a,**k:None)
    with pytest.raises(ExecutionInterrupted):r.execute('put-part')
    before=s.captures
    with pytest.raises(ExecutionInterrupted,match='unchanged'):r.execute('put-part')
    assert s.captures==before


def test_graph_scope_blocks_model_selected_out_of_scope_action():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.execution.sandbox import ExecutionInterrupted
    class Skills:
        phase='empty';held=None;anchors={'x':True}
        def acquire(self,r):raise AssertionError('out of scope dispatch')
        def deposit(self,r):raise AssertionError('out of scope dispatch')
        def verify(self,k):raise AssertionError('out of scope capture')
    r=ModelContractRuntime(Skills(),task(),group='G1',limits=dict(skill_budget=8,episode_seconds=60,recovery_version=27),emit=lambda *a,**k:None)
    r.recovering=True;r.work_scope={'allowed_intents':['get-part']}
    with pytest.raises(ExecutionInterrupted,match='outside parent graph'):r.execute('get-peer')


def test_expired_effect_is_reobserved_without_replaying_producer():
    c=fixture();t=c.admit('get-part');c.finish(t,{'held:part':'TRUE'})
    c.observe('placed:part','UNKNOWN',{'expired':True})
    p=ConditionPlanner(c)
    assert p.plan(['placed:part'])['keys']==['placed:part']
    c.observe('placed:part','TRUE',{'source':'new_current_depth'})
    assert p.plan(['placed:part'])['work']=='complete'


def test_public_geometry_failure_does_not_falsify_unrelated_hold():
    from capx.execution.vision_skills import RegionSkills
    class Scene:source='current'
    s=RegionSkills(Scene(),None,None,None,{'schema':'capx.upright_development_bounds.v1','offline_evidence':['fixture']})
    s.held={'object':'widget'};s.effects['held:widget']='TRUE';s.active_condition='target_geometry:0'
    with pytest.raises(SkillBoundaryStop) as caught:s.stop('target not observed')
    assert 'held:widget' not in caught.value.effects
    assert s.effects['held:widget']=='TRUE'


def test_g0_preserves_completed_goal_even_when_resource_effect_is_shared():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.experiments.native_vision_skills import region_task
    class Skills:
        phase='retreated';held=None;anchors={'x':True}
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,ks):
            return {k:dict(truth=('FALSE' if k.startswith('held:') or k=='placed:b' else 'TRUE'),evidence={}) for k in ks}
    r=ModelContractRuntime(Skills(),region_task({'a':'A','b':'B'},grounded=True),group='G0',
        limits=dict(skill_budget=8,episode_seconds=60,recovery_version=27),emit=lambda *a,**k:None)
    r.broker.refresh(list(r.c.definition['condition_specs']))
    r.last_intent='deposit-0'
    assert r.prepare_recovery()
    assert r.work_scope['allowed_intents']==['acquire-1']


def test_known_workspace_revalidates_needed_region_without_rebinding_roles():
    from capx.execution.vision_skills import RegionSkills
    class Scene:
        source='current';obs={}
        def frame(self):pass
        def bind_regions(self):raise AssertionError('unneeded global rebinding')
        def observe_region(self,role):
            assert role=='source';return {'status':'PASS'}
    s=RegionSkills(Scene(),None,None,None,{'schema':'capx.upright_development_bounds.v1','offline_evidence':['fixture']})
    s.anchors={'source':{},'destination':{}}
    assert s.verify(['workspace_geometry'])['workspace_geometry']['truth']=='TRUE'


def test_consumed_effects_are_synchronized_with_current_public_verifier():
    c=fixture();c.intents['get-part']['invalidates']=['ready']
    physical={'ready':'TRUE','geometry':'TRUE','held:part':'FALSE'}
    def act(row):physical.update({'ready':'FALSE','held:part':'TRUE'})
    b=ContractBroker(c,{'get':act},lambda ks:{k:dict(truth=physical[k],evidence={'capture':'current-depth'}) for k in ks},sync_invalidates=True)
    b.execute('get-part')
    assert c.truth('ready')=='FALSE' and c.truth('held:part')=='TRUE'
    node=c.graph.nodes[c.conditions['held:part']]
    assert node['data']['evidence']['capture']=='current-depth'


def test_unchanged_recovery_actor_frontier_does_not_buy_more_llm_requests():
    from capx.experiments.contract_models import ModelContractRuntime
    class Skills:
        phase='empty';held=None;anchors={'x':True}
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,ks):
            return {k:dict(truth=('FALSE' if k.startswith(('held:','placed:')) else 'TRUE'),evidence={}) for k in ks}
    r=ModelContractRuntime(Skills(),task(),group='G1',limits=dict(skill_budget=8,episode_seconds=60,recovery_version=27,max_same_frontier=1),emit=lambda *a,**k:None)
    assert r.prepare_recovery()
    assert not r.prepare_recovery()
    assert r.terminal_reason=='recovery_frontier_unchanged'


def test_shared_unrecoverable_condition_cuts_irrelevant_evidence_work():
    c=fixture();c.observe('geometry','UNKNOWN',{})
    c.observe('ready','UNKNOWN',{})
    p=ConditionPlanner(c);p.exhaust(['geometry'])
    d=p.plan()
    assert d['work']=='blocked' and not d['keys']


def test_independent_unknown_frontier_is_captured_together():
    c=fixture()
    for g in c.definition['goals']:c.observe(g,'UNKNOWN',{})
    d=ConditionPlanner(c).plan()
    assert d['work']=='evidence' and set(d['keys'])==set(c.definition['goals'])


def test_no_graph_repair_retains_models_legal_evidence_capability():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.experiments.native_vision_skills import region_task
    class Skills:
        phase='retreated';held=None;anchors={'x':True}
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,ks):
            return {k:dict(truth=('FALSE' if k.startswith('held:') else 'TRUE'),evidence={}) for k in ks}
    r=ModelContractRuntime(Skills(),region_task({'a':'A','b':'B'},grounded=True),group='G1',ablation='no_graph_repair',
        limits=dict(skill_budget=8,episode_seconds=60,recovery_version=27),emit=lambda *a,**k:None)
    r.last_intent='deposit-0';r.c.observe('placed:a','UNKNOWN',{'expired':True})
    assert r.prepare_recovery()
    assert 'acquire-1' in r.work_scope['evidence_allowed_intents']
    assert not r.graph_repair


def test_new_physical_view_can_reopen_exhausted_evidence_but_new_id_cannot():
    c=fixture();c.observe('geometry','UNKNOWN',{'capture':1})
    p=ConditionPlanner(c);p.motion_epoch(0);p.exhaust(['geometry'])
    c.observe('geometry','UNKNOWN',{'capture':2});p.motion_epoch(0)
    assert p.plan()['work']=='blocked'
    p.motion_epoch(1)
    assert p.plan()['work']=='evidence'
