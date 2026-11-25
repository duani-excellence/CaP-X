"""V29 symbolic mechanism tests; these are not Vision or physical successes."""
import copy
import pytest
from capx.execution.skill_contracts import ContractGraph
from capx.execution.causal_plan import CausalPlan, ground_library, program_order, validate_sequence, repair_plan


def definition():
    return dict(schema='capx.contract_task.v1',objects=['unnamed_widget'],resources=['left','right'],
        goals=['installed','peer_done'],condition_specs={'geometry':{'observable':True}},
        intents=[dict(id='take',skill='take',requires=['free','geometry'],effects=['held'],deletes=['free'],resources=['left']),
                 dict(id='put',skill='put',requires=['held','aligned'],effects=['installed','free'],deletes=['held'],resources=['left']),
                 dict(id='peer',skill='peer',requires=['peer_ready'],effects=['peer_done'],resources=['right'])],
        skill_library=[dict(name='align',bindings=[{}],contract=dict(skill='align',requires=['held'],effects=['aligned'],resources=['left'])),
                       dict(name='sense_support',bindings=[{}],contract=dict(skill='restore',requires=['free'],effects=['aligned'],resources=['left']))])


def fixture():
    c=ContractGraph(definition())
    for key in ['free','geometry','aligned','peer_ready']:c.observe(key,'TRUE',{'fixture':True})
    for key in ['held','installed','peer_done']:c.observe(key,'FALSE',{'fixture':True})
    return c


def test_program_compiler_never_hoists_conditionals_or_dynamic_calls():
    assert program_order("execute_skill('take')\nexecute_skill('put')",{'take','put'})==['take','put']
    for code in ["if False:\n execute_skill('take')", "execute_skill(x)", "for i in execution_state(): execute_skill(i)"]:
        with pytest.raises(ValueError):program_order(code,{'take'})


def test_automatic_causality_resources_and_data_without_after():
    c=fixture();c.intents['put']['data_requires']=['geometry']
    p=CausalPlan(c,['take','put','peer'])
    assert any(e['source']=='take' and e['target']=='put' and e['kind']=='causal' and e['condition']=='held' for e in p.edges)
    assert any(e['kind']=='resource' for e in p.edges)
    assert any(e['kind']=='data' and e['condition']=='geometry' for e in p.edges)
    assert not any(e['source']=='put' and e['target']=='peer' for e in p.edges)
    assert c.truth('held')=='FALSE'  # predicted effects never establish public truth
    assert any(n['kind']=='causal_action' for n in c.graph.nodes.values())


def test_support_invalidation_affects_only_causal_branch_and_not_completed_history():
    c=fixture();p=CausalPlan(c,['take','put','peer'])
    t=c.admit('take');c.finish(t,{'held':'TRUE'});c.observe('aligned','FALSE',{})
    assert p.affected(['aligned'])==['put']
    assert 'take' not in p.pending()
    assert any(x['condition']=='held' for x in p.live_support())
    assert c.graph.nodes[c.completed['take']]['state']=='done'


def test_local_repair_composes_new_library_skill_retains_independent_branch():
    c=fixture();p=CausalPlan(c,['take','put','peer'])
    t=c.admit('take');c.finish(t,{'held':'TRUE'});c.observe('free','FALSE',{});c.observe('aligned','FALSE',{})
    d=repair_plan(p,ground_library(c.definition),failed=['aligned'])
    assert d['status']=='ready' and [r['skill'] for r in d['fragment']]==['align']
    assert d['retained']==['put','peer'] and d['affected']==['put'] and d['replaced']==[]
    assert d['validation']['valid']
    assert 'take' not in d['retained']


def test_multistep_library_search_not_limited_to_declared_intents():
    c=fixture();c.observe('aligned','FALSE',{});c.observe('held','FALSE',{});c.observe('tool','FALSE',{})
    c.definition['skill_library']=[dict(name='tool',bindings=[{}],contract=dict(skill='fetch',requires=['free'],effects=['tool'])),
       dict(name='fix',bindings=[{}],contract=dict(skill='fix',requires=['tool'],effects=['aligned']))]
    p=CausalPlan(c,['take','put','peer']);d=repair_plan(p,ground_library(c.definition),failed=['aligned'])
    assert [r['skill'] for r in d['fragment']]==['fetch','fix']
    assert d['retained']==['take','put','peer']


def test_fragment_which_breaks_retained_condition_rejected_before_commit():
    c=fixture();p=CausalPlan(c,['take','put','peer']);before=len(c.graph.nodes)
    c.observe('aligned','FALSE',{})
    harmful=dict(id='bad',skill='bad',requires=['free'],effects=['aligned'],deletes=['peer_ready'])
    d=repair_plan(p,[harmful],failed=['aligned'])
    assert d['status']=='blocked'
    assert len(c.intents)==3
    v=validate_sequence({'free':'TRUE','geometry':'TRUE','peer_ready':'TRUE'},[harmful]+[c.intents[i] for i in p.order],c.definition['goals'],{})
    assert not v['valid'] and v['condition']=='peer_ready'


def test_unknown_is_evidence_false_is_physical_and_quarantine_never_searches():
    c=fixture();p=CausalPlan(c,['take','put','peer']);c.observe('geometry','UNKNOWN',{})
    d=repair_plan(p,ground_library(c.definition));assert d['status']=='evidence' and d['keys']==['geometry']
    c.observe('aligned','FALSE',{});c.quarantined=True
    assert repair_plan(p,ground_library(c.definition))['reason']=='quarantined'


def test_evidence_exhaustion_and_search_budget_are_bounded():
    c=fixture();p=CausalPlan(c,['take','put','peer']);c.observe('geometry','UNKNOWN',{})
    assert repair_plan(p,ground_library(c.definition),exhausted={'geometry':'UNKNOWN'})['status']=='blocked'
    c.observe('geometry','TRUE',{});c.observe('aligned','FALSE',{})
    assert repair_plan(p,ground_library(c.definition),max_expansions=0)['status']=='blocked'


def test_unseen_object_binding_and_negative_precondition():
    d=definition();d['skill_library']=[dict(name='release',bindings=[{'object':'new part !'}],
        contract=dict(skill='release',object='{object}',requires=['held:{object}'],requires_false=['obstruction'],effects=['free'],deletes=['held:{object}']))]
    lib=ground_library(d);assert lib[0]['object']=='new part !'
    assert not validate_sequence({'held:new part !':'TRUE','obstruction':'UNKNOWN'},lib,['free'],{})['valid']
    assert validate_sequence({'held:new part !':'TRUE','obstruction':'FALSE'},lib,['free'],{})['valid']


def test_atomic_patch_preserves_history_rejects_changed_state():
    c=fixture();p=CausalPlan(c,['take','put','peer']);c.observe('aligned','FALSE',{})
    d=repair_plan(p,ground_library(c.definition));before=set(c.intents)
    c.observe('free','FALSE',{})
    with pytest.raises(ValueError,match='state changed'):p.commit(d)
    assert set(c.intents)==before


def test_patch_new_ids_and_live_versions_retained():
    c=fixture();p=CausalPlan(c,['take','put','peer']);c.observe('aligned','FALSE',{})
    d=repair_plan(p,ground_library(c.definition));p2=p.commit(d)
    assert p2.order[-3:]==['take','put','peer']
    assert p2.order[0] not in p.order
    assert c.truth('aligned')=='FALSE'
    assert p2.revision!=p.revision
    assert any(n['kind']=='causal_patch' for n in c.graph.nodes.values())


def test_destroy_and_evidence_invalidation_have_different_semantics():
    rows=[dict(id='move',requires=['free'],effects=['held'],deletes=['free'],invalidates=['geometry']),
          dict(id='finish',requires=['held','geometry'],effects=['goal'])]
    v=validate_sequence({'free':'TRUE','geometry':'TRUE'},rows,['goal'],{'geometry':{'observable':True}},conditional=True)
    assert v['valid'] and v['obligations']==['geometry']
    rows[0]['deletes'].append('geometry')
    assert not validate_sequence({'free':'TRUE','geometry':'TRUE'},rows,['goal'],{'geometry':{'observable':True}},conditional=True)['valid']


def test_completed_effect_broken_does_not_replay_old_call():
    c=fixture();p=CausalPlan(c,['take','put','peer'])
    for i in ['take','put','peer']:
        for k in c.intents[i]['requires']:c.observe(k,'TRUE',{})
        t=c.admit(i);c.finish(t,{k:'TRUE' for k in c.intents[i]['effects']})
    c.observe('installed','FALSE',{});c.observe('held','FALSE',{});c.observe('free','TRUE',{})
    lib=[dict(c.intents[i],after=[]) for i in ['take','put']]
    d=repair_plan(p,lib,failed=['installed']);assert d['status']=='ready'
    q=p.commit(d)
    assert not set(q.order)&set(c.completed)
    assert [c.intents[i]['skill'] for i in q.order]==['take','put']


def test_common_broker_checks_negative_data_and_never_asserts_predicted_delete():
    from capx.execution.skill_contracts import ContractBroker
    d=definition();d['intents'][0].update(requires_false=['hazard'],data_requires=['calibration'],deletes=['free'])
    c=ContractGraph(d);seen=[]
    facts={'free':'TRUE','geometry':'TRUE','hazard':'UNKNOWN','calibration':'TRUE','held':'TRUE'}
    def verify(keys):
        seen.append(keys);return {k:dict(truth=facts.get(k,'UNKNOWN'),evidence={}) for k in keys}
    b=ContractBroker(c,{'take':lambda row:None},verify,sync_invalidates=True)
    with pytest.raises(ValueError):b.execute('take')
    facts['hazard']='FALSE';b.execute('take')
    assert 'calibration' in seen[0] and 'hazard' in seen[0]
    assert c.truth('free')=='TRUE'  # verifier, not the predicted delete, wins


def test_runtime_compiles_actual_model_order_and_guards_unexecuted_conflict():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.execution.sandbox import ExecutionInterrupted
    class Skills:
        phase='empty';held=None;anchors={};motion_version=0
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,ks):return {k:dict(truth='TRUE',evidence={}) for k in ks}
    r=ModelContractRuntime(Skills(),definition(),group='G1',limits=dict(recovery_version=29,skill_budget=10,episode_seconds=60),emit=lambda *a,**k:None)
    r.compile_program("execute_skill('take')\nexecute_skill('put')\nexecute_skill('peer')")
    with pytest.raises(ExecutionInterrupted,match='causal predecessor'):r.execute('put')
    assert not r.c.completed


def test_local_patch_can_resume_without_llm_via_existing_broker():
    from capx.experiments.contract_models import ModelContractRuntime
    class Skills:
        phase='empty';held=None;anchors={};motion_version=0
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,ks):return {k:dict(truth='TRUE',evidence={}) for k in ks}
        def evidence_features(self,k):return {'frame':1}
    r=ModelContractRuntime(Skills(),definition(),group='G1',limits=dict(recovery_version=29,skill_budget=10,episode_seconds=60),emit=lambda *a,**k:None)
    r.c=fixture();r.planner.c=r.c;r.broker.contracts=r.c;r.recovery.c=r.c
    r.compile_program("execute_skill('take')\nexecute_skill('put')\nexecute_skill('peer')")
    assert r.prepare_recovery()
    assert r.deterministic_code()=="execute_skill('take')"
    assert any(e['kind']=='causal_repair_decision' for e in r.events)


def test_future_transient_evidence_is_deferred_until_consuming_action():
    c=fixture();c.intents['peer']['requires'].append('future_grasp')
    c.definition['condition_specs']['future_grasp']={'observable':True,'kind':'tool_data'}
    p=CausalPlan(c,['take','put','peer']);c.observe('aligned','FALSE',{})
    d=repair_plan(p,ground_library(c.definition),failed=['aligned'])
    assert d['status']=='ready' and d['deferred_evidence']==['future_grasp']
    assert 'future_grasp' in d['validation']['obligations']
    assert c.truth('future_grasp')=='UNKNOWN'


def test_invalid_patch_resources_never_partially_register():
    c=fixture();p=CausalPlan(c,['take','put','peer']);c.observe('aligned','FALSE',{})
    d=repair_plan(p,ground_library(c.definition));d['fragment']=[dict(id='x',skill='x',effects=['x']),dict(id='y',skill='y',effects=['aligned'],resources=['missing'])]
    before=set(c.intents)
    with pytest.raises(ValueError):p.commit(d)
    assert set(c.intents)==before


def test_continuation_validation_ablation_is_distinct_but_not_motion_permission():
    c=fixture();c.observe('aligned','FALSE',{})
    p=CausalPlan(c,['take','put','peer'])
    row=dict(id='fix',skill='fix',requires=['free'],effects=['aligned'],invalidates=['peer_ready'])
    strict=repair_plan(p,[row],failed=[])
    ablated=repair_plan(p,[row],failed=[],check_continuation=False)
    assert strict['status']=='blocked'
    assert ablated['status']=='ready' and not ablated['validation']['valid']
    assert c.truth('aligned')=='FALSE'


def test_incremental_support_links_actual_observation_and_call_to_future_plan():
    c=fixture();p=CausalPlan(c,['take','put','peer']);c.observe('aligned','TRUE',{'source':'new RGBD'})
    support=[n for n in c.graph.nodes.values() if n['kind']=='live_causal_support']
    assert any(n['data']['condition']=='aligned' and n['data']['node']==c.conditions['aligned'] for n in support)
    t=c.admit('take');c.finish(t,{'held':'TRUE'})
    assert any(n['kind']=='causal_execution' and n['data']['intent']=='take' for n in c.graph.nodes.values())
    current=[n['id'] for n in support if n['data']['condition']=='aligned']
    c.observe('aligned','FALSE',{})
    assert all(not c.graph.nodes[n]['valid'] for n in current)


def test_analyzer_includes_v29_and_identifies_policy_variant(tmp_path):
    import importlib.util,json
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('capx_contract_analyzer',Path(__file__).resolve().parents[2]/'scripts/local/analyze_contract_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'config.json').write_text(json.dumps({'services':{}}));d=tmp_path/'G1-no_causal';d.mkdir()
    (d/'result.json').write_text(json.dumps({'schema':'capx.contract_models.v29','group':'G1','actors':[],'request_paths':[]}))
    rows=module.analyze(tmp_path)['episodes'];assert len(rows)==1 and rows[0]['variant']=='G1-no_causal'


def test_negative_live_support_tracks_expected_false_not_missing_true():
    c=fixture();c.intents['take']['requires_false']=['hazard'];c.observe('hazard','FALSE',{})
    p=CausalPlan(c,['take','put','peer'])
    edge=next(e for e in p.live_support() if e['condition']=='hazard')
    assert edge['expected']=='FALSE' and c.truth('hazard')==edge['expected']


def test_actual_deepseek_literal_sequence_loop_is_compiled_without_execution():
    code="ids=['take','put','peer']\nfor item in ids:\n execute_skill(item)\nprint('done')"
    assert program_order(code,{'take','put','peer'})==['take','put','peer']
    assert program_order("for item in ('take','put'): execute_skill(item)",{'take','put'})==['take','put']
    bad=["ids=['take']\nfor item in ids:\n ids.append('put')\n execute_skill(item)",
         "ids=execution_state()\nfor item in ids: execute_skill(item)",
         "ids=['take']\nfor item in ids:\n if False: execute_skill(item)",
         "ids=['take']\nids=['put']\nfor item in ids: execute_skill(item)",
         "for item in ['take']:\n execute_skill(item)\nelse: execute_skill('put')"]
    for code in bad:
        with pytest.raises(ValueError):program_order(code,{'take','put'})
