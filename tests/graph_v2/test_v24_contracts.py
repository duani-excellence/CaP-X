import copy
import pytest

from capx.execution.skill_contracts import ContractGraph, validate_contract_task
from capx.execution.graph_executive import GraphExecutive


def definition():
    return dict(schema='capx.contract_task.v1', objects=['a', 'b', 'tray'],
        goals=['inside:a', 'inside:b'], resources=['arm'],
        intents=[dict(id=o, skill='transfer', requires=['localized:'+o, 'localized:tray'],
                      effects=['inside:'+o], resources=['arm'], after=[])
                 for o in ['a', 'b']])


def test_multiple_independent_goals_and_shared_evidence():
    c=ContractGraph(definition())
    for k in ['localized:a', 'localized:b', 'localized:tray']: c.observe(k, 'TRUE', {'capture':'current'})
    assert GraphExecutive().next_contract(c)['intent']=='a'
    token=c.admit('a'); c.finish(token, {'inside:a':'TRUE'})
    assert GraphExecutive().next_contract(c)['intent']=='b'
    c.observe('localized:b','FALSE',{'capture':'changed'})
    assert c.truth('inside:a')=='TRUE'
    assert GraphExecutive().next_contract(c)['work']=='evidence'


def test_final_capture_invalidates_permission_and_resources():
    c=ContractGraph(definition())
    for k in ['localized:a', 'localized:b', 'localized:tray']: c.observe(k,'TRUE',{})
    t=c.admit('a')
    with pytest.raises(ValueError,match='resource'): c.admit('b')
    c.observe('localized:tray','UNKNOWN',{'reason':'occluded'})
    with pytest.raises(ValueError,match='stale'): c.check_permission(t)
    c.abort(t,known_effect=False)
    with pytest.raises(ValueError,match='quarantined'): c.admit('a')


def test_completed_calls_never_replayed_but_effect_can_be_revoked():
    c=ContractGraph(definition())
    for k in ['localized:a','localized:tray']: c.observe(k,'TRUE',{})
    t=c.admit('a'); c.finish(t,{'inside:a':'TRUE'})
    with pytest.raises(ValueError,match='completed'): c.admit('a')
    c.observe('inside:a','FALSE',{'capture':'new contradiction'})
    assert not c.complete()
    # Repair requires a NEW explicit intent, not flipping the completed call.
    new=copy.deepcopy(c.intents['a']);new['id']='repair-a'
    c.register(new)
    assert c.admit('repair-a')


def test_no_completion_from_skill_return_or_model_claim():
    c=ContractGraph(definition())
    for k in ['localized:a','localized:tray']:c.observe(k,'TRUE',{})
    t=c.admit('a');c.finish(t,{})
    assert c.truth('inside:a')=='UNKNOWN'
    assert not c.complete()


def test_selective_evidence_scope_vs_strong_conservative_baseline():
    c=ContractGraph(definition())
    for k in ['localized:a','localized:b','localized:tray']:c.observe(k,'TRUE',{})
    assert set(c.recovery_scope(['localized:a'],'G1'))=={'localized:a'}
    assert set(c.recovery_scope(['localized:a'],'G0'))=={'localized:a','localized:b','localized:tray'}
    assert c.complexity()['width']==2
    assert c.complexity()['shared_conditions']==1


@pytest.mark.parametrize('rename',[{'a':'bolt','b':'cup','tray':'shelf'}, {'a':'blue','b':'small','tray':'receiver'}])
def test_role_names_do_not_change_scheduling(rename):
    d=definition()
    def transform(x):
        if isinstance(x,str):return ':'.join(rename.get(k,k) for k in x.split(':'))
        if isinstance(x,list):return [transform(v) for v in x]
        if isinstance(x,dict):return {k:transform(v) for k,v in x.items()}
        return x
    d=transform(d)
    c=ContractGraph(d)
    decision=GraphExecutive().next_contract(c)
    assert decision['work']=='evidence'


def test_cycle_and_unknown_resource_rejected():
    d=definition();d['intents'][0]['after']=['b'];d['intents'][1]['after']=['a']
    with pytest.raises(ValueError,match='cycle'):validate_contract_task(d)
    d=definition();d['intents'][0]['resources']=['third-arm']
    with pytest.raises(ValueError,match='resource'):validate_contract_task(d)


def test_ready_independent_branch_runs_before_blocked_branch():
    c=ContractGraph(definition())
    for k in ['localized:b','localized:tray']:c.observe(k,'TRUE',{})
    assert GraphExecutive().next_contract(c)['intent']=='b'


def test_handover_contract_refuses_donor_release_without_receiver_hold():
    d=dict(schema='capx.contract_task.v1',objects=['tool'],resources=['left','right'],goals=['transferred'],
        intents=[dict(id='release',skill='release',requires=['receiver_hold','donor_hold'],
            effects=['transferred'],resources=['left','right'],after=[])])
    c=ContractGraph(d);c.observe('donor_hold','TRUE',{});c.observe('receiver_hold','UNKNOWN',{})
    with pytest.raises(ValueError,match='precondition'):c.admit('release')


def test_generic_broker_final_refresh_precedes_dispatch():
    from capx.execution.skill_contracts import ContractBroker
    c=ContractGraph(definition());calls=[]
    for k in ['localized:a','localized:tray']:c.observe(k,'TRUE',{})
    broker=ContractBroker(c,{'transfer':lambda row: calls.append(row['id'])},
        verify=lambda keys:{k:dict(truth='UNKNOWN',evidence={'capture':'final'}) for k in keys})
    with pytest.raises(ValueError,match='precondition'):broker.execute('a')
    assert calls==[]


def test_generic_broker_unknown_effect_quarantines_all_actors():
    from capx.execution.skill_contracts import ContractBroker
    c=ContractGraph(definition())
    def broken(row):raise RuntimeError('transport error with unknown motion')
    broker=ContractBroker(c,{'transfer':broken},
        verify=lambda keys:{k:dict(truth='TRUE',evidence={'capture':'fresh'}) for k in keys})
    with pytest.raises(RuntimeError):broker.execute('a')
    assert c.quarantined


def test_fork_join_and_reordered_declarations():
    d=definition();d['intents'].insert(0,dict(id='join',skill='verify_all',requires=['inside:a','inside:b'],
        effects=['all_done'],resources=[],after=['a','b']))
    d['goals']=['all_done'];c=ContractGraph(d)
    assert c.complexity()['joins']==1
    assert c.complexity()['depth']==2
    assert len([e for e in c.graph.edges if e['kind']=='precondition'])==2


def test_revoked_effect_regresses_join_and_creates_new_minimal_repair():
    d=definition();d['intents'].append(dict(id='join',skill='verify',requires=['inside:a','inside:b'],
        effects=['all_done'],effect_dependencies={'all_done':['inside:a','inside:b']},resources=[],after=['a','b']))
    d['goals']=['all_done'];c=ContractGraph(d)
    for k in ['localized:a','localized:b','localized:tray']:c.observe(k,'TRUE',{})
    for i in ['a','b','join']:
        t=c.admit(i);c.finish(t,{k:'TRUE' for k in c.intents[i]['effects']})
    assert c.complete()
    c.observe('inside:a','FALSE',{'capture':'object displaced'})
    assert not c.complete()
    decision=GraphExecutive().next_contract(c)
    assert decision['intent'] not in ('a','b','join')
    assert c.intents[decision['intent']]['repair_of']=='a'
    assert c.truth('inside:b')=='TRUE'
    assert c.completed.keys()>={'a','b','join'}


def test_unchanged_conditions_cannot_generate_unbounded_new_ids():
    c=ContractGraph(definition())
    for k in ['localized:a','localized:b','localized:tray']:c.observe(k,'TRUE',{})
    for i in ['a','b']:
        t=c.admit(i);c.finish(t,{'inside:'+i:'FALSE' if i=='a' else 'TRUE'})
    i=GraphExecutive().next_contract(c)['intent'];t=c.admit(i);c.finish(t,{'inside:a':'FALSE'})
    assert GraphExecutive().next_contract(c)['work']=='stop'


def test_verified_existing_effect_can_discharge_optional_skill_without_fake_call():
    d=definition();d['intents'].append(dict(id='join',skill='verify',requires=['inside:a','inside:b'],
        effects=['all_done'],resources=[],after=['a','b'],allow_verified_predecessors=True))
    d['goals']=['all_done'];c=ContractGraph(d)
    for k in ['inside:a','inside:b']:c.observe(k,'TRUE',{'capture':'existing physical effect'})
    assert GraphExecutive().next_contract(c)['intent']=='join'
    assert not c.completed
    assert c.admit('join')


def test_required_returned_order_is_not_fabricated_from_current_state():
    d=definition();d['intents'][1]['after']=['a'];c=ContractGraph(d)
    for k in ['inside:a','localized:b','localized:tray']:c.observe(k,'TRUE',{})
    with pytest.raises(ValueError,match='predecessor'):c.admit('b')


def test_unobservable_branch_does_not_prevent_other_branch_evidence():
    c=ContractGraph(definition());executive=GraphExecutive()
    first=executive.next_contract(c)
    second=executive.next_contract(c,excluded=[first['intent']])
    assert second['intent']!=first['intent']
    assert executive.next_contract(c,excluded=list(c.intents))['work']=='stop'
    assert not c.repair_count
