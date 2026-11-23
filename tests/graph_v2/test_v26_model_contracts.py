import copy
import pytest
from capx.execution.skill_contracts import ContractGraph, ContractBroker, SkillBoundaryStop
from capx.execution.contract_recovery import ContractRecovery


def task():
    return dict(schema='capx.contract_task.v1', objects=['part','other'], resources=['arm'],
        goals=['done:part','done:other'], intents=[
            dict(id=n, skill='transfer', object=n, requires=['ready'], effects=['done:'+n],
                 resources=['arm'], after=[]) for n in ('part','other')])


def test_scope_is_selective_without_replaying_completed_actions():
    calls=[]
    def verify(keys):
        calls.append(keys)
        return {k:dict(truth='TRUE',evidence={'frame':'current'}) for k in keys}
    for group,expected in [('G0',{'ready','done:part','done:other'}),('G1',{'ready','done:other'})]:
        c=ContractGraph(task());c.observe('ready','TRUE',{});t=c.admit('part');c.finish(t,{'done:part':'TRUE'})
        old=c.conditions['done:part']
        r=ContractRecovery(c,ContractBroker(c,{'transfer':lambda row:None},verify),group=group)
        result=r.refresh_for('other')
        assert set(calls[-1])==expected
        assert 'part' in c.completed
        assert (c.conditions['done:part']==old)==(group=='G1')
        assert ('done:part' in result['preserved'])==(group=='G1')


def test_new_id_repair_has_budget_and_no_false_progress_from_node_ids():
    c=ContractGraph(task());c.stopped.add('part')
    broker=ContractBroker(c,{},lambda ks:{k:dict(truth='TRUE',evidence={}) for k in ks})
    r=ContractRecovery(c,broker,group='G1',max_repairs=2,max_same_frontier=1)
    new=r.repair('part');assert new!='part' and c.intents[new]['repair_of']=='part'
    c.stopped.add(new)
    with pytest.raises(ValueError,match='progress'):r.repair(new)


def test_unknown_effect_never_repaired():
    c=ContractGraph(task());c.quarantined=True
    r=ContractRecovery(c,ContractBroker(c,{},lambda k:{}),group='G1')
    with pytest.raises(ValueError,match='quarantined'):r.repair('part')


def test_effect_verified_does_not_create_duplicate_physical_work():
    c=ContractGraph(task());c.stopped.add('part');c.observe('done:part','TRUE',{})
    r=ContractRecovery(c,ContractBroker(c,{},lambda k:{}),group='G1')
    with pytest.raises(ValueError,match='verified'):r.repair('part')


def test_skill_effect_invalidates_consumed_conditions():
    d=task();d['intents'][0]['invalidates']=['ready']
    c=ContractGraph(d)
    broker=ContractBroker(c,{'transfer':lambda r:None},
        lambda ks:{k:dict(truth='TRUE',evidence={}) for k in ks})
    broker.execute('part')
    assert c.truth('done:part')=='TRUE' and c.truth('ready')=='UNKNOWN'


def test_stopped_acquire_can_resume_declared_deposit_after_verified_hold():
    from capx.experiments.native_vision_skills import region_task
    c=ContractGraph(region_task({'renamed-part':'descriptor'}))
    c.stopped.add('acquire-0');c.observe('held:renamed-part','TRUE',{})
    assert c.order_satisfied(c.intents['deposit-0'])


def test_renaming_does_not_change_recovery_policy():
    for name in ['red','widget','some_new_task_object']:
        d=task();d['intents'][0]['object']=name;d['objects'][0]=name
        c=ContractGraph(d);c.stopped.add('part')
        r=ContractRecovery(c,ContractBroker(c,{},lambda k:{}),group='G1')
        assert c.intents[r.repair('part')]['object']==name


def test_real_runtime_dispatch_logs_continuation_and_interrupts_known_error():
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.execution.sandbox import ExecutionInterrupted
    from capx.experiments.native_vision_skills import region_task
    class Skills:
        phase='empty';held=None;anchors={'ready':True}
        def acquire(self,row):pass
        def deposit(self,row):raise SkillBoundaryStop('uncertain_geometry',{})
        def verify(self,keys):return {k:dict(truth='TRUE',evidence={}) for k in keys}
    logs=[]
    def emit(kind,**row):logs.append(dict(kind=kind,**row))
    r=ModelContractRuntime(Skills(),region_task({'x':'label'}),group='G1',
        limits=dict(skill_budget=8,episode_seconds=60),emit=emit)
    r.execute('acquire-0')
    with pytest.raises(ExecutionInterrupted,match='uncertain_geometry'):r.execute('deposit-0')
    assert r.c.stopped=={'deposit-0'} and not r.c.quarantined
    assert logs


def test_effect_diagnosis_reaches_actual_program_and_completed_call():
    c=ContractGraph(task());program=c.register_program('execute_skill("part")','request-42')
    c.observe('ready','TRUE',{'capture':'public-1'})
    token=c.admit('part');c.finish(token,{'done:part':'TRUE'})
    ancestry=c.graph.ancestors([c.conditions['done:part']])
    assert program in ancestry and c.completed['part'] in ancestry
    # Replacing historical action inputs cannot invalidate a physically verified
    # independent effect merely because its old permission was superseded.
    c.observe('ready','UNKNOWN',{'capture':'public-2'})
    assert c.truth('done:part')=='TRUE'


def test_partial_known_stop_invalidates_possible_effects_too():
    d=task();d['intents'][0]['invalidates']=['ready'];c=ContractGraph(d)
    def stopped(row):raise SkillBoundaryStop('stopped after partially moving',{})
    b=ContractBroker(c,{'transfer':stopped},lambda ks:{k:dict(truth='TRUE',evidence={}) for k in ks})
    with pytest.raises(SkillBoundaryStop):b.execute('part')
    assert c.truth('ready')=='UNKNOWN' and not c.quarantined


def test_no_physical_work_and_no_ready_continuation_ends_replanning():
    from capx.experiments.contract_models import ready_continuation
    c=ContractGraph(task());c.observe('ready','UNKNOWN',{})
    assert not ready_continuation(c)
    c.observe('ready','TRUE',{})
    assert ready_continuation(c)
    c.stopped.update(c.intents)
    assert not ready_continuation(c)

