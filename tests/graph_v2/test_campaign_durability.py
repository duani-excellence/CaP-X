"""No-model regression tests for complete-plan budgets and durable batch control."""
import json
from pathlib import Path
import pytest
from test_campaign import config
from capx.experiments import graph_campaign as campaign


def fake_formal_gates(monkeypatch):
    monkeypatch.setattr(campaign,'_acceptance',lambda *a,**kw:{'fixture':True})
    monkeypatch.setattr(campaign,'formal_provenance',lambda *a,**kw:{'fixture':True})


def test_formal_normal_requires_full_plan_unique_request_coverage(tmp_path,monkeypatch):
    fake_formal_gates(monkeypatch)
    c=config(tmp_path);c['tasks']=['stack','lift','restack'];c['budgets'].update(max_blocks=8,max_requests=13799)
    p=campaign.make_plan(c['tasks'],20,100,2)
    with pytest.raises(ValueError,match='13800'):
        campaign.freeze_campaign(tmp_path/'insufficient',c,p,phase='formal')
    assert not (tmp_path/'insufficient').exists()
    c['budgets']['max_requests']=13800
    frozen=campaign.freeze_campaign(tmp_path/'enough',c,p,phase='formal')
    coverage=frozen['request_coverage']
    assert coverage['units']==600 and coverage['required_requests']==13800
    assert coverage['requests_per_unit']==23 and coverage['sufficient']


def test_development_may_use_small_cap_without_claiming_full_coverage(tmp_path):
    c=config(tmp_path);c['budgets'].update(max_blocks=8,max_requests=1)
    frozen=campaign.freeze_campaign(tmp_path/'dev',c,campaign.make_plan(c['tasks'],20,100,2),phase='development')
    assert frozen['request_coverage']['sufficient'] is False
    assert frozen['request_coverage']['required_requests']==40*23


def test_formal_stress_uses_actual_graph_groups_and_fault_units(tmp_path,monkeypatch):
    fake_formal_gates(monkeypatch)
    c=config(tmp_path);c.update(groups=['G0','G1'],faults=['target_shift','support_refresh'],study='stress')
    c['budgets'].update(max_blocks=8,max_requests=1)
    frozen=campaign.freeze_campaign(tmp_path/'stress',c,campaign.make_plan(c['tasks'],1,2,2),phase='formal')
    assert frozen['request_coverage']['units']==8
    assert frozen['request_coverage']['required_requests']==8*15
    assert frozen['request_coverage']['sufficient'] is False  # Normal-only mandatory gate.


def pair(kw):
    return {'episodes':[dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],group=g,fault=kw['fault'],reward=0.,status='task_failure') for g in kw['groups']]}


def test_request_stop_finishes_current_pair_then_resume_skips_it(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],2,1,1),phase='development')
    calls=[]
    def runner(**kw):
        calls.append(kw['seed'])
        if len(calls)==1:campaign.request_stop(root,'maintenance after current pair')
        return pair(kw)
    report=campaign.run_campaign(root,runner=runner)
    assert len(calls)==1 and report['complete_units']==1
    assert report['stop_reason']=='requested_boundary_stop'
    report=campaign.run_campaign(root,runner=runner)
    assert len(calls)==2 and len(set(calls))==2 and report['stop_reason']=='complete'
    assert len(list((root/'stop-requests').glob('*.json')))==1


def test_checkpoint_cannot_drop_a_failed_or_pending_scenario(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],2,1,1),phase='development')
    state=campaign.read(root/'state.json');state['units'].pop();campaign.atomic(root/'state.json',state)
    with pytest.raises(ValueError,match='planned unit ledger'):
        campaign.run_campaign(root,runner=lambda **kw:pair(kw))


def test_done_result_missing_group_not_reported_as_complete(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],1,1,1),phase='development')
    campaign.run_campaign(root,runner=lambda **kw:pair(kw))
    result=Path(campaign.read(root/'state.json')['units'][0]['result']);body=campaign.read(result)
    body['episodes'].pop();campaign.atomic(result,body)
    with pytest.raises(ValueError,match='groups'):campaign.status(root)


def test_completed_formal_campaign_does_not_require_services_on_resume(tmp_path,monkeypatch):
    fake_formal_gates(monkeypatch)
    monkeypatch.setattr(campaign,'verify_service_inventory',lambda *a,**kw:{'fixture':True})
    c=config(tmp_path);root=tmp_path/'formal';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],1,1,1),phase='formal')
    campaign.run_campaign(root,runner=lambda **kw:pair(kw))
    def unavailable(*a,**kw):raise AssertionError('No service needed after all episodes completed')
    monkeypatch.setattr(campaign,'verify_service_inventory',unavailable)
    assert campaign.run_campaign(root,runner=unavailable)['stop_reason']=='complete'


def test_batch_limit_must_be_positive(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],1,1,1),phase='development')
    with pytest.raises(ValueError,match='max_units'):campaign.run_campaign(root,max_units=0,runner=lambda **kw:pair(kw))


def test_stop_requests_are_unique_and_preserved_even_with_same_clock(tmp_path,monkeypatch):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],1,1,1),phase='development')
    monkeypatch.setattr(campaign.time,'time_ns',lambda:123)
    a=campaign.request_stop(root,'first');b=campaign.request_stop(root,'second')
    assert a['request']!=b['request']
    assert campaign.read(a['request'])['reason']=='first'
    assert campaign.read(b['request'])['reason']=='second'


def test_recovered_completed_pair_does_not_double_count_persisted_runtime(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],1,1,1),phase='development')
    campaign.run_campaign(root,runner=lambda **kw:pair(kw))
    state=campaign.read(root/'state.json');state['units'][0]['state']='started'
    before=state['active_seconds'];campaign.atomic(root/'state.json',state)
    assert campaign.run_campaign(root,runner=lambda **kw:pytest.fail('must not replay'))['active_seconds']==before


def test_uncertain_request_stops_before_initializing_another_unit(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],1,1,1),phase='development')
    campaign.atomic(root/'request-ledger.json',{'requests':[{'state':'uncertain','reserved_usd':.1,'label':'previous'}]})
    result=campaign.run_campaign(root,runner=lambda **kw:pytest.fail('must not start another unit'))
    assert result['stop_reason']=='uncertain_request_requires_audit' and result['pending_units']==1


def test_partial_campaign_analysis_cannot_claim_formal_ready(tmp_path,monkeypatch):
    from capx.experiments import graph_statistics
    c=config(tmp_path);root=tmp_path/'dev';campaign.freeze_campaign(root,c,campaign.make_plan(c['tasks'],2,1,1),phase='development')
    campaign.run_campaign(root,runner=lambda **kw:pair(kw),max_units=1)
    monkeypatch.setattr(graph_statistics,'analyze',lambda *a,**kw:{'strata':{'none':{'G1_vs_G0':{'formal_estimate_ready':True},'G1_vs_B0':{'formal_estimate_ready':True}}}})
    output=tmp_path/'analysis.json';campaign.main(['analyze','--campaign',str(root),'--output',str(output)])
    result=campaign.read(output)
    assert result['campaign_complete'] is False
    assert result['missing_planned_episodes']==3
    assert not result['strata']['none']['G1_vs_G0']['formal_estimate_ready']
