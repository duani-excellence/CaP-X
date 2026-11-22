"""No-model tests for preparing real freeze inputs and declared graph ablations."""
import copy
import json
from pathlib import Path
import pytest
import test_campaign
from capx.experiments import graph_campaign as c


def snapshot_factory(calls,fail_seed=None):
    def prepare(task,seed,dest):
        calls.append((task,seed));Path(dest).mkdir(parents=True)
        if seed==fail_seed:raise RuntimeError('unprepared scene')
        return {'valid':True,'snapshot_replay_verified':True},{'task':task,'seed':seed,'state_hash':f'state-{seed}','model_hash':'model'}
    return prepare


def test_snapshot_preparation_has_no_provider_and_resumes_without_repeating(tmp_path,monkeypatch):
    config=test_campaign.config(tmp_path);plan=c.make_plan(['stack'],2,3,2)
    monkeypatch.setattr('capx.experiments.graph_v2.configure_services',lambda config:None)
    calls=[];root=tmp_path/'snapshots'
    first=c.prepare_snapshots(root,config,plan,prepare=snapshot_factory(calls),max_scenes=1)
    assert first['completed']==1 and not first['complete']
    result=c.prepare_snapshots(root,config,plan,resume=True,prepare=snapshot_factory(calls))
    assert result['complete'] and result['completed']==3 and len(calls)==3
    manifest=c.read(root/'scenes.json')
    assert {r['seed'] for r in manifest['scenes']}=={r['seed'] for r in plan['test']}
    assert not ({r['seed'] for r in manifest['scenes']}&{r['seed'] for r in plan['development']})
    assert all(c.read(root/r['snapshot'])['state_hash']==r['state_hash'] for r in manifest['scenes'])
    with pytest.raises(FileExistsError):c.prepare_snapshots(root,config,plan,prepare=snapshot_factory(calls))
    c.prepare_snapshots(root,config,plan,resume=True,prepare=lambda *a:pytest.fail('must not replay'))
    assert not list(root.glob('**/request-ledger.json'))


def test_preparation_failure_keeps_seed_and_never_selects_success_replacement(tmp_path,monkeypatch):
    config=test_campaign.config(tmp_path);plan=c.make_plan(['stack'],1,2,2);calls=[]
    monkeypatch.setattr('capx.experiments.graph_v2.configure_services',lambda config:None)
    root=tmp_path/'snapshots';failed=plan['test'][0]['seed']
    result=c.prepare_snapshots(root,config,plan,prepare=snapshot_factory(calls,failed))
    assert result['failed']==1 and not result['complete'] and result['completed']==1
    c.prepare_snapshots(root,config,plan,resume=True,prepare=lambda *a:pytest.fail('failed scenes cannot silently retry'))
    assert len(calls)==2 and c.read(root/'scenes.json')['failed_scenes'][0]['seed']==failed


def test_snapshot_resume_rejects_drift_and_unknown_started(tmp_path,monkeypatch):
    config=test_campaign.config(tmp_path);plan=c.make_plan(['stack'],1,2,1)
    monkeypatch.setattr('capx.experiments.graph_v2.configure_services',lambda config:None)
    root=tmp_path/'snapshots';c.prepare_snapshots(root,config,plan,prepare=snapshot_factory([]),max_scenes=1)
    state=c.read(root/'preparation.json');state['scenes'][1]['state']='started';c.atomic(root/'preparation.json',state)
    with pytest.raises(RuntimeError,match='started'):c.prepare_snapshots(root,config,plan,resume=True,prepare=snapshot_factory([]))
    changed=copy.deepcopy(config);changed['model']='other'
    with pytest.raises(ValueError,match='signature'):c.prepare_snapshots(root,changed,plan,resume=True,prepare=snapshot_factory([]))


def test_development_cli_requires_service_identity_before_creating_campaign(tmp_path):
    config=test_campaign.config(tmp_path);path=tmp_path/'config.json';c.atomic(path,config);root=tmp_path/'dev'
    with pytest.raises(ValueError,match='service_inventory'):
        c.main(['prepare-development','--config',str(path),'--output',str(root)])
    assert not root.exists()


def test_development_cli_explicit_exploratory_is_not_formal_eligible(tmp_path,capsys):
    config=test_campaign.config(tmp_path);path=tmp_path/'config.json';c.atomic(path,config);root=tmp_path/'dev'
    c.main(['prepare-development','--config',str(path),'--output',str(root),'--exploratory'])
    assert c.read(root/'frozen.json')['config']['exploratory_only'] is True
    assert 'formal' in capsys.readouterr().err.lower()


def declared_fixture(tmp_path,monkeypatch):
    original=test_campaign.config
    full={'diagnostic_slice':True,'selective_invalidation':True,'selective_recovery':True}
    variant=dict(full,diagnostic_slice=False)
    def cfg(path):
        value=original(path);value.update(graph_options=full,allowed_graph_ablations=[variant]);return value
    monkeypatch.setattr(test_campaign,'config',cfg)
    return (*test_campaign.completed_development_fixture(tmp_path),variant)


def test_predeclared_graph_ablation_reuses_normal_dev_without_changing_base(tmp_path,monkeypatch):
    config,plan,dev,variant=declared_fixture(tmp_path,monkeypatch)
    config.update(study='mechanism_ablation',groups=['G0','G1'],graph_options=variant)
    result=c._acceptance(config,plan,dev)
    assert result['graph_ablation']['predeclared'] is True
    assert result['graph_ablation']['options']==variant


@pytest.mark.parametrize('change',['undeclared','multiple_options','episode_budget','model_budget','source','runtime'])
def test_ablation_cannot_smuggle_other_changes(tmp_path,monkeypatch,change):
    config,plan,dev,variant=declared_fixture(tmp_path,monkeypatch)
    config.update(study='mechanism_ablation',groups=['G0','G1'],graph_options=dict(variant))
    if change=='undeclared':config['graph_options']={'diagnostic_slice':True,'selective_invalidation':False,'selective_recovery':True}
    elif change=='multiple_options':config['graph_options']['selective_recovery']=False
    elif change=='episode_budget':config['budgets']['api_budget']+=1
    elif change=='model_budget':config['budgets']['max_tokens']+=1
    elif change=='source':Path(config['sources'][0]).write_text('CHANGED=1')
    elif change=='runtime':config['runtime_environment']['OMP_NUM_THREADS']='3'
    with pytest.raises(ValueError):c._acceptance(config,plan,dev)
    assert not (tmp_path/'bad').exists()


def test_ablation_allows_only_campaign_scale_budget_growth(tmp_path,monkeypatch):
    config,plan,dev,variant=declared_fixture(tmp_path,monkeypatch)
    config.update(study='mechanism_ablation',groups=['G0','G1'],graph_options=variant)
    for name in ('max_requests','cost_limit_usd','wall_seconds'):config['budgets'][name]*=5
    assert c._acceptance(config,plan,dev)['graph_ablation']['predeclared']
