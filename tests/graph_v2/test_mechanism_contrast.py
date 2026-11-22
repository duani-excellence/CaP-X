"""Direct full-G1 versus ablated-G1 effects, never differences of significance."""
import copy
from pathlib import Path
import pytest
from capx.experiments import graph_campaign as campaign,shared_initial as shared
from test_ablation_scene_subset import configs


def fixture(tmp_path,monkeypatch):
    normal,variant,plan=configs(tmp_path)
    normal['tasks']=variant['tasks']=['stack']
    plan=campaign.make_plan(['stack'],20,22,2)
    roots=[tmp_path/'normal',tmp_path/'ablated']
    source=dict(config=normal,plan=plan,phase='formal',signature='source-signature')
    target=dict(config=variant,plan=plan,phase='formal',signature='target-signature')
    initial=tmp_path/'initials.json';variant['initial_manifest']=str(initial)
    refs={};states={};rows={}
    for root,frozen in zip(roots,[source,target]):
        root.mkdir();units=campaign.planned_units(frozen['config'],plan,'formal')
        states[str(root)]={'signature':frozen['signature'],'units':units,'active_seconds':1,'stop_reason':'complete'}
        rows[str(root)]=[]
        for unit in units:
            unit.update(state='done',result=str(root/(unit['id']+'.json')))
            identity=shared.identity(unit)
            entry=dict(task=unit['task'],seed=unit['seed'],repeat=unit['repeat'],fault='none',source_campaign=str(roots[0]),source_signature=source['signature'],source_unit_id=unit['id'],source_request_uid='uid-'+unit['id'])
            if root==roots[0]:refs[identity]=entry
            group_rows=[]
            for group in unit['groups']:
                row=dict(task=unit['task'],seed=unit['seed'],repeat=unit['repeat'],fault='none',group=group,
                         ablation='full' if root==roots[0] else 'without_diagnostic_slice',executed=True,
                         status='success',reward=1,execution_status='ok',snapshot_hash='snapshot-'+str(unit['seed']),initial_prompt_hash='prompt',api_docs_hash='docs',code_hash='code')
                if root==roots[1]:
                    row.update(initial_reused=True,initial_source=dict(entry,generated_in_this_run=False,initial_requests_this_run=0))
                    # Deliberately all ablated policies fail; original physical state may still attain.
                    row.update(status='controlled_exit',reward=1,policy_success=False)
                group_rows.append(row)
            campaign.atomic(Path(unit['result']),{'episodes':group_rows});rows[str(root)]+=group_rows
        campaign.atomic(root/'state.json',states[str(root)])
        campaign.atomic(root/'request-ledger.json',{'requests':[]})
    monkeypatch.setattr(campaign,'load_campaign',lambda root,**kw: {str(roots[0]):source,str(roots[1]):target}[str(root)])
    def validate(path,config,units=None,**kw):
        assert path==str(initial) or Path(path)==initial
        return {shared.identity(u):refs[shared.identity(u)] for u in units}
    monkeypatch.setattr(shared,'validate_initial_manifest',validate)
    return roots,source,target,states,refs


def run(roots,**kwargs):
    from capx.experiments.graph_contrast import paired_mechanism_contrast
    return paired_mechanism_contrast(*roots,n_bootstrap=100,**kwargs)


def test_direct_contrast_clusters_both_draws_and_keeps_raw_diagnostic_separate(tmp_path,monkeypatch):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch);out=run(roots)
    p=out['policy_contrast'];raw=out['physical_attainment_contrast']
    assert p['treatment']=='full_G1' and p['control']=='ablated_G1'
    assert p['paired_generations']==40 and p['independent_scenarios']==20
    assert p['macro_success_difference']==1 and p['ci95']==[1,1]
    assert raw['macro_success_difference']==0 and raw['is_task_success_metric'] is False
    assert out['formal_estimate_ready'] is True
    assert out['selection']['normal_units']==44 and out['selection']['selected_units']==40


@pytest.mark.parametrize('field',['snapshot_hash','initial_prompt_hash','api_docs_hash','code_hash','initial_reused','initial_source'])
def test_contrast_rejects_changed_pair_or_source(tmp_path,monkeypatch,field):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch)
    unit=states[str(roots[1])]['units'][0];body=campaign.read(unit['result']);row=next(r for r in body['episodes'] if r['group']=='G1')
    row[field]=False if field=='initial_reused' else {} if field=='initial_source' else 'drift'
    campaign.atomic(Path(unit['result']),body)
    with pytest.raises(ValueError,match='initial|hash|source|reus'):run(roots)


@pytest.mark.parametrize('change',['pending','missing_repeat','uncertain_request','incomplete_normal'])
def test_incomplete_campaign_never_produces_formal_conclusion(tmp_path,monkeypatch,change):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch)
    root=roots[0] if change=='incomplete_normal' else roots[1]
    if change=='uncertain_request':campaign.atomic(root/'request-ledger.json',{'requests':[{'state':'uncertain'}]})
    else:
        state=states[str(root)]
        for unit in state['units'][:2 if change=='missing_repeat' else 1]:unit['state']='pending';unit.pop('result')
        campaign.atomic(root/'state.json',state)
    out=run(roots)
    assert out['formal_estimate_ready'] is False
    assert out['policy_contrast']['formal_estimate_ready'] is False


@pytest.mark.parametrize('change',['phase','fault','options','source_campaign'])
def test_contrast_is_only_declared_formal_normal_subset(tmp_path,monkeypatch,change):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch)
    if change=='phase':t['phase']='development'
    elif change=='fault':t['config']['faults']=['target_shift']
    elif change=='options':t['config']['graph_options']['selective_invalidation']=False
    else:next(iter(refs.values()))['source_campaign']=str(tmp_path/'unrelated')
    with pytest.raises(ValueError):run(roots)


def test_known_invalid_initials_remain_paired_policy_failures(tmp_path,monkeypatch):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch)
    for root in roots:
        unit=states[str(root)]['units'][0];body=campaign.read(unit['result'])
        for row in body['episodes']:row.update(status='invalid_program',reward=0,code_hash='sha-of-null',stop_reason='initial_generation_failed')
        campaign.atomic(Path(unit['result']),body)
    out=run(roots)
    assert out['policy_contrast']['paired_generations']==40
    assert out['policy_contrast']['generation_ties']==1
    assert out['policy_contrast']['macro_success_difference']==pytest.approx(.975)
    assert out['formal_estimate_ready']


def test_contrast_cli_exclusive_output_without_provider(tmp_path,monkeypatch,capsys):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch)
    path=tmp_path/'contrast.json'
    args=['contrast','--normal',str(roots[0]),'--ablation',str(roots[1]),'--output',str(path),'--bootstrap','10']
    campaign.main(args)
    assert campaign.read(path)['policy_contrast']['treatment']=='full_G1'
    with pytest.raises(FileExistsError):campaign.main(args)


def test_source_integrity_failure_prevents_formal_mechanism_conclusion(tmp_path,monkeypatch):
    roots,s,t,states,refs=fixture(tmp_path,monkeypatch)
    unit=states[str(roots[1])]['units'][0];body=campaign.read(unit['result'])
    for row in body['episodes']:row.update(status='source_changed',policy_success=False,source_changes=['edited-business.py'])
    campaign.atomic(Path(unit['result']),body)
    out=run(roots)
    assert out['ablation_status']['source_integrity_complete'] is False
    assert out['formal_estimate_ready'] is False
