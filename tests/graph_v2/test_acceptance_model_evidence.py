"""Formal task readiness must come from the validated development execution.

Fixtures simulate recorded receipts, not an actual provider request. These tests
precede the acceptance implementation while online v7 source remains frozen.
"""
import json
from pathlib import Path
import pytest
from test_campaign import completed_development_fixture
from capx.experiments import graph_campaign as campaign


def acceptance_reference(tmp_path,config,path):
    target=tmp_path/'final-acceptance.json'
    campaign.atomic(target,{'validated_tasks':['stack'],'evidence_files':[str(path)]})
    config['acceptance_file']=str(target)


@pytest.mark.parametrize('kind',['oracle','no_receipt','copied_result'])
def test_unbound_success_cannot_satisfy_model_task_acceptance(tmp_path,kind):
    config,plan,dev=completed_development_fixture(tmp_path)
    unit=campaign.read(dev/'state.json')['units'][0]
    row=campaign.read(unit['result'])['episodes'][1]
    row.update(reward=1.,status='success',policy_success=True,stop_reason='physical_success')
    if kind=='oracle':row.update(model='oracle',provider='oracle',request_paths=[])
    if kind=='no_receipt':row.update(model='deepseek-v4-pro',request_paths=[])
    # Even copied real-shaped receipts cannot authorize a standalone success
    # which is absent from the validated development result ledger.
    external=tmp_path/(kind+'-success.json');campaign.atomic(external,row)
    acceptance_reference(tmp_path,config,external)
    with pytest.raises(ValueError,match='acceptance|validated.*development|development.*success'):
        campaign.freeze_campaign(tmp_path/'formal',config,plan,development_root=dev)
    assert not (tmp_path/'formal').exists()


def promote_recorded_result(dev,group='G0',policy=True):
    unit=campaign.read(dev/'state.json')['units'][0]
    path=Path(unit['result']);body=campaign.read(path)
    row=next(r for r in body['episodes'] if r['group']==group)
    row.update(reward=1.,status='success',policy_success=policy,stop_reason='physical_success')
    persisted=path.parent/'episodes'/group/'result.json'
    campaign.atomic(path,body);campaign.atomic(persisted,row)
    return persisted


def test_recorded_g0_success_from_validated_development_is_accepted(tmp_path):
    config,plan,dev=completed_development_fixture(tmp_path)
    persisted=promote_recorded_result(dev)
    acceptance_reference(tmp_path,config,persisted)
    result=campaign.freeze_campaign(tmp_path/'formal',config,plan,development_root=dev)
    assert str(persisted.resolve()) in result['acceptance']['development_evidence_hashes']
    assert result['acceptance']['executed_development_scenes']=={'stack':20}


@pytest.mark.parametrize('group,policy',[('B0',True),('G0',False)])
def test_baseline_only_or_policy_rejected_success_does_not_accept_grounded_task(tmp_path,group,policy):
    config,plan,dev=completed_development_fixture(tmp_path)
    persisted=promote_recorded_result(dev,group,policy)
    acceptance_reference(tmp_path,config,persisted)
    with pytest.raises(ValueError,match='acceptance|validated.*development|development.*success'):
        campaign.freeze_campaign(tmp_path/'formal',config,plan,development_root=dev)
    assert not (tmp_path/'formal').exists()


def test_persisted_success_without_deepseek_receipt_cannot_accept_task(tmp_path):
    config,plan,dev=completed_development_fixture(tmp_path)
    persisted=promote_recorded_result(dev)
    row=campaign.read(persisted)
    for request in row['request_paths']:
        receipt=Path(request)/'receipt.json'
        body=campaign.read(receipt);body['response_model']='oracle'
        campaign.atomic(receipt,body)
    acceptance_reference(tmp_path,config,persisted)
    with pytest.raises(ValueError,match='20 completed|receipt|acceptance'):
        campaign.freeze_campaign(tmp_path/'formal',config,plan,development_root=dev)
    assert not (tmp_path/'formal').exists()
