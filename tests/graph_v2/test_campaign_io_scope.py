"""Full campaign provenance, bounded per-unit input archiving, invocation-only cache."""
import json
from pathlib import Path
import pytest
from capx.experiments import graph_campaign as campaign, shared_initial as shared
from test_shared_initial import exports
from test_runner_v2 import fake


def test_runtime_inventory_excludes_unselected_run_inputs(tmp_path,fake,monkeypatch):
    root,c,v,u,s,path,ref=exports(tmp_path,fake,monkeypatch)
    v['initial_manifest']=str(path)
    full=campaign.source_inventory(v)
    runtime=campaign.source_inventory(v,include_run_inputs=False)
    assert set(runtime)<set(full)
    assert str(path.resolve()) not in runtime and ref['initial_file'] not in runtime
    assert str(Path(campaign.__file__).resolve()) in runtime


def test_manifest_loads_source_once_per_invocation_not_once_per_entry(tmp_path,fake,monkeypatch):
    root,c,v,u,s,path,ref=exports(tmp_path,fake,monkeypatch)
    calls=[];original=campaign.load_campaign
    def counted(*args,**kwargs):
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(campaign,'load_campaign',counted)
    shared.validate_initial_manifest(path,v)
    assert len(calls)==1
    shared.validate_initial_manifest(path,v)
    assert len(calls)==2  # Never cache verification across invocation/resume.
    Path(ref['initial_file']).write_text('{}')
    with pytest.raises(ValueError,match='drift'):shared.validate_initial_manifest(path,v)


def test_unit_archive_keeps_selected_bytes_not_global_normal_ledger(tmp_path,fake,monkeypatch):
    from capx.experiments import graph_v2 as runner
    root,c,v,u,s,path,ref=exports(tmp_path,fake,monkeypatch)
    v['initial_manifest']=str(path)
    # fake patches archive for simulated execution; use the original callable.
    archive=ORIGINAL_ARCHIVE
    out=tmp_path/'unit-archive';out.mkdir()
    hashes=archive(out,v,initial_ref=ref)
    archived=campaign.read(out/'source-archive.json')
    assert ref['initial_file'] in archived
    assert str(path.resolve()) not in archived
    assert str(root/'request-ledger.json') not in archived
    assert str(root/'state.json') not in archived
    assert all(hashes[p]==h for p,h in ref['files'].items() if p in archived)
    Path(ref['initial_file']).write_text('{}')
    with pytest.raises(ValueError,match='drift|changed'):
        other=tmp_path/'bad-archive';other.mkdir();archive(other,v,initial_ref=ref)


from capx.experiments.graph_v2 import archive as ORIGINAL_ARCHIVE


def test_campaign_reference_is_small_and_drift_blocks_unit(tmp_path,fake,monkeypatch):
    root,c,v,u,s,path,ref=exports(tmp_path,fake,monkeypatch)
    frozen=root/'frozen.json'
    context=dict(root=str(root),signature=campaign.read(frozen)['signature'],frozen_sha256=shared.file_hash(frozen),_initial_validation_cache={('private','cache'):object()})
    out=tmp_path/'bounded';out.mkdir()
    hashes=ORIGINAL_ARCHIVE(out,v,initial_ref=ref,campaign_ref=context)
    assert hashes[str(frozen.resolve())]==context['frozen_sha256']
    assert not any(k.startswith('_') for k in campaign.read(out/'campaign-reference.json'))
    assert str(frozen.resolve()) not in campaign.read(out/'source-archive.json')
    frozen.write_text('{}')
    other=tmp_path/'drift';other.mkdir()
    with pytest.raises(ValueError,match='drift'):ORIGINAL_ARCHIVE(other,v,campaign_ref=context)


def test_validate_cache_does_not_retain_every_large_pair_payload(tmp_path,fake,monkeypatch):
    root,c,v,u,s,path,ref=exports(tmp_path,fake,monkeypatch)
    cache={};shared.validate_initial_manifest(path,v,_cache=cache)
    assert ('json',str(Path(ref['result_file']).resolve())) not in cache


def test_unit_archive_cannot_bless_business_edit_after_campaign_check(tmp_path):
    from test_campaign import config
    c=config(tmp_path);root=tmp_path/'race-campaign'
    frozen=campaign.freeze_campaign(root,c,campaign.make_plan(['stack'],1,1,1),phase='development')
    context=dict(root=str(root),signature=frozen['signature'],frozen_sha256=shared.file_hash(root/'frozen.json'),
                 _runtime_hashes=campaign.source_inventory(c,include_run_inputs=False))
    Path(c['sources'][0]).write_text('edited_between_guard_and_archive = True\n')
    out=tmp_path/'edited-unit';out.mkdir()
    with pytest.raises(ValueError,match='source.*drift|source.*differs'):
        ORIGINAL_ARCHIVE(out,c,campaign_ref=context)


def test_snapshot_loading_verifies_the_exact_bytes_it_parses(tmp_path):
    snap=tmp_path/'snapshot.json';manifest=tmp_path/'scenes.json'
    campaign.atomic(snap,{'task':'stack','seed':7,'state_hash':'old','model_hash':'m'})
    campaign.atomic(manifest,{'scenes':[{'task':'stack','seed':7,'snapshot':'snapshot.json'}]})
    expected={str(p.resolve()):shared.file_hash(p) for p in (snap,manifest)}
    campaign.atomic(snap,{'task':'stack','seed':7,'state_hash':'changed','model_hash':'m'})
    with pytest.raises(ValueError,match='Snapshot.*drift'):
        campaign._snapshots({'snapshot_manifest':str(manifest)},expected_hashes=expected)


def test_last_unit_source_drift_is_preserved_but_never_completed_as_success(tmp_path):
    from test_campaign import config
    c=config(tmp_path);root=tmp_path/'last-unit'
    campaign.freeze_campaign(root,c,campaign.make_plan(['stack'],1,1,1),phase='development')
    def changed(**kw):
        Path(c['sources'][0]).write_text('changed_during_last_unit = True\n')
        return {'episodes':[dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],fault=kw['fault'],group=g,
            status='success',reward=1,policy_success=True,executed=True) for g in kw['groups']]}
    report=campaign.run_campaign(root,runner=changed)
    assert report['stop_reason']=='source_changed'
    assert report['source_integrity_complete'] is False
    rows=campaign.records(root)
    assert all(r['reward']==1 and r['policy_success'] is False and r['status']=='source_changed' for r in rows)
    assert report['complete_units']==1  # Retain the attempt, never retry physical actions.


@pytest.mark.parametrize('stage',['export','reuse'])
def test_source_drift_history_cannot_be_reused_after_bytes_restored(tmp_path,fake,monkeypatch,stage):
    root,c,v,u,s,path,ref=exports(tmp_path,fake,monkeypatch)
    result=Path(ref['result_file']);body=campaign.read(result)
    for row in body['episodes']:row.update(status='source_changed',source_changes=['temporary-edit.py'],policy_success=False)
    campaign.atomic(result,body)
    if stage=='export':
        with pytest.raises(ValueError,match='source.*integrity|Source.*integrity'):
            shared.export_initial_manifest(root,tmp_path/'new-initials.json')
    else:
        ref['files'][str(result.resolve())]=shared.file_hash(result)
        with pytest.raises(ValueError,match='source.*integrity|Source.*integrity'):
            shared.load_initial_reference(ref,v,task='stack',seed=u['seed'],repeat=0,fault='none',snapshot=s)
