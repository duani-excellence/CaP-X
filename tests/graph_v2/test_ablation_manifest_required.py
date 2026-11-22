"""Acceptance guards for formal ablation initial sharing; no model requests.

These tests intentionally precede the implementation, while live v7 execution
keeps business source frozen. Existing unrelated formal gates are mocked so a
missing shared-initial contract cannot be hidden by a different error.
"""
import pytest
from test_campaign import config
from capx.experiments import graph_campaign as campaign


def formal_ablation(tmp_path,monkeypatch):
    c=config(tmp_path)
    c.update(study='mechanism_ablation',groups=['G0','G1'],faults=['none'],
             graph_options={'diagnostic_slice':False,'selective_invalidation':True,'selective_recovery':True})
    monkeypatch.setattr(campaign,'_acceptance',lambda *a,**kw:{'fixture':True})
    monkeypatch.setattr(campaign,'formal_provenance',lambda *a,**kw:{'fixture':True})
    return c,campaign.make_plan(c['tasks'],1,1,1)


@pytest.mark.parametrize('value',['omitted',None,''])
def test_formal_mechanism_ablation_requires_initial_manifest_before_artifacts(tmp_path,monkeypatch,value):
    c,plan=formal_ablation(tmp_path,monkeypatch)
    if value!='omitted':c['initial_manifest']=value
    root=tmp_path/'formal'
    with pytest.raises(ValueError,match='initial_manifest'):
        campaign.freeze_campaign(root,c,plan,phase='formal')
    assert not root.exists()


def test_formal_mechanism_ablation_rejects_missing_manifest_file(tmp_path,monkeypatch):
    c,plan=formal_ablation(tmp_path,monkeypatch);c['initial_manifest']=str(tmp_path/'missing.json')
    root=tmp_path/'formal'
    with pytest.raises((ValueError,FileNotFoundError)):
        campaign.freeze_campaign(root,c,plan,phase='formal')
    assert not root.exists()


def test_formal_mechanism_ablation_rejects_nonmanifest_json(tmp_path,monkeypatch):
    c,plan=formal_ablation(tmp_path,monkeypatch);path=tmp_path/'not-a-manifest.json';path.write_text('{}')
    c['initial_manifest']=str(path);root=tmp_path/'formal'
    with pytest.raises(ValueError,match='manifest'):
        campaign.freeze_campaign(root,c,plan,phase='formal')
    assert not root.exists()
