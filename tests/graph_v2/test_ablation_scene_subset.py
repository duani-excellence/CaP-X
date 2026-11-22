"""Predeclared scene selection is independent of success and model output."""
import copy
import pytest
from test_campaign import config
from capx.experiments import graph_campaign as campaign


def configs(tmp_path):
    normal=config(tmp_path)
    normal.update(tasks=['stack','lift','restack'],study='normal',groups=['B0','G0','G1'],faults=['none'],
        graph_options={'diagnostic_slice':True,'selective_invalidation':True,'selective_recovery':True},
        mechanism_scene_subset={'rule':'first_scene_index','scenes_per_task':20})
    normal['allowed_graph_ablations']=[dict(normal['graph_options'],diagnostic_slice=False)]
    variant=copy.deepcopy(normal)
    variant.update(study='mechanism_ablation',groups=['G0','G1'],graph_options=normal['allowed_graph_ablations'][0])
    return normal,variant,campaign.make_plan(normal['tasks'],20,100,2)


def test_formal_ablation_uses_exact_preregistered_scene_subset_with_all_draws(tmp_path):
    normal,variant,plan=configs(tmp_path)
    units=campaign.planned_units(variant,plan,'formal')
    expected={(s['task'],s['seed'],r,'none') for s in plan['test'] if s['scene_index']<20 for r in range(2)}
    actual={(u['task'],u['seed'],u['repeat'],u['fault']) for u in units}
    assert actual==expected and len(units)==120
    assert all(set(u['groups'])=={'G0','G1'} for u in units)
    assert len(campaign.planned_units(normal,plan,'formal'))==600
    assert len(campaign.planned_units(variant,plan,'development'))==120


@pytest.mark.parametrize('change',[None,{'rule':'first_scene_index','scenes_per_task':10},{'rule':'successful_scenes','scenes_per_task':20}])
def test_subset_contract_cannot_be_added_changed_or_removed_after_development(tmp_path,change):
    normal,variant,plan=configs(tmp_path)
    if change is None:variant.pop('mechanism_scene_subset')
    else:variant['mechanism_scene_subset']=change
    with pytest.raises(ValueError,match='mechanism_scene_subset|subset'):
        campaign.compatible_development(variant,normal)


@pytest.mark.parametrize('subset',[
    {'rule':'successful_scenes','scenes_per_task':20},
    {'rule':'first_scene_index','scenes_per_task':0},
    {'rule':'first_scene_index','scenes_per_task':101},
    {'rule':'first_scene_index','scenes_per_task':True},
    {'rule':'first_scene_index','scenes_per_task':20,'seeds':[123]},
])
def test_invalid_or_result_based_subset_rejected(tmp_path,subset):
    normal,variant,plan=configs(tmp_path);variant['mechanism_scene_subset']=subset
    with pytest.raises(ValueError,match='subset|scene'):
        campaign.planned_units(variant,plan,'formal')


def test_larger_formal_power_plan_keeps_advance_rule_and_all_repeats(tmp_path):
    normal,variant,plan=configs(tmp_path)
    expanded=campaign.make_plan(plan['tasks'],20,160,2,plan['allocation_seed'])
    assert expanded['development']==plan['development']
    units=campaign.planned_units(variant,expanded,'formal')
    assert len(units)==120
    selected={(s['task'],s['seed']) for s in expanded['test'] if s['scene_index']<20}
    assert {(u['task'],u['seed']) for u in units}==selected
    assert not {u['seed'] for u in units}&{s['seed'] for s in plan['development']}


def manifest_fixture(tmp_path,monkeypatch):
    from capx.experiments import shared_initial as shared
    normal,variant,plan=configs(tmp_path)
    source={'phase':'formal','config':normal,'plan':plan,'signature':'normal-signature'}
    root=tmp_path/'normal'
    entries=[dict(task=u['task'],seed=u['seed'],repeat=u['repeat'],fault=u['fault'],
                  source_campaign=str(root),source_signature=source['signature'],source_phase='formal',available=True)
             for u in campaign.planned_units(normal,plan,'formal')]
    body={'schema':shared.SCHEMA,'source_campaign':str(root),'source_signature':source['signature'],'source_phase':'formal','entries':entries}
    path=tmp_path/'initials.json';campaign.atomic(path,body)
    monkeypatch.setattr(shared,'_source',lambda entry,config=None,cache=None:source)
    checked=[]
    def contents(entry,source,cache=None):
        checked.append(shared.identity(entry))
        if not entry['available']:raise ValueError('Source initial unavailable')
        return {'code':None,'error_status':'invalid_program'}
    monkeypatch.setattr(shared,'_contents',contents)
    # This fixture isolates allocation; test_shared_initial covers real-shaped
    # receipt, source, prompt, code and Snapshot validation without these mocks.
    return shared,variant,plan,path,body,checked


def test_full_manifest_is_retained_but_only_fixed_subset_returned(tmp_path,monkeypatch):
    shared,c,plan,path,body,checked=manifest_fixture(tmp_path,monkeypatch)
    target=campaign.planned_units(c,plan,'formal')
    selected=shared.validate_initial_manifest(path,c,target,phase='formal',plan=plan)
    assert len(body['entries'])==600 and len(selected)==120
    assert set(selected)=={shared.identity(u) for u in target}
    assert campaign.read(path)==body


@pytest.mark.parametrize('change',['missing_source','extra_source','target_repeat','target_other_scene','target_plan'])
def test_manifest_and_target_cannot_choose_arbitrary_successful_subset(tmp_path,monkeypatch,change):
    shared,c,plan,path,body,checked=manifest_fixture(tmp_path,monkeypatch)
    target=campaign.planned_units(c,plan,'formal');supplied_plan=plan
    if change=='missing_source':body['entries'].pop()
    elif change=='extra_source':body['entries'].append(dict(body['entries'][0],seed=123))
    elif change=='target_repeat':target[0]['repeat']=2
    elif change=='target_other_scene':target[0]['seed']=plan['test'][20]['seed']
    elif change=='target_plan':supplied_plan=copy.deepcopy(plan);supplied_plan['test'][-1]['seed']+=1
    campaign.atomic(path,body)
    with pytest.raises(ValueError,match='allocation|plan|subset'):
        shared.validate_initial_manifest(path,c,target,phase='formal',plan=supplied_plan)


def test_unavailable_nonselected_source_is_preserved_not_used_to_select(tmp_path,monkeypatch):
    shared,c,plan,path,body,checked=manifest_fixture(tmp_path,monkeypatch)
    unselected=body['entries'][-1];unselected.update(available=False,reason='environment did not initialize')
    campaign.atomic(path,body)
    selected=shared.validate_initial_manifest(path,c,phase='formal',plan=plan)
    assert len(selected)==120 and shared.identity(unselected) not in selected
    assert campaign.read(path)['entries'][-1]['available'] is False
    body['entries'][0].update(available=False,reason='environment did not initialize');campaign.atomic(path,body)
    with pytest.raises(ValueError,match='unavailable'):
        shared.validate_initial_manifest(path,c,phase='formal',plan=plan)


def test_pair_reference_outside_preregistered_subset_rejected(tmp_path,monkeypatch):
    shared,c,plan,path,body,checked=manifest_fixture(tmp_path,monkeypatch)
    entry=body['entries'][-1]
    with pytest.raises(ValueError,match='subset'):
        shared.load_initial_reference(entry,c,task=entry['task'],seed=entry['seed'],repeat=entry['repeat'],fault='none',snapshot={})
    assert not checked
