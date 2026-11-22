import pytest
from capx.experiments.graph_power import design_from_development


def rows(n=20):
    result=[]
    for task in ('stack','lift','restack'):
        for seed in range(n):
            for repeat in range(2):
                for group in ('B0','G0','G1'):
                    result.append(dict(task=task,seed=seed,repeat=repeat,group=group,fault='none',reward=int(seed%2==0),status='success' if seed%2==0 else 'task_failure',total_seconds=10,initial_generated_here=group!='G1',new_llm_requests=1 if group!='G1' else 0))
    return result


def test_repeats_do_not_double_independent_scenes():
    result=design_from_development(rows(),minimum_scenes=100)
    assert all(x['scenes']==20 for x in result['development'].values())
    assert result['formal_scenes_per_task']>=100
    assert result['planned_episodes']==result['formal_scenes_per_task']*3*2*3
    assert result['max_requests']==result['formal_scenes_per_task']*3*2*(2+3*5)


def test_no_variance_does_not_claim_perfect_power():
    result=design_from_development(rows())
    assert result['variance_floor_used']
    assert result['precision_warning']


def test_requires_all_three_groups_and_twenty_independent_scenes():
    with pytest.raises(ValueError,match='20'):design_from_development(rows(19))
    with pytest.raises(ValueError,match='incomplete'):design_from_development([x for x in rows() if x['group']!='B0'])


def test_stress_data_cannot_select_normal_sample_size():
    rr=rows();rr[0]['fault']='target_shift'
    with pytest.raises(ValueError,match='normal'):design_from_development(rr)


def test_invalid_effect_and_duplicate_records_rejected():
    with pytest.raises(ValueError):design_from_development(rows(),detectable_effect=0)
    rr=rows();rr.append(dict(rr[0]))
    with pytest.raises(ValueError,match='Duplicate'):design_from_development(rr)


def test_unprepared_scene_cannot_determine_formal_sample_size():
    rr=rows();rr[0].update(executed=False,status='scene_preparation_error')
    with pytest.raises(ValueError,match='unprepared'):design_from_development(rr)


def test_missing_repeat_cannot_silently_change_scene_variance():
    rr=[r for r in rows() if not (r['task']=='stack' and r['seed']==0 and r['repeat']==1)]
    with pytest.raises(ValueError,match='repeat'):design_from_development(rr)
