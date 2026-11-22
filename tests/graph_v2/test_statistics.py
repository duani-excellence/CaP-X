import pytest
from capx.experiments.graph_statistics import analyze, paired_summary


def row(task, seed, repeat, group, success, **extra):
    return dict(task=task, seed=seed, repeat=repeat, group=group, fault='none', reward=float(success), status='success' if success else 'task_failure', **extra)


def test_cluster_is_scene_not_generation_and_task_macro_is_equal_weight():
    rows=[]
    for seed in range(10):
        for repeat in range(2):
            rows += [row('stack', seed, repeat, 'G0', False),row('stack', seed, repeat, 'G1', True)]
    for repeat in range(2):
        rows += [row('lift', 50, repeat, 'G0', True),row('lift', 50, repeat, 'G1', False)]
    out=paired_summary(rows, 'G1', 'G0', n_bootstrap=99)
    assert out['independent_scenarios']==11
    assert out['paired_generations']==22
    assert out['macro_success_difference']==0
    assert out['ci95']==[0,0]
    assert out['generation_wins']==20 and out['generation_losses']==2


def test_two_repeats_never_add_independent_samples():
    rows=[row('stack', 1, r, g, g=='G1') for r in range(2) for g in ('G0','G1')]
    out=paired_summary(rows,'G1','G0',n_bootstrap=19)
    assert out['independent_scenarios']==1
    assert out['inference_warning']


def test_failures_and_budget_exhaustion_count_not_dropped():
    rows=[row('stack',1,0,'G0',False),row('stack',1,0,'G1',True)]
    rows[0].update(reward=None,status='system_error')
    out=analyze(rows,n_bootstrap=19)
    assert out['strata']['none']['groups']['G0']['attempted_episodes']==1
    assert out['strata']['none']['groups']['G0']['success_rate']==0
    assert out['strata']['none']['G1_vs_G0']['macro_success_difference']==1


def test_stress_is_not_pooled_with_normal():
    rows=[row('stack',1,0,g,g=='G1') for g in ('G0','G1')]
    rows += [dict(row('stack',2,0,g,g=='G0'),fault='support_refresh') for g in ('G0','G1')]
    out=analyze(rows,n_bootstrap=19)
    assert out['strata']['none']['G1_vs_G0']['macro_success_difference']==1
    assert out['strata']['support_refresh']['G1_vs_G0']['macro_success_difference']==-1


def test_duplicate_episode_rejected():
    r=row('stack',1,0,'G0',False)
    with pytest.raises(ValueError,match='Duplicate'):
        analyze([r,r],n_bootstrap=19)


def test_incomplete_pairs_reported_not_synthesized():
    out=paired_summary([row('stack',1,0,'G1',True)],'G1','G0',n_bootstrap=19)
    assert out['paired_generations']==0 and out['incomplete_generations']==1
    assert out['macro_success_difference'] is None


def test_bootstrap_reproducible_and_observed_detection_only():
    rows=[row('stack',s,r,g,(s+r+(g=='G1'))%3==0) for s in range(10) for r in range(2) for g in ('G0','G1')]
    a=analyze(rows,n_bootstrap=99,random_seed=7)
    assert a==analyze(rows,n_bootstrap=99,random_seed=7)
    assert a['strata']['none']['groups']['G1']['detection']['labeled_checks']==0
    assert a['strata']['none']['groups']['G1']['detection']['false_allow_rate'] is None


def test_zero_bootstraps_invalid():
    with pytest.raises(ValueError):analyze([],n_bootstrap=0)


def test_hash_mismatch_is_fatal_to_pairing():
    rows=[row('stack',1,0,g,True,snapshot_hash=g) for g in ('G0','G1')]
    with pytest.raises(ValueError,match='hash mismatch'):analyze(rows,n_bootstrap=19)


def test_detection_rates_do_not_infer_truth_from_reward():
    rows=[row('stack',1,0,'G1',False,detection={'true_allow':3,'false_allow':1,'true_block':4,'false_block':2,'unknown':9})]
    d=analyze(rows,n_bootstrap=19)['strata']['none']['groups']['G1']['detection']
    assert d['labeled_checks']==10 and d['unknown']==9
    assert d['false_allow_rate']==.25 and d['false_block_rate']==pytest.approx(1/3)


def test_ablations_cannot_be_pooled_even_with_same_task_seed_fault():
    rows=[dict(row('stack',1,0,g,g=='G1'),ablation=abl) for abl in ('full','no_diagnostic_slice') for g in ('G0','G1')]
    result=analyze(rows,n_bootstrap=19)
    assert set(result['strata'])=={'full:none','no_diagnostic_slice:none'}
    assert all(v['G1_vs_G0']['paired_generations']==1 for v in result['strata'].values())


def test_unexecuted_scene_setup_kept_as_missing_not_physical_failure():
    rows=[row('stack',1,0,g,True) for g in ('G0','G1')]
    rows += [dict(row('stack',2,0,g,False),executed=False,status='scene_preparation_error') for g in ('G0','G1')]
    out=analyze(rows,n_bootstrap=19)['strata']['none']
    assert out['groups']['G0']['recorded_episodes']==2
    assert out['groups']['G0']['attempted_episodes']==1
    assert out['groups']['G0']['unexecuted_episodes']==1
    assert out['groups']['G0']['success_rate']==1
    assert out['groups']['G0']['statuses']['scene_preparation_error']==1
    assert out['G1_vs_G0']['paired_generations']==1
    assert out['G1_vs_G0']['incomplete_generations']==1


def test_direct_pair_summary_rejects_mixed_fault_or_ablation_estimands():
    rows=[row('stack',1,0,g,True) for g in ('G0','G1')]
    rows += [dict(row('stack',2,0,g,True),fault='support_refresh') for g in ('G0','G1')]
    with pytest.raises(ValueError,match='strata'):paired_summary(rows,'G1','G0',n_bootstrap=19)


def test_model_generation_failure_without_motion_is_still_attempted_failure():
    rows=[row('stack',1,0,'G0',True),dict(row('stack',1,0,'G1',False),executed=False,status='invalid_program')]
    out=analyze(rows,n_bootstrap=19)['strata']['none']
    assert out['groups']['G1']['attempted_episodes']==1 and out['groups']['G1']['success_rate']==0
    assert out['G1_vs_G0']['paired_generations']==1 and out['G1_vs_G0']['macro_success_difference']==-1


def test_partial_scene_and_missing_pair_hashes_are_explicit():
    rows=[row('stack',1,0,g,g=='G1') for g in ('G0','G1')]
    rows += [row('stack',1,1,'G1',False)]
    out=paired_summary(rows,'G1','G0',n_bootstrap=19)
    assert out['partial_scenarios']==1 and out['complete_scenarios']==0
    assert out['pairing_verified_generations']==0
    assert {x['field'] for x in out['missing_pair_hashes']}=={'snapshot_hash','initial_prompt_hash','api_docs_hash','code_hash'}
    assert out['formal_estimate_ready'] is False
    assert out['incomplete_identities'][0]['repeat']==1


def test_expected_repeats_detects_absent_whole_generation():
    rows=[row('stack',1,0,g,g=='G1',snapshot_hash='s',initial_prompt_hash='p',api_docs_hash='a',code_hash='c') for g in ('G0','G1')]
    out=paired_summary(rows,'G1','G0',n_bootstrap=19,expected_repeats=2)
    assert out['incomplete_generations']==1 and out['partial_scenarios']==1
    assert out['pairing_verified_generations']==1
    assert out['formal_estimate_ready'] is False


def test_controlled_exit_with_stale_positive_reward_is_not_success():
    rows=[dict(row('stack',1,0,'G1',True),status='interrupted_by_operator')]
    assert analyze(rows,n_bootstrap=19)['strata']['none']['groups']['G1']['successes']==0
