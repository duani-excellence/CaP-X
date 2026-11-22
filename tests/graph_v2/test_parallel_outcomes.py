"""Identical physical criteria stay separate from heterogeneous completion contracts."""
import json
import pytest
from capx.experiments import graph_statistics as stats


def rows(seeds=1,repeats=2):
    return [dict(task='stack',seed=s,repeat=r,group=g,fault='none',ablation='full',
            reward=1.,info={'physical_success':True,'execution_status':'ok'},
            status='controlled_exit' if g=='G1' else 'success',policy_success=g!='G1',
            completion_contract='unmonitored_original_api' if g=='B0' else 'grounded_public_skill_phase',
            snapshot_hash=f'snapshot-{s}',initial_prompt_hash='grounded' if g!='B0' else 'raw',
            api_docs_hash='grounded' if g!='B0' else 'raw',code_hash=f'program-{s}-{r}' if g!='B0' else 'b0')
            for s in range(seeds) for r in range(repeats) for g in ('B0','G0','G1')]


def test_physical_success_remains_one_when_policy_stops():
    out=stats.analyze(rows(),n_bootstrap=19,expected_repeats=2)['strata']['none']
    assert out['groups']['G1']['success_rate']==0
    physical=out['groups']['G1']['physical_outcome']
    assert physical['attained']==2 and physical['attainment_rate']==1 and physical['unknown']==0
    assert physical['original_reward_positive']==2


def test_primary_graph_policy_and_parallel_physics_use_same_scene_clusters():
    out=stats.analyze(rows(20),n_bootstrap=99,expected_repeats=2)['strata']['none']
    p=out['G1_vs_G0'];raw=out['G1_vs_G0_physical']
    assert p['macro_success_difference']==-1 and p['ci95']==[-1,-1]
    assert raw['macro_success_difference']==0 and raw['ci95']==[0,0]
    assert p['outcome']=='policy_success' and raw['outcome']=='original_physical_attainment'
    assert p['independent_scenarios']==raw['independent_scenarios']==20
    assert p['paired_generations']==raw['paired_generations']==40


def test_b0_primary_uses_same_physical_criterion_and_policy_is_heterogeneous():
    out=stats.analyze(rows(20),n_bootstrap=29,expected_repeats=2)['strata']['none']
    assert out['G1_vs_B0']['outcome']=='common_eligible_success'
    assert out['G1_vs_B0']['macro_success_difference']==-1
    assert out['G1_vs_B0_physical']['macro_success_difference']==0
    assert out['G1_vs_B0_physical']['is_task_success_metric'] is False
    other=out['G1_vs_B0_policy']
    assert other['macro_success_difference']==-1
    assert other['completion_contract_comparable'] is False
    assert other['formal_estimate_ready'] is False
    assert 'heterogeneous' in other['interpretation_warning'].lower()


def test_physical_flag_overrides_guarded_reward_without_rewriting_it():
    values=rows();target=values[-1];target.update(reward=0.,status='interrupted_by_operator')
    out=stats.analyze(values,n_bootstrap=19)['strata']['none']['groups']['G1']
    assert out['physical_outcome']['attained']==2
    assert out['physical_outcome']['original_reward_positive']==1
    assert out['successes']==0 and target['reward']==0


def test_unknown_physical_outcome_does_not_become_zero_or_formal_evidence():
    values=rows();values[-1].update(reward=None,info={},status='system_error')
    out=stats.analyze(values,n_bootstrap=19,expected_repeats=2)['strata']['none']
    assert out['groups']['G1']['physical_outcome']['unknown']==1
    assert out['G1_vs_G0_physical']['incomplete_generations']==1
    assert out['G1_vs_G0_physical']['formal_estimate_ready'] is False


def test_offline_entry_retains_completion_contract(tmp_path):
    from capx.experiments.graph_v2_analysis import analyze_episode
    value=rows()[0];path=tmp_path/'result.json';path.write_text(json.dumps(value))
    out=analyze_episode(path)
    assert out['completion_contract']=='unmonitored_original_api'
    assert out['physical_task_success'] is True


def test_common_eligibility_ignores_only_public_completion_gate():
    values=rows();target=values[-1]
    target.update(status='task_failure',policy_checks={'original_task_success':True,'execution_ok':True,'not_truncated':True,'no_terminal_exit':True,'within_wall_budget':True,'public_stage_complete':False})
    assert stats.success(target)==0 and stats.common_eligible_success(target)==1
    for field in ('within_wall_budget','no_terminal_exit','execution_ok'):
        modified=dict(target,policy_checks=dict(target['policy_checks'],**{field:False}))
        assert stats.common_eligible_success(modified)==0


@pytest.mark.parametrize('status,reason',[('controlled_exit','held_surface_evidence_unrecoverable'),('budget_exceeded','episode_or_global_wall_budget'),('task_failure','episode_code_block_budget'),('interrupted_by_operator','interrupted_by_operator')])
def test_common_eligibility_never_counts_controlled_or_budget_exit(status,reason):
    target=rows()[-1];target.update(status=status,stop_reason=reason)
    assert stats.physical_success(target)==1
    assert stats.common_eligible_success(target)==0
