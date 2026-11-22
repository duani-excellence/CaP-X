"""Scene-clustered analysis of frozen B0/G0/G1 trials; never imports a simulator.

The unit resampled is (task, seed): both model draws and every system stay in
that cluster. Tasks receive equal weight. Failed/aborted executed episodes are
zero successes; unexecuted episodes stay missing, never fabricated successes or
failures. Stress settings are separate estimands.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

GROUPS = ('B0','G0','G1')


def key(row):
    return row['task'], int(row['seed']), int(row.get('repeat',0)), row.get('fault','none'), row.get('ablation','full'), row['group']


def validate(rows):
    seen=set()
    for row in rows:
        k=key(row)
        if k in seen:raise ValueError('Duplicate episode identity: '+str(k))
        seen.add(k)
        if row['group'] not in GROUPS:raise ValueError('Unknown group')
    return rows


def unprepared(row):
    return row.get('executed') is False and row.get('status')=='scene_preparation_error'


def success(row):
    # Original reward is retained even when policy completion was too late or
    # terminal. Old valid records do not need newly introduced policy fields.
    if row.get('reward')!=1 or row.get('status')!='success':return 0
    if row.get('policy_success') is False or row.get('truncated') or row.get('terminal_reason'):return 0
    for state in (row.get('execution_status'),(row.get('info') or {}).get('execution_status')):
        if state is not None and state!='ok':return 0
    reason=str(row.get('stop_reason') or '')
    if (reason.endswith('_budget') or 'budget_exhausted' in reason or reason in {
            'environment_timeout_or_truncation','interrupted_by_operator','unknown_physical_effect_terminal_stop',
            'hold_evidence_unrecoverable','held_surface_evidence_unrecoverable','model_finish_claim_without_reward_change',
            'inconsistent_or_non_ok_environment_completion','global_model_or_episode_wall_budget',
            'resource_budget_exhausted'}):return 0
    return 1




def common_eligible_success(row):
    """Common task-success eligibility, without the extra public phase gate.

    It never turns physical attainment during a controlled/budget exit into a
    task success. This is an endpoint rule, not a counterfactual rerun of B0.
    """
    if row.get('status') not in ('success','task_failure'):return 0
    state=row.get('execution_status',(row.get('info') or {}).get('execution_status'))
    if state is None and row.get('status')!='success':return 0
    checks=row.get('policy_checks') or {}
    if any(value is False for key,value in checks.items() if key!='public_stage_complete'):return 0
    return success(dict(row,status='success',policy_success=True))


def physical_success(row):
    """Independent evaluator outcome, without changing execution eligibility.

    Original reward includes guards in some tasks, so preserve its independent
    physical flag when available. Missing measurements remain unknown.
    """
    info=row.get('info') or {}
    value=info.get('physical_success') if 'physical_success' in info else row.get('reward')
    if value is None or value not in (False,True,0,1):return None
    return int(bool(value))


def quantile(values,p):
    values=sorted(values);position=(len(values)-1)*p;lo=int(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def paired_summary(rows, treatment, control, *, n_bootstrap=2000, random_seed=0, expected_repeats=None, outcome='policy_success'):
    if n_bootstrap<1:raise ValueError('n_bootstrap must be positive')
    validate(rows)
    judges={'policy_success':success,'common_eligible_success':common_eligible_success,'original_physical_attainment':physical_success}
    if outcome not in judges:raise ValueError('Unknown outcome')
    judge=judges[outcome]
    if len({(r.get('fault','none'),r.get('ablation','full')) for r in rows})>1:raise ValueError('Analyze fault/ablation strata separately')
    pairs=defaultdict(dict)
    for row in rows:
        if row['group'] in (treatment,control):pairs[key(row)[:-1]][row['group']]=row
    if expected_repeats is not None:
        if not isinstance(expected_repeats,int) or expected_repeats<1:raise ValueError('expected_repeats must be positive')
        for task,seed,_,fault,ablation in list(pairs):
            for repeat in range(expected_repeats):pairs.setdefault((task,seed,repeat,fault,ablation),{})
    scene_states=defaultdict(list);missing_hashes=[];incomplete_identities=[];verified=0
    differences=defaultdict(lambda:defaultdict(list));wins=losses=ties=incomplete=0
    paired_metrics=defaultdict(list)
    mismatches=[]
    for k,pair in sorted(pairs.items()):
        if treatment not in pair or control not in pair or any(unprepared(r) for r in pair.values()):
            incomplete+=1;scene_states[k[:2]].append(False)
            incomplete_identities.append(dict(task=k[0],seed=k[1],repeat=k[2],fault=k[3],ablation=k[4],
                missing_groups=[g for g in (treatment,control) if g not in pair],unprepared_groups=[g for g,r in pair.items() if unprepared(r)]))
            continue
        a,b=pair[treatment],pair[control]
        if judge(a) is None or judge(b) is None:
            incomplete+=1;scene_states[k[:2]].append(False)
            incomplete_identities.append(dict(task=k[0],seed=k[1],repeat=k[2],fault=k[3],ablation=k[4],reason='unknown_physical_outcome'))
            continue
        scene_states[k[:2]].append(True);verified_pair=True
        if treatment=='G1' and control=='G0':
            # Explicit missing hashes are reported; mismatch is never pooled silently.
            for field in ('snapshot_hash','initial_prompt_hash','api_docs_hash','code_hash'):
                if not a.get(field) or not b.get(field):
                    verified_pair=False;missing_hashes.append(dict(identity=k,field=field))
                elif a[field]!=b[field]:mismatches.append(dict(identity=k,field=field))
        elif treatment=='G1' and control=='B0':
            # B0 has its own legal API/program; only the physical start is shared.
            if not a.get('snapshot_hash') or not b.get('snapshot_hash'):
                verified_pair=False;missing_hashes.append(dict(identity=k,field='snapshot_hash'))
            elif a['snapshot_hash']!=b['snapshot_hash']:mismatches.append(dict(identity=k,field='snapshot_hash'))
        verified+=verified_pair
        d=judge(a)-judge(b);differences[k[0]][k[1]].append(d)
        wins+=d>0;losses+=d<0;ties+=d==0
        for field in ('actual_api_calls','replans','total_seconds','end_to_end_seconds','input_tokens','output_tokens'):
            if isinstance(a.get(field),(float,int)) and isinstance(b.get(field),(float,int)):
                paired_metrics[field].append(a[field]-b[field])
    if mismatches:raise ValueError('Paired hash mismatch: '+str(mismatches[:3]))
    by_task={task:{seed:sum(ds)/len(ds) for seed,ds in clusters.items()} for task,clusters in differences.items()}
    task_effects={t:sum(v.values())/len(v) for t,v in by_task.items()}
    estimate=sum(task_effects.values())/len(task_effects) if task_effects else None
    rng=random.Random(random_seed);bootstrap=[]
    for _ in range(n_bootstrap if task_effects else 0):
        sampled=[]
        for clusters in by_task.values():
            values=list(clusters.values());sampled.append(sum(rng.choices(values,k=len(values)))/len(values))
        bootstrap.append(sum(sampled)/len(sampled))
    return dict(treatment=treatment,control=control,outcome=outcome,is_task_success_metric=outcome!='original_physical_attainment',
                interpretation_warning='Original physical state attainment only, including controlled/budget exits; this is not task success' if outcome=='original_physical_attainment' else None,independent_scenarios=sum(len(v) for v in by_task.values()),
                paired_generations=wins+losses+ties,incomplete_generations=incomplete,
                complete_scenarios=sum(all(flags) for flags in scene_states.values()),
                partial_scenarios=sum(any(flags) and not all(flags) for flags in scene_states.values()),
                unpaired_scenarios=sum(not any(flags) for flags in scene_states.values()),
                incomplete_identities=incomplete_identities,missing_pair_hashes=missing_hashes,
                pairing_verified_generations=verified,formal_estimate_ready=bool(task_effects) and not incomplete and not missing_hashes and all(len(v)>=20 for v in by_task.values()),
                missingness_warning='Available-pair descriptive estimate; incomplete scene draws or missing pairing provenance prevent a formal conclusion' if incomplete or missing_hashes else None,
                generation_wins=wins,generation_losses=losses,generation_ties=ties,
                scenario_wins=sum(v>0 for ss in by_task.values() for v in ss.values()),
                scenario_losses=sum(v<0 for ss in by_task.values() for v in ss.values()),
                scenario_ties=sum(v==0 for ss in by_task.values() for v in ss.values()),
                macro_success_difference=estimate,ci95=[quantile(bootstrap,.025),quantile(bootstrap,.975)] if bootstrap else None,
                tasks={t:dict(scenarios=len(by_task[t]),success_difference=effect) for t,effect in task_effects.items()},
                paired_mean_differences={k:sum(v)/len(v) for k,v in paired_metrics.items()},
                bootstrap_samples=n_bootstrap,bootstrap_seed=random_seed,
                inference_warning='Fewer than 20 independent scenes in at least one task; exploratory interval only' if any(len(v)<20 for v in by_task.values()) else None)


def summarize_interventions(rows):
    """Role counts are observational associations, never causal repair success."""
    known=[r for r in rows if isinstance(r.get('interventions'),list)]
    summary={}
    for role in ('physical_correction','evidence_acquisition','llm_replan'):
        episodes=[r for r in known if any(i.get('role')==role for i in r['interventions'])]
        attempts=[i for r in episodes for i in r['interventions'] if i.get('role')==role]
        final_successes=sum(success(r) for r in episodes)
        summary[role]=dict(attempts=len(attempts),episodes=len(episodes),
            final_task_successes=final_successes,final_task_success_rate=final_successes/len(episodes) if episodes else None,
            final_physical_task_successes=sum(bool((r.get('info') or {}).get('physical_success',r.get('reward')==1)) for r in episodes),
            detector_passes=sum(i.get('detector_passed') is True for i in attempts),
            detector_clearances=sum(i.get('detector_cleared') is True for i in attempts),
            detector_outcome_unknown=sum(i.get('detector_passed') is None for i in attempts),
            physical_risk_clearances=sum(i.get('physical_risk_cleared') is True for i in attempts),
            physical_risk_unknown=sum(i.get('physical_risk_cleared') is None for i in attempts),
            physical_error_improvements=sum(i.get('physical_error_improved') is True for i in attempts),
            normal_verification_attempts=sum(i.get('trigger')=='normal_verification' for i in attempts),
            failure_triggered_attempts=sum(i.get('trigger')=='failed_or_uncertain_checkpoint' for i in attempts),
            trigger_unknown_attempts=sum(i.get('trigger')=='unknown' for i in attempts))
    return summary,dict(known_episodes=len(known),missing_episodes=len(rows)-len(known))


def summarize_group(rows):
    recorded=list(rows);rows=[r for r in recorded if not unprepared(r)]
    interventions,intervention_coverage=summarize_interventions(rows)
    n=len(rows);recoveries=[r for r in rows if r.get('replans',0)>0 or r.get('local_recoveries',0)>0]
    physical=[physical_success(r) for r in rows];known_physical=[x for x in physical if x is not None]
    detection=Counter()
    for row in rows:
        for k,v in (row.get('detection') or {}).items():
            if k in ('true_allow','false_allow','true_block','false_block','unknown') and isinstance(v,int) and v>=0:detection[k]+=v
    allow=detection['true_allow']+detection['false_allow'];block=detection['true_block']+detection['false_block']
    totals={}
    for field in ('actual_api_calls','replans','total_seconds','end_to_end_seconds','input_tokens','output_tokens','reasoning_tokens','llm_requests','cost_usd','invalid_recovery_actions'):
        values=[r[field] for r in rows if isinstance(r.get(field),(int,float))]
        totals[field]=dict(known=len(values),missing=n-len(values),total=sum(values),mean=sum(values)/len(values) if values else None)
    return dict(interventions=interventions,intervention_coverage=intervention_coverage,
                completion_contracts=dict(Counter(r.get('completion_contract','legacy_unspecified') for r in rows)),
                common_eligible_successes=sum(common_eligible_success(r) for r in rows),
                common_eligible_success_rate=sum(common_eligible_success(r) for r in rows)/n if n else None,
                physical_outcome=dict(known=len(known_physical),unknown=n-len(known_physical),attained=sum(known_physical),
                    attainment_rate=sum(known_physical)/len(known_physical) if known_physical else None,is_task_success_metric=False,
                    original_reward_positive=sum(r.get('reward')==1 for r in rows),
                    definition='info.physical_success when present, otherwise original reward; controlled exits stay separate policy failures'),
                recorded_episodes=len(recorded),unexecuted_episodes=len(recorded)-n,attempted_episodes=n,successes=sum(success(r) for r in rows),success_rate=sum(success(r) for r in rows)/n if n else None,
                statuses=dict(Counter(r.get('status','unknown') for r in recorded)),
                stop_reasons=dict(Counter(r.get('stop_reason','unknown') for r in recorded)),
                recovery_attempted_episodes=len(recoveries),recovery_successes=sum(success(r) for r in recoveries),
                recovery_success_rate=sum(success(r) for r in recoveries)/len(recoveries) if recoveries else None,
                normal_successes=sum(success(r) for r in rows if r not in recoveries),
                detection=dict(detection,labeled_checks=allow+block,false_allow_rate=detection['false_allow']/allow if allow else None,
                               false_block_rate=detection['false_block']/block if block else None),metrics=totals)


def analyze(rows, *, n_bootstrap=2000, random_seed=0, expected_repeats=None):
    if n_bootstrap<1:raise ValueError('n_bootstrap must be positive')
    rows=validate(list(rows));strata={}
    variants=sorted({(r.get('ablation','full'),r.get('fault','none')) for r in rows})
    plain=all(a=='full' for a,_ in variants)
    for ablation,fault in variants:
        subset=[r for r in rows if r.get('fault','none')==fault and r.get('ablation','full')==ablation]
        label=fault if plain else ablation+':'+fault
        strata[label]=dict(ablation=ablation,fault=fault,groups={g:summarize_group([r for r in subset if r['group']==g]) for g in GROUPS},
            G1_vs_G0=paired_summary(subset,'G1','G0',n_bootstrap=n_bootstrap,random_seed=random_seed,expected_repeats=expected_repeats),
            G1_vs_G0_physical=paired_summary(subset,'G1','G0',n_bootstrap=n_bootstrap,random_seed=random_seed,expected_repeats=expected_repeats,outcome='original_physical_attainment'),
            G1_vs_B0=paired_summary(subset,'G1','B0',n_bootstrap=n_bootstrap,random_seed=random_seed,expected_repeats=expected_repeats,outcome='common_eligible_success'),
            G1_vs_B0_physical=paired_summary(subset,'G1','B0',n_bootstrap=n_bootstrap,random_seed=random_seed,expected_repeats=expected_repeats,outcome='original_physical_attainment'),
            G1_vs_B0_policy=paired_summary(subset,'G1','B0',n_bootstrap=n_bootstrap,random_seed=random_seed,expected_repeats=expected_repeats))
        strata[label]['G1_vs_B0_policy'].update(completion_contract_comparable=False,formal_estimate_ready=False,
            interpretation_warning='Heterogeneous completion contracts: grounded public skill/retreat verification versus B0 unmonitored original API. Descriptive system completion only; not a same-contract causal effect.')
    return dict(schema='capx-graph-cluster-analysis-v2',episodes=len(rows),strata=strata,
        definitions={'primary':'G1_vs_G0 is policy success under the same grounded completion contract; G1_vs_G0_physical describes raw physical state attainment separately, never task success. Equal-task scene means; cluster percentile bootstrap resamples whole scenes within task',
        'baseline_comparison':'G1_vs_B0 uses common eligibility: original reward 1, clean return, no truncation/terminal/budget exit, excluding the extra public phase requirement. G1_vs_B0_policy is heterogeneous/descriptive. The two *_physical comparisons measure raw physical state attainment only, never task success',
        'success':'Original physical reward == 1 AND successful policy completion, with no explicit truncation, terminal exit or non-ok execution. New policy_success=False overrides reward; legacy valid records without the new field remain eligible. FINISH and budget exits are not success',
        'missing':'Explicit executed=False scene_preparation_error environment-setup failures remain recorded in statuses and unexecuted counts; excluded from attempted physical-effect rate and paired estimator. All attempted failures remain zero. Incomplete pairs are listed, not imputed or silently retried',
        'recovery':'Legacy broad count: at least one recorded replan/local recovery, including active visibility sensing; not a physical-repair count. Final policy success requires original physical success'
        ,'interventions':'Separate physical_correction/evidence_acquisition/llm_replan attempts and episode outcomes. Initial verify_hold is normal verification unless preceded by an actual failed/uncertain hold checkpoint. Detector clearance, independent physical risk clearance and final task success are separate; temporal association is not causal attribution. Missing legacy role evidence stays missing',
        'detection':'Only independently offline-labeled decisions. No inference from final reward; unknown evidence stays unknown',
        'false_allow_rate':'false_allow / (true_allow + false_allow); precision-style erroneous-release fraction, not FPR',
        'false_block_rate':'false_block / (true_block + false_block); precision-style erroneous-block fraction, not FNR',
        'time':'Use end_to_end_seconds for total elapsed cost; policy_elapsed_seconds stops after evaluation/verification and before final video/artifact cleanup. Shared initial latency is charged to each policy, billed once globally'})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('records',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=0);a=p.parse_args(argv)
    rows=[json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
    result=analyze(rows,n_bootstrap=a.bootstrap,random_seed=a.seed)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
