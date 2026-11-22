"""Development-only scene-cluster power/budget planning; no provider imports.

Normal approximation is a planning heuristic, not a reported treatment effect.
Formal inference remains the separately frozen scene-cluster bootstrap. A positive
variance floor avoids declaring certain power after an all-tie development sample.
"""
import argparse
import hashlib
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, variance
from capx.experiments.graph_statistics import validate, success


def design_from_development(rows, *, minimum_scenes=100, repeats=2, max_blocks=6,
                            detectable_effect=.10, alpha=.05, power=.8, variance_floor=.10, runtime_budgets=None):
    if runtime_budgets is not None and runtime_budgets.get('max_blocks',max_blocks)!=max_blocks:
        raise ValueError('max_blocks differs from runtime configuration')
    rows=validate(list(rows))
    if not rows or any(r.get('fault','none')!='none' or r.get('ablation','full')!='full' for r in rows):
        raise ValueError('Use normal full-system development data only')
    if not 0<detectable_effect<1 or not 0<alpha<1 or not .5<power<1 or minimum_scenes<20 or repeats<1 or max_blocks<1:
        raise ValueError('Invalid design parameters')
    if any(r.get('executed') is False and r.get('status')=='scene_preparation_error' for r in rows):raise ValueError('Exclude unprepared scenes from development design; complete replacement development scenes first')
    pairs=defaultdict(dict)
    for row in rows:pairs[(row['task'],row['seed'],row.get('repeat',0))][row['group']]=row
    if any(set(p)!=set(('B0','G0','G1')) for p in pairs.values()):raise ValueError('incomplete development pairing')
    observed=defaultdict(set)
    for task,seed,repeat in pairs:observed[(task,seed)].add(repeat)
    if any(value!=set(range(repeats)) for value in observed.values()):raise ValueError('Development repeats must match planned repeats in every scene')
    clusters=defaultdict(lambda:defaultdict(list));units=[]
    for (task,seed,repeat),pair in pairs.items():
        clusters[task][seed].append(success(pair['G1'])-success(pair['G0']))
        units.append(sum(float(r.get('total_seconds',0)) for r in pair.values()))
    if len(clusters)<3 or any(len(s)<20 for s in clusters.values()):raise ValueError('At least 3 tasks and 20 independent development scenes per task required')
    development={};used=False;variances=[]
    for task,scenes in clusters.items():
        values=[mean(ds) for ds in scenes.values()];raw=variance(values);v=max(raw,variance_floor);used|=v!=raw;variances.append(v)
        development[task]=dict(scenes=len(scenes),generations=sum(len(ds) for ds in scenes.values()),
            mean_paired_difference=mean(values),observed_scene_variance=raw,planning_scene_variance=v)
    z=NormalDist().inv_cdf(1-alpha/2)+NormalDist().inv_cdf(power)
    # Equal n per task: Var(equal-task macro effect)=sum(sigma_t^2)/(T^2*n).
    required=math.ceil(z*z*sum(variances)/(len(clusters)**2*detectable_effect**2))
    n=max(minimum_scenes,required);unit_count=n*len(clusters)*repeats
    sorted_time=sorted(units);p95=sorted_time[min(len(units)-1,math.ceil(.95*len(units))-1)]
    resources=resource_plan(unit_count,max_blocks,runtime_budgets or {})
    return dict(schema='capx-development-power-plan-v2',development=development,
        detectable_effect=detectable_effect,alpha=alpha,power=power,formal_scenes_per_task=n,
        minimum_scenes=minimum_scenes,repeats=repeats,tasks=len(clusters),planned_units=unit_count,
        planned_episodes=unit_count*3,max_blocks=max_blocks,max_requests=resources['max_requests'],resources=resources,
        mean_unit_seconds=mean(units),p95_unit_seconds=p95,
        expected_serial_seconds=mean(units)*unit_count,wall_budget_seconds=math.ceil(max(p95*2,mean(units)*2)*unit_count),
        variance_floor=variance_floor,variance_floor_used=used,
        precision_warning='Planning only: 20-scene variance is uncertain; no power guarantee. Floor protects all-tie samples. Report achieved CI even if effect is null.',
        accounting='One unit = one independent initial generation pair + B0; G0/G1 initial request billed once. All failures included; no selection by success.')


def resource_plan(units,max_blocks,budgets):
    """Counting upper bounds, separate from empirical time and billed cost.

    No prompt-size ceiling is enforced by ModelBudget. Consequently a finite
    full input+output worst-case cost cannot be claimed from max_tokens alone.
    """
    episodes=units*3
    requests=units*(2+3*(max_blocks-1))  # two initial draws, three recovery chains
    for key in ('api_budget','episode_seconds','max_tokens','max_requests','cost_limit_usd','wall_seconds'):
        value=budgets.get(key)
        if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0):
            raise ValueError('Invalid runtime budget '+key)
    output_tokens=requests*budgets['max_tokens'] if 'max_tokens' in budgets else None
    output_cost=output_tokens*3.96/1e6 if output_tokens is not None else None
    runtime=episodes*budgets['episode_seconds'] if 'episode_seconds' in budgets else None
    physical=episodes*budgets['api_budget'] if 'api_budget' in budgets else None
    coverage={
        'requests_sufficient':budgets['max_requests']>=requests if 'max_requests' in budgets else None,
        'cost_covers_output_component':budgets['cost_limit_usd']>=output_cost if output_cost is not None and 'cost_limit_usd' in budgets else None,
        'wall_covers_episode_runtime_upper':budgets['wall_seconds']>=runtime if runtime is not None and 'wall_seconds' in budgets else None,
        'full_cost_sufficiency':'unknown_without_enforced_input_limit',
        'whole_campaign_wall_sufficiency':'unknown_setup_and_archiving_are_not_episode_bounded'}
    return dict(max_requests=requests,requests_per_unit=2+3*(max_blocks-1),
        max_physical_api_calls=physical,episode_runtime_upper_seconds=runtime,
        max_output_tokens=output_tokens,output_component_upper_usd=output_cost,
        full_cost_upper_usd=None,output_price_peak_usd_per_million=3.96,
        configured_limits=dict(budgets),coverage=coverage,
        scope='Normal B0/G0/G1 only. No automatic retry. G0/G1 initial generation counted once; every group may use all remaining blocks.',
        limitations=['Output component is a conservative peak-price allowance, not a total cost or actual invoice.',
            'Input prompt bytes are not globally capped; CostLedger also requires reservation headroom.',
            'Episode runtime excludes unbounded environment setup, source archiving and final video encoding.',
            'Physical API count excludes model and read-only graph calls; sandbox RPC has separate limits.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('records',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--effect',type=float,default=.10);p.add_argument('--minimum-scenes',type=int,default=100)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--config',type=Path)
    p.add_argument('--max-blocks',type=int);a=p.parse_args(argv)
    config=json.loads(a.config.read_text()) if a.config else {}
    budgets=config.get('config',config).get('budgets',{})
    configured=budgets.get('max_blocks')
    if configured is not None and a.max_blocks is not None and configured!=a.max_blocks:
        raise ValueError('max_blocks override differs from configuration')
    max_blocks=configured if configured is not None else a.max_blocks
    if max_blocks is None:raise ValueError('Provide --config or --max-blocks; no hidden block budget default')
    rows=[json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
    result=design_from_development(rows,detectable_effect=a.effect,minimum_scenes=a.minimum_scenes,repeats=a.repeats,max_blocks=max_blocks,runtime_budgets=budgets)
    result['planning_config_sha256']=hashlib.sha256(a.config.read_bytes()).hexdigest() if a.config else None
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
