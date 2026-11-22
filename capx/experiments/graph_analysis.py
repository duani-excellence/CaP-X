"""Read-only G0/G1 experiment accounting. Does not import simulation or providers.

Outputs are evidence summaries, not significance claims. Shared initial requests
are billed once in the observed request ledger and charged once per episode in
logical standalone-policy accounting. Costs are estimates, never an invoice.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

PAIR_FIELDS = ('snapshot_hash', 'initial_prompt_hash', 'api_docs_hash', 'code_hash')
CONFIG_FIELDS = ('max_tokens', 'max_blocks', 'api_budget', 'max_local', 'request_timeout',
                 'reasoning_effort', 'source_hashes')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path, warnings):
    if not path or not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1:
                warnings.append('Incomplete final JSONL row ignored: ' + str(path))
            else:
                raise ValueError('Malformed interior JSONL row: ' + str(path)) from None
    return rows


def request_row(path, run):
    raw = read_json(path, {})
    usage = raw.get('usage') or {}
    label = raw.get('label', path.parent.name)
    match = re.match(r'^replan-(G[01])-seed-(-?\d+)-fault-(.+)-(\d+)$', label)
    known = all(isinstance(usage.get(key), (int, float)) for key in ('prompt_tokens', 'completion_tokens'))
    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', usage.get('reasoning_tokens'))
    inp, out = (usage['prompt_tokens'], usage['completion_tokens']) if known else (None, None)
    hit = usage.get('prompt_cache_hit_tokens', 0) if known else None
    miss = usage.get('prompt_cache_miss_tokens', inp-hit) if known else None
    estimate = (hit*.044 + miss*1.32 + out*3.96)/1e6 if known else None
    return dict(path=str(path.resolve()), run=str(run), label=label,
                kind='recovery' if match else 'shared_initial' if label.startswith('initial-') else 'unassigned',
                owner_group=match.group(1) if match else None,
                owner_seed=int(match.group(2)) if match else None,
                owner_fault=match.group(3) if match else None,
                recovery_number=int(match.group(4)) if match else None,
                status=raw.get('status'), http_status=raw.get('http_status'),
                attempts=raw.get('attempts', 0), seconds=raw.get('seconds', 0),
                requested_model=raw.get('requested_model'), response_model=raw.get('response_model'),
                finish_reason=raw.get('finish_reason'), input_tokens=inp, output_tokens=out,
                reasoning_tokens=reasoning, cache_hit_tokens=hit, cache_miss_tokens=miss,
                cache_aware_peak_estimate_usd=estimate,
                recorded_cost_upper_usd=raw.get('cost_upper_usd'), unknown_usage=not known)


def request_totals(rows):
    return dict(requests=len(rows), http_attempts=sum(r.get('attempts', 0) or 0 for r in rows),
                input_tokens=sum(r.get('input_tokens') or 0 for r in rows),
                output_tokens=sum(r.get('output_tokens') or 0 for r in rows),
                reasoning_tokens=sum(r.get('reasoning_tokens') or 0 for r in rows),
                reasoning_usage_missing=sum(r.get('reasoning_tokens') is None for r in rows),
                seconds=sum(r.get('seconds') or 0 for r in rows),
                shared_initial_requests=sum(r['kind']=='shared_initial' for r in rows),
                recovery_requests=sum(r['kind']=='recovery' for r in rows),
                unknown_usage_requests=sum(r['unknown_usage'] for r in rows),
                cache_aware_peak_estimate_usd=sum(r.get('cache_aware_peak_estimate_usd') or 0 for r in rows),
                known_usage_off_peak_estimate_usd=sum(r.get('cache_aware_peak_estimate_usd') or 0 for r in rows)/2,
                unknown_usage_reserved_usd=sum(r.get('recorded_cost_upper_usd') or 0 for r in rows if r['unknown_usage']),
                missing_cost_bound_requests=sum(r.get('recorded_cost_upper_usd') is None and r['unknown_usage'] for r in rows),
                recorded_cost_upper_usd=sum(r.get('recorded_cost_upper_usd') or 0 for r in rows))


def initial_receipt(reference, root, requests):
    if not reference:
        return None
    directory = Path(reference)
    candidates = [directory/'receipt.json', root/'llm'/directory.name/'receipt.json', root/directory/'receipt.json']
    for path in candidates:
        canonical = str(path.resolve())
        if canonical in requests:
            return requests[canonical]
        if path.exists():
            # External cached initial requests contribute only logical accounting.
            return request_row(path, root)
    return None


def analyze_episode(path, root, requests, warnings):
    result = read_json(path)
    dest = path.parent
    reference = result.get('trace_path') or result.get('info', {}).get('trace_path')
    trace = Path(reference) if reference else None
    if trace is not None and not trace.exists():
        trace = None
    if trace is None:
        trace = next(iter(sorted(dest.glob('trace/**/trace.jsonl'))), None)
    rows = read_jsonl(trace, warnings)
    apis = [row for row in rows if row.get('kind')=='api']
    feedback = [row for row in rows if row.get('kind')=='feedback'] or result.get('feedback_events', [])
    events = read_jsonl(dest/'graph-events.jsonl', warnings) or result.get('graph_events', [])
    invalidations = [e for e in events if e.get('action')=='invalidate']
    changes = [e for e in invalidations if e.get('invalidated')]
    first_change = min((e.get('sequence', 0) for e in changes), default=float('inf'))
    passes_after = Counter()
    for event in events:
        node = event.get('node', {})
        data = node.get('data') or {}
        if event.get('action')=='add' and node.get('kind')=='check' and data.get('status')=='PASS' and event.get('sequence', 0)>first_change:
            passes_after[data.get('check', 'unknown')] += 1
    by_api = Counter(a.get('name', 'unknown') for a in apis)
    localization = Counter(str((a.get('args') or [a.get('kwargs', {}).get('object_name')])[0])
                           for a in apis if a.get('name')=='get_object_pose')
    signatures = Counter(json.dumps([a.get('name'), a.get('args'), a.get('kwargs')], sort_keys=True) for a in apis)
    tool_counts, tool_seconds, tool_errors = Counter(), Counter(), Counter()
    for tool_path in sorted((dest/'tools').glob('*.json')):
        tool = read_json(tool_path, {})
        name = tool_path.stem.rsplit('-', 1)[0]
        tool_counts[name] += 1
        tool_seconds[name] += tool.get('elapsed_seconds', tool.get('seconds', 0)) or 0
        tool_errors[name] += bool(tool.get('error'))
    group, seed, fault = result['group'], result['seed'], result.get('fault', 'none')
    initial = initial_receipt(result.get('initial_request'), root, requests)
    recoveries = [r for r in requests.values() if r['run']==str(root) and
                  (r['owner_group'], r['owner_seed'], r['owner_fault'])==(group, seed, fault)]
    logical = ([initial] if initial else []) + recoveries
    blocks = result.get('blocks', [])
    reward = result.get('reward')
    successful_followups = sum(b.get('block', i)>0 and b.get('reward')==1 for i,b in enumerate(blocks))
    recovery_scope = []
    for context in sorted(dest.glob('recovery-context-*.json')):
        payload = read_json(context, {})
        public = payload.get('feedback', {})
        recovery_scope.append(dict(context=str(context), **public.get('recovery_scope', {}),
                                   available_result_ids=[v.get('id') for v in public.get('available_results', [])]))
    local_recoveries = sum(e.get('check')=='recovery' for e in feedback)
    recovered = bool(result.get('replans') or local_recoveries or len(blocks)>1)
    classification = ('recovery_success' if recovered else 'normal_completion') if reward==1 else 'unrecovered_exit'
    info = result.get('info') or {}
    if trace is None:
        warnings.append('Missing trace: ' + str(dest))
    if result.get('actual_api_calls') is not None and trace and result['actual_api_calls']!=len(apis):
        warnings.append('API result/trace count mismatch: ' + str(dest))
    initial_generated_here = result.get('new_llm_requests', 0)>result.get('replans', 0)
    return dict(run=str(root), episode=str(dest), group=group, seed=seed, fault=fault, fault_fired=result.get('fault_fired'),
                reward=reward, reward_success=reward==1, physical_success=info.get('physical_success'),
                raw_success=info.get('raw_success'), status=result.get('status'),
                stop_reason=result.get('stop_reason'), classification=classification,
                recovery_protocol_misclassification=bool(
                    result.get('stop_reason')=='model_finish_claim_without_reward_change'
                    and str(result.get('finish_claim') or '').lstrip().startswith('```')),
                infrastructure_failure=result.get('status')=='system_error' or bool(result.get('truncated')),
                paired_hashes={field:result.get(field) for field in PAIR_FIELDS},
                execution_seconds=sum(b.get('seconds', 0) or 0 for b in blocks),
                observed_total_seconds=result.get('total_seconds'),
                standalone_seconds_estimate=(result.get('total_seconds') or 0)+(0 if initial_generated_here or not initial else initial.get('seconds') or 0),
                blocks=blocks,
                replanning=dict(requested=result.get('replans', 0), receipts=len(recoveries),
                                successful_followup_blocks=successful_followups,
                                final_success_after_replan=bool(recoveries and reward==1)),
                logical_llm=request_totals(logical), logical_request_paths=[r['path'] for r in logical],
                missing_initial_receipt=initial is None, recovery_scopes=recovery_scope,
                local_recoveries=local_recoveries,
                failure_reasons=dict(Counter(e.get('reason') for e in feedback if e.get('status')=='FAIL')),
                graph=dict(events=len(events), versions=sum(e.get('action')=='version' for e in events),
                           replacing_versions=sum(e.get('action')=='version' and e.get('previous') is not None for e in events),
                           invalidate_events=len(invalidations), effective_invalidate_events=len(changes),
                           unique_invalidated_nodes=len({n for e in changes for n in e['invalidated']}),
                           conservative_invalidations=sum(bool(e.get('conservative')) for e in invalidations),
                           dependency_blocks=sum(e.get('check')=='dependency_guard' and e.get('status')=='FAIL' for e in feedback),
                           block_reasons=dict(Counter(e.get('reason') for e in feedback if e.get('check')=='dependency_guard' and e.get('status')=='FAIL')),
                           consumed_plans=sum(e.get('action')=='consume_plan' for e in events),
                           returned_calls=sum(e.get('action')=='returned_call' for e in events),
                           passing_checks_after_first_invalidation=dict(passes_after)),
                tools=dict(actual_api_calls=result.get('actual_api_calls'), api_trace_calls=len(apis),
                           actual_api_by_name=dict(by_api), localizations_by_object=dict(localization),
                           repeated_localizations=sum(max(0,n-1) for n in localization.values()),
                           repeated_identical_api_arguments=sum(max(0,n-1) for n in signatures.values()),
                           tool_calls=dict(tool_counts), tool_seconds=dict(tool_seconds), tool_errors=dict(tool_errors),
                           http_requests=result.get('http_requests')))


def pair_episodes(episodes):
    by_key = defaultdict(dict)
    for row in episodes:
        key = (row['run'], row['seed'], row['fault'])
        if row['group'] in by_key[key]:
            raise ValueError('Duplicate episode for paired group: ' + str(key))
        by_key[key][row['group']] = row
    pairs = []
    for (run, seed, fault), groups in sorted(by_key.items()):
        complete = all(g in groups for g in ('G0', 'G1'))
        bad = [field for field in PAIR_FIELDS if complete and
               (groups['G0']['paired_hashes'][field] is None or groups['G0']['paired_hashes'][field]!=groups['G1']['paired_hashes'][field])]
        exposed = {g:row.get('fault_fired') for g,row in groups.items()}
        fault_complete = complete and (fault=='none' or all(exposed.get(g) is True for g in ('G0','G1')))
        excluded = sorted({reason for row in groups.values() for reason in row.get('effect_exclusion_reasons', [])})
        if not complete: excluded.append('incomplete_pair')
        if bad: excluded.append('paired_hash_mismatch')
        if fault!='none' and not fault_complete: excluded.append('fault_not_exposed_both_groups')
        item = dict(run=run,seed=seed,fault=fault,complete=complete,hash_match=complete and not bad,mismatched_fields=bad,
                    fault_fired=exposed,fault_exposure_complete=fault_complete,
                    valid_for_effect=not excluded,exclusion_reasons=excluded,
                    groups={g:row['episode'] for g,row in groups.items()})
        for row in groups.values():
            row['valid_for_effect'] = item['valid_for_effect']
            row['paired_exclusion_reasons'] = list(excluded)
        if complete:
            g0,g1=groups['G0'],groups['G1']
            item.update(reward={g:groups[g]['reward'] for g in ('G0','G1')},
                        physical_success={g:groups[g]['physical_success'] for g in ('G0','G1')},
                        g1_minus_g0=dict(reward_success=int(g1['reward_success'])-int(g0['reward_success']),
                          actual_api_calls=(g1['tools']['actual_api_calls'] or 0)-(g0['tools']['actual_api_calls'] or 0),
                          repeated_localizations=g1['tools']['repeated_localizations']-g0['tools']['repeated_localizations'],
                          llm_requests=g1['logical_llm']['requests']-g0['logical_llm']['requests'],
                          standalone_seconds=g1['standalone_seconds_estimate']-g0['standalone_seconds_estimate']))
        pairs.append(item)
    return pairs


def analyze(roots):
    roots = list(dict.fromkeys(Path(p).resolve() for p in roots))
    requests, run_meta, warnings = {}, [], []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        config = read_json(root/'experiment.json', {})
        abort = read_json(root/'engineering-abort.json')
        engineering_excluded = abort is not None and abort.get('excluded_from_effect_comparison', True)
        if engineering_excluded:
            warnings.append('Engineering-aborted run retained in raw accounting but excluded from effects: ' + str(root))
        chosen = {key:config.get(key) for key in CONFIG_FIELDS}
        config_hash = hashlib.sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()
        receipts = sorted((root/'llm').glob('*/receipt.json'))
        for receipt in receipts:
            requests[str(receipt.resolve())] = request_row(receipt, root)
        run_meta.append(dict(root=str(root), configuration_fingerprint=config_hash, configuration=chosen,
                             excluded_from_effect_comparison=bool(engineering_excluded),
                             engineering_abort_reason=(abort or {}).get('reason'),
                             complete_summary_exists=(root/'summary.json').exists(),
                             unfinished_episodes=[str(p) for p in sorted((root/'episodes').glob('*')) if p.is_dir() and not (p/'result.json').exists()],
                             pending_requests=[str(p.parent) for p in sorted((root/'llm').glob('*/prompt.json')) if not (p.parent/'receipt.json').exists()]))
    episodes = [analyze_episode(path, root, requests, warnings)
                for root in roots for path in sorted((root/'episodes').glob('*/result.json'))]
    meta_by_run = {meta['root']:meta for meta in run_meta}
    for episode in episodes:
        reasons = []
        if meta_by_run[episode['run']]['excluded_from_effect_comparison']:
            reasons.append('engineering_abort')
        if episode['infrastructure_failure']:
            reasons.append('infrastructure_failure')
        if episode.get('recovery_protocol_misclassification'):
            reasons.append('recovery_protocol_misclassification')
        if str(episode.get('stop_reason') or '').startswith('global_'):
            reasons.append('global_budget_abort')
        episode['effect_exclusion_reasons'] = reasons
        episode['excluded_from_effect_comparison'] = bool(reasons)
    pairs = pair_episodes(episodes)
    groups = {}
    for group in ('G0','G1'):
        subset = [r for r in episodes if r['group']==group]
        if not subset: continue
        logical = []
        for ep in subset:
            for path in ep['logical_request_paths']:
                logical.append(requests.get(path) or request_row(Path(path), Path(ep['run'])))
        groups[group] = dict(episodes=len(subset), effect_eligible_episodes=sum(r['valid_for_effect'] for r in subset),
            fault_fired_true=sum(r.get('fault_fired') is True for r in subset),
            reward_successes=sum(r['reward_success'] for r in subset),
            physical_success_true=sum(r['physical_success'] is True for r in subset),
            physical_success_unknown=sum(r['physical_success'] is None for r in subset),
            classification=dict(Counter(r['classification'] for r in subset)),
            statuses=dict(Counter(r['status'] for r in subset)),
            infrastructure_failures=sum(r['infrastructure_failure'] for r in subset),
            actual_api_calls=sum(r['tools']['actual_api_calls'] or 0 for r in subset),
            repeated_localizations=sum(r['tools']['repeated_localizations'] for r in subset),
            dependency_blocks=sum(r['graph']['dependency_blocks'] for r in subset),
            unique_invalidated_nodes_sum=sum(r['graph']['unique_invalidated_nodes'] for r in subset),
            successful_followup_blocks=sum(r['replanning']['successful_followup_blocks'] for r in subset),
            recovery_requests=sum(r['replanning']['receipts'] for r in subset),
            observed_total_seconds=sum(r['observed_total_seconds'] or 0 for r in subset),
            mean_standalone_seconds_estimate=sum(r['standalone_seconds_estimate'] for r in subset)/len(subset),
            logical_llm=request_totals(logical))
    valid_pairs=[p for p in pairs if p['valid_for_effect']]
    actual=list(requests.values())
    return dict(schema='capx.graph_feedback_analysis.v1', runs=run_meta, groups=groups, episodes=episodes,
                pairs=pairs, paired_effects=dict(valid_pairs=len(valid_pairs), excluded_pairs=len(pairs)-len(valid_pairs),
                    exclusion_reasons=dict(Counter(reason for p in pairs for reason in p['exclusion_reasons'])),
                    g1_only_reward_success=sum(p['reward']['G1']==1 and p['reward']['G0']!=1 for p in valid_pairs),
                    g0_only_reward_success=sum(p['reward']['G0']==1 and p['reward']['G1']!=1 for p in valid_pairs),
                    both_reward_success=sum(p['reward']['G0']==1 and p['reward']['G1']==1 for p in valid_pairs)),
                billing=request_totals(actual), request_ledger=actual,
                mixed_run_configurations=len({r['configuration_fingerprint'] for r in run_meta})>1,
                warnings=warnings,
                notes=['Counts are descriptive; more graph nodes, invalidations or checks are not evidence of better task performance.',
                       'Reward is the unchanged execution-qualified P0 reward; physical_success and raw_success are separately reported.',
                       'Pairs are matched within each run by seed and fault, then checked against all four hashes. Effect comparisons additionally exclude engineering aborts, infrastructure failures, global-budget interruptions, incomplete pairs, and requested faults not actually exposed to both groups. Raw group outcomes and billing remain visible.',
                       'Shared initial requests appear once in billing and once per policy episode in logical_llm. Request ledger counts observed invocations, not invoice entries.',
                       'Unknown usage stays explicit. Token totals cover known usage only; reasoning is included in output tokens and is not charged twice.',
                       'Cost estimates reuse recorded DeepSeek rates: cached input $0.044, uncached input $1.32, output $3.96 per million, off-peak half; billing time/discounts are not inferred. Unknown request reservations are listed separately.',
                       'Standalone time adds initial-generation latency to episodes that reused it; observed elapsed time is not a hardware-isolated benchmark.',
                       'Repeated localization and identical arguments are counts, not presumed waste: changed physical state can require the same API again.',
                       'Recovery-success classification means success after a local or model recovery attempt; it does not establish that recovery caused the success.',
                       'Passing checks after invalidation are revalidation evidence only, not proof that an invalidated plan was safely recovered.',
                       'Incomplete runs/episodes and pending model requests are explicit and are not fabricated into completed outcomes.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError(args.output)
    result=analyze(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'groups':result['groups'],'paired_effects':result['paired_effects'],'billing':result['billing'],'output':str(args.output)},indent=2))


if __name__=='__main__':main()
