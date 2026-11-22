"""Offline accounting must not double-bill shared initial programs or infer reward."""
import json
from pathlib import Path
import pytest
from capx.experiments.graph_analysis import analyze, main


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(tmp_path):
    run=tmp_path/'run'
    put(run/'experiment.json', {'max_tokens':32768, 'max_blocks':3, 'source_hashes':{'core':'a'}})
    initial=run/'llm'/'001-initial-seed-7-fault-none'
    recovery=run/'llm'/'002-replan-G0-seed-7-fault-none-1'
    for path,label,seconds,usage in [
        (initial,'initial-seed-7-fault-none',10,{'prompt_tokens':100,'completion_tokens':20,'prompt_cache_hit_tokens':40,'completion_tokens_details':{'reasoning_tokens':8}}),
        (recovery,'replan-G0-seed-7-fault-none-1',20,{'prompt_tokens':200,'completion_tokens':30,'completion_tokens_details':{'reasoning_tokens':10}})]:
        put(path/'receipt.json',dict(label=label,status='received',attempts=1,seconds=seconds,usage=usage,cost_upper_usd=.01,requested_model='deepseek-v4-pro',response_model='deepseek-v4-pro',http_status=200))
    hashes=dict(snapshot_hash='state',initial_prompt_hash='prompt',api_docs_hash='docs',code_hash='code')
    events=[{'sequence':1,'action':'version','previous':None,'current':'o1','invalidation_enabled':False,'invalidated':[]},
            {'sequence':2,'action':'invalidate','roots':['o1'],'reason':'new_support','conservative':False,'invalidated':['o1','p1']},
            {'sequence':3,'action':'add','node':{'id':'c2','kind':'check','data':{'status':'PASS','check':'retreat_boundary'}}},
            {'sequence':4,'action':'consume_plan','node_id':'p2','execution':'consumed'},
            {'sequence':5,'action':'returned_call','node_id':'a1','state':'done','valid':True}]
    for group in ['G0','G1']:
        episode=run/'episodes'/f'{group}-seed-7-fault-none'
        trace=episode/'trace'/'trace.jsonl'
        rows=[{'kind':'code','block':1},
              {'kind':'api','name':'get_object_pose','args':['green cube'],'kwargs':{}},
              {'kind':'api','name':'get_object_pose','args':['green cube'],'kwargs':{}},
              {'kind':'api','name':'goto_pose','args':[[1,2,3],[1,0,0,0]],'kwargs':{}},
              {'kind':'feedback','layer':'L3','check':'dependency_guard','status':'FAIL','reason':'stale_target'},
              {'kind':'feedback','layer':'L2','check':'recovery','status':'UNCERTAIN','reason':'align'}]
        trace.parent.mkdir(parents=True,exist_ok=True)
        trace.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        result=dict(group=group,seed=7,fault='none',reward=1. if group=='G0' else 0.,status='success' if group=='G0' else 'interrupted',info={'physical_success':True,'raw_success':True},replans=1 if group=='G0' else 0,replan_effective=group=='G0',blocks=[{'block':0,'reward':0,'seconds':2},{'block':1,'reward':1,'seconds':3}] if group=='G0' else [{'block':0,'reward':0,'seconds':2}],actual_api_calls=3,http_requests=5,new_llm_requests=2 if group=='G0' else 0,total_seconds=35 if group=='G0' else 4,trace_path=str(trace),initial_request=str(initial),recoveries=1,graph_events=events,**hashes)
        put(episode/'result.json',result)
        put(episode/'recovery-context-1.json',{'feedback':{'recovery_scope':{'recompute':['p1'],'reobserve':['o1'],'never_replay':['a0']},'available_results':[{'id':'o2'}]}})
    return run


def test_shared_initial_billing_and_logical_usage_are_separate(tmp_path):
    out=analyze([fixture_run(tmp_path)])
    assert out['billing']['requests']==2
    assert out['billing']['input_tokens']==300 and out['billing']['output_tokens']==50
    assert out['billing']['reasoning_tokens']==18
    assert out['groups']['G0']['logical_llm']['requests']==2
    assert out['groups']['G1']['logical_llm']['requests']==1
    assert out['groups']['G1']['logical_llm']['seconds']==10
    assert out['billing']['shared_initial_requests']==1
    # Reasoning is a subset of completion tokens, not added again to output cost.
    expected=(40*.044+260*1.32+50*3.96)/1e6
    assert out['billing']['cache_aware_peak_estimate_usd']==pytest.approx(expected)


def test_reward_and_physical_state_and_graph_counts_are_separate(tmp_path):
    out=analyze([fixture_run(tmp_path)])
    assert out['groups']['G0']['reward_successes']==1
    assert out['groups']['G1']['reward_successes']==0
    assert out['groups']['G1']['physical_success_true']==1
    ep=out['episodes'][0]
    assert ep['classification']=='recovery_success'
    assert ep['graph']['unique_invalidated_nodes']==2
    assert ep['graph']['dependency_blocks']==1
    assert ep['graph']['returned_calls']==1
    assert ep['tools']['repeated_localizations']==1
    assert ep['tools']['actual_api_by_name']['get_object_pose']==2
    assert ep['replanning']['successful_followup_blocks']==1
    assert ep['recovery_scopes'][0]['recompute']==['p1']
    assert out['pairs'][0]['hash_match'] is True


def test_hash_mismatch_is_reported_not_silently_pooled_as_valid_pair(tmp_path):
    run=fixture_run(tmp_path)
    path=run/'episodes'/'G1-seed-7-fault-none'/'result.json'
    row=json.loads(path.read_text());row['code_hash']='other';put(path,row)
    out=analyze([run])
    assert not out['pairs'][0]['hash_match']
    assert out['pairs'][0]['mismatched_fields']==['code_hash']
    assert out['paired_effects']['valid_pairs']==0


def test_duplicate_roots_do_not_double_count(tmp_path):
    run=fixture_run(tmp_path)
    out=analyze([run,run.resolve()])
    assert len(out['episodes'])==2 and out['billing']['requests']==2


def test_unknown_usage_is_not_zero_cost(tmp_path):
    run=fixture_run(tmp_path)
    put(run/'llm'/'003-replan-G1-seed-7-fault-none-1'/'receipt.json',{'label':'replan-G1-seed-7-fault-none-1','status':'error','attempts':1,'seconds':4,'cost_upper_usd':.15,'http_status':None})
    out=analyze([run])
    assert out['billing']['unknown_usage_requests']==1
    assert out['billing']['unknown_usage_reserved_usd']==.15
    assert out['billing']['requests']==3


def test_incomplete_episode_and_pending_request_are_explicit(tmp_path):
    run=fixture_run(tmp_path)
    put(run/'episodes'/'G0-seed-8-fault-none'/'initial-prompt.json',[])
    put(run/'llm'/'004-initial-seed-8-fault-none'/'prompt.json',[])
    out=analyze([run])
    assert len(out['runs'][0]['unfinished_episodes'])==1
    assert len(out['runs'][0]['pending_requests'])==1
    assert len(out['episodes'])==2


def test_output_is_exclusive_and_help_does_not_import_simulator(tmp_path,capsys):
    run=fixture_run(tmp_path);output=tmp_path/'analysis.json'
    main(['--run',str(run),'--output',str(output)])
    before=output.read_bytes()
    with pytest.raises(FileExistsError):main(['--run',str(run),'--output',str(output)])
    assert output.read_bytes()==before
    with pytest.raises(SystemExit) as exc:main(['--help'])
    assert exc.value.code==0


def test_instrumented_tool_elapsed_seconds_are_counted(tmp_path):
    run=fixture_run(tmp_path)
    put(run/'episodes'/'G0-seed-7-fault-none'/'tools'/'sam3-001.json',
        {'tool':'sam3','elapsed_seconds':1.25,'error':None})
    out=analyze([run])
    tools=out['episodes'][0]['tools']
    assert tools['tool_calls']=={'sam3':1}
    assert tools['tool_seconds']['sam3']==1.25


def update_result(run,group,**changes):
    path=run/'episodes'/f'{group}-seed-7-fault-none'/'result.json'
    row=json.loads(path.read_text());row.update(changes);put(path,row)


def test_requested_fault_must_fire_for_both_groups_before_effect_comparison(tmp_path):
    run=fixture_run(tmp_path)
    update_result(run,'G0',fault='support_refresh',fault_fired=True)
    update_result(run,'G1',fault='support_refresh',fault_fired=False)
    out=analyze([run]);pair=out['pairs'][0]
    assert pair['hash_match']
    assert pair['fault_fired']=={'G0':True,'G1':False}
    assert not pair['fault_exposure_complete']
    assert not pair['valid_for_effect'] and 'fault_not_exposed_both_groups' in pair['exclusion_reasons']
    assert out['paired_effects']['valid_pairs']==0
    assert out['groups']['G0']['reward_successes']==1  # Raw outcome remains visible.


def test_both_fault_exposures_allow_pair_when_no_other_abort(tmp_path):
    run=fixture_run(tmp_path)
    for group in ['G0','G1']:update_result(run,group,fault='support_refresh',fault_fired=True)
    out=analyze([run])
    assert out['pairs'][0]['fault_exposure_complete']
    assert out['pairs'][0]['valid_for_effect']
    assert out['paired_effects']['valid_pairs']==1


def test_engineering_abort_retains_raw_outcomes_but_excludes_effects(tmp_path):
    run=fixture_run(tmp_path)
    put(run/'engineering-abort.json',{'excluded_from_effect_comparison':True,'reason':'runtime changed during smoke'})
    out=analyze([run])
    assert out['runs'][0]['excluded_from_effect_comparison']
    assert out['runs'][0]['engineering_abort_reason']=='runtime changed during smoke'
    assert out['episodes'][0]['excluded_from_effect_comparison']
    assert not out['pairs'][0]['valid_for_effect']
    assert 'engineering_abort' in out['pairs'][0]['exclusion_reasons']
    assert out['paired_effects']['valid_pairs']==0
    assert out['billing']['requests']==2 and out['groups']['G0']['reward_successes']==1


@pytest.mark.parametrize('change,reason',[
    ({'status':'system_error'},'infrastructure_failure'),
    ({'stop_reason':'global_model_or_wall_budget'},'global_budget_abort'),
])
def test_infrastructure_and_global_abort_are_not_effect_failures(tmp_path,change,reason):
    run=fixture_run(tmp_path);update_result(run,'G1',**change)
    out=analyze([run])
    assert out['pairs'][0]['hash_match']
    assert reason in out['pairs'][0]['exclusion_reasons']
    assert not out['pairs'][0]['valid_for_effect']
    assert out['groups']['G1']['episodes']==1


def test_episode_api_budget_is_a_comparable_policy_outcome(tmp_path):
    run=fixture_run(tmp_path)
    update_result(run,'G1',stop_reason='episode_api_budget')
    out=analyze([run])
    assert out['pairs'][0]['valid_for_effect']


def test_code_only_response_misclassified_as_finish_excludes_effect_not_cost(tmp_path):
    run=fixture_run(tmp_path)
    update_result(run,'G1',stop_reason='model_finish_claim_without_reward_change',finish_claim='```python\nread_robot()\n```')
    out=analyze([run])
    assert out['paired_effects']['valid_pairs']==0
    assert 'recovery_protocol_misclassification' in out['pairs'][0]['exclusion_reasons']
    assert out['billing']['requests']==2 and out['groups']['G0']['reward_successes']==1


def test_explicit_finish_is_a_policy_outcome_not_parser_error(tmp_path):
    run=fixture_run(tmp_path)
    update_result(run,'G1',stop_reason='model_finish_claim_without_reward_change',finish_claim='FINISH')
    out=analyze([run])
    assert out['pairs'][0]['valid_for_effect']
