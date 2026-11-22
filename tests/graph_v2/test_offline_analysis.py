import json
from pathlib import Path
import pytest
from capx.experiments.graph_v2_analysis import label_check, analyze_episode, analyze, billing


def geometry():
    return {'base_world':{'position':[0,0,0],'rotation':[[1,0,0],[0,1,0],[0,0,1]]},'objects':{'red':{'half_size':[.02,.02,.02]},'green':{'half_size':[.03,.03,.03]}},'table_top_world_z':0.,'table_center_world':[0,0,0],'table_half_extent_xy':[.5,.5]}


def state(red=(0,0,.086),green=(0,0,.03),red_contact=True,green_contact=False):
    return {'canonical':{'sim_time':1,'frame':'world','tcp_world':{'position':[0,0,.1],'quaternion_wxyz':[1,0,0,0]},'objects':{
        'red':{'position_world':list(red),'quaternion_wxyz':[1,0,0,0],'bilateral_contact':red_contact},
        'green':{'position_world':list(green),'quaternion_wxyz':[1,0,0,0],'bilateral_contact':green_contact}}}}


def event(check='pre_release',status='PASS',**kw):
    return {'check':check,'status':status,'metrics':{},'reason':'observed','layer':'L2',**kw}


def test_release_labels_actual_geometry_not_final_reward():
    good=label_check(event(),{'state':state(),'held_object':'red cube','placement_skill':'place'},geometry())
    bad=label_check(event(),{'state':state(red=(.1,0,.086)),'held_object':'red cube','placement_skill':'place'},geometry())
    assert good['classification']=='true_allow'
    assert bad['classification']=='false_allow'


def test_uncertain_release_is_real_block_but_not_confident_detection():
    item=label_check(event(status='UNCERTAIN'),{'state':state(),'held_object':'red cube','placement_skill':'place'},geometry())
    assert item['classification']=='false_block' and item['uncertain']


def test_control_reuses_world_base_transform():
    e=event('tcp_tracking','PASS',metrics={'target_tcp':[0,0,.1,1,0,0,0]})
    assert label_check(e,{'state':state()},geometry())['classification']=='true_allow'
    e['metrics']['target_tcp'][0]=.3
    assert label_check(e,{'state':state()},geometry())['classification']=='false_allow'


def test_green_hold_does_not_borrow_red_contact():
    p={'state':state(red_contact=True,green_contact=False),'held_object':'green cube'}
    assert label_check(event('hold_probe'),p,geometry())['classification']=='false_allow'
    p['state']=state(red_contact=False,green_contact=True)
    assert label_check(event('hold_probe'),p,geometry())['classification']=='true_allow'


def test_missing_private_evidence_unknown_not_success():
    assert label_check(event(),None,geometry())['classification']=='unknown'
    assert label_check(event(),{'state':state(),'held_object':'red cube'},None)['classification']=='unknown'


def test_table_release_not_assessed_against_red_block():
    s=state(red=(.3,0,.02),green=(0,0,.036),red_contact=False,green_contact=True)
    p={'state':s,'held_object':'green cube','placement_skill':'table_place'}
    assert label_check(event(),p,geometry())['classification']=='true_allow'
    s['canonical']['objects']['green']['position_world'][2]=.15
    assert label_check(event(),p,geometry())['classification']=='false_allow'


def test_lift_truth_has_independent_per_grasp_reference():
    p={'state':state(red=(0,0,.15)),'held_object':'red cube'}
    assert label_check(event('lift'),p,geometry(),baseline_z={'red':.02})['classification']=='true_allow'
    assert label_check(event('lift'),p,geometry(),baseline_z={})['classification']=='unknown'


def fixture_episode(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True);g=geometry();(tmp_path/'evaluation-geometry.json').write_text(json.dumps(g))
    rows=[]
    for i,(status,x,attempt) in enumerate([('FAIL',.08,0),('PASS',0,1)]):
        rows += [{'kind':'feedback',**event(status=status)}, {'kind':'feedback_evaluation','event_index':i,'state':state(red=(x,0,.086)),'held_object':'red cube','phase':'holding_verified','placement_skill':'place','local_attempts':attempt}]
    trace=tmp_path/'trace.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result={'task':'stack','seed':1,'repeat':0,'group':'G1','fault':'none','ablation':'full','reward':0.,'status':'task_failure','trace_path':str(trace),'blocks':[],'request_paths':[],'replans':0}
    p=tmp_path/'result.json';p.write_text(json.dumps(result));return p


def test_alignment_risk_improvement_not_task_success(tmp_path):
    out=analyze_episode(fixture_episode(tmp_path))
    assert out['detection']['true_block']==1 and out['detection']['true_allow']==1
    recovery=out['recoveries'][0]
    assert recovery['physical_risk_cleared'] and recovery['physical_error_improved']
    assert out['task_success'] is False


def test_global_shared_request_counted_once(tmp_path):
    request=tmp_path/'request';request.mkdir();receipt=request/'receipt.json'
    receipt.write_text(json.dumps({'response_id':'id1','request_id':'r1','usage':{'prompt_tokens':10,'completion_tokens':20,'completion_tokens_details':{'reasoning_tokens':12}},'http_status':200,'attempts':1,'cost_upper_usd':.01,'seconds':3,'label':'initial'}))
    rows=[{'request_paths':[str(request)]},{'request_paths':[str(request)]}]
    out=billing([tmp_path],rows)
    assert out['unique_requests']==1 and out['input_tokens']==10 and out['output_tokens']==20 and out['reasoning_tokens']==12


def test_unknown_reserved_request_remains_billed_bound(tmp_path):
    (tmp_path/'request-ledger.json').write_text(json.dumps({'requests':[{'label':'uncertain','state':'started','directory':str(tmp_path/'missing'),'reserved_usd':.2}]}))
    out=billing([tmp_path],[])
    assert out['unknown_usage_requests']==1 and out['cost_upper_usd']==.2


def test_independent_bad_state_and_block_is_true_block():
    out=label_check(event(status='FAIL'),{'state':state(red=(.1,0,.086)),'held_object':'red cube','placement_skill':'place'},geometry())
    assert out['classification']=='true_block'


def test_shared_receipt_copies_and_overlapping_roots_not_double_billed(tmp_path):
    receipt={'response_id':'shared','http_status':200,'usage':{'prompt_tokens':100,'completion_tokens':50,'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20},'cost_upper_usd':.1,'seconds':3}
    paths=[]
    for name in ('one','two'):
        p=tmp_path/name;p.mkdir();(p/'receipt.json').write_text(json.dumps(receipt));paths.append(str(p))
    out=billing([tmp_path,tmp_path/'one'],[{'request_paths':paths}])
    assert out['unique_requests']==1 and out['cost_upper_usd']==.1
    estimate=out['known_usage_cache_aware_estimate_range_usd']
    assert estimate['peak']==pytest.approx((80*.044+20*1.32+50*3.96)/1e6)
    assert estimate['offpeak']==pytest.approx(estimate['peak']/2)


def test_table_unknown_geometry_never_becomes_true_allow():
    g=geometry();g['table_top_world_z']=None
    out=label_check(event(),{'state':state(),'held_object':'red cube','placement_skill':'table_place'},g)
    assert out['classification']=='unknown'


def test_cli_offline_analysis_preserves_failures_and_exclusive_output(tmp_path):
    import subprocess,sys
    fixture_episode(tmp_path/'run'/'episodes'/'G1');dest=tmp_path/'report.json'
    cmd=[sys.executable,'-B','-m','capx.experiments.graph_v2_analysis','--run',str(tmp_path/'run'),'--output',str(dest),'--bootstrap','10']
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    report=json.loads(dest.read_text())
    assert report['episodes'][0]['task_success'] is False
    assert report['billing']['unique_requests']==0
    assert subprocess.run(cmd,capture_output=True).returncode!=0


def test_precise_tcp_uses_strict_independent_fk_and_command_is_not_detection():
    private={'state':state(),'held_object':'red cube'}
    metrics={'stage':'final','target_tcp':[0,0,.11,1,0,0,0],'actual_tcp':[0,0,.11,1,0,0,0]}
    labeled=label_check(event('precise_tcp','FAIL',metrics=metrics),private,geometry())
    assert labeled['classification']=='true_block' and labeled['position_error_m']==pytest.approx(.01)
    assert labeled['position_tolerance_m']==.003
    pre=label_check(event('precise_tcp_pre_descent','PASS',metrics=metrics),private,geometry())
    assert pre['classification']=='false_allow'
    command=label_check(event('precise_tcp_command','UNCERTAIN',metrics=metrics),private,geometry())
    assert command['classification']=='not_applicable' and command['decision']=='announcement'


def test_conditional_attachment_or_support_never_replaces_private_truth():
    for check in ('attachment_consistency','conditional_support'):
        result=label_check(event(check,'PASS',metrics={'truth_bad':False,'conditional_attachment':True}),None,geometry())
        assert result['classification'] not in ('true_allow','true_block')
    result=label_check(event('precise_tcp','PASS',metrics={'target_tcp':[0,0,.1,1,0,0,0],'actual_tcp':[0,0,.1,1,0,0,0]}),None,geometry())
    assert result['classification']=='unknown'


def test_precise_recovery_only_compares_identical_stage_and_target(tmp_path):
    path=fixture_episode(tmp_path);result=json.loads(path.read_text());rows=[]
    for i,(stage,target,actual,correction,status) in enumerate([('above_target',.2,.18,False,'FAIL'),('above_target',.2,.2,True,'PASS'),('final',.1,.1,True,'PASS')]):
        s=state();s['canonical']['tcp_world']['position'][2]=actual
        rows += [{'kind':'feedback',**event('precise_tcp',status,metrics={'stage':stage,'target_tcp':[0,0,target,1,0,0,0],'correction':correction})},
                 {'kind':'feedback_evaluation','event_index':i,'state':s,'held_object':'red cube'}]
    Path(result['trace_path']).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=analyze_episode(path)
    assert len(out['recoveries'])==1
    assert out['recoveries'][0]['kind']=='precision_control' and out['recoveries'][0]['physical_error_improved']


def test_frozen_oracle_v2_precise_records_have_independent_labels():
    fixture=json.loads((Path(__file__).parent/'fixtures/oracle_v2_precise.json').read_text())
    labels=[label_check(row['event'],row['private'],fixture['geometry']) for row in fixture['records']]
    assert [x['classification'] for x in labels]==['not_applicable','true_block','true_allow','true_allow','unknown']
    assert fixture['source']=='unchanged_handwritten_grounded_oracle_fixture'


def test_precise_and_legacy_truth_labels_are_plain_json_booleans():
    for check in ('precise_tcp','precise_tcp_pre_descent','tcp_tracking'):
        row=label_check(event(check,'PASS',metrics={'target_tcp':[0,0,.1,1,0,0,0]}),{'state':state()},geometry())
        assert type(row['truth_bad']) is bool
        json.dumps(row,allow_nan=False)
