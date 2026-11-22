"""Offline intervention roles must not turn active sensing into a repaired failure."""
import json
from pathlib import Path
import pytest
from capx.experiments.graph_v2_analysis import analyze_episode, analyze
from capx.experiments.graph_statistics import summarize_group
from test_offline_analysis import event, geometry, state


def episode(tmp_path, events, *, success=True, local=0, replans=0, decisions=(), blocks=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i,(e,s) in enumerate(events):
        if e.get('kind')=='api': rows.append(e); continue
        index=sum(r.get('kind')=='feedback' for r in rows)
        rows.extend([{'kind':'feedback',**e},{'kind':'feedback_evaluation','event_index':index,
                     'state':s,'held_object':'red cube','phase':'holding_verified',
                     'placement_skill':'place','local_attempts':i}])
    trace=tmp_path/'trace.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'evaluation-geometry.json').write_text(json.dumps(geometry()))
    value=dict(task='lift',seed=0,repeat=0,group='G1',fault='none',reward=float(success),
               status='success' if success else 'task_failure',policy_success=success,
               trace_path=str(trace),local_recoveries=local,replans=replans,
               replan_llm_requests=replans,recovery_decisions=list(decisions),blocks=list(blocks),request_paths=[])
    path=tmp_path/'result.json';path.write_text(json.dumps(value));return path


def phase(operation):
    return event('phase_precondition','PASS',metrics={'operation':operation,'held_object':'red cube'})


def motion(): return {'kind':'api','name':'goto_pose'}


def test_normal_verify_hold_is_evidence_not_physical_recovery(tmp_path):
    path=episode(tmp_path,[
        (event('grasp'),state(red=(0,0,.02))),
        (event('lift'),state(red=(0,0,.1))),
        (phase('verify_hold'),state()),(motion(),None),(motion(),None),
        (event('hold_probe'),state())])
    report=analyze_episode(path)
    item=report['interventions'][0]
    assert item['role']=='evidence_acquisition' and item['kind']=='hold_evidence_probe'
    assert item['trigger']=='normal_verification' and item['motion_calls']==2
    assert item['detector_passed'] is True and item['detector_cleared'] is False
    assert item['physical_risk_cleared'] is False
    row=dict(report['raw_result'],interventions=report['interventions'])
    summary=summarize_group([row])
    assert summary['normal_successes']==1 and summary['recovery_attempted_episodes']==0
    assert summary['interventions']['evidence_acquisition']['attempts']==1
    assert summary['interventions']['physical_correction']['attempts']==0
    assert summary['interventions']['evidence_acquisition']['normal_verification_attempts']==1


def test_visibility_is_evidence_and_not_independent_physical_clearance(tmp_path):
    path=episode(tmp_path,[
        (event('attachment_consistency','UNCERTAIN'),state()),
        (event('visibility_probe_clearance'),state()),
        (phase('visibility_probe'),state()),(motion(),None),
        (event('attachment_consistency','PASS'),state())],local=1)
    out=analyze_episode(path);item=out['interventions'][0]
    assert item['kind']=='active_visibility' and item['role']=='evidence_acquisition'
    assert item['trigger']=='failed_or_uncertain_checkpoint'
    assert item['motion_calls']==1 and item['detector_cleared'] is True
    assert item['physical_risk_cleared'] is None and item['physical_error_improved'] is None
    assert item['final_task_success'] is True
    summary=summarize_group([dict(out['raw_result'],interventions=out['interventions'])])
    assert summary['recovery_successes']==1  # compatible broad recovery field
    assert summary['interventions']['physical_correction']['attempts']==0
    assert summary['interventions']['evidence_acquisition']['final_task_successes']==1


def test_visibility_missing_after_check_stays_unknown_not_recovered(tmp_path):
    path=episode(tmp_path,[(event('attachment_consistency','UNCERTAIN'),state()),
             (phase('visibility_probe'),state()),(motion(),None)],local=1,success=False)
    item=analyze_episode(path)['interventions'][0]
    assert item['detector_passed'] is None and item['detector_cleared'] is None
    assert item['physical_risk_cleared'] is None and item['final_task_success'] is False


def test_refused_visibility_clearance_is_not_attempted_motion(tmp_path):
    path=episode(tmp_path,[(event('attachment_consistency','UNCERTAIN'),state()),
                 (event('visibility_probe_clearance','UNCERTAIN'),state())],success=False)
    assert analyze_episode(path)['interventions']==[]


def test_visibility_attempts_stop_at_next_probe_and_count_separately(tmp_path):
    path=episode(tmp_path,[(event('attachment_consistency','UNCERTAIN'),state()),
             (phase('visibility_probe'),state()),(motion(),None),
             (event('attachment_consistency','UNCERTAIN'),state()),
             (phase('visibility_probe'),state()),(motion(),None),
             (event('attachment_consistency','PASS'),state())],local=2)
    rows=analyze_episode(path)['interventions']
    assert len(rows)==2 and [r['motion_calls'] for r in rows]==[1,1]
    assert [r['detector_cleared'] for r in rows]==[False,True]


def test_physical_risk_clearance_does_not_imply_final_task_success(tmp_path):
    path=episode(tmp_path,[(event('pre_release','FAIL'),state(red=(.08,0,.086))),
             (motion(),None),(event('pre_release','PASS'),state())],local=1,success=False)
    item=analyze_episode(path)['interventions'][0]
    assert item['role']=='physical_correction'
    assert item['detector_cleared'] and item['physical_risk_cleared']
    assert item['final_task_success'] is False


def test_llm_finish_and_recovered_code_are_separate_attempt_outcomes(tmp_path):
    path=episode(tmp_path,[],replans=2,decisions=[{'block':1,'decision':'regenerate'},
                {'block':2,'decision':'finish'}],blocks=[{'block':0,'reward':0},
                {'block':1,'reward':0,'policy_success':False}],success=False)
    rows=analyze_episode(path)['interventions']
    assert len(rows)==2 and all(r['role']=='llm_replan' for r in rows)
    assert rows[0]['code_executed'] is True and rows[1]['code_executed'] is False
    assert all(r['final_task_success'] is False for r in rows)
    assert all(r['physical_risk_cleared'] is None for r in rows)


def test_detection_reports_coverage_without_labeling_conditional_geometry(tmp_path):
    path=episode(tmp_path,[(event('attachment_consistency'),state()),
             (event('table_clearance'),state()),(event('hold_probe'),state())])
    out=analyze_episode(path);coverage=out['detection_coverage']
    assert coverage['all_feedback_events']==3 and coverage['eligible_checkpoints']==2
    assert coverage['excluded_by_check']=={'attachment_consistency':1}
    assert coverage['independently_labeled_checkpoints']==1
    # A table decision without a recorded region remains unknown, not inferred
    # from the online PASS flag or the eventual task result.
    assert coverage['unknown_checkpoints']==1
    assert out['detection_by_type']['table']=={'unknown':1}


def test_unknown_legacy_intervention_roles_not_inferred_from_local_count():
    r=dict(task='stack',seed=0,group='G1',reward=1,status='success',local_recoveries=2,replans=0)
    result=summarize_group([r])
    assert result['recovery_successes']==1
    assert result['intervention_coverage']=={'known_episodes':0,'missing_episodes':1}
    assert result['interventions']['physical_correction']['attempts']==0


@pytest.mark.parametrize('operation,prior_check,after_check',[
    ('verify_hold','lift','hold_probe'),
    ('visibility_probe','attachment_consistency','attachment_consistency')])
def test_internal_authorization_gates_do_not_split_one_active_sensing_attempt(tmp_path,operation,prior_check,after_check):
    # Actual Lift0-v4 trace: verify_hold gate, internal authorization, goto,
    # tracking, another internal authorization/goto, then hold_probe result.
    path=episode(tmp_path,[(event('grasp'),state(red=(0,0,.02))),
        (event(prior_check,'PASS' if operation=='verify_hold' else 'UNCERTAIN'),state(red=(0,0,.1))),
        (phase(operation),state()),
        (event('phase_precondition','PASS',reason='authorized_current_boundary'),state()),
        (motion(),None),
        (event('tcp_tracking','PASS',metrics={'target_tcp':[0,0,.1,1,0,0,0]}),state()),
        (event('phase_precondition','PASS',reason='authorized_current_boundary'),state()),
        (motion(),None),(event(after_check),state())])
    items=analyze_episode(path)['interventions']
    assert len(items)==1
    assert items[0]['motion_calls']==2 and items[0]['detector_passed'] is True


@pytest.mark.parametrize('change_view',[False,True])
def test_table_reobservation_is_one_evidence_attempt_including_optional_yaw(tmp_path,change_view):
    events=[(event('attachment_consistency','PASS'),state()),
            (event('table_clearance','UNCERTAIN'),state()),
            (phase('reobserve_table'),state())]
    if change_view:
        events.extend([(phase('visibility_probe'),state()),
            (event('phase_precondition','PASS',reason='authorized_current_boundary'),state()),
            (motion(),None),(event('attachment_consistency','PASS'),state())])
    events.append((event('table_clearance','PASS'),state()))
    out=analyze_episode(episode(tmp_path,events,local=int(change_view)))
    assert len(out['interventions'])==1
    item=out['interventions'][0]
    assert item['kind']=='table_reobservation' and item['role']=='evidence_acquisition'
    assert item['trigger']=='failed_or_uncertain_checkpoint'
    assert item['motion_calls']==int(change_view)
    assert item['included_visibility_probes']==int(change_view)
    assert item['detector_passed'] is True and item['detector_cleared'] is True
    assert item['physical_risk_cleared'] is None and item['physical_error_improved'] is None


def test_held_geometry_recovered_does_not_mean_table_clearance_recovered(tmp_path):
    events=[(event('table_clearance','UNCERTAIN'),state()),(phase('reobserve_table'),state()),
        (phase('visibility_probe'),state()),(motion(),None),
        (event('attachment_consistency','PASS'),state()),
        (event('table_clearance','UNCERTAIN'),state())]
    item=analyze_episode(episode(tmp_path,events,local=1,success=False))['interventions'][0]
    assert item['kind']=='table_reobservation' and item['detector_passed'] is False
    assert item['detector_cleared'] is False and item['physical_risk_cleared'] is None


def test_table_reobservation_without_final_table_check_has_unknown_outcome(tmp_path):
    events=[(event('table_clearance','UNCERTAIN'),state()),(phase('reobserve_table'),state()),
        (phase('visibility_probe'),state()),(motion(),None),
        (event('attachment_consistency','PASS'),state())]
    item=analyze_episode(episode(tmp_path,events,local=1,success=False))['interventions'][0]
    assert item['kind']=='table_reobservation'
    assert item['detector_passed'] is None and item['detector_cleared'] is None
