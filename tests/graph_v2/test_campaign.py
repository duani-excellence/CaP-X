import argparse,json
from pathlib import Path
import pytest
from capx.experiments.graph_campaign import make_plan, fingerprint, freeze_campaign, load_campaign, run_campaign, PersistentModelBudget, status


def config(tmp_path):
    src=tmp_path/'source.py';src.write_text('VERSION=1\n')
    prompt=tmp_path/'prompt.txt';prompt.write_text('task prompt')
    score=tmp_path/'score.py';score.write_text('success unchanged')
    return dict(tasks=['stack'],model='deepseek-v4-pro',services={'sam3':'http://local/sam3'},sources=[str(src)],prompt_files=[str(prompt)],scoring_files=[str(score)],budgets={'max_requests':100,'cost_limit_usd':100.,'max_tokens':32768,'wall_seconds':10000,'max_blocks':6,'api_budget':128,'max_local':3,'request_timeout':400,'reasoning_effort':'low'})


def test_seed_partitions_unique_and_repeated_only_within_scene():
    p=make_plan(['stack','lift','restack'],20,100,2,123)
    dev={s['seed'] for s in p['development']};test={s['seed'] for s in p['test']}
    assert len(dev)==60 and len(test)==300 and not dev&test
    assert p==make_plan(['stack','lift','restack'],20,100,2,123)


def test_freeze_refuses_overwrite_and_source_drift(tmp_path):
    c=config(tmp_path);p=make_plan(c['tasks'],1,1,2,123);root=tmp_path/'campaign'
    freeze_campaign(root,c,p,phase='development')
    with pytest.raises(FileExistsError):freeze_campaign(root,c,p,phase='development')
    Path(c['sources'][0]).write_text('VERSION=2\n')
    with pytest.raises(ValueError,match='drift'):load_campaign(root)


def test_formal_requires_real_task_acceptance(tmp_path):
    c=config(tmp_path);p=make_plan(c['tasks'],20,100,2,123)
    with pytest.raises(ValueError,match='acceptance'):freeze_campaign(tmp_path/'formal',c,p,phase='formal')


def test_secrets_rejected_before_artifact_write(tmp_path):
    c=config(tmp_path);c['api_key']='not-a-real-key'
    with pytest.raises(ValueError,match='secret'):freeze_campaign(tmp_path/'bad',c,make_plan(c['tasks'],1,1,1,12),phase='development')
    assert not (tmp_path/'bad').exists()


def test_run_resume_keeps_all_results_and_does_not_repeat(tmp_path):
    c=config(tmp_path);p=make_plan(c['tasks'],2,1,1,123);root=tmp_path/'campaign'
    freeze_campaign(root,c,p,phase='development')
    calls=[]
    def pair(**kw):
        calls.append(kw['seed'])
        return {'episodes':[dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],group=g,fault=kw['fault'],reward=0.,status='budget_exceeded',snapshot_hash='same') for g in kw['groups']]}
    run_campaign(root,runner=pair,max_units=1)
    assert status(root)['complete_units']==1
    run_campaign(root,runner=pair)
    assert len(calls)==2 and status(root)['complete_units']==2
    assert status(root)['attempted_episodes']==6


def test_uncertain_started_blocks_automatic_rerun(tmp_path):
    c=config(tmp_path);root=tmp_path/'campaign';freeze_campaign(root,c,make_plan(c['tasks'],1,1,1,123),phase='development')
    state=json.loads((root/'state.json').read_text());state['units'][0]['state']='started';(root/'state.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='uncertain'):run_campaign(root,runner=lambda **kw:None)


def test_partial_result_not_marked_complete(tmp_path):
    c=config(tmp_path);root=tmp_path/'campaign';freeze_campaign(root,c,make_plan(c['tasks'],1,1,1,123),phase='development')
    with pytest.raises(ValueError,match='groups'):run_campaign(root,runner=lambda **kw:{'episodes':[]})
    assert status(root)['complete_units']==0


def test_pending_request_not_automatically_reissued(tmp_path):
    state={'version':1,'requests':[{'label':'same','state':'started','reserved_usd':1.,'directory':'does-not-exist'}]}
    (tmp_path/'request-ledger.json').write_text(json.dumps(state))
    budget=PersistentModelBudget(tmp_path,{'max_requests':5,'cost_limit_usd':10,'max_tokens':100,'wall_seconds':100})
    with pytest.raises(RuntimeError,match='uncertain'):budget.query([], 'new')


def test_fingerprint_order_stable():
    assert fingerprint({'b':2,'a':1})==fingerprint({'a':1,'b':2})

class FakeBudget:
    calls=0
    def __init__(self,root,*args):self.root=Path(root);self.rows=[]
    def query(self,prompt,label):
        type(self).calls+=1;dest=self.root/'llm'/'response';dest.mkdir(parents=True)
        answer={'content':'generated program','response_model':'deepseek-v4-pro','finish_reason':'stop'}
        (dest/'response.json').write_text(json.dumps(answer))
        self.rows.append({'attempts':1,'http_status':200,'cost_upper_usd':.1,'usage':{'prompt_tokens':1,'completion_tokens':1}})
        return answer,str(dest)


def test_received_request_reused_across_restart_no_extra_charge(tmp_path):
    FakeBudget.calls=0;opts=dict(max_requests=2,cost_limit_usd=2.,max_tokens=100,wall_seconds=100)
    budget=PersistentModelBudget(tmp_path,opts,factory=FakeBudget)
    first=budget.query([{'role':'user','content':'x'}],'request')
    restored=PersistentModelBudget(tmp_path,opts,factory=FakeBudget)
    assert restored.query([{'role':'user','content':'x'}],'request')==first
    assert FakeBudget.calls==1 and restored.count==1
    assert restored.ledger.spent['deepseek']==.1
    with pytest.raises(ValueError,match='prompt changed'):
        restored.query([{'role':'user','content':'different'}],'request')


def test_global_request_cap_survives_restart(tmp_path):
    opts=dict(max_requests=1,cost_limit_usd=2.,max_tokens=100,wall_seconds=100)
    PersistentModelBudget(tmp_path,opts,factory=FakeBudget).query([], 'first')
    with pytest.raises(RuntimeError,match='request budget'):
        PersistentModelBudget(tmp_path,opts,factory=FakeBudget).query([], 'second')


def test_failure_does_not_refund_uncertain_charge(tmp_path):
    class Crash(FakeBudget):
        def query(self,prompt,label):raise TimeoutError('unknown response')
    opts=dict(max_requests=5,cost_limit_usd=2.,max_tokens=100,wall_seconds=100)
    budget=PersistentModelBudget(tmp_path,opts,factory=Crash)
    with pytest.raises(TimeoutError):budget.query([], 'first')
    ledger=json.loads((tmp_path/'request-ledger.json').read_text())
    assert ledger['requests'][0]['state']=='uncertain'
    assert ledger['requests'][0]['cost_upper_usd']>0
    with pytest.raises(RuntimeError,match='uncertain'):
        PersistentModelBudget(tmp_path,opts,factory=FakeBudget).query([], 'second')


def test_snapshot_task_and_provenance_preserved_in_runner(tmp_path):
    c=config(tmp_path);p=make_plan(c['tasks'],1,1,1,123);seed=p['development'][0]['seed']
    snapshot=tmp_path/'snapshot.json';snapshot.write_text(json.dumps({'seed':seed,'state_hash':'stable'}))
    manifest=tmp_path/'scenes.json';manifest.write_text(json.dumps({'task':'stack','scenes':[{'seed':seed,'snapshot':snapshot.name}]}));c['snapshot_manifest']=str(manifest)
    root=tmp_path/'campaign';freeze_campaign(root,c,p,phase='development')
    def runner(**kw):
        assert kw['snapshot']=={'seed':seed,'state_hash':'stable'}
        return {'episodes':[dict(task='stack',seed=seed,repeat=0,fault='none',group=g,reward=0.,status='task_failure') for g in kw['groups']]}
    run_campaign(root,runner=runner)
    assert status(root)['complete_units']==1


def test_formal_freeze_checks_completed_development_count(tmp_path):
    c=config(tmp_path);evidence=tmp_path/'real-oracle.json';evidence.write_text(json.dumps({'task':'stack','online_vision':True,'reward':1,'status':'success','snapshot_hash':'test-fixture'}))
    acceptance=tmp_path/'acceptance.json';acceptance.write_text(json.dumps({'validated_tasks':['stack'],'evidence_files':[str(evidence)]}));c['acceptance_file']=str(acceptance)
    p=make_plan(c['tasks'],1,1,1,123);dev=tmp_path/'dev';freeze_campaign(dev,c,p,phase='development')
    with pytest.raises(ValueError,match='20 completed'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_snapshot_file_contents_are_frozen(tmp_path):
    c=config(tmp_path);p=make_plan(c['tasks'],1,1,1,123)
    snap=tmp_path/'snapshot.json';snap.write_text('{"state":"before"}')
    manifest=tmp_path/'scenes.json';manifest.write_text(json.dumps({'task':'stack','scenes':[{'seed':p['development'][0]['seed'],'snapshot':snap.name}]}));c['snapshot_manifest']=str(manifest)
    root=tmp_path/'campaign';freeze_campaign(root,c,p,phase='development');snap.write_text('{"state":"after"}')
    with pytest.raises(ValueError,match='drift'):load_campaign(root)


def test_cost_precheck_matches_existing_ledger_before_reserving_request(tmp_path):
    from capx.experiments.pilot import CostLedger
    opts=dict(max_requests=5,cost_limit_usd=.001,max_tokens=100,wall_seconds=100)
    with pytest.raises(RuntimeError,match='cost'):
        PersistentModelBudget(tmp_path,opts,factory=FakeBudget).query([], 'first')
    assert not (tmp_path/'request-ledger.json').exists()


def test_formal_acceptance_names_and_existing_empty_files_not_enough(tmp_path):
    c=config(tmp_path);evidence=tmp_path/'empty.json';evidence.write_text('{}')
    acceptance=tmp_path/'acceptance.json';acceptance.write_text(json.dumps({'validated_tasks':['stack'],'evidence_files':[str(evidence)]}));c['acceptance_file']=str(acceptance)
    with pytest.raises(ValueError,match='online physical success'):
        freeze_campaign(tmp_path/'formal',c,make_plan(c['tasks'],20,100,2,123),phase='formal',development_root=tmp_path/'missing')


def test_pair_result_numpy_values_are_serializable_without_losing_physics(tmp_path):
    import numpy as np
    c=config(tmp_path);root=tmp_path/'campaign';freeze_campaign(root,c,make_plan(c['tasks'],1,1,1,123),phase='development')
    def runner(**kw):
        return {'episodes':[dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],fault=kw['fault'],group=g,reward=0.,status='task_failure',final_physics={'tcp':np.array([1.,2.,3.])}) for g in kw['groups']]}
    run_campaign(root,runner=runner)
    from capx.experiments.graph_campaign import records
    assert records(root)[0]['final_physics']['tcp']==[1.,2.,3.]


def test_cli_prepare_status_analyze_and_import_are_model_free(tmp_path):
    import subprocess,sys
    c=config(tmp_path);cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c));root=tmp_path/'cli'
    command=[sys.executable,'-B','-m','capx.experiments.graph_campaign']
    done=subprocess.run(command+['prepare-development','--exploratory','--config',str(cfg),'--output',str(root),'--development-scenes','1','--test-scenes','1','--repeats','1'],check=True,capture_output=True,text=True)
    assert json.loads(done.stdout)['planned_episodes']==3
    checked=subprocess.run(command+['status','--campaign',str(root)],check=True,capture_output=True,text=True)
    assert json.loads(checked.stdout)['requests']==0
    analysis=tmp_path/'analysis.json'
    subprocess.run(command+['analyze','--campaign',str(root),'--output',str(analysis)],check=True,capture_output=True,text=True)
    assert json.loads(analysis.read_text())['episodes']==0
    imported=subprocess.run([sys.executable,'-B','-c','import sys;import capx.experiments.graph_campaign;import capx.experiments.graph_statistics;assert "capx.envs" not in sys.modules;assert "capx.llm.providers" not in sys.modules'],capture_output=True,text=True)
    assert imported.returncode==0,imported.stderr


def test_stress_can_schedule_only_the_comparable_graph_groups(tmp_path):
    c=config(tmp_path);c['groups']=['G0','G1'];c['faults']=['support_refresh']
    root=tmp_path/'stress';freeze_campaign(root,c,make_plan(c['tasks'],1,1,1,123),phase='development')
    state=json.loads((root/'state.json').read_text())
    assert set(state['units'][0]['groups'])=={'G0','G1'}
    assert status(root)['planned_episodes']==2


def completed_development_fixture(tmp_path, *, executed=True, mock=False, omit_group=None, different_snapshot=False, generation_failure=False):
    c=config(tmp_path);p=make_plan(c['tasks'],20,1,1,123)
    evidence=tmp_path/'accepted-online.json'
    evidence.write_text(json.dumps({'task':'stack','online_vision':True,'reward':1,'status':'success','snapshot_hash':'accepted-real-physics'}))
    acceptance=tmp_path/'acceptance.json';acceptance.write_text(json.dumps({'validated_tasks':['stack'],'evidence_files':[str(evidence)]}))
    c['acceptance_file']=str(acceptance)
    from capx.experiments.graph_v2 import RUNTIME_ENVIRONMENT
    c['runtime_environment']={key:'fixture' for key in RUNTIME_ENVIRONMENT}
    c['services']={name:'http://local/'+name for name in ('sam3','graspnet','pyroki')}
    cli=tmp_path/'launch.sh';cli.write_text('#!/bin/sh\ntrue\n');c['cli_files']=[str(cli)]
    snapshots=[]
    for scene in p['test']:
        snap=tmp_path/('snapshot-'+str(scene['seed'])+'.json');snap.write_text(json.dumps({'version':1,'task':scene['task'],'seed':scene['seed'],'state_hash':'initial','model_hash':'model','physics':[]}))
        snapshots.append(dict(scene,snapshot=str(snap),state_hash='initial'))
    manifest=tmp_path/'formal-scenes.json';manifest.write_text(json.dumps({'scenes':snapshots}));c['snapshot_manifest']=str(manifest)
    service_source=tmp_path/'service.py';service_source.write_text('SERVICE_VERSION=1\n')
    import hashlib
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'services':{name:{'url':url,'openapi':{'http_status':200,'body_sha256':'schema'},'seed_request_response':True,'processes':[{'pid':1,'source_discovery':'recursive_local_ast_with_initializers','source_files':{str(service_source):{'sha256':hashlib.sha256(service_source.read_bytes()).hexdigest(),'modified_after_process_start':False}},'dependency_versions':{}}]} for name,url in c['services'].items()},'runtime':{'runtime_environment':c['runtime_environment']}}));c['service_inventory']=str(inventory)
    dev=tmp_path/'dev';freeze_campaign(dev,c,p,phase='development')
    def pair(**kw):
        root=kw['output'];rows=[]
        for group in kw['groups']:
            dest=root/'episodes'/group;dest.mkdir(parents=True)
            trace=dest/'trace.jsonl';trace.write_text(json.dumps({'kind':'vision_boundary','seed':kw['seed'],'initial_state_hash':'same','snapshot_replay_verified':True})+'\n'+(json.dumps({'kind':'api','name':'goto_pose','before':{'sim_time':0},'after':{'sim_time':1}})+'\n' if not generation_failure else ''))
            (dest/'initial-evaluation-state.json').write_text(json.dumps({'sim_time':0,'canonical':{'task':kw['task']}}))
            request=root/'initial'/('B0' if group=='B0' else 'shared');request.mkdir(parents=True,exist_ok=True)
            (request/'response.json').write_text(json.dumps({'response_model':'deepseek-v4-pro','finish_reason':'length' if generation_failure else 'stop','content':'partial' if generation_failure else '```python\npass\n```'}))
            (request/'receipt.json').write_text(json.dumps({'http_status':200,'attempts':1,'response_model':'deepseek-v4-pro','response_id':str(kw['seed'])+group,'usage':{'prompt_tokens':10,'completion_tokens':10}}))
            row=dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],fault=kw['fault'],group=group,
                     status='task_failure' if executed else 'scene_preparation_error',reward=0.,executed=executed,
                     mock=mock,online_vision=executed,snapshot_hash=('different' if different_snapshot and group=='G1' else 'same') if executed else None,
                     blocks=[{'block':0,'execution_status':'ok','api_calls':1}] if executed else [],actual_api_calls=1 if executed else 0,
                     trace_path=str(trace),request_paths=[str(request)],code_hash='shared-code' if group!='B0' else 'baseline-code',
                     initial_prompt_hash='shared-prompt' if group!='B0' else 'baseline-prompt',api_docs_hash='grounded' if group!='B0' else 'raw')
            row['stop_reason']='initial_generation_failed' if generation_failure else 'episode_code_block_budget'
            if generation_failure:row.update(status='invalid_program',executed=False,blocks=[],actual_api_calls=0)
            (dest/'result.json').write_text(json.dumps(row));rows.append(row)
        (root/'experiment.json').write_text(json.dumps({'snapshot_hash':'same'}))
        if omit_group:
            (root/'episodes'/omit_group/'result.json').unlink()
        return {'episodes':rows}
    # This fixture installs synthetic runtime values (including EGL='fixture').
    # Keep those inside the mock campaign; later real-renderer tests must see
    # their original environment. Assertions and campaign semantics unchanged.
    from unittest.mock import patch
    with patch.dict('os.environ'):
        run_campaign(dev,runner=pair)
    # Acceptance points to a real-shaped persisted record in the verified
    # development ledger, not the standalone Oracle-like setup file.
    if not generation_failure and executed and not mock and not omit_group and not different_snapshot:
        unit=json.loads((dev/'state.json').read_text())['units'][0]
        result=Path(unit['result']);body=json.loads(result.read_text())
        row=next(r for r in body['episodes'] if r['group']=='G0')
        row.update(status='success',reward=1.,policy_success=True,stop_reason='physical_success')
        persisted=result.parent/'episodes/G0/result.json'
        result.write_text(json.dumps(body));persisted.write_text(json.dumps(row))
        final_acceptance=tmp_path/'recorded-acceptance.json'
        final_acceptance.write_text(json.dumps({'validated_tasks':['stack'],'evidence_files':[str(persisted)]}))
        c['acceptance_file']=str(final_acceptance)
    return c,p,dev


@pytest.mark.parametrize('variant',[{'executed':False},{'mock':True},{'omit_group':'G0'},{'different_snapshot':True}])
def test_formal_gate_rejects_unexecuted_mock_incomplete_or_unpaired_development(tmp_path,variant):
    c,p,dev=completed_development_fixture(tmp_path,**variant)
    with pytest.raises(ValueError,match='20 completed'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_formal_gate_retains_actual_failed_execution_and_freezes_evidence(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    frozen=freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)
    assert frozen['acceptance']['executed_development_scenes']=={'stack':20}
    assert len(frozen['acceptance']['development_evidence_hashes'])>=60


def test_formal_gate_cannot_substitute_new_business_sources_after_development(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    replacement=tmp_path/'replacement.py';replacement.write_text('VERSION=2\n');c['sources']=[str(replacement)]
    with pytest.raises(ValueError,match='source set'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_formal_gate_requires_every_planned_repeat_for_each_counted_scene(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    state=json.loads((dev/'state.json').read_text());extra=dict(state['units'][0],repeat=1,id='unexecuted-repeat',state='pending')
    state['units'].append(extra);(dev/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='20 completed'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_formal_resume_checks_development_evidence_hashes(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path);root=tmp_path/'formal'
    frozen=freeze_campaign(root,c,p,phase='formal',development_root=dev)
    evidence=next(p for p in frozen['acceptance']['development_evidence_hashes'] if p.endswith('trace.jsonl'))
    Path(evidence).write_text('changed historical execution')
    with pytest.raises(ValueError,match='development evidence drift'):load_campaign(root)


def test_source_freeze_follows_transitive_local_imports(tmp_path):
    c=config(tmp_path);src=Path(c['sources'][0]);src.write_text('import helper\n')
    helper=tmp_path/'helper.py';helper.write_text('VALUE=1\n')
    root=tmp_path/'dev';freeze_campaign(root,c,make_plan(c['tasks'],1,1,1),phase='development')
    helper.write_text('VALUE=2\n')
    with pytest.raises(ValueError,match='source drift'):load_campaign(root)


def test_formal_requires_snapshots_service_inventory_and_cli_provenance(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    c.pop('snapshot_manifest',None)
    with pytest.raises(ValueError,match='Snapshot manifest'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_formal_rejects_modified_service_source_since_inventory(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    inventory=json.loads(Path(c['service_inventory']).read_text())
    source=next(iter(inventory['services']['sam3']['processes'][0]['source_files']))
    Path(source).write_text('SERVICE_VERSION=2\n')
    with pytest.raises(ValueError,match='source drift'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_formal_cannot_change_runtime_from_development(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path);c['runtime_environment']['MUJOCO_GL']='different'
    with pytest.raises(ValueError,match='runtime_environment'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_missing_planned_development_repeat_cannot_be_erased_from_ledger(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    state=json.loads((dev/'state.json').read_text());state['units'][0]['seed']+=1;(dev/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='planned development'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_runtime_service_drift_fails_before_formal_execution(tmp_path):
    from capx.experiments.graph_campaign import verify_service_inventory
    c,p,dev=completed_development_fixture(tmp_path);body=json.loads(Path(c['service_inventory']).read_text())
    def changed(name,url,timeout):
        row=json.loads(json.dumps(body['services'][name]));row['openapi']['body_sha256']='changed';return row
    with pytest.raises(ValueError,match='Service runtime drift'):
        verify_service_inventory(c,inspector=changed)


def test_static_robot_calibration_sources_are_frozen_and_checked(tmp_path):
    from capx.experiments.graph_campaign import source_inventory
    import hashlib
    c=config(tmp_path);robot=tmp_path/'robot.xml';robot.write_text('<robot/>')
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps({'sources':[{'role':'robot_mjcf','path':str(robot),'sha256':hashlib.sha256(robot.read_bytes()).hexdigest()}]}))
    c['robot_calibration_file']=str(calibration)
    inventory=source_inventory(c)
    assert str(robot) in inventory and str(calibration) in inventory
    robot.write_text('<changed/>')
    with pytest.raises(ValueError,match='calibration source drift'):source_inventory(c)


def test_real_initial_generation_failure_counts_as_development_attempt_without_selection(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path,generation_failure=True)
    from capx.experiments.graph_campaign import development_execution_evidence,read
    counts,hashes,rejected=development_execution_evidence(dev,read(dev/'state.json'),c)
    assert counts=={'stack':20} and hashes and not rejected
    # Every failed attempt counts; that alone does not prove task readiness.
    with pytest.raises(ValueError,match='acceptance'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_missing_environment_initialization_is_not_development_evidence(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path,generation_failure=True)
    for path in dev.glob('units/*/episodes/*/initial-evaluation-state.json'):path.unlink()
    with pytest.raises(ValueError,match='20 completed'):freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_real_failed_initial_request_is_counted_without_retry(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path,generation_failure=True)
    state=json.loads((dev/'state.json').read_text());unit=state['units'][0];pair_path=Path(unit['result']);body=json.loads(pair_path.read_text())
    row=next(x for x in body['episodes'] if x['group']=='B0')
    row.update(status='system_error',request_paths=[])
    (pair_path.parent/'episodes/B0/result.json').write_text(json.dumps(row));pair_path.write_text(json.dumps(body))
    receipt={'requested_model':'deepseek-v4-pro','attempts':1,'status':'error','http_status':503,'error':'upstream unavailable'}
    directory=pair_path.parent/'failed-request';dest=directory/'llm/001-request';dest.mkdir(parents=True)
    (dest/'receipt.json').write_text(json.dumps(receipt))
    label=unit['id']+f":initial-stack-seed-{unit['seed']}-repeat-{unit['repeat']}-B0"
    (dev/'request-ledger.json').write_text(json.dumps({'requests':[{'label':label,'state':'failed','directory':str(directory),'receipt':receipt}]}))
    from capx.experiments.graph_campaign import development_execution_evidence,read
    counts,hashes,rejected=development_execution_evidence(dev,read(dev/'state.json'),c)
    assert counts=={'stack':20} and hashes and not rejected


def test_formal_cannot_add_unrecorded_service_version_after_development(tmp_path):
    c,p,dev=completed_development_fixture(tmp_path)
    original=json.loads(Path(c['service_inventory']).read_text())
    original['services']['graspnet']['openapi']['body_sha256']='new-service'
    changed=tmp_path/'new-inventory.json';changed.write_text(json.dumps(original));c['service_inventory']=str(changed)
    with pytest.raises(ValueError,match='service identity'):
        freeze_campaign(tmp_path/'formal',c,p,phase='formal',development_root=dev)


def test_freeze_archives_exact_source_closure_bytes_and_detects_archive_corruption(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev'
    frozen=freeze_campaign(root,c,make_plan(c['tasks'],1,1,1),phase='development')
    entry=frozen['source_archive'][c['sources'][0]]
    archived=root/entry['archive_file']
    assert archived.read_bytes()==Path(c['sources'][0]).read_bytes()
    archived.write_text('corrupt archive')
    with pytest.raises(ValueError,match='archive drift'):load_campaign(root)


def test_campaign_source_change_stops_before_next_unit(tmp_path):
    c=config(tmp_path);root=tmp_path/'dev';freeze_campaign(root,c,make_plan(c['tasks'],2,1,1),phase='development')
    calls=[]
    def pair(**kw):
        calls.append(kw['seed']);Path(c['sources'][0]).write_text('changed during episode')
        return {'episodes':[dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],group=g,fault=kw['fault'],reward=0.,status='task_failure') for g in kw['groups']]}
    result=run_campaign(root,runner=pair)
    assert calls and len(calls)==1
    assert result['stop_reason']=='source_changed' and result['complete_units']==1
