import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from capx.experiments import graph_v2 as runner


class Actor:
    def __init__(self):self.apis={};self.call_site={};self.closed=0
    def close(self):self.closed+=1

class API:
    def combined_doc(self):return 'raw goto_pose, get_object_pose, open_gripper, close_gripper'

class Env:
    created=[]
    outcomes=[True]
    def __init__(self,task,root,seed,snapshot,api_budget):
        self.task=task;self._task_prompt='Task '+task;self.cfg=SimpleNamespace();self._apis={'api':API()};self._functions={};self._actor=Actor()
        self._full_prompt=[{'role':'system','content':'robot'},{'role':'user','content':'Task '+task}]
        self.reset_info={'initial_state_hash':snapshot['state_hash']};self._api_calls=0;self._request_audit=SimpleNamespace(count=0)
        self._trace_path=Path(root)/'trace.jsonl';self._trace_path.parent.mkdir(parents=True,exist_ok=True);self._trace_path.write_text('')
        self.low_level_env=SimpleNamespace(_frame_timestamps=[]);self.steps=[];self.closed=False;Env.created.append(self)
    def _get_observation(self):return {'robot_joint_pos':[0]*7,'robot_cartesian_pos':[0,0,.2,1,0,0,0,.1],'full_prompt':self._full_prompt,'hidden_truth':999}
    def enable_video_capture(self,*a,**kw):pass
    def evaluation_state(self):return {'private_evaluation_only':999}
    def _write(self,kind,**kw):pass
    def step(self,code):
        self.steps.append(code);self._api_calls+=1;done=Env.outcomes[min(len(self.steps)-1,len(Env.outcomes)-1)]
        info={'execution_status':'ok','api_calls':self._api_calls,'stdout':'visible output','stderr':'','physical_success':done,'private_evaluation_only':999}
        return self._get_observation(),float(done),done,False,info
    def get_video_frames(self):return []
    def close(self):self.closed=True

class Graph:
    events=[]
    def snapshot(self):return {'nodes':[]}
class Monitor:
    def __init__(self,*a,**kw):self.graph=Graph();self.recoveries=0;self.events=[];self.fault_fired=False;self.terminal_reason=None;self.local_attempts=0;self.begin=[];self.phase='ready'
    def capture(self):pass
    def begin_program(self,code,block):self.begin.append(block)
    def finish_check(self):return {'status':'PASS'}
    def summary(self):return {'public_feedback':'only','phase':self.phase,'recovery_scope':{'strategy':'needed'}}
    def end_block_failure(self,*a):pass

class Budget:
    def __init__(self):self.count=0;self.max_requests=100;self.deadline=1e30;self.rows=[];self.calls=[];self.ledger=SimpleNamespace(spent={'deepseek':0})
    def query(self,prompt,label):
        self.count+=1;self.calls.append((copy.deepcopy(prompt),label));self.rows.append({})
        code='print("initial-%d")'%self.count
        return {'content':'```python\n'+code+'\n```','response_model':'deepseek-v4-pro','finish_reason':'stop'},'nonexistent-receipt'

@pytest.fixture
def fake(monkeypatch):
    Env.created=[];Env.outcomes=[True]
    monkeypatch.setattr(runner,'make_environment',lambda *a,**kw:Env(*a,**kw))
    monkeypatch.setattr(runner,'instrument_tools',lambda *a:None)
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:Monitor())
    monkeypatch.setattr(runner,'validated_program',lambda answer:answer['content'].split('```python\n')[1].split('```')[0].strip())
    monkeypatch.setattr(runner,'archive',lambda *a,**kw:{})
    return {'tasks':['stack'],'model':'deepseek-v4-pro','services':{},'budgets':{'api_budget':128,'max_blocks':6,'max_local':3,'max_episode_local':8,'block_seconds':90,'episode_seconds':1800}}


def test_graph_pair_shares_one_initial_b0_independent_and_all_new_envs(tmp_path,fake):
    b=Budget();out=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,b)
    assert len(out['episodes'])==3 and b.count==2
    rows={r['group']:r for r in out['episodes']}
    assert rows['G0']['code_hash']==rows['G1']['code_hash']!=rows['B0']['code_hash']
    assert rows['G0']['initial_prompt_hash']==rows['G1']['initial_prompt_hash']
    assert len(Env.created)==3 and all(e.closed for e in Env.created)
    assert all(r['snapshot_hash']=='same' and r['reward']==1 for r in rows.values())
    assert 'private_evaluation_only' not in json.dumps(b.calls)


def test_block_recovery_retains_env_monitor_budgets_and_new_actor(tmp_path,fake):
    Env.outcomes=[False,True];b=Budget()
    out=runner.run_pair(tmp_path,'lift',2,0,{'state_hash':'same'},fake,b,groups=('G1',))
    assert len(Env.created)==1 and len(Env.created[0].steps)==2
    assert Env.created[0]._actor.closed==2
    assert out['episodes'][0]['replans']==1
    assert out['episodes'][0]['actual_api_calls']==2
    assert 'DO NOT release' in json.dumps(b.calls[0][0])


def test_b0_feedback_has_original_program_output_no_graph_truth(tmp_path,fake):
    Env.outcomes=[False,True];b=Budget()
    runner.run_pair(tmp_path,'restack',2,0,{'state_hash':'same'},fake,b,groups=('B0',))
    text=json.dumps(b.calls[1][0])
    assert 'visible output' in text and 'REGENERATE' in text
    assert 'public_feedback' not in text and 'private_evaluation_only' not in text and 'physical_success' not in text
    assert 'green' in json.dumps(b.calls[0][0])


def test_all_failures_are_preserved_at_block_budget(tmp_path,fake):
    Env.outcomes=[False];fake['budgets']['max_blocks']=2;b=Budget()
    out=runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,b)
    assert len(out['episodes'])==3 and all(r['reward']==0 for r in out['episodes'])
    assert all(r['stop_reason']=='episode_code_block_budget' for r in out['episodes'])
    assert all((tmp_path/'episodes'/g/'result.json').exists() for g in ('B0','G0','G1'))


def test_only_literal_finish_stops_but_not_success(tmp_path,fake):
    Env.outcomes=[False]
    class Finisher(Budget):
        def query(self,prompt,label):
            answer,path=super().query(prompt,label)
            if 'replan' in label:answer['content']='FINISH'
            return answer,path
    out=runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,Finisher(),groups=('G1',))
    assert out['episodes'][0]['reward']==0 and out['episodes'][0]['stop_reason']=='model_finish_claim_without_reward_change'


def test_prepare_snapshot_once_for_three_groups(tmp_path,fake,monkeypatch):
    calls=[]
    def prepare(task,seed,dest):calls.append((task,seed));return {'state_hash':'same'},{'state_hash':'same'}
    monkeypatch.setattr(runner,'prepare_scene',prepare)
    runner.run_pair(tmp_path,'stack',2,0,None,fake,Budget())
    assert calls==[('stack',2)]


def test_provider_failure_initial_not_retried_for_other_graph_group(tmp_path,fake):
    class ErrorBudget(Budget):
        def query(self,prompt,label):self.count+=1;raise ConnectionError('upstream failed')
    b=ErrorBudget();out=runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,b,groups=('G0','G1'))
    assert b.count==1 and all(r['reward']==0 for r in out['episodes'])
    assert all(r['status']=='system_error' for r in out['episodes'])


def test_budgets_and_identity_validation(tmp_path,fake):
    with pytest.raises(ValueError):runner.run_pair(tmp_path,'fake',2,0,{},fake,Budget())
    with pytest.raises(ValueError):runner.run_pair(tmp_path,'stack',2,0,{},fake,Budget(),groups=('G1','G1'))


def test_import_and_cli_help_do_not_cache_vision_service_configuration():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-B','-c','import sys;from capx.experiments import graph_v2;assert "capx.envs" not in sys.modules;assert "capx.integrations.franka.control" not in sys.modules'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    p=subprocess.run([sys.executable,'-B','-m','capx.experiments.graph_v2','--help'],capture_output=True,text=True)
    assert p.returncode==0 and '--groups' in p.stdout


def test_unknown_b0_evidence_fault_is_not_called_an_equal_stress_test(tmp_path,fake):
    with pytest.raises(ValueError,match='equivalent'):
        runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,Budget(),fault='missing_baseline')


def test_global_budget_exhaustion_has_explicit_exit_status(tmp_path,fake):
    class Exhausted(Budget):
        def query(self,prompt,label):raise RuntimeError('Global request budget exhausted')
    out=runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,Exhausted(),groups=('G0','G1'))
    assert all(r['status']=='budget_exceeded' and r['reward']==0 for r in out['episodes'])


def test_scene_preparation_failure_kept_for_every_planned_system(tmp_path,fake,monkeypatch):
    def fail(*a):raise RuntimeError('invalid scene')
    monkeypatch.setattr(runner,'prepare_scene',fail)
    out=runner.run_pair(tmp_path,'stack',2,0,None,fake,Budget())
    assert len(out['episodes'])==3 and all(r['status']=='scene_preparation_error' and not r['executed'] for r in out['episodes'])


def test_ablation_is_recorded_with_effective_group_options(tmp_path,fake):
    fake['graph_options']={'diagnostic_slice':False,'selective_invalidation':True,'selective_recovery':True}
    out=runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,Budget(),groups=('G0','G1'))
    rows={r['group']:r for r in out['episodes']}
    assert rows['G1']['ablation']=='no_diagnostic_slice'
    assert rows['G1']['graph_options']==fake['graph_options']
    assert all(v is False for v in rows['G0']['graph_options'].values())


def test_unknown_ablation_flag_rejected(tmp_path,fake):
    fake['graph_options']={'fake_option':False}
    with pytest.raises(ValueError,match='graph_options'):
        runner.run_pair(tmp_path,'stack',2,0,{'state_hash':'same'},fake,Budget())


def test_operator_interrupt_persists_partial_execution_and_does_not_run_next_group(tmp_path,fake,monkeypatch):
    Env.outcomes=[False]
    original=Env.step
    def interrupted(self,code):
        if len(self.steps):
            self._api_calls+=1
            raise KeyboardInterrupt()
        return original(self,code)
    monkeypatch.setattr(Env,'step',interrupted)
    budget=Budget()
    with pytest.raises(KeyboardInterrupt):
        runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,groups=('G0','G1'))
    result=json.loads((tmp_path/'episodes'/'G0'/'result.json').read_text())
    assert result['status']==result['stop_reason']=='interrupted_by_operator'
    assert result['executed'] is True and result['reward']==0
    assert result['reward_is_last_completed_evaluation'] is True
    assert result['actual_api_calls']==2 and result['llm_requests']==2 and result['new_llm_requests']==2
    assert len(Env.created)==1 and Env.created[0].closed
    assert not (tmp_path/'episodes'/'G1').exists()


def test_interrupt_during_initial_request_is_not_physical_execution(tmp_path,fake):
    class Interrupted(Budget):
        def query(self,prompt,label):self.count+=1;raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Interrupted(),groups=('G1',))
    result=json.loads((tmp_path/'episodes'/'G1'/'result.json').read_text())
    assert result['status']=='interrupted_by_operator' and not result['executed']
    assert result['new_llm_requests']==1 and result['llm_requests']==1


def test_shared_failed_request_is_counted_logically_twice_billed_once(tmp_path,fake):
    class Error(Budget):
        def query(self,prompt,label):self.count+=1;raise ConnectionError('upstream failed')
    out=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Error(),groups=('G0','G1'))
    assert [r['llm_requests'] for r in out['episodes']]==[1,1]
    assert [r['new_llm_requests'] for r in out['episodes']]==[1,0]


def test_replan_budget_precheck_not_counted_as_sent_request(tmp_path,fake):
    Env.outcomes=[False]
    class Capped(Budget):
        def query(self,prompt,label):
            if self.count:raise RuntimeError('Global request budget exhausted')
            return super().query(prompt,label)
    out=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Capped(),groups=('G1',))
    result=out['episodes'][0]
    assert result['replans']==1 and result['llm_requests']==1 and result['new_llm_requests']==1


def test_new_geometry_analysis_and_preflight_sources_are_frozen():
    expected={'capx/execution/grip_offset.py','capx/execution/support_geometry.py','capx/execution/grounded_control.py',
              'capx/experiments/graph_v2_analysis.py','capx/experiments/graph_power.py','capx/experiments/graph_preflight.py'}
    assert expected<=set(runner.SOURCES)
    root=Path(runner.__file__).resolve().parents[2]
    for path in (root/'env_configs'/'graph_feedback').glob('*.json'):
        config=json.loads(path.read_text())
        assert expected<=set(config['sources'])|set(config.get('analysis_sources',[]))


def test_runtime_environment_applied_before_environment_import(fake,monkeypatch):
    monkeypatch.delenv('MUJOCO_GL',raising=False);monkeypatch.delenv('JAX_PLATFORMS',raising=False)
    fake['runtime_environment']={'MUJOCO_GL':'egl','JAX_PLATFORMS':'cpu'}
    runner.configure_services(fake)
    import os
    assert os.environ['MUJOCO_GL']=='egl' and os.environ['JAX_PLATFORMS']=='cpu'
    fake['runtime_environment']={'PYTHONPATH':'unsafe'}
    with pytest.raises(ValueError,match='runtime_environment'):runner.configure_services(fake)


def test_interrupt_after_completed_physical_score_preserves_that_score(tmp_path,fake,monkeypatch):
    class InterruptingMonitor(Monitor):
        def finish_check(self):raise KeyboardInterrupt()
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:InterruptingMonitor())
    with pytest.raises(KeyboardInterrupt):
        runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Budget(),groups=('G1',))
    result=json.loads((tmp_path/'episodes'/'G1'/'result.json').read_text())
    assert result['status']=='interrupted_by_operator' and result['reward']==1
    assert result['reward_is_last_completed_evaluation'] is True


def test_install_monitor_maps_public_motion_once_and_archives_static_calibration(tmp_path,monkeypatch):
    from capx.execution import grounded_feedback,control_frames,bounded_ik
    from capx.experiments.layered_feedback import instrument_tools
    asset=tmp_path/'panda.xml';asset.write_text('<robot/>')
    urdf=tmp_path/'panda.urdf';urdf.write_text('<robot name="fixture"/>')
    import hashlib
    calibration={'schema':'capx.public_robot_calibration.v1','calibration_hash':'calibration-fixture',
                 'sources':[{'role':role,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                            for role,path in [('robot_mjcf',asset),('ik_urdf',urdf)]]}
    logged=[];calls=[];order=[];wrapped=[]
    raw_solver=lambda **kw:[]
    api=SimpleNamespace(ik_solve_fn=raw_solver)
    env=SimpleNamespace(public_robot_calibration=lambda:calibration,_functions={},_actor=Actor(),
                        _apis={'FrankaControlApi':api},_get_observation=lambda:{'robot_joint_pos':[.1]*7},
                        _write=lambda kind,**kw:logged.append((kind,kw)))
    def wrapper(solve,**kw):
        order.append('bounded_ik');wrapped.append((solve,kw));return SimpleNamespace(solve=solve,successful_calls=0,**kw)
    monkeypatch.setattr(bounded_ik,'BoundedIK',wrapper)
    def make_monitor(*args,**kw):order.append('monitor');return SimpleNamespace(**kw)
    monkeypatch.setattr(grounded_feedback,'GroundedMonitor',make_monitor)
    def transform(cal,p,q):calls.append((cal,p,q));return [1.,2.,3.],[1.,0.,0.,0.]
    monkeypatch.setattr(control_frames,'transform_target',transform)
    dest=tmp_path/'episode';dest.mkdir()
    instrument_tools(env,dest)
    instrumented_solver=api.ik_solve_fn
    assert instrumented_solver is not raw_solver
    monitor=runner.install_monitor(env,'G1','stack',dest,{'max_local':6,'max_episode_local':24,'api_budget':192},'none')
    assert order==['bounded_ik','monitor']
    assert len(wrapped)==1 and wrapped[0][0] is instrumented_solver
    kwargs=wrapped[0][1]
    assert kwargs['budget'].max_calls==192*8 and kwargs['budget'].calls==0
    assert kwargs['max_iterations']==8 and kwargs['urdf_path']==str(urdf)
    assert kwargs['observe_joints']()==[.1]*7 and kwargs['calibration_hash']=='calibration-fixture'
    kwargs['emit']({'status':'PASS'});assert ('ik_verification',{'status':'PASS'}) in logged
    assert api.ik_solve_fn.solve is instrumented_solver
    assert monitor.ik_success_count()==0
    api.ik_solve_fn.successful_calls=3
    assert monitor.ik_success_count()==3
    original_p=[.1,.2,.3];original_q=[1.,0.,0.,0.]
    assert monitor.motion_transform(original_p,original_q)==([1.,2.,3.],[1.,0.,0.,0.])
    assert len(calls)==1 and original_p==[.1,.2,.3]
    row=next(v for kind,v in logged if kind=='motion_frame_transform')
    assert row['calibration_hash']=='calibration-fixture'
    assert row['public_target']['position']==original_p and row['mapped_target']['position']==[1.,2.,3.]
    assert json.loads((dest/'robot-calibration.json').read_text())==calibration
    manifest=json.loads((dest/'calibration-assets'/'manifest.json').read_text())
    assert len(manifest)==2
    for source in manifest:
        assert (dest/'calibration-assets'/source['archive_file']).read_bytes()==Path(source['path']).read_bytes()


@pytest.mark.parametrize('missing',['file','ik_urdf_role'])
def test_missing_configured_ik_urdf_fails_closed_before_wrapper_or_monitor(tmp_path,monkeypatch,missing):
    from capx.execution import grounded_feedback,bounded_ik
    import hashlib
    # Config-style sources are real files; the missing path must never invoke a
    # resolver download or fall back to an unverified IK solution.
    asset=tmp_path/'robot.xml';asset.write_text('<robot/>')
    sources=[{'role':'robot_mjcf','path':str(asset),'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}]
    if missing=='file':sources.append({'role':'ik_urdf','path':str(tmp_path/'missing.urdf'),'sha256':'missing'})
    calls=[];raw=lambda **kw:calls.append('solver')
    env=SimpleNamespace(public_robot_calibration=lambda:{'calibration_hash':'fixture','sources':sources},
                        _apis={'FrankaControlApi':SimpleNamespace(ik_solve_fn=raw)})
    monkeypatch.setattr(bounded_ik,'BoundedIK',lambda *a,**kw:calls.append('wrapper'))
    monkeypatch.setattr(grounded_feedback,'GroundedMonitor',lambda *a,**kw:calls.append('monitor'))
    with pytest.raises(FileNotFoundError if missing=='file' else StopIteration):
        runner.install_monitor(env,'G0','stack',tmp_path,{'max_local':6,'max_episode_local':24,'api_budget':192},'none')
    assert not calls and env._apis['FrankaControlApi'].ik_solve_fn is raw


def test_calibration_asset_drift_blocks_monitor_before_execution(tmp_path,monkeypatch):
    asset=tmp_path/'panda.xml';asset.write_text('<different/>')
    env=SimpleNamespace(public_robot_calibration=lambda:{'calibration_hash':'fixture','sources':[{'role':'robot','path':str(asset),'sha256':'wrong'}]})
    with pytest.raises(ValueError,match='calibration source drift'):
        runner.install_monitor(env,'G0','stack',tmp_path,{'max_local':6,'max_episode_local':24},'none')


def test_source_drift_after_first_group_blocks_remaining_groups_without_requests(tmp_path,fake,monkeypatch):
    source=tmp_path/'execution.py';source.write_text('version=1')
    import hashlib
    monkeypatch.setattr(runner,'archive',lambda *a,**kw:{str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
    original=Env.step
    def mutate(self,code):
        result=original(self,code);source.write_text('version=2');return result
    monkeypatch.setattr(Env,'step',mutate);budget=Budget()
    result=runner.run_pair(tmp_path/'run','stack',2,0,{'state_hash':'same'},fake,budget,groups=('B0','G0','G1'))
    assert budget.count==1 and len(Env.created)==1
    rows={r['group']:r for r in result['episodes']}
    assert rows['B0']['reward']==1
    assert rows['G0']['status']==rows['G1']['status']=='source_changed'
    assert rows['G0']['executed'] is False and rows['G1']['llm_requests']==0
    assert (tmp_path/'run/episodes/G1/result.json').exists()


def test_source_edit_during_model_request_blocks_program_execution(tmp_path,fake,monkeypatch):
    source=tmp_path/'execution.py';source.write_text('v1')
    import hashlib
    monkeypatch.setattr(runner,'archive',lambda *a,**kw:{str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
    class EditingBudget(Budget):
        def query(self,prompt,label):
            answer=super().query(prompt,label);source.write_text('v2');return answer
    budget=EditingBudget();out=runner.run_pair(tmp_path/'run','stack',2,0,{'state_hash':'same'},fake,budget,groups=('G0','G1'))
    assert budget.count==1 and len(Env.created)==1 and not Env.created[0].steps
    assert out['episodes'][0]['status']=='source_changed' and out['episodes'][0]['executed'] is False
    assert out['episodes'][1]['status']=='source_changed'


def test_missing_renderer_calibration_is_infrastructure_not_program_failure(tmp_path,fake,monkeypatch):
    from capx.execution.public_camera import DepthCalibrationUnavailable
    class MissingCalibration(Monitor):
        def begin_program(self,code,block):
            raise DepthCalibrationUnavailable('Public renderer depth sampling calibration unavailable')
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:MissingCalibration())
    budget=Budget()
    result=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,groups=('G1',))['episodes'][0]
    assert result['status']=='system_error'
    assert result['executed'] is False and result['policy_success'] is False
    assert result['replans']==0 and budget.count==1
    assert Env.created[0].closed and Env.created[0].steps==[]


@pytest.mark.parametrize('boundary',['final_step','environment_close'])
def test_final_source_drift_revokes_policy_success_but_keeps_physics(tmp_path,fake,monkeypatch,boundary):
    import hashlib
    watched=tmp_path/'watched.py';watched.write_text('VERSION=1\n')
    hashes={str(watched):hashlib.sha256(watched.read_bytes()).hexdigest()}
    if boundary=='final_step':
        original=Env.step
        def changed(self,code):
            watched.write_text('VERSION=2\n')
            return original(self,code)
        monkeypatch.setattr(Env,'step',changed)
    else:
        original=Env.close
        def changed(self):
            watched.write_text('VERSION=2\n')
            return original(self)
        monkeypatch.setattr(Env,'close',changed)
    budget=Budget()
    row,_=runner.run_episode(tmp_path/'episode','stack',21,0,{'state_hash':'same'},fake,budget,'G1',source_hashes=hashes)
    assert row['reward']==1. and row['actual_api_calls']==1
    assert row['policy_success'] is False and row['status']=='source_changed'
    assert row['stop_reason']=='frozen_source_drift'
    assert row['source_postcheck']['unchanged'] is False
    assert row['source_postcheck']['checked_files']==1
    assert row['source_changes']==[str(watched)]
    assert budget.count==1 and row['replan_llm_requests']==0
    assert json.loads((tmp_path/'episode/episodes/G1/result.json').read_text())['status']=='source_changed'


def test_final_source_postcheck_records_verified_unchanged_success(tmp_path,fake):
    import hashlib
    watched=tmp_path/'watched.py';watched.write_text('VERSION=1\n')
    hashes={str(watched):hashlib.sha256(watched.read_bytes()).hexdigest()}
    row,_=runner.run_episode(tmp_path/'episode','stack',21,0,{'state_hash':'same'},fake,Budget(),'G1',source_hashes=hashes)
    assert row['reward']==1. and row['policy_success'] is True
    assert row['source_postcheck']=={'unchanged':True,'checked_files':1}
