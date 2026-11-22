"""Keep the original physical reward separate from bounded policy completion."""
import json
import pytest
from test_runner_v2 import fake, Env, Monitor, Budget
from test_offline_analysis import fixture_episode
from capx.experiments import graph_v2 as runner, graph_statistics as statistics
from capx.experiments.graph_v2_analysis import analyze_episode


@pytest.mark.parametrize('group',['B0','G0','G1'])
def test_post_actor_evaluation_past_deadline_keeps_reward_but_not_policy_success(tmp_path,fake,monkeypatch,group):
    clock=[100.];monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0]);fake['budgets']['episode_seconds']=100
    original=Env.step
    def step(self,code):
        out=original(self,code);clock[0]+=101.;return out
    monkeypatch.setattr(Env,'step',step)
    result=runner.run_pair(tmp_path,'lift',1,0,{'state_hash':'same'},fake,Budget(),groups=(group,))['episodes'][0]
    assert result['reward']==1 and result['info']['physical_success']
    assert result['status']=='budget_exceeded' and not result['policy_success']
    assert result['stop_reason']=='episode_or_global_wall_budget'
    assert result['policy_elapsed_seconds']==101. and statistics.success(result)==0


def test_post_monitor_check_past_deadline_is_not_success(tmp_path,fake,monkeypatch):
    clock=[100.];monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0]);fake['budgets']['episode_seconds']=100
    def check(self):clock[0]+=101.;return {'status':'PASS'}
    monkeypatch.setattr(Monitor,'finish_check',check)
    result=runner.run_pair(tmp_path,'lift',1,0,{'state_hash':'same'},fake,Budget(),groups=('G1',))['episodes'][0]
    assert result['reward']==1 and result['status']=='budget_exceeded' and not result['policy_success']


@pytest.mark.parametrize('variant',['terminal','truncated','raw_error'])
def test_contradictory_physical_success_cannot_override_exit_condition(tmp_path,fake,monkeypatch,variant):
    if variant=='terminal':
        def check(self):self.terminal_reason='local_recovery_budget_exhausted';return {'status':'PASS'}
        monkeypatch.setattr(Monitor,'finish_check',check)
    else:
        original=Env.step
        def step(self,code):
            obs,reward,done,truncated,info=original(self,code)
            if variant=='truncated':truncated=True
            else:info['execution_status']='interrupted'
            return obs,reward,done,truncated,info
        monkeypatch.setattr(Env,'step',step)
    result=runner.run_pair(tmp_path,'lift',1,0,{'state_hash':'same'},fake,Budget(),groups=('G1',))['episodes'][0]
    assert result['reward']==1 and result['status']!='success' and not result['policy_success']
    assert result['stop_reason']!='physical_success' and statistics.success(result)==0


def test_success_at_last_permitted_api_call_is_still_success(tmp_path,fake):
    fake['budgets']['api_budget']=1
    result=runner.run_pair(tmp_path,'lift',1,0,{'state_hash':'same'},fake,Budget(),groups=('G1',))['episodes'][0]
    assert result['actual_api_calls']==1 and result['reward']==1 and result['policy_success']
    assert statistics.success(result)==1


@pytest.mark.parametrize('extra',[{'policy_success':False},{'truncated':True},{'terminal_reason':'unknown_physical_effect_terminal_stop'},
    {'info':{'execution_status':'interrupted'}},{'execution_status':'timeout'},
    {'stop_reason':'episode_code_block_budget'},{'stop_reason':'local_recovery_budget_exhausted'},
    {'policy_success':True,'info':{'execution_status':'system_error'}}])
def test_statistics_defends_against_conflicting_success_fields(extra):
    assert statistics.success(dict(reward=1,status='success',**extra))==0


def test_legacy_success_without_new_policy_fields_remains_valid():
    assert statistics.success({'reward':1.,'status':'success'})==1
    assert statistics.success({'reward':1.,'status':'success','info':{'execution_status':'ok'},'stop_reason':'physical_success'})==1


def test_offline_report_separates_physical_result_from_budget_exit(tmp_path):
    path=fixture_episode(tmp_path);row=json.loads(path.read_text())
    row.update(reward=1.,status='budget_exceeded',stop_reason='episode_or_global_wall_budget',policy_success=False)
    path.write_text(json.dumps(row));out=analyze_episode(path)
    assert out['physical_reward']==1 and out['physical_task_success'] is True
    assert out['policy_success'] is False and out['task_success'] is False


def test_offline_retains_independent_physical_flag_even_if_guarded_reward_is_zero(tmp_path):
    path=fixture_episode(tmp_path);row=json.loads(path.read_text())
    row.update(reward=0.,status='interrupted',info={'physical_success':True,'execution_status':'interrupted'})
    path.write_text(json.dumps(row));out=analyze_episode(path)
    assert out['physical_reward']==0 and out['physical_task_success'] is True
    assert out['task_success'] is False


def test_offline_legacy_valid_success_keeps_old_boolean_shape(tmp_path):
    path=fixture_episode(tmp_path);row=json.loads(path.read_text());row.update(reward=1.,status='success')
    path.write_text(json.dumps(row));out=analyze_episode(path)
    assert out['task_success'] is True and out['policy_success'] is True
    assert out['physical_task_success'] is True
