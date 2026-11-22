from types import SimpleNamespace
import pytest
from test_runner_v2 import Env, Budget, fake
from capx.experiments import graph_v2 as runner

@pytest.mark.parametrize('group',['B0','G0','G1'])
@pytest.mark.parametrize('raises_during_step',[False,True])
def test_native_horizon_stops_all_replanning_without_changing_reward(tmp_path, fake, monkeypatch, group, raises_during_step):
    Env.outcomes=[False]
    class Expired(Env):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.low_level_env.robosuite_env=SimpleNamespace(done=False,timestep=99,horizon=100)
            self.low_level_env._sim_step_count=70
            self.low_level_env.max_steps=100
        def step(self,code):
            value=super().step(code)
            self.low_level_env.robosuite_env.done=True
            self.low_level_env.robosuite_env.timestep=100
            if raises_during_step:raise ValueError('executing action in terminated episode')
            obs,reward,done,truncated,info=value
            info.update(execution_status='runtime_error',stderr='ValueError: executing action in terminated episode')
            return obs,reward,done,False,info
    monkeypatch.setattr(runner,'make_environment',lambda *a,**kw:Expired(*a,**kw))
    budget=Budget()
    result,_=runner.run_episode(tmp_path,'restack',1,0,{'state_hash':'same'},fake,budget,group)
    assert result['stop_reason']=='simulator_episode_terminated'
    assert result['truncated'] and not result['policy_success'] and result['reward']==0
    assert budget.count==1 and len(Env.created[-1].steps)==1
    assert Env.created[-1].closed

def test_generated_error_text_is_not_authoritative_lifecycle(tmp_path, fake, monkeypatch):
    Env.outcomes=[False];fake['budgets']['max_blocks']=2
    class Nonterminal(Env):
        def step(self,code):
            obs,reward,done,truncated,info=super().step(code)
            info.update(execution_status='runtime_error',stderr='ValueError: executing action in terminated episode')
            return obs,reward,done,truncated,info
    monkeypatch.setattr(runner,'make_environment',lambda *a,**kw:Nonterminal(*a,**kw))
    budget=Budget();result,_=runner.run_episode(tmp_path,'stack',1,0,{'state_hash':'same'},fake,budget,'B0')
    assert budget.count==2 and result['stop_reason']=='episode_code_block_budget'
