"""Review-only resource-parity and planning checks; no API requests."""
import json
import pytest
from test_runner_v2 import fake,Env,Budget
from test_power import rows
from capx.experiments import graph_v2 as runner,graph_power


def test_cached_initial_generation_does_not_grant_extra_policy_wall_time(tmp_path,fake,monkeypatch):
    clock=[100.];monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    fake['budgets']['episode_seconds']=100;Env.outcomes=[False,True]
    class TimedBudget(Budget):
        def __init__(self):super().__init__();self.deadlines=[]
        def query(self,prompt,label):
            self.deadlines.append((label,self.deadline,clock[0]))
            if label.startswith('initial-'):clock[0]+=60.
            return super().query(prompt,label)
    b=TimedBudget();runner.run_pair(tmp_path,'lift',2,0,{'state_hash':'same'},fake,b,groups=('G0','G1'))
    first=next(d for label,d,now in b.deadlines if 'replan-lift-G0' in label)
    second=next(d for label,d,now in b.deadlines if 'replan-lift-G1' in label)
    assert second==first,'The cached-initial arm received an extra 60 seconds under the same total policy budget'


def test_formal_budget_cli_can_use_the_frozen_eight_block_limit(tmp_path):
    records=tmp_path/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    output=tmp_path/'design.json'
    graph_power.main([str(records),'--output',str(output),'--max-blocks','8'])
    result=json.loads(output.read_text())
    assert result['max_requests']==result['planned_units']*(2+3*7)
    assert result['planned_episodes']==1800 and result['max_requests']==13800


def test_power_uses_frozen_runtime_limits_and_reports_insufficient_campaign_caps(tmp_path):
    records=tmp_path/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    config=tmp_path/'config.json';config.write_text(json.dumps({'budgets':{
        'max_blocks':8,'api_budget':192,'episode_seconds':2400,'max_tokens':32768,
        'max_requests':2400,'cost_limit_usd':500,'wall_seconds':604800}}))
    output=tmp_path/'design.json'
    graph_power.main([str(records),'--output',str(output),'--config',str(config)])
    result=json.loads(output.read_text());resource=result['resources']
    assert result['max_blocks']==8 and result['max_requests']==13800
    assert resource['max_physical_api_calls']==345600
    assert resource['episode_runtime_upper_seconds']==4320000
    assert resource['max_output_tokens']==13800*32768
    assert resource['output_component_upper_usd']==pytest.approx(1790.705664)
    assert resource['full_cost_upper_usd'] is None
    assert resource['coverage']['requests_sufficient'] is False
    assert resource['coverage']['cost_covers_output_component'] is False
    assert resource['coverage']['wall_covers_episode_runtime_upper'] is False
    assert result['planning_config_sha256']


def test_cli_rejects_an_unmatched_block_override(tmp_path):
    records=tmp_path/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    config=tmp_path/'config.json';config.write_text(json.dumps({'budgets':{'max_blocks':8}}))
    with pytest.raises(ValueError,match='max_blocks'):
        graph_power.main([str(records),'--output',str(tmp_path/'out.json'),'--config',str(config),'--max-blocks','6'])


def test_cli_requires_explicit_block_limit_instead_of_hidden_six_default(tmp_path):
    records=tmp_path/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    with pytest.raises(ValueError,match='config|blocks'):
        graph_power.main([str(records),'--output',str(tmp_path/'out.json')])
