"""Public stage completion is distinct from immutable independent reward."""
import json
import pytest
from test_runner_v2 import fake, Env, Monitor, Budget
from capx.experiments import graph_v2 as runner, graph_statistics
from capx.execution.grounded_feedback import GroundedMonitor


@pytest.mark.parametrize('group',['G0','G1'])
@pytest.mark.parametrize('check',[{'status':'UNCERTAIN'},{'status':'FAIL'},None,{}])
def test_incomplete_or_missing_public_finish_keeps_reward_but_cannot_succeed(tmp_path,fake,monkeypatch,group,check):
    fake['budgets']['max_blocks']=1
    monkeypatch.setattr(Monitor,'finish_check',lambda self:check)
    row=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Budget(),groups=(group,))['episodes'][0]
    assert row['reward']==1 and row['info']['physical_success']
    assert not row['policy_success'] and not row['policy_checks']['public_stage_complete']
    assert row['stop_reason']=='episode_code_block_budget' and row['replans']==0
    assert graph_statistics.success(row)==0


class ReleasedMonitor(Monitor):
    def capture(self):pass
    def record(self,layer,check,status,reason,metrics,**kw):
        return dict(layer=layer,check=check,status=status,reason=reason,metrics=metrics)
    def finish_check(self):
        self.task='stack';self.closed=False
        self.phase='released' if len(self.begin)==1 else 'ready'
        self.released=self.phase=='released';self.retreat_completed=not self.released
        return GroundedMonitor.finish_check(self)
    def summary(self):
        return dict(super().summary(),recovery_instruction='after release fresh retreat only; do not replay completed physical actions',
                    recovery_scope={'never_replay':['completed_release'],'strategy':'fresh_retreat_only'})


RETREAT="robot=read_robot()\nplan=plan_skill('retreat',{'robot':robot['id']})\nrun_plan(plan)"
class RetreatBudget(Budget):
    def query(self,prompt,label):
        answer,path=super().query(prompt,label)
        code=RETREAT if label.startswith('replan-') else 'open_gripper()'
        answer['content']='```python\n'+code+'\n```'
        return answer,path


@pytest.mark.parametrize('group',['G0','G1'])
def test_raw_done_enters_existing_recovery_and_only_new_retreat_program_runs(tmp_path,fake,monkeypatch,group):
    monitor=ReleasedMonitor();monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:monitor)
    budget=RetreatBudget()
    row=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,groups=(group,))['episodes'][0]
    assert row['reward']==1 and row['policy_success'] and row['stop_reason']=='physical_success'
    assert [b['reward'] for b in row['blocks']]==[1.,1.]
    assert [b['policy_success'] for b in row['blocks']]==[False,True]
    assert [b['finish_check']['status'] for b in row['blocks']]==['UNCERTAIN','PASS']
    assert row['replans']==1 and len(Env.created)==1 and Env.created[0]._actor.closed==2
    assert Env.created[0].steps==['open_gripper()',RETREAT]
    context=json.loads((tmp_path/'episodes'/group/'recovery-context-1.json').read_text())['feedback']
    assert context['phase']=='released' and context['completion_check']['status']=='UNCERTAIN'
    assert context['recovery_scope']['never_replay']==['completed_release']
    prompt=json.dumps(budget.calls[1][0])
    assert 'fresh retreat only' in prompt and 'private_evaluation_only' not in prompt
    assert '"physical_success":' not in json.dumps(context)


@pytest.mark.parametrize('exit_kind',['terminal','api_budget','request_budget','finish_claim'])
def test_public_completion_failure_obeys_existing_exit_conditions_without_rewriting_reward(tmp_path,fake,monkeypatch,exit_kind):
    monitor=ReleasedMonitor();monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:monitor)
    class ExitBudget(RetreatBudget):
        def query(self,prompt,label):
            answer,path=super().query(prompt,label)
            if label.startswith('replan-'):answer['content']='FINISH'
            return answer,path
    budget=ExitBudget()
    if exit_kind=='terminal':monitor.terminal_reason='held_surface_evidence_unrecoverable'
    elif exit_kind=='api_budget':fake['budgets']['api_budget']=1
    elif exit_kind=='request_budget':budget.max_requests=1
    row=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,groups=('G1',))['episodes'][0]
    expected={'terminal':'held_surface_evidence_unrecoverable','api_budget':'episode_api_budget',
              'request_budget':'global_model_or_episode_wall_budget','finish_claim':'model_finish_claim_without_reward_change'}
    assert row['reward']==1 and row['info']['physical_success'] and not row['policy_success']
    assert row['stop_reason']==expected[exit_kind] and graph_statistics.success(row)==0
    assert len(Env.created[0].steps)==1


def test_b0_records_absence_of_grounded_completion_contract(tmp_path,fake):
    row=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Budget(),groups=('B0',))['episodes'][0]
    assert row['reward']==1 and row['policy_success']
    assert row['completion_contract']=='unmonitored_original_api'
    assert 'public_stage_complete' not in row['policy_checks']
    assert row['blocks'][0]['finish_check'] is None
