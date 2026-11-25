import os,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('tool_data_invalid_while_held,failed_approach',[(False,False),(True,False),(True,True)])
def test_official_logiv_executive_dispatches_common_skills(monkeypatch,tmp_path,tool_data_invalid_while_held,failed_approach):
    path=os.environ.get('CAPX_LOGIV_SOURCE')
    if not path:pytest.skip('Pinned official checkout required')
    from capx.experiments.external_logiv import execute_official
    from capx.experiments.contract_models import ModelContractRuntime
    from capx.experiments.native_vision_skills import region_task
    from capx.execution.skill_contracts import SkillBoundaryStop
    class Skills:
        phase='empty';held=None;motion_version=0
        def __init__(self):self.done=set();self.stalled=None;self.injected=False
        def acquire(self,row):
            if failed_approach and not self.injected:
                self.injected=True;self.stalled=row['object']
                raise SkillBoundaryStop('synthetic control stop',{})
            self.held={'object':row['object']};self.phase='held'
        def deposit(self,row):self.held=None;self.done.add(row['object']);self.phase='released'
        def retreat_empty(self,row):self.stalled=None
        def capabilities(self):return dict(acquire=self.acquire,deposit=self.deposit,retreat_empty=self.retreat_empty)
        def stage_conditions(self):
            return {'approach_stalled:'+o:dict(truth='TRUE' if self.stalled==o else 'FALSE',evidence={}) for o in ['unseen','another']}
        def verify(self,keys):
            def truth(k):
                if k.startswith('held:'):return self.held is not None and self.held['object']==k.split(':',1)[1]
                if k.startswith('placed:'):return k.split(':',1)[1] in self.done
                if k=='robot_ready':return self.held is None and self.stalled is None
                if k.startswith('approach_stalled:'):return self.stalled==k.split(':',1)[1]
                if k.startswith('attachment_candidate:'):return False
                return True
            return {k:dict(truth='UNKNOWN' if tool_data_invalid_while_held and self.held and k.startswith(('geometry:','grasp_ready:'))
                           else ('TRUE' if truth(k) else 'FALSE'),evidence={}) for k in keys}
    class Actor:
        def __init__(self,functions=None,**kw):self.functions=functions
        def close(self):pass
        def run(self,code,**kw):
            import ast
            key=ast.literal_eval(ast.parse(code).body[0].value.args[0])
            try:self.functions['execute_skill'](key)
            except SkillBoundaryStop as exc:return {'ok':False,'status':'interrupted','error':str(exc)}
            return {'ok':True,'status':'success'}
    monkeypatch.setattr('capx.execution.sandbox.SandboxProcess',Actor)
    s=Skills();d=region_task({'unseen':'part','another':'different'},grounded=True,causal=True)
    if failed_approach:
        from capx.execution.skill_recovery import recovery_library
        recovery_library(d)
    r=ModelContractRuntime(s,d,group='G0',limits=dict(recovery_version=29,skill_budget=8,episode_seconds=60),emit=lambda *a,**k:None)
    b=SimpleNamespace(root=tmp_path,actor=Actor(),call=None)
    rev=subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
    code="execute_skill('acquire-0')\nexecute_skill('deposit-0')\nexecute_skill('acquire-1')\nexecute_skill('deposit-1')"
    rows,reason=execute_official(r,b,code,{'budgets':{'block_seconds':30},'tasks':{'pick_place':{'skill_budget':8}}},{'path':path,'revision':rev})
    assert reason=='external_LOGIV_GOAL_VERIFIED' and len(rows)==(6 if failed_approach else 4) and r.c.complete()
