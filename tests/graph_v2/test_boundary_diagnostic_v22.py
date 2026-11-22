import importlib.util
from pathlib import Path
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_executive_v22 import positioned,execute_recipe
from capx.execution.grounded_recovery import GroundedRecovery

def diagnostic_module():
    path=Path(__file__).resolve().parents[2]/'scripts/local/replay_graph_initial.py'
    spec=importlib.util.spec_from_file_location('v22_diagnostic',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('group',['G0','G1'])
def test_release_pause_is_once_only_and_never_dispatches_an_extra_action(group):
    m,s,v,c,raw,*_=positioned(group)
    diagnostic_module().install_release_pause(m)
    before=len(c)
    with pytest.raises(ExecutionInterrupted,match='diagnostic_boundary_pause'):
        m.invoke('open_gripper',[],{},raw)
    assert len(c)==before and m.closed
    ctrl=GroundedRecovery();recipe=ctrl.propose(m,execution_status='interrupted')
    execute_recipe(m,recipe,raw);ctrl.complete(m,'ok')
    assert m.finish_check()['status']=='PASS'
    assert len([e for e in m.graph.events if e['action']=='diagnostic_release_pause'])==1
