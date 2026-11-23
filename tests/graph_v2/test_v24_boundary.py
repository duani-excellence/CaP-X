import pytest
from capx.experiments.native_tasks import actor_functions, physical_reward


def test_vision_cannot_export_oracle_or_raw_motion_capabilities():
    f={k:lambda:None for k in ['oracle_object','oracle_goal','oracle_insertion','open_gripper','goto_pose','sample_grasp_pose','get_object_pose','observe_robot']}
    assert set(actor_functions(f,'vision'))=={'sample_grasp_pose','get_object_pose','observe_robot'}
    assert set(actor_functions(f,'oracle'))==set(f)
    with pytest.raises(ValueError):actor_functions(f,'unknown')


@pytest.mark.parametrize('ok,motion,raw,expected',[(True,0,True,0),(True,1,False,0),(False,1,True,0),(True,1,True,1)])
def test_native_physical_score_not_program_claim(ok,motion,raw,expected):
    assert physical_reward(ok,motion,raw)==expected


def test_contract_skill_is_reachable_through_existing_real_sandbox():
    from capx.execution.skill_contracts import ContractGraph,ContractBroker
    from capx.execution.sandbox import SandboxProcess
    d=dict(schema='capx.contract_task.v1',objects=['x'],goals=['done'],resources=['arm'],
        intents=[dict(id='work',skill='mock_physical_action',requires=['ready'],effects=['done'],resources=['arm'],after=[])])
    c=ContractGraph(d);actions=[]
    broker=ContractBroker(c,{'mock_physical_action':lambda row:actions.append(row['id'])},
        verify=lambda keys:{k:dict(truth='TRUE',evidence={'kind':'MOCK_not_physics'}) for k in keys})
    actor=SandboxProcess({'execute_skill':broker.execute})
    try:
        result=actor.run("execute_skill('work')",timeout=5)
        assert result['ok'] and actions==['work']
        result=actor.run("execute_skill('work')",timeout=5)
        assert not result['ok'] and actions==['work']
        result=actor.run("print(env)",timeout=5)
        assert not result['ok']
    finally:actor.close()
