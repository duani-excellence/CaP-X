import pytest
from capx.multitask.catalog import TASKS,require_ready

def test_seven_different_tasks():
 assert len(TASKS)==7
 assert len({x['module'] for x in TASKS.values()})==7

def test_unknown_task_fail_closed():
 with pytest.raises(ValueError):require_ready('invented')

def test_task_specific_success_not_dense_reward():
 from capx.multitask.runtime import qualifies
 s=dict(raw_success=True,still=True,grasp=[False,False],released=True)
 assert not qualifies('lift',s)
 assert not qualifies('two_arm_lift',s)
 assert qualifies('nut_assembly',s)
 assert qualifies('wipe',s)
 s.update(grasp=[True,True],released=False)
 assert qualifies('lift',s)
 assert qualifies('two_arm_lift',s)
 assert not qualifies('nut_assembly',s)
 s['raw_success']=False
 assert not any(qualifies(k,s) for k in TASKS if k!='stack')
