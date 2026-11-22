import pytest
from capx.execution.grounded_recovery import GroundedRecovery
from test_grounded_monitor import fixture
@pytest.mark.parametrize("mode", ["G0", "G1"])
def test_table_replacement_does_not_raise_without_occlusion_evidence(mode):
    m,*_=fixture(mode=mode,task="restack")
    m.closed=True;m.lifted=True;m.phase="holding_verified";m.held_object="green cube"
    m.skill_intent={"skill":"table_place","inputs":{},"params":{"gap":.006}}
    recipe=GroundedRecovery().propose(m,execution_status="interrupted")
    assert recipe["kind"]=="replace_placement"
    assert "plan_skill('lift'" not in recipe["code"], recipe["code"]

@pytest.mark.parametrize("mode", ["G0", "G1"])
def test_known_table_evidence_gap_revalidates_original_region(mode):
    m,*_=fixture(mode=mode,task="restack")
    m.closed=True;m.lifted=True;m.phase="holding_verified";m.held_object="green cube"
    m.skill_intent={"skill":"table_place","inputs":{},"params":{"gap":.006}}
    m.record("L2","dependency_guard","FAIL","table_clearance_unresolved_nonheld_elevated_points",{"recoverable":True,"region_id":"historical-region","next_action":"reobserve_table_same_region_then_new_plan_or_stop"})
    recipe=GroundedRecovery().propose(m,execution_status="interrupted")
    assert "reobserve_table('historical-region', change_view=True)" in recipe["code"]
    assert "observe_table('green cube')" not in recipe["code"]
    assert "plan_skill('lift'" not in recipe["code"]
