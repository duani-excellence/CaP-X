"""V22 logical mechanisms: public evidence, never independent physical reward."""
import json
import pytest
from capx.execution.dependency_graph import DependencyGraph


def conditions(task='restack', mode='G1'):
    from capx.execution.task_conditions import TaskConditions
    g = DependencyGraph(mode)
    return g, TaskConditions(g, task)


def test_trivalent_evidence_expiry_and_new_version():
    g, c = conditions()
    assert c.get('holding:red cube')['truth'] == 'UNKNOWN'
    obs = g.add('observation', {'public': True})
    a = c.set('holding:red cube', 'TRUE', evidence=[obs], reason='verified_comotion')
    p = g.add('plan', {'skill': 'place'}, parents=[(a, 'precondition')])
    call = g.add('call', {'api': 'goto_pose'}, parents=[p]);g.returned_call(call)
    g.invalidate([obs], 'camera_evidence_changed')
    assert c.get('holding:red cube')['truth'] == 'UNKNOWN'
    assert not g.nodes[p]['valid'] and g.nodes[call]['state'] == 'done'
    now = g.add('observation', {'public': True})
    b = c.set('holding:red cube', 'FALSE', evidence=[now], reason='empty_jaws')
    assert b != a and c.get('holding:red cube')['truth'] == 'FALSE'
    assert g.nodes[call]['state'] == 'done' and not g.nodes[p]['valid']


def test_expected_effect_and_returned_action_do_not_establish_condition():
    g, c = conditions()
    call = g.add('call', {'api':'goto_pose'});g.returned_call(call)
    c.returned('table_place', 'green cube', 'table', call)
    assert c.get('placed:green cube:table')['truth'] == 'UNKNOWN'
    assert c.history['positioned:green cube:table']
    assert any(n['kind'] == 'expected_effect' for n in g.nodes.values())


def test_open_return_does_not_claim_verified_loss_of_hold():
    from test_executive_v22 import holding
    m,*_=holding()
    call=m.graph.add('call',{'api':'open_gripper'});m.graph.returned_call(call);m.last_call=call
    m.conditions.motion_effect(m,'open_gripper')
    assert m.conditions.get('holding:red cube')['truth']=='UNKNOWN'


def test_later_motion_invalidates_alignment_even_with_returned_place_history():
    from test_executive_v22 import positioned
    m,*_=positioned();m.summary()
    assert m.conditions.get('aligned:red cube:green cube')['truth']=='TRUE'
    m.conditions.motion_effect(m,'goto_pose')
    assert m.conditions.get('aligned:red cube:green cube')['truth']=='UNKNOWN'
    assert 'positioned:red cube:green cube' in m.conditions.history


def test_restack_intermediate_stage_is_not_full_goal():
    g, c = conditions()
    fact = g.add('verification_fact', {'public':True})
    c.set('placed:green cube:table','TRUE',evidence=[fact],reason='visible_table_relation')
    c.set('safe_retreat','TRUE',evidence=[fact],reason='verified_controller_return')
    assert not c.complete()
    assert 'placed:red cube:green cube' in c.remaining_goals()


def test_frontier_regresses_task_dependencies_and_preserves_completed_stage():
    g,c=conditions()
    goal,path=c.repair_frontier()
    assert goal=='placed:green cube:table'
    assert path==['placed:red cube:green cube','placed:green cube:table']
    f=g.add('verification_fact',{'public':True})
    c.set(goal,'TRUE',evidence=[f],reason='verified_table_relation')
    assert c.repair_frontier()[0]=='placed:red cube:green cube'


def test_support_change_preserves_holding_but_invalidates_dependent_work():
    g, c = conditions()
    support = g.add('estimate', {'object':'green cube'})
    grasp = g.add('verification_fact', {'check':'hold_probe','status':'PASS'})
    hold = c.set('holding:red cube','TRUE',evidence=[grasp],reason='comotion')
    s = c.set('support:green cube','TRUE',evidence=[support],reason='visible_top')
    p = g.add('plan', {'skill':'place'}, parents=[hold,s])
    g.invalidate([support], 'real_public_support_update')
    assert c.get('holding:red cube')['truth'] == 'TRUE'
    assert c.get('support:green cube')['truth'] == 'UNKNOWN'
    assert not g.nodes[p]['valid']


def test_diagnosis_lists_and_payload_remain_bounded():
    g = DependencyGraph()
    for i in range(3000):g.add('estimate', {'huge': 'x'*3000})
    failed = g.add('check', {'status':'FAIL','reason':'critical_release_permission_expired'})
    d = g.diagnose([failed],max_nodes=24)
    assert len(d['preserved']) <= 24
    assert len(json.dumps(d)) < 30000
    assert failed in d['roots']
    assert 'critical_release_permission_expired' in json.dumps(d)
    assert d['diagnostic_stats']['graph_nodes'] == 3001
    assert len(g.snapshot()['nodes']) == 3001


def test_unknown_physical_effect_never_has_automatic_repair():
    from capx.execution.graph_executive import GraphExecutive
    from test_grounded_monitor import fixture
    m,*_=fixture();g,c=conditions('stack');m.graph=g;m.conditions=c
    m.reconciliation_required={'status':'unknown_partial_effects'}
    d=GraphExecutive().decide(m)
    assert d['selected_actions']==[] and d['recoverable'] is False


@pytest.mark.parametrize('height_error,truth',[(0.,'TRUE'),(.03,'FALSE')])
def test_table_effect_does_not_compare_raw_centroid_with_refined_center(height_error,truth):
    """Regression from live Restack frame 249: a 14.6 mm false XY error.

    Both estimates are PUBLIC and from one capture. A horizontal table plane
    has no distinguished XY center; its reference must use the selected
    current object estimate. Height thresholds still reject elevated objects.
    """
    from types import SimpleNamespace
    g,c=conditions();call=g.add('call',{'api':'open_gripper'});g.returned_call(call)
    c.returned('release','green cube','table',call)
    obs=g.add('observation',{'public':True})
    raw={'center':[.39053,.26237,-.10182],'hi':[.39934,.28159,-.08211],
         'top_xy_bounds':[[.38188,.24411],[.39918,.28063]],'top_pixels':80}
    refined={'center':[.37662,.26670,-.10206],'hi':[.39959,.28968,-.0821142+height_error]}
    m=SimpleNamespace(last_observation=obs,closed=False,held_object=None,
        geometry={'green cube':{'frame':'robot_base','shape':'cube','height_m':.03988587,
            'orientation_verified':True}},object_epochs={'green cube':1},table_z=-.12200007,
        _surface=lambda obj:raw if obj=='green cube' else None,
        _support_surface=lambda obj,allow_retained=False:refined if obj=='green cube' else None)
    c.refresh(m)
    assert c.get('placed:green cube:table')['truth']==truth


def test_returned_destination_is_actual_skill_target_not_task_goal():
    g,c=conditions();call=g.add('call',{'api':'goto_pose'});g.returned_call(call)
    # Legal temporary table placement of red must not claim a red-on-green
    # positioning receipt just because that is the task's eventual goal.
    c.returned('table_place','red cube','green cube',call)
    c.returned('release','red cube','green cube',call)
    assert 'positioned:red cube:table' in c.history
    assert 'release:red cube:table' in c.history
    assert 'positioned:red cube:green cube' not in c.history
    assert 'release:red cube:green cube' not in c.history


def test_place_receipt_binds_the_plan_support_source():
    g,c=conditions();support=g.add('estimate',{'type':'surface','object':'green cube'})
    plan=g.add('plan',{'skill':'place','inputs':{'support_surface':support}})
    call=g.add('call',{'api':'goto_pose'},parents=[plan]);g.returned_call(call)
    c.returned('place','red cube','wrong_task_default',call,plan)
    assert 'positioned:red cube:green cube' in c.history
