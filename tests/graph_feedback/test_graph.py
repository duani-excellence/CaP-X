"""The graph must govern future work, while completed physical calls stay facts."""
import json
import pytest
from capx.execution.dependency_graph import DependencyGraph


def placement_graph(mode='G1'):
    g = DependencyGraph(mode)
    program = g.add('program', {'code': 'place_then_retreat'}, state='done')
    support = g.version('support', 'estimate', {'position': [0, 0, 1]})
    red = g.version('red', 'estimate', {'position': [1, 0, 1]}, invalidate=False)
    target = g.add('plan', {'skill': 'place'}, [(support, 'data')])
    check = g.add('check', {'status': 'PASS'}, [(target, 'data')])
    action = g.add('call', {'api': 'goto_pose'}, [(check, 'precondition'), (program, 'data')])
    return g, program, support, red, target, check, action


def test_support_update_invalidates_old_target_and_action_only():
    g, program, support, red, target, check, action = placement_graph()
    fresh = g.version('support', 'estimate', {'position': [.02, 0, 1]})
    assert g.latest['support'] == fresh
    assert g.nodes[fresh]['version'] == 2
    for node in (support, target, check, action):
        assert not g.nodes[node]['valid']
        with pytest.raises(RuntimeError, match='invalid'): g.require_valid(node)
    assert g.require_valid(red)['valid']
    assert g.require_valid(program)['valid']
    assert g.require_valid(fresh)['valid']


def test_local_alignment_invalidates_retreat_and_revalidated_plan_runs():
    g = DependencyGraph()
    state = g.version('place_reference', 'estimate', {'tcp': [0, 0, 1]})
    release = g.add('call', {'api': 'open_gripper'}, [(state, 'precondition')])
    g.mark_done(release)
    retreat = g.add('plan', {'skill': 'retreat'}, [(state, 'data'), (release, 'precondition')])
    new_state = g.version('place_reference', 'estimate', {'tcp': [.02, 0, .98]}, reason='local_align')
    with pytest.raises(RuntimeError, match='invalid'): g.require_valid(retreat)
    assert g.nodes[release]['state'] == 'done'
    fresh_plan = g.add('plan', {'skill': 'retreat'}, [(new_state, 'data'), (release, 'precondition')])
    verification = g.add('check', {'status': 'PASS'}, [(fresh_plan, 'data')])
    action = g.add('call', {'api': 'goto_pose'}, [(verification, 'precondition')])
    g.require_valid(action)
    g.mark_done(action)
    assert g.nodes[action]['state'] == 'done'


def test_g0_discards_all_unexecuted_work_but_keeps_immutable_and_done_calls():
    g, program, support, red, target, check, action = placement_graph('G0')
    capability = g.add('capability', {'api': 'goto_pose'}, state='done')
    independent_obs = g.add('observation', {'frame': 'red'}, [])
    independent_call = g.add('call', {'api': 'close_gripper'}, [(red, 'data')])
    g.mark_done(independent_call)
    fresh = g.version('support', 'estimate', {'position': [0, 0, 2]})
    assert not g.nodes[red]['valid']
    assert not g.nodes[independent_obs]['valid']
    assert g.nodes[program]['valid'] and g.nodes[capability]['valid']
    assert g.nodes[independent_call]['state'] == 'done'
    assert g.nodes[independent_call]['valid']
    assert g.nodes[fresh]['valid']


def test_completed_actions_cannot_be_replayed_but_propagation_crosses_them():
    g = DependencyGraph()
    estimate = g.version('pose', 'estimate')
    call = g.add('call', {'api': 'close_gripper'}, [(estimate, 'precondition')])
    g.mark_done(call)
    following = g.add('plan', {'skill': 'lift'}, [(call, 'precondition')])
    changed = g.invalidate([estimate], 'pose_changed')
    assert call not in changed and following in changed
    assert g.nodes[call]['state'] == 'done'
    with pytest.raises(RuntimeError, match='already completed'): g.mark_done(call)


def test_initialization_version_without_invalidation_preserves_existing_work():
    g, _, support, _, target, _, _ = placement_graph('G0')
    newer = g.version('support', 'estimate', invalidate=False)
    assert g.nodes[support]['valid'] and g.nodes[target]['valid']
    assert g.nodes[newer]['version'] == 2


def test_diagnosis_contains_reverse_dependency_slice_and_recovery_scope():
    g, program, support, red, target, check, action = placement_graph()
    g.invalidate([support], 'new_measurement')
    assert g.ancestors([action]) == [program, support, target, check, action]
    report = g.diagnose([action])
    ids = [n['id'] for n in report['nodes']]
    assert ids == [program, support, target, check, action]
    assert red in report['preserved']
    assert target in report['invalidated']
    assert report['recovery_scope']['recompute'] == [target]
    assert report['recovery_scope']['revalidate'] == [check]
    assert report['recovery_scope']['blocked_calls'] == [action]
    assert report['edges'] and not report['truncated']
    json.dumps(report)


def test_missing_parent_rejected_without_partial_mutation():
    g = DependencyGraph()
    before = g.snapshot()
    with pytest.raises(KeyError): g.add('plan', parents=[('missing', 'data')])
    assert g.snapshot() == before


def test_invalid_parent_creates_invalid_child_and_cannot_complete():
    g = DependencyGraph()
    parent = g.add('estimate')
    g.invalidate([parent], 'superseded')
    child = g.add('plan', parents=[(parent, 'data')])
    with pytest.raises(RuntimeError, match='invalid'): g.require_valid(child)
    with pytest.raises(RuntimeError, match='invalid'): g.mark_done(child)


def test_event_log_is_append_only_and_snapshot_is_independent():
    emitted = []
    g = DependencyGraph(emit=emitted.append)
    first = g.version('p', 'estimate', {'x': 1})
    original = json.dumps(g.events, sort_keys=True)
    snapshot = g.snapshot()
    g.version('p', 'estimate', {'x': 2})
    assert json.dumps(g.events[:len(snapshot['events'])], sort_keys=True) == original
    snapshot['nodes'][first]['data']['x'] = -99
    assert g.nodes[first]['data']['x'] == 1
    assert emitted == g.events
    json.dumps(g.snapshot())


def test_invalid_edge_kind_and_mode_rejected():
    with pytest.raises(ValueError): DependencyGraph('not-a-group')
    g = DependencyGraph()
    parent = g.add('estimate')
    with pytest.raises(ValueError): g.add('call', parents=[(parent, 'magic')])


def test_diagnosis_is_bounded_and_completed_ids_are_available():
    g = DependencyGraph()
    previous = g.add('program')
    call = g.add('call', {'api': 'close_gripper'}, [(previous, 'data')])
    g.mark_done(call)
    previous = call
    for _ in range(5): previous = g.add('plan', parents=[(previous, 'data')])
    report = g.diagnose([previous], max_nodes=3)
    assert report['truncated'] and len(report['nodes']) == 3
    assert previous in [n['id'] for n in report['nodes']]
    assert call in report['completed_calls']
    selected = {n['id'] for n in report['nodes']}
    assert all(e['source'] in selected and e['target'] in selected for e in report['edges'])


def test_explicit_update_invalidates_older_live_versions_not_only_latest():
    g = DependencyGraph()
    first = g.version('tcp_reference', 'estimate')
    plan = g.add('plan', parents=[(first, 'data')])
    g.version('tcp_reference', 'estimate', invalidate=False)
    fresh = g.version('tcp_reference', 'estimate', reason='local_alignment')
    assert not g.nodes[first]['valid'] and not g.nodes[plan]['valid']
    assert g.nodes[fresh]['valid'] and g.nodes[fresh]['version'] == 3


def test_returned_call_reconciles_actual_effect_without_reviving_descendants():
    g = DependencyGraph()
    observation = g.add('estimate', {'x': 1})
    call = g.add('call', {'api': 'goto_pose'}, [(observation, 'precondition')])
    future = g.add('plan', {'skill': 'retreat'}, [(call, 'precondition')])
    g.invalidate([observation], 'changed_during_call')
    assert not g.nodes[call]['valid']
    g.returned_call(call)
    assert g.nodes[call]['state'] == 'done' and g.nodes[call]['valid']
    assert not g.nodes[future]['valid']
    event = g.events[-1]
    assert event['action'] == 'returned_call'
    assert event['node_id'] == call
    assert event['previous_state'] == 'invalid' and event['previous_valid'] is False
    assert event['state'] == 'done' and event['valid'] is True
    with pytest.raises(RuntimeError, match='already completed'): g.returned_call(call)


def test_returned_call_only_records_calls_and_cannot_repeat_mark_done():
    g = DependencyGraph()
    plan = g.add('plan')
    with pytest.raises(ValueError, match='call'): g.returned_call(plan)
    call = g.add('call', {'api': 'open_gripper'})
    g.mark_done(call)
    with pytest.raises(RuntimeError, match='already completed'): g.returned_call(call)


def test_consume_plan_is_once_only_and_never_claims_physical_completion():
    g = DependencyGraph()
    plan = g.add('plan', {'skill': 'retreat'})
    g.consume_plan(plan)
    assert g.nodes[plan]['data']['execution'] == 'consumed'
    assert g.nodes[plan]['state'] == 'pending'
    assert g.nodes[plan]['valid']
    assert 'physical_success' not in g.nodes[plan]['data']
    assert g.events[-1] == {'sequence': 2, 'action': 'consume_plan', 'node_id': plan,
                           'execution': 'consumed'}
    before = g.snapshot()
    with pytest.raises(RuntimeError, match='consumed'): g.consume_plan(plan)
    assert g.snapshot() == before


def test_consume_plan_rejects_invalid_and_wrong_kind_without_mutation():
    g = DependencyGraph()
    call = g.add('call')
    with pytest.raises(ValueError, match='plan'): g.consume_plan(call)
    plan = g.add('plan')
    g.invalidate([plan], 'obsolete')
    before = g.snapshot()
    with pytest.raises(RuntimeError, match='invalid'): g.consume_plan(plan)
    assert g.snapshot() == before


def test_consume_plan_without_payload_can_be_recorded():
    g = DependencyGraph()
    plan = g.add('plan')
    g.consume_plan(plan)
    assert g.nodes[plan]['data'] == {'execution': 'consumed'}


def test_recovery_scope_includes_affected_suffix_but_evidence_stays_backward():
    g = DependencyGraph()
    green = g.version('green', 'estimate', {'position': [0,0,0]})
    red = g.version('red', 'estimate', {'position': [1,0,0]}, invalidate=False)
    place = g.add('plan', {'skill': 'place'}, parents=[green])
    retreat = g.add('plan', {'skill': 'retreat'}, parents=[place])
    retreat_check = g.add('check', {'status': 'PASS'}, parents=[retreat])
    retreat_call = g.add('call', {'api': 'goto_pose'}, parents=[(retreat_check,'precondition')])
    unrelated = g.add('plan', {'skill': 'old_unrelated'})
    g.invalidate([unrelated], 'independent_historical_failure')
    g.version('green', 'estimate', {'position': [.02,0,0]})
    diagnosis = g.diagnose([place])
    assert [n['id'] for n in diagnosis['nodes']] == [green,place]
    assert diagnosis['recovery_scope']['recompute'] == [place,retreat]
    assert diagnosis['recovery_scope']['revalidate'] == [retreat_check]
    assert diagnosis['recovery_scope']['blocked_calls'] == [retreat_call]
    assert unrelated not in diagnosis['recovery_scope']['impacted']
    assert unrelated not in diagnosis['invalidated']
    assert red in diagnosis['recovery_scope']['preserved']


def test_recovery_scope_crosses_done_call_without_scheduling_replay():
    g = DependencyGraph()
    state = g.version('placement', 'estimate')
    place = g.add('plan', {'skill':'place'}, parents=[state])
    completed = g.add('call', {'api':'goto_pose'}, parents=[place])
    g.returned_call(completed)
    retreat = g.add('plan', {'skill':'retreat'}, parents=[(completed,'precondition')])
    g.version('placement', 'estimate', reason='local_alignment')
    diagnosis = g.diagnose([place])
    assert retreat in diagnosis['recovery_scope']['recompute']
    assert completed not in diagnosis['recovery_scope']['impacted']
    assert completed in diagnosis['recovery_scope']['never_replay']
    assert g.nodes[completed]['state']=='done'
