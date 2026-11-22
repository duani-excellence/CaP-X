"""Graph links must expose the real public sources and current permission."""
import copy
import pytest
from test_grounded_monitor import fixture
from test_phase_edges import _approach
from capx.execution.sandbox import ExecutionInterrupted


def _lift(m,raw):
    _approach(m,raw)
    m.invoke('close_gripper',[],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['lift',{'robot':robot['id']}],{},raw)
    m.invoke('run_plan',[plan],{},raw)
    return plan


def _parents(m,node_id,kind=None):
    return [m.graph.nodes[e['source']] for e in m.graph.edges
            if e['target']==node_id and (kind is None or e['kind']==kind)]


@pytest.mark.parametrize('mode',['G0','G1'])
def test_lift_check_has_explicit_historical_baseline_parent(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    _lift(m,raw)
    check=[n for n in m.graph.nodes.values()
           if n['kind']=='check' and n['data'].get('check')=='lift'][-1]
    assert m.baseline['source_id'] in {n['id'] for n in _parents(m,check['id'])}
    baseline=m.graph.nodes[m.baseline['source_id']]
    # It must reach the actual pregrasp public observation, not inferred truth.
    assert any(m.graph.nodes[i]['kind']=='observation'
               for i in m.graph.ancestors([baseline['id']]))
    assert check['data']['metrics']['baseline_available'] is True


@pytest.mark.parametrize('mode',['G0','G1'])
def test_each_returned_physical_call_has_current_phase_precondition(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    m.invoke('open_gripper',[],{},raw)
    _lift(m,raw)
    physical=[n for n in m.graph.nodes.values() if n['kind']=='call'
              and n['data']['api'] in ('goto_pose','open_gripper','close_gripper','home_pose')]
    assert len(physical)==4
    for call in physical:
        guards=[n for n in _parents(m,call['id'],'precondition')
                if n['kind']=='check' and n['data'].get('check')=='phase_precondition']
        assert len(guards)==1
        assert guards[0]['data']['status']=='PASS'
        assert guards[0]['data']['metrics']['phase']
    assert m.active_precondition is None


@pytest.mark.parametrize('mode',['G0','G1'])
def test_denied_motion_phase_has_failed_check_in_diagnosis(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['retreat',{'robot':robot['id']}],{},raw)
    with pytest.raises(ExecutionInterrupted,match='retreat_requires_release'):
        m.invoke('run_plan',[plan],{},raw)
    checks=[n for n in m.graph.nodes.values() if n['kind']=='check'
            and n['data'].get('check')=='phase_precondition']
    assert checks and checks[-1]['data']['status']=='FAIL'
    assert checks[-1]['id'] in m.graph.ancestors(m.summary()['diagnosis']['roots'])
    assert not calls


def test_g0_new_actor_rechecks_phase_without_replaying_prior_calls():
    m,state,surfaces,calls,raw=fixture('G0')
    _lift(m,raw)
    prior_ids={n['id'] for n in m.graph.nodes.values() if n['kind']=='call' and n['state']=='done'}
    m.end_block_failure('runtime_error','ValueError')
    m.begin_program('recovery uses measured current surfaces',1)
    held=m.invoke('observe_surfaces',['red cube'],{},raw)
    support=m.invoke('observe_surfaces',['green cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['place',{'robot':robot['id'],'held_surface':held['id'],'support_surface':support['id']}],{},raw)
    observed=[]
    def check_guard(name,args,kwargs):
        if name=='goto_pose':
            gate=m.graph.nodes.get(m.active_precondition,{})
            observed.append(copy.deepcopy(gate))
        return raw(name,args,kwargs)
    before=len(calls)
    m.invoke('run_plan',[plan],{},check_guard)
    # The shared controller now ascends, traverses and descends; all are new
    # calls, and every stage must recheck the current permission.
    assert len(calls)==before+3
    assert all(name=='goto_pose' for name,_ in calls[before:])
    assert len(observed)==3 and all(g['valid'] for g in observed)
    assert len({g['id'] for g in observed})==3
    assert all(g['data']['check']=='phase_precondition' for g in observed)
    assert all(g['data']['metrics']['phase']=='holding_verified' for g in observed)
    assert all(m.graph.nodes[i]['state']=='done' for i in prior_ids)
    assert len([e for e in m.graph.events if e['action']=='returned_call' and e['node_id'] in prior_ids])==len(prior_ids)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_place_cannot_use_carried_object_as_its_own_support(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    _lift(m,raw)
    held=m.invoke('observe_surfaces',['red cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    before=len(calls)
    with pytest.raises(ExecutionInterrupted,match='support'):
        m.invoke('plan_skill',['place',{'robot':robot['id'],'held_surface':held['id'],'support_surface':held['id']}],{},raw)
    assert len(calls)==before


@pytest.mark.parametrize('mode',['G0','G1'])
def test_table_region_footprint_is_bound_to_observed_object(mode,monkeypatch):
    m,state,surfaces,calls,raw=fixture(mode,'restack')
    region={'frame':'robot_base','center':[.75,.15,0.],'top_z':0.,'clearance_verified':True}
    monkeypatch.setattr('capx.execution.grounded_feedback.choose_table_region',lambda *a:region)
    selected=m.invoke('observe_table',['green cube'],{},raw)
    assert selected.get('object')=='green cube'
    _lift(m,raw)  # red is now held, so the green-sized cleared region is not its source.
    held=m.invoke('observe_surfaces',['red cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    with pytest.raises(ExecutionInterrupted,match='table_region'):
        m.invoke('plan_skill',['table_place',{'robot':robot['id'],'held_surface':held['id'],'table_region':selected['id']}],{},raw)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_next_grasp_stage_clears_prior_restack_retreat_completion(mode):
    m,state,surfaces,calls,raw=fixture(mode,'restack')
    m.phase='ready';m.retreat_completed=True;m.released=False
    _approach(m,raw)
    assert not m.retreat_completed and not m.released
    m.invoke('close_gripper',[],{},raw)
    assert not m.retreat_completed and not m.released


@pytest.mark.parametrize('mode',['G0','G1'])
def test_old_retreat_cannot_make_held_second_restack_stage_finish(mode):
    m,state,surfaces,calls,raw=fixture(mode,'restack')
    m.phase='holding_verified';m.closed=True;m.lifted=True;m.held_object='red cube'
    m.retreat_completed=True
    assert m.finish_check()['status']=='UNCERTAIN'
