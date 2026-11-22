"""Public source updates must stop stale targets at multi-stage API boundaries."""
import copy
import numpy as np
import pytest
from capx.execution.grounded_feedback import GroundedMonitor
from capx.execution.sandbox import ExecutionInterrupted


def fixture(mode="G1"):
    state={'robot_cartesian_pos':np.array([.5,0,.14,0,0,1,0,.5]),'robot_joint_pos':np.zeros(8)}
    surfaces={'red':{'center':[.5,0,.02],'lo':[.48,-.02,0.],'hi':[.52,.02,.04]},
              'green':{'center':[.65,0,.025],'lo':[.625,-.025,0.],'hi':[.675,.025,.05]}}
    for value in surfaces.values():
        value.update(top_xy_bounds=[value['lo'][:2].copy(),value['hi'][:2].copy()],top_pixels=400,pixels=600)
    m=GroundedMonitor(mode,'stack',lambda:state)
    m.geometry_fn=lambda obs:(surfaces,0.,np.empty((0,3)))
    m.closed=True;m.lifted=True;m.phase='holding_verified';m.held_object='red cube';m.close_tcp_z=.02
    m.geometry['red cube']={'height_m':.04,'orientation_verified':True}
    calls=[]
    def raw(name,args,kwargs):
        calls.append((name,copy.deepcopy(args)))
        assert name=='goto_pose'
        delta=np.asarray(args[0])-state['robot_cartesian_pos'][:3]
        state['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
        for key in ('center','lo','hi'):
            surfaces['red'][key]=(np.asarray(surfaces['red'][key])+delta).tolist()
    m.begin_program('place = plan_skill(...); run_plan(place)',0)
    held=m.invoke('observe_surfaces',['red cube'],{},raw)
    support=m.invoke('observe_surfaces',['green cube'],{},raw)
    robot=m.invoke('read_robot',[],{},raw)
    plan=m.invoke('plan_skill',['place',{'held_surface':held['id'],'support_surface':support['id'],'robot':robot['id']}],{},raw)
    return m,state,surfaces,calls,raw,plan,support


def install_update(m,surfaces,calls,plan,support,*,after_calls,kind):
    old_capture=m.capture;fired=[]
    def capture():
        if m.active_plan==plan and len(calls)>=after_calls and not fired:
            fired.append(True)
            if kind=='pose':
                for key in ('center','lo','hi'):surfaces['green'][key][0]+=.02
                for bound in surfaces['green']['top_xy_bounds']:bound[0]+=.02
            else:
                old_capture()
                # Same numerical estimate, but explicitly a new parent-owned source version.
                data=copy.deepcopy(m.graph.nodes[support['id']]['data'])
                m._new('surface:green cube',data)
        return old_capture()
    m.capture=capture
    return fired


@pytest.mark.parametrize('kind',['pose','version'])
def test_update_after_plan_consumed_but_before_first_motion_is_blocked(kind):
    m,s,v,c,raw,plan,support=fixture()
    fired=install_update(m,v,c,plan,support,after_calls=0,kind=kind)
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},raw)
    assert fired and not c
    assert not m.graph.nodes[plan]['valid']
    assert m.closed and m.phase=='holding_verified'


@pytest.mark.parametrize('kind',['pose','version'])
def test_update_between_precision_stages_stops_remaining_suffix_without_replay(kind):
    m,s,v,c,raw,plan,support=fixture()
    fired=install_update(m,v,c,plan,support,after_calls=1,kind=kind)
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},raw)
    assert fired and len(c)==1
    assert len(m.call_history)==1  # Completed ascent remains factual; no replay/rollback.
    assert m.graph.nodes[m.call_history[0]]['state']=='done'
    assert m.closed and m.phase=='holding_verified'


def test_own_returned_motion_effects_do_not_block_unchanged_multistage_place():
    m,s,v,c,raw,plan,support=fixture()
    result=m.invoke('run_plan',[plan],{},raw)
    assert result['status']=='returned'
    assert len(c)==3 and len(m.call_history)==3
    assert m.last_place['skill']=='place'
    assert not m.graph.nodes[plan]['valid']  # Expected own-state invalidation is historical, not permission reuse.
    assert m.closed and m.phase=='holding_verified'
    assert all(m.graph.nodes[node]['state']=='done' for node in m.call_history)


def test_cumulative_subthreshold_drift_is_compared_with_bound_plan_geometry():
    m,s,v,c,raw,plan,support=fixture()
    old=m.capture
    def capture():
        if m.active_plan==plan and c:
            for key in ('center','lo','hi'):v['green'][key][0]+=.003
            for bound in v['green']['top_xy_bounds']:bound[0]+=.003
        return old()
    m.capture=capture
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[plan],{},raw)
    assert len(c)==1
    assert m.phase=='holding_verified'
