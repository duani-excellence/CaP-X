"""Software-boundary reproductions; these are not claimed physical accidents."""
import copy
import numpy as np
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from test_release_permission_edges import release_ready
from test_inflight_invalidation import fixture

@pytest.mark.parametrize('mode',['G0','G1'])
def test_support_update_in_final_release_capture_revokes_permission(mode):
    m,s,v,c,f,_=release_ready(mode);old=m.capture;changed=[]
    def capture():
        auth=m.graph.nodes.get(m.active_authorization,{})
        if auth.get('data',{}).get('type')=='boundary_release_authorization' and not changed:
            changed.append(True)
            for key in ('center','lo','hi'):v['green'][key][0]+=.02
            for bound in v['green']['top_xy_bounds']:bound[0]+=.02
        return old()
    m.capture=capture
    with pytest.raises(ExecutionInterrupted):m.invoke('open_gripper',[],{},f)
    assert changed and not c and m.closed
    assert any(n['data'].get('type')=='boundary_release_authorization' and n['valid'] for n in m.graph.nodes.values())
    assert any(n['data'].get('check')=='pre_release' and not n['valid'] for n in m.graph.nodes.values())

@pytest.mark.parametrize('mode',['G0','G1'])
def test_no_support_evidence_after_first_place_motion_blocks_suffix(mode):
    m,s,v,c,f,p,support=fixture(mode);old=m.capture
    def capture():
        if c:v.pop('green',None)
        return old()
    m.capture=capture
    with pytest.raises(ExecutionInterrupted,match='evidence'):m.invoke('run_plan',[p],{},f)
    assert len(c)==1 and m.graph.nodes[m.call_history[0]]['state']=='done'
    checks=[e for e in m.graph.events if e['action']=='active_dependency_revalidation']
    assert checks[-1]['status']=='UNCERTAIN'

@pytest.mark.parametrize('mode',['G0','G1'])
def test_partial_consistent_support_can_allow_clear_transit_but_not_descent(mode):
    m,s,v,c,f,p,support=fixture(mode);ref=copy.deepcopy(v['green']);old=m.capture
    # Current visible top points are consistent, but too few for full localization.
    pts=np.tile([.65,0,.05],(40,1));m.color_points_fn=lambda obs,name:pts if name=='green cube' else np.empty((0,3))
    def capture():
        if c:v['green']={**copy.deepcopy(ref),'top_pixels':20}
        return old()
    m.capture=capture
    with pytest.raises(ExecutionInterrupted,match='evidence'):m.invoke('run_plan',[p],{},f)
    assert len(c)==2  # ascend and clear transit; no final descent
    checks=[e for e in m.graph.events if e['action']=='active_dependency_revalidation']
    assert any(e.get('evidence_policy')=='conditional_history_clear_transit' for e in checks)
    assert checks[-1]['status']=='UNCERTAIN'

@pytest.mark.parametrize('mode',['G0','G1'])
def test_table_verifier_recapture_cannot_pair_with_older_held_pose(mode):
    m,s,v,c,f,_=release_ready(mode)
    region=m.graph.add('estimate',{'type':'table_region','center':[.65,0,.05],'top_z':.05})
    m.last_place={'skill':'table_place','inputs':{'table_region':region}}
    def table_check(_):
        # Nested table verification obtains a newer observation.
        if m.release_permission is not None:
            s['robot_cartesian_pos'][0]+=.04
            v['red']['center'][0]+=.04
        m.capture()
        m.record('L2','table_clearance','PASS','fixture_clear_table')
    m._current_table_check=table_check;m.max_local=0
    with pytest.raises(ExecutionInterrupted):m.invoke('open_gripper',[],{},f)
    assert not c
