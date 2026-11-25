from capx.execution.dependency_graph import DependencyGraph


def test_partial_information_is_not_replaced_by_three_valued_conditions():
    from capx.execution.belief_state import BeliefState
    b=BeliefState(DependencyGraph())
    b.record('region:destination',{'status':'UNKNOWN','coverage':.5,'missing_axes':[0]},source='frame1')
    r=b.query('region:destination')
    assert r['values']['coverage']==.5 and r['permission'] is False
    b.record('region:destination',{'status':'UNKNOWN','coverage':1.},source='frame2')
    assert len(b.history['region:destination'])==2
    assert b.history['region:destination'][0]['kind']=='historical_measurement'


def test_history_not_permission_and_contradiction_remains_visible():
    from capx.execution.belief_state import BeliefState
    b=BeliefState(DependencyGraph())
    b.record('region:r',{'status':'PASS','lo':[0,0]},source='frame1')
    b.record('region:r',{'status':'UNKNOWN','history_rejected':'changed'},source='frame2')
    r=b.query('region:r')
    assert r['status']=='UNKNOWN' and r['permission'] is False
    assert r['values']['history_rejected']=='changed'
    assert not b.graph.nodes[b.history['region:r'][0]['id']]['valid']


def test_geometry_diagnostics_are_decision_focused_not_raw_clouds():
    from capx.execution.belief_state import BeliefState
    b=BeliefState(DependencyGraph())
    b.record('object:new',{'center':[1,2,3],'points':[[0,0,0]]*5000,'identity_confidence':None},source='archive')
    q=b.query()
    assert 'points' not in q['object:new']['values']
    assert q['object:new']['values']['point_count']==5000
    assert q['object:new']['source']=='archive'
    b.record('object:count_only',{'points':103},source='archive')
    assert b.query('object:count_only')['values']['point_count']==103


def test_missing_data_does_not_assert_static_state_or_release():
    from capx.execution.belief_state import BeliefState
    b=BeliefState(DependencyGraph())
    b.record('object:new',{'status':'PASS'},source='old')
    b.robot_motion('fresh')
    assert b.query('object:new')['kind']=='historical_measurement'
    assert not b.query('object:new')['permission']


def test_region_memory_indexes_fitted_walls_not_object_surfaces():
    import numpy as np
    from capx.experiments.native_vision import PublicScene
    s=PublicScene(None,None,{})
    s.obs={};s.source='old'
    geom={'lo':[0,0],'hi':[1,1],'wall_planes':[[0,1],[0,1]],'floor_z':0.}
    s.remember_region('target',geom,[{'walls':[(0,np.array([[0,.5,.1],[.5,.5,.1],[1,.5,.1]]))]}])
    assert np.allclose(s.region_memory['target']['walls'][0][1][:,0],[0,1])


def test_actual_plan_start_detects_changes_even_without_version_increment():
    from types import SimpleNamespace
    import numpy as np
    import pytest
    from capx.execution.open_motion import check_plan_start
    obs={'robot_cartesian_pos':np.array([0,0,1,0,0,1,0]),'robot_joint_pos':np.zeros(7),'robot_gripper_joint_pos':np.array([.04,-.04])}
    p={'held':None,'start':{k:v.copy() for k,v in obs.items()}}
    s=SimpleNamespace(held=None,scene=SimpleNamespace(obs=obs));check_plan_start(p,s)
    obs['robot_joint_pos'][0]=.03
    with pytest.raises(ValueError,match='encoder'):check_plan_start(p,s)


def test_sandbox_open_program_calls_real_parent_with_branch_calculation_loop():
    from capx.execution.sandbox import SandboxProcess
    calls=[]
    def plan(skill,params):calls.append((skill,params));return {'status':'READY_FOR_LIVE_VERIFICATION','id':'p'+str(len(calls))}
    with_parent=SandboxProcess({'plan_skill':plan,'run_plan':lambda p:calls.append(p)})
    try:
        result=with_parent.run("for n in range(2):\n    p=plan_skill('deposit', params={'object':'unseen','preferred_xy':[n*0.01,0.1]})\n    if p.get('status') == 'READY_FOR_LIVE_VERIFICATION':\n        run_plan(p['id'])")
        assert result['ok'],result
        assert len(calls)==4 and calls[2][1]['preferred_xy'][0]==.01
    finally:with_parent.close()


def test_sandbox_budget_failure_preserves_attempt_count():
    from capx.execution.sandbox import SandboxProcess
    worker=SandboxProcess({'read_robot':lambda:None})
    try:
        r=worker.run('for i in range(66):\n    read_robot()')
        assert not r['ok'] and r['api_calls']==65
    finally:worker.close()


def test_gripper_return_and_observation_are_not_grasp_verification(tmp_path):
    from types import SimpleNamespace
    from capx.execution.agent_events import EventStore
    from capx.execution.persistent_harness import ExecutionMonitor
    s=EventStore(tmp_path);b=SimpleNamespace(actor=SimpleNamespace(call_site=None),execution_version=34)
    m=ExecutionMonitor(s,lambda:{'source':'before'},b)
    m.invoke('close_gripper',[],{},lambda:None)
    m.observe_returned({'capture':1,'robot':{'gripper':[.01,-.01]}})
    assert s.actions['p0:call1']['state']=='OBSERVED'
    assert s.actions['p0:call1']['result']['verification_pending']
