"""V23 software contracts; these MOCK tests are not physical success evidence."""
import ast
import inspect
import copy
import numpy as np
import pytest
from capx.execution.dependency_graph import DependencyGraph


def definition():
    return dict(name='rearranged_names', objects={'payload':{'sensor_key':'red','perception_name':'red cube'},
        'pedestal':{'sensor_key':'green','perception_name':'green cube'}},
        subgoals=[dict(condition='placed:pedestal:table',object='pedestal',destination='table',requires=[]),
                  dict(condition='placed:payload:pedestal',object='payload',destination='pedestal',requires=['placed:pedestal:table'])],
        goals=['placed:payload:pedestal','safe_retreat'])


def test_generic_goals_and_future_intents_exist_before_any_call():
    from capx.execution.task_conditions import TaskConditions
    g=DependencyGraph();c=TaskConditions(g,definition())
    assert c.repair_frontier()[0]=='placed:pedestal:table'
    intents=[n for n in g.nodes.values() if n['kind']=='skill_intent']
    assert {n['data']['object'] for n in intents}=={'payload','pedestal'}
    assert all(n['data']['permission'] is False for n in intents)
    c.set('placed:pedestal:table','TRUE',reason='mock_verified')
    assert c.repair_frontier()[0]=='placed:payload:pedestal'
    assert c.required('approach','payload')==['placed:pedestal:table']


def test_core_executive_has_no_task_object_or_fault_dispatch():
    from capx.execution.graph_executive import GraphExecutive
    source=inspect.getsource(GraphExecutive)
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('task','fault') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Constant) and n.value in ('red cube','green cube','stack','lift','restack') for n in ast.walk(tree))


def test_renamed_goals_choose_remaining_acquisition_without_completed_replay():
    from test_grounded_monitor import fixture
    from capx.execution.task_conditions import TaskConditions
    from capx.execution.graph_executive import GraphExecutive
    m,*_=fixture();m.conditions=TaskConditions(m.graph,definition())
    m.conditions.set('placed:pedestal:table','TRUE',reason='mock_verified')
    d=GraphExecutive().decide(m)
    assert d['kind']=='acquire_remaining' and d['object']=='payload'
    assert d['destination']=='pedestal'


def test_task_dependency_cycle_and_unknown_objects_rejected():
    from capx.execution.task_conditions import TaskConditions
    d=definition();d['subgoals'][0]['requires']=['placed:payload:pedestal']
    with pytest.raises(ValueError):TaskConditions(DependencyGraph(),d)
    d=definition();d['subgoals'][0]['object']='invented'
    with pytest.raises(ValueError):TaskConditions(DependencyGraph(),d)


def profile():
    # Synthetic calibrated fixture. Production never enables this test profile.
    return dict(schema='capx.memory_bounds.v1',calibration_id='mock-only',validated=True,
        frame='robot_base',xy_m=.001,z_m=.001,height_m=.001,
        translation_rate=.005,rotation_rate=.002,age_rate=.00001,
        static_rate=.00001,max_age_seconds=30.,max_travel_m=.5,max_rotation_rad=.5,
        scope='synthetic_test',evidence=['mock_calibration'])


def memory():
    from capx.execution.state_memory import StateMemory
    g=DependencyGraph();mem=StateMemory(g,profile());obs=g.add('observation',{'public':True})
    surface=dict(center=[.5,0,.12],lo=[.48,-.02,.10],hi=[.52,.02,.14],bottom_z=.10,frame='robot_base')
    mem.observe('payload',surface,dict(height_m=.04),source=obs,calibration='c1',stamp=0.,
        evidence_ref='observations/00001.npz',tcp=[.5,0,.12,1,0,0,0,.5],attached=True,grasp_epoch=1)
    return g,mem,surface


def test_expected_occlusion_propagates_motion_but_is_not_current_detection():
    g,mem,_=memory()
    x=mem.estimate('payload',tcp=[.51,0,.12,1,0,0,0,.5],stamp=1.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected')
    assert x['status']=='PREDICTED' and x['surface']['center'][0]==pytest.approx(.51)
    assert x['bounds']['xy_m']>.001 and x['evidence_ref'].endswith('.npz')
    assert x['current_visual_detection'] is False


@pytest.mark.parametrize('event',['touch','possible_contact','slip','grasp_lost','unknown_effect','calibration_changed'])
def test_invalidating_events_expire_downstream_permission(event):
    g,mem,_=memory();source=mem.records['payload']['node']
    p=g.add('plan',{},parents=[source]);mem.invalidate('payload',event)
    assert not g.nodes[p]['valid']
    x=mem.estimate('payload',tcp=[.5,0,.12,1,0,0,0,.5],stamp=1.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected')
    assert x['status']=='UNKNOWN'


@pytest.mark.parametrize('update',[{'calibration':'changed'},{'stamp':99.},{'hold_verified':False},
                                 {'grasp_epoch':2},{'occlusion':'unexplained'}])
def test_missing_model_conditions_cannot_authorize_prediction(update):
    _,mem,_=memory();args=dict(tcp=[.5,0,.12,1,0,0,0,.5],stamp=1.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected');args.update(update)
    assert mem.estimate('payload',**args)['status']=='UNKNOWN'


def test_contradiction_is_not_replaced_by_prediction_or_same_id_refresh():
    g,mem,s=memory();s=copy.deepcopy(s);s['center'][0]+=.04
    mem.contradict('payload',s,source=g.add('observation',{}))
    assert mem.records['payload']['status']=='CONTRADICTED'
    assert mem.estimate('payload',tcp=[.5,0,.12,1,0,0,0,.5],stamp=1.,calibration='c1',
        grasp_epoch=1,hold_verified=True,occlusion='expected')['status']=='UNKNOWN'


def test_release_uses_entire_error_interval_not_nominal_center():
    from capx.execution.state_memory import bounded_place_metrics
    h=dict(center=[.5,0,.076],hi=[.52,.02,.096],bottom_z=.056)
    s=dict(center=[.5,0,.025],hi=[.525,.025,.05]);geom=dict(height_m=.04)
    small=dict(xy_m=.001,z_m=.001,height_m=.001)
    assert bounded_place_metrics(h,s,geom,small,small)['status']=='PASS'
    large=dict(xy_m=.006,z_m=.009,height_m=.002)
    assert bounded_place_metrics(h,s,geom,large,large)['status']=='UNCERTAIN'


def test_uncalibrated_bounds_cannot_enable_prediction():
    from capx.execution.state_memory import StateMemory
    with pytest.raises(ValueError):StateMemory(DependencyGraph(),dict(profile(),validated=False))


def test_slip_latch_cannot_be_reset_by_same_grasp_new_source():
    g,mem,s=memory();mem.invalidate('payload','slip')
    assert mem.observe('payload',s,dict(height_m=.04),source=g.add('observation',{}),
        calibration='c1',stamp=2.,evidence_ref='new.npz',tcp=[.5,0,.12,1,0,0,0,.5],
        attached=True,grasp_epoch=1) is None


def test_unexplained_occlusion_does_not_hide_out_and_back_travel():
    _,mem,_=memory()
    for x in (.6,.5):
        mem.estimate('payload',tcp=[x,0,.12,1,0,0,0,.5],stamp=1.,calibration='c1',
            grasp_epoch=1,hold_verified=True,occlusion='unexplained')
    assert mem.records['payload']['path_m']==pytest.approx(.2)


def test_untrusted_future_declaration_cannot_change_parent_goal_or_bind_other_role():
    from capx.execution.task_conditions import TaskConditions
    g=DependencyGraph();c=TaskConditions(g,definition());program=g.add('program',{})
    with pytest.raises(ValueError):c.declare([dict(skill='place',object='payload',destination='table')],program)
    with pytest.raises(ValueError):c.declare([dict(skill='approach',object='payload',after=[0])],program)
    ids=c.declare([dict(skill='approach',object='payload')],program)
    assert len(ids)==1 and g.nodes[ids[0]]['data']['permission'] is False
    assert c.repair_frontier()[0]=='placed:pedestal:table'


def test_rgbd_occlusion_distinguishes_dropout_and_out_of_frame():
    from capx.execution.state_memory import depth_occlusion
    c=dict(intrinsics=np.array([[100.,0,50],[0,100,50],[0,0,1]]),pose_mat=np.eye(4),
        images={'depth':np.full((100,100),.8)})
    surface=dict(center=[0,0,1])
    assert depth_occlusion({'camera':c},surface,.04)=='expected'
    c['images']['depth'][:]=0
    assert depth_occlusion({'camera':c},surface,.04)=='unexplained'
    c['images']['depth'][:]=.8
    assert depth_occlusion({'camera':c},dict(center=[4,4,1]),.04)=='unexplained'


@pytest.mark.parametrize('mode',['G0','G1'])
def test_monitor_predicted_release_path_keeps_final_guard(mode,monkeypatch):
    """Actual monitor, synthetic sensors and calibrated error model only."""
    import time
    import capx.execution.state_memory as sm
    from test_executive_v22 import positioned
    from capx.execution.sandbox import ExecutionInterrupted
    m,state,visible,calls,raw,*_=positioned(mode)
    m.memory=sm.StateMemory(m.graph,profile())
    m._memory_calibration=lambda:'c1'
    obj=m.held_object;surface=copy.deepcopy(visible['red']);surface['bottom_z']=surface['hi'][2]-.04
    m.memory.observe(obj,surface,dict(height_m=.04),source=m.last_observation,calibration='c1',stamp=time.monotonic(),
        evidence_ref='mock-frame.npz',tcp=state['robot_cartesian_pos'],attached=True,grasp_epoch=m.grasp_epoch)
    # Deterministic fixture lacks a camera; explicitly inject known occlusion.
    monkeypatch.setattr(sm,'depth_occlusion',lambda *a:'expected')
    # This test supplies a calibrated DIRECT support on every capture. The
    # legacy top-screen fixture alone is outside the production face envelope.
    def refresh_support():
        m.memory.observe('green cube',copy.deepcopy(visible['green']),dict(height_m=.05),
            source=m.last_observation,calibration='c1',stamp=time.monotonic(),evidence_ref='mock-support.npz',
            tcp=state['robot_cartesian_pos'],attached=False)
    m._refresh_memory=refresh_support
    m._held_surface=lambda:m._predicted_surface(obj)
    m.invoke('open_gripper',[],{},raw)
    assert any(name=='open_gripper' for name,_ in calls)
    checks=[e for e in m.events if e['check']=='pre_release']
    assert checks and all(e['metrics']['evidence'].startswith('conditional_current') for e in checks)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_predicted_permission_is_revoked_by_last_capture_slip(mode,monkeypatch):
    import time
    import capx.execution.state_memory as sm
    from test_executive_v22 import positioned
    from capx.execution.sandbox import ExecutionInterrupted
    m,state,visible,calls,raw,*_=positioned(mode)
    m.memory=sm.StateMemory(m.graph,profile());m._memory_calibration=lambda:'c1'
    obj=m.held_object;surface=copy.deepcopy(visible['red']);surface['bottom_z']=surface['hi'][2]-.04
    m.memory.observe(obj,surface,dict(height_m=.04),source=m.last_observation,calibration='c1',stamp=time.monotonic(),
        evidence_ref='mock-frame.npz',tcp=state['robot_cartesian_pos'],attached=True,grasp_epoch=m.grasp_epoch)
    monkeypatch.setattr(sm,'depth_occlusion',lambda *a:'expected')
    def refresh_support():
        m.memory.observe('green cube',copy.deepcopy(visible['green']),dict(height_m=.05),
            source=m.last_observation,calibration='c1',stamp=time.monotonic(),evidence_ref='mock-support.npz',
            tcp=state['robot_cartesian_pos'],attached=False)
    m._refresh_memory=refresh_support
    m._held_surface=lambda:m._predicted_surface(obj)
    old=m.capture
    def capture():
        if m.release_permission:m.memory.invalidate(obj,'slip')
        return old()
    m.capture=capture;before=len(calls)
    with pytest.raises(ExecutionInterrupted):m.invoke('open_gripper',[],{},raw)
    assert not any(name=='open_gripper' for name,_ in calls[before:])


def test_table_check_accepts_prediction_provenance_without_losing_safety_gate():
    from test_executive_v22 import holding
    m,*_=holding();node=m.graph.add('state_prediction',{'test':True})
    m._held_surface=lambda:dict(memory_id=node)
    m.table_evidence_fn=lambda *a,**kw:dict(validation=dict(status='UNCERTAIN',reason='mock_missing_clearance'),evidence={})
    result=m._table_evaluation(dict(center=[.5,0],top_z=0))
    assert result[0]['status']=='UNCERTAIN'


def test_partial_visible_contradiction_is_refutation_not_a_pose_measurement():
    _,mem,_=memory();tcp=[.5,0,.12,1,0,0,0,.5]
    assert mem.partial_conflict('payload',np.repeat([[.65,0,.12]],40,axis=0),tcp)
    assert not mem.partial_conflict('payload',np.repeat([[.5,0,.12]],40,axis=0),tcp)


def test_positioned_but_unknown_geometry_selects_evidence_not_repeat_motion():
    from test_executive_v22 import positioned
    from capx.execution.graph_executive import GraphExecutive
    m,*_=positioned();m.conditions.set('aligned:red cube:green cube','UNKNOWN',reason='occluded')
    decision=GraphExecutive().decide(m)
    assert decision['kind']=='verify_placement' and decision['work_kind']=='acquire_evidence'


def test_future_intent_is_bound_to_actual_gripper_call_without_proving_hold():
    from test_grounded_monitor import fixture
    m,_,_,_,raw=fixture();m.enable_task_conditions()
    m.invoke('observe_object',['red cube'],{},raw)
    grasp=m.invoke('observe_grasp',['red cube'],{},raw)
    plan=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{},raw)
    m.invoke('run_plan',[plan],{},raw);m.invoke('close_gripper',[],{},raw)
    intent=m.conditions.intent('close_gripper','red cube')
    assert m.graph.nodes[intent]['data']['execution']=='returned_awaiting_effect_verification'
    assert m.conditions.get('holding:red cube')['truth']=='UNKNOWN'


@pytest.mark.parametrize('swap',[False,True])
def test_goal_role_permutation_is_data_not_a_new_executive_branch(swap):
    from capx.execution.task_conditions import TaskConditions
    from capx.execution.graph_executive import GraphExecutive
    from test_grounded_monitor import fixture
    d=definition()
    if swap:
        # Complete role permutation, including ordering dependencies and goal.
        import json
        d=json.loads(json.dumps(d).replace('payload','TEMP').replace('pedestal','payload').replace('TEMP','pedestal'))
    m,*_=fixture();m.conditions=TaskConditions(m.graph,d)
    first,second=d['subgoals'];m.conditions.set(first['condition'],'TRUE',reason='mock_verified')
    decision=GraphExecutive().decide(m)
    assert decision['object']==second['object']
    assert decision['future_intent_id']==m.conditions.intent('approach',second['object'])


def test_subgoal_declaration_order_is_not_robot_action_order():
    from capx.execution.task_conditions import TaskConditions
    from capx.execution.graph_executive import GraphExecutive
    from test_grounded_monitor import fixture
    d=definition();d['subgoals'].reverse()
    m,*_=fixture();m.conditions=TaskConditions(m.graph,d)
    decision=GraphExecutive().decide(m)
    assert decision['object']=='pedestal' and decision['destination']=='table'
