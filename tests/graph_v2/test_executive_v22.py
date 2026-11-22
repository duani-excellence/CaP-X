"""Real monitor/recipe execution against explicit MOCK public geometry and APIs.

These are deterministic software mechanisms, not online physical success.
"""
import copy
import json
import numpy as np
import pytest
from capx.execution.grounded_recovery import GroundedRecovery
from capx.execution.graph_executive import GraphExecutive
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture

def execute_recipe(m,recipe,raw):
    from capx.execution.grounded_feedback import APIS
    m.begin_program(recipe['code'],m.block+1)
    scope={name:(lambda *a,_name=name,**kw:m.invoke(_name,list(a),kw,raw))
           for name in APIS+['open_gripper','close_gripper','home_pose']}
    exec(recipe['code'],{'__builtins__':{}},scope)


def holding(mode='G1', task='stack', obj='red cube'):
    m,s,v,c,f=fixture(mode=mode,task=task)
    m.enable_task_conditions()
    m.invoke('observe_object',[obj],{},f)
    m.geometry[obj]['orientation_verified']=True
    m.closed=True;m.lifted=True;m.held_object=obj;m.phase='holding_verified'
    m.record('L2','lift','PASS','mock_public_rise',{'object':obj})
    return m,s,v,c,f


def positioned(mode='G1'):
    m,s,v,c,f=holding(mode)
    h=m.invoke('observe_surfaces',['red cube'],{},f)
    t=m.invoke('observe_surfaces',['green cube'],{},f)
    r=m.invoke('read_robot',[],{},f)
    p=m.invoke('plan_skill',['place',{'held_surface':h['id'],'support_surface':t['id'],'robot':r['id']}],{},f)
    m.invoke('run_plan',[p],{},f)
    return m,s,v,c,f,p,t


def test_graph_chooses_pending_release_while_g0_recomputes_place():
    counts={};recipes={}
    for mode in ('G0','G1'):
        m,s,v,c,f,p,t=positioned(mode);done=list(m.call_history)
        ctrl=GroundedRecovery();r=ctrl.propose(m,execution_status='interrupted')
        recipes[mode]=r;start=len(c)
        execute_recipe(m,r,f);out=ctrl.complete(m,'ok')
        counts[mode]=len(c)-start
        assert m.finish_check()['status']=='PASS'
        assert out['progress']
        assert all(m.graph.nodes[i]['state']=='done' for i in done)
        assert p not in out['new_plan_ids']
    assert recipes['G1']['kind']=='release_retreat'
    assert recipes['G0']['kind']=='replace_placement'
    assert counts['G0']>counts['G1']
    assert "plan_skill('place'" not in recipes['G1']['code']


@pytest.mark.parametrize('mode',['G0','G1'])
def test_last_capture_still_blocks_release_despite_graph_alignment(mode):
    m,s,v,c,f,p,t=positioned(mode)
    ctrl=GroundedRecovery();r=ctrl.propose(m,execution_status='interrupted')
    old=m.capture;fire=[]
    def capture():
        if m.release_permission and not fire:
            fire.append(True)
            for key in ('center','lo','hi'):v['green'][key][0]+=.03
        return old()
    m.capture=capture;before=len(c)
    with pytest.raises(ExecutionInterrupted):execute_recipe(m,r,f)
    ctrl.complete(m,'interrupted')
    assert fire and not any(n=='open_gripper' for n,_ in c[before:])


def test_support_version_updates_keep_verified_hold_without_regrasp():
    m,s,v,c,f=holding()
    h=m.invoke('observe_surfaces',['red cube'],{},f);t=m.invoke('observe_surfaces',['green cube'],{},f)
    r=m.invoke('read_robot',[],{},f)
    p=m.invoke('plan_skill',['place',{'held_surface':h['id'],'support_surface':t['id'],'robot':r['id']}],{},f)
    m.skill_intent=m.graph.nodes[p]['data'].copy()
    m.invoke('observe_surfaces',['green cube'],{},f)
    assert not m.graph.nodes[p]['valid']
    recipe=GroundedRecovery().propose(m,execution_status='interrupted')
    assert recipe['kind']=='replace_placement'
    assert 'close_gripper' not in recipe['code'] and "plan_skill('lift'" not in recipe['code']
    hold=m.conditions.get('holding:red cube')
    assert hold['truth']=='TRUE' and hold['id'] in recipe['condition_decision']['preserved_nodes']


def completed_green(mode='G1'):
    m,s,v,c,f=fixture(mode=mode,task='restack');m.enable_task_conditions()
    m.invoke('observe_object',['green cube'],{},f)
    m.geometry['green cube']['orientation_verified']=True
    call=m.graph.add('call',{'api':'open_gripper'});m.graph.returned_call(call);m.call_history.append(call)
    m.conditions.returned('release','green cube','table',call)
    retreat=m.graph.add('call',{'api':'goto_pose'});m.graph.returned_call(retreat);m.call_history.append(retreat)
    fact=m.conditions.returned('retreat','green cube','table',retreat)
    m.conditions.set('safe_retreat','TRUE',evidence=[fact],reason='mock_verified_retreat')
    m.retreat_completed=True;m.phase='ready';m.last_place={'skill':'table_place','object':'green cube'}
    m.capture()
    return m,s,v,c,f


@pytest.mark.parametrize('mode',['G0','G1'])
def test_restack_stage_continuation_does_not_repeat_green_manipulation(mode):
    m,s,v,c,f=completed_green(mode)
    assert m.conditions.get('placed:green cube:table')['truth']=='TRUE'
    assert m.finish_check()['status']=='UNCERTAIN'
    recipe=GroundedRecovery().propose(m,execution_status='ok')
    assert recipe['kind']=='acquire_remaining' and recipe['object']=='red cube'
    assert "observe_grasp('red cube')" in recipe['code']
    assert "observe_grasp('green cube')" not in recipe['code']
    if mode=='G1':assert 'green cube:table' in recipe['condition_decision']['retained_task_stages']


def test_completed_effect_destroyed_does_not_undo_historical_calls():
    m,s,v,c,f=completed_green();historical=list(m.call_history)
    for key in ('center','lo','hi'):v['green'][key][2]+=.08
    m.capture()
    assert m.conditions.get('placed:green cube:table')['truth']=='FALSE'
    d=GraphExecutive().decide(m)
    assert d['repair_goals']==['placed:green cube:table'] and d['object']=='green cube'
    assert all(m.graph.nodes[i]['state']=='done' for i in historical)
    assert not set(historical)&set(d['affected_nodes'])


def test_broken_prior_stage_blocks_later_place_even_while_holding_red():
    m,s,v,c,f=completed_green()
    m.invoke('observe_object',['red cube'],{},f);m.geometry['red cube']['orientation_verified']=True
    m.closed=m.lifted=True;m.held_object='red cube';m.phase='holding_verified'
    m.record('L2','lift','PASS','mock_verified_red_hold')
    for key in ('center','lo','hi'):v['green'][key][2]+=.08
    m.capture()
    d=GraphExecutive().decide(m)
    assert d['repair_goals']==['placed:green cube:table']
    assert not d['recoverable'] and d['selected_actions']==[]
    h=m.invoke('observe_surfaces',['red cube'],{},f);t=m.invoke('observe_surfaces',['green cube'],{},f)
    r=m.invoke('read_robot',[],{},f)
    p=m.invoke('plan_skill',['place',{'held_surface':h['id'],'support_surface':t['id'],'robot':r['id']}],{},f)
    before=len(c)
    with pytest.raises(ExecutionInterrupted,match='task_condition'):m.invoke('run_plan',[p],{},f)
    assert len(c)==before


def test_occluded_completed_effect_requests_evidence_not_repeated_place():
    m,s,v,c,f=completed_green()
    m.geometry_fn=lambda obs:({'red':v['red'],'green':None},0,np.empty((0,3)))
    m.color_points_fn=lambda *a:np.empty((0,3))
    recipe=GroundedRecovery().propose(m,execution_status='interrupted')
    assert recipe['kind']=='revalidate_effect'
    assert 'run_plan' not in recipe['code'] and 'close_gripper' not in recipe['code']


@pytest.mark.parametrize('mode',['G0','G1'])
def test_unknown_hold_repairs_evidence_with_no_duplicate_lift(mode):
    m,s,v,c,f=fixture(mode=mode,task='lift');m.enable_task_conditions()
    m.closed=True;m.held_object='red cube';m.phase='holding_unverified';m.lift_attempts=1
    r=GroundedRecovery().propose(m,execution_status='interrupted')
    assert r['code']=='verify_hold()'


def test_progress_is_condition_establishment_not_fresh_plan_id():
    m,s,v,c,f=completed_green()
    m.conditions.set('placed:green cube:table','UNKNOWN',reason='missing_evidence')
    m.geometry_fn=lambda obs:({'red':v['red'],'green':None},0,np.empty((0,3)))
    m.color_points_fn=lambda *a:np.empty((0,3))
    ctrl=GroundedRecovery(max_no_progress=1);r=ctrl.propose(m,execution_status='interrupted')
    result=ctrl.complete(m,'ok')
    assert not result['progress']
    assert ctrl.propose(m,execution_status='ok') is None


def test_fresh_source_without_restored_effect_does_not_reset_stall_budget():
    m,s,v,c,f=completed_green()
    m.geometry_fn=lambda obs:({'red':v['red'],'green':None},0,np.empty((0,3)))
    m.color_points_fn=lambda *a:np.empty((0,3))
    ctrl=GroundedRecovery(max_no_progress=1)
    recipe=ctrl.propose(m,execution_status='interrupted')
    assert recipe['kind']=='revalidate_effect'
    source=m.graph.add('estimate',{'type':'surface','object':'green cube'})
    m.conditions.set('source:surface:green cube','TRUE',evidence=[source],reason='fresh_handle_only')
    result=ctrl.complete(m,'ok')
    assert not result['progress']
    assert ctrl.propose(m,execution_status='ok') is None


def test_recovery_prompt_bounded_with_all_critical_conditions_retained():
    from capx.experiments.graph_feedback import graph_recovery_prompt
    m,s,v,c,f=holding()
    for i in range(1200):m.graph.add('estimate',{'garbage':list(range(100))})
    m.terminal_reason='unknown_physical_effect_terminal_stop'
    feedback=m.summary()
    feedback['diagnostic_source']='print(1)\n'*10000
    prompt,meta=graph_recovery_prompt({'full_prompt':[{'role':'user','content':'same task'}]},
        'print(1)\n'*10000,feedback,'interrupted','G1',api_doc='same contract')
    text=json.dumps(prompt)
    assert len(text)<40000
    assert 'unknown_physical_effect_terminal_stop' in text
    assert 'holding:red cube' in text and meta['context_budget']['code']['truncated']
    assert len(m.graph.nodes)>1200


def test_approached_recovery_only_closes_and_lifts():
    m,s,v,c,f=fixture();m.enable_task_conditions()
    m.invoke('observe_object',['red cube'],{},f)
    g=m.invoke('observe_grasp',['red cube'],{},f)
    p=m.invoke('plan_skill',['approach',{'grasp':g['id']}],{},f)
    m.invoke('run_plan',[p],{},f)
    recipe=GroundedRecovery().propose(m,execution_status='interrupted')
    assert recipe['kind']=='complete_grasp'
    assert 'observe_grasp' not in recipe['code'] and "plan_skill('approach'" not in recipe['code']
    assert recipe['code'].startswith('close_gripper()')
