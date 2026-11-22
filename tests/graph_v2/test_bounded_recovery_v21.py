import pytest
from capx.execution.grounded_recovery import GroundedRecovery
from capx.execution.sandbox import ExecutionInterrupted
from test_inflight_invalidation import fixture
from test_grounded_monitor import fixture as basic

def execute_recipe(m,recipe,raw):
    m.begin_program(recipe['code'],m.block+1)
    from capx.execution.grounded_feedback import APIS
    scope={name:(lambda *a,_name=name,**kw:m.invoke(_name,list(a),kw,raw)) for name in APIS}
    exec(recipe['code'],{'__builtins__':{}},scope)

@pytest.mark.parametrize('mode',['G0','G1'])
def test_support_recovery_preserves_roles_and_builds_fresh_plan(mode):
    m,s,v,c,raw,p,sup=fixture(mode)
    m.skill_intent=m.graph.nodes[p]['data'].copy()
    m.graph.invalidate([p],'test_old_target')
    controller=GroundedRecovery()
    recipe=controller.propose(m,execution_status='interrupted')
    assert recipe['kind']=='replace_placement' and recipe['object']=='red cube'
    assert recipe['roles']['support']=='green cube' and 'close_gripper' not in recipe['code']
    assert str(p) not in recipe['code']
    assert recipe['selected_sources'] if mode=='G1' else not recipe['selected_sources']
    assert m.local_attempts==1

def test_released_only_retreats_and_reports_actual_new_plan_reuse():
    m,s,v,c,raw=basic();m.phase='released';m.released=True
    m.invoke('read_robot',[],{},raw)
    controller=GroundedRecovery();recipe=controller.propose(m,execution_status='interrupted')
    assert recipe['kind']=='fresh_retreat' and 'open_gripper' not in recipe['code']
    execute_recipe(m,recipe,raw)
    outcome=controller.complete(m,'ok')
    assert m.phase=='ready' and m.retreat_completed
    assert [name for name,_ in c]==['goto_pose']
    assert outcome['actual_reused_source_ids'] and outcome['new_plan_ids']
    assert outcome['progress']

def test_missing_baseline_calls_verify_not_another_lift_or_close():
    m,s,v,c,raw=basic(task='lift');m.phase='holding_unverified';m.closed=True;m.held_object='red cube';m.lift_attempts=1
    ctrl=GroundedRecovery();r=ctrl.propose(m,execution_status='interrupted')
    assert r['code']=='verify_hold()'
    execute_recipe(m,r,raw);ctrl.complete(m,'ok')
    assert m.phase=='holding_verified' and m.hold_probes==1 and m.lift_attempts==1

def test_unknown_physical_effect_and_terminal_budget_are_not_auto_recovered():
    m,*_=basic();ctrl=GroundedRecovery();m.phase='reconciliation_required';m.reconciliation_required={'status':'unknown_partial_effects'}
    assert ctrl.propose(m,execution_status='interrupted') is None
    m.reconciliation_required=None;m.phase='released';m.local_attempts=m.max_episode_local
    assert ctrl.propose(m,execution_status='interrupted') is None

@pytest.mark.parametrize('mode',['G0','G1'])
def test_repeated_no_progress_is_bounded_across_actors(mode):
    m,*_=basic(mode=mode);m.phase='released';m.released=True;ctrl=GroundedRecovery(max_no_progress=1)
    assert ctrl.propose(m,execution_status='interrupted')
    ctrl.complete(m,'interrupted')
    m.begin_program('new actor',1)
    assert ctrl.propose(m,execution_status='interrupted') is None
    assert ctrl.events[-1]['reason']=='no_progress_limit'

def test_lost_grasp_rebuilds_scene_before_regrasp():
    m,*_=basic();m.last_grasp_loss={'object':'red cube'};m.phase='ready'
    ctrl=GroundedRecovery();r=ctrl.propose(m,execution_status='interrupted')
    assert r['kind']=='rebuild_after_grasp_loss'
    assert r['code'].index('observe_object')<r['code'].index('observe_grasp')<r['code'].index('close_gripper')
    assert not r['selected_sources']  # affected object must be current

def test_runner_runs_recipe_in_next_actor_without_model_request(tmp_path,monkeypatch):
    from test_runner_v2 import Env,Monitor,Budget
    from capx.experiments import graph_v2 as runner
    import capx.execution.grounded_recovery as module
    Env.created=[];Env.outcomes=[False,True]
    monkeypatch.setattr(runner,'make_environment',lambda *a,**kw:Env(*a,**kw))
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:Monitor())
    monkeypatch.setattr(runner,'instrument_tools',lambda *a:None)
    monkeypatch.setattr(runner,'archive',lambda *a,**kw:{})
    monkeypatch.setattr(runner,'eager_execution_imports',lambda:None)
    class Recovery:
        pending=None;events=[]
        def __init__(self,**kw):pass
        def complete(self,*a):pass
        def propose(self,*a,**kw):return {'code':'print("recipe")','id':'r1'}
    monkeypatch.setattr(module,'GroundedRecovery',Recovery)
    budget=Budget();config={'budgets':{'bounded_recovery':True,'max_blocks':3}}
    r,_=runner.run_episode(tmp_path,'stack',1,0,{'state_hash':'same'},config,budget,'G1')
    assert r['policy_success'] and budget.count==1 and r['replans']==0
    assert len(Env.created[0].steps)==2 and Env.created[0]._actor.closed==2
    import json
    b=json.loads((tmp_path/'episodes/G1/block-1.json').read_text())
    assert b['code_origin']=='bounded_skill_recipe' and b['model_request'] is None

def test_diagnostic_surface_envelopes_are_not_reusable_skill_handles():
    m,s,v,c,raw,p,sup=fixture()
    m.skill_intent=m.graph.nodes[p]['data'].copy()
    invalid=[]
    for name in ('red cube','green cube'):
        invalid.append(m.graph.add('estimate',{'type':'surface','object':name,
            'evidence_role':'pre_release_held','surface':None}))
    ctrl=GroundedRecovery();recipe=ctrl.propose(m,execution_status='interrupted')
    selected={v['id'] for v in recipe['selected_sources'].values()}
    assert not selected.intersection(invalid)
    assert sup['id'] in selected

def test_recovery_keeps_declared_skill_parameters_with_fresh_sources():
    m,s,v,c,raw,p,sup=fixture();m.skill_intent=m.graph.nodes[p]['data'].copy()
    m.skill_intent['params']={'gap':.008,'z_approach':.12}
    recipe=GroundedRecovery().propose(m,execution_status='interrupted')
    assert "'gap': 0.008" in recipe['code'] and "'z_approach': 0.12" in recipe['code']
