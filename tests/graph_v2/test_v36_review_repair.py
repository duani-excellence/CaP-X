"""Reviewed counterexamples; fixtures are explicit, never physical acceptance."""
import copy
import time
from types import SimpleNamespace as S
import pytest
from capx.execution.persistent_harness import ProgressTracker, PersistentHarness, TaskSupervisor, RecoveryOrchestrator, ExecutionMonitor, classify_failure
from capx.execution.agent_events import EventStore
from capx.execution.skill_contracts import ContractGraph, SkillBoundaryStop
from capx.execution.open_recovery import OpenRecoveryAPIs


def state(truth):
    return {'remaining_goals':['placed:B'], 'conditions':{'placed:A':{'truth':truth}}}


def test_restored_obligation_is_progress_but_chatter_is_bounded():
    p=ProgressTracker();p.update(state('TRUE'),{})
    for _ in range(3):p.update(state('UNKNOWN'),{})
    assert p.update(state('TRUE'),{})['changed']
    gains=[]
    for _ in range(12):
        p.update(state('UNKNOWN'),{})
        gains.append(p.update(state('TRUE'),{})['changed'])
    assert not any(gains[-4:]) and p.no_progress>=4


def test_negative_required_condition_is_verified_progress():
    p=ProgressTracker();s=state('UNKNOWN')
    s['intents']=[{'requires_false':['occupied']}]
    s['conditions']['occupied']={'truth':'UNKNOWN'}
    p.update(s,{})
    s['conditions']['occupied']={'truth':'FALSE'}
    assert p.update(s,{})['changed']
    s['conditions']['occupied']['node']='new-id'
    assert not p.update(s,{})['changed']


def test_final_progress_is_after_evidence_refresh_once():
    s=state('UNKNOWN');r=S(state=lambda:s,open_runtime=True,skills=S(),open_tools=S(information={},search_results=[]),last_error='',terminal_reason=None,c=S(quarantined=False),broker=S(max_calls=30,calls=0),limit=30,refreshes=0)
    b=S(spec={'episode_seconds':100},started=time.monotonic(),calls=0)
    h=PersistentHarness.__new__(PersistentHarness);h.r=r
    h.supervisor=TaskSupervisor(b,r,S(max_requests=4,count=0),3)
    for _ in range(4):h.supervisor.progress()
    def prepare():s['conditions']['placed:A']['truth']='TRUE';return True
    r.prepare_recovery=prepare;r.deterministic_code=lambda:'execute_skill("continue")'
    h.store=S(unresolved=lambda:[],append=lambda *a:None);h.orchestrator=RecoveryOrchestrator(4);h.history=[];h.checkpoint=lambda a:None
    decision,code=h.next({'status':'ok'})
    assert code and decision['level']=='R1' and h.supervisor.no_progress==0


def test_observe_hold_unknown_synchronizes_without_dropping_identity(tmp_path):
    c=ContractGraph({'schema':'capx.contract_task.v1','objects':['widget'],'resources':['arm'],'goals':['done'],'intents':[{'id':'put','skill':'deposit','requires':['held:widget'],'effects':['done'],'resources':['arm']}]})
    c.observe('held:widget','TRUE',{'source':'old'});c.observe('unrelated','TRUE',{})
    scene=S(bindings={'widget':'public descriptor'},source='fresh',frame=lambda:None)
    s=S(scene=scene,phase='held_verified',held={'object':'widget'},anchors={'source':{}},effects={})
    def stopped():raise SkillBoundaryStop('contradictory registration',{'held:widget':{'truth':'UNKNOWN','evidence':{'source':'fresh'}}})
    s.current_held=stopped;r=S(c=c,skills=s,refreshes=0,limit=10)
    a=OpenRecoveryAPIs(r,EventStore(tmp_path),True)
    assert a.observe_object('widget')['status']=='UNKNOWN'
    assert c.truth('held:widget')=='UNKNOWN' and s.phase=='closed_unverified'
    assert s.held['object']=='widget' and c.truth('unrelated')=='TRUE'


@pytest.mark.parametrize('error',['AttributeError','KeyError','IndexError','ZeroDivisionError'])
def test_python_errors_are_not_physical(error):
    assert classify_failure({'status':'error','stderr':error+': invalid code'})=='PROGRAM_ERROR'


def test_predispatch_failure_does_not_quarantine(tmp_path):
    b=S(actor=S(call_site=None),execution_version=36,spec={},dispatch_receipt=lambda:{'started':0,'returned':0})
    store=EventStore(tmp_path);m=ExecutionMonitor(store,lambda:{'source':'same','robot':{}},b)
    def fail():raise ValueError('planner failed')
    with pytest.raises(ValueError):m.invoke('goto_pose',[],{},fail)
    assert not store.unresolved()
    assert m.last_failure['dispatch_state']=='NOT_DISPATCHED'


def test_actual_unreconciled_dispatch_stays_quarantined(tmp_path):
    count=[0];b=S(actor=S(call_site=None),execution_version=36,spec={},dispatch_receipt=lambda:{'started':count[0],'returned':0})
    store=EventStore(tmp_path);m=ExecutionMonitor(store,lambda:{'source':'same','robot':{}},b)
    def fail():count[0]+=1;raise RuntimeError('driver lost')
    with pytest.raises(RuntimeError):m.invoke('goto_pose',[],{},fail)
    assert store.unresolved() and m.last_failure['dispatch_state']=='DISPATCHED_UNRECONCILED'


def test_capability_return_links_original_source_and_keeps_done_call(tmp_path):
    from capx.execution.semantic_capabilities import CapabilityRegistry
    from capx.execution.dependency_graph import DependencyGraph
    from test_v36_semantic import proposal
    g=DependencyGraph();origin=g.add('estimate',{'value':2.})
    def measure(label):return {'value':2.,'id':origin,'result_ref':origin,'dependency_refs':[origin]}
    r=CapabilityRegistry(tmp_path,{'measure':measure},graph=g,tool_effects={'measure':[]},episode='one')
    row=r.register(proposal());result=r.run(row['id'],{'label':'item'})
    call=g.nodes[result['node_id']]['data']['calls'][0]
    future=g.add('plan',{},parents=[result['output_node']])
    g.invalidate([origin],reason='new contradictory measurement')
    assert not g.nodes[call['result_node']]['valid']
    assert not g.nodes[result['output_node']]['valid'] and not g.nodes[future]['valid']
    assert g.nodes[call['call_node']]['state']=='done'


def test_record_ids_do_not_establish_generalization(tmp_path):
    from capx.execution.semantic_capabilities import CapabilityRegistry
    from capx.execution.dependency_graph import DependencyGraph
    from test_v36_semantic import proposal
    def measure(label):return {'value':2.,'id':'record-one'}
    a=CapabilityRegistry(tmp_path,{'measure':measure},graph=DependencyGraph(),tool_effects={'measure':[]},episode='one')
    row=a.register(proposal());a.run(row['id'],{'label':'item'})
    b=CapabilityRegistry(tmp_path,{'measure':measure},graph=DependencyGraph(),tool_effects={'measure':[]},episode='two',callback=lambda n,a,k:{'value':2.,'id':'record-two'})
    result=b.run(row['id'],{'label':'item'})
    assert result['status']=='VERIFIED' and len(result['verified_contexts'])==1


def test_trajectory_partial_dispatch_invalidates_and_increments_once(tmp_path):
    import numpy as np
    from capx.execution.dependency_graph import DependencyGraph
    g=DependencyGraph();old=g.add('estimate',{});plan=g.add('plan',{})
    epochs=[];counter=[0]
    scene=S(source='new',frame=lambda:None,obs={'robot_cartesian_pos':np.array([0.,0.,.5,1.,0.,0.,0.]),'robot_gripper_joint_pos':[.04,-.04]})
    s=S(scene=scene,held=None,phase='approach_stopped',stalled={'object':'widget'},effects={},motion_version=7)
    c=S(quarantined=False,locks={},graph=g,observe=lambda *a,**k:None)
    r=S(skills=s,c=c,library=[],planner=S(motion_epoch=epochs.append))
    a=OpenRecoveryAPIs(r,EventStore(tmp_path),True)
    a.monitor=S(boundary=S(dispatch_receipt=lambda:{'started':counter[0],'returned':counter[0]}))
    a.results={old:{'type':'robot'}}
    a.plans[plan]={'kind':'trajectory','used':False,'inputs':{},'trajectory':{},'target':[0,0,.5],'q':[1,0,0,0],'held':None,'empty_scope':{'reconcile_stalled':True}}
    def partial(*args):counter[0]+=1;raise SkillBoundaryStop('later waypoint stopped',{})
    s.motion=partial
    with pytest.raises(SkillBoundaryStop):a.run_plan(plan)
    assert s.motion_version==8 and not g.nodes[old]['valid'] and not c.locks
    assert epochs==[8] and a.plans[plan]['used']


def test_motion_stop_propagates_explicit_effects(tmp_path):
    from capx.execution.vision_skills import RegionSkills
    s=RegionSkills.__new__(RegionSkills);s.held={'object':'w'};s.effects={};s.phase='held_verified'
    s.scene=S(source='fresh');s.active_condition='held:w';updates=[]
    s.validation_sink=lambda effects,reason:updates.append(effects)
    with pytest.raises(SkillBoundaryStop):s.stop('changed attachment')
    assert updates[-1]['held:w']['truth']=='UNKNOWN'


def symbolic_runtime(group='G1',ablation='none'):
    from capx.experiments.contract_models import ModelContractRuntime
    d={'schema':'capx.contract_task.v1','objects':['widget'],'resources':['arm'],'goals':['done'],
       'condition_specs':{k:{'observable':True} for k in ('ready','held','done')},
       'intents':[dict(id='take',skill='take',requires=['ready'],effects=['held'],deletes=['ready'],resources=['arm']),dict(id='put',skill='put',requires=['held'],effects=['done','ready'],deletes=['held'],resources=['arm'])]}
    d['skill_library']=[dict(name=r['skill'],bindings=[{}],contract={k:v for k,v in r.items() if k!='id'}) for r in d['intents']]
    physical={'ready':'TRUE','held':'FALSE','done':'FALSE'}
    def skill(row):
        physical.update({k:'TRUE' for k in row['effects']});physical.update({k:'FALSE' for k in row.get('deletes',[])})
    skills=S(phase='empty',held=None,motion_version=0,capabilities=lambda:{'take':skill,'put':skill},verify=lambda keys:{k:{'truth':physical[k],'evidence':{'public_fixture':True}} for k in keys},evidence_features=lambda key:{'capture':1})
    r=ModelContractRuntime(skills,d,group=group,limits={'harness_version':34,'recovery_version':29,'skill_budget':8,'episode_seconds':30},emit=lambda *a,**k:None,ablation=ablation)
    for k,v in physical.items():r.c.observe(k,v,{})
    return r


def test_open_python_has_live_incremental_plan_without_static_restriction():
    r=symbolic_runtime();r.compile_program("x=execution_state()\nif x:\n    execute_skill('take')")
    assert r.causal is not None and not r.causal.order
    r.execute('take')
    assert 'take' in r.causal.order and 'take' in r.c.completed
    assert r.prepare_recovery() and r.deterministic_code()
    assert r.c.truth('held')=='TRUE'


def test_open_literal_calls_have_real_causal_ablation():
    a=symbolic_runtime();b=symbolic_runtime(ablation='no_causal')
    for r in (a,b):r.compile_program("execute_skill('take')\nexecute_skill('put')")
    assert a.causal and b.causal
    assert any(e['kind']=='causal' and e['source']=='take' and e['target']=='put' for e in a.causal.edges)
    assert not any(e['kind']=='causal' for e in b.causal.edges)


def test_registered_straight_contract_composition_is_planner_candidate(tmp_path):
    from capx.execution.semantic_capabilities import CapabilityRegistry
    r=symbolic_runtime();r.compile_program('print("open")')
    reg=CapabilityRegistry(tmp_path,{'execute_skill':r.execute},graph=r.c.graph,tool_effects={'execute_skill':['parent_guarded_possible_physical_effect']},episode='one',condition_state=lambda:r.state()['conditions'],qualify=r.qualify_capability)
    p=dict(capability_id='compose',version='1',purpose='Compose bound trusted skills',
        code="def capability():\n    execute_skill('take')\n    execute_skill('put')\n    return {}",
        inputs={},outputs={},tools=['execute_skill'],expected_effects=['done'],physical_side_effects=['parent_guarded_possible_physical_effect'],verification_refs=['done'],validation_status='PROPOSED')
    entry=reg.register(p)
    assert entry['planner_qualification']['status']=='QUALIFIED_COMPOSITION'
    row=next(x for x in r.library if x.get('capability_id')==entry['id'])
    assert row['requires']==['ready'] and 'done' in row['effects']
    r.execute(row['id'])
    assert r.c.truth('done')=='TRUE' and r.broker.calls==2
    assert 'take' not in r.c.completed  # new IDs; not a replay of immutable calls
    assert any(e.get('kind')=='composed_capability_result' for e in r.events)


def test_joint_grasp_selection_rejects_first_and_preserves_selected_pose(monkeypatch):
    import numpy as np
    import capx.execution.preplanning as pp
    from capx.execution.preplanning import choose_joint_grasp
    seen=[]
    def future(s,row,shape,reg,destination,**kw):
        q=s.held['grasp_quaternion'];seen.append(q)
        return {'selected':None if q[0]==1 else {'position':[.4,0,.1]},'status':'NO_SOLUTION_FOUND' if q[0]==1 else 'KINEMATIC_CANDIDATE'}
    monkeypatch.setattr(pp,'search_placements',future)
    s=S(scene=S(source='public'),profile={},reachable=lambda *a:True,region=lambda role:{},preplanner=S(targets={'w':'0'}))
    candidates=[dict(id='first',q=np.array([1.,0,0,0]),score=1),dict(id='second',q=np.array([0.,1,0,0]),score=.5)]
    selected,results=choose_joint_grasp(s,'w',{},np.zeros(3),candidates)
    assert selected['id']=='second' and len(results)==2 and len(seen)==2


def test_old_occupied_solid_is_not_sparse_surface_distance():
    import numpy as np,trimesh
    from capx.execution.trajectory_execution import retained_solid_mask
    mesh=trimesh.creation.box(extents=[1,1,1])
    mask=retained_solid_mask(mesh,np.array([[0,0,0],[.6,0,0],[.499,0,0]]))
    assert mask.tolist()==[True,False,True]
    mesh.update_faces(np.arange(len(mesh.faces)-1))
    assert not retained_solid_mask(mesh,np.array([[0,0,0]]))[0]


def test_search_record_ids_do_not_renew_progress():
    p=ProgressTracker();s=state('UNKNOWN')
    p.update(s,{},search=[dict(strategy='acquire',parameters={'object':'w','candidate_id':'a'},outcome='NO_SOLUTION_FOUND')])
    same=p.update(s,{},search=[dict(strategy='acquire',parameters={'object':'w','candidate_id':'b'},outcome='NO_SOLUTION_FOUND')])
    assert not same['search_progress'] and not same['changed']


def test_nested_reconciled_motion_has_consistent_dispatch_receipt(tmp_path):
    from capx.execution.sandbox import ExecutionInterrupted
    count=[0]
    b=S(actor=S(call_site=None),execution_version=36,spec={},
        dispatch_receipt=lambda:{'started':count[0],'returned':0},reconcile_dispatch=lambda:True)
    store=EventStore(tmp_path);m=ExecutionMonitor(store,lambda:{'source':'fresh','robot':{}},b)
    def guarded():count[0]+=1;raise SkillBoundaryStop('controlled force stop',{})
    def skill():
        try:m.invoke('goto_pose',[],{},guarded)
        except SkillBoundaryStop:raise ExecutionInterrupted('skill stopped')
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[],{},skill)
    assert not store.unresolved()
    assert m.last_failure['dispatch_state']=='DISPATCHED_RECONCILED'


def test_nested_unknown_motion_cannot_be_reconciled_by_wrapper(tmp_path):
    from capx.execution.sandbox import ExecutionInterrupted
    count=[0]
    b=S(actor=S(call_site=None),execution_version=36,spec={},
        dispatch_receipt=lambda:{'started':count[0],'returned':0},reconcile_dispatch=lambda:True)
    store=EventStore(tmp_path);m=ExecutionMonitor(store,lambda:{'source':'fresh','robot':{}},b)
    def driver():count[0]+=1;raise RuntimeError('lost acknowledgement')
    def skill():
        try:m.invoke('goto_pose',[],{},driver)
        except RuntimeError:raise ExecutionInterrupted('skill stopped')
    with pytest.raises(ExecutionInterrupted):m.invoke('run_plan',[],{},skill)
    assert store.unresolved() and m.last_failure['dispatch_state']=='DISPATCHED_UNRECONCILED'


def test_capability_two_projection_layers_invalidate_from_original(tmp_path):
    from capx.execution.semantic_capabilities import CapabilityRegistry
    from capx.execution.dependency_graph import DependencyGraph
    from test_v36_semantic import proposal
    g=DependencyGraph();source=g.add('estimate',{'value':2.})
    def measure(label):return {'value':2.,'id':source,'result_ref':source}
    first=CapabilityRegistry(tmp_path/'first',{'measure':measure},graph=g,tool_effects={'measure':[]},episode='one')
    entry=first.register(proposal());out=first.run(entry['id'],{'label':'widget'})
    def read_summary(label):return out
    second=CapabilityRegistry(tmp_path/'second',{'read_summary':read_summary},graph=g,tool_effects={'read_summary':[]},episode='one')
    p=proposal();p['tools']=['read_summary'];p['code']="def capability(label):\n    x=read_summary(label)\n    return {'value':x['output']['value']}"
    p['outputs']={'value':'number'}
    p['output_projections']={'value':{'call':0,'tool':'read_summary','path':['output','value']}}
    entry2=second.register(p);out2=second.run(entry2['id'],{'label':'widget'})
    plan=g.add('plan',{},parents=[out2['output_node']])
    assert out2['status']=='VERIFIED'
    g.invalidate([source],reason='contradiction')
    assert not g.nodes[plan]['valid']
    for row in (out,out2):
        assert not g.nodes[row['output_node']]['valid']
        assert g.nodes[row['node_id']]['state']=='done'


def test_compiled_composition_enters_search_and_preserves_verified_sibling():
    from capx.execution.causal_plan import repair_plan
    r=symbolic_runtime();r.compile_program('print("open")')
    r.c.definition['goals'].append('sibling')
    sibling=r.c.observe('sibling','TRUE',{'public_verifier':True})
    before=repair_plan(r.causal,r.library,max_depth=1)
    assert before['status']!='ready'
    proposal={'code':"def capability():\n    execute_skill('take')\n    execute_skill('put')\n    return {}",'inputs':{},'expected_effects':['done']}
    assert r.qualify_capability('candidate-composition',proposal)['status']=='QUALIFIED_COMPOSITION'
    after=repair_plan(r.causal,r.library,max_depth=1)
    assert after['status']=='ready' and after['fragment'][0]['components']
    r.causal=r.causal.commit(after)
    r.execute(r.causal.order[0])
    assert r.c.truth('done')=='TRUE' and r.c.conditions['sibling']==sibling
    assert r.broker.calls==2


def test_early_return_cannot_invent_child_effects():
    r=symbolic_runtime()
    p={'code':"def capability():\n    return {}\n    execute_skill('take')\n    execute_skill('put')",'inputs':{},'expected_effects':['done']}
    assert r.qualify_capability('bad',p)['status']=='CALLABLE_ONLY'


def test_joint_preplanning_keeps_failure_diagnostics_and_bounded_search(monkeypatch):
    import capx.execution.preplanning as pp
    from capx.execution.preplanning import SkillPreplanner,planning_summary
    s=S(scene=S(),phase='empty',effects={},motion_version=0,held=None,
        joint_grasp_search={'w':[{'candidate_id':'g','future_placement':{'status':'NO_SOLUTION_FOUND','tested':[{'checks':[{'position_error_m':.02}]}],'candidate_count':8,'remaining_candidates':0}}]})
    def fail(name):raise SkillBoundaryStop('finite candidates exhausted',{})
    s._acquire_plan=fail;p=SkillPreplanner(s);p.context=lambda:'same'
    out=p.preview({'skill':'acquire','object':'w','future_positions':12})
    assert out['joint_search'][0]['future_placement']['tested'][0]['checks'][0]['position_error_m']==.02
    assert planning_summary(out)['future_search']['max_positions']==12
    assert out['physical_unreachable'] is False
    with pytest.raises(ValueError,match='future_positions'):p.preview({'skill':'acquire','object':'w','future_positions':13})


def test_joint_search_expansion_is_bounded_and_applies_to_all_objects(monkeypatch):
    import capx.execution.preplanning as pp
    import numpy as np
    seen=[]
    def search(*args,**kwargs):
        seen.append(kwargs['max_positions']);return {'selected':None,'status':'NO_SOLUTION_FOUND'}
    monkeypatch.setattr(pp,'search_placements',search)
    s=S(scene=S(source='public'),profile={},reachable=lambda *a:True,region=lambda role:{},
        preplanner=S(targets={'renamed':'0'}),future_search_positions=12)
    pp.choose_joint_grasp(s,'renamed',{},np.zeros(3),[{'id':'c','q':[1,0,0,0]}])
    assert seen==[12]
