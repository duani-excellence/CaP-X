import copy
import numpy as np
import pytest


def test_exact_semantic_cache_requires_identical_input_and_returns_owned_masks():
    from capx.execution.perception_evidence import ExactEvidenceCache
    cache=ExactEvidenceCache(max_entries=2);calls=[]
    image=np.zeros((8,8,3),np.uint8)
    def query():
        calls.append(1);return [{'mask':np.ones((8,8),bool),'score':.9}]
    key=('camera','calibration','label',cache.digest(image))
    first,hit,_=cache.get_or_compute(key,query,'frame-1')
    assert not hit
    first[0]['mask'][:]=False
    second,hit,origin=cache.get_or_compute(key,query,'frame-2')
    assert hit and origin=='frame-1' and second[0]['mask'].all() and len(calls)==1
    image[0,0]=1
    cache.get_or_compute(('camera','calibration','label',cache.digest(image)),query,'frame-3')
    cache.get_or_compute(('camera','new-calibration','label',cache.digest(image)),query,'frame-4')
    assert len(calls)==3 and len(cache.entries)==2


def test_exact_observation_identity_includes_depth_calibration_and_robot_state():
    from capx.execution.perception_evidence import ExactEvidenceCache
    obs={'cam':{'images':{'rgb':np.zeros((4,4,3),np.uint8),'depth':np.ones((4,4))},
                'intrinsics':np.eye(3),'pose_mat':np.eye(4)},'robot_joint_pos':np.zeros(7)}
    original=ExactEvidenceCache.digest(obs)
    for mutate in [lambda o:o['cam']['images']['depth'].__setitem__((0,0),2),
                   lambda o:o['cam']['pose_mat'].__setitem__((0,3),.01),
                   lambda o:o['robot_joint_pos'].__setitem__(0,.01)]:
        other=copy.deepcopy(obs);mutate(other)
        assert ExactEvidenceCache.digest(other)!=original


def test_frame_cache_reuses_semantics_but_never_old_depth():
    from capx.experiments.native_vision import PublicScene
    obs={'cam':{'images':{'rgb':np.zeros((12,12,3),np.uint8),'depth':np.ones((12,12))},
                'intrinsics':np.array([[100.,0,6],[0,100.,6],[0,0,1]]),'pose_mat':np.eye(4)}}
    calls=[]
    def segment(image,**kw):
        calls.append(1);return [{'mask':np.ones(image.shape[:2],bool),'score':.9}]
    s=PublicScene(lambda:(copy.deepcopy(obs),'frame'),segment,{'part':'thing'},optimize=True)
    s.frame();a=s.candidates('thing')[0]['points'].copy()
    obs['cam']['images']['depth'][:]=2
    s.frame();b=s.candidates('thing')[0]['points']
    assert len(calls)==1 and np.allclose(b,2*a)
    obs['cam']['pose_mat'][0,3]=.1
    s.frame();s.candidates('thing')
    assert len(calls)==2


def test_cost_frontier_batches_free_future_information_not_speculative_ik():
    from capx.execution.contract_recovery import ConditionPlanner
    from capx.execution.skill_contracts import ContractGraph
    d=dict(schema='capx.contract_task.v1',objects=['renamed'],resources=['arm'],goals=['done'],
           condition_specs={k:{'observable':True} for k in ['done','shape','ready','held']},
           intents=[dict(id='a',skill='pick',requires=['shape','ready'],effects=['held'],resources=['arm']),
                    dict(id='b',skill='put',requires=['held'],effects=['done'],resources=['arm'])])
    c=ContractGraph(d);p=ConditionPlanner(c)
    requirements={'done':{'rgbd':1,'segment':4},'shape':{'rgbd':1,'segment':4},
                  'held':{'rgbd':1},'ready':{'rgbd':1,'segment':4,'grasp_ik':10}}
    plan=p.costed_evidence(p.plan(),lambda k:requirements[k])
    assert set(plan['keys'])=={'done','shape','held'} and 'ready' not in plan['keys']
    assert plan['estimated_cost']==5 and plan['avoided_incremental_cost']==0
    c.observe('shape','TRUE',{});c.observe('held','FALSE',{});c.observe('done','FALSE',{})
    p.exhaust(['ready'])
    assert p.plan()['work']=='blocked'


def test_cost_planning_never_reobserves_exhausted_or_true_sibling():
    from capx.execution.contract_recovery import ConditionPlanner
    from capx.execution.skill_contracts import ContractGraph
    d=dict(schema='capx.contract_task.v1',objects=['part'],resources=[],goals=['a','b','c'],
           intents=[dict(id='make-'+k,skill='make',requires=[],effects=[k]) for k in ['a','b','c']],
           condition_specs={k:{'observable':True} for k in ['a','b','c']})
    c=ContractGraph(d);c.observe('b','TRUE',{});p=ConditionPlanner(c);p.exhaust(['c'])
    q=p.costed_evidence(p.plan(),lambda k:{'same_frame':1})
    assert q['keys']==['a']


def test_cost_bundle_combines_required_sensor_work_but_not_multiple_motion_plans():
    from capx.execution.contract_recovery import ConditionPlanner
    from capx.execution.skill_contracts import ContractGraph
    d=dict(schema='capx.contract_task.v1',objects=['x','y'],resources=[],goals=['a','b'],
           intents=[dict(id='make-'+k,skill='make',requires=[],effects=[k]) for k in ['a','b']],
           condition_specs={k:{'observable':True} for k in ['a','b']})
    p=ConditionPlanner(ContractGraph(d))
    result=p.costed_evidence(p.plan(),lambda k:{'capture':1,'mask:'+k:2})
    assert set(result['keys'])=={'a','b'} and result['estimated_cost']==5
    result=p.costed_evidence(p.plan(),lambda k:{'capture':1,'plan:'+k:{'cost':3,'valid_until':'motion'}})
    assert len(result['keys'])==1 and result['estimated_cost']==4


def test_cost_planning_is_opt_in_and_no_selective_keeps_conservative_scope():
    from capx.experiments.contract_models import ModelContractRuntime
    from test_v27_recovery import task
    class Skills:
        phase='empty';held=None
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,keys):return {k:dict(truth='UNKNOWN',evidence={}) for k in keys}
        def evidence_features(self,key):return {'frame':1}
    for group,ablation,expected in [('G1','none',True),('G0','none',False),('G1','no_selective',False),('G1','no_graph_repair',True)]:
        r=ModelContractRuntime(Skills(),task(),group=group,ablation=ablation,
            limits={'skill_budget':10,'episode_seconds':60,'recovery_version':28},emit=lambda *a,**k:None)
        assert r.costed_evidence is expected


def test_v28_authority_ablation_keeps_same_costed_evidence_but_all_legal_choices():
    from capx.experiments.contract_models import ModelContractRuntime
    from test_v27_recovery import task
    class Skills:
        phase='empty';held=None
        def __init__(self):self.scopes=[]
        def acquire(self,r):pass
        def deposit(self,r):pass
        def verify(self,keys):
            self.scopes.append(list(keys))
            return {k:dict(truth='FALSE' if k.startswith(('held:','placed:')) else 'TRUE',evidence={}) for k in keys}
        def evidence_features(self,key):return {'frame':1}
    variants=[]
    for ablation in ['none','no_graph_repair']:
        s=Skills();r=ModelContractRuntime(s,task(),group='G1',ablation=ablation,
            limits={'skill_budget':10,'episode_seconds':60,'recovery_version':28},emit=lambda *a,**k:None)
        r.c.observe('ready','TRUE',{});r.c.observe('geometry','TRUE',{})
        assert r.prepare_recovery()
        variants.append(r)
    a,b=variants
    assert a.skills.scopes==b.skills.scopes
    assert a.work_scope['allowed_intents']==['get-part']
    assert set(b.work_scope['allowed_intents'])=={'get-part','get-peer'}
    assert a.work_scope['enforced'] and not b.work_scope['enforced']


def test_motion_plan_reuse_requires_current_depth_robot_effects_and_motion_epoch():
    from capx.execution.vision_skills import RegionSkills
    class Scene:
        optimize=True;observation_key='fresh-rgbd-robot';source='f1'
    s=Scene();r=RegionSkills(s,None,None,None,
        {'schema':'capx.upright_development_bounds.v1','offline_evidence':['fixture']})
    calls=[]
    def compute(name):calls.append(name);return {'checked':name}
    r._compute_acquire_plan=compute
    r._acquire_plan('part');s.source='f2';r._acquire_plan('part')
    assert len(calls)==1
    r.effects.update({'geometry:part':'TRUE','grasp_ready:part':'TRUE'})
    r._acquire_plan('part')
    assert len(calls)==1  # A verifier conclusion is not a new physical effect.
    for mutate in [lambda:setattr(s,'observation_key','changed-depth-or-robot'),
                   lambda:setattr(r,'motion_version',1),
                   lambda:r.effects.update({'placed:peer':'UNKNOWN'})]:
        mutate();r._acquire_plan('part')
    assert len(calls)==4


def test_same_center_but_different_crop_extent_does_not_reuse_geometry():
    from capx.experiments.native_vision import PublicScene
    obs={'cam':{'images':{'rgb':np.zeros((64,64,3),np.uint8),'depth':np.ones((64,64))},
                'intrinsics':np.array([[100.,0,32],[0,100.,32],[0,0,1]]),'pose_mat':np.eye(4)}}
    s=PublicScene(lambda:(obs,'f'),lambda image,**kw:[],{},optimize=True)
    s.frame()
    a=dict(center=[0,0,1],lo=[-.1,-.1],hi=[.1,.1],floor_z=1)
    b=dict(a,lo=[-.3,-.3],hi=[.3,.3])
    s.candidates('part',a);s.candidates('part',b)
    assert len(s.cache)==2


def test_failed_semantic_request_has_provenance_and_is_not_cached():
    from capx.experiments.native_vision import PublicScene
    obs={'cam':{'images':{'rgb':np.zeros((4,4,3),np.uint8),'depth':np.ones((4,4))},
                'intrinsics':np.eye(3),'pose_mat':np.eye(4)}}
    events=[]
    def fail(*a,**kw):raise RuntimeError('tool unavailable')
    s=PublicScene(lambda:(obs,'f'),fail,{},optimize=True,emit=lambda kind,**kw:events.append(dict(kind=kind,**kw)))
    s.frame()
    for _ in range(2):
        with pytest.raises(RuntimeError):s.candidates('thing')
    rows=[x for x in events if x['kind']=='perception_request']
    assert len(rows)==2 and all(x['error']=='RuntimeError' and not x['reused'] for x in rows)
    assert not s.semantic_memory.entries


def test_variant_keeps_full_physical_snapshot_gate_and_only_changes_execution_flags():
    from capx.experiments.contract_models import shared_boundary_spec
    base={'environment':'SomeEnvironment','api_budget':10,'recovery_version':28,'perception_optimization':True}
    variant=dict(base,recovery_version=27,perception_optimization=False)
    assert shared_boundary_spec(variant,base)==base
    assert variant['recovery_version']==27
    with pytest.raises(ValueError,match='non-policy'):
        shared_boundary_spec(dict(variant,api_budget=11),base)


def ready_frontier_definition():
    return dict(schema='capx.contract_task.v1',objects=['renamed-a','renamed-b'],resources=[],goals=['done-a','done-b'],
        condition_specs={k:{'observable':True} for k in ['done-a','done-b','ready-a','ready-b']},
        intents=[dict(id='op-a',skill='acquire',object='renamed-a',requires=['ready-a'],effects=['done-a'],resources=[]),
                 dict(id='op-b',skill='acquire',object='renamed-b',requires=['ready-b'],effects=['done-b'],resources=[])])


def test_ready_action_consumes_motion_limited_plan_before_unrelated_evidence():
    from capx.execution.contract_recovery import ConditionPlanner
    from capx.execution.skill_contracts import ContractGraph
    c=ContractGraph(ready_frontier_definition());p=ConditionPlanner(c)
    for key in ['done-a','done-b']:c.observe(key,'FALSE',{})
    c.observe('ready-a','TRUE',{})
    base=p.plan();assert base['work']=='evidence'
    selected=p.costed_evidence(base,lambda k:{'capture':1,'transient-plan':{'cost':2,'valid_until':'motion'}})
    assert selected['work']=='skill' and selected['allowed_intents']==['op-a']
    assert c.truth('ready-b')=='UNKNOWN'


def test_runtime_does_not_refresh_after_cost_planner_selects_ready_work():
    from capx.experiments.contract_models import ModelContractRuntime
    class Skills:
        phase='empty';held=None;motion_version=0
        def acquire(self,row):pass
        def deposit(self,row):pass
        def verify(self,keys):raise AssertionError('Unrelated evidence before ready work')
        def evidence_features(self,key):return {'capture':1,'plan':{'cost':2,'valid_until':'motion'}}
    for variant in ['none','no_graph_repair']:
        r=ModelContractRuntime(Skills(),ready_frontier_definition(),group='G1',ablation=variant,
            limits={'skill_budget':10,'episode_seconds':60,'recovery_version':28},emit=lambda *a,**k:None)
        for key in ['done-a','done-b']:r.c.observe(key,'FALSE',{})
        r.c.observe('ready-a','TRUE',{})
        assert r.prepare_recovery()
        assert r.refreshes==0 and r.work_scope['allowed_intents']==['op-a']
