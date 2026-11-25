"""Stage recovery gates: synthetic public measurements, never physics success."""
import copy
import numpy as np
import pytest
from capx.execution.skill_recovery import attachment_proposal, attachment_consistency, recovery_library
from capx.execution.causal_plan import CausalPlan,ground_library,repair_plan
from capx.execution.skill_contracts import ContractGraph
from capx.experiments.native_vision_skills import region_task


def fit(**kw):
    r=dict(transform=np.eye(4).tolist(),coverage=.95,residual_95_m=.002,spread_m=[.02,.01,.005],innovation_m=.005,innovation_deg=18.,status='UNKNOWN')
    r.update(kw);return r


def test_changed_attachment_is_hypothesis_not_old_permission():
    assert attachment_proposal(fit())
    for key,value in [('coverage',.4),('spread_m',[.1,.02,0]),('residual_95_m',.01),('transform',None)]:
        assert not attachment_proposal(fit(**{key:value}))


def test_attachment_probe_requires_actual_motion_and_rigid_consistency():
    a=np.eye(4);b=a.copy();b[2,3]=.01
    assert attachment_consistency(a,b,a,b,.04,.04)['status']=='PASS'
    assert attachment_consistency(a,a,a,a,.04,.04)['status']!='PASS'
    slipped=b.copy();slipped[0,3]=.02
    assert attachment_consistency(a,b,a,slipped,.04,.04)['status']!='PASS'
    assert attachment_consistency(a,b,a,b,.04,.06)['status']!='PASS'


def test_graph_composes_attachment_confirmation_before_retained_deposit():
    d=region_task({'novel widget':'part'},grounded=True,causal=True);recovery_library(d)
    c=ContractGraph(d);p=CausalPlan(c,['acquire-0','deposit-0'])
    for k in d['condition_specs']:c.observe(k,'FALSE',{})
    for k in ['attachment_candidate:novel widget','probe_ready:novel widget','target_geometry:0']:c.observe(k,'TRUE',{})
    c.stopped.add('acquire-0')
    c.observe('held:novel widget','UNKNOWN',{})
    r=repair_plan(p,ground_library(d),failed=['held:novel widget'],exhausted={'held:novel widget':'UNKNOWN'})
    assert r['status']=='ready'
    assert [x['skill'] for x in r['fragment']]==['confirm_attachment']
    assert r['retained']==['deposit-0']
    p2=p.commit(r)
    assert p2.order[0]!='acquire-0' and c.truth('held:novel widget')=='UNKNOWN'


def test_unknown_probe_cannot_authorize_repair():
    d=region_task({'renamed':'part'},grounded=True,causal=True);recovery_library(d)
    c=ContractGraph(d);p=CausalPlan(c,['acquire-0','deposit-0']);c.stopped.add('acquire-0')
    for k in d['condition_specs']:c.observe(k,'UNKNOWN',{})
    exhausted={k:'UNKNOWN' for k in d['condition_specs']}
    assert repair_plan(p,ground_library(d),exhausted=exhausted)['status']=='blocked'


def test_library_is_shared_parameterized_and_no_new_initial_solution():
    d=region_task({'part-a':'a','part-b':'b'},grounded=True,causal=True);old=copy.deepcopy(d['intents']);recovery_library(d)
    assert d['intents']==old
    skills=ground_library(d)
    assert {r['object'] for r in skills if r['skill']=='confirm_attachment'}=={'part-a','part-b'}
    assert all(not r.get('after') for r in skills)


def test_repair_discovers_evidence_needed_by_new_library_capability():
    d=region_task({'new part':'part'},grounded=True,causal=True);recovery_library(d)
    c=ContractGraph(d);p=CausalPlan(c,['acquire-0','deposit-0']);c.stopped.add('acquire-0')
    for k in d['condition_specs']:c.observe(k,'FALSE',{})
    for k in ['attachment_candidate:new part','target_geometry:0']:c.observe(k,'TRUE',{})
    for k in ['held:new part','probe_ready:new part']:c.observe(k,'UNKNOWN',{})
    r=repair_plan(p,ground_library(d),failed=['held:new part'],exhausted={'held:new part':'UNKNOWN'})
    assert r['status']=='evidence' and r['keys']==['probe_ready:new part']


def test_repair_can_retreat_then_regain_tool_evidence_without_claiming_it_true():
    from capx.execution.causal_plan import signature
    d=region_task({'part':'part'},grounded=True,causal=True);recovery_library(d)
    c=ContractGraph(d);p=CausalPlan(c,['acquire-0','deposit-0']);c.stopped.add('acquire-0')
    for k in d['condition_specs']:c.observe(k,'TRUE',{})
    for k in ['held:part','robot_ready','placed:part','attachment_candidate:part']:c.observe(k,'FALSE',{})
    for k in ['geometry:part','grasp_ready:part']:c.observe(k,'UNKNOWN',{})
    r=repair_plan(p,ground_library(d),failed=['robot_ready'],exhausted={'geometry:part':'UNKNOWN','grasp_ready:part':'UNKNOWN'},forbidden=[signature(c.intents['acquire-0'])])
    assert r['status']=='ready'
    assert [x['skill'] for x in r['fragment']]==['retreat_empty','acquire']
    assert 'geometry:part' in r['validation']['obligations']
    assert c.truth('geometry:part')=='UNKNOWN'


def test_pointcloud_grasp_requests_seed_and_rejects_old_service(monkeypatch):
    from capx.integrations.vision import graspnet
    sent=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):return dict(seed=None,grasps_base64=graspnet._numpy_to_base64(np.empty((0,4,4))),scores_base64=graspnet._numpy_to_base64(np.array([])),contact_pts_base64=graspnet._numpy_to_base64(np.empty((0,3))))
    monkeypatch.setattr(graspnet,'audited_post',lambda url,**kw:(sent.append(kw) or Response()))
    with pytest.raises(ConnectionError,match='seed'):
        graspnet.init_contact_graspnet_point_clouds()(np.ones((5,3)),np.ones((5,3)),seed=17,max_retries=2)
    assert sent[0]['json']['seed']==17 and sent[0]['timeout']>0


def bare_skills():
    from types import SimpleNamespace
    from capx.execution.skill_recovery import RecoverableRegionSkills
    s=RecoverableRegionSkills.__new__(RecoverableRegionSkills)
    s.scene=SimpleNamespace(source='public-frame',serial=1,frame=lambda:None,
        obs={'robot_cartesian_pos':[0,0,.2,1,0,0,0],'robot_gripper_joint_pos':[.02,-.02]})
    s.held=None;s.stalled=None;s.active_condition=None;s.effects={};s.recovery_attempts={}
    s.failed_candidates={};s.plans={};s.phase='empty';s.emit=lambda *a,**kw:None
    return s


def test_failed_approach_never_dispatches_close_or_claims_hold():
    from capx.execution.vision_skills import SkillBoundaryStop
    s=bare_skills();calls=[];s.motion=lambda name,*a:calls.append(name)
    s._acquire_plan=lambda name:({'points':np.ones((4,3))},np.zeros(3),np.array([1,0,0,0]),{})
    s.move=lambda *a:s.stop('measured nonconvergence')
    with pytest.raises(SkillBoundaryStop) as error:s.acquire({'object':'new-part'})
    assert calls==['open_gripper'] and s.held is None and s.phase=='approach_stopped'
    assert error.value.effects['held:new-part']['truth']=='FALSE'


def test_attachment_probe_failure_rolls_back_hypothesis_and_never_releases():
    from capx.execution.vision_skills import SkillBoundaryStop
    s=bare_skills();calls=[];s.motion=lambda name,*a:calls.append(name)
    s.phase='closed_unverified';s.held={'object':'novel','tcp':np.eye(4),'shape':{'center':[0,0,0]}}
    original=copy.deepcopy(s.held);s._corridor=lambda *a,**kw:(np.array([0,0,.21]),np.array([1,0,0,0]))
    transform=np.eye(4);transform[0,3]=.005
    s._held_measurement=lambda:(None,fit(transform=transform.tolist()),.04)
    s.move=lambda *a:s.stop('probe control failed')
    with pytest.raises(SkillBoundaryStop):s.confirm_attachment({'object':'novel'})
    assert np.array_equal(s.held['tcp'],original['tcp']) and s.phase=='closed_unverified'
    assert 'open_gripper' not in calls and s.effects['held:novel']=='UNKNOWN'


def test_stage_recovery_is_bounded_even_without_new_observations():
    from capx.execution.vision_skills import SkillBoundaryStop
    s=bare_skills()
    s._begin_repair('part','retreat_empty');s._begin_repair('part','retreat_empty')
    with pytest.raises(SkillBoundaryStop,match='budget'):s._begin_repair('part','retreat_empty')
