import numpy as np
import pytest
from capx.execution.table_clearance import current_table_evidence
from test_table_clearance import fixture,Robot
from test_held_mask_boundary import current_prediction
from test_grounded_monitor import fixture as monitor_fixture


def scenario(modify=None,permit=True):
    obs,region,pred,geom,mask=fixture();pred=current_prediction(pred)
    mask[56:58,56:58]=False
    if modify:modify(obs,pred)
    calls=[];reservations=[]
    def segment(image,text_prompt):
        calls.append(image.shape)
        if len(calls)==1:return [{'mask':mask,'score':.9}]
        return [{'mask':image[:,:,1]>image[:,:,0],'score':.85}]
    def reserve():reservations.append(1);return permit
    result=current_table_evidence(obs,region,'green cube',pred,Robot(),segment,
        observation_id='current',held_identity_verified=True,nonheld_geometry=geom,
        static_scene_assumption=True,reserve_semantic_refinement=reserve)
    return result,calls,reservations


def test_actual_second_segmentation_can_repair_semantic_hole_without_wider_geometry():
    out,calls,res=scenario()
    assert out['validation']['status']=='PASS'
    assert len(calls)==2 and calls[1][0]<calls[0][0] and res==[1]
    e=out['evidence']['semantic_refinement']
    assert e['initial_validation']['status']=='UNCERTAIN' and e['attempted']
    assert e['independent_sensor'] is False and out['evidence']['sam3_requests']==2
    assert out['evidence']['held_mask_boundary_refinement']['surface_tolerance_m']==.002
    assert out['masks']['held'][56,56]


def test_budget_refusal_keeps_original_unknown_and_never_requests_crop():
    out,calls,res=scenario(permit=False)
    assert out['validation']['status']=='UNCERTAIN' and len(calls)==1 and res==[1]
    assert not out['evidence']['semantic_refinement']['attempted']


@pytest.mark.parametrize('kind',['foreign_depth','nonheld_colour','prior_only'])
def test_refinement_does_not_override_geometry_identity_or_current_source(kind):
    def modify(o,p):
        if kind=='foreign_depth':o['camera']['images']['depth'][56,56]=.90
        if kind=='nonheld_colour':o['camera']['images']['rgb'][56,56]=[180,0,0]
        if kind=='prior_only':p['source']='rigid_prior_only'
    out,calls,res=scenario(modify)
    assert out['validation']['status']!='PASS'
    assert not out['masks']['held'][56,56]
    assert len(calls)<=2
    if kind=='prior_only':assert len(calls)==1 and not res


@pytest.mark.parametrize('mode',['G0','G1'])
def test_perception_reservation_is_shared_finite_across_actors(mode):
    m,s,v,c,f=monitor_fixture(mode);m.max_episode_local=2
    assert m._reserve_semantic_refinement()
    m.begin_program('print(1)',1)
    assert m._reserve_semantic_refinement()
    assert not m._reserve_semantic_refinement()
    assert m.local_attempts==2 and m.perception_refinements==2 and not c
