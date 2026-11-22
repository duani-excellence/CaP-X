import numpy as np
import pytest
import capx.execution.grounded_feedback as module
from test_grounded_monitor import fixture
from capx.execution.sandbox import ExecutionInterrupted


def setup(monkeypatch, points):
    monkeypatch.setattr(module,'color_points',lambda obs,name:points if name=='green cube' else np.empty((0,3)))
    m,s,v,c,raw=fixture()
    m.color_points_fn=lambda obs,name:points if name=='green cube' else np.empty((0,3))
    m.invoke('observe_object',['red cube'],{},raw)
    g=m.invoke('observe_grasp',['red cube'],{},raw)
    plan=m.invoke('plan_skill',['approach',{'grasp':g['id']}],{},raw)
    c.clear()
    return m,s,v,c,raw,plan


def test_public_neighbor_collision_blocks_approach(monkeypatch):
    points=np.tile([.45,0,.04],(40,1))
    m,s,v,c,raw,plan=setup(monkeypatch,points)
    with pytest.raises(ExecutionInterrupted,match='gripper_sweep'):m.invoke('run_plan',[plan],{},raw)
    assert not c
    assert m.phase=='ready'
    assert any(e['check']=='gripper_sweep' and e['status']=='FAIL' for e in m.events)


def test_unseen_known_neighbor_does_not_manufacture_clearance(monkeypatch):
    m,s,v,c,raw,plan=setup(monkeypatch,np.empty((0,3)))
    with pytest.raises(ExecutionInterrupted,match='gripper_sweep'):m.invoke('run_plan',[plan],{},raw)
    assert not c


def test_public_clear_sweep_allows_approach_with_explicit_scope(monkeypatch):
    m,s,v,c,raw,plan=setup(monkeypatch,np.tile([.8,.2,.04],(40,1)))
    m.invoke('run_plan',[plan],{},raw)
    assert len(c)==1 and m.phase=='approached'
    event=next(e for e in m.events if e['check']=='gripper_sweep')
    assert event['metrics']['proof']=='visible_points_only_not_collision_free_certificate'


def test_asymmetric_fingers_block_approach_until_opened(monkeypatch):
    m,s,v,c,raw,plan=setup(monkeypatch,np.tile([.8,.2,.04],(40,1)))
    s['robot_gripper_joint_pos']=np.array([.04,-.02])
    with pytest.raises(ExecutionInterrupted,match='open_gripper_before_approach'):
        m.invoke('run_plan',[plan],{},raw)
    assert not c


def test_missing_second_encoder_cannot_claim_open_finger_sweep(monkeypatch):
    m,s,v,c,raw,plan=setup(monkeypatch,np.tile([.8,.2,.04],(40,1)))
    s.pop('robot_gripper_joint_pos',None)
    with pytest.raises(ExecutionInterrupted,match='finger_encoder_evidence_required'):
        m.invoke('run_plan',[plan],{},raw)
    assert not c
