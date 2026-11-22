"""Uncertain partial visibility triggers bounded public active perception."""
import numpy as np
import pytest
import capx.execution.grounded_feedback as module
from capx.execution.sandbox import ExecutionInterrupted
from test_grounded_monitor import fixture


def held(monkeypatch):
    m,state,surfaces,calls,raw=fixture()
    m.phase='holding_verified';m.closed=m.lifted=True;m.held_object='red cube'
    m.attachment={'public_only':True};m.geometry['red cube']={'height_m':.04}
    monkeypatch.setattr(module,'predict_attachment',lambda a,t:{'center':[.5,0,.2],'bottom_z':.18,'orientation_verified':True})
    m._held_surface=lambda:None if not calls else {'center':[.5,0,.2],'hi':[.52,.02,.22],'lo':[.48,-.02,.18],'bottom_z':.18,'frame':'robot_base'}
    return m,state,surfaces,calls,raw


def test_uncertain_surface_can_get_a_new_view_without_regrasp_or_repeated_lift(monkeypatch):
    m,s,v,c,f=held(monkeypatch)
    old=m.invoke('read_robot',[],{},f)
    result=m.invoke('observe_surfaces',['red cube'],{},f)
    assert result['bottom_z']==.18
    assert [x[0] for x in c]==['goto_pose']
    assert np.allclose(c[0][1][0],[.5,0,.14])
    assert not np.allclose(c[0][1][1],[0,0,1,0])
    assert not m.graph.nodes[old['id']]['valid']
    assert m.phase=='holding_verified' and m.lift_attempts==0
    assert m.visibility_probes==1 and m.local_attempts==1


def test_low_clearance_never_rotates_carried_object(monkeypatch):
    m,s,v,c,f=held(monkeypatch)
    monkeypatch.setattr(module,'predict_attachment',lambda a,t:{'center':[.5,0,.03],'bottom_z':.01,'orientation_verified':True})
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['red cube'],{},f)
    assert c==[]
    assert m.phase=='holding_verified'


def test_unobservable_scene_stops_after_two_views_and_persists_across_actor(monkeypatch):
    m,s,v,c,f=held(monkeypatch);m._held_surface=lambda:None
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['red cube'],{},f)
    assert len(c)==2 and m.visibility_probes==2
    assert m.terminal_reason=='held_surface_evidence_unrecoverable'
    m.begin_program('recovery',1)
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['red cube'],{},f)
    assert len(c)==2


def test_missing_neighbor_visibility_cannot_authorize_wrist_sweep(monkeypatch):
    m,s,v,c,f=held(monkeypatch);v['green']=None
    m.color_points_fn=lambda obs,name:np.empty((0,3))
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['red cube'],{},f)
    assert c==[]


def test_episode_local_budget_applies_to_active_perception(monkeypatch):
    m,s,v,c,f=held(monkeypatch);m.local_attempts=m.max_episode_local
    with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['red cube'],{},f)
    assert c==[] and m.terminal_reason=='local_recovery_budget_exhausted'
