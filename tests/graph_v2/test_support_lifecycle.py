import numpy as np
import pytest
from test_grounded_monitor import fixture
from capx.execution.sandbox import ExecutionInterrupted


def setup(monkeypatch):
 import capx.execution.grounded_feedback as mod
 m,s,v,c,f=fixture();m.invoke('observe_object',['green cube'],{},f);m.capture()
 monkeypatch.setattr(mod,'current_support',lambda *a,camera_origin=None:None)
 monkeypatch.setattr(mod,'retained_consistent',lambda *a:True)
 # Current partially visible points do not independently locate the support.
 v['green']['top_pixels']=10
 return m,s,v,c,f


def test_unchanged_support_can_be_retained_for_planning_with_explicit_history(monkeypatch):
 m,s,v,c,f=setup(monkeypatch)
 record=m.invoke('observe_surfaces',['green cube'],{},f)
 assert record['source']=='retained_public_support_pose'
 assert record['full_current_localization'] is False
 assert record['prior_source_id'] in m.graph.nodes


def test_possible_contact_revokes_history_even_if_current_points_do_not_contradict(monkeypatch):
 m,s,v,c,f=setup(monkeypatch)
 m.object_epochs['green cube']+=1
 with pytest.raises(ExecutionInterrupted):m.invoke('observe_surfaces',['green cube'],{},f)


def test_release_requires_current_geometry_even_if_unchanged_epoch(monkeypatch):
 m,s,v,c,f=setup(monkeypatch)
 assert m._support_surface('green cube',allow_retained=False) is None
