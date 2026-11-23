import numpy as np
import pytest


def test_ring_hole_and_handle_are_bound_by_geometry_not_name():
    from capx.execution.insertion_geometry import flat_ring_features
    x,y=np.meshgrid(np.arange(-.04,.081,.001),np.arange(-.04,.041,.001))
    keep=((abs(x)<=.04)&(abs(y)<=.04)&~((abs(x)<.02)&(abs(y)<.02)))|((x>.04)&(abs(y)<.012))
    p=np.c_[x[keep],y[keep],np.zeros(keep.sum())+.02]
    f=flat_ring_features(p)
    assert f['status']=='PASS'
    assert np.linalg.norm(f['hole_center'][:2])<.003
    assert .05<f['handle_center'][0]<.075
    assert abs(f['handle_center'][1])<.004
    # Renaming, rotating and translating geometry needs no task branch.
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    t=np.array([.3,.4,.5]);g=flat_ring_features(p@r.T+t)
    assert g['status']=='PASS'
    assert np.linalg.norm(np.asarray(g['hole_center'])-(r@f['hole_center']+t))<.004


def test_closed_block_is_not_an_insertion_hole():
    from capx.execution.insertion_geometry import flat_ring_features
    x,y=np.meshgrid(np.arange(-.04,.081,.001),np.arange(-.04,.041,.001))
    assert flat_ring_features(np.c_[x.ravel(),y.ravel(),np.ones(x.size)*.02])['status']=='UNKNOWN'
