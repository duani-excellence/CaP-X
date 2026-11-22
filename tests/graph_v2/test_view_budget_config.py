"""Configured public-view budget must reach both graph arms without fallback."""
import hashlib
from types import SimpleNamespace
import pytest
from capx.experiments import graph_v2 as runner


def test_legacy_view_limit_stays_two_and_new_budget_is_explicit():
    assert runner.limits({})['max_episode_views']==2
    assert runner.limits({'budgets':{'max_episode_views':8}})['max_episode_views']==8
    assert runner.limits({'budgets':{'max_episode_views':0}})['max_episode_views']==0


@pytest.mark.parametrize('value',[-1,1.5,True,'8',None])
def test_invalid_view_limit_rejected_before_environment_creation(value):
    with pytest.raises(ValueError,match='view'):
        runner.limits({'budgets':{'max_episode_views':value}})


@pytest.mark.parametrize('group',['G0','G1'])
def test_install_passes_identical_configured_view_budget(tmp_path,monkeypatch,group):
    from capx.execution import grounded_feedback,bounded_ik
    asset=tmp_path/'robot.urdf';asset.write_text('<robot/>')
    calibration={'calibration_hash':'fixture','sources':[{'role':'ik_urdf','path':str(asset),'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}]}
    env=SimpleNamespace(public_robot_calibration=lambda:calibration,_functions={},
        _actor=SimpleNamespace(apis={},call_site={}),
        _apis={'FrankaControlApi':SimpleNamespace(ik_solve_fn=lambda:None)})
    monkeypatch.setattr(bounded_ik,'BoundedIK',lambda *a,**kw:SimpleNamespace(successful_calls=0))
    monkeypatch.setattr(grounded_feedback,'GroundedMonitor',lambda *a,**kw:SimpleNamespace(**kw))
    m=runner.install_monitor(env,group,'stack',tmp_path,runner.limits({'budgets':{'max_episode_views':8}}),'none')
    assert m.max_episode_views==8
