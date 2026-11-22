from types import SimpleNamespace
from capx.experiments import vision_tasks

class Forbidden:
    def __getattr__(self,name):raise AssertionError('private state accessed: '+name)


def test_public_depth_getter_only_reads_current_render_context(monkeypatch):
    import capx.execution.depth_sampling as module
    calls=[]
    class Probe:
        def measure(self,ctx):calls.append(ctx);return {'status':'PASS','public':'calibration'}
    monkeypatch.setattr(module,'DepthSamplingProbe',Probe)
    ctx=object();sim=SimpleNamespace(_render_context_offscreen=ctx,data=Forbidden(),model=Forbidden())
    low=SimpleNamespace(robosuite_env=SimpleNamespace(sim=sim),render_camera_names=['camera'])
    env=SimpleNamespace(low_level_env=low,evaluation_state=Forbidden())
    first=vision_tasks.public_depth_sampling(env);probe=env._depth_sampling_probe
    assert first=={'camera':{'status':'PASS','public':'calibration'}}
    second=vision_tasks.public_depth_sampling(env)
    assert env._depth_sampling_probe is probe and calls==[ctx,ctx]
    first['camera']['status']='changed';assert second['camera']['status']=='PASS'


def test_missing_render_context_returns_unknown_without_private_fallback():
    sim=SimpleNamespace(_render_context_offscreen=None,data=Forbidden(),model=Forbidden())
    env=SimpleNamespace(low_level_env=SimpleNamespace(robosuite_env=SimpleNamespace(sim=sim),render_camera_names=['camera']))
    assert vision_tasks.public_depth_sampling(env)['camera']['status']=='UNCERTAIN'
