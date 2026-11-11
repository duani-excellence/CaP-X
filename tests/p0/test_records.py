import json
from types import SimpleNamespace
import numpy as np
import pytest


def test_video_timestamp_alignment():
    from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
    e = object.__new__(RobosuiteBaseEnv)
    e._record_frames=True; e._record_wrist_camera=False
    e._frame_buffer=[]; e._frame_timestamps=[]
    e.save_camera_name='x'; e._render_width=e._render_height=2
    e.robosuite_env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=1.25),render=lambda **kw:np.zeros((2,2,3))))
    e._record_frame()
    assert len(e._frame_timestamps) == len(e._frame_buffer) == 1
    assert e._frame_timestamps[0] == 1.25
    e.get_video_frames(clear=True)
    assert not e._frame_timestamps


def test_request_count_and_no_secret(tmp_path, monkeypatch):
    from capx.utils.request_audit import RequestAudit, audited_post
    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **kw:SimpleNamespace(status_code=200))
    path=tmp_path/'requests.jsonl'
    with RequestAudit(path) as audit:
        audited_post('http://127.0.0.1/x', headers={'Authorization':'SECRET'}, json={'model':'fake','messages':['hello']})
        audited_post('http://127.0.0.1/x', headers={'Authorization':'SECRET'}, json={'model':'fake','messages':['hello']})
        assert audit.count == 2
    text=path.read_text()
    assert 'SECRET' not in text
    assert json.loads(text.splitlines()[0])['payload']['messages']==['hello']


def test_transport_is_system_error(monkeypatch):
    import requests
    from capx.utils.serve_utils import post_with_retries, RobotAPIConnectionError
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError('offline')))
    with pytest.raises(RobotAPIConnectionError):
        post_with_retries('http://localhost/fake', {}, retry_interval=0,max_retries=1)


def test_seeded_service_rng_scope():
    import asyncio
    import random
    import torch
    from capx.serving.launch_contact_graspnet_server import _run_on_gpu
    req=SimpleNamespace(seed=19)
    def sample(req):
        return {'values': [random.random(),float(np.random.rand()),float(torch.rand(1))]}
    a=asyncio.run(_run_on_gpu(sample,req))
    b=asyncio.run(_run_on_gpu(sample,req))
    assert a==b


def test_registry_modes():
    from capx.envs.tasks import get_config
    a=get_config('franka_pick_place_code_env')
    assert a.low_level=='franka_robosuite_cubes_low_level' and a.privileged
    b=get_config('franka_robosuite_pick_place_vision_code_env')
    assert not b.privileged and b.apis==['FrankaControlApi']

def test_sam3_outage_is_not_empty_detection(monkeypatch):
    from capx.integrations.vision import sam3
    from capx.utils.serve_utils import RobotAPIConnectionError
    monkeypatch.setattr(sam3,'post_with_retries',lambda *a,**kw:(_ for _ in ()).throw(RobotAPIConnectionError('offline')))
    with pytest.raises(RobotAPIConnectionError):
        sam3.init_sam3()(np.zeros((2,2,3),dtype=np.uint8),'cube')

def test_video_manifest_and_no_overwrite(tmp_path,monkeypatch):
    from capx.utils import video_utils as v
    class Writer:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def append_data(self,x):pass
    seen={}
    def writer(path,**kwargs):
        seen.update(kwargs);path.touch();return Writer()
    monkeypatch.setattr(v.imageio,'get_writer',writer)
    frames=[np.zeros((2,2,3),dtype=np.uint8)]*3
    v._write_video(frames,str(tmp_path),suffix='test',timestamps=[1.,1.25,1.5])
    manifest=json.loads((tmp_path/'video_test.timestamps.json').read_text())
    assert manifest['frames'][1]=={'frame':1,'sim_time':1.25}
    assert seen['fps']==4
    with pytest.raises(FileExistsError):
        v._write_video(frames,str(tmp_path),suffix='test',timestamps=[1.,1.25,1.5])
