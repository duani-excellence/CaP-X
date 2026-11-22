import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from capx.execution.depth_sampling import DepthSamplingProbe, resolve_profile

POSITIONS=[[.375,.125],[.875,.375],[.125,.625],[.625,.875]]

def trials(selected=0, rule=None):
    out=[]
    for i in range(4):
        d=np.roll([.2,.4,.6,.8],i)
        value=float(d[selected]) if rule is None else float(rule(d))
        out.append(dict(assigned_sample_depths=d.tolist(),resolved_depths=[value]*3))
    return out

def test_fixed_sample_measured_not_fitted_to_scene():
    r=resolve_profile(POSITIONS,trials(),sample_count=4)
    assert r['status']=='PASS' and r['selected_sample']==0
    assert r['depth_offset_from_pixel_center_xy']==[-.125,.375]
    assert not r['fitted_to_scene'] and len(r['profile_sha256'])==64
    assert json.loads(json.dumps(r))==r

@pytest.mark.parametrize('rule',[np.mean,np.min,np.max])
def test_non_fixed_depth_resolve_is_unknown(rule):
    assert resolve_profile(POSITIONS,trials(rule=rule),sample_count=4)['status']=='UNCERTAIN'

def test_spatially_varying_sample_is_unknown():
    t=trials();t[0]['resolved_depths'][1]=.4
    assert resolve_profile(POSITIONS,t,sample_count=4)['status']=='UNCERTAIN'

def test_not_enough_depth_permutations_is_unknown():
    assert resolve_profile(POSITIONS,trials()[:1],sample_count=4)['status']=='UNCERTAIN'

def test_single_sample_is_known_center():
    r=resolve_profile([],[],sample_count=0)
    assert r['status']=='PASS' and r['depth_offset_from_pixel_center_xy']==[0.,0.]

class Backend:
    def __init__(self):
        self.description={'samples':4,'width':640,'format':42,'framebuffer':7};self.probes=0
    def describe(self,con):return copy.deepcopy(self.description)
    def sample(self,description):self.probes+=1;return POSITIONS,trials()

def context():return SimpleNamespace(con=object(),gl_ctx=SimpleNamespace(make_current=lambda:None))

def test_cache_tracks_context_size_format_and_framebuffer_and_is_copy():
    b=Backend();p=DepthSamplingProbe(backend=b);c=context()
    first=p.measure(c);first['status']='mutated'
    assert p.measure(c)['status']=='PASS' and b.probes==1
    for key,val in [('width',641),('format',43),('framebuffer',8)]:
        b.description[key]=val;p.measure(c)
    assert b.probes==4
    c.con=object();p.measure(c);assert b.probes==5
    c.gl_ctx=SimpleNamespace(make_current=lambda:None);p.measure(c);assert b.probes==6

def test_unknown_backend_failure_redacts_exception_data():
    b=Backend()
    def fail(con):raise RuntimeError('Authorization: private-secret')
    b.describe=fail;r=DepthSamplingProbe(backend=b).measure(context())
    assert r['status']=='UNCERTAIN' and 'private-secret' not in json.dumps(r)
    assert r['error_type']=='RuntimeError'

def test_no_renderer_is_unknown():
    assert DepthSamplingProbe(backend=Backend()).measure(None)['status']=='UNCERTAIN'
