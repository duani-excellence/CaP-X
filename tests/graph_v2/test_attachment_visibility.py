"""Current visible faces must constrain centre; partial centroids are not centres."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from capx.execution.grip_offset import check_surface,calibrate,predict
FIXTURES=Path(__file__).with_name('fixtures')
def predicted():return dict(center=[.5,.1,.12],hi=[.52,.12,.14],lo=[.48,.08,.10],height_m=.04,orientation_verified=True)
def geometry():return dict(frame='robot_base',shape='cube',height_m=.04,orientation_verified=True)
def cloud(shift=(0,0,0),face_count=2):
 h=.04;grid=np.linspace(-h/2,h/2,25);a,b=np.meshgrid(grid,grid);a=a.ravel();b=b.ravel()
 top=np.c_[a[a<-.008],b[a<-.008],np.full(sum(a<-.008),h/2)]
 faces=[np.c_[np.full(len(a),-h/2),a,b],np.c_[a,np.full(len(a),-h/2),b]]
 return np.vstack([top,*faces[:face_count]])+np.array([.5,.1,.12])+shift

def partial(p):
 lo,hi=np.quantile(p,[.01,.99],axis=0);top=p[p[:,2]>hi[2]-.002]
 return dict(center=[*np.median(top[:,:2],axis=0),(lo[2]+hi[2])/2],lo=lo.tolist(),hi=hi.tolist(),top_xy_bounds=np.quantile(top[:,:2],[.01,.99],axis=0).tolist(),top_pixels=len(top),pixels=len(p))

def test_partial_centroid_bias_uses_two_current_faces_not_center_recalibration():
 p=cloud();v=partial(p);v['center'][0]-=.01;before=copy.deepcopy(predicted())
 out=check_surface(before,v,points=p,geometry=geometry())
 assert out['status']=='PASS' and out['visibility']=='partial_top'
 assert out['center_observable'] is True and out['independent_verification'] is False
 assert out['localized_center_disagreement_m']<.003
 assert out['conditional'] is True and before==predicted()

@pytest.mark.parametrize('shift',[(.025,0,0),(0,0,-.02)])
def test_real_shift_or_vertical_slip_contradicts_current_faces(shift):
 p=cloud(shift);out=check_surface(predicted(),partial(p),points=p,geometry=geometry())
 assert out['status']=='FAIL'

def test_one_side_and_partial_top_do_not_approve_a_rigid_prior():
 p=cloud(face_count=1);out=check_surface(predicted(),partial(p),points=p,geometry=geometry())
 assert out['status']=='UNCERTAIN' and out['center_observable'] is False

@pytest.mark.parametrize('p',[np.zeros((0,3)),np.tile([.5,.1,.14],(300,1)),np.full((50,3),np.nan)])
def test_insufficient_or_degenerate_points_never_pass(p):
 out=check_surface(predicted(),{'center':[.5,.1,.12],'hi':[.52,.12,.14]},points=p,geometry=geometry())
 assert out['status']=='UNCERTAIN'

def test_points_inside_prior_solid_are_not_surface_or_no_slip_proof():
 p=np.random.default_rng(3).uniform([.49,.09,.11],[.51,.11,.13],(800,3))
 assert check_surface(predicted(),partial(p),points=p,geometry=geometry())['status']!='PASS'

def test_full_top_preserves_threshold_and_tilt_is_not_overridden():
 v=dict(center=[.516,.1,.12],lo=[.496,.08,.1],hi=[.536,.12,.14],top_xy_bounds=[[.496,.08],[.536,.12]],top_pixels=200)
 assert check_surface(predicted(),v,geometry=geometry())['status']=='FAIL'
 q=predicted();q['orientation_verified']=False;p=cloud()
 assert check_surface(q,partial(p),points=p,geometry=geometry())['status']=='FAIL'

@pytest.mark.parametrize('task,expected',[('stack','UNCERTAIN'),('restack','PASS')])
def test_real_v3_frozen_public_frames(task,expected):
 meta=json.loads((FIXTURES/f'oracle_v3_{task}_held.json').read_text());points=np.load(FIXTURES/f'oracle_v3_{task}_held.npz')['points']
 assert meta['private_truth_used'] is False
 out=check_surface(meta['predicted'],meta['visible'],points=points,geometry=meta['geometry'])
 assert out['visibility']=='partial_top' and out['status']==expected
 assert out['independent_verification'] is False
 if expected=='PASS':assert out['localized_center_disagreement_m']<.003

def test_yaw_preserves_conditional_upright_height_but_pitch_does_not():
 s=dict(center=[.51,.1,.12],hi=[.53,.12,.14],lo=[.49,.08,.1]);a=calibrate(s,geometry(),[.5,.1,.12,1,0,0,0],source_id='s',tcp_id='t')
 yaw=predict(a,[.5,.1,.12,np.cos(.3),0,0,np.sin(.3)])
 pitch=predict(a,[.5,.1,.12,np.cos(.3),0,np.sin(.3),0])
 assert yaw['orientation_verified'] and not pitch['orientation_verified']
 assert yaw['center'][:2]==pytest.approx([.5+.01*np.cos(.6),.1+.01*np.sin(.6)])
