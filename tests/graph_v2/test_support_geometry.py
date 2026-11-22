"""Public-only geometry contracts, including deliberately unobservable cases."""
import numpy as np
import pytest
from capx.execution.support_geometry import color_points,current_support,reliable_top,retained_consistent


def geometry(height=.04):
 return {'frame':'robot_base','object':'green cube','height_m':height,'shape':'cube',
         'orientation_policy':'upright_only','uncertainty_kind':'heuristic_uncalibrated'}


def faces(yaw=.3,tilt=0.,floor=0.,which=('x-','y-'),samples=19):
 t=np.linspace(-.019,.019,samples);a,b=np.meshgrid(t,t)
 p=[]
 for name in which:
  axis='xyz'.index(name[0]);other=[i for i in range(3) if i!=axis]
  q=np.zeros((a.size,3));q[:,axis]=-.02 if name[1]=='-' else .02
  q[:,other[0]]=a.ravel();q[:,other[1]]=b.ravel();p.append(q)
 points=np.concatenate(p)
 rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
 rx=np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
 return points@(rz@rx).T+[.5,.1,.02+floor]


def prior():
 return {'frame':'robot_base','center':[.5,.1,.02],'lo':[.472,.072,0.],
         'hi':[.528,.128,.04],'top_xy_bounds':[[.472,.072],[.528,.128]],'top_pixels':400}


@pytest.mark.parametrize('yaw',[0.,.3,.8,1.4])
def test_two_current_faces_recover_public_cube_center(yaw):
 surface=current_support(faces(yaw),0.,geometry())
 assert surface is not None
 assert np.linalg.norm(np.asarray(surface['center'])-[.5,.1,.02])<.001
 assert surface['frame']=='robot_base' and abs(surface['top_z']-.04)<.001
 assert surface['fit_evidence']['independent_sensor'] is False
 assert surface['fit_evidence']['current_vertical_faces']==2
 assert 'upright' in ' '.join(surface['assumptions'])


def test_noisy_faces_are_deterministic_without_rng_or_truth():
 p=faces();p+=np.random.default_rng(8).normal(0,.00015,p.shape)
 a=current_support(p,0.,geometry());b=current_support(p,0.,geometry())
 assert a==b and a is not None
 assert np.linalg.norm(np.asarray(a['center'])[:2]-[.5,.1])<.001


@pytest.mark.parametrize('points',[
 faces(which=('y-',)),faces(tilt=np.deg2rad(26)),faces(floor=.025),
 faces()[faces()[:,2]>.012],np.repeat(faces()[:2],200,axis=0),np.empty((0,3))])
def test_unobservable_or_non_upright_support_is_unknown(points):
 assert current_support(points,0.,geometry()) is None


@pytest.mark.parametrize('height',[.02,.08])
def test_incompatible_public_shape_prior_is_rejected(height):
 assert current_support(faces(),0.,geometry(height)) is None


def test_unknown_table_or_non_cube_prior_is_rejected():
 assert current_support(faces(),None,geometry()) is None
 assert current_support(faces(),0.,{**geometry(),'shape':'cylinder'}) is None


def test_full_top_needs_both_visible_dimensions_and_enough_pixels():
 assert reliable_top(prior(),geometry())
 assert not reliable_top({**prior(),'top_xy_bounds':[[.495,.072],[.505,.128]]},geometry())
 assert not reliable_top({**prior(),'top_pixels':10},geometry())
 assert not reliable_top(None,geometry())


def test_current_partial_points_can_only_confirm_noncontradiction():
 partial=faces(which=('y-',))
 assert current_support(partial,0.,geometry()) is None
 assert retained_consistent(partial,prior(),geometry()) is True
 assert retained_consistent(partial+[.03,0,0],prior(),geometry()) is False
 assert retained_consistent(np.empty((0,3)),prior(),geometry()) is False


def test_retained_surface_rejects_elevation_and_malformed_values():
 assert not retained_consistent(faces(floor=.025),prior(),geometry())
 assert not retained_consistent(np.array([[np.nan,0,0]]),prior(),geometry())
 assert not retained_consistent(faces(),{},geometry())


def test_color_points_uses_only_public_rgbd_and_calibration():
 rgb=np.array([[[220,20,20],[20,220,20]],[[220,20,20],[20,20,220]]],dtype=np.uint8)
 depth=np.array([[1.,2.],[0.,1.]])
 t=np.eye(4);t[:3,3]=[.1,.2,.3]
 obs={'camera':{'intrinsics':np.eye(3),'pose_mat':t,'images':{'rgb':rgb,'depth':depth}}}
 assert np.allclose(color_points(obs,'red cube'),[[.1,.2,1.3]])
 assert np.allclose(color_points(obs,'green cube'),[[2.1,.2,2.3]])
 with pytest.raises(ValueError):color_points(obs,'unknown')
 assert color_points({},'red cube').shape==(0,3)
