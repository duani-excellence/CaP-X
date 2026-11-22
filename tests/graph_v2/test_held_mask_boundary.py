import numpy as np
from capx.execution.table_clearance import current_table_evidence
from test_table_clearance import fixture,Robot

def current_prediction(pred):
 return {**pred,'yaw_rad':0.,'source':'current_two_visible_vertical_faces_and_top_with_prior_public_cube_side','face_point_counts':[60,70]}

def run_case(o,r,p,g,mask):
 return current_table_evidence(o,r,'green cube',p,Robot(),lambda *a,**k:[{'mask':mask,'score':.9}],observation_id='current',held_identity_verified=True,nonheld_geometry=g,static_scene_assumption=True)

def test_one_pixel_semantic_edge_with_current_cube_ray_surface_can_be_completed():
 o,r,p,g,m=fixture();p=current_prediction(p);m[56,:]=False
 out=run_case(o,r,p,g,m)
 assert out['masks']['held'][56,56:73].all()
 assert out['validation']['status']=='PASS'
 assert out['evidence']['held_mask_boundary_refinement']['added_pixels']==17
 assert out['evidence']['held_mask_boundary_refinement']['surface_tolerance_m']==.002

def test_foreign_depth_in_adjacent_mask_pixel_never_erased():
 o,r,p,g,m=fixture();p=current_prediction(p);m[56,:]=False;o['camera']['images']['depth'][56,56:73]=.90
 out=run_case(o,r,p,g,m)
 assert not out['masks']['held'][56,56:73].any()
 assert out['validation']['status']=='FAIL'

def test_prior_only_pose_cannot_complete_current_semantic_mask():
 o,r,p,g,m=fixture();m[56,:]=False
 out=run_case(o,r,p,g,m)
 assert not out['masks']['held'][56,56:73].any()

def test_nonheld_color_wins_over_adjacent_cube_surface_completion():
 o,r,p,g,m=fixture();p=current_prediction(p);m[56,:]=False;o['camera']['images']['rgb'][56,56:73]=[180,0,0]
 out=run_case(o,r,p,g,m)
 assert not out['masks']['held'][56,56:73].any()

def test_nonadjacent_points_not_filled_even_if_on_same_cube_surface():
 o,r,p,g,m=fixture();p=current_prediction(p);m[56:59,:]=False
 out=run_case(o,r,p,g,m)
 assert not out['masks']['held'][56:58,56:73].any()


def _edge_sample(focal=500.,size=3,depth=.98):
 from capx.execution.table_clearance import _complete_held_boundary
 center=size//2;cx=center-.0206*focal/depth;k=np.array([[focal,0,cx],[0,focal,center],[0,0,1.]])
 yy,xx=np.indices((size,size));points=np.stack([(xx-cx)*depth/focal,(yy-center)*depth/focal,np.full((size,size),depth)],axis=-1)
 mask=np.zeros((size,size),bool);mask[center,center-1]=True
 pred=current_prediction({'center':[0,0,1.],'height_m':.04,'orientation_verified':True})
 return _complete_held_boundary(points,mask,np.zeros_like(mask),pred,np.zeros(3),camera_intrinsics=k,camera_rotation=np.eye(3)),center


def test_fitted_edge_subpixel_coverage_uses_current_camera_cell_not_box_inflation():
 (mask,evidence),center=_edge_sample(focal=500.)
 assert mask[center,center]
 assert evidence['ray_footprint_pixels']==.5
 (mask,_),center=_edge_sample(focal=2000.)
 assert not mask[center,center]


def test_pixel_footprint_agrees_across_resolution_at_fixed_calibration():
 for size in (3,7):
  (mask,_),center=_edge_sample(focal=500.,size=size)
  assert mask[center,center]


def test_pixel_footprint_keeps_foreign_foreground_and_background_depth():
 for depth in (.97,.99):
  (mask,_),center=_edge_sample(depth=depth)
  assert not mask[center,center]


def test_subpixel_grazing_corner_requires_continuous_cell_not_five_ray_sampling():
 from capx.execution.table_clearance import _complete_held_boundary
 camera=np.array([.10,-.70,.33]);z=-camera/np.linalg.norm(camera);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x);rotation=np.column_stack([x,y,z])
 points=np.tile(np.array([.020012,.020012,.005]),(2,2,1));mask=np.zeros((2,2),bool);mask[1,0]=True
 pred=current_prediction({'center':[0,0,0.],'height_m':.04,'orientation_verified':True})
 out,evidence=_complete_held_boundary(points,mask,np.zeros_like(mask),pred,camera,camera_intrinsics=np.diag([618.,618.,1.]),camera_rotation=rotation)
 assert out[1,1]
 assert evidence['footprint_method']=='continuous_visible_cube_surface_projection'
