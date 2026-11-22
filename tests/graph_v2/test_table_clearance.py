import copy
import numpy as np
import pytest
from capx.execution.table_clearance import current_table_evidence


def fixture(held=True):
    rgb=np.full((128,128,3),130,np.uint8);depth=np.ones((128,128),float)
    mask=np.zeros(depth.shape,bool)
    if held:
        mask[56:73,56:73]=True;rgb[mask]=[0,160,0];depth[mask]=.96
        rgb[56,56:73]=[60,75,60]  # public mixed-color edge: only instance mask covers it
    rgb[92:109,92:109]=[180,0,0];depth[92:109,92:109]=.96
    t=np.diag([1.,-1.,-1.,1.]);t[:3,3]=[.5,0,1]
    obs={'camera':{'images':{'rgb':rgb,'depth':depth},'intrinsics':np.array([[400,0,64],[0,400,64],[0,0,1.]]),'pose_mat':t},'robot_joint_pos':np.r_[np.zeros(7),.5],'robot_gripper_joint_pos':np.array([.02,-.019]),'robot_cartesian_pos':np.array([.5,0,.05,0,1,0,0,.5])}
    region={'frame':'robot_base','center':[.5,0,0],'lo':[.47,-.03,0],'hi':[.53,.03,0],'footprint_half_extent_xy':[.03,.03],'clearance_verified':True,'footprint_coverage_fraction':1.,'center_coverage_verified':True,'visible_plane_points':1000,'id':'region-1'}
    pred={'center':[.5,0,.02],'height_m':.04,'orientation_verified':True}
    geometry={'red cube':{'frame':'robot_base','height_m':.04,'shape':'cube','source_id':'red-geometry'}}
    return obs,region,pred,geometry,mask


class Robot:
    def __init__(self,all_robot=False):self.calls=[];self.all_robot=all_robot
    def classify(self,p,c,q,**kwargs):
        self.calls.append(kwargs)
        return {'robot_mask':(np.ones(len(p),bool)&~kwargs['protected_mask']) if self.all_robot else np.zeros(len(p),bool),'evidence':{'kind':'mock_current_mesh','status':'PASS'}}


def run(obs,region,pred,geometry,mask,robot=None,**kwargs):
    calls=[]
    def segment(image,text_prompt):calls.append(text_prompt);return [{'mask':mask,'score':.9,'label':text_prompt}]
    result=current_table_evidence(obs,region,'green cube',pred,robot or Robot(),segment,observation_id='current-1',held_identity_verified=True,nonheld_geometry=geometry,static_scene_assumption=True,**kwargs)
    return result,calls


def test_current_mask_rgbd_fk_and_static_context_end_to_end():
    o,r,p,g,m=fixture();robot=Robot();out,calls=run(o,r,p,g,m,robot)
    assert calls==['green cube']
    assert robot.calls[0]['finger_positions_m']==pytest.approx([.02,.019])
    assert out['context']['all_known_nonheld_visible_outside'] is True
    assert out['context']['held_identity_verified'] is True
    assert out['validation']['status']=='PASS'
    assert out['validation']['conditional_static_scene'] is True
    assert len(out['held_points'])==m.sum()
    assert out['evidence']['new_llm_requests']==0


def test_nonheld_named_colour_never_removed_even_sam_mask_overlaps():
    o,r,p,g,m=fixture();m[92:109,92:109]=True;out,_=run(o,r,p,g,m,Robot(all_robot=True))
    assert not out['masks']['held'][92:109,92:109].any()
    assert not out['masks']['robot'][92:109,92:109].any()


def test_held_mask_geometry_incompatible_fails_closed():
    o,r,p,g,m=fixture();p['center']=[.7,0,.02];out,_=run(o,r,p,g,m)
    assert out['validation']['status']=='UNCERTAIN'
    assert not len(out['held_points'])


def test_invalid_mask_shape_fails_closed():
    o,r,p,g,m=fixture();out,_=run(o,r,p,g,np.ones((4,4),bool))
    assert out['validation']['status']=='UNCERTAIN'
    assert not len(out['held_points'])


def test_segmentation_exception_redacted_and_not_obstacle_claim():
    o,r,p,g,m=fixture()
    def segment(*args,**kwargs):raise RuntimeError('private authorization example must not leak')
    out=current_table_evidence(o,r,'green cube',p,Robot(),segment,observation_id='o',held_identity_verified=True,nonheld_geometry=g)
    assert out['validation']['status']=='UNCERTAIN'
    assert 'private authorization' not in str(out['evidence'])
    assert out['evidence']['segmentation_error_type']=='RuntimeError'


def test_missing_second_finger_stays_missing():
    o,r,p,g,m=fixture();o.pop('robot_gripper_joint_pos');robot=Robot();run(o,r,p,g,m,robot)
    assert robot.calls[0]['finger_positions_m'] is None


def test_no_known_grasp_still_accepts_fully_visible_clear_table():
    o,r,p,g,m=fixture(False)
    def segment(*args,**kwargs):raise AssertionError('No need to segment a held object without verified identity')
    out=current_table_evidence(o,r,None,None,Robot(),segment,observation_id='o',held_identity_verified=False,nonheld_geometry=g)
    assert out['validation']['status']=='PASS'
    assert not out['validation']['conditional_static_scene']
    assert not len(out['held_points'])


def test_unknown_nonheld_full_bounds_cannot_authorize_hidden_patch():
    o,r,p,g,m=fixture();out,_=run(o,r,p,{},m)
    assert out['context']['all_known_nonheld_visible_outside'] is False
    assert out['validation']['status']=='UNCERTAIN'


def test_unknown_current_obstacle_preserved():
    o,r,p,g,m=fixture();o['camera']['images']['depth'][58:61,73:76]=.90;o['camera']['images']['rgb'][58:61,73:76]=[0,0,180];m[58:61,73:76]=False
    out,_=run(o,r,p,g,m)
    assert out['validation']['status']=='FAIL'
    assert out['validation']['obstacle_points']>=6


def test_artifact_callback_receives_current_aligned_masks():
    o,r,p,g,m=fixture();saved=[]
    def callback(arrays):saved.append(arrays);return {'artifact_id':'mask-1'}
    out,_=run(o,r,p,g,m,artifact_callback=callback)
    assert len(saved)==1
    assert saved[0]['held'].shape==(128,128)
    assert out['evidence']['mask_artifact']=={'artifact_id':'mask-1'}


def test_depth_valid_index_mapping_is_exact():
    o,r,p,g,m=fixture();o['camera']['images']['depth'][0,0]=float('nan');o['camera']['images']['depth'][1,1]=0
    out,_=run(o,r,p,g,m)
    assert len(out['filtered_points'])==128*128-2
    assert not out['masks']['valid_depth'][0,0]
    assert not out['masks']['valid_depth'][1,1]


def test_missing_observation_id_rejected():
    o,r,p,g,m=fixture()
    with pytest.raises(ValueError):current_table_evidence(o,r,'green cube',p,Robot(),lambda *a,**k:[],observation_id='',held_identity_verified=True,nonheld_geometry=g)


def test_rejected_mask_reports_geometry_metrics():
    o,r,p,g,m=fixture();p['center']=[.7,0,.02];out,_=run(o,r,p,g,m)
    assert out['evidence']['held_mask_candidates'][0]['reason']=='current_mask_geometry_inconsistent'
    assert out['evidence']['held_mask_candidates'][0]['compatible_fraction']==0


def test_invalid_service_response_stays_uncertain():
    o,r,p,g,m=fixture()
    out=current_table_evidence(o,r,'green cube',p,Robot(),lambda *a,**k:None,observation_id='o',held_identity_verified=True,nonheld_geometry=g)
    assert out['validation']['status']=='UNCERTAIN'
