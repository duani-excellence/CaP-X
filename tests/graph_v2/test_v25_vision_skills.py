import numpy as np
import pytest
from capx.execution.skill_contracts import ContractGraph,ContractBroker


def test_known_phase_stop_retains_verified_effects_without_replaying_call():
    from capx.execution.skill_contracts import SkillBoundaryStop
    d=dict(schema='capx.contract_task.v1',objects=['part'],resources=['arm'],goals=['placed'],intents=[
        dict(id='a',skill='acquire',requires=[],effects=['held'],resources=['arm']),
        dict(id='p',skill='deposit',requires=['held'],effects=['placed'],resources=['arm'],after=['a'])])
    c=ContractGraph(d)
    def stop(row):raise SkillBoundaryStop('missing release evidence',{'held':{'truth':'TRUE','evidence':{'public':'hold check'}}})
    b=ContractBroker(c,{'acquire':lambda r:None,'deposit':stop},lambda ks:{k:dict(truth='TRUE',evidence={}) for k in ks})
    b.execute('a')
    with pytest.raises(SkillBoundaryStop):b.execute('p')
    assert not c.quarantined and c.truth('held')=='TRUE'
    assert 'a' in c.completed and not c.permissions and not c.locks
    with pytest.raises(ValueError,match='already completed'):b.execute('a')


def test_registration_tracks_rigid_motion_and_rejects_wrong_object():
    from capx.execution.vision_skills import register_rigid
    rng=np.random.default_rng(6);p=rng.uniform(-.03,.03,(2000,3));t=np.eye(4);t[:3,3]=[.2,-.1,.3]
    r=register_rigid(p,p+t[:3,3],t)
    assert r['status']=='PASS'
    assert np.linalg.norm(np.array(r['transform'])-t)<1e-5
    assert register_rigid(p,p+1,t)['status']=='UNKNOWN'


def test_pose_propagation_rotates_object_offset_not_only_tcp():
    from capx.execution.vision_skills import carried_shape
    s=dict(center=[.1,0,.02],height_m=.04,half_extent_xy=[.02,.04],bottom_z=0.,top_z=.04)
    a=np.eye(4);a[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];a[:3,3]=[.3,0,.1]
    q=carried_shape(s,a)
    assert q['center']==pytest.approx([.3,.1,.12]);assert q['half_extent_xy']==pytest.approx([.04,.02])
    assert q['bottom_z']==pytest.approx(.1)


def test_public_tcp_settling_is_bounded_and_does_not_relax_tolerance():
    from capx.execution.vision_skills import RegionSkills
    class Scene:
        obs={'robot_cartesian_pos':np.r_[.008,0,0,1,0,0,0]}
        def frame(self):return self.obs
    s=Scene();calls=[]
    def motion(name,*args):
        calls.append(name)
        if name=='wait':s.obs['robot_cartesian_pos'][0]*=.5
    profile={'schema':'capx.upright_development_bounds.v1','offline_evidence':['fixture']}
    skills=RegionSkills(s,motion,None,None,profile)
    skills.move([0,0,0],[1,0,0,0])
    assert calls==['goto_pose','wait','wait']
    s.obs['robot_cartesian_pos'][0]=1
    with pytest.raises(Exception,match='did not converge'):skills.move([0,0,0],[1,0,0,0])
    assert len(calls)<=8


def test_signed_finger_encoders_measure_separation_not_cancelling_sum():
    from capx.execution.vision_skills import jaw_separation
    assert jaw_separation([.02,-.02])==pytest.approx(.04)
    assert jaw_separation([.02,.02])==pytest.approx(.04)
    with pytest.raises(ValueError):jaw_separation([.1,-.02])


def test_public_depth_association_is_current_measurement_not_prediction_fill():
    from capx.execution.vision_skills import depth_association
    cam={'images':{'depth':np.ones((30,30))},'intrinsics':np.array([[100.,0,15],[0,100.,15],[0,0,1]]),'pose_mat':np.eye(4)}
    shape=dict(center=[0,0,1],half_extent_xy=[.05,.05],height_m=.1)
    p=depth_association({'camera':cam},shape,.002,.002)
    assert len(p)>30 and np.allclose(p[:,2],1)
    shape['center']=[0,0,.5]
    assert len(depth_association({'camera':cam},shape,.002,.002))==0


def test_prism_memory_can_match_newly_visible_face_but_is_marked_modelled():
    from capx.execution.vision_skills import prism_surface,register_rigid
    s=dict(center=[0,0,.05],extent_xy=[.04,.04],height_m=.1,basis_xy=np.eye(2).tolist())
    model=prism_surface(s)
    assert len(model)>100 and np.max(model[:,2])==pytest.approx(.1)
    current=model[(model[:,0]>.019)|(model[:,1]>.019)]
    assert register_rigid(model,current,np.eye(4))['status']=='PASS'


def test_release_correction_targets_interval_interior_not_edge_and_is_bounded():
    from capx.execution.vision_skills import release_correction
    shape=dict(center=[.09,.06,.1],half_extent_xy=[.03,.03],bottom_z=.001)
    region=dict(lo=[0,0],hi=[.2,.2],floor_z=0.)
    d=release_correction(shape,region,.014,.010)
    assert np.linalg.norm(d)<=.03 and d[1]>0 and d[2]>0
    shape['center']=[.6,.06,.1]
    with pytest.raises(ValueError):release_correction(shape,region,.014,.010)


def test_settled_estimate_uses_current_depth_and_retains_canonical_shape_source():
    from capx.execution.vision_skills import settled_prediction,carried_shape
    s=dict(center=[0,0,.05],height_m=.1,half_extent_xy=[.02,.02],bottom_z=0.,top_z=.1)
    t=np.eye(4);t[:3,3]=[.2,.3,.02]
    predicted=settled_prediction(s,t,.0)
    assert carried_shape(s,predicted)['bottom_z']==pytest.approx(0.)
    assert np.asarray(predicted)[:2,3]==pytest.approx([.2,.3])


def test_geometric_grasp_proposals_are_public_shape_and_jaw_limited():
    from capx.execution.vision_skills import planar_jaw_proposals
    p=planar_jaw_proposals(dict(basis_xy=np.eye(2),extent_xy=[.04,.10]))
    assert p and all(x['width_m']<=.074 for x in p)
    assert all(x['source']=='public_prism_geometry_not_GraspNet' for x in p)
    assert not planar_jaw_proposals(dict(basis_xy=np.eye(2),extent_xy=[.11,.12]))


def test_horizontal_rim_is_retained_as_obstacle_not_only_vertical_planes():
    from capx.execution.vision_skills import receptacle_obstacles
    from capx.execution.grasp_geometry import panda_open_sweep
    x,y=np.meshgrid(np.linspace(-.02,.02,20),np.linspace(.08,.10,20))
    rim=np.c_[x.ravel(),y.ravel(),np.full(x.size,.06)]
    target=np.tile([0,0,.02],(100,1))
    obstacles=receptacle_obstacles([dict(points=np.r_[rim,target])],target)
    assert len(obstacles)==len(rim)
    result=panda_open_sweep([0,0,0],[0,1,0,0],{'receptacle':obstacles})
    assert result['status']=='FAIL' and 'palm' in result['collisions']['receptacle']


def test_historical_goal_does_not_survive_missing_current_placement_evidence():
    from capx.execution.vision_skills import RegionSkills
    from capx.execution.skill_contracts import SkillBoundaryStop
    class Scene:
        source='current'
        obs={'robot_joint_pos':np.zeros(7)}
        def frame(self):return self.obs
    s=RegionSkills(Scene(),None,None,None,{'schema':'capx.upright_development_bounds.v1','offline_evidence':['fixture']})
    s.effects['placed:any-name']='TRUE'
    s.released['any-name']={}
    def missing(name):raise SkillBoundaryStop('current instance hidden',{})
    s.observe_placement=missing
    row=s.verify(['placed:any-name'])['placed:any-name']
    assert row['truth']=='UNKNOWN'
    assert s.effects['placed:any-name']=='UNKNOWN'
    assert row['evidence']['source']=='current'


def test_unverified_attachment_is_not_advertised_as_empty_robot():
    from capx.execution.vision_skills import RegionSkills
    class Scene:
        source='current'
        obs={'robot_joint_pos':np.zeros(7)}
        def frame(self):return self.obs
    s=RegionSkills(Scene(),None,None,None,{'schema':'capx.upright_development_bounds.v1','offline_evidence':['fixture']})
    s.held={'object':'part'};s.effects['held:part']='TRUE'
    with pytest.raises(Exception):s.stop('registration conflict')
    assert s.effects['held:part']=='UNKNOWN'
    assert s.verify(['robot_ready'])['robot_ready']['truth']=='FALSE'
