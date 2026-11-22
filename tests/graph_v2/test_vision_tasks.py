"""Vision task adapters: public observation, independent reward, replay and broker."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from capx.experiments import vision_tasks as v


def raw_observation():
    return {'robot_joint_pos':np.zeros(8), 'robot_cartesian_pos':np.r_[.5,0,.3,1,0,0,0,1],
            'cube_poses':{'primary':[999]*7},'raw_success':True,'sim':object(),
            'robot0_robotview':{'pose':np.r_[0,0,0,1,0,0,0],'pose_mat':np.eye(4),
                'intrinsics':np.eye(3),'images':{'rgb':np.zeros((3,3,3),np.uint8),
                    'depth':np.ones((3,3,1)),'segmentation':np.ones((3,3,1))},'truth':'hidden'}}


class FakeLow:
    render_camera_names=['robot0_robotview']
    max_steps=1500
    _sim_step_count=0
    _frame_timestamps=[]
    def get_observation(self):return raw_observation()
    def move_to_joints_blocking(self,value):self.last_joints=value
    def _set_gripper(self,value):self.fraction=value
    def _step_once(self):self._sim_step_count+=1
    def enable_video_capture(self,*a,**kw):pass
    def get_video_frames(self,**kw):return []


class FakeRuntime:
    instances=[]
    def __init__(self,task,seed,root,render=True):
        self.task=task;self.seed=seed;self.root=Path(root);self.root.mkdir(parents=True)
        self.trace=self.root/'trace.jsonl';self.low=FakeLow();self.calls=self.motion=0
        self.active=False;self.stable=False;self.actor=SimpleNamespace(close=lambda:None)
        self.task_module=SimpleNamespace(PROMPT='Goal: '+task)
        self.initial={'task':task,'seed':seed,'state_hash':'fixed','model_hash':'model','physics':{'stable':True}}
        self.closed=False;FakeRuntime.instances.append(self)
    def state(self):return {'raw_success':self.stable,'still':True,'grasp':[self.stable],
                            'released':False,'objects':{'cube':{'truth':999}}}
    def call(self,name,args,kwargs):
        self.calls+=1
        if name=="close_gripper":self.motion+=1
        return self.functions[name](*args,**kwargs)
    def execute(self,code,timeout=90):
        result=self.actor.run(code,timeout=timeout)
        good=result['ok'] and self.motion>0 and self.stable
        return {'reward':float(good),'truncated':False,'info':{'physical_success':self.stable,
                    'execution_status':result['status'],'stdout':result['stdout'],'stderr':result['stderr'],
                    'api_calls':self.calls,'motion_calls':self.motion}}
    def close(self):self.closed=True;self.actor.close()


class FakeAPI:
    def __init__(self,low):self.low=low
    def get_object_pose(self,name,return_bbox_extent=False):
        assert 'cube_poses' not in self.low.get_observation()
        return [0,0,0],[1,0,0,0],[.04]*3 if return_bbox_extent else None
    def sample_grasp_pose(self,name):return [0,0,0],[1,0,0,0]
    def close_gripper(self):pass
    def functions(self):return {n:getattr(self,n) for n in ['get_object_pose','sample_grasp_pose','close_gripper']}
    def combined_doc(self):return 'PUBLIC VISION APIS'


@pytest.fixture
def fake(monkeypatch):
    FakeRuntime.instances=[]
    monkeypatch.setattr(v,'_runtime_factory',FakeRuntime)
    monkeypatch.setattr(v,'_vision_api_factory',FakeAPI)
    monkeypatch.setattr(v,'_robot_tcp',lambda low,raw:np.asarray(raw['robot_cartesian_pos']))


def test_public_filter_only_sensors_and_copies_arrays():
    raw=raw_observation();clean=v.public_sensors(raw,['robot0_robotview'])
    assert set(clean)=={'robot_joint_pos','robot_cartesian_pos','robot0_robotview'}
    assert set(clean['robot0_robotview'])=={'pose','pose_mat','intrinsics','images'}
    assert set(clean['robot0_robotview']['images'])=={'rgb','depth'}
    clean['robot_joint_pos'][0]=1
    assert raw['robot_joint_pos'][0]==0


def test_proxy_blocks_hidden_state_and_has_seeded_grasp_rng(fake):
    a=v.PublicLowLevel(FakeLow(),21);b=v.PublicLowLevel(FakeLow(),21)
    assert [a.next_grasp_seed() for _ in range(5)]==[b.next_grasp_seed() for _ in range(5)]
    for name in ['robosuite_env','cube_poses','initial','object_pose','state']:
        with pytest.raises(AttributeError):getattr(a,name)
    a.cube_points=np.ones((3,3))
    assert a.cube_points.shape==(3,3)
    assert 'cube_poses' not in a.get_observation()


def test_seed_validation_rejects_bool_negative_and_unknown_task(tmp_path):
    for seed in [True,-1,1.5]:
        with pytest.raises(ValueError):v.make_environment('lift',tmp_path/'unused',seed)
    with pytest.raises(ValueError):v.make_environment('nut_assembly',tmp_path/'unused',0)


def test_snapshot_mismatch_fails_closed_and_closes_runtime(fake,tmp_path):
    with pytest.raises(ValueError,match='Snapshot'):
        v.make_environment('lift',tmp_path/'x',21,{'task':'lift','seed':21,'state_hash':'wrong'})
    assert FakeRuntime.instances[-1].closed


def test_real_sandbox_calls_only_vision_and_actor_has_no_env(fake,tmp_path):
    env=v.make_environment('lift',tmp_path/'x',21)
    try:
        assert env.reset_info['initial_state_hash']=='fixed'
        assert env.cfg.privileged is False
        assert 'PUBLIC VISION APIS' in str(env._full_prompt)
        assert 'cube_poses' not in env._get_observation()
        _,reward,done,_,info=env.step('p,q,e=get_object_pose("red cube")\nprint(p)')
        assert reward==0 and not done and info['api_calls']==1
        assert info['execution_status']=='ok'
        assert env._actor.call_site['line']==1
        _,reward,_,_,info=env.step('print(env)')
        assert reward==0 and info['execution_status']!='ok'
    finally:env.close()


def test_monitor_boundary_receives_old_callback_and_budget_covers_recovery(fake,tmp_path):
    from capx.execution.sandbox import ExecutionInterrupted
    env=v.make_environment('lift',tmp_path/'x',0,api_budget=1)
    seen=[]
    class Monitor:
        def invoke(self,name,args,kwargs,fn):
            seen.append(name);return fn(name,args,kwargs)
    env._feedback_monitor=Monitor()
    try:
        _,reward,_,_,info=env.step('get_object_pose("red cube")\nget_object_pose("red cube")')
        assert seen==['get_object_pose','get_object_pose']
        assert info['execution_status']=='interrupted' and info['api_calls']==1 and reward==0
        with pytest.raises(ExecutionInterrupted):env._execute_api('sample_grasp_pose',['red cube'],{})
    finally:env.close()


def test_reset_rebuilds_fresh_runtime_clears_monitor_preserves_old_trace(fake,tmp_path):
    env=v.make_environment('restack',tmp_path/'x',0)
    first=env._trace_path;first_runtime=env._runtime
    env._feedback_monitor=object()
    env.reset(seed=0,options={'snapshot':copy.deepcopy(first_runtime.initial)})
    try:
        assert first_runtime.closed and env._trace_path!=first and first.exists()
        assert not hasattr(env,'_feedback_monitor')
        assert env._api_calls==0 and env.reset_info['initial_state_hash']=='fixed'
    finally:env.close()


def test_prepare_scene_writes_exclusive_manifest_and_replays(fake,tmp_path):
    entry,snapshot=v.prepare_scene('lift',21,tmp_path/'scene')
    assert entry['task']=='lift' and entry['state_hash']==snapshot['state_hash']
    assert (tmp_path/'scene'/entry['snapshot']).exists()
    assert (tmp_path/'scene'/'scenes.json').exists()
    assert entry['snapshot_replay_verified'] is True
    with pytest.raises(FileExistsError):v.prepare_scene('lift',21,tmp_path/'scene')


def test_stack_delegates_to_original_p0_and_snapshot(fake,monkeypatch,tmp_path):
    from capx.experiments import pilot
    calls=[]
    stack=SimpleNamespace(cfg=SimpleNamespace(),low_level_env=SimpleNamespace(capture_snapshot=lambda:{'state_hash':'a'}),reset=lambda **kw:(None,calls.append(kw) or {'initial_state_hash':'a'}))
    monkeypatch.setattr(pilot,'environment',lambda mode,root:stack)
    snapshot={'seed':21,'state_hash':'a'}
    assert v.make_environment('stack',tmp_path/'stack',21,snapshot,api_budget=19) is stack
    assert calls==[{'seed':21,'options':{'snapshot':snapshot}}]
    assert stack.cfg.feedback_api_budget==19
    assert stack.capture_snapshot()=={'state_hash':'a'}
    assert callable(stack.evaluation_state)


def test_tcp_uses_live_robot_hand_fk_and_stack_convention():
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    base_q=np.array([1.,0,0,0]);hand_q=np.array([0.,0.,1.,0.])
    data=SimpleNamespace(xquat=np.array([hand_q]),xpos=np.array([[.6,.2,1.2]]))
    low=SimpleNamespace(robosuite_env=SimpleNamespace(sim=SimpleNamespace(data=data,model=SimpleNamespace(body_name2id=lambda name:0))),base_link_wxyz_xyz=np.r_[base_q,0,0,.8],gripper_metric_length=.04)
    raw={'robot0_gripper_qpos':[.02]}
    expected=np.r_[.6,.2,1.2-.8-.107,hand_q,.5]
    assert v._robot_tcp(low,raw)==pytest.approx(expected)


def diagnostic_low():
    obj=lambda name:SimpleNamespace(name=name,root_body=name+'_body',size=np.array([.02,.02,.02]),contact_geoms=[name+'_geom'])
    red,green=obj('cubeA'),obj('cubeB')
    body_ids={'base':0,'robot0_right_hand':1,'eef':2,'cubeA_body':3,'cubeB_body':4}
    geom_names=['red_geom_unused','cubeA_geom','cubeB_geom','left_pad','right_pad']
    positions=np.array([[0,0,.8],[.5,.1,1.2],[.5,.1,1.103],[.5,.1,.83],[.55,.1,.9]])
    contacts=[SimpleNamespace(geom1=1,geom2=3,dist=-.001,pos=np.zeros(3)),
              SimpleNamespace(geom1=3,geom2=2,dist=-.001,pos=np.zeros(3)),
              SimpleNamespace(geom1=2,geom2=4,dist=-.001,pos=np.zeros(3))]
    data=SimpleNamespace(time=1.2,xpos=positions,xquat=np.tile([1,0,0,0],(5,1)),xmat=np.tile(np.eye(3).reshape(9),(5,1)),
                         contact=contacts,ncon=len(contacts),get_body_xvelp=lambda _:np.zeros(3),get_body_xvelr=lambda _:np.zeros(3))
    model=SimpleNamespace(body_name2id=body_ids.__getitem__,geom_id2name=geom_names.__getitem__)
    gripper=SimpleNamespace(important_geoms={'left_fingerpad':['left_pad'],'right_fingerpad':['right_pad']})
    env=SimpleNamespace(sim=SimpleNamespace(data=data,model=model),cubeA=red,cubeB=green,cube=red,
                         robots=[SimpleNamespace(arms=['right'],gripper={'right':gripper})],
                         _get_observations=lambda **kw:{'robot0_gripper_qpos':np.array([.02,-.02])})
    return SimpleNamespace(robosuite_env=env,base_link_idx=0,gripper_link_idx=2)


def test_private_geometry_has_real_names_sizes_and_calibrated_base():
    low=diagnostic_low();geometry=v._evaluation_geometry(low,'restack')
    assert geometry['objects']['red']['object_name']=='cubeA'
    assert geometry['objects']['green']['object_name']=='cubeB'
    assert geometry['objects']['green']['half_size']==pytest.approx([.02]*3)
    assert geometry['base_world']['position']==pytest.approx([0,0,.8])
    assert geometry['tcp_offset_in_hand_m']==pytest.approx([0,0,.107])
    assert set(v._evaluation_geometry(low,'lift')['objects'])=={'red'}


def test_private_canonical_evaluation_separates_green_and_red_bilateral_contact():
    low=diagnostic_low();state=v._canonical_evaluation(low,'restack')
    assert state['objects']['red']['contact_left_fingerpad'] is True
    assert state['objects']['red']['contact_right_fingerpad'] is False
    assert state['objects']['red']['bilateral_contact'] is False
    assert state['objects']['green']['bilateral_contact'] is True
    assert state['tcp_world']['position']==pytest.approx([.5,.1,1.307])
    assert state['tcp_world']['quaternion_wxyz']==pytest.approx([1,0,0,0])
    assert state['gripper_opening_m']==pytest.approx(.04)
    # Explicit evaluator output never augments deployment observations.
    clean=v.public_sensors(raw_observation(),['robot0_robotview'])
    assert 'canonical' not in clean and 'objects' not in clean


def test_private_table_geometry_is_explicitly_unknown_without_model_metadata():
    table=v._table_geometry(diagnostic_low())
    assert table['table_top_world_z'] is None
    assert table['table_center_world'] is None
    assert table['table_half_extent_xy'] is None


def test_private_table_geometry_uses_actual_axis_aligned_collision_box():
    low=diagnostic_low();sim=low.robosuite_env.sim
    sim.model.geom_name2id=lambda name:0
    sim.model.geom_type=np.array([6]);sim.model.geom_size=np.array([[.4,.3,.025]])
    sim.data.geom_xpos=np.array([[.1,.2,.775]]);sim.data.geom_xmat=np.eye(3).reshape(1,9)
    table=v._table_geometry(low)
    assert table['table_top_world_z']==pytest.approx(.8)
    assert table['table_center_world']==pytest.approx([.1,.2,.775])
    assert table['table_half_extent_xy']==pytest.approx([.4,.3])
    angle=np.pi/4;sim.data.geom_xmat=np.array([[np.cos(angle),-np.sin(angle),0,np.sin(angle),np.cos(angle),0,0,0,1]])
    assert v._table_geometry(low)['table_top_world_z'] is None


def test_stack_public_observation_preserves_both_encoders_without_hidden_state():
    raw=raw_observation();raw['robot0_gripper_qpos']=np.array([.031,-.017])
    low=SimpleNamespace(render_camera_names=['robot0_robotview'],get_observation=lambda:raw)
    env=SimpleNamespace(low_level_env=low,_full_prompt=[{'role':'user','content':'same task'}])
    obs=v.stack_public_observation(env)
    assert obs['robot_gripper_joint_pos']==pytest.approx([.031,-.017])
    assert 'cube_poses' not in obs and 'sim' not in obs and 'raw_success' not in obs
    assert obs['full_prompt']==env._full_prompt
    obs['robot_gripper_joint_pos'][0]=9
    assert raw['robot0_gripper_qpos'][0]==.031


def test_stack_adapter_installs_public_wrapper_and_missing_finger_is_not_inferred(monkeypatch,tmp_path):
    from capx.experiments import pilot
    raw=raw_observation();raw['robot0_gripper_qpos']=np.array([.04,-.02])
    low=SimpleNamespace(render_camera_names=['robot0_robotview'],get_observation=lambda:raw,capture_snapshot=lambda:{'seed':21})
    env=SimpleNamespace(cfg=SimpleNamespace(),low_level_env=low,_full_prompt=[],reset=lambda **kw:(None,{'initial_state_hash':'a'}),_get_observation=lambda:{'legacy':True})
    monkeypatch.setattr(pilot,'environment',lambda mode,root:env)
    v.make_environment('stack',tmp_path/'scene',21)
    assert env._get_observation()['robot_gripper_joint_pos']==pytest.approx([.04,-.02])
    del raw['robot0_gripper_qpos']
    assert 'robot_gripper_joint_pos' not in env._get_observation()
