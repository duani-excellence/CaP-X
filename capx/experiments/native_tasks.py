"""One native Robosuite boundary for parameterized task benchmarks.

This is environment plumbing, not a Graph Executive policy. Native success is
always read by the private evaluator. Actor-visible observations are restricted
to images/calibration and encoders/FK. No reward or object qpos is exposed.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from capx.envs.reproducibility import json_value, state_hash


def actor_functions(functions,mode):
    if mode=='oracle':return dict(functions)
    if mode=='vision':return {k:v for k,v in functions.items() if k in ('get_object_pose','sample_grasp_pose','observe_robot')}
    raise ValueError('Unknown mode')


def physical_reward(program_ok,new_motion,raw_success):
    return float(program_ok and new_motion>0 and raw_success)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,default=json_value,indent=2,allow_nan=False)


def bind_rng(env,seed):
    env.seed=seed;env.rng=np.random.default_rng(seed)
    def bind(s):
        if s is None:return
        s.rng=env.rng
        for child in getattr(s,'samplers',{}).values():bind(child)
    bind(getattr(env,'placement_initializer',None))


def create_native(spec,seed,render=True):
    import robosuite
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
    np.random.seed(seed)
    kwargs=copy.deepcopy(spec.get('environment_kwargs',{}))
    robots=spec.get('robots',['Panda'])
    controller=load_composite_controller_config(controller=spec['controller'])
    env=robosuite.make(spec['environment'],robots=robots,controller_configs=controller,
        has_renderer=False,has_offscreen_renderer=render,use_camera_obs=render,
        camera_names=spec.get('cameras',['robot0_robotview']) if render else [],
        camera_depths=True,camera_heights=512,camera_widths=512,
        reward_shaping=False,hard_reset=False,horizon=spec.get('horizon',6000),
        seed=seed,**kwargs)
    bind_rng(env,seed);np.random.seed(seed);env.reset()
    return env


def physical_state(env):
    """Private, task-agnostic native evaluator; never returned by public API."""
    d,m=env.sim.data,env.sim.model
    success=bool(env._check_success())
    objects={}
    for obj in env.model.mujoco_objects:
        bid=m.body_name2id(obj.root_body)
        objects[obj.name]=dict(position=d.xpos[bid].copy(),quaternion=d.xquat[bid].copy(),
            linear_velocity=d.get_body_xvelp(obj.root_body).copy(),
            angular_velocity=d.get_body_xvelr(obj.root_body).copy())
    progress={}
    # These are native evaluator fields, not alternative success thresholds.
    for name in ('objects_in_bins','objects_on_pegs','wiped_markers'):
        if hasattr(env,name):progress[name]=copy.deepcopy(getattr(env,name))
    return dict(raw_success=success,sim_time=float(d.time),objects=objects,
                native_progress=progress,contacts=[dict(geom1=m.geom_id2name(c.geom1),
                    geom2=m.geom_id2name(c.geom2),distance=float(c.dist)) for c in d.contact[:d.ncon]])


def snapshot(env,seed,spec):
    physics=dict(state=env.sim.get_state().flatten(),ctrl=env.sim.data.ctrl.copy(),
                 warmstart=env.sim.data.qacc_warmstart.copy(),rng=env.rng.bit_generator.state)
    return json.loads(json.dumps(dict(schema='capx.native_initial.v1',seed=seed,spec=spec,
        model_hash=hashlib.sha256(env.model.get_xml().encode()).hexdigest(),
        state_hash=state_hash(physics),physics=physics),default=json_value))


def make_low(spec,seed,render=True):
    """Shared single-arm joint-control wrapper, using existing camera/control code."""
    from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
    class NativeLow(RobosuiteBaseEnv):
        def __init__(self):
            super().__init__(controller_cfg=spec['controller'],max_steps=spec.get('horizon',6000),seed=seed)
            self.robosuite_env=create_native(spec,seed,render)
            if len(self.robosuite_env.robots)!=1:raise ValueError('Single-arm capability only; dual-arm needs a qualified coordinated controller')
            self.render_camera_names=spec.get('cameras',['robot0_robotview']) if render else []
            self.save_camera_name=spec.get('video_camera',self.render_camera_names[0]) if render else 'robot0_robotview'
            self._init_robot_links()
            robot=self.robosuite_env.robots[0]
            self._current_joints=self.robosuite_env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
            self._gripper_fraction=1.
            self.home_joint_position=self._current_joints.copy()
            self.viser_debug=False
            for _ in range(30):self._step_once()
            self._sim_step_count=0
        def get_observation(self):
            raw=self.robosuite_env._get_observations(force_update=True)
            self._process_camera_observations(raw);self._compute_gripper_obs(raw)
            return raw
        def _update_viser_server(self):pass
        def compute_reward(self):return float(self.robosuite_env._check_success())
        def task_completed(self):return bool(self.robosuite_env._check_success())
    return NativeLow()


class NativeBoundary:
    """Reuses SandboxProcess and the existing public Vision/control API.

    Motion/observations and independent native physics traces are recorded around
    every RPC. Oracle-only functions are constructed only in explicit oracle
    mode; no fallback to them exists in vision mode.
    """
    def __init__(self,spec,seed,root,mode='vision',expected_snapshot=None,sensor_config=None):
        from capx.experiments.vision_tasks import PublicLowLevel,public_robot_calibration
        from capx.integrations.franka.control import FrankaControlApi
        from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
        from capx.execution.control_frames import transform_target
        from capx.execution.sandbox import SandboxProcess
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.spec=spec;self.seed=seed;self.mode=mode;self.calls=0;self.motion=0
        self.started=time.monotonic();self.low=make_low(spec,seed)
        self.low_level_env=self.low
        self.public=PublicLowLevel(self.low,seed)
        self.api=FrankaControlApi(self.public);self._apis={'FrankaControlApi':self.api}
        self.initial=snapshot(self.low.robosuite_env,seed,spec)
        save(self.root/'snapshot.json',self.initial)
        if expected_snapshot is not None and self.initial!=expected_snapshot:
            self.low.close();raise ValueError('Initial snapshot replay mismatch')
        self.calibration=public_robot_calibration(self)
        save(self.root/'calibration.json',self.calibration)
        self.force_sensor=None;self.force_enabled=False;self.force_guard=False
        sensor_config=sensor_config or spec.get('force_sensor')
        if sensor_config:
            from capx.integrations.sensors.wrist_ft import MujocoWristFT
            def record_force(row):
                with (self.root/'force.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.force_sensor=MujocoWristFT(self.low,sensor_config,record_force)
            save(self.root/'force-calibration.json',self.force_sensor.calibration)
            def force_tick():
                row=self.force_sensor.sample()
                # Emergency boundary stop; no force-only movement/hold permission.
                if self.force_enabled and self.force_guard:
                    f=np.linalg.norm(row['external_base'][:3]);t=np.linalg.norm(row['external_base'][3:])
                    if not row['valid'] or f>sensor_config.get('hard_force_n',40) or t>sensor_config.get('hard_torque_nm',4):
                        from capx.execution.skill_contracts import SkillBoundaryStop
                        self.low._current_joints=self.low.robosuite_env.sim.data.qpos[self.low.robosuite_env.robots[0]._ref_joint_pos_indexes].copy()
                        self.log('force_guard_stop',sample=row)
                        raise SkillBoundaryStop('Wrist force guard stopped motion',{})
            self.low._physics_observer=force_tick
            self.force_sensor.sample()
        urdf=next(x['path'] for x in self.calibration['sources'] if x['role']=='ik_urdf')
        self.api.ik_solve_fn=BoundedIK(self.api.ik_solve_fn,urdf_path=urdf,
            budget=IKSolveBudget(spec.get('api_budget',256)*8),
            observe_joints=lambda:self.public.get_observation()['robot_joint_pos'],
            emit=lambda row:self.log('ik',**row))
        self.functions=self.api.functions()
        self.functions['goto_pose']=self.goto_pose
        self.functions['observe_robot']=lambda:self.public.get_observation()['robot_cartesian_pos']
        if mode=='oracle':
            self.functions.update(oracle_object=self.oracle_object,oracle_goal=self.oracle_goal,oracle_region=self.oracle_region,
                                  oracle_bindings=lambda:list(enumerate(self.spec['objects'])),
                                  oracle_insertion_bindings=lambda:list(range(len(self.spec.get('insertion_bindings',[])))),
                                  oracle_insertion=lambda index:self.oracle_insertions()[index])
        elif mode=='vision':
            # A registered raw controller is NOT a qualified non-cube Skill.
            # Until capability-specific admission is installed, this boundary
            # exposes perception only and cannot bypass V23 safety via raw RPC.
            self.functions=actor_functions(self.functions,mode)
        else:raise ValueError('Unknown mode')
        self.actor=SandboxProcess(self.functions,callback=self.call)
        self.low.enable_video_capture()
        self.log('reset',mode=mode,private_evaluation=physical_state(self.low.robosuite_env))

    def log(self,kind,**row):
        with (self.root/'trace.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,**row),default=json_value)+'\n')

    def capture(self):
        obs=self.public.get_observation();index=getattr(self,'captures',0);self.captures=index+1
        if self.force_enabled and self.force_sensor is not None:
            obs['wrist_ft']=copy.deepcopy(self.force_sensor.sample())
        directory=self.root/'observations';directory.mkdir(exist_ok=True)
        files={}
        for camera in self.low.render_camera_names:
            data=obs[camera];path=directory/f'{index:05d}-{camera}.npz'
            np.savez_compressed(path,rgb=data['images']['rgb'],depth=data['images']['depth'],
                intrinsics=data['intrinsics'],pose=data['pose'],pose_mat=data['pose_mat'],
                tcp=obs['robot_cartesian_pos'],joints=obs['robot_joint_pos'],gripper=obs['robot_gripper_joint_pos'])
            files[camera]=str(path)
        self.log('public_capture',capture=index,files=files,sim_time=float(self.low.robosuite_env.sim.data.time))
        sink=getattr(self,'public_capture_sink',None)
        if sink:sink(dict(capture=index,files=files,robot={k:obs[k] for k in ('robot_cartesian_pos','robot_joint_pos','robot_gripper_joint_pos','wrist_ft') if k in obs}))
        return obs

    def goto_pose(self,position,quaternion_wxyz,z_approach=0.):
        from capx.execution.control_frames import transform_target
        pos,quat=transform_target(self.calibration,position,quaternion_wxyz)
        return self.api.goto_pose(pos,quat,z_approach=z_approach)

    def call(self,name,args,kwargs):
        return self._invoke(name,args,kwargs,self.functions[name])

    def trusted_motion(self,name,*args,**kwargs):
        """Parent skill registry only; absent from actor RPC capabilities."""
        functions={'goto_pose':self.goto_pose,'open_gripper':self.api.open_gripper,
                   'close_gripper':self.api.close_gripper,'wait':self.wait_control}
        if name not in functions:raise ValueError('Unsupported trusted primitive')
        self.force_guard=bool(self.force_enabled and name=='goto_pose')
        try:return self._invoke(name,args,kwargs,functions[name])
        finally:self.force_guard=False

    def wait_control(self,seconds):
        if not np.isfinite(seconds) or not 0<seconds<=1:raise ValueError('Bounded hold duration required')
        for _ in range(int(np.ceil(seconds*self.low.robosuite_env.control_freq))):self.low._step_once()

    def _invoke(self,name,args,kwargs,function):
        from capx.execution.sandbox import ExecutionInterrupted
        if self.calls>=self.spec.get('api_budget',256):raise ExecutionInterrupted('Native episode API budget exhausted')
        if time.monotonic()-self.started>self.spec.get('episode_seconds',900):raise ExecutionInterrupted('Native episode deadline')
        self.calls+=1
        if name in ('goto_pose','open_gripper','close_gripper','home_pose'):self.motion+=1
        before=physical_state(self.low.robosuite_env);started=time.monotonic();error=None;response=None
        try:
            observer=getattr(self,'action_observer',None)
            response=observer(name,args,kwargs,function) if observer else function(*args,**kwargs)
            return response
        except Exception as exc:error=str(exc);raise
        finally:self.log('api',name=name,args=args,kwargs=kwargs,response=response,error=error,
                         seconds=time.monotonic()-started,before=before,after=physical_state(self.low.robosuite_env))

    def oracle_object(self,name):
        """PRIVILEGED ORACLE ONLY. Native object center/orientation in robot base."""
        from scipy.spatial.transform import Rotation
        e=self.low.robosuite_env;d=e.sim.data;m=e.sim.model
        obj=next(o for o in e.model.mujoco_objects if o.name==name)
        bid=m.body_name2id(obj.root_body);base=self.low.base_link_idx
        R=d.xmat[base].reshape(3,3).T
        pos=R@(d.xpos[bid]-d.xpos[base]);rot=R@d.xmat[bid].reshape(3,3)
        points=[]
        for name in obj.contact_geoms:
            gid=m.geom_name2id(name);mesh=int(m.geom_dataid[gid])
            if int(m.geom_type[gid])==7 and mesh>=0:
                start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
                vertices=np.asarray(m.mesh_vert[start:start+count])
            elif int(m.geom_type[gid])==6:
                import itertools
                vertices=np.asarray(list(itertools.product([-1,1],repeat=3)))*m.geom_size[gid]
            else:continue
            world=vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
            points.extend((world-d.xpos[bid])@d.xmat[bid].reshape(3,3))
        if not points:raise ValueError('Oracle shape has no supported collision geometry')
        lo,hi=np.min(points,axis=0),np.max(points,axis=0);extent=hi-lo
        center=(lo+hi)/2
        # Parallel jaws close along tool Y: align that axis with the narrow
        # horizontal shape dimension, independent of object/task identity.
        basis=np.array([[0,1,0],[1,0,0],[0,0,-1]]) if extent[0]<extent[1] else np.diag([1,-1,-1])
        grasp_rot=rot@basis
        depth=min(float(self.spec.get('grasp_depth_m',.02)),float(extent[2]/2))
        grasp=center+np.array([0.,0.,extent[2]/2-depth])
        return dict(position=pos+rot@grasp,root_position=pos,
                    quaternion=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]],
                    grasp_quaternion=Rotation.from_matrix(grasp_rot).as_quat()[[3,0,1,2]],
                    bottom_offset=np.array([0,0,-(extent[2]-depth)]),horizontal_radius=float(np.linalg.norm(extent[:2])/2),
                    geometry_extent=extent,grasp_policy='top_band_parallel_jaw_with_palm_clearance',grasp_depth_m=depth)

    def oracle_goal(self,index):
        """PRIVILEGED ORACLE ONLY. Native receptacle center, no online exposure."""
        e=self.low.robosuite_env;d=e.sim.data
        if not hasattr(e,'target_bin_placements'):raise ValueError('Native bin geometry capability unavailable')
        return d.xmat[self.low.base_link_idx].reshape(3,3).T@(e.target_bin_placements[index]-d.xpos[self.low.base_link_idx])

    def oracle_region(self,index):
        """PRIVILEGED ORACLE ONLY. Region bounds for unconstrained-yaw placement."""
        e=self.low.robosuite_env
        return dict(center=self.oracle_goal(index),half_extent_xy=np.asarray(e.bin_size[:2])/4)

    def oracle_insertions(self):
        """PRIVILEGED reference feature binding, never available in Vision mode.

        Object handle sites and fixture body names come from environment metadata.
        The same rigid relative transform is used for every compatible peg pair.
        """
        from scipy.spatial.transform import Rotation
        e=self.low.robosuite_env;d=e.sim.data;m=e.sim.model;base=self.low.base_link_idx
        R=d.xmat[base].reshape(3,3).T;origin=d.xpos[base];rows=[]
        for binding in self.spec.get('insertion_bindings',[]):
            obj=next(o for o in e.model.mujoco_objects if o.name==binding['object'])
            bid=m.body_name2id(obj.root_body);sid=m.site_name2id(obj.important_sites['handle'])
            peg=m.body_name2id(binding['fixture_body']);gid=int(m.body_geomadr[peg])
            shape=int(m.geom_type[gid]);size=m.geom_size[gid]
            half_z=float(size[2] if shape==6 else size[1] if shape==5 else np.nan)
            if not np.isfinite(half_z):raise ValueError('Unsupported reference peg geometry')
            center=R@(d.xpos[bid]-origin);handle=R@(d.site_xpos[sid]-origin)
            grasp_rotation=R@d.xmat[bid].reshape(3,3)@np.diag([-1.,1.,-1.])
            target_rotation=R@d.xmat[peg].reshape(3,3)@np.diag([1.,-1.,-1.])
            peg_top=R@(d.xpos[peg]+d.xmat[peg].reshape(3,3)@np.array([0.,0.,half_z])-origin)
            relative=grasp_rotation.T@(handle-center)
            rows.append(dict(object=binding['object'],handle=handle,
                grasp_quaternion=Rotation.from_matrix(grasp_rotation).as_quat()[[3,0,1,2]],
                target=peg_top+target_rotation@relative,
                target_quaternion=Rotation.from_matrix(target_rotation).as_quat()[[3,0,1,2]]))
        return rows

    def execute(self,code):
        save(self.root/'program.json',dict(code=code,origin='reference_oracle' if self.mode=='oracle' else 'vision_reference_NOT_LLM'))
        before_motion=self.motion
        result=self.actor.run(code,timeout=self.spec.get('episode_seconds',900))
        if result['ok'] and self.motion>before_motion:
            for _ in range(15):self.low._step_once()
        state=physical_state(self.low.robosuite_env)
        system_error=result['status'] in ('timeout','system_error')
        truncated=self.low._sim_step_count>=self.low.max_steps
        result.update(raw_success=state['raw_success'],reward=None if system_error else physical_reward(result['ok'] and not truncated,self.motion-before_motion,state['raw_success']),
            system_error=system_error,truncated=truncated,
            physics=state,actual_api=self.calls,seconds=time.monotonic()-self.started,new_llm_requests=0)
        save(self.root/'result.json',result)
        return result

    def close(self):
        if hasattr(self,'actor'):self.actor.close()
        if hasattr(self,'low'):
            frames=self.low.get_video_frames()
            if frames:
                import imageio.v2 as imageio
                imageio.mimsave(self.root/'video.mp4',frames,fps=10)
                save(self.root/'video-timestamps.json',self.low._frame_timestamps)
            self.low.close()
