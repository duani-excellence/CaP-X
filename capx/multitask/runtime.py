"""Fresh-process deterministic starts and independent task-specific evaluation.
No legacy SimpleExecutor, dense reward, or actor access to simulator state.
"""
import hashlib,importlib,json,time
from pathlib import Path
import numpy as np
from capx.envs.reproducibility import json_value,state_hash
from capx.execution.sandbox import SandboxProcess
from capx.multitask.catalog import TASKS


def qualifies(task,s):
    if not s['raw_success'] or not s['still']:return False
    if task=='lift':return bool(s['grasp'][0])
    if task=='two_arm_lift':return all(s['grasp'][:2]) and len(s['grasp'])>=2
    if task in ('restack','nut_assembly'):return s['released']
    if task in ('wipe','handover'):return True
    raise ValueError('No criterion for task '+task)


def load(path,prefix):
    module,name=path.rsplit('.',1)
    return getattr(importlib.import_module(prefix+module),name)


class Runtime:
    def __init__(self,task,seed,root,render=True):
        self.task=task;self.seed=seed;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.trace=self.root/'trace.jsonl';self.since=None;self.stable=False;self.motion=0;self.calls=0;self.active=False
        self.low=None;self.actor=None
        np.random.seed(seed)
        spec=TASKS[task]
        self.task_module=importlib.import_module('capx.envs.tasks.franka.'+spec['module'])
        self.low=load(spec['low'],'capx.envs.simulators.')(seed=seed,privileged=True,enable_render=render,viser_debug=False)
        env=self.low.robosuite_env
        env.seed=seed;env.rng=np.random.default_rng(seed)
        def bind(sampler):
            if sampler is None:return
            sampler.rng=env.rng
            for c in getattr(sampler,'samplers',{}).values():bind(c)
        bind(getattr(env,'placement_initializer',None))
        np.random.seed(seed);self.low.reset(seed=seed)
        if hasattr(self.low,'_current_joints'):
            robot=env.robots[0]
            self.low._current_joints=env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
        # Replay the same complete reset/settle/controller history in a fresh process.
        since=None
        for _ in range(100):
            self.low._step_once();s=self.state()
            since=(s['sim_time'] if since is None else since) if s['still'] else None
            if since is not None and s['sim_time']-since>=.5:break
        else:raise RuntimeError('Initial objects failed to settle')
        if self.state()['raw_success']:raise ValueError('Task already successful at reset')
        self.low._sim_step_count=0
        self.initial=self.snapshot()
        self.api=load(spec['api'],'capx.integrations.franka.')(self.low)
        self.functions=self.api.functions()
        if task in ('lift','restack'):
            self.functions['get_object_pose']=self.object_pose
            self.functions['sample_grasp_pose']=self.grasp_pose
        # All legacy motion paths eventually step this simulator, including dual-arm ones.
        original=env.step
        def stepped(*a,**kw):
            value=original(*a,**kw)
            if self.active:
                state=self.state();good=qualifies(self.task,state)
                self.since=(state['sim_time'] if self.since is None else self.since) if good else None
                self.stable=self.since is not None and state['sim_time']-self.since>=.5
                self.log('physics',state=state)
            return value
        env.step=stepped
        if not hasattr(self.low,'_frame_timestamps'):
            self.low._frame_timestamps=[]
            original_frame=self.low._record_frame
            def frame():
                before=len(self.low._frame_buffer);original_frame()
                if len(self.low._frame_buffer)>before:self.low._frame_timestamps.append(float(env.sim.data.time))
            self.low._record_frame=frame
        self.actor=SandboxProcess(self.functions,callback=self.call)
        self.log('reset',task=task,seed=seed,snapshot=self.initial)

    def snapshot(self):
        env=self.low.robosuite_env;sim=env.sim
        physics=dict(state=sim.get_state().flatten(),ctrl=sim.data.ctrl.copy(),warmstart=sim.data.qacc_warmstart.copy(),
                     rng=env.rng.bit_generator.state,wiped=list(getattr(env,'wiped_markers',[])))
        return json.loads(json.dumps(dict(version=1,task=self.task,seed=self.seed,model_hash=hashlib.sha256(env.model.get_xml().encode()).hexdigest(),physics=physics,state_hash=state_hash(physics)),default=json_value))

    def state(self):
        env=self.low.robosuite_env;d=env.sim.data;m=env.sim.model
        objects={}
        for obj in env.model.mujoco_objects:
            bid=m.body_name2id(obj.root_body)
            objects[obj.name]=dict(position=d.xpos[bid].copy(),quaternion_wxyz=d.xquat[bid].copy(),speed=float(np.linalg.norm(d.get_body_xvelp(obj.root_body))),angular_speed=float(np.linalg.norm(d.get_body_xvelr(obj.root_body))))
        obs=env._get_observations(force_update=True);grasps=[];openings=[];ees=[]
        for i,robot in enumerate(env.robots):
            arm=robot.arms[0];gripper=robot.gripper[arm]
            openings.append(float(np.abs(obs.get(f'robot{i}_gripper_qpos',[])).sum()))
            site=robot.eef_site_id[arm];ees.append(d.site_xpos[site].copy())
            if self.task=='lift':geoms=env.cube.contact_geoms
            elif self.task=='two_arm_lift':geoms=env.pot.handle0_geoms if i==0 else env.pot.handle1_geoms
            elif self.task=='restack':geoms=env.cubeA.contact_geoms
            elif self.task=='nut_assembly':geoms=env.nuts[0].contact_geoms
            elif self.task=='handover':geoms=env.hammer.contact_geoms
            else:geoms=[]
            grasps.append(bool(env._check_grasp(gripper,geoms)) if geoms else False)
        contacts=[dict(geom1=m.geom_id2name(c.geom1),geom2=m.geom_id2name(c.geom2),distance=float(c.dist)) for c in d.contact[:d.ncon]]
        return dict(sim_time=float(d.time),objects=objects,ee=ees,gripper_openings=openings,grasp=grasps,
                    contacts=contacts,raw_success=bool(env._check_success()),released=all(x>.05 for x in openings) and not any(grasps),
                    still=all(x['speed']<.02 and x['angular_speed']<.2 for x in objects.values()),frame='world')

    def object_pose(self,object_name,return_bbox_extent=False):
        """Object pose in robot base: XYZ metres, WXYZ quaternion, full base-axis bbox extent. Names: red cube / green cube."""
        from scipy.spatial.transform import Rotation
        env=self.low.robosuite_env
        if 'red' in object_name:obj=env.cube if self.task=='lift' else env.cubeA
        elif 'green' in object_name and self.task=='restack':obj=env.cubeB
        else:raise ValueError('Unknown object')
        d=env.sim.data;m=env.sim.model;bid=m.body_name2id(obj.root_body);base=self.low.base_link_idx
        R=d.xmat[base].reshape(3,3).T;rotation=R@d.xmat[bid].reshape(3,3)
        quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        return R@(d.xpos[bid]-d.xpos[base]),quat,np.abs(rotation)@(2*np.asarray(obj.size)) if return_bbox_extent else None

    def grasp_pose(self,object_name):
        """Top-down cube grasp in robot base, XYZ metres and WXYZ quaternion."""
        return self.object_pose(object_name)[0],np.array([0.,0.,1.,0.])

    def log(self,kind,**kw):
        with self.trace.open('a') as f:f.write(json.dumps(dict(kind=kind,**kw),default=json_value)+'\n')

    def call(self,name,args,kwargs):
        before=self.state();self.calls+=1
        if name.startswith(('goto','open_gripper','close_gripper')):self.motion+=1
        error=None;response=None
        try:response=self.functions[name](*args,**kwargs);return response
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            if not isinstance(exc,(ValueError,TypeError,TimeoutError)):raise ConnectionError(error) from exc
            raise
        finally:self.log('api',name=name,args=args,kwargs=kwargs,response=response,error=error,before=before,after=self.state())

    def execute(self,code,timeout=90.):
        if getattr(self,'invalidated',False):raise RuntimeError('Fresh environment required after worker timeout/error')
        self.active=True;self.log('code',code=code)
        start_motion=self.motion
        result=self.actor.run(code,timeout=timeout)
        self.invalidated=result['status'] in ('timeout','system_error')
        if result['ok'] and self.motion:
            for _ in range(15):self.low._step_once()
        state=self.state();truncated=self.low._sim_step_count>=self.low.max_steps
        success=bool(result['ok'] and self.motion>start_motion and self.stable and not truncated)
        info=dict(physical_success=self.stable,raw_success=state['raw_success'],sandbox_rc=0 if result['ok'] else 1,
                  execution_status=result['status'],motion_calls=self.motion,api_calls=self.calls,stdout=result['stdout'],stderr=result['stderr'])
        self.log('result',reward=float(success),**info)
        return dict(status=('success' if success else 'task_failure') if result['status'] not in ('timeout','system_error') else result['status'],reward=float(success),truncated=truncated,info=info)

    def close(self):
        if self.actor:self.actor.close()
        if self.low:self.low.close()
