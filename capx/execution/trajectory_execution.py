"""Parent-only waypoint execution with the existing native controller.

No new IK/global planner. Shared opt-in backend for mature and open motions.
Unknown geometry remains a stop, and no collision status is model-authored.
"""
import time
import numpy as np


def retained_solid_mask(mesh,points):
    """Containment in closed configured solids, not a surface-distance proxy.

    Open/non-watertight mesh parts prove no interior. Unknown stays unknown.
    """
    points=np.asarray(points,float);mask=np.zeros(len(points),bool)
    for part in mesh.split(only_watertight=True,repair=False):
        bounds=part.bounds
        ids=np.flatnonzero(~mask & np.all(points>bounds[0],axis=1) & np.all(points<bounds[1],axis=1))
        if len(ids):mask[ids]=part.contains(points[ids])
    return mask


def public_obstacles(skills,geometry,calibration):
    """Conservative voxel spheres from CURRENT RGB-D, never unknown-space fill."""
    from .region_geometry import deproject
    from .robot_self_geometry import classify_mesh_depth
    s=skills;frame=s.scene.obs;cloud=[];sources=[]
    mesh,meta=geometry.mesh(frame['robot_joint_pos'][:7],finger_positions_m=np.abs(frame['robot_gripper_joint_pos']),observed_tcp=frame['robot_cartesian_pos'])
    for name,cam in frame.items():
        if not isinstance(cam,dict) or 'images' not in cam:continue
        xyz=deproject(cam)[::8,::8].reshape(-1,3);xyz=xyz[np.isfinite(xyz).all(1)]
        if not len(xyz):continue
        identity=classify_mesh_depth(xyz,cam['pose_mat'][:3,3],mesh,tolerance_m=.002)
        cloud.append(xyz[~identity['robot_mask']]);sources.append(dict(camera=name,points=len(xyz),robot_points=int(identity['robot_mask'].sum())))
    if not cloud:return [],dict(status='UNKNOWN',reason='no public depth')
    xyz=np.concatenate(cloud);t=np.array(calibration['T_ik_base_public_base'])
    xyz=xyz@t[:3,:3].T+t[:3,3]
    cell=.04;centers=(np.unique(np.floor(xyz/cell).astype(int),axis=0)+.5)*cell
    # All sampled occupied voxels retained. Partial depth is NOT full world knowledge.
    return [dict(type='sphere_cloud',centers=centers.tolist(),radius=float(np.sqrt(3)*cell/2+.004))],dict(
        status='PARTIAL_CURRENT_OBSTACLES',source=s.scene.source,calibration=calibration['calibration_hash'],
        sources=sources,obstacle_count=len(centers),voxel_size_m=cell,complete_public_world=False,attachment_included=False)


def execute_waypoints(path,read,admit,dispatch,emit,*,deadline=None):
    path=np.asarray(path,float)
    if path.ndim!=2 or path.shape[1]!=7 or len(path)<2 or not np.isfinite(path).all():raise ValueError('Invalid waypoint path')
    before=read()
    if np.max(abs(path[0]-before['joints']))>.002:raise ValueError('Trajectory start changed')
    for i,q in enumerate(path[1:],1):
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Shared trajectory budget exhausted')
        before=read();check=admit(np.asarray(before['joints']),q)
        if check['status']!='PASS':raise ValueError('Trajectory admission: '+str(check))
        emit('trajectory_waypoint_dispatch',index=i,target=q.tolist(),before=before,admission=check)
        dispatch(q)
        after=read();error=float(np.max(abs(q-after['joints'])))
        emit('trajectory_waypoint_observed',index=i,after=after,joint_error_rad=error)
        if error>.002:raise ValueError('Trajectory tracking failed')
        if before['source']==after['source']:raise ValueError('Trajectory verification needs fresh encoders')
    return dict(status='EFFECT_VERIFIED',scope='waypoint_joint_tracking_only',waypoints=len(path)-1)


class SharedTrajectoryMotion:
    def __init__(self,boundary,skills,geometry):
        self.b=boundary;self.s=skills;self.geometry=geometry;self.plans=[]

    def check(self,path):
        """Existing dense robot/attachment public-ray check, no truth access."""
        from scipy.spatial import cKDTree
        from scipy.spatial.transform import Rotation
        from .open_motion import tool_volume,observed_free
        s=self.s;frame=s.scene.obs;fingers=np.abs(frame['robot_gripper_joint_pos'])
        mesh,meta=self.geometry.mesh(frame['robot_joint_pos'][:7],finger_positions_m=fingers,observed_tcp=frame['robot_cartesian_pos'])
        original_mesh=mesh.copy()
        tree=cKDTree(np.r_[mesh.vertices,mesh.triangles_center,tool_volume(s,frame['robot_cartesian_pos'][:7])])
        checks=[]
        for start,end in zip(path[:-1],path[1:]):
            for alpha in np.linspace(0,1,max(2,int(np.ceil(np.max(abs(end[:7]-start[:7]))/.01))+1))[1:]:
                if len(checks)>=256:return dict(status='UNKNOWN',reason='public_sweep_sample_budget',checks=checks)
                mesh,meta=self.geometry.mesh((start*(1-alpha)+end*alpha)[:7],finger_positions_m=fingers)
                tcp=np.asarray(meta['public_fk_tcp']);q=np.roll(Rotation.from_matrix(tcp[:3,:3]).as_quat(),1)
                points=np.r_[mesh.vertices,mesh.triangles_center,tool_volume(s,np.r_[tcp[:3,3],q])];d,_=tree.query(points)
                changed=points[d>.004]
                interior=retained_solid_mask(original_mesh,changed)
                check=observed_free(changed[~interior],frame)
                check.update(old_solid_interior_points=int(interior.sum()),surface_distance_new_points=len(changed))
                checks.append(check)
                if check['status']!='PASS':return dict(status='UNKNOWN',reason='INSUFFICIENT_GEOMETRIC_EVIDENCE',checks=checks,attachment_included=bool(s.held))
        if s.held:
            # The held envelope is included above, but non-grasp-link contact
            # exclusions still lack calibration. Never pretend this is complete.
            return dict(status='UNKNOWN',reason='attachment_to_robot_collision_contract_missing',checks=checks,attachment_included=True)
        return dict(status='PASS',scope='sampled_public_robot_and_tool_sweep',checks=checks,attachment_included=False)

    def plan(self,p,q):
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from .control_frames import transform_target
        from capx.integrations.motion.public_planning import plan_pyroki
        b=self.b;s=self.s;s.scene.frame()
        if s.held:s.current_held()
        pos,quat=transform_target(b.calibration,p,q)
        target=np.r_[quat,pos+Rotation.from_quat(np.roll(quat,-1)).apply(b.calibration['api_tcp_to_hand_offset_m'])]
        start=np.r_[s.scene.obs['robot_joint_pos'][:7],abs(s.scene.obs['robot_gripper_joint_pos'][0])]
        urdf=next(x['path'] for x in b.calibration['sources'] if x['role']=='ik_urdf')
        spheres=Path(__file__).resolve().parents[1]/'serving/assets/panda_spheres.json'
        obstacles,world=public_obstacles(s,self.geometry,b.calibration)
        row=plan_pyroki(urdf,spheres,start,target,obstacles,held=bool(s.held))
        row['public_world']=world
        row.update(source=s.scene.source,motion_version=s.motion_version,held=s.held['object'] if s.held else None)
        if row['validation']['status']=='REJECTED':row['admission']=dict(status='FAIL',reason=row['validation']['reason'])
        elif row['self_collision']['status']!='PASS':row['admission']=dict(status='FAIL',reason='self_collision')
        else:row['admission']=self.check(np.array(row['trajectory']))
        self.plans.append(row);b.log('shared_trajectory_plan',**row)
        return row

    def __call__(self,p,q,z_approach=0.):
        from .skill_contracts import SkillBoundaryStop
        if z_approach:raise SkillBoundaryStop('Shared backend needs explicit approach waypoint',{ })
        row=self.plan(p,q)
        return self.execute(row)

    def execute(self,row):
        from .skill_contracts import SkillBoundaryStop
        if row['admission']['status']!='PASS':raise SkillBoundaryStop('Shared trajectory rejected: '+str(row['admission'].get('reason')),{})
        if row['motion_version']!=self.s.motion_version or row['held']!=(self.s.held['object'] if self.s.held else None):raise SkillBoundaryStop('Trajectory state version changed',{})
        def read():
            self.s.scene.frame()
            return dict(joints=np.asarray(self.s.scene.obs['robot_joint_pos'])[:7],source=self.s.scene.source)
        def admit(start,end):return self.check(np.array([start,end]))
        def dispatch(joints):
            # Existing controller, every waypoint, original F/T tick guard.
            self.b.public.move_to_joints_blocking(joints,tolerance=.001,max_steps=100)
        try:
            result=execute_waypoints(np.array(row['trajectory'])[:,:7],read,admit,dispatch,self.b.log,
                deadline=self.b.started+self.b.spec['episode_seconds'])
        except ValueError as exc:raise SkillBoundaryStop(str(exc),{}) from exc
        self.b.api.cfg=np.asarray(row['trajectory'][-1]);self.b.log('shared_trajectory_result',**result)
        return result
