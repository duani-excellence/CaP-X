"""Public RGB-D / configured-robot validation for bounded Cartesian proposals.

No task/seed/object truth. This is local sampled planning, not a global cuRobo
replacement. Unknown swept volume is rejected. Parent rechecks each segment.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .grasp_geometry import _panda_boxes
from .vision_skills import pose_matrix,jaw_separation


def empty_motion_scope(skills):
    """Eligibility to propose, NOT a path or contact-force permission."""
    if skills.held:return dict(may_propose=False,permission=False,reason='attachment requires held verifier')
    if skills.phase in ('empty','retreated','released'):
        return dict(may_propose=True,permission=False,reconcile_stalled=False)
    # Existing Skill sets stalled only on a controlled stop BEFORE close was
    # dispatched. Fresh open-jaw encoders are additional required evidence.
    known_open=(skills.phase=='approach_stopped' and bool(getattr(skills,'stalled',None))
        and jaw_separation(skills.scene.obs['robot_gripper_joint_pos'])>=.075)
    return dict(may_propose=known_open,permission=False,reconcile_stalled=known_open,
        reason='known pre-close interruption with measured open jaws' if known_open else 'unresolved physical phase')


def check_plan_start(plan,skills):
    if plan['held']!=(skills.held['object'] if skills.held else None):raise ValueError('Motion plan attachment changed')
    for key,limit in (('robot_joint_pos',.002),('robot_gripper_joint_pos',.001)):
        if np.max(abs(np.asarray(plan['start'][key])-skills.scene.obs[key]))>limit:raise ValueError('Motion plan start encoder state changed')
    old=np.asarray(plan['start']['robot_cartesian_pos']);now=np.asarray(skills.scene.obs['robot_cartesian_pos'])
    a=Rotation.from_quat(np.roll(old[3:7],-1));b=Rotation.from_quat(np.roll(now[3:7],-1))
    if np.linalg.norm(old[:3]-now[:3])>.002 or (a.inv()*b).magnitude()>np.deg2rad(1):raise ValueError('Motion plan start TCP changed')


def observation_opportunities(skills,role):
    """Geometry-directed fixed-camera occluder proposals, NEVER permissions.

    A suggestion requires a historical wall ray blocked near the current tool.
    No unseen wall is invented, no camera is moved, no completeness prerequisite
    for asking about information actions. Each actual plan rechecks full safety.
    """
    scene=skills.scene;prior=scene.region_memory.get(role);diag=scene.region_diagnostics.get(role,{})
    if not prior:return dict(status='UNKNOWN',reason='no indexed wall rays',candidates=[],permission=False)
    tcp=np.asarray(scene.obs['robot_cartesian_pos']);candidates=[]
    for axis,points in prior['walls']:
        if axis not in diag.get('missing_axes',[]):continue
        p=np.asarray(points)
        if not len(p):continue
        for name,cam in scene.obs.items():
            if not isinstance(cam,dict) or 'images' not in cam:continue
            t=cam['pose_mat'];local=(p-t[:3,3])@t[:3,:3];uv=local@cam['intrinsics'].T
            ij=np.rint(uv[:,:2]/np.maximum(uv[:,2,None],1e-9)).astype(int);d=cam['images']['depth'].squeeze();h,w=d.shape
            good=(local[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
            ids=np.flatnonzero(good);observed=d[ij[ids,1],ij[ids,0]]
            blocked=ids[(observed>0)&(observed<local[ids,2]-.004)]
            if len(blocked)<30:continue
            target=np.median(p[blocked],axis=0);ray=target-t[:3,3];ray/=np.linalg.norm(ray)
            closest=t[:3,3]+np.dot(tcp[:3]-t[:3,3],ray)*ray;away=tcp[:3]-closest
            if np.linalg.norm(away)>.10:continue
            if np.linalg.norm(away)<.005:away=np.cross(ray,[0,0,1.])
            if np.linalg.norm(away)<1e-6:continue
            proposal=tcp[:3]+away/np.linalg.norm(away)*.03
            candidates.append(dict(position=proposal.tolist(),quaternion_wxyz=tcp[3:7].tolist(),
                intent='uncover missing wall axis '+str(axis),camera=name,blocked_reference_samples=len(blocked),
                source=scene.source,historical_source=prior['source'],
                expected_information='hypothesis: remove tool occlusion; must measure actual gain',
                permission=False,requires='plan_motion and current parent sweep/attachment/force checks'))
    return dict(status='CANDIDATES' if candidates else 'UNKNOWN',candidates=candidates[:4],
        reason='fixed camera occlusion analysis; no camera actuation',permission=False)


def interpolate_targets(tcp,position,quaternion):
    start=np.asarray(tcp,float);p=np.asarray(position,float);q=np.asarray(quaternion,float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[start,p,q]).all() or abs(np.linalg.norm(q)-1)>.001:raise ValueError('Finite XYZ / unit WXYZ required')
    rots=Rotation.from_quat(np.array([np.roll(start[3:7],-1),np.roll(q,-1)]))
    angle=(rots[0].inv()*rots[1]).magnitude();distance=np.linalg.norm(p-start[:3])
    if distance>.15 or angle>np.deg2rad(30):raise ValueError('Local planner scope: at most 0.15 m and 30 degrees per plan')
    if distance<.0005 and angle<np.deg2rad(.2):raise ValueError('No meaningful proposed motion')
    count=max(1,int(np.ceil(distance/.01)),int(np.ceil(angle/np.deg2rad(3))))
    ts=np.linspace(0,1,count+1)[1:];qs=Slerp([0,1],rots)(ts).as_quat()
    return [(start[:3]*(1-t)+p*t,np.roll(qi,1)) for t,qi in zip(ts,qs)]


def observed_free(points,observation,margin=.004):
    """Require a current unobstructed RGB-D ray for every sampled volume point."""
    p=np.asarray(points,float);free=np.zeros(len(p),bool);seen=np.zeros(len(p),bool)
    for cam in observation.values():
        if not isinstance(cam,dict) or 'images' not in cam:continue
        t=np.asarray(cam['pose_mat']);local=(p-t[:3,3])@t[:3,:3];uv=local@np.asarray(cam['intrinsics']).T
        xy=np.rint(uv[:,:2]/np.maximum(local[:,2,None],1e-9)).astype(int)
        d=np.asarray(cam['images']['depth']).squeeze();h,w=d.shape
        ok=(local[:,2]>0)&(xy[:,0]>=1)&(xy[:,0]<w-1)&(xy[:,1]>=1)&(xy[:,1]<h-1)
        ids=np.flatnonzero(ok)
        if not len(ids):continue
        # Min in a 3x3 pixel footprint avoids isolated missing/edge pixels.
        samples=np.stack([d[xy[ids,1]+y,xy[ids,0]+x] for y,x in itertools.product((-1,0,1),repeat=2)])
        valid=np.isfinite(samples).all(0)&(samples>0).all(0);depth=samples.min(0)
        seen[ids[valid]]=True;free[ids[valid&(depth>local[ids,2]+margin)]]=True
    return dict(status='PASS' if free.all() else 'UNKNOWN',reason='current_depth_free_rays' if free.all() else 'swept_volume_occupied_or_unobserved',
        points=len(p),free_points=int(free.sum()),unobserved_points=int((~seen).sum()),coverage=float(free.mean()) if len(p) else 1.)


def box_points(lo,hi,spacing=.008):
    return np.array(list(itertools.product(*[np.linspace(a,b,max(2,int(np.ceil((b-a)/spacing))+1)) for a,b in zip(lo,hi)])))


def tool_volume(skills,tcp):
    current=pose_matrix(skills.scene.obs['robot_cartesian_pos']);target=pose_matrix(tcp)
    opening=jaw_separation(skills.scene.obs['robot_gripper_joint_pos'])
    boxes=list(_panda_boxes(opening).values());points=[]
    for lo,hi in boxes:
        xyz=box_points(np.asarray(lo)-.004,np.asarray(hi)+.004)
        points.append(xyz@target[:3,:3].T+target[:3,3])
    if skills.held:
        h=skills.held;shape=h['shape'];half=np.r_[shape['half_extent_xy'],shape['height_m']/2]
        half += [skills.profile['xy_error_m'],skills.profile['xy_error_m'],skills.profile['z_error_m']]
        xyz=box_points(np.asarray(shape['center'])-half,np.asarray(shape['center'])+half)
        transform=target@np.linalg.inv(h['tcp']);points.append(xyz@transform[:3,:3].T+transform[:3,3])
    return np.concatenate(points)


def check_tool_segment(skills,p,q):
    from scipy.spatial import cKDTree
    start=np.asarray(skills.scene.obs['robot_cartesian_pos'])[:7]
    old=tool_volume(skills,start);tree=cKDTree(old);checks=[]
    for target,quat in interpolate_targets(start,p,q):
        volume=tool_volume(skills,np.r_[target,quat]);distance,_=tree.query(volume)
        # Only the same currently occupied tool/held volume is exempt. Newly
        # swept volume needs positive current depth, not lack of detections.
        changed=volume[distance>.004]
        check=observed_free(changed,skills.scene.obs);checks.append(check)
        if check['status']!='PASS':return dict(check,scope='tool_and_attached_shape',checks=checks)
    return dict(status='PASS',checks=checks,scope='tool_and_attached_shape')


class PublicJointSweep:
    """Validate actual IK outputs against public robot FK and current RGB-D.

    Called on the IK output used for dispatch, not on a discarded candidate.
    It retains configured geometry; it never reads the simulated scene model.
    """
    def __init__(self,geometry,scene,emit):self.geometry=geometry;self.scene=scene;self.emit=emit

    def __call__(self,joints):
        from scipy.spatial import cKDTree
        obs=self.scene.obs;start=np.asarray(obs['robot_joint_pos'])[:7];end=np.asarray(joints)[:7]
        if np.max(abs(end-start))>.20:raise ValueError('Local IK branch jump exceeds checked joint-segment scope')
        fingers=np.abs(obs['robot_gripper_joint_pos'])
        old,meta=self.geometry.mesh(start,finger_positions_m=fingers,observed_tcp=obs['robot_cartesian_pos'])
        # Dense deterministic surface samples plus face centers. Points inside
        # old robot are exempt only within the mesh sampling tolerance.
        oldp=np.r_[old.vertices,old.triangles_center];tree=cKDTree(oldp)
        count=max(1,int(np.ceil(np.max(abs(end-start))/.01)))
        for alpha in np.linspace(0,1,count+1)[1:]:
            mesh,_=self.geometry.mesh(start*(1-alpha)+end*alpha,finger_positions_m=fingers)
            points=np.r_[mesh.vertices,mesh.triangles_center];d,_=tree.query(points)
            check=observed_free(points[d>.004],obs)
            self.emit('open_joint_sweep',alpha=float(alpha),check=check,calibration=meta['calibration_hash'],source=self.scene.source)
            if check['status']!='PASS':raise ValueError('Current RGB-D cannot certify swept robot geometry')
