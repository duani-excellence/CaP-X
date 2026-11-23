"""Shared staged public-observation skills; no simulator, task or evaluator API.

Development capabilities, not a claim of qualification on unseen objects. All
motion is dispatched through a parent callback and all effects require evidence.
"""
import copy
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .skill_contracts import SkillBoundaryStop
from .region_geometry import region_permission,foreground_component,region_cells
from .grasp_geometry import cgn_upright_panda,panda_open_sweep


def jaw_separation(encoders):
    q=np.asarray(encoders,float)
    if q.shape!=(2,) or not np.isfinite(q).all() or np.max(abs(q))>.041:
        raise ValueError('Two calibrated Panda slide encoders required')
    return float(np.sum(abs(q)))


def depth_association(observation,shape,xy_error,z_error):
    """Current measured points in a predicted volume; never fill absent depth.

    The association is a hypothesis, not a hold certificate. Registration must
    constrain 3D geometry and reject contradictory points / insufficient views.
    """
    from .region_geometry import deproject
    c=np.asarray(shape['center']);half=np.r_[shape['half_extent_xy'],shape['height_m']/2]+[xy_error,xy_error,z_error]
    rows=[]
    for camera in observation.values():
        if not isinstance(camera,dict) or 'images' not in camera:continue
        p=deproject(camera).reshape(-1,3);valid=np.isfinite(p).all(1)&np.all(abs(p-c)<=half,axis=1)
        rows.append(p[valid])
    return np.concatenate(rows) if rows else np.empty((0,3))


def pose_matrix(tcp):
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(np.roll(np.asarray(tcp)[3:7],-1)).as_matrix();t[:3,3]=tcp[:3]
    return t


def carried_shape(shape,transform):
    t=np.asarray(transform);c=t[:3,:3]@shape['center']+t[:3,3]
    half=np.abs(t[:3,:3])@np.r_[shape['half_extent_xy'],shape['height_m']/2]
    return dict(shape,center=c.tolist(),half_extent_xy=half[:2].tolist(),
                bottom_z=float(c[2]-half[2]),top_z=float(c[2]+half[2]),
                source='registered_current_RGBD_with_rigid_shape_memory')


def prism_surface(shape):
    """Inferred upright prism surface, explicitly NOT an observed point cloud.

    Its template error is part of the non-cube offline development profile.
    New current depth is still required; occlusion alone never authorizes release.
    """
    ext=np.r_[shape['extent_xy'],shape['height_m']];points=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        a,b=np.meshgrid(*[np.linspace(-ext[i]/2,ext[i]/2,max(5,int(ext[i]/.002))) for i in other])
        for sign in (-1,1):
            p=np.zeros((a.size,3));p[:,axis]=sign*ext[axis]/2;p[:,other[0]]=a.ravel();p[:,other[1]]=b.ravel();points.append(p)
    basis=np.eye(3);basis[:2,:2]=shape['basis_xy']
    return np.concatenate(points)@basis.T+shape['center']


def release_correction(shape,region,xy_error,z_error):
    half=np.asarray(shape['half_extent_xy']);lo=np.asarray(region['lo'])+half+xy_error+.025
    hi=np.asarray(region['hi'])-half-xy_error-.025
    if np.any(hi<=lo) or z_error>=.0125:raise ValueError('No admissible interval interior')
    c=np.asarray(shape['center']);desired=np.clip(c[:2],lo+(hi-lo)*.1,hi-(hi-lo)*.1)
    delta=np.r_[desired-c[:2],.0125-(shape['bottom_z']-region['floor_z'])]
    if np.linalg.norm(delta)>.03:raise ValueError('Correction exceeds local motion scope')
    return delta


def settled_prediction(shape,transform,floor_z):
    t=np.asarray(transform).copy();prior=carried_shape(shape,t)
    t[2,3]+=floor_z-prior['bottom_z']
    return t


def planar_jaw_proposals(shape):
    basis=np.asarray(shape['basis_xy']);extent=np.asarray(shape['extent_xy']);result=[]
    for y in [basis[:,0],basis[:,1],np.array([1.,0.]),np.array([0.,1.])]:
        width=float(np.abs(basis.T@y)@extent)
        if width+.006>.08:continue
        for sign in (-1,1):
            closing=np.r_[sign*y,0.];down=np.array([0.,0.,-1.]);r=np.c_[np.cross(closing,down),closing,down]
            result.append(dict(quaternion=np.roll(Rotation.from_matrix(r).as_quat(),1),width_m=width,
                               source='public_prism_geometry_not_GraspNet'))
    return result


def receptacle_obstacles(rows,target_points):
    """Keep every measured receptacle surface, including horizontal rims.

    A semantic tray mask can contain the target. Only samples coincident with
    its independently segmented current surface are excluded. No bounding-box
    erasure, hidden wall extrapolation, or assumption of free unseen space.
    """
    points=np.concatenate([r['points'] for r in rows]) if rows else np.empty((0,3))
    target=np.asarray(target_points)
    if len(target) and len(points):
        distance,_=cKDTree(target).query(points)
        points=points[distance>.003]
    return points


def register_rigid(reference,current,initial):
    """Partial observed surface registration against a kinematic prediction.

    Innovation and observed support are reported separately. Agreement with this
    model is not independent proof of unseen shape or stable force closure.
    """
    ref=np.asarray(reference)[::max(1,len(reference)//4000)]
    cur=np.asarray(current)[::max(1,len(current)//4000)];t=np.asarray(initial).copy()
    if len(ref)<100 or len(cur)<100:return dict(status='UNKNOWN',reason='insufficient_registration_points')
    for _ in range(12):
        pred=ref@t[:3,:3].T+t[:3,3];d,i=cKDTree(pred).query(cur);good=d<.012
        if good.mean()<.6 or good.sum()<100:return dict(status='UNKNOWN',reason='shape_prediction_contradicted',coverage=float(good.mean()))
        a=pred[i[good]];b=cur[good];ac=a.mean(0);bc=b.mean(0)
        u,s,v=np.linalg.svd((a-ac).T@(b-bc));r=v.T@u.T
        if np.linalg.det(r)<0:v[-1]*=-1;r=v.T@u.T
        delta=np.eye(4);delta[:3,:3]=r;delta[:3,3]=bc-r@ac;t=delta@t
        if np.linalg.norm(delta-np.eye(4))<1e-6:break
    pred=ref@t[:3,:3].T+t[:3,3];d,i=cKDTree(pred).query(cur);good=d<.006
    innovation=t@np.linalg.inv(initial);angle=Rotation.from_matrix(innovation[:3,:3]).magnitude()
    shift=np.linalg.norm((t[:3,:3]-initial[:3,:3])@ref.mean(0)+t[:3,3]-initial[:3,3])
    # A single planar patch cannot constrain all 6 rigid degrees of freedom.
    spread=np.linalg.svd(cur[good]-cur[good].mean(0),compute_uv=False)/np.sqrt(max(1,good.sum())) if good.sum() else np.zeros(3)
    passed=good.mean()>=.75 and good.sum()>=100 and shift<.012 and angle<np.deg2rad(8) and spread[-1]>.001
    return dict(status='PASS' if passed else 'UNKNOWN',reason='current_surface_registration' if passed else 'unconstrained_or_conflicting_registration',
                transform=t.tolist(),coverage=float(good.mean()),residual_95_m=float(np.quantile(d[good],.95)) if good.any() else None,
                innovation_m=float(shift),innovation_deg=float(np.degrees(angle)),spread_m=spread.tolist())


class RegionSkills:
    """Acquire / region deposit / effect verification, shared across policies.

    Task declarations bind object descriptors and cell IDs. These skills know no
    task names, object identity semantics, fault labels or simulator properties.
    """
    def __init__(self,scene,motion,grasp,reachable,profile,*,seed=0,emit=None):
        self.scene=scene;self.motion=motion;self.grasp=grasp;self.reachable=reachable
        self.profile=profile;self.seed=seed;self.emit=emit or (lambda *a,**k:None)
        self.held=None;self.released={};self.effects={};self.phase='empty';self.anchors={};self.targets={}
        if not profile.get('offline_evidence') or profile.get('schema')!='capx.upright_development_bounds.v1':
            raise ValueError('Explicit non-cube offline error evidence required')

    def stop(self,reason):
        effects={}
        if self.held is not None:
            key='held:'+self.held['object'];self.effects[key]='UNKNOWN'
            effects[key]=dict(truth='UNKNOWN',evidence={'reason':reason,'phase':self.phase})
        raise SkillBoundaryStop(reason,effects)

    def verify(self,keys):
        """Current public effect verification, never a cached success lookup.

        The released instance model is an association hypothesis. Missing current
        depth means UNKNOWN and can revoke a previously verified task condition.
        Private task reward is deliberately unavailable to this verifier.
        """
        self.scene.frame();rows={}
        for key in keys:
            truth='UNKNOWN';reason='unsupported_condition'
            try:
                if key=='robot_ready':
                    truth='TRUE' if self.held is None and np.isfinite(self.scene.obs['robot_joint_pos']).all() else 'FALSE'
                    reason='current_robot_and_attachment_state'
                elif key.startswith('held:'):
                    obj=key.split(':',1)[1]
                    if self.held is None:truth='FALSE';reason='no_attachment'
                    elif self.held['object']==obj:
                        self.current_held();truth='TRUE';reason='current_registered_surface_and_jaw'
                elif key.startswith('placed:'):
                    obj=key.split(':',1)[1]
                    if obj in self.released:
                        self.observe_placement(obj);truth='TRUE';reason='current_registered_instance_in_target'
            except (SkillBoundaryStop,ValueError) as exc:reason=str(exc)
            self.effects[key]=truth
            rows[key]=dict(truth=truth,evidence={'source':self.scene.source,'phase':self.phase,'reason':reason})
        return rows

    def observe_placement(self,name):
        item=self.released[name];h=item['attachment'];region=item['region']
        initial=settled_prediction(h['shape'],item['release_transform'],region['floor_z'])
        proposal=carried_shape(h['shape'],initial)
        points=depth_association(self.scene.obs,proposal,self.profile['xy_error_m'],self.profile['z_error_m'])
        points=points[points[:,2]>region['floor_z']+.003]
        reg=register_rigid(h['model_points'],points,initial)
        self.emit('skill_placed_surface_evidence',object=name,registration=reg,capture=self.scene.source,
                  evidence_kind='current_RGBD_with_retained_instance_model_NOT_semantic_marker')
        if reg['status']!='PASS':self.stop('Placed instance geometry not observable or inconsistent')
        shape=carried_shape(h['shape'],reg['transform']);xy=np.asarray(shape['center'])[:2]
        if not (np.all(xy>region['lo']) and np.all(xy<region['hi'])):self.stop('Placed instance outside target')
        return shape

    def prepare(self):
        self.scene.frame();roles=self.scene.bind_regions();self.anchors=copy.deepcopy(roles)
        self.targets={str(i):r for i,r in enumerate(region_cells(roles['destination'],[2,2]))}
        return roles

    def region(self,role):
        try:return self.scene.observe_region(role)
        except ValueError as exc:self.stop(str(exc))

    def move(self,p,q):
        self.motion('goto_pose',np.asarray(p),np.asarray(q))
        for attempt in range(4):
            self.scene.frame()
            tcp=np.asarray(self.scene.obs['robot_cartesian_pos']);error=np.linalg.norm(tcp[:3]-p)
            angular=np.degrees((Rotation.from_quat(np.roll(tcp[3:7],-1)).inv()*Rotation.from_quat(np.roll(q,-1))).magnitude())
            self.emit('skill_control_check',status='PASS' if error<.004 and angular<3 else 'FAIL',target=list(p),actual=tcp.tolist(),position_error_m=float(error),angle_deg=float(angular),settle_attempt=attempt)
            if error<.004 and angular<3:break
            if attempt<3:self.motion('wait',.25)
        if error>=.004 or angular>=3:self.stop('physical TCP did not converge')

    def current_held(self):
        h=self.held
        if h is None:self.stop('No established attachment')
        now=pose_matrix(self.scene.obs['robot_cartesian_pos']);prediction=now@np.linalg.inv(h['tcp'])
        proposal=carried_shape(h['shape'],prediction)
        center=np.asarray(proposal['center']);half=np.maximum(.08,np.asarray(proposal['half_extent_xy'])+.03)
        roi=dict(lo=(center[:2]-half).tolist(),hi=(center[:2]+half).tolist(),
                 center=center.tolist(),floor_z=proposal['bottom_z']-.02,ceiling_z=proposal['top_z']+.02)
        # Kinematics only proposes a segmentation crop; it cannot grant hold or
        # release. Current RGB-D must still pass independent surface fitting.
        rows=self.scene.candidates(self.scene.bindings[h['object']],roi)
        rows=[r for r in rows if self.scene.in_region(r,roi)]
        if rows:
            cloud,screen=foreground_component(np.concatenate([r['points'] for r in rows]));source='current_semantic_RGBD'
        else:
            cloud=depth_association(self.scene.obs,proposal,self.profile['xy_error_m'],self.profile['z_error_m'])
            source='current_depth_in_kinematic_association_volume'
        self.emit('skill_tracking_source',object=h['object'],source=source,measured_points=len(cloud),capture=self.scene.source,
                  prediction_is_only_association=True)
        reg=register_rigid(h['model_points'],cloud,prediction)
        width=jaw_separation(self.scene.obs['robot_gripper_joint_pos'])
        self.emit('skill_hold_evidence',object=h['object'],registration=reg,gripper_width_m=width)
        if reg['status']!='PASS' or width<.004:self.stop('Hold check failed: '+reg['reason'])
        shape=carried_shape(h['shape'],reg['transform']);self.emit('skill_hold_check',object=h['object'],registration=reg,gripper_width_m=width,shape=shape)
        return shape,reg

    def acquire(self,row):
        if self.held is not None:self.stop('Cannot acquire while another attachment exists')
        name=row['object'];self.scene.frame();roles={'source':self.region('source')}
        try:g=self.scene.object(name,roles['source'])
        except ValueError as exc:self.stop(str(exc))
        limits=self.profile['shape_range_m']
        if not limits[0]<=g['height_m']<=limits[1] or max(g['extent_xy'])>limits[2]:self.stop('Shape outside calibrated capability')
        view=max(g['views'],key=lambda r:r['mask'].sum());cam=self.scene.obs[view['camera']]
        transforms,scores,_=self.grasp(cam['images']['depth'].squeeze(),cam['intrinsics'],view['mask'].astype(np.int32),1,seed=self.seed,max_retries=2)
        if not len(scores):
            alternatives=[r for r in g['views'] if r['camera']!=view['camera']]
            if alternatives:
                view=max(alternatives,key=lambda r:r['mask'].sum());cam=self.scene.obs[view['camera']]
                transforms,scores,_=self.grasp(cam['images']['depth'].squeeze(),cam['intrinsics'],view['mask'].astype(np.int32),1,seed=self.seed,max_retries=2)
        candidates=[];depth=min(.02,g['height_m']/2);p=np.array(g['center']);p[2]=g['top_z']-depth
        # Existing calibrated CGN->Panda axis mapping. Projection changes the
        # candidate, so every projected pose is screened again before motion.
        for i in np.argsort(np.asarray(scores).ravel())[::-1][:32]:
            t=cam['pose_mat']@np.asarray(transforms)[i];q=np.roll(Rotation.from_matrix(t[:3,:3]).as_quat(),1)
            try:q=np.array(cgn_upright_panda(q)['quaternion_wxyz'])
            except ValueError:continue
            rot=Rotation.from_quat(np.roll(q,-1)).as_matrix();xy=np.asarray(g['points'])[:,:2]@rot[:2,1]
            width=float(np.ptp(xy))
            if width+.006>.08:continue
            candidates.append((float(scores[i]),q,width))
        self.emit('skill_grasp_candidates',object=name,raw_count=len(scores),qualified_width_count=len(candidates),source=self.scene.source)
        geometric=planar_jaw_proposals(g)
        candidates.extend((0.,r['quaternion'],r['width_m']) for r in geometric)
        self.emit('skill_geometric_grasp_proposals',object=name,candidates=geometric,source=self.scene.source)
        if not candidates:self.stop('No grasp candidate fits jaw aperture')
        # All known loose objects remain explicit obstacles; absence is UNKNOWN.
        neighbors={}
        for other in self.scene.bindings:
            if other==name or self.effects.get('placed:'+other)=='TRUE':continue
            try:neighbors[other]=self.scene.object(other,roles['source'])['points']
            except ValueError:self.stop('Missing neighboring object geometry')
        # Axis-normal walls omit horizontal rims that can hit the palm. Keep
        # the full currently observed receptacle, minus exact target surfaces.
        neighbors['observed_receptacle']=receptacle_obstacles(
            self.scene.candidates(self.scene.region_query),g['points'])
        selected=None
        for score,q,width in candidates:
            sweep=panda_open_sweep(p,q,neighbors,z_approach=.20)
            if sweep['status']!='PASS':continue
            if self.reachable(p,q) and self.reachable(p+[0,0,.2],q):selected=q;break
        if selected is None:self.stop('No reachable collision-screened grasp')
        q=selected;self.emit('skill_grasp_plan',object=name,position=p.tolist(),quaternion=q.tolist(),shape={k:v for k,v in g.items() if k not in ('points','views')},sweep=sweep)
        self.motion('open_gripper');self.move(p+[0,0,.2],q);self.move(p,q)
        self.motion('close_gripper');self.scene.frame();tcp=pose_matrix(self.scene.obs['robot_cartesian_pos'])
        self.held=dict(object=name,points=g['points'],model_points=prism_surface(g),shape={k:v for k,v in g.items() if k not in ('points','views')},tcp=tcp,grasp_quaternion=q)
        self.emit('skill_shape_memory',object=name,evidence_kind='inferred_upright_prism_NOT_current_observation',
                  observed_points=len(g['points']),model_points=len(self.held['model_points']),profile=self.profile['schema'])
        self.phase='closed_unverified';self.move(p+[0,0,.18],q)
        shape,reg=self.current_held()
        if shape['bottom_z']-roles['source']['floor_z']<.10:self.stop('No visible lift effect')
        self.phase='held_verified';self.effects['held:'+name]='TRUE'

    def deposit(self,row):
        name=row['object']
        if self.held is None or self.held['object']!=name:self.stop('Required object is not held')
        self.scene.frame();shape,reg=self.current_held();destination=self.region('destination')
        region=region_cells(destination,[2,2])[int(row['target'])]
        h=self.held;original=h['shape'];q=np.array([0.,0.,1.,0.]);rot=Rotation.from_quat(np.roll(q,-1)).as_matrix()
        # Full object--TCP transform, including off-centre attachment and yaw.
        relative=np.linalg.inv(h['tcp']);local=relative[:3,:3]@original['center']+relative[:3,3]
        delta=rot@h['tcp'][:3,:3].T;half=np.abs(delta)@np.r_[original['half_extent_xy'],original['height_m']/2]
        margin=half[:2]+self.profile['xy_error_m']+reg['residual_95_m']+.025
        lo=np.array(region['lo'])+margin;hi=np.array(region['hi'])-margin
        if np.any(hi<lo):self.stop('Object uncertainty does not fit target region')
        positions=[]
        for alpha in [0.5,.25,.75,0.,1.]:
            for beta in [.5,.25,.75,0.,1.]:positions.append(lo+(hi-lo)*[alpha,beta])
        positions.sort(key=lambda xy:np.linalg.norm(xy-np.asarray(shape['center'])[:2]))
        target=None
        for xy in positions[:12]:
            center=np.r_[xy,region['floor_z']+half[2]+.0125];p=center-rot@local
            if self.reachable(p,q) and self.reachable(p+[0,0,.2],q):target=p;break
        if target is None:self.stop('No reachable interior placement in bounded search')
        self.emit('skill_place_plan',object=name,target=target.tolist(),region=region,attachment_local=local.tolist())
        self.move(target+[0,0,.2],q);self.current_held();self.phase='transported'
        self.move(target+[0,0,.05],q);self.current_held()
        self.move(target,q);self.phase='pre_release'
        # Final capture is followed by BOTH live support and hold checks. A prior
        # PASS is historical evidence, never a reusable release permission.
        self.scene.frame();shape,reg=self.current_held();destination=self.region('destination')
        current=region_cells(destination,[2,2])[int(row['target'])]
        previous_error=None
        for correction in range(3):
            xy_error=self.profile['xy_error_m']+reg['residual_95_m'];z_error=self.profile['z_error_m']+reg['residual_95_m']
            permission=region_permission(shape,current,xy_error=xy_error,z_error=z_error,edge_margin=.025)
            self.emit('skill_release_permission',object=name,permission=permission,source=self.scene.source,correction=correction)
            if permission['status']=='PASS' or correction==2:break
            try:delta=release_correction(shape,current,xy_error,z_error)
            except ValueError as exc:self.stop(str(exc))
            magnitude=np.linalg.norm(delta)
            if previous_error is not None and magnitude>=previous_error-.0005:self.stop('Local alignment made no measurable progress')
            previous_error=magnitude
            tcp=np.asarray(self.scene.obs['robot_cartesian_pos']);new_target=tcp[:3]+delta
            if not self.reachable(new_target,q):self.stop('Local alignment unreachable')
            self.emit('skill_local_alignment',object=name,delta=delta.tolist(),new_plan_id='alignment-'+str(self.scene.serial),
                      reason='current_complete_geometry_interval_not_contained')
            self.move(new_target,q);self.scene.frame();shape,reg=self.current_held();destination=self.region('destination')
            current=region_cells(destination,[2,2])[int(row['target'])]
        if permission['status']!='PASS':self.stop('Current release interval rejected')
        self.motion('open_gripper');self.released[name]=dict(attachment=h,release_transform=reg['transform'],region=current)
        self.held=None;self.phase='released';self.effects['held:'+name]='FALSE'
        self.move(target+[0,0,.2],q);self.phase='retreated'
        self.scene.frame();observed=self.observe_placement(name)
        xy=np.array(observed['center'])[:2];inside=bool(np.all(xy>current['lo']) and np.all(xy<current['hi']))
        # A second independent time sample is required for visible stability.
        first=observed;self.motion('wait',.5);self.scene.frame();second=self.observe_placement(name)
        stable=np.linalg.norm(np.array(first['center'])-second['center'])<.003
        self.effects['placed:'+name]='TRUE' if inside and stable else 'UNKNOWN'
        self.effects['held:'+name]='FALSE'
        self.emit('skill_placement_verified',object=name,inside=inside,stable=bool(stable),source=self.scene.source)
        if not inside or not stable:self.stop('Placed object not independently observed stable in target')
