"""Bounded, parent-owned stage recovery on the existing Vision Skills.

MTC's fallback principle is used at verified stage boundaries, not by replaying
an entire successful prefix. Candidate poses retain the existing public sweep,
IK and TCP gates. A changed attachment is a hypothesis until a new co-motion
experiment passes; old hold permissions are never recycled.
"""
import copy
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from .vision_skills import RegionSkills,pose_matrix,jaw_separation,prism_surface,carried_shape,receptacle_obstacles
from .grasp_geometry import panda_open_sweep
from .skill_contracts import SkillBoundaryStop
from .perception_evidence import evidence_scope


def attachment_proposal(reg):
    """Observable surface fit only, deliberately not a hold permission."""
    try:
        t=np.asarray(reg['transform'],float);spread=np.asarray(reg['spread_m'],float)
        return bool(t.shape==(4,4) and np.isfinite(t).all() and np.allclose(t[3],[0,0,0,1])
            and np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6)
            and np.linalg.det(t[:3,:3])>0 and reg['coverage']>=.75
            and 0<=reg['residual_95_m']<.006 and spread.shape==(3,) and spread[-1]>.001)
    except (KeyError,TypeError,ValueError):return False


def attachment_consistency(tcp0,tcp1,object0,object1,width0,width1):
    """New relative transform verified across actual motion, unchanged limits."""
    a=np.asarray(tcp0);b=np.asarray(tcp1);x=np.asarray(object0);y=np.asarray(object1)
    predicted=b@np.linalg.inv(a)@x;delta=y@np.linalg.inv(predicted)
    distance=float(np.linalg.norm(b[:3,3]-a[:3,3]))
    # Compare attached transforms at the measured object origin, not the world
    # origin (rotation and translation of a remote origin are correlated).
    err=float(np.linalg.norm((y-predicted)[:3,3]))
    angle=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude()))
    valid=distance>=.006 and distance<=.02 and err<.004 and angle<3 and min(width0,width1)>.004 and abs(width1-width0)<.003
    return dict(status='PASS' if valid else 'UNKNOWN',robot_motion_m=distance,relative_error_m=err,
                relative_angle_deg=angle,jaw_change_m=abs(width1-width0))


def recovery_library(definition):
    values=[dict(object=x) for x in definition['objects']]
    definition['skill_library'] += [
        dict(name='confirm_attachment',bindings=values,contract=dict(skill='confirm_attachment',object='{object}',
             requires=['attachment_candidate:{object}','probe_ready:{object}'],effects=['held:{object}'],
             invalidates=['attachment_candidate:{object}','probe_ready:{object}'],resources=['arm0'],cost=1.)),
        dict(name='retreat_empty',bindings=values,contract=dict(skill='retreat_empty',object='{object}',
             requires=['approach_stalled:{object}','retreat_ready:{object}'],effects=['robot_ready'],
             deletes=['approach_stalled:{object}'],invalidates=['geometry:{object}','grasp_ready:{object}'],resources=['arm0'],cost=1.))]
    for x in definition['objects']:
        for kind in ['attachment_candidate','probe_ready','approach_stalled','retreat_ready']:
            definition['condition_specs'][kind+':'+x]=dict(observable=True,kind='physical' if kind.endswith(('candidate','stalled')) else 'tool_data')


class RecoverableRegionSkills(RegionSkills):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stalled=None;self.acquisition=None;self.failed_candidates={};self.recovery_attempts={}
        self.candidate_filter=self.candidate_allowed

    @evidence_scope
    def _propose_grasps(self,g):
        transforms,scores,cam=super()._propose_grasps(g)
        if len(scores) or not hasattr(self,'grasp_cloud'):return transforms,scores,cam
        # ASPIRE's object+scene point-cloud protocol, using the existing seeded
        # Contact-GraspNet service. No extra model or simulated mesh is involved.
        from .region_geometry import deproject
        clouds=[deproject(c).reshape(-1,3) for c in self.scene.obs.values() if isinstance(c,dict) and 'images' in c]
        full=np.concatenate(clouds);full=full[np.isfinite(full).all(1)]
        full=full[::max(1,len(full)//80000)]
        t=np.linalg.inv(cam['pose_mat'])
        to_camera=lambda x:np.asarray(x)@t[:3,:3].T+t[:3,3]
        transforms,scores,_=self.grasp_cloud(to_camera(full),to_camera(g['points']),seed=self.seed,
            forward_passes=1,max_retries=2)
        self.emit('skill_multiview_grasp_fallback',source=self.scene.source,scene_points=len(full),
            object_points=len(g['points']),raw_candidates=len(scores),protocol='ASPIRE_object_and_scene_cloud',
            candidate_translation_policy='existing_calibrated_upright_projection_not_raw_tcp')
        return transforms,scores,cam

    def candidate_allowed(self,name,q):
        # Reject the same physical orientation across quaternion signs. A new
        # source ID alone must not resurrect a failed candidate.
        for old in self.failed_candidates.get(name,[]):
            if abs(float(np.dot(q,old)))>np.cos(np.deg2rad(5)/2):return False
        return True

    def stage_conditions(self):
        # Internal execution facts, not object truth or geometric certificates.
        return {kind+':'+name:dict(truth='TRUE' if active else 'FALSE',
                    evidence=dict(reason='parent_stage_record',phase=self.phase))
                for name in self.scene.bindings for kind,active in [
                    ('attachment_candidate',bool(self.held and self.held['object']==name and self.phase=='closed_unverified')),
                    ('approach_stalled',bool(self.stalled and self.stalled['object']==name and self.held is None))]}

    def capabilities(self):
        return dict(acquire=self.acquire,deposit=self.deposit,confirm_attachment=self.confirm_attachment,retreat_empty=self.retreat_empty)

    def _begin_repair(self,name,skill):
        key=(name,skill);n=self.recovery_attempts.get(key,0)
        if n>=2:self.stop('Stage recovery budget exhausted')
        self.recovery_attempts[key]=n+1
        self.emit('skill_recovery_stage',stage=skill,object=name,attempt=n+1,plan_id=uuid.uuid4().hex)

    def _corridor(self,name,probe=False):
        a=self.acquisition if probe else self.stalled
        if not a or a['object']!=name:raise ValueError('No recorded stage corridor')
        tcp=np.asarray(self.scene.obs['robot_cartesian_pos']);p=np.asarray(a['p']);q=np.asarray(a['q'])
        angle=np.degrees((Rotation.from_quat(np.roll(tcp[3:7],-1)).inv()*Rotation.from_quat(np.roll(q,-1))).magnitude())
        dz=tcp[2]-p[2]
        if np.linalg.norm(tcp[:2]-p[:2])>.004 or angle>=3 or not -.004<=dz<=.20:
            raise ValueError('Outside recorded vertical approach scope')
        if probe and not .10<=dz<=.19:raise ValueError('Probe requires lifted state inside observed approach')
        if not probe and (self.held is not None or jaw_separation(self.scene.obs['robot_gripper_joint_pos'])<.075):
            raise ValueError('Retreat requires confirmed open hand and no attachment')
        neighbors={}
        # Retained region is an association ROI, not current free-space proof.
        # The retreat sweep below uses newly measured neighbor/receptacle points.
        region=self.anchors.get('source')
        if region is None:raise ValueError('No bounded association ROI')
        for other in self.scene.bindings:
            if other==name or self.effects.get('placed:'+other)=='TRUE':continue
            neighbors[other]=self.scene.object(other,region)['points']
        neighbors['receptacle']=receptacle_obstacles(self.scene.candidates(self.scene.region_query),a['points'])
        sweep=panda_open_sweep(p,q,neighbors,z_approach=.20)
        self.emit('skill_recovery_corridor',object=name,sweep=sweep,source=self.scene.source,probe=probe)
        if sweep['status']!='PASS':raise ValueError('Current visible recovery corridor rejected')
        target=tcp[:3]+[0,0,.01] if probe else p+[0,0,.20]
        if not self.reachable(target,q):raise ValueError('Recovery target unreachable')
        return target,q

    @evidence_scope
    def verify(self,keys):
        extra=[k for k in keys if k.startswith(('attachment_candidate:','probe_ready:','approach_stalled:','retreat_ready:'))]
        regular=[k for k in keys if k not in extra];rows=super().verify(regular)
        if extra and not regular:self.scene.frame()
        for key in extra:
            kind,name=key.split(':',1);truth='FALSE';reason='stage_not_applicable'
            try:
                if kind=='attachment_candidate':
                    truth='TRUE' if self.held and self.held['object']==name and self.phase=='closed_unverified' else 'FALSE'
                elif kind=='approach_stalled':truth='TRUE' if self.stalled and self.stalled['object']==name and self.held is None else 'FALSE'
                elif kind=='probe_ready':
                    if not self.held or self.held['object']!=name or self.phase!='closed_unverified':raise ValueError('No attachment candidate')
                    self._corridor(name,probe=True);_,reg,width=self._held_measurement()
                    if not attachment_proposal(reg) or width<.004:raise ValueError('No observable attachment proposal')
                    truth='TRUE'
                elif kind=='retreat_ready':self._corridor(name);truth='TRUE'
                reason='current_stage_and_public_evidence'
            except (ValueError,SkillBoundaryStop) as exc:truth='UNKNOWN';reason=str(exc)
            rows[key]=dict(truth=truth,evidence=dict(source=self.scene.source,phase=self.phase,reason=reason))
            self.effects[key]=truth
        if 'robot_ready' in rows and self.stalled:
            rows['robot_ready']=dict(truth='FALSE',evidence=dict(reason='unresolved_approach_stage',source=self.scene.source))
        self.condition_rows.update(copy.deepcopy(rows));return rows

    def acquire(self,row):
        name=row['object']
        if self.held is not None or self.stalled:self.stop('Unresolved stage must be recovered before acquire')
        cached=self.plans.pop(name,None)
        if cached and cached[0]==self.scene.serial:g,p,q,roles=cached[1]
        else:self.scene.frame();g,p,q,roles=self._acquire_plan(name)
        self.acquisition=dict(object=name,p=np.array(p),q=np.array(q),points=g['points'])
        self.phase='approaching';self.active_condition='held:'+name
        self.motion('open_gripper')
        try:self.move(p+[0,0,.2],q);self.move(p,q)
        except SkillBoundaryStop as exc:
            self.stalled=copy.deepcopy(self.acquisition);self.failed_candidates.setdefault(name,[]).append(np.array(q))
            self.phase='approach_stopped'
            raise SkillBoundaryStop(str(exc),{'approach_stalled:'+name:dict(truth='TRUE',evidence={'source':self.scene.source}),
                'held:'+name:dict(truth='FALSE',evidence={'reason':'close_not_dispatched'}),'robot_ready':dict(truth='FALSE',evidence={'reason':'approach_unresolved'})}) from exc
        self.motion('close_gripper');self.scene.frame();tcp=pose_matrix(self.scene.obs['robot_cartesian_pos'])
        self.held=dict(object=name,points=g['points'],model_points=prism_surface(g),shape={k:v for k,v in g.items() if k not in ('points','views')},tcp=tcp,grasp_quaternion=q)
        self.phase='closed_unverified'
        try:
            self.move(p+[0,0,.18],q);shape,reg=self.current_held()
            if shape['bottom_z']-roles['source']['floor_z']<.10:self.stop('No visible lift effect')
        except SkillBoundaryStop as exc:
            effects=dict(exc.effects);effects['attachment_candidate:'+name]=dict(truth='TRUE',evidence={'source':self.scene.source,'not_hold_permission':True})
            raise SkillBoundaryStop(str(exc),effects) from exc
        self.phase='held_verified';self.effects['held:'+name]='TRUE';self.active_condition=None

    def retreat_empty(self,row):
        name=row['object'];self._begin_repair(name,'retreat_empty');self.scene.frame()
        try:p,q=self._corridor(name)
        except ValueError as exc:self.stop(str(exc))
        self.move(p,q);self.stalled=None;self.phase='empty';self.active_condition=None
        self.emit('skill_recovery_success',stage='retreat_empty',object=name,source=self.scene.source)

    def _settle_attachment(self,name):
        # Closing can still be settling after lift. Require measured jaw
        # convergence before testing a NEW attachment hypothesis; do not widen
        # the attachment or release thresholds to accommodate ongoing slip.
        self.scene.frame();previous=jaw_separation(self.scene.obs['robot_gripper_joint_pos'])
        settled=False
        for sample in range(3):
            self.motion('wait',.25);self.scene.frame()
            width=jaw_separation(self.scene.obs['robot_gripper_joint_pos'])
            settled=width>.004 and abs(width-previous)<.0005
            self.emit('skill_attachment_settle',object=name,sample=sample,width_m=width,
                      change_m=abs(width-previous),status='PASS' if settled else 'UNKNOWN')
            if settled:break
            previous=width
        if not settled:self.stop('Jaw did not settle within attachment recovery budget')

    def confirm_attachment(self,row):
        name=row['object'];self._begin_repair(name,'confirm_attachment')
        if not self.held or self.held['object']!=name or self.phase!='closed_unverified':self.stop('No pending attachment')
        self._settle_attachment(name)
        try:p,q=self._corridor(name,probe=True)
        except ValueError as exc:self.stop(str(exc))
        _,first,w0=self._held_measurement()
        if not attachment_proposal(first) or w0<.004:self.stop('No current attachment geometry proposal')
        old=copy.deepcopy(self.held);t0=pose_matrix(self.scene.obs['robot_cartesian_pos']);r0=np.asarray(first['transform'])
        # Hypothesis scoped to this experiment; rollback on ANY failure. The
        # executor's held condition stays UNKNOWN until broker effect verification.
        self.held['tcp']=np.linalg.inv(r0)@t0
        try:
            self._attachment_model(first)
            self.move(p,q);_,second,w1=self._held_measurement();t1=pose_matrix(self.scene.obs['robot_cartesian_pos'])
            if not attachment_proposal(second):self.stop('Probe surface unobservable')
            # Registration transforms act on original world points; compare at
            # original object origin to avoid remote-origin rotation bias.
            origin=np.eye(4);origin[:3,3]=old['shape']['center']
            check=attachment_consistency(t0,t1,r0@origin,np.asarray(second['transform'])@origin,w0,w1)
            self.emit('skill_attachment_probe',object=name,check=check,first=first,second=second,source=self.scene.source)
            if check['status']!='PASS' or second['status']!='PASS':self.stop('Attachment co-motion not established')
            # Keep first independently tested anchor, not continually chase slip.
            self.motion('wait',.25);self.scene.frame();self.current_held()
        except BaseException:
            self.held=old;self.phase='closed_unverified';raise
        self.phase='held_verified';self.effects['held:'+name]='TRUE';self.active_condition=None
        self.emit('skill_recovery_success',stage='confirm_attachment',object=name,source=self.scene.source,
            attachment_id=uuid.uuid4().hex,evidence='new_current_surface_fit_and_measured_rigid_comotion')

    def _attachment_model(self,registration):
        """V30 retains its original reference. Subclasses must keep probe gates."""
        return
