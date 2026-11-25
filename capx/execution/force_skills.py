"""Shared V32 physical evidence and bounded attachment recovery.

F/T never replaces current geometry, rigid co-motion, TCP or release checks.
No task/object identities or simulator handles occur in these skills.
"""
import numpy as np
from .skill_recovery import RecoverableRegionSkills
from .vision_skills import jaw_separation


def observed_attachment_template(cloud,transform):
    """Observed surface in the existing shape-memory frame; not a new shape bound."""
    p=np.asarray(cloud,float);t=np.asarray(transform,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<100 or not np.isfinite(p).all():raise ValueError('Insufficient observed attachment surface')
    spread=np.linalg.svd(p-p.mean(0),compute_uv=False)/np.sqrt(len(p))
    if spread[-1]<=.001:raise ValueError('Planar surface cannot constrain attachment')
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6):raise ValueError('Invalid attachment reference frame')
    return (p-t[:3,3])@t[:3,:3]


class ForceRegionSkills(RecoverableRegionSkills):
    def __init__(self,*args,force_evidence=None,archive_cloud=None,**kwargs):
        super().__init__(*args,**kwargs);self.force_evidence=force_evidence
        self.force_graph=None;self.load_anchors={};self.force_decisions=[];self.archive_cloud=archive_cloud

    def _force_record(self,kind,result):
        result=dict(result)
        if 'timestamp' in result:result.setdefault('expires_at',result['timestamp']+.15)
        if self.force_graph is not None and 'timestamp' in result:
            g=self.force_graph.graph
            expired=[n for n,v in g.nodes.items() if v.get('valid') and v.get('kind')=='observation'
                and str(v.get('key','')).startswith('wrist_evidence:')
                and v.get('data',{}).get('expires_at',float('inf'))<result['timestamp']]
            if expired:g.invalidate(expired,reason='wrist_evidence_expired')
        if self.force_decisions and self.force_decisions[-1]==dict(kind=kind,**result):return result
        if self.force_graph is not None:
            g=self.force_graph.graph;obj=self.held['object'] if self.held else 'unbound'
            node=g.version('wrist_evidence:'+kind+':'+obj,'observation',dict(source='force.jsonl',**result),reason='new_force_window')
            self.force_graph.observe(kind+':'+obj,'TRUE' if result.get('status')=='PASS' else 'UNKNOWN',
                dict(source_node=node,**result),parents=[node])
        self.force_decisions.append(dict(kind=kind,**result));self.emit(kind,**result)
        return result

    def verify_grasp_force(self):
        if self.force_evidence is None:return dict(status='UNKNOWN',reason='force_disabled',hold_permission=False)
        value=self.force_evidence();return self._force_record('force_load',value)

    def detect_contact(self):
        if self.force_evidence is None:return dict(status='UNKNOWN',reason='force_disabled')
        row=self.scene.obs.get('wrist_ft')
        if not row or not row['valid']:return self._force_record('force_contact',dict(status='UNKNOWN',reason='no_current_wrench'))
        payload=self.load_anchors.get(self.held['object'],0.) if self.held else 0.
        f=float(np.linalg.norm(np.asarray(row['external_base'][:3])+[0,0,payload]));t=float(np.linalg.norm(row['external_base'][3:]))
        return self._force_record('force_contact',dict(status='PASS' if f>2 else 'UNKNOWN',force_n=f,torque_nm=t,
            estimated_payload_load_n=payload,payload_torque_compensated=False,
            reason='wrench_change_is_contact_candidate_not_contact_identity',timestamp=row['timestamp']))

    def monitor_slip(self,registration=None):
        if self.force_evidence is None:return dict(status='UNKNOWN',reason='force_disabled')
        e=self.force_evidence();obj=self.held['object'] if self.held else None;anchor=self.load_anchors.get(obj)
        lost=bool(anchor and e.get('load_interval_n',[anchor,anchor])[1]<anchor*.5)
        moving=e.get('jaw_drift_m',0)>.0005
        conflict=bool(registration and registration.get('status')!='PASS')
        status='FAIL' if lost and (moving or conflict) else ('PASS' if e['status']=='PASS' and not conflict else 'UNKNOWN')
        return self._force_record('force_slip',dict(e,status=status,load_loss=lost,vision_conflict=conflict,
            reason='multimodal_relative_stability_not_force_closure'))

    def force_guarded_motion(self,p,q):
        # Parent samples actual native wrench at each 20 Hz control boundary and
        # arrests the current target on overload. No motion cancellation claim
        # inside a MuJoCo integration substep or a real hardware controller.
        super().move(p,q)
        if self.force_evidence is not None:self.detect_contact()

    def move(self,p,q):return self.force_guarded_motion(p,q)

    def acquire(self,row):
        if self.held is None:self.load_anchors.pop(row['object'],None)
        return super().acquire(row)

    def _attachment_model(self,registration):
        source,cloud=self.last_held_cloud
        if source!=self.scene.source:self.stop('Observed attachment surface is stale')
        try:template=observed_attachment_template(cloud,registration['transform'])
        except ValueError as exc:self.stop(str(exc))
        self.held['model_points']=template
        reference=self.archive_cloud(template) if self.archive_cloud else None
        self.emit('skill_attachment_surface_reference',object=self.held['object'],source=source,points=len(template),
            reference=reference,shape_bounds_changed=False,permission=False,
            reason='current_observed_surface_for_cross_motion_registration; pending unchanged co-motion test')

    def _settle_attachment(self,name):
        if self.force_evidence is None:return super()._settle_attachment(name)
        self.scene.frame();previous=jaw_separation(self.scene.obs['robot_gripper_joint_pos']);no_progress=0;previous_drift=float('inf')
        # Same 0.5 mm limit as V30, bounded extra settling conditional on measured
        # load. No object-dependent time extension or replacement threshold.
        for sample in range(20):
            self.motion('wait',.25);self.scene.frame();width=jaw_separation(self.scene.obs['robot_gripper_joint_pos'])
            drift=abs(width-previous);e=self.verify_grasp_force()
            self.emit('skill_attachment_settle',object=name,sample=sample,width_m=width,change_m=drift,
                status='PASS' if width>.004 and drift<.0005 and e['status']=='PASS' else 'UNKNOWN',force=e)
            if width>.004 and drift<.0005 and e['status']=='PASS':
                return
            if sample>=2 and ('load_n' in e and not 3*e.get('force_error_bound_n',.005)<e['load_n']<15):self.stop('Attachment load missing or outside verified range')
            no_progress=no_progress+1 if drift>=previous_drift*.98 else 0
            if no_progress>=6:self.stop('No progress in force/jaw convergence')
            if self.scene.obs['wrist_ft']['valid'] is not True:self.stop('Wrist sensor invalid')
            previous=width;previous_drift=drift
        self.stop('Force-guided attachment settling exhausted')

    def recover_attachment(self,row):
        # Existing geometric hypothesis and actual 6–20 mm co-motion validation
        # remain mandatory after settling. Unknown physical effects stay unknown.
        return self.confirm_attachment(row)

    def current_held(self):
        shape,reg=super().current_held()
        if self.force_evidence is not None:
            evidence=self.monitor_slip(reg)
            if evidence['status']=='FAIL':self.stop('Load loss with changing attachment')
            if evidence['status']=='PASS':self.load_anchors.setdefault(self.held['object'],evidence['load_n'])
        return shape,reg
