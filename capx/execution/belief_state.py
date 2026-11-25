"""Versioned index of existing public estimates, separate from action truth.

This is a journal/query layer over PublicScene and Skill/StateMemory outputs,
not another perception pipeline or a source of motion/release permission.
"""
import copy
import time
from .agent_events import plain


class BeliefState:
    def __init__(self,graph,emit=None):
        self.graph=graph;self.emit=emit or (lambda *a,**k:None)
        self.current={};self.history={};self.version=0

    def record(self,key,values,*,source,kind='direct_measurement',parents=(),assumptions=()):
        values=plain(values)
        if 'points' in values:
            points=values.pop('points')
            values['point_count']=points if isinstance(points,int) else len(points)
        values.pop('views',None)
        self.version+=1;old=self.current.get(key)
        if old:
            old['kind']='historical_measurement'
            self.graph.invalidate([old['id']],reason='belief superseded; historical event retained')
        status=values.get('status','UNKNOWN')
        row=dict(key=key,version=self.version,source=source,frame='robot_base',
            timestamp_monotonic=time.monotonic(),kind=kind,status=status,values=values,
            assumptions=list(assumptions),permission=False,
            uncertainty=values.get('uncertainty',values.get('bounds','not_established')))
        node=self.graph.add('belief_state',row,parents=list(parents))
        row=dict(row,id=node);self.current[key]=row;self.history.setdefault(key,[]).append(row)
        self.emit('BeliefUpdated',**copy.deepcopy(row));return row

    def robot_motion(self,source):
        # No old object pose is silently promoted to current truth. Rigid
        # propagation must be supplied by the qualified existing Skill model.
        for key,row in self.current.items():
            if key.startswith(('object:','region:')) and row['kind']=='direct_measurement':
                row['kind']='historical_measurement'
                self.emit('BeliefAged',key=key,id=row['id'],source=source,permission=False)

    def query(self,key=None,history=False):
        if key is None:return copy.deepcopy(self.current)
        if key not in self.current:return dict(status='UNKNOWN',reason='no_public_evidence',permission=False)
        return copy.deepcopy(self.history[key][-4:] if history else self.current[key])

    def ingest(self,kind,data,source):
        """Only existing measured diagnostics; no evaluator or task score input."""
        if kind in ('region_geometry_diagnostic','single_region_revalidation'):
            return self.record('region:'+data['role'],data.get('fit',data.get('geometry',{})),source=source)
        if kind=='object_geometry':
            return self.record('object:'+data['object'],dict(data['geometry'],identity_confidence=None,
                identity_basis='existing_multiview_instance_binding'),source=source)
        if kind in ('skill_hold_check','skill_hold_evidence'):
            return self.record('attachment:'+data['object'],data,source=source,
                kind='measured_registration_with_rigid_shape_model',assumptions=['existing attachment verifier applies'])
        if kind=='skill_shape_memory':
            return self.record('shape:'+data['object'],data,source=source,kind='model_inference')
        if kind=='placement_quality_observed':
            return self.record('quality:'+data['object'],data,source=source)
