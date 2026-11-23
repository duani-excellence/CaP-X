"""Frozen PUBLIC sensor memory replay; offline physics is used only afterwards.

No services, model client or robot execution. The trace supplies parent runtime
phase/held identity and already recorded public fits. Physics never initializes
memory or changes its output. This is NOT an online end-to-end experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import hashlib
from collections import Counter
import numpy as np


def load_public(path):
    result={}
    with np.load(path) as f:
        for key in f.files:
            if key.startswith('robot_'):result[key]=f[key];continue
            camera,prop=key.rsplit('_',1) if not key.endswith('_pose_mat') else (key[:-9],'pose_mat')
            value=result.setdefault(camera,{'images':{}})
            if prop in ('rgb','depth'):value['images'][prop]=f[key]
            else:value[prop]=f[key]
    return result


def replay(episode,profile):
    from capx.execution.dependency_graph import DependencyGraph
    from capx.execution.state_memory import StateMemory,depth_occlusion,bounded_place_metrics
    from capx.execution.grip_offset import rotation,predict
    from capx.execution.public_geometry import capture_geometry
    from capx.execution.support_geometry import current_support,color_points
    graph=DependencyGraph();memory=StateMemory(graph,profile);geometry={};attachment=None
    feedback=None;rows=[];old_held=None;grasp_epoch=0;cache={}
    for line in next(episode.glob('trace/*/trace.jsonl')).open():
        d=json.loads(line)
        if d['kind']=='graph_event' and d['event']['action']=='add':
            v=d['event']['node'].get('data') or {}
            if v.get('type') in ('public_geometry_calibration','conditional_upright_grasp','public_support_size'):geometry[v['object']]=v
            if v.get('type')=='public_rigid_grasp_attachment':attachment=v
        elif d['kind']=='feedback':feedback=d
        elif d['kind']=='feedback_evaluation' and feedback and feedback['check']=='attachment_consistency':
            # These two fields are parent execution state, not physical truth.
            obj=d.get('held_object');phase=d.get('phase');frame=feedback['observation_file']
            if not obj or not attachment or phase!='holding_verified':continue
            if obj!=old_held:
                for name in memory.records:memory.invalidate(name,'grasp_transition')
                grasp_epoch+=1;old_held=obj
            if frame not in cache:
                public=load_public(episode/'observations'/frame)
                surfaces,table,_=capture_geometry(public)
                camera=next(v for v in public.values() if isinstance(v,dict) and 'pose_mat' in v)
                meta=json.loads((episode/'observations'/frame).with_suffix('.camera.json').read_text())['calibration']
                calibration=hashlib.sha256(json.dumps([v for _,v in sorted(meta.items())],sort_keys=True).encode()).hexdigest()
                cache[frame]=(public,surfaces,table,np.asarray(camera['pose_mat'])[:3,3],calibration)
            public,surfaces,table,camera,calibration=cache[frame]
            tcp=public['robot_cartesian_pos'];now=feedback['elapsed_seconds'];source=graph.add('observation',dict(frame=frame))
            fit=feedback['metrics'].get('current_surface')
            if feedback['status']=='FAIL':memory.contradict(obj,surfaces.get(obj.split()[0]),source=source)
            elif feedback['status']=='PASS' and fit:
                surface={**predict(attachment,tcp),**fit}
                memory.observe(obj,surface,geometry[obj],source=source,calibration=calibration,stamp=now,
                    evidence_ref=frame,tcp=tcp,attached=True,grasp_epoch=grasp_epoch)
            elif feedback['status']=='UNCERTAIN' and obj in memory.records:
                record=memory.records[obj];s=copy.deepcopy(record['surface'])
                delta=tcp[:3]+rotation(tcp[3:7])@np.asarray(record['offset'])-np.asarray(s['center'])
                for k in ('center','lo','hi'):s[k]=(np.asarray(s[k])+delta).tolist()
                occlusion=depth_occlusion(public,s,geometry[obj]['height_m'])
                points=color_points(public,obj)
                if memory.partial_conflict(obj,points,tcp):memory.contradict(obj,surfaces.get(obj.split()[0]),source=source)
                estimate=memory.estimate(obj,tcp=tcp,stamp=now,calibration=calibration,grasp_epoch=grasp_epoch,
                    hold_verified=True,occlusion=occlusion)
                row=dict(frame=frame,object=obj,occlusion=occlusion,estimate=estimate)
                if estimate['status']=='PREDICTED':
                    other=next((name for name in geometry if name!=obj),None)
                    support=current_support(color_points(public,other),table,geometry[other],camera_origin=camera) if other else None
                    bound={k:profile[k] for k in ('xy_m','z_m','height_m')}
                    # Only the actual recorded object-on-object placement is
                    # evaluated here. A table placement has a different trusted
                    # footprint gate; do not substitute the other object for it.
                    row['release_check']=(bounded_place_metrics(estimate['surface'],support,geometry[obj],estimate['bounds'],bound)
                        if d.get('placement_skill')=='place' else
                        dict(status='NOT_EVALUATED',reason='table_or_unpositioned_state_requires_its_own_contract'))
                    # Evaluation-only: no field from here feeds memory above.
                    state=d['state']['canonical'];truth=next(v for v in state['objects'].values() if v['label']==obj)
                    base=state['base_world'];R=np.asarray(base['rotation'])
                    center=(np.asarray(truth['position_world'])-base['position'])@R
                    extent=np.abs(R.T@np.asarray(truth['rotation_world']))@truth['half_size']
                    errors=dict(xy_m=float(np.linalg.norm(np.asarray(estimate['surface']['center'])[:2]-center[:2])),
                        z_m=abs(float(estimate['surface']['hi'][2]-center[2]-extent[2])))
                    row['offline_error']=errors;row['within_bound']=all(errors[k]<=estimate['bounds'][k] for k in errors)
                rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=replay(a.episode,json.loads(a.profile.read_text()))
    summary=dict(kind='frozen_public_perception_OFFLINE_NOT_live_vision',new_model_requests=0,
        counts=dict(Counter(r['estimate']['status'] for r in rows)),
        release=dict(Counter((r.get('release_check') or {}).get('status','not_evaluated') for r in rows)),
        error_bound_violations=sum(r.get('within_bound') is False for r in rows))
    (a.output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
