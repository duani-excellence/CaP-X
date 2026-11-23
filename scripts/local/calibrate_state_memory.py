"""OFFLINE ONLY: measure V22 public geometry errors against archived physics.

Never imported by the online monitor. Exports scalar error envelopes and source
hashes, never object poses or fitted offsets. This is development calibration,
not a distribution-free guarantee or an independent test-set result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def samples(episode):
    from capx.execution.grip_offset import predict
    from capx.execution.public_geometry import measured_surface
    from capx.execution.support_geometry import reliable_top,current_support,color_points
    nodes={};geometry={};attachment=None;robot=None;feedback=None;rows=[];observation=None;cache={}
    trace=next(episode.glob('trace/*/trace.jsonl'))
    for line in trace.open():
        d=json.loads(line)
        if d['kind']=='graph_event' and d['event']['action']=='add':
            n=d['event']['node'];nodes[n['id']]=n;v=n.get('data') or {}
            if v.get('type') in ('public_geometry_calibration','conditional_upright_grasp','public_support_size'):
                geometry[v['object']]=v
            if v.get('type')=='public_rigid_grasp_attachment':attachment=v
            if n['kind']=='observation':robot=v.get('robot');observation=v
        elif d['kind']=='feedback':feedback=d
        elif d['kind']=='feedback_evaluation' and feedback and robot and observation:
            state=d['state'].get('canonical');held=d.get('held_object')
            if not state:continue
            base=state['base_world'];R=np.array(base['rotation']);origin=np.array(base['position'])
            candidates=[]
            if feedback['check']=='attachment_consistency' and feedback['status']=='PASS' and attachment and held and feedback['metrics'].get('current_surface'):
                surface={**predict(attachment,robot['robot_cartesian_pos']),**feedback['metrics'].get('current_surface',{})}
                candidates.append((held,surface,'attached'))
            # Identifiable current face geometry only. Legacy centroid or an
            # attachment prediction cannot independently initialize memory.
            for name,g in geometry.items():
                if name==held:continue
                key=(feedback['observation_file'],name)
                if key not in cache:
                    data=np.load(episode/'observations'/key[0]);public={}
                    for field in data.files:
                        if field.startswith('robot_'):public[field]=data[field]
                        else:
                            camera,prop=field.rsplit('_',1) if not field.endswith('_pose_mat') else (field[:-9],'pose_mat')
                            v=public.setdefault(camera,{'images':{}})
                            if prop in ('rgb','depth'):v['images'][prop]=data[field]
                            else:v[prop]=data[field]
                    cameras=[v for v in public.values() if isinstance(v,dict) and 'pose_mat' in v]
                    cam=np.asarray(cameras[0]['pose_mat'])[:3,3] if cameras else None
                    cache[key]=current_support(color_points(public,name),observation.get('table_z'),g,camera_origin=cam)
                if cache[key]:candidates.append((name,cache[key],'static'))
            for name,s,kind in candidates:
                obj=next((o for o in state['objects'].values() if o['label']==name),None)
                if not obj:continue
                center=(np.array(obj['position_world'])-origin)@R
                extent=np.abs(R.T@np.array(obj['rotation_world']))@np.array(obj['half_size'])
                top=center[2]+extent[2];height=2*extent[2];geom=geometry[name]
                rows.append(dict(episode=str(episode),frame=feedback['observation_file'],object=name,kind=kind,
                    xy_m=float(np.linalg.norm(np.array(s['center'])[:2]-center[:2])),
                    z_m=abs(float(s['hi'][2]-top)),height_m=abs(float(geom['height_m']-height)),
                    error_xyz=(np.asarray(s['center'])-center).tolist(),
                    top_error=float(s['hi'][2]-top),height_error=float(geom['height_m']-height),
                    sim_time=state['sim_time'],public_tcp=robot['robot_cartesian_pos']))
    # Repeated checks in one captured frame are one calibration observation.
    return list({(r['frame'],r['object'],r['kind']):r for r in rows}.values()),trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[];sources=[]
    for e in a.episodes:
        r,t=samples(e);rows+=r;sources.append(dict(path=str(t.resolve()),sha256=hashlib.sha256(t.read_bytes()).hexdigest()))
    if not rows:raise ValueError('No aligned calibration samples')
    # Full observed envelope plus one millimetre measurement reserve. NO bias
    # subtraction, task-specific geometry correction or threshold modification.
    bounds={k:max(r[k] for r in rows)+.001 for k in ('xy_m','z_m','height_m')}
    identity=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    profile=dict(schema='capx.memory_bounds.v1',calibration_id=identity,validated=True,
        validation_kind='archived_development_empirical_envelope_NOT_heldout_guarantee',
        frame='robot_base',**bounds,translation_rate=.01,rotation_rate=.01,
        age_rate=.0001,static_rate=.0001,max_age_seconds=10.,max_travel_m=.35,max_rotation_rad=.4,
        scope='same public upright cube estimator, renderer and robot calibration; short noncontact rigid transport only',
        evidence=sources,rate_policy='conservative engineering growth allowance, NOT fitted probability',
        estimator_policy='identifiable_current_faces_only',
        external_disturbance_policy='contradiction/touch/slip/loss invalidates; arbitrary hidden disturbance not covered')
    (a.output/'samples.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(dict(samples=len(rows),episodes=len(a.episodes),
        max_errors={k:bounds[k]-.001 for k in bounds},bounds=bounds,
        independent_holdout=False,permission_requires_interval_inclusion=True),indent=2)+'\n')
    print(json.dumps(dict(samples=len(rows),bounds=bounds)))


if __name__=='__main__':main()
