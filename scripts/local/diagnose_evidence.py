"""Offline V23 error accounting and current-mask table boundary diagnosis.

Inputs are archived artifacts; no model, network, private feedback or execution.
V23's anchor envelope lumps attachment/calibration/pose errors: this script
reports that identifiability limit rather than inventing individual variances.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from replay_state_memory import load_public


def errors(rows,profile):
    result=[]
    for row in rows:
        est=row['estimate']
        if est['status']!='PREDICTED':
            result.append(dict(frame=row['frame'],object=row['object'],status=est['status'],reason=est['reason']))
            continue
        c=row.get('release_check',{})
        terms={x:est[field]*profile[rate] for x,field,rate in (
            ('travel','travel_m','translation_rate'),('rotation','rotation_rad','rotation_rate'),('age','age_seconds','age_rate'))}
        growth=sum(terms.values())
        result.append(dict(frame=row['frame'],object=row['object'],status=est['status'],
            initial_xy_each_m=profile['xy_m'],initial_top_z_each_m=profile['z_m'],intrinsic_height_m=profile['height_m'],
            growth_terms_m=terms,xy_bound_sum_m=2*profile['xy_m']+growth,
            v23_gap_bound_sum_m=2*profile['z_m']+profile['height_m']+2*growth,
            corrected_gap_bound_before_tilt_m=2*profile['z_m']+profile['height_m']+growth,
            nominal_xy_m=c.get('xy_error_m'),nominal_gap_m=c.get('gap_m'),original_release=c,
            source=est.get('evidence_ref'),age=est['age_seconds'],travel=est['travel_m'],rotation=est['rotation_rad'],
            correlation_cancellation_m=0,offline_error=row.get('offline_error'),
            limitation='Initial pose, attachment and calibration are lumped in the empirical anchor envelope; no identifiable shared-error covariance. Corrected formula is not permission; actual tilt and freshness still required.'))
    return result


def table(episode):
    from capx.execution.public_geometry import capture_geometry
    nodes=json.loads((episode/'graph-final.json').read_text())['nodes']
    if not isinstance(nodes,dict):nodes={n['id']:n for n in nodes}
    rows=[]
    for line in next(episode.glob('trace/*/trace.jsonl')).open():
        d=json.loads(line)
        if d['kind']!='feedback' or d['check']!='table_clearance' or d['status']=='PASS':continue
        m=d['metrics'];e=m['evidence']
        if not e.get('mask_artifact'):continue
        region=nodes[m['region_id']]['data'];pub=load_public(episode/'observations'/d['observation_file'])
        c=next(v for v in pub.values() if isinstance(v,dict) and 'images' in v)
        depth=c['images']['depth'].squeeze();rgb=c['images']['rgb'];k=c['intrinsics'];t=c['pose_mat'];yy,xx=np.indices(depth.shape)
        pts=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],-1)@t[:3,:3].T+t[:3,3]
        with np.load(e['mask_artifact']) as masks:
            near=(np.abs(pts[:,:,:2]-region['center'][:2])<=region['footprint_half_extent_xy']).all(-1)
            z=capture_geometry(pub)[1];mask=near&masks['valid_depth']&~masks['held']&~masks['robot']&(pts[:,:,2]>z+.012)
            dist=distance_transform_edt(~masks['held_semantic'])[mask]
            rows.append(dict(frame=d['observation_file'],validation=m['validation'],residual_count=int(mask.sum()),
                semantic_edge_distance_pixels=dist.tolist(),residual_rgb=rgb[mask].tolist(),residual_xyz=pts[mask].tolist(),
                region=region,boundary=e.get('held_mask_boundary_refinement'),
                interpretation='Edge proximity is not object identity or clearance; no residual point is removed by this diagnostic.'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-rows',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--table-episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('error-decomposition',errors(json.loads(a.frozen_rows.read_text()),json.loads(a.profile.read_text()))),
                      ('table-residuals',table(a.table_episode))]:
        (a.output/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
        print(name,len(rows),'offline records; zero new LLM requests')


if __name__=='__main__':main()
