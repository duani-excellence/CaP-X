"""Read-only stage measurements for the Oracle diagnostic; distances in mm."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from capx.experiments.pilot import dump


def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def orientation_error(q1,q2):return float(np.rad2deg(Rotation.from_matrix(rot(q1).T@rot(q2)).magnitude()))

def summarize(result,geometry,interventions=None):
    rows=[json.loads(l) for l in Path(result['trace_path']).read_text().splitlines()]
    apis=[r for r in rows if r['kind']=='api'];states=[r['state'] for r in rows if r['kind']=='physics']
    R=np.array(geometry['base_rotation_world']);b=np.array(geometry['base_position_world'])
    mm=lambda x:(np.asarray(x)*1000).tolist()
    base=lambda p:R.T@(np.asarray(p)-b)
    primary=lambda s:np.array(s['objects']['primary']['position'])
    secondary=lambda s:np.array(s['objects']['secondary']['position'])
    out=dict(status=result['status'],reward=result['reward'],snapshot_hash=result.get('snapshot_hash'),info={k:result.get('info',{}).get(k) for k in ['grasp_success','slipped','physical_success','llm_requests','api_calls','execution_status']},estimates=[],moves=[])
    for a in apis:
        if a['name']=='get_object_pose':
            object_key='primary' if a['args'][0]=='red cube' else 'secondary'
            # These dimensions are read from saved real geometry, not inferred from vision.
            # Derive local box sizes from initial rotation + extent using least squares.
            obj=geometry['red'] if object_key=='primary' else geometry['green'];size=np.linalg.lstsq(np.abs(rot(obj[1])),obj[2],rcond=None)[0]
            truth=a['before']['objects'][object_key];gt=base(truth['position']);ext=np.abs(R.T@rot(truth['quaternion_wxyz']))@size
            response=a['response']
            out['estimates'].append(dict(name=a['args'][0],time=a['before']['sim_time'],estimated_base=response[0],truth_base=gt.tolist(),position_error_mm=mm(np.array(response[0])-gt),extent_mm=None if response[2] is None else mm(response[2]),truth_extent_mm=mm(ext),orientation_error_deg=orientation_error(response[1],Rotation.from_matrix(R.T@rot(truth['quaternion_wxyz'])).as_quat()[[3,0,1,2]])))
        if a['name']=='goto_pose':
            s=a['after'];target=np.asarray(a['args'][0]);q=a['args'][1]
            if interventions:
                edited=[r for r in interventions if r.get('kind')=='place_tcp_offset_counterfactual']
                if len(out['moves'])==2 and edited:target=np.asarray(edited[0]['executed'])
            out['moves'].append(dict(requested_base_tcp=a['args'][0],executed_base_tcp=target.tolist(),actual_world_tcp=s['tcp']['position'],actual_world_ee=s['ee']['position'],tcp_error_base_mm=mm(base(s['tcp']['position'])-target),tcp_rotation_error_deg=orientation_error(q,Rotation.from_matrix(R.T@rot(s['tcp']['quaternion_wxyz'])).as_quat()[[3,0,1,2]]),object_minus_tcp_base_mm=mm(R.T@(primary(s)-s['tcp']['position'])),grasp=s['grasp_contact'],time=s['sim_time']))
    release=next((r for r in apis if r['name']=='open_gripper'),None)
    if release:
        for label,s in [('pre_release',release['before']),('post_release',release['after']),('final',states[-1])]:
            delta=primary(s)-secondary(s)
            out[label]=dict(time=s['sim_time'],red_world=s['objects']['primary']['position'],green_world=s['objects']['secondary']['position'],delta_mm=mm(delta),xy_mm=float(np.linalg.norm(delta[:2])*1000),grasp=s['grasp_contact'],opening_mm=s['gripper_opening']*1000,raw_success=s['raw_success'])
        during=[s for s in states if release['before']['sim_time']<=s['sim_time']<=release['after']['sim_time']]
        # Retain all contact names to avoid falsely claiming an event from a guessed label.
        out['release_contacts']=sorted(set((c['geom1'],c['geom2']) for s in during for c in s['contacts']))
        out['release_drift_mm']=mm(primary(release['after'])-primary(release['before']))
        out['release_timeline']=[dict(dt=s['sim_time']-release['before']['sim_time'],delta_mm=mm(primary(s)-secondary(s)),grasp=s['grasp_contact'],contacts=list({(c['geom1'],c['geom2']) for c in s['contacts'] if 'cubeA' in str(c['geom1']) or 'cubeA' in str(c['geom2'])})) for s in during]
    out['max_lift_mm']=max((s['objects']['primary']['position'][2]-apis[0]['before']['objects']['primary']['position'][2])*1000 for s in states)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--historical-run',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.run);summary={}
    for d in sorted(root.iterdir()):
        if not (d/'geometry.json').exists():continue
        results=list(d.glob('oracle/vision/seed-*/result.json'))
        if not results:continue
        r=json.loads(results[0].read_text());g=json.loads((d/'geometry.json').read_text());ints=json.loads((d/'interventions.json').read_text())
        summary[d.name]=summarize(r,g,ints)
        if d.name.startswith('baseline'):
            h=json.loads((Path(a.historical_run)/f"oracle/vision/seed-{r['seed']}/result.json").read_text());summary[f"historical-seed{r['seed']}"]=summarize(h,g)
            def apirows(x):return [v for v in map(json.loads,Path(x['trace_path']).read_text().splitlines()) if v['kind']=='api']
            pairs=list(zip(apirows(r),apirows(h)));summary[d.name]['historical_replay_comparison']=dict(snapshot_equal=r['snapshot_hash']==h['snapshot_hash'],api_names_equal=[v['name'] for v in apirows(r)]==[v['name'] for v in apirows(h)],max_tcp_difference_mm=max(float(np.linalg.norm(np.array(x['after']['tcp']['position'])-y['after']['tcp']['position']))*1000 for x,y in pairs),max_red_difference_mm=max(float(np.linalg.norm(np.array(x['after']['objects']['primary']['position'])-y['after']['objects']['primary']['position']))*1000 for x,y in pairs),max_sim_time_difference=max(abs(x['after']['sim_time']-y['after']['sim_time']) for x,y in pairs))
    dump(a.output,summary)
    for name,s in summary.items():print(name,s['status'],'release_xy',round(s.get('pre_release',{}).get('xy_mm',-1),2),'release_z',round(s.get('pre_release',{}).get('delta_mm',[0,0,-1])[2],2),'final_xy',round(s.get('final',{}).get('xy_mm',-1),2))
if __name__=='__main__':main()
