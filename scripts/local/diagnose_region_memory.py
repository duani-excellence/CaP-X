"""Frozen PUBLIC RGB-D + live segmentation diagnostic; zero LLM, zero actuation."""
import argparse
import json
from pathlib import Path
import numpy as np


def load_frame(root,index):
    calibration=json.loads((root/'observations'/f'{index:05d}-calibration.json').read_text())
    obs={}
    for camera,cal in calibration.items():
        with np.load(root/'observations'/f'{index:05d}-{camera}.npz') as row:
            obs[camera]=dict(images=dict(rgb=row['rgb'],depth=row['depth']),intrinsics=np.array(cal['output_intrinsics']),pose_mat=row['pose_mat'])
            obs.update(robot_cartesian_pos=row['tcp'],robot_joint_pos=row['joints'],robot_gripper_joint_pos=row['gripper'])
    return obs,str(root/'observations'/f'{index:05d}-calibration.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',required=True);p.add_argument('--anchor',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--counterexamples',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import os
    cfg=json.loads(a.config.read_text())
    for key,var in (('sam3','SAM3_SERVICE_URL'),('graspnet','GRASPNET_SERVICE_URL'),('pyroki','PYROKI_SERVICE_URL')):
        if cfg.get('services',{}).get(key):os.environ[var]=cfg['services'][key]
    from capx.experiments.graph_v2 import configure_services
    from capx.experiments.native_tasks import save
    from capx.experiments.native_vision import PublicScene
    cfg=json.loads(a.config.read_text());configure_services(cfg)
    from capx.integrations.vision.sam3 import init_sam3
    from capx.utils.request_audit import RequestAudit
    index=[a.anchor];events=[]
    scene=PublicScene(lambda:load_frame(a.episode,index[0]),init_sam3(),
        cfg['tasks']['pick_place']['vision_skill']['bindings'],optimize=True,emit=lambda kind,**kw:events.append(dict(kind=kind,**kw)))
    scene.allow_region_memory=True;results=[]
    with RequestAudit(a.output/'http.jsonl'):
        scene.frame();scene.bind_regions();save(a.output/'anchor.json',scene.roles)
        for frame in a.frames:
            index[0]=frame;scene.frame();row=dict(frame=frame,results={})
            for role in ('source','destination'):
                result={}
                for memory in (False,True):
                    scene.allow_region_memory=memory
                    try:fit=scene.observe_region(role)
                    except ValueError as exc:fit=dict(scene.region_diagnostics.get(role,{}),exception=str(exc))
                    result['current_plus_history' if memory else 'current_only']=fit
                row['results'][role]=result
            results.append(row)
        if a.counterexamples:
            import copy
            original=copy.deepcopy(scene.obs);checks=[]
            partial={'status':'UNKNOWN','reason':'controlled_missing_semantics'}
            for case in ('missing_depth','visible_wall_removed','calibration_changed'):
                scene.obs=copy.deepcopy(original)
                for camera in scene.obs.values():
                    if not isinstance(camera,dict):continue
                    if case=='missing_depth':camera['images']['depth'][:]=0
                    elif case=='visible_wall_removed':camera['images']['depth'][:]=10
                    else:camera['intrinsics'][0,2]+=1
                outcome=scene.corroborate_region('destination',partial,[np.empty((0,3))],[])
                checks.append(dict(case=case,kind='controlled_PUBLIC_SENSOR_counterexample_NOT_physical_fault',outcome=outcome))
                assert outcome['status']!='PASS',(case,outcome)
            scene.obs=original;save(a.output/'counterexamples.json',checks)
    save(a.output/'results.json',dict(mode='frozen_public_RGBD_live_SAM3_NO_LLM_NO_PHYSICS',results=results))
    save(a.output/'events.json',events)
    print(json.dumps(results,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),indent=2))


if __name__=='__main__':main()
