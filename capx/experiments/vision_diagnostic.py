"""Oracle-only diagnostic harness; process-local interventions, no algorithm edits.
Historical vision receipts are frozen to isolate downstream physics/control.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from capx.experiments import pilot
from capx.envs.physical_evaluation import physical_state

VARIANTS=('baseline','green_center','green_xy','green_z','green_extent','red_extent','all_geometry','all_geometry_offset','offset_only')


def truth(low,name):
    env=low.robosuite_env;d=env.sim.data
    obj,bid=(env.cubeA,env.cubeA_body_id) if name=='red cube' else (env.cubeB,env.cubeB_body_id)
    base=low.base_link_idx;R=d.xmat[base].reshape(3,3);p=d.xpos[base]
    Robj=R.T@d.xmat[bid].reshape(3,3)
    return [R.T@(d.xpos[bid]-p),Rotation.from_matrix(Robj).as_quat()[[3,0,1,2]],np.abs(Robj)@(2*np.asarray(obj.size))]


def run(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    history=Path(args.historical_run).resolve();manifest=history/'manifest/scenes.json'
    entry=next(e for e in json.loads(manifest.read_text())['scenes'] if e['seed']==args.seed)
    prior=history/f'oracle/vision/seed-{args.seed}'
    result=json.loads((prior/'result.json').read_text())
    trace=Path(result['trace_path'])
    receipts=[r for r in map(json.loads,trace.read_text().splitlines()) if r['kind']=='api' and r['name'] in ('get_object_pose','sample_grasp_pose')]
    code=(prior/'generated.py').read_text()
    from capx.envs.tasks.franka.franka_pick_place import ORACLE_CODE
    if code!=ORACLE_CODE:raise ValueError('Historical Oracle differs from current Oracle; review before diagnosis')
    # Explicit CPU invocation of the existing server implementation, with the same float32 client cast.
    solve=None
    if args.ik=='cpu':
        import jax
        if jax.default_backend()!='cpu':raise RuntimeError('CPU diagnostic requires JAX_PLATFORMS=cpu')
        from capx.serving import launch_pyroki_server as server
        server.init_pyroki_server()
        # Warm both JIT paths before the unchanged 90 s actor deadline.
        reqs=trace.parent/'requests.jsonl'
        ik=[r['payload'] for r in map(json.loads,reqs.read_text().splitlines()) if r['url'].endswith('/ik')]
        for item in (ik[0],next(i for i in ik if i.get('prev_cfg') is not None)):
            server._do_solve_ik(np.asarray(item['target_pose_wxyz_xyz']),np.asarray(item['prev_cfg']) if item.get('prev_cfg') is not None else None)
        def solve(target_pose_wxyz_xyz,prev_cfg=None):
            t=time.monotonic();j=np.asarray(server._do_solve_ik(np.asarray(target_pose_wxyz_xyz),np.asarray(prev_cfg) if prev_cfg is not None else None),dtype=np.float32)
            with (root/'ik.jsonl').open('a') as f:f.write(json.dumps(dict(target=np.asarray(target_pose_wxyz_xyz).tolist(),previous=None if prev_cfg is None else np.asarray(prev_cfg).tolist(),joints=j.tolist(),seconds=time.monotonic()-t))+'\n')
            return j
    interventions=[];original_environment=pilot.environment
    def environment(mode,artifact_root):
        env=original_environment(mode,artifact_root);reset=env.reset
        def reset_hook(*a,**kw):
            obs,info=reset(*a,**kw);low=env.low_level_env;d=low.robosuite_env.sim.data;base=low.base_link_idx
            pilot.dump(root/'geometry.json',dict(base_position_world=d.xpos[base],base_rotation_world=d.xmat[base].reshape(3,3),red=truth(low,'red cube'),green=truth(low,'green cube'),camera_pose_base=obs['robot0_robotview']['pose'],camera_intrinsics=obs['robot0_robotview']['intrinsics']))
            for api in env._apis.values():
                if solve is not None:api.ik_solve_fn=solve
            index=0
            def receipt(name,callargs,callkwargs):
                nonlocal index
                r=receipts[index];index+=1
                if r['name']!=name or r['args']!=list(callargs) or r['kwargs']!=callkwargs:raise ValueError('Historical perception call order mismatch')
                output=[np.asarray(v) if v is not None else None for v in r['response']]
                raw=copy.deepcopy(output)
                if name=='get_object_pose':
                    obj=callargs[0];gt=truth(low,obj)
                    if args.variant in ('all_geometry','all_geometry_offset'):
                        output=[gt[0],gt[1],gt[2] if output[2] is not None else None]
                    elif args.variant=='green_center' and obj=='green cube':output[0]=gt[0]
                    elif args.variant=='green_xy' and obj=='green cube':output[0][:2]=gt[0][:2]
                    elif args.variant=='green_z' and obj=='green cube':output[0][2]=gt[0][2]
                    elif ((args.variant=='green_extent' and obj=='green cube') or (args.variant=='red_extent' and obj=='red cube')) and output[2] is not None:output[2]=gt[2]
                    interventions.append(dict(call=index,name=name,object=obj,historical=raw,truth=gt,returned=output,physical_state=physical_state(low)))
                else:interventions.append(dict(call=index,name=name,historical=raw,returned=output,physical_state=physical_state(low)))
                return tuple(output)
            env._functions['get_object_pose']=lambda *a,**kw:receipt('get_object_pose',a,kw)
            env._functions['sample_grasp_pose']=lambda *a,**kw:receipt('sample_grasp_pose',a,kw)
            goto=env._functions['goto_pose'];move=0
            def goto_hook(position,quaternion_wxyz,z_approach=0.):
                nonlocal move
                move+=1;pos=np.asarray(position).copy()
                if move==3 and args.variant in ('offset_only','all_geometry_offset'):
                    state=physical_state(low);R=d.xmat[base].reshape(3,3)
                    offset=R.T@(state['objects']['primary']['position']-state['tcp']['position'])
                    pos-=offset
                    interventions.append(dict(kind='place_tcp_offset_counterfactual',requested=position,offset_base=offset,executed=pos,physical_state=state))
                return goto(pos,quaternion_wxyz,z_approach=z_approach)
            env._functions['goto_pose']=goto_hook
            return obs,info
        env.reset=reset_hook
        return env
    pilot.environment=environment
    pilot.dump(root/'provenance.json',dict(seed=args.seed,variant=args.variant,historical_run=str(history),historical_trace=str(trace),historical_trace_sha256=__import__('hashlib').sha256(trace.read_bytes()).hexdigest(),snapshot_digest=entry['snapshot_digest'],oracle_sha256=__import__('hashlib').sha256(code.encode()).hexdigest(),vision_source='frozen historical API returns; no live SAM3/GraspNet',ik=args.ik,actor_timeout_seconds=90,notes='No SFT data. Parent-only truth interventions. Same physics, controller, initial snapshot, sandbox and success criteria. CPU backend differs from historical GPU; baseline must be validated.'))
    try:pilot.run_trial(root,manifest,entry,'vision','oracle',pilot.CostLedger(0))
    finally:
        pilot.environment=original_environment
        pilot.dump(root/'interventions.json',interventions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--variant',choices=VARIANTS,default='baseline');p.add_argument('--ik',choices=['cpu','service'],default='service')
    run(p.parse_args())
if __name__=='__main__':main()
