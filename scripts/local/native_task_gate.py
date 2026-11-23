#!/usr/bin/env python3
"""Native task feasibility gates; no model construction and no paid requests."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--tasks',nargs='+',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--output',required=True)
    p.add_argument('--gate',choices=['environment','vision-probe','oracle','oracle-graph','vision-graph','vision-skills'],default='environment')
    p.add_argument('--group',choices=['G0','G1'],default='G1')
    p.add_argument('--program',help='Explicit reference Python program for oracle gate; never model-labelled')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    if set(args.tasks)-set(cfg['tasks']):p.error('Unknown configured task')
    for k,v in cfg['runtime_environment'].items():os.environ[k]=str(v)
    for key,env in [('sam3','SAM3_SERVICE_URL'),('graspnet','GRASPNET_SERVICE_URL'),('pyroki','PYROKI_SERVICE_URL')]:os.environ[env]=cfg['services'][key]
    import numpy as np
    from capx.experiments.native_tasks import create_native,make_low,snapshot,physical_state,NativeBoundary,save
    from capx.utils.request_audit import RequestAudit
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);save(out/'config.json',cfg)
    import hashlib
    from capx.experiments.graph_campaign import archive_inventory,verify_source_hashes
    sources=[Path(x) for x in cfg.get('sources',[])]+[Path(args.config)]
    if args.program:sources.append(Path(args.program))
    hashes={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    save(out/'source-archive.json',archive_inventory(out,hashes))
    results=[]
    for name in args.tasks:
        spec=cfg['tasks'][name];start=time.monotonic();row=dict(task=name,seed=args.seed,gate=args.gate,new_llm_requests=0)
        env=low=boundary=None
        try:
            if args.gate=='environment':
                snaps=[]
                for i in range(2):
                    if len(spec['robots'])==1:low=make_low(spec,args.seed);env=low.robosuite_env
                    else:env=create_native(spec,args.seed)
                    snap=snapshot(env,args.seed,spec);snaps.append(snap);save(out/f'{name}-snapshot-{i}.json',snap)
                    state=physical_state(env);save(out/f'{name}-physics-{i}.json',state)
                    if i==0:
                        from PIL import Image
                        camera=spec['cameras'][0]
                        Image.fromarray(env._get_observations(force_update=True)[camera+'_image'][::-1]).save(out/f'{name}.png')
                    env.close();env=low=None
                row.update(snapshot_replay=snaps[0]==snaps[1],initial_success=state['raw_success'],
                    passed=snaps[0]==snaps[1] and not state['raw_success'])
            else:
                with RequestAudit(out/f'{name}-requests.jsonl') as audit:
                    boundary=NativeBoundary(spec,args.seed,out/name,mode='oracle' if args.gate in ('oracle','oracle-graph') else 'vision')
                    if args.gate=='oracle':
                        if not args.program:raise ValueError('--program is required; no implicit canned success')
                        row.update(boundary.execute(Path(args.program).read_text()))
                    elif args.gate=='oracle-graph':
                        if not args.program:raise ValueError('--program required')
                        from capx.experiments.native_reference_graph import run_reference_graph
                        row.update(run_reference_graph(boundary,spec,Path(args.program).read_text(),args.group))
                    elif args.gate=='vision-skills':
                        from capx.experiments.native_vision_skills import run_skills
                        row.update(run_skills(boundary,spec,args.group))
                    elif args.gate=='vision-probe':
                        from capx.experiments.native_vision import probe
                        row.update(probe(boundary))
                    else:
                        from capx.experiments.native_vision import run_graph
                        row.update(run_graph(boundary,spec))
                    row['actual_http']=audit.count
        except Exception as exc:
            row.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        finally:
            if boundary:boundary.close()
            elif env:env.close()
        row['wall_seconds']=time.monotonic()-start;results.append(row);save(out/f'{name}-gate.json',row)
        print(name,json.dumps({k:v for k,v in row.items() if k in ('passed','raw_success','reward','error','wall_seconds','snapshot_replay')},ensure_ascii=False),flush=True)
    save(out/'summary.json',results)
    save(out/'source-postcheck.json',dict(changed=verify_source_hashes(hashes),checked_files=len(hashes)))


if __name__=='__main__':main()
