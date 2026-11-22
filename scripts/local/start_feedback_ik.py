"""Run the unchanged CaP-X IK solver on CPU without replacing existing services."""
import argparse
import json
import os
from pathlib import Path
import socket


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-file',type=Path,required=True)
    p.add_argument('--warmup-requests',type=Path,required=True,help='Existing P0 requests.jsonl with IK requests')
    p.add_argument('--port',type=int,default=0,help='0 lets the OS choose a free loopback port')
    args=p.parse_args()
    if args.runtime_file.exists():raise FileExistsError(args.runtime_file)
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import numpy as np
    import uvicorn
    from capx.serving import launch_pyroki_server as original
    if jax.default_backend()!='cpu':raise RuntimeError('This diagnostic service must use CPU')
    original.init_pyroki_server()
    rows=[json.loads(l) for l in args.warmup_requests.read_text().splitlines()]
    requests=[r['payload'] for r in rows if r['url'].endswith('/ik')]
    for previous in (False,True):
        row=next(r for r in requests if (r.get('prev_cfg') is not None)==previous)
        original._do_solve_ik(np.asarray(row['target_pose_wxyz_xyz']),np.asarray(row['prev_cfg']) if previous else None)
    sock=socket.socket();sock.bind(('127.0.0.1',args.port));sock.listen(128)
    port=sock.getsockname()[1]
    args.runtime_file.parent.mkdir(parents=True,exist_ok=True)
    with args.runtime_file.open('x') as f:
        json.dump(dict(url=f'http://127.0.0.1:{port}',pid=os.getpid(),backend='cpu',solver='unchanged capx.serving.launch_pyroki_server',warmup_source=str(args.warmup_requests.resolve())),f,indent=2)
    uvicorn.Server(uvicorn.Config(original.app,host='127.0.0.1',port=port)).run(sockets=[sock])

if __name__=='__main__':main()
