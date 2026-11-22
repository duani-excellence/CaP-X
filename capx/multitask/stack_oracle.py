"""Read-only acceptance through the original Stack runner."""
import argparse
from pathlib import Path
from capx.experiments.pilot import prepare,run_trial,CostLedger
from capx.training.teacher import read,dump,successful
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
r=Path(a.root);r.mkdir(parents=True,exist_ok=False);m=r/'manifest/scenes.json';prepare(m,r,[a.seed])
result=run_trial(r,m,read(m)['scenes'][0],'privileged','oracle',CostLedger())
dump(r/'acceptance.json',dict(task='stack',seed=a.seed,status='ready' if successful(result) else 'unverified',oracle_status=result['status'],snapshot_replay=True))
