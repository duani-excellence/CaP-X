"""One controlled public-motion capability probe; NEVER a model success trial.

Reuses the original initial program, Snapshot and existing Runner. A single
handwritten recovery fragment tries the documented view motion and verification.
No model call, simulator truth, release or additional skill is introduced.
"""
import argparse,copy,json,hashlib
from pathlib import Path
from capx.experiments.unified import RunSpec,resolve,NoNewLLM,atomic
from capx.experiments.graph_v2 import configure_services
from capx.experiments.graph_campaign import archive_inventory,verify_source_hashes
from capx.experiments.contract_models import run_episode
from capx.execution.graph_journal import GraphJournal
from capx.execution.agent_events import plain

CODE='''s = execution_state()
x = capture_observation()
p = x['robot']['tcp'][:3]
q = x['robot']['tcp'][3:7]
p[2] = p[2] + 0.01
move_tcp(p, q, 'controlled-public-view-1')
g = inspect_region('destination')
print(g)
keys = []
for item in s['intents']:
    if item.get('object') == s['attachment'] and item.get('skill') == 'deposit':
        keys.extend(item.get('requires', []))
print(verify_conditions(keys))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--snapshot-group',required=True);p.add_argument('--ik-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--group',choices=['G0','G1'],default='G1')
    a=p.parse_args();snapshot=json.loads((a.reference/a.snapshot_group/'snapshot.json').read_text())
    _,cfg,_=resolve(RunSpec(harness='persistent',skill='v32',force_mode='on',methods=[a.group],seeds=[snapshot['seed']],ik_runtime=str(a.ik_runtime)))
    configure_services(cfg);common=json.loads((a.reference/'config.json').read_text())['tasks']['pick_place']
    initial=json.loads((a.reference/'initial.json').read_text());initial['historical_replay']=True
    a.output.mkdir(parents=True,exist_ok=False)
    atomic(a.output/'config.json',cfg);atomic(a.output/'initial.json',initial)
    hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in cfg['sources']+[__file__]}
    atomic(a.output/'source-archive.json',archive_inventory(a.output,hashes))
    atomic(a.output/'probe-program.json',dict(code=CODE,origin='controlled_public_probe_NOT_LLM',new_llm_requests=0))
    with GraphJournal(a.output/a.group/'graph-events.jsonl'):
        result,_,_=run_episode(a.output/a.group,copy.deepcopy(cfg['tasks']['pick_place']),snapshot['seed'],a.group,NoNewLLM(),cfg,initial,snapshot,
            environment_spec=common,recorded_recoveries=[dict(code=CODE,origin='controlled_public_probe_NOT_LLM',reference=str(a.output/'probe-program.json'))])
    atomic(a.output/'source-postcheck.json',dict(changed=verify_source_hashes(hashes)))
    atomic(a.output/'provenance.json',dict(mode='controlled_public_capability_probe_NOT_LLM',new_llm_requests=0,source_initial=str(a.reference/'initial.json')))
    print(json.dumps(plain({k:result.get(k) for k in ['raw_success','native_progress','stop_reason','error']})))

if __name__=='__main__':main()
