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
    p.add_argument('--harness',choices=['persistent','open'],default='persistent')
    p.add_argument('--initial-program',type=Path,help='Explicitly labelled controlled public program, never a model output')
    p.add_argument('--config',type=Path);p.add_argument('--decision-encoding',choices=['legacy','encoded'])
    p.add_argument('--recorded-recovery',type=Path,action='append',default=[])
    p.add_argument('--live-recovery-requests',type=int,default=0,
        help='Optional new model requests AFTER recorded programs; default 0, same Runner/budget')
    a=p.parse_args();snapshot=json.loads((a.reference/a.snapshot_group/'snapshot.json').read_text())
    if not 0<=a.live_recovery_requests<=3:p.error('live recovery requests must be 0..3 for this small probe')
    _,cfg,_=resolve(RunSpec(harness=a.harness,skill='v32',force_mode='on',methods=[a.group],seeds=[snapshot['seed']],ik_runtime=str(a.ik_runtime),config=str(a.config) if a.config else None,decision_encoding=a.decision_encoding))
    configure_services(cfg);common=copy.deepcopy(snapshot.get('spec',json.loads((a.reference/'config.json').read_text())['tasks']['pick_place']))
    initial=json.loads((a.reference/'initial.json').read_text());initial['historical_replay']=True
    if a.initial_program:
        from capx.experiments.graph_v2 import digest
        initial.update(code=a.initial_program.read_text(),request=None,origin='controlled_public_probe_NOT_LLM')
        initial['code_hash']=digest(initial['code'])
    cfg['budgets']['max_requests']=min(cfg['budgets']['max_requests'],a.live_recovery_requests)
    cfg['tasks']['pick_place']['harness_max_requests']=a.live_recovery_requests
    a.output.mkdir(parents=True,exist_ok=False)
    atomic(a.output/'config.json',cfg);atomic(a.output/'initial.json',initial)
    hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in cfg['sources']+[__file__]}
    atomic(a.output/'source-archive.json',archive_inventory(a.output,hashes))
    atomic(a.output/'probe-program.json',dict(code=CODE,origin='controlled_public_probe_NOT_LLM',new_llm_requests=0))
    recoveries=[dict(code=json.loads(f.read_text())['code'],origin='archived_model_program',reference=str(f)) for f in a.recorded_recovery]
    if not recoveries and not a.initial_program:recoveries=[dict(code=CODE,origin='controlled_public_probe_NOT_LLM',reference=str(a.output/'probe-program.json'))]
    from capx.experiments.graph_campaign import PersistentModelBudget
    budget=PersistentModelBudget(a.output,cfg['budgets']) if a.live_recovery_requests else NoNewLLM()
    with GraphJournal(a.output/a.group/'graph-events.jsonl'):
        result,_,_=run_episode(a.output/a.group,copy.deepcopy(cfg['tasks']['pick_place']),snapshot['seed'],a.group,budget,cfg,initial,snapshot,
            environment_spec=common,recorded_recoveries=recoveries)
    atomic(a.output/'source-postcheck.json',dict(changed=verify_source_hashes(hashes)))
    atomic(a.output/'provenance.json',dict(mode='recorded_prefix_with_live_recovery' if a.live_recovery_requests else 'controlled_public_capability_probe_NOT_LLM',
        new_llm_requests=budget.count,source_initial=str(a.reference/'initial.json')))
    print(json.dumps(plain({k:result.get(k) for k in ['raw_success','native_progress','stop_reason','error']})))

if __name__=='__main__':main()
