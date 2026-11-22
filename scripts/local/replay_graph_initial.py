"""Diagnostic ONLY: historical DeepSeek initial + live Vision + bounded recovery.

No LLM client is constructed. Missing/unsupported recovery exits; it cannot call
DeepSeek. This is not a fresh model trial and is not a campaign/formal result.
Uses the existing graph_v2 runner, Sandbox, Snapshot checks and physical score.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-unit',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fault',choices=('none','support_refresh','missing_baseline','target_shift'),default='none')
    p.add_argument('--groups',nargs='+',choices=('G0','G1'),default=['G0','G1'])
    a=p.parse_args(argv)
    from capx.experiments import graph_v2 as runner
    config=json.loads(a.config.read_text());runner.configure_services(config)
    runner.eager_execution_imports()
    original=a.source_unit/'initial/G0-G1.json';data=original.read_bytes();initial=json.loads(data)
    experiment=json.loads((a.source_unit/'experiment.json').read_text())
    code=runner.validated_program(initial['response'])
    if code!=initial['code'] or runner.digest(code)!=initial['code_hash']:raise ValueError('Historical code/response/hash mismatch')
    if initial['response']['response_model']!=config.get('model'):raise ValueError('Model mismatch')
    snapshot=json.loads(a.snapshot.read_text())
    if snapshot['state_hash']!=initial['snapshot_hash']:raise ValueError('Historical snapshot mismatch')
    receipt=Path(initial['request'])/'receipt.json'
    if not receipt.is_file() or json.loads(receipt.read_text()).get('http_status')!=200:
        raise ValueError('A real successful historical response receipt is required')
    a.output.mkdir(parents=True,exist_ok=False)
    provenance=dict(kind='historical_model_live_vision_diagnostic',formal_evaluation=False,
        new_model_requests=0,initial_file=str(original.resolve()),sha256=hashlib.sha256(data).hexdigest(),
        historical_receipt=str(receipt.resolve()),source_unit=str(a.source_unit.resolve()),
        fault=a.fault,limitations='No fresh initial or LLM replanning; only bounded local recipes. Historical cost is not charged again.')
    runner.save(a.output/'provenance.json',provenance)
    runner.save(a.output/'initial/G0-G1.json',initial)
    initial['cache_source']=provenance
    hashes=runner.archive(a.output,config)
    class NoModel:
        count=0;max_requests=1;deadline=time.monotonic()+config['budgets']['wall_seconds']
        def query(self,*args,**kwargs):raise RuntimeError('Global request budget exhausted: diagnostic replay cannot call a model')
    budget=NoModel();rows=[]
    for group in a.groups:
        row,_=runner.run_episode(a.output,experiment['task'],experiment['seed'],experiment.get('repeat',0),
            snapshot,config,budget,group,copy.deepcopy(initial),fault=a.fault,source_hashes=hashes)
        rows.append(row)
    runner.save(a.output/'diagnostic-summary.json',dict(provenance=provenance,episodes=rows,
        new_model_requests=0,new_model_cost_usd=0,uses_actual_online_vision=True))


if __name__=='__main__':main()
