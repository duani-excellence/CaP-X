"""Online-vision diagnostic replay of recorded DeepSeek responses; no new LLM calls.

Reuse the production graph runner. A recorded recovery is consumed only when
the current public prompt exactly matches both its saved prompt and HTTP payload.
Any mismatch stops this diagnostic; there is no provider fallback.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from capx.experiments.pilot import dump


class RecordedResponses:
    def __init__(self, source, output, wall_seconds):
        self.source=Path(source).resolve();self.output=Path(output).resolve()
        self.deadline=time.monotonic()+wall_seconds
        self.count=0  # Actual new model requests, always zero.
        self.rows=[];self.records={};self.consumed=set()
        for path in sorted((self.source/'llm').glob('*/receipt.json')):
            receipt=json.loads(path.read_text());label=receipt['label']
            if label in self.records:raise ValueError('Duplicate recorded request label')
            self.records[label]=(path.parent,receipt)
        self.max_requests=max(1,len(self.records))

    def query(self,prompt,label):
        if time.monotonic()>=self.deadline:raise ValueError('Recorded replay wall budget exhausted')
        if label in self.consumed:raise ValueError('Recorded response already consumed')
        if label not in self.records:raise ValueError('Missing recorded response; no paid fallback')
        directory,receipt=self.records[label]
        if (receipt.get('http_status')!=200 or receipt.get('finish_reason')!='stop'
                or any(receipt.get(k)!='deepseek-v4-pro' for k in ('requested_model','response_model'))):
            raise ValueError('Invalid original receipt')
        expected=json.loads((directory/'prompt.json').read_text())
        if prompt!=expected:raise ValueError('Recorded recovery prompt drift; replay refused')
        transport=json.loads((directory/'http.json').read_text())
        if transport.get('payload',{}).get('messages')!=expected:
            raise ValueError('Original transport prompt mismatch')
        response_bytes=(directory/'response.json').read_bytes()
        answer=json.loads(response_bytes)
        if answer.get('response_model')!='deepseek-v4-pro' or answer.get('finish_reason')!='stop':
            raise ValueError('Invalid recorded response')
        row=dict(label=label,original_request=str(directory),prompt_exact_match=True,
            transport_prompt_exact_match=True,new_llm_requests=0,
            response_sha256=hashlib.sha256(response_bytes).hexdigest(),
            response_source='historical_real_DeepSeek_response',
            source_files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                          for name in ('receipt.json','prompt.json','http.json','response.json')})
        dump(self.output/'recorded-responses'/f'{len(self.rows)+1:03d}.json',row)
        self.rows.append(row);self.consumed.add(label)
        return copy.deepcopy(answer),str(directory)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fault',choices=['none','support_refresh'],default='none')
    parser.add_argument('--groups',nargs='+',choices=['G0','G1'],default=['G0','G1'])
    parser.add_argument('--wall-seconds',type=float,default=600)
    opts=parser.parse_args(argv)
    if len(set(opts.groups))!=len(opts.groups) or opts.wall_seconds<=0:
        parser.error('Distinct groups and a positive wall budget are required')
    source=opts.source_run.resolve();root=opts.output.resolve()
    config=json.loads((source/'experiment.json').read_text())
    snapshot=json.loads((source/'snapshot-source.json').read_text())
    entry=next(e for e in snapshot['entries'] if e['seed']==opts.seed)
    # Reuse the verified service URLs and all task/model execution budgets.
    for name in ('sam3','graspnet','pyroki'):
        os.environ[name.upper()+'_SERVICE_URL']=config[name+'_url']
    # Integration modules read service URLs at import time. Configure first.
    from capx.experiments.graph_feedback import archive_sources, load_initial_run, run_episode
    root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    names=list(config['source_hashes'])+['capx/experiments/graph_replay.py']
    hashes=archive_sources(repo,root,names)
    dump(root/'experiment.json',dict(schema='capx.recorded_graph_replay.v1',
        source_run=str(source),seed=opts.seed,fault=opts.fault,groups=opts.groups,
        online_vision=True,new_llm_requests=0,source_hashes=hashes,
        original_configuration=config,
        limitation='Historical model responses with live vision/physics; not fresh online model generation or an additional independent sample'))
    dump(root/'snapshot-source.json',snapshot)
    initial=load_initial_run(source,root,opts.seed,opts.fault)
    budget=RecordedResponses(source,root,opts.wall_seconds)
    results=[];start=time.monotonic()
    try:
        for group in opts.groups:
            result,_=run_episode(root,Path(snapshot['manifest']),entry,group,
                copy.deepcopy(initial),budget,SimpleNamespace(**config),fault=opts.fault)
            results.append(result)
            if result.get('error') or result.get('status') in ('system_error','timeout'):
                raise RuntimeError('Recorded diagnostic failed; inspect new artifacts')
    finally:
        dump(root/'summary.json',dict(schema='capx.recorded_graph_replay.v1',
            online_vision=True,new_llm_requests=0,new_model_cost_usd=0,
            model_generation='reused_real_DeepSeek_responses_not_new_requests',
            results=results,recorded_responses=budget.rows,seconds=time.monotonic()-start))


if __name__=='__main__':main()
