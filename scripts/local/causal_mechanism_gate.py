"""Reproducible exact-symbolic mechanism gate, NOT robot/LLM/Vision evidence."""
import argparse
import json
import time
from pathlib import Path
from capx.execution.skill_contracts import ContractGraph,ContractBroker
from capx.execution.causal_plan import CausalPlan,ground_library,repair_plan,apply_prediction


def run(automatic):
    # A support can be restored only with a free resource; three actions must be
    # composed before retaining the original assembly and independent branch.
    definition=dict(schema='capx.contract_task.v1',objects=['novel_part','novel_support'],resources=['arm','other'],
        goals=['assembled','peer_done'],condition_specs={'geometry':{'observable':True}},intents=[
            dict(id='assemble',skill='assemble',requires=['held','aligned','geometry'],effects=['assembled','free'],deletes=['held'],resources=['arm']),
            dict(id='peer',skill='peer',requires=['peer_ready'],effects=['peer_done'],resources=['other'])],skill_library=[
            dict(name='park',bindings=[{}],contract=dict(skill='park',requires=['held'],effects=['parked','free'],deletes=['held'],resources=['arm'])),
            dict(name='align',bindings=[{}],contract=dict(skill='align',requires=['free'],effects=['aligned'],resources=['arm'])),
            dict(name='take',bindings=[{}],contract=dict(skill='take',requires=['parked','free'],effects=['held'],deletes=['free','parked'],resources=['arm']))])
    facts=dict(held='TRUE',aligned='TRUE',geometry='TRUE',peer_ready='TRUE',free='FALSE',parked='FALSE',assembled='FALSE',peer_done='FALSE')
    c=ContractGraph(definition);calls=[];events=[]
    for k,v in facts.items():c.observe(k,v,{'source':'explicit_symbolic_fixture'})
    p=CausalPlan(c,['assemble','peer'],automatic=automatic)
    facts['aligned']='FALSE';c.observe('aligned','FALSE',{'source':'explicit_symbolic_update'})
    before=dict(c.conditions);start=time.monotonic()
    decision=repair_plan(p,ground_library(definition),failed=['aligned'])
    if decision['status']!='ready':raise RuntimeError(decision)
    q=p.commit(decision)
    def execute(row):
        calls.append(row['skill']);facts.update(apply_prediction(facts,row))
    def verify(keys):return {k:dict(truth=facts.get(k,'UNKNOWN'),evidence={'source':'exact_symbolic_fixture_NOT_sensor'}) for k in keys}
    broker=ContractBroker(c,{k:execute for k in ['park','align','take','assemble','peer']},verify,emit=events.append,sync_invalidates=True)
    for key in q.order:broker.execute(key)
    return dict(mode='exact_symbolic_fixture_NOT_physical_NOT_DeepSeek',automatic=automatic,decision=decision,
        actual_calls=calls,complete=c.complete(),seconds=time.monotonic()-start,
        preserved_peer_evidence=before['peer_ready']==c.conditions['peer_ready'],
        graph=dict(nodes=c.graph.nodes,edges=c.graph.edges,events=c.graph.events),broker_events=events)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for automatic in [False,True]:
        row=run(automatic);path=a.output/('automatic.json' if automatic else 'conservative.json')
        path.write_text(json.dumps(row,indent=2));print(path.name,row['complete'],row['actual_calls'],row['decision']['affected'])

if __name__=='__main__':main()
