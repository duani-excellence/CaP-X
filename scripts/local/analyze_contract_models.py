"""Offline accounting only; private physics is never returned to an actor."""
import argparse
import collections
import json
from pathlib import Path


def analyze(root):
    result=[];unique={}
    config=json.loads((Path(root)/'config.json').read_text())
    for file in sorted(Path(root).glob('*/result.json')):
        r=json.loads(file.read_text())
        if r.get('schema') not in ('capx.contract_models.v26','capx.contract_models.v27','capx.contract_models.v28'):continue
        events=[];trace=file.parent/'trace.jsonl'
        if trace.exists():events=[json.loads(x) for x in trace.read_text().splitlines()]
        http=file.parent.parent/(file.parent.name+'-http.jsonl');times=collections.defaultdict(float);counts=collections.Counter()
        if http.exists():
            for line in http.read_text().splitlines():
                h=json.loads(line);url=h['url'];role='other'
                for service,base in config.get('services',{}).items():
                    if url.startswith(base.rstrip('/')+'/'):
                        role={'sam3':'perception','graspnet':'grasp','pyroki':'ik'}.get(service,service)
                if isinstance(h.get('payload'),dict) and str(h['payload'].get('model','')).startswith('deepseek-'):role='llm'
                times[role]+=h['elapsed_seconds'];counts[role]+=1
        for request in r.get('request_paths',[]):
            receipt=Path(request)/'receipt.json'
            if receipt.exists():unique[str(receipt.resolve())]=json.loads(receipt.read_text())
        goals=r.get('native_progress',{}).get('objects_in_bins',[]);n=sum(float(x)>0 for x in goals)
        interruptions=[i for i,a in enumerate(r.get('actors',[])) if a['status']=='interrupted']
        recovery_calls=[]
        for i in interruptions:
            before=json.loads((file.parent/f'actor-{i:02d}-public.json').read_text())['completed_calls']
            later=set(r.get('completed_calls',[]))-set(before)
            recovery_calls.append(dict(after_block=i,new_completed_calls=sorted(later)))
        policy=[e for e in events if e['kind']=='model_contract']
        api_names=collections.Counter(e['name'] for e in events if e['kind']=='api')
        expiry=next(((i,e['condition']) for i,e in enumerate(events) if e['kind']=='controlled_event' and e.get('event')=='evidence_expired'),None)
        restored=None
        if expiry:
            restored=next((i for i in range(expiry[0]+1,len(events)) if events[i]['kind']=='model_contract' and
                events[i].get('event') in ('recovery_evidence','planned_evidence') and events[i].get('observed',{}).get(expiry[1],{}).get('truth')=='TRUE'),None)
        row={k:r.get(k) for k in ('group','seed','study','ablation','raw_success','reward','seconds','end_to_end_seconds','execution_seconds','api_calls','captures','llm_requests','recovery_llm_requests','input_tokens','output_tokens','cost_usd','last_error','error','fault_triggered','repairs')}
        row.update(native_goals=n,api_per_native_goal=r.get('api_calls',0)/n if n else None,
            api_breakdown=dict(api_names),physical_motion_calls=sum(api_names[k] for k in ('goto_pose','open_gripper','close_gripper','wait')),
            evidence_expiry_restored=restored is not None,
            post_revalidation_motion_calls=sum(e['kind']=='api' and e.get('name') in ('goto_pose','open_gripper','close_gripper','wait') for e in events[restored+1:]) if restored is not None else 0,
            http_counts=dict(counts),http_seconds=dict(times),post_interrupt_progress=recovery_calls,
            policy_events=[e for e in policy if e.get('event') in ('recovery_evidence','repair_intent','continuation','planned_evidence','repair_decision','failed_frontier','bounded_exit')],
            rejection_counts=dict(collections.Counter(e.get('reason') for e in policy if e.get('event')=='boundary_interrupt')),
            planned_refreshes=sum(e.get('event')=='planned_evidence' for e in policy),
            physical_perturbations=[e for e in events if e['kind']=='controlled_physical_disturbance'],
            stop_reason=r.get('stop_reason'),
            local_alignment_count=sum(e['kind']=='skill_local_alignment' for e in events),
            source=str(file))
        result.append(row)
    return dict(episodes=result,unique_paid_requests=len(unique),
        estimated_unique_cost_usd=sum(x.get('cost_upper_usd',0) for x in unique.values()),
        input_tokens=sum(x.get('usage',{}).get('prompt_tokens',0) for x in unique.values()),
        output_tokens=sum(x.get('usage',{}).get('completion_tokens',0) for x in unique.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    with a.output.open('x') as f:json.dump(analyze(a.root),f,indent=2)
