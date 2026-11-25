"""Read-only aggregation of real V34 receipts, actors and parent events."""
import argparse
import collections
import json
from pathlib import Path
from urllib.parse import urlsplit


def summarize(root):
    rows=[]
    for path in sorted(root.glob('*/summary.json')):
        summary=json.loads(path.read_text());run=path.parent
        ledger=run/'request-ledger.json'
        requests=json.loads(ledger.read_text())['requests'] if ledger.exists() else []
        for episode in summary.get('episodes',[]):
            dest=run/'seed-0-repeat-0'/episode.get('group','G0')
            actors=[]
            for file in sorted(dest.glob('actor-*-result.json')):
                actor=json.loads(file.read_text());codefile=file.with_name(file.name.replace('-result','-program'))
                program=json.loads(codefile.read_text()) if codefile.exists() else {}
                actors.append(dict(file=file.name,result=actor,origin=program.get('origin'),program=codefile.name))
            events=dest/'harness/events.jsonl';lifecycle=collections.Counter();actions={};belief_kinds=collections.Counter();decisions=[]
            if events.exists():
                for line in events.open():
                    event=json.loads(line);data=event['data']
                    if event['type']=='ActionResult':lifecycle[data['state']]+=1
                    if event['type']=='ActionStarted':actions[data['action_id']]=data['name']
                    if event['type']=='BeliefUpdated':belief_kinds[data.get('kind','unknown')]+=1
                    if event['type']=='RecoveryProposed':decisions.append(data)
            http=dest.with_name(dest.name+'-http.jsonl');counts=collections.Counter();seconds=collections.Counter()
            if http.exists():
                for line in http.open():
                    request=json.loads(line);route=urlsplit(request['url']).path
                    counts[route]+=1;seconds[route]+=request['elapsed_seconds']
            receipts=[q['receipt'] for q in requests if q.get('receipt')]
            rows.append(dict(run=run.name,episode={k:v for k,v in episode.items() if k not in ('request_paths','actors')},
                new_requests=len(requests),request_usage=[dict(label=q['label'],state=q['state'],
                    usage=q.get('receipt',{}).get('usage'),cost_upper_usd=q.get('cost_upper_usd')) for q in requests],
                new_cost_upper_usd=sum(q.get('cost_upper_usd',0) for q in requests),actors=actors,
                lifecycle=dict(lifecycle),actual_call_names=dict(collections.Counter(actions.values())),
                belief_evidence_kinds=dict(belief_kinds),recovery_decisions=decisions,
                http_counts=dict(counts),http_seconds=dict(seconds),
                interpretation='Historical receipts are not new requests; event/graph counts are not physical success.'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as f:json.dump(summarize(a.root),f,indent=2,ensure_ascii=False)


if __name__=='__main__':main()
