"""Read-only per-request provenance and exact duplicate accounting (no API calls).

V27 attribution reconstructs the ordered full/crop requests from trace events;
V28 uses explicit call context. Identical image/query is engineering redundancy,
not proof that a changed observation or an unsuccessful query was unnecessary.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(episode):
    events=[json.loads(l) for l in (episode/'trace.jsonl').read_text().splitlines()]
    attributed=[e for e in events if e['kind']=='perception_request' and not e['reused']]
    legacy=[];crops={};captures={};capture_id=None
    for e in events:
        if e['kind']=='public_capture':capture_id=e['capture'];captures[capture_id]=[]
        if capture_id is not None:captures[capture_id].append(e)
        if e['kind']=='public_region_crop':crops[(e['source'],e['label'],e['camera'])]=e['box']
        if e['kind']=='instance_query':
            key=(e['source'],e['label'],e['camera']);base=dict(label=e['label'],camera=e['camera'],source=e['source'],capture=capture_id)
            legacy.append(dict(base,crop=None))
            if key in crops:legacy.append(dict(base,crop=crops.pop(key)))
    files=sorted((episode/'tools').glob('sam3-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
    mapping=attributed if attributed else legacy
    if len(mapping)!=len(files):raise ValueError(f'Tool/trace mismatch: {episode}: {len(mapping)} != {len(files)}')
    seen={};rows=[];cache_hits=[e for e in events if e['kind']=='perception_request' and e['reused']]
    for file,meta in zip(files,mapping):
        record=json.loads(file.read_text());bundle=np.load(file.with_suffix('.npz'))
        image=bundle[record['args'][0]['npz_key']];label=record['kwargs']['text_prompt']
        if label!=meta['label']:raise ValueError('Request label order drift')
        ih=hashlib.sha256(image.tobytes()).hexdigest();key=(ih,str(image.shape),label)
        h=hashlib.sha256()
        for det in record['response'] or []:h.update(bundle[det['mask']['npz_key']].tobytes())
        rh=h.hexdigest();prior=seen.get(key);context=record.get('context') or meta.get('context')
        cap=int(Path(meta['source']).name.split('-')[0]);frame=captures[cap]
        components=[x['component'] for x in context or []]
        if any('current_held' in c for c in components):trigger='held_surface_validation'
        elif any('_acquire_plan' in c or '_compute_acquire_plan' in c for c in components):trigger='grasp_plan_and_obstacles'
        elif any('verify' in c for c in components):trigger='contract_verification'
        elif any('deposit' in c for c in components):trigger='deposit_geometry'
        elif any(e['kind']=='skill_tracking_source' for e in frame):trigger='held_surface_validation_inferred'
        elif any(e['kind']=='model_contract' and e.get('event')=='planned_evidence' for e in frame):trigger='recovery_evidence_inferred'
        elif any(e['kind']=='skill_grasp_plan' for e in frame):trigger='grasp_plan_and_obstacles_inferred'
        else:trigger='geometry_or_admission_inferred'
        # Compare actual request pixels to the archived public image/crop.
        obs=np.load(episode/'observations'/f'{cap:05d}-{meta["camera"]}.npz')['rgb']
        if meta['crop']:
            (x0,y0),(x1,y1)=meta['crop'];obs=obs[y0:y1,x0:x1]
        if not np.array_equal(obs,image):raise ValueError('Request image is not recorded public observation/crop')
        row=dict(tool_file=str(file),trigger=trigger,attribution='explicit_context' if context else 'ordered_trace_inference',
            capture=cap,camera=meta['camera'],label=label,crop=meta['crop'],image_hash=ih,response_mask_hash=rh,
            seconds=record['elapsed_seconds'],duplicate_of=prior['tool_file'] if prior else None,
            conflicting_repeat=bool(prior and prior['response_mask_hash']!=rh),context=context,
            raw_result_count=len(record['response'] or []),source=meta['source'])
        rows.append(row);seen.setdefault(key,row)
    http=episode.parent/(episode.name+'-http.jsonl');http_cost=collections.defaultdict(lambda:dict(count=0,seconds=0.))
    if http.exists():
        for line in http.open():
            row=json.loads(line);endpoint=row['url'].rsplit('/',1)[-1]
            if endpoint in ('segment','grasp','graspnet','ik','predict'):
                http_cost[endpoint]['count']+=1;http_cost[endpoint]['seconds']+=row['elapsed_seconds']
    result=json.loads((episode/'result.json').read_text())
    return dict(episode=str(episode),requests=rows,
        sam3_requests=len(rows),sam3_seconds=sum(x['seconds'] for x in rows),
        exact_duplicates=sum(x['duplicate_of'] is not None for x in rows),
        duplicate_seconds=sum(x['seconds'] for x in rows if x['duplicate_of']),
        conflicting_repeats=sum(x['conflicting_repeat'] for x in rows),
        by_label=dict(collections.Counter(x['label'] for x in rows)),
        by_trigger=dict(collections.Counter(x['trigger'] for x in rows)),
        crop_requests=sum(x['crop'] is not None for x in rows),
        full_requests=sum(x['crop'] is None for x in rows),
        semantic_reuses=len(cache_hits),cross_capture_reuses=sum(x['origin']!=x['source'] for x in cache_hits),
        geometry_reuses=sum(e['kind']=='perception_geometry_reuse' for e in events),
        plan_reuses=sum(e['kind']=='perception_plan_reuse' and e['reused'] for e in events),
        failed_alias_cascades=sum(e['kind']=='perception_binding' and e['selected'] is None for e in events),
        capture_count=sum(e['kind']=='public_capture' for e in events),http=dict(http_cost),
        capture_and_archive_seconds=sum(e.get('capture_and_archive_seconds',0.) for e in events if e['kind']=='perception_frame'),
        capture_timing_available=any('capture_and_archive_seconds' in e for e in events if e['kind']=='perception_frame'),
        observed_condition_changes=sum(sum(e['before'][k]!=v for k,v in e['after'].items()) for e in events if e['kind']=='perception_verification'),
        result={k:result.get(k) for k in ('raw_success','native_progress','api_calls','captures','llm_requests','end_to_end_seconds','stop_reason','initial_code_hash','snapshot_hash')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('unit',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    episodes=[summarize(f.parent) for f in sorted(a.unit.glob('*/result.json'))]
    with a.output.open('x') as f:json.dump(dict(episodes=episodes),f,indent=2)
    for e in episodes:print(Path(e['episode']).name,{k:e[k] for k in ('sam3_requests','exact_duplicates','semantic_reuses','cross_capture_reuses','plan_reuses','capture_count')})


if __name__=='__main__':main()
