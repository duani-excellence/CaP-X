"""Immutable normal-to-ablation initial-program references, with no provider calls.

Only completed fixed allocations are exported. Invalid programs and known API
failures remain entries; unknown completions never trigger an automatic requery.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

SCHEMA='capx.shared_initial.v1'


def read(path):return json.loads(Path(path).read_text())
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(row):return row['task'],row['seed'],row['repeat'],row.get('fault','none')


def _cached(cache,kind,path,loader):
    # Cache lifetime is one validation/export/run invocation, never disk or a
    # global. Selected execution bytes are rechecked by the runner archive guard.
    if cache is None:return loader()
    key=(kind,str(Path(path).resolve()))
    if key not in cache:cache[key]=loader()
    return cache[key]


def _read(path,cache=None):return _cached(cache,'json',path,lambda:read(path))
def _hash(path,cache=None):return _cached(cache,'hash',path,lambda:file_hash(path))


def _check_source_integrity(rows):
    if any(r.get('status')=='source_changed' or r.get('source_changes') or
           (r.get('source_postcheck') or {}).get('unchanged') is False for r in rows):
        raise ValueError('Shared source integrity failed during execution; restored files do not erase that history')


def selected_reference_files(entry):
    """Per-unit payload; shared global metadata lives in the campaign archive."""
    root=Path(entry['source_campaign']).resolve()
    shared={str(root/name) for name in ('frozen.json','state.json','request-ledger.json')}
    return {p:h for p,h in entry.get('files',{}).items() if str(Path(p).resolve()) not in shared}



def reference_files(manifest):
    """Return only already-declared immutable reference files for source archiving."""
    body=read(manifest)
    if body.get('schema')!=SCHEMA:raise ValueError('Unknown initial manifest schema')
    result={str(Path(manifest).resolve())}
    for entry in body['entries']:result.update(entry.get('files',{}))
    return sorted(result)


def _source(entry,config=None,cache=None):
    from capx.experiments.graph_campaign import load_campaign,compatible_development,declared_sources,service_identity,validate_checkpoint
    for path,digest in entry.get('files',{}).items():
        if _hash(path,cache)!=digest:raise ValueError('Shared initial file drift: '+path)
    source=_cached(cache,'campaign',entry['source_campaign'],lambda:load_campaign(entry['source_campaign']))
    if source['signature']!=entry['source_signature']:raise ValueError('Shared initial source signature drift')
    if entry.get('source_phase')!=source['phase']:raise ValueError('Shared initial source phase mismatch')
    state=_read(Path(entry['source_campaign'])/'state.json',cache)
    _cached(cache,'checkpoint',entry['source_campaign'],lambda:validate_checkpoint(source,state))
    index=_cached(cache,'units',entry['source_campaign'],lambda:{u['id']:u for u in state['units']})
    units=[index[entry['source_unit_id']]] if entry['source_unit_id'] in index else []
    if len(units)!=1 or units[0]['state']!='done' or identity(units[0])!=identity(entry):raise ValueError('Shared source unit identity is not completed')
    if not entry.get('available'):
        _check_source_integrity(read(units[0]['result'])['episodes'])
    if entry.get('available'):
        folder=Path(units[0]['result']).resolve().parent
        if Path(entry['result_file']).resolve()!=Path(units[0]['result']).resolve() or Path(entry['initial_file']).resolve()!=folder/'initial'/'G0-G1.json':raise ValueError('Shared source artifact location mismatch')
        label=entry['source_unit_id']+f":initial-{entry['task']}-seed-{entry['seed']}-repeat-{entry['repeat']}-shared-G0-G1"
        if entry['request_label']!=label:raise ValueError('Shared source request label mismatch')
    c=source['config']
    if c.get('study','normal')!='normal' or c.get('faults',['none'])!=['none'] or not all(c.get('graph_options',{}).values()):
        raise ValueError('Shared initial requires a full normal source')
    if config is not None:
        if config.get('study')!='mechanism_ablation':raise ValueError('Shared normal initial is for declared mechanism ablations only')
        if not compatible_development(config,c):raise ValueError('Initial reuse requires a declared graph ablation')
        if declared_sources(config)!=declared_sources(c):raise ValueError('Shared initial business source set differs')
        if bool(config.get('service_inventory'))!=bool(c.get('service_inventory')):raise ValueError('Shared initial service identity missing')
        if c.get('service_inventory'):
            old=_read(c['service_inventory'],cache);new=_read(config['service_inventory'],cache)
            if {k:service_identity(v) for k,v in old['services'].items()}!={k:service_identity(v) for k,v in new['services'].items()}:raise ValueError('Shared initial service identity differs')
    return source


def _contents(entry,source,cache=None):
    """Verify the existing provider output/receipt chain, including failure tombstones."""
    from capx.experiments.graph_campaign import fingerprint
    from capx.experiments.graph_v2 import digest,validated_program
    if entry.get('available') is not True:raise ValueError('Source initial unavailable: '+str(entry.get('reason')))
    directory=Path(entry['request_directory'])
    required=[Path(entry[k]) for k in ('initial_file','snapshot_file','api_docs_file','result_file')]+[Path(entry['source_campaign'])/n for n in ('frozen.json','state.json','request-ledger.json')]+[directory/n for n in ('prompt.json','receipt.json','http.json')]
    if (directory/'response.json').exists():required.append(directory/'response.json')
    if not {str(p.resolve()) for p in required}<=set(entry.get('files',{})):raise ValueError('Unhashed shared initial reference file')
    initial=read(entry['initial_file']);snapshot=read(entry['snapshot_file']);config=source['config']
    if snapshot.get('seed')!=entry['seed'] or snapshot.get('task',entry['task'])!=entry['task'] or not snapshot.get('model_hash'):
        raise ValueError('Shared initial Snapshot identity missing')
    if snapshot.get('state_hash')!=initial.get('snapshot_hash'):raise ValueError('Shared initial Snapshot hash drift')
    ledger=_read(Path(entry['source_campaign'])/'request-ledger.json',cache)
    matches=[r for r in ledger['requests'] if r['label']==entry['request_label']]
    if len(matches)!=1:raise ValueError('Shared initial ledger reference is not unique')
    request=matches[0]
    if request['state'] not in ('received','failed'):raise ValueError('Shared initial request completion uncertain')
    if sorted(p.resolve() for p in Path(request['directory']).glob('llm/*/receipt.json'))!=[(directory/'receipt.json').resolve()]:raise ValueError('Shared request directory differs from ledger')
    receipt=read(directory/'receipt.json');prompt=read(directory/'prompt.json')
    payload=read(directory/'http.json')['payload']
    if receipt!=request.get('receipt') or request['prompt_hash']!=fingerprint(prompt):raise ValueError('Shared initial receipt/prompt ledger mismatch')
    if receipt.get('attempts')!=1 or receipt.get('mock') or receipt.get('requested_model')!=config['model']:
        raise ValueError('Source is not a recorded single DeepSeek request')
    for key,expected in [('model',config['model']),('max_tokens',config['budgets']['max_tokens']),('reasoning_effort',config['budgets'].get('reasoning_effort','low'))]:
        if payload.get(key)!=expected:raise ValueError('Shared initial request parameter mismatch: '+key)
    if payload.get('messages')!=prompt or digest(prompt)!=initial.get('initial_prompt_hash'):
        raise ValueError('Shared initial prompt bytes mismatch')
    docs=read(entry['api_docs_file'])['documentation']
    if digest(docs)!=initial.get('api_docs_hash'):raise ValueError('Shared initial API documentation mismatch')
    if initial.get('request_count')!=1 or initial.get('code_hash')!=digest(initial.get('code')):
        raise ValueError('Shared initial code/count mismatch')
    rows=read(entry['result_file'])['episodes'];_check_source_integrity(rows)
    paired={r['group']:r for r in rows if r['group'] in ('G0','G1')}
    if set(paired)!={'G0','G1'}:raise ValueError('Shared source graph pair incomplete')
    for row in paired.values():
        if identity(row)!=identity(entry) or row.get('mock'):raise ValueError('Shared initial source identity mismatch')
        for key in ('snapshot_hash','initial_prompt_hash','api_docs_hash','code_hash'):
            if row.get(key)!=initial.get(key):raise ValueError('Shared source pair hash mismatch: '+key)
    if request['state']=='received':
        response=read(directory/'response.json')
        if initial.get('response')!=response or receipt.get('http_status')!=200 or receipt.get('response_model')!=config['model'] or response.get('response_model')!=config['model']:
            raise ValueError('Shared initial response/model mismatch')
        if response.get('response_id')!=receipt.get('response_id'):raise ValueError('Shared initial response receipt identity differs')
        try:code=validated_program(response)
        except (ValueError,SyntaxError):
            if initial.get('code') is not None or initial.get('error_status')!='invalid_program':raise ValueError('Invalid source program failure not retained')
        else:
            if code!=initial.get('code'):raise ValueError('Shared initial parsed program differs')
    elif initial.get('code') is not None or initial.get('response') is not None or not initial.get('error_status'):
        raise ValueError('Failed source request has inconsistent initial result')
    return initial


def export_initial_manifest(source_root,output):
    from capx.experiments.graph_campaign import load_campaign,validate_checkpoint,fingerprint,_validate_result
    root=Path(source_root).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    frozen=load_campaign(root);state=read(root/'state.json');validate_checkpoint(frozen,state)
    if frozen['config'].get('study','normal')!='normal' or frozen['config'].get('faults',['none'])!=['none']:
        raise ValueError('Initial export requires a normal campaign')
    if not all(u['state']=='done' for u in state['units']):raise ValueError('Export requires the complete fixed allocation; no outcome selection')
    ledger_path=root/'request-ledger.json';ledger=read(ledger_path) if ledger_path.exists() else {'requests':[]}
    if any(r['state'] in ('started','uncertain') for r in ledger['requests']):raise ValueError('Source request uncertain; audit before export')
    entries=[];cache={}
    for unit in state['units']:
        result=Path(unit['result']).resolve();rows=_validate_result(read(result),unit);_check_source_integrity(rows)
        if set(unit['groups'])!={'B0','G0','G1'}:raise ValueError('Source allocation lacks normal three-system comparison')
        folder=result.parent;initial=folder/'initial'/'G0-G1.json'
        common=[root/'frozen.json',root/'state.json',result]
        entry=dict(task=unit['task'],seed=unit['seed'],repeat=unit['repeat'],fault=unit['fault'],source_campaign=str(root),source_phase=frozen['phase'],
            source_signature=frozen['signature'],source_unit_id=unit['id'],available=initial.exists())
        if not initial.exists():
            entry.update(reason='normal source never produced an initialized graph request',files={str(p):_hash(p,cache) for p in common});entries.append(entry);continue
        label=unit['id']+f":initial-{unit['task']}-seed-{unit['seed']}-repeat-{unit['repeat']}-shared-G0-G1"
        requests=[r for r in ledger['requests'] if r['label']==label]
        if len(requests)!=1:raise ValueError('Expected one attempted initial request per source unit')
        request=requests[0];receipt_paths=list(Path(request['directory']).glob('llm/*/receipt.json'))
        if len(receipt_paths)!=1:raise ValueError('Missing or ambiguous source receipt')
        directory=receipt_paths[0].parent.resolve();snapshot=folder/'scene'/'snapshot.json'
        if not snapshot.exists():
            scene_manifest=folder/'scene'/'scenes.json';record=read(scene_manifest)['scenes'][0];snapshot=scene_manifest.parent/record['snapshot'];common.append(scene_manifest)
        docs=folder/'episodes'/'G0'/'api-docs.json'
        files=common+[initial,snapshot,docs,ledger_path]+[directory/name for name in ('prompt.json','receipt.json','http.json')]
        if (directory/'response.json').exists():files.append(directory/'response.json')
        entry.update(initial_file=str(initial.resolve()),snapshot_file=str(snapshot.resolve()),api_docs_file=str(docs.resolve()),result_file=str(result),
            request_directory=str(directory),request_label=label,source_request_uid=fingerprint({'campaign':frozen['signature'],'label':label,'prompt_hash':request['prompt_hash']}),
            files={str(p.resolve()):_hash(p,cache) for p in files})
        _contents(entry,frozen,cache);entries.append(entry)
    body=dict(schema=SCHEMA,source_campaign=str(root),source_signature=frozen['signature'],source_phase=frozen['phase'],
        entries=entries,selection='all fixed allocation units including failed programs and known API failures; no reward filtering')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(body,stream,indent=2,allow_nan=False)
    return dict(manifest=str(output.resolve()),entries=len(entries),unavailable=sum(not e['available'] for e in entries),model_requests=0)


def validate_initial_manifest(path,config,units=None,*,phase=None,plan=None,_cache=None):
    from capx.experiments.graph_campaign import planned_units,fingerprint
    cache={} if _cache is None else _cache
    body=read(path)
    if body.get('schema')!=SCHEMA:raise ValueError('Unknown initial manifest schema')
    entries=body['entries'];index={identity(row):row for row in entries}
    if len(index)!=len(entries):raise ValueError('Duplicate initial manifest identity')
    if not entries:raise ValueError('Empty initial manifest allocation')
    if phase=='formal' and body.get('source_phase')!='formal':raise ValueError('Formal ablations require formal normal initials; development inputs forbidden')
    source=_source(entries[0],config,cache)
    if plan is not None and fingerprint(plan)!=fingerprint(source['plan']):
        raise ValueError('Shared initial target must retain the complete source plan')
    full={identity(u) for u in planned_units(source['config'],source['plan'],source['phase'])}
    if set(index)!=full:raise ValueError('Shared initial manifest must preserve exact complete source allocation')
    target={identity(u) for u in planned_units(config,source['plan'],phase or source['phase'])}
    if units is not None and ({identity(u) for u in units}!=target or len(units)!=len(target)):
        raise ValueError('Shared initial target allocation differs from the predeclared complete plan/subset')
    for entry in entries:
        if entry.get('fault')!='none':raise ValueError('Normal initial reuse requires none fault')
        checked=_source(entry,config,cache)
        if checked['signature']!=body['source_signature'] or checked['phase']!=body['source_phase'] or str(Path(entry['source_campaign']).resolve())!=str(Path(body['source_campaign']).resolve()):raise ValueError('Shared initial manifest source mismatch')
        if phase=='formal' and checked['phase']!='formal':raise ValueError('Formal source required')
        # Full source allocation remains in the manifest, including unavailable
        # attempts outside the predeclared subset. Selected unavailable inputs
        # fail closed; their absence never silently triggers a new paid request.
        if entry.get('available') or identity(entry) in target:_contents(entry,checked,cache)
    return {key:index[key] for key in index if key in target}


def load_initial_reference(entry,config,*,task,seed,repeat,fault,snapshot,_cache=None):
    if identity(entry)!=(task,seed,repeat,fault) or fault!='none':raise ValueError('Shared initial identity/fault mismatch')
    cache={} if _cache is None else _cache
    source=_source(entry,config,cache)
    if source['phase']=='formal':
        from capx.experiments.graph_campaign import planned_units
        if identity(entry) not in {identity(u) for u in planned_units(config,source['plan'],'formal')}:
            raise ValueError('Shared initial is outside the predeclared formal subset')
    initial=copy.deepcopy(_contents(entry,source,cache));saved=_read(entry['snapshot_file'],cache)
    if snapshot.get('state_hash')!=saved['state_hash'] or snapshot.get('model_hash')!=saved['model_hash']:
        raise ValueError('Shared initial target Snapshot mismatch')
    # Retain the canonical receipt location: billing deduplicates it across roots,
    # including failed requests without a provider response id.
    initial['request']=entry['request_directory']
    initial['cache_source']=dict(source_campaign=entry['source_campaign'],source_unit_id=entry['source_unit_id'],
        source_signature=entry['source_signature'],source_request_uid=entry['source_request_uid'],initial_file=entry['initial_file'],
        generated_in_this_run=False,initial_requests_this_run=0)
    return initial
