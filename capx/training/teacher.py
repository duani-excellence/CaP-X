"""DeepSeek Teacher collection and text-chat SFT export; no training entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from capx.experiments.pilot import (
    CostLedger, DEFAULT_MAX_TOKENS, FATAL, check_services, digest, dump,
    load_scene, one_program, prepare, run_trial,
)

MODEL = 'deepseek-v4-pro'


def read(path):
    return json.loads(Path(path).read_text())


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def successful(result):
    info = result.get('info', {})
    return (result.get('status') == 'success' and result.get('reward') == 1
            and result.get('truncated') is False and info.get('physical_success') is True
            and info.get('sandbox_rc') == 0 and info.get('execution_status') == 'ok'
            and info.get('motion_calls', 0) > 0)


def split_group(snapshot_hash, fraction):
    value = int(hashlib.sha256(snapshot_hash.encode()).hexdigest()[:16],16)/2**64
    return 'validation' if value < fraction else 'train'


def training_example(row, target):
    """Export only teacher text/code; evaluator truth and reasoning stay in raw audit."""
    if row.get('source') not in ('teacher','transfer'):
        return None, 'not_teacher'
    if row.get('mode') != target:
        return None, 'wrong_observation_mode'
    try:
        artifacts = Path(row['artifacts'])
        result = read(artifacts/'result.json')
        if not successful(result):return None,'physical_or_execution_failure'
        if row.get('snapshot_hash') != result.get('snapshot_hash'):
            return None,'snapshot_mismatch'
        code = one_program((artifacts/'generated.py').read_text())
        prompt = read(artifacts/'prompt.json')
        if digest(prompt) != result.get('prompt_hash'):return None,'prompt_hash_mismatch'
        if row['source'] == 'teacher':
            response = read(artifacts/'model-response.json')
            if response.get('response_model') != MODEL:return None,'wrong_teacher_model'
            if response.get('finish_reason') != 'stop':return None,'incomplete_generation'
            if code != one_program(response.get('content')):return None,'code_mismatch'
        else:
            source = row.get('teacher_source')
            if not source or source.get('source') != 'teacher' or source.get('mode') != 'privileged':
                return None,'unverified_transfer_source'
            teacher, reason = training_example(source,'privileged')
            if reason:return None,'invalid_transfer_source:'+reason
            if source['snapshot_hash'] != row['snapshot_hash'] or teacher['messages'][-1]['content'] != code:
                return None,'transfer_pair_mismatch'
        messages=[]
        for message in prompt:
            if message.get('role') not in ('system','user'):return None,'unexpected_prompt_role'
            content=message.get('content')
            if isinstance(content,list):
                if any(p.get('type') != 'text' for p in content):return None,'non_text_prompt'
                content='\n'.join(p['text'] for p in content)
            if not isinstance(content,str):return None,'invalid_prompt'
            messages.append({'role':message['role'],'content':content})
        messages.append({'role':'assistant','content':code})
        return {'messages':messages},None
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        return None,'invalid_artifact:'+type(exc).__name__


def export_data(runs, output, target, validation_fraction):
    if not 0 <= validation_fraction < 1:raise ValueError('validation fraction must be in [0,1)')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    counts={'accepted':0,'rejected':0,'train':0,'validation':0,'duplicates':0}
    seen=set();groups=set()
    with (output/'train.jsonl').open('x') as train, (output/'validation.jsonl').open('x') as val, (output/'index.jsonl').open('x') as index, (output/'rejections.jsonl').open('x') as rejected:
        for run in runs:
            for line in (Path(run)/'records.jsonl').read_text().splitlines():
                row=json.loads(line);sample,reason=training_example(row,target)
                if reason:
                    counts['rejected']+=1
                    rejected.write(json.dumps({'id':row.get('id'),'reason':reason})+'\n');continue
                fingerprint=digest(sample)
                if fingerprint in seen:
                    counts['duplicates']+=1;continue
                seen.add(fingerprint);groups.add(row['snapshot_hash'])
                split=split_group(row['snapshot_hash'],validation_fraction)
                (train if split=='train' else val).write(json.dumps(sample,ensure_ascii=False)+'\n')
                index.write(json.dumps({'id':row['id'],'split':split,'snapshot_hash':row['snapshot_hash'],'source':row['source'],'mode':target,'sample_digest':fingerprint,'artifacts':row['artifacts']})+'\n')
                counts['accepted']+=1;counts[split]+=1
    dump(output/'manifest.json',dict(format='messages-jsonl',target=target,teacher=MODEL,counts=counts,snapshot_groups=len(groups),validation_fraction=validation_fraction,split_rule='SHA256(snapshot_hash); keep same physical scene in one split',warning='Empty/small validation splits are not a benchmark. Expand scenes before training.'))
    print(json.dumps(counts),flush=True)
    return counts


def copy_manifest(source, root, seeds=None):
    source=Path(source).resolve();document=read(source)
    entries=document['scenes']
    if seeds is not None:
        entries=[e for e in entries if e['seed'] in seeds]
        if {e['seed'] for e in entries} != set(seeds):raise ValueError('Requested seed missing from manifest')
    if not entries or len({e['seed'] for e in entries}) != len(entries):raise ValueError('Empty or duplicate seeds')
    rows=[]
    for entry in entries:
        snapshot=load_scene(source,entry)
        if snapshot.get('version') != 2:raise ValueError('A settled v2 Snapshot is required')
        target=root/'manifest'/'snapshots'/f'seed-{entry["seed"]}.json'
        dump(target,snapshot)
        rows.append({**entry,'snapshot':str(target.relative_to(root/'manifest'))})
    manifest=root/'manifest'/'scenes.json';dump(manifest,{'version':1,'scenes':rows})
    return manifest,rows


def make_record(sample_id, mode, source, artifacts):
    artifacts=Path(artifacts).resolve();r=read(artifacts/'result.json')
    return dict(id=sample_id,mode=mode,source=source,artifacts=str(artifacts),
                snapshot_hash=r.get('snapshot_hash'),seed=r.get('seed'),status=r['status'])


def worker(job_path):
    job=read(job_path);root=Path(job['root']);manifest=Path(job['manifest']);entry=job['entry'];mode=job['mode']
    if job['mock']:
        import capx.llm.client as client
        from capx.envs.tasks.franka.franka_pick_place import ORACLE_CODE
        # Mock never invokes a provider, and its records cannot be exported as Teacher labels.
        os.environ['DEEPSEEK_API_KEY']='offline-mock';os.environ['DEEPSEEK_MODEL']=MODEL
        client.query_model=lambda *a,**k:dict(content=ORACLE_CODE,response_model=MODEL,finish_reason='stop',usage={},attempts=0)
    ledger=CostLedger(limit=job['cost_limit'])
    control=run_trial(root,manifest,entry,mode,'oracle',ledger)
    if not successful(control):
        record=make_record(job['id'],mode,'oracle_gate_rejected',root/'oracle'/mode/f'seed-{entry["seed"]}')
        dump(root/'worker-records.json',[record]);return
    result=run_trial(root,manifest,entry,mode,'deepseek',ledger,max_tokens=job['max_tokens'],max_attempts=1)
    record=make_record(job['id'],mode,'mock' if job['mock'] else 'teacher',root/'deepseek'/mode/f'seed-{entry["seed"]}')
    records=[record]
    if job['distill_vision'] and mode=='privileged' and successful(result):
        # Build the Vision prompt/API contract from the same snapshot, never from teacher truth.
        run_trial(root,manifest,entry,'vision','oracle',ledger)
        program=(Path(record['artifacts'])/'generated.py').read_text()
        run_trial(root,manifest,entry,'vision','replay',ledger,replay_code=program)
        transfer=make_record(job['id']+'-vision-replay','vision','mock' if job['mock'] else 'transfer',root/'replay'/'vision'/f'seed-{entry["seed"]}')
        transfer['teacher_source']=record
        records.append(transfer)
    dump(root/'worker-records.json',records)


def collect(args):
    if args.max_requests < 1 or args.max_cost_usd <= 0 or args.max_tokens < 1 or args.worker_timeout <= 0:
        raise ValueError('Budgets must be positive')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    manifest,entries=copy_manifest(args.manifest,root,args.seeds)
    services=check_services(root)
    needed=['8116']+(['8114','8215'] if 'vision' in args.modes or args.distill_vision else [])
    if not all(services[p]['ok'] for p in needed):raise RuntimeError('Required local service unavailable; no model calls sent')
    jobs=[(e,m) for e in entries for m in dict.fromkeys(args.modes)]
    if len(jobs)>args.max_requests:raise ValueError('Requested samples exceed --max-requests; use fewer seeds/modes')
    if not args.mock:
        from capx.llm.providers import provider_config
        _,model,_=provider_config('deepseek')
        if model != MODEL:raise ValueError('DeepSeek model mismatch')
    dump(root/'collection.json',dict(version=1,model=MODEL,modes=args.modes,mock=args.mock,max_requests=args.max_requests,max_cost_usd=args.max_cost_usd,max_tokens=args.max_tokens,worker_timeout=args.worker_timeout,distill_vision=args.distill_vision,max_attempts=1,argv_note='Keys and private environment are never serialized'))
    records=root/'records.jsonl';records.touch(exist_ok=False)
    spent=0.;attempts=0;usage_complete=True
    for i,(entry,mode) in enumerate(jobs):
        sample_id=f'{i:04d}-{mode}-seed-{entry["seed"]}';sample=root/'samples'/sample_id;sample.mkdir(parents=True)
        job=dict(id=sample_id,root=str(sample),manifest=str(manifest),entry=entry,mode=mode,mock=args.mock,
                 max_tokens=args.max_tokens,cost_limit=args.max_cost_usd-spent,distill_vision=args.distill_vision)
        if job['cost_limit']<=0:break
        dump(sample/'job.json',job)
        with (sample/'worker.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B','-m','capx.training.teacher','_worker','--job',str(sample/'job.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:process.wait(timeout=args.worker_timeout)
            except KeyboardInterrupt:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
                append(records,dict(id=sample_id,mode=mode,source="interrupted",artifacts=str(sample),status="interrupted"))
                dump(root/"aborted.json",{"reason":"user_interrupt","sample":sample_id,"cost_known":False})
                raise
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                append(records,dict(id=sample_id,mode=mode,source='worker_timeout',artifacts=str(sample),status='timeout'))
                dump(root/'aborted.json',{'reason':'worker_timeout','sample':sample_id,'cost_known':False})
                break
        if process.returncode or not (sample/'worker-records.json').exists():
            append(records,dict(id=sample_id,mode=mode,source='worker_error',artifacts=str(sample),status='system_error'))
            dump(root/'aborted.json',{'reason':'worker_error','sample':sample_id,'returncode':process.returncode})
            break
        rows=read(sample/'worker-records.json')
        for row in rows:append(records,row)
        for path in (sample/'deepseek').rglob('http-attempt-*.json'):
            charge=read(path);spent+=charge['accounting_charge_usd'];attempts+=1
            usage_complete = usage_complete and not charge.get('charge_is_upper_bound', True)
        if any(r['status'] in FATAL for r in rows):
            dump(root/'aborted.json',{'reason':'fatal_trial_status','sample':sample_id,'statuses':[r['status'] for r in rows]})
            break
    dump(root/'summary.json',dict(http_attempts=attempts,accounting_cost_usd=spent,mock=args.mock,aborted=(root/'aborted.json').exists(),usage_complete=usage_complete and not (root/'aborted.json').exists(),notes='Accounting estimates/upper bounds, not billing. Inspect http-attempt files. Interrupted calls may have unknown usage.'))
    print(str(root),flush=True)


def import_pilot(source, output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    copy_manifest(source/'manifest/scenes.json',output)
    record_path=output/'records.jsonl';record_path.touch(exist_ok=False)
    for result_path in sorted((source/'deepseek').glob('*/seed-*/result.json')):
        result=read(result_path);mode=result['mode']
        if mode not in ('vision','privileged'):continue
        record=make_record('import-'+result['trial_id'],mode,'teacher',result_path.parent)
        append(record_path,record)
    dump(output/'import.json',{'source':str(source),'note':'Read-only references to existing pilot artifacts; keep original run available.'})
    print(str(output))


def run_pipeline(args):
    """Prepare, collect, and export using the existing stage implementations."""
    from datetime import datetime
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be nonempty and unique')
    if len(args.seeds)*len(set(args.modes)) > args.max_requests:
        raise ValueError('Increase --max-requests or reduce seeds/modes')
    if min(args.max_requests,args.max_cost_usd,args.max_tokens,args.worker_timeout) <= 0:
        raise ValueError('Budgets and timeout must be positive')
    if not 0 <= args.validation_fraction < 1:
        raise ValueError('validation-fraction must be in [0, 1)')
    root=Path(args.output).resolve() if args.output else (
        Path(__file__).resolve().parents[2]/'data/teacher_runs'/
        datetime.now().strftime('teacher_%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(root/'numba-cache')
    print(f'Teacher run: {root}',flush=True)
    scenes=root/'scenes';scenes.mkdir()
    manifest=scenes/'manifest/scenes.json'
    prepare(manifest,scenes,args.seeds)
    options=vars(args).copy()
    options.update(output=str(root/'raw'),manifest=str(manifest))
    collect(argparse.Namespace(**options))
    if (root/'raw/aborted.json').exists():
        raise RuntimeError(f'Collection aborted; inspect {root}/raw/aborted.json')
    counts={}
    for target in ('vision','privileged'):
        counts[target]=export_data([str(root/'raw')],root/'sft'/target,target,args.validation_fraction)
    dump(root/'summary.json',dict(collection=read(root/'raw/summary.json'),exports=counts))
    print(f'Completed: {root} (SFT: sft/vision, sft/privileged)',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('run',help='One command: snapshots, bounded collection, replay and SFT export')
    p.add_argument('--output',help='New output directory; default data/teacher_runs/teacher_TIMESTAMP')
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--modes',choices=['privileged','vision'],nargs='+',default=['privileged','vision'])
    p.add_argument('--max-requests',type=int,default=2)
    p.add_argument('--max-cost-usd',type=float,default=1.)
    p.add_argument('--max-tokens',type=int,default=DEFAULT_MAX_TOKENS)
    p.add_argument('--worker-timeout',type=float,default=360)
    p.add_argument('--no-distill-vision',dest='distill_vision',action='store_false')
    p.add_argument('--mock',action='store_true',help='No LLM calls; mock samples are excluded from SFT')
    p.add_argument('--validation-fraction',type=float,default=.2)
    from .campaign import add_arguments
    add_arguments(p)
    p=commands.add_parser('prepare',help='Create settled snapshots; no model API calls')
    p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,21])
    p=commands.add_parser('collect',help='Serial bounded DeepSeek collection; incurs API charges unless --mock')
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modes',choices=['privileged','vision'],nargs='+',default=['privileged'])
    p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--max-requests',type=int,default=2)
    p.add_argument('--max-cost-usd',type=float,default=1.);p.add_argument('--max-tokens',type=int,default=DEFAULT_MAX_TOKENS)
    p.add_argument('--worker-timeout',type=float,default=360);p.add_argument('--distill-vision',action='store_true')
    p.add_argument('--mock',action='store_true',help='Fixed Oracle code instead of LLM; never eligible for Teacher SFT export')
    p=commands.add_parser('export',help='Filter successful labels and export messages JSONL')
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--target',choices=['privileged','vision'],required=True)
    p.add_argument('--validation-fraction',type=float,default=.2)
    p=commands.add_parser('import-pilot',help='Index existing DeepSeek pilot results without modifying them')
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=commands.add_parser('_worker',help='Internal process entry; use collect instead');p.add_argument('--job',required=True)
    args=parser.parse_args()
    if args.command in ('run','prepare','collect','_worker'):
        os.environ['GRASPNET_SERVICE_URL']='http://127.0.0.1:8215'
    if args.command=='run':
        if args.target_success is not None:
            from .campaign import run
            run(args)
        elif args.resume:
            parser.error('--resume requires --target-success')
        else:
            if args.seeds is None:args.seeds=[0]
            run_pipeline(args)
    elif args.command=='prepare':
        root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
        prepare(root/'manifest/scenes.json',root,args.seeds)
        print(root/'manifest/scenes.json')
    elif args.command=='collect':collect(args)
    elif args.command=='export':export_data(args.runs,args.output,args.target,args.validation_fraction)
    elif args.command=='import-pilot':import_pilot(args.source,args.output)
    else:worker(args.job)


if __name__=='__main__':main()
