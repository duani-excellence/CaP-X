"""Multi-task Privileged entry point. Existing Stack pipeline is reused unchanged."""
import argparse,functools,json,os,signal,subprocess,sys,time
from pathlib import Path
from capx.multitask.catalog import TASKS,ROOT,catalog,require_ready


def process(command,log,timeout):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:return p.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(p.pid,signal.SIGINT)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise


def collect_episode(task,args):
    from capx.training.teacher import read,dump,append
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    job=dict(manifest=str(Path(args.manifest).resolve()),max_tokens=args.max_tokens,cost_limit=args.max_cost_usd,mock=args.mock)
    dump(root/'job.json',job)
    rc=process([sys.executable,'-B','-m','capx.multitask.worker','episode','--task',task,'--root',str(root.resolve()),'--job',str((root/'job.json').resolve())],root/'worker.log',args.worker_timeout)
    rows=read(root/'worker-records.json') if (root/'worker-records.json').exists() else []
    control=read(root/'control-result.json') if (root/'control-result.json').exists() else {}
    records=root/'records.jsonl';records.touch()
    for row in rows:append(records,row)
    if not rows:append(records,dict(id=task,mode='privileged',source='oracle_gate_rejected',status=control.get('status','system_error'),task=task,artifacts=str(root)))
    receipts=[read(p) for p in root.glob('**/http-attempt-*.json')]
    fatal=bool(rc or not control or control.get('status') in ('system_error','timeout','api_error') or any(r['status'] in ('system_error','timeout','api_error') for r in rows))
    dump(root/'summary.json',dict(aborted=fatal,http_attempts=len(receipts),accounting_cost_usd=sum(r['accounting_charge_usd'] for r in receipts),usage_complete=not fatal))
    if fatal:dump(root/'aborted.json',dict(reason='task_worker_failure',returncode=rc))


def task_campaign(task,options):
    from capx.training import campaign
    # Explicit backend injection inside this dedicated task process, never the Web or Stack process.
    if task!='stack':
        from capx.multitask.worker import prepare
        campaign.prepare=functools.partial(prepare,task)
        campaign.collect=functools.partial(collect_episode,task)
    campaign.run(argparse.Namespace(**options))


def parse_targets(items):
    targets={}
    for item in items:
        name,value=item.split('=',1)
        if name not in TASKS or name in targets or int(value)<1:raise ValueError('Use unique task=positive_integer targets')
        targets[name]=int(value)
    return targets


def export(root,output):
    from capx.training.teacher import training_example,split_group,digest,dump
    root=Path(root).resolve();config=json.loads((root/'multitask.json').read_text());output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    groups={}
    for task in config['targets']:
        views=sorted((root/task/'exports').glob('*/records.jsonl'))
        rows=[json.loads(l) for l in views[-1].read_text().splitlines()] if views else []
        valid=[];seen=set()
        for row in rows:
            sample,reason=training_example(row,'privileged')
            if reason:continue
            key=digest(sample)
            if key in seen:continue
            seen.add(key);valid.append((row,sample,key))
        groups[task]=valid
    cap=min(map(len,groups.values()),default=0)
    counts={task:dict(available=len(rows),exported=cap) for task,rows in groups.items()}
    with (output/'train.jsonl').open('x') as train,(output/'validation.jsonl').open('x') as validation,(output/'index.jsonl').open('x') as index:
        # Round-robin equal task quotas; Stack never dominates the combined view.
        for i in range(cap):
            for task,rows in groups.items():
                row,sample,key=rows[i];split=split_group(task+':'+row['snapshot_hash'],.2)
                (train if split=='train' else validation).write(json.dumps(sample,ensure_ascii=False)+'\n')
                index.write(json.dumps(dict(task=task,source=row['source'],mode='privileged',snapshot_hash=row['snapshot_hash'],split=split,sample_digest=key,artifacts=row['artifacts']))+'\n')
    dump(output/'manifest.json',dict(counts=counts,balanced_cap=cap,format='messages-jsonl',task_metadata='index.jsonl',note='Zero for any selected task yields empty balanced export; task-local exports retain all samples.'))
    print(json.dumps(counts),flush=True)


def main():
    signal.signal(signal.SIGTERM,lambda *unused: (_ for _ in ()).throw(KeyboardInterrupt()))
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='action',required=True)
    sp.add_parser('list')
    q=sp.add_parser('collect');q.add_argument('--targets',nargs='+',required=True);q.add_argument('--output',required=True);q.add_argument('--resume',action='store_true')
    q.add_argument('--mock',action='store_true',help='Fixed Oracle code; never exported as Teacher SFT')
    q.add_argument('--max-requests-per-task',type=int,default=10);q.add_argument('--max-cost-usd-per-task',type=float,default=2.)
    q.add_argument('--max-scenes',type=int,default=1000);q.add_argument('--start-seed',type=int,default=0);q.add_argument('--max-hours',type=float,default=4.);q.add_argument('--max-tokens',type=int,default=32768)
    q=sp.add_parser('infer');q.add_argument('--tasks',nargs='+',choices=TASKS,required=True);q.add_argument('--output',required=True);q.add_argument('--max-tokens',type=int,default=32768)
    q=sp.add_parser('oracle');q.add_argument('--tasks',nargs='+',choices=TASKS,required=True);q.add_argument('--seed',type=int,default=0);q.add_argument('--output',required=True)
    q=sp.add_parser('export');q.add_argument('--run',required=True);q.add_argument('--output',required=True)
    q=sp.add_parser('_campaign');q.add_argument('--task',required=True);q.add_argument('--job',required=True)
    a=p.parse_args()
    if a.action=='list':print(json.dumps(catalog(),ensure_ascii=False,indent=2));return
    if a.action=='export':export(a.run,a.output);return
    if a.action=='_campaign':task_campaign(a.task,json.loads(Path(a.job).read_text()));return
    if a.action=='infer':
        a.action='collect';a.targets=[t+'=1' for t in a.tasks];a.resume=False;a.max_requests_per_task=1;a.max_cost_usd_per_task=1.;a.start_seed=0;a.max_scenes=1;a.max_hours=1.;a.mock=False
    root=Path(a.output).resolve()
    if a.action=='oracle':
        root.mkdir(parents=True,exist_ok=False)
        for task in a.tasks:
            try:
                if task=='stack':
                    rc=process([sys.executable,'-B','-m','capx.multitask.stack_oracle','--root',str(root/task),'--seed',str(a.seed)],root/f'{task}.log',300)
                else:rc=process([sys.executable,'-B','-m','capx.multitask.worker','accept','--task',task,'--root',str(root/task),'--seed',str(a.seed)],root/f'{task}.log',300)
            except subprocess.TimeoutExpired:
                rc=124
            print(task,'worker_exit',rc,flush=True)
        return
    targets=parse_targets(a.targets)
    for task in targets:require_ready(task)
    from capx.training.teacher import dump,read,digest
    # Pin all new task code without changing legacy Stack checkpoint format.
    revision=digest({f.name:f.read_text() for f in Path(__file__).parent.glob('*.py')})
    config=dict(targets=targets,mock=a.mock,max_requests=a.max_requests_per_task,max_cost_usd=a.max_cost_usd_per_task,start_seed=a.start_seed,max_scenes=a.max_scenes,max_tokens=a.max_tokens,revision=revision)
    if a.resume:
        if read(root/'multitask.json')!=config:raise ValueError('Multi-task configuration/code mismatch')
    else:root.mkdir(parents=True,exist_ok=False);dump(root/'multitask.json',config)
    import fcntl
    with (root/'multitask.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes={}
        for task,target in targets.items():
            output=root/task;resume=(output/'state.json').exists()
            options=dict(output=str(output),resume=resume,target_success=target,target_modes=['privileged'],modes=['privileged'],start_seed=a.start_seed,max_scenes=a.max_scenes,max_requests=a.max_requests_per_task,max_cost_usd=a.max_cost_usd_per_task,request_cost_cap=1.,retries=1,max_hours=a.max_hours,min_free_gb=5.,max_per_template=20,max_tokens=a.max_tokens,worker_timeout=360,distill_vision=False,mock=a.mock,validation_fraction=.2)
            stamp=str(time.time_ns());job=root/'jobs'/f'{task}-{stamp}.json';dump(job,options)
            try:rc=process([sys.executable,'-B','-m','capx.multitask.cli','_campaign','--task',task,'--job',str(job)],root/'logs'/f'{task}-{stamp}.log',a.max_hours*3600+400)
            except subprocess.TimeoutExpired:rc=124
            outcomes[task]=dict(exit=rc,state=str(output/'state.json'));print(task,outcomes[task],flush=True)
        dump(root/'invocations'/f'{time.time_ns()}.json',outcomes)
        export(root,root/'combined'/str(time.time_ns()))

if __name__=='__main__':main()
