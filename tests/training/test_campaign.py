from argparse import Namespace
from pathlib import Path
import pytest
from capx.training import teacher


def args(tmp_path, **kwargs):
    values=dict(output=str(tmp_path/'campaign'),resume=False,target_success=2,target_modes=['vision'],
                modes=['vision'],start_seed=0,max_scenes=10,max_requests=2,max_cost_usd=10.,
                request_cost_cap=1.,retries=1,max_hours=1.,min_free_gb=0.,
                max_per_template=1,max_tokens=32768,worker_timeout=360,distill_vision=False,
                mock=True,validation_fraction=.2)
    values.update(kwargs)
    return Namespace(**values)


def fake_runtime(monkeypatch, fail=False):
    from capx.training import campaign as c
    calls=[]
    def prepare(manifest,root,seeds):
        teacher.dump(manifest,{'scenes':[{'seed':seeds[0]}]})
    def collect(a):
        calls.append(a.seeds[0]);root=Path(a.output);root.mkdir(parents=True)
        teacher.dump(root/'summary.json',{'http_attempts':0,'accounting_cost_usd':0,'aborted':False})
        teacher.append(root/'records.jsonl',{'id':str(root),'seed':a.seeds[0],'mode':'vision',
            'source':'mock','status':'task_failure' if fail else 'success','snapshot_hash':str(a.seeds[0]),'artifacts':str(root)})
    monkeypatch.setattr(c,'prepare',prepare);monkeypatch.setattr(c,'collect',collect)
    return calls


def test_resume_no_duplicate_calls(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    a=args(tmp_path,max_requests=4,min_free_gb=1.)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:Namespace(free=0 if len(calls)>=2 else 10**12))
    assert c.run(a)['stop_reason']=='disk_limit'
    assert calls==[0,1]
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:Namespace(free=10**12))
    a.resume=True;c.run(a)
    assert calls==[0,1,2,3]
    state=teacher.read(Path(a.output)/'state.json')
    assert state['stats']['accepted']['vision']==0 # mock must never meet target


def test_retry_is_bounded(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch,True)
    c.run(args(tmp_path,max_requests=4))
    assert calls==[0,0,1,1]


def test_unknown_attempt_blocks_resume(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    a=args(tmp_path);c.run(a)
    p=Path(a.output)/'state.json';s=teacher.read(p)
    s['attempts'][-1]['state']='started';s['attempts'][-1]['raw']=str(tmp_path/'missing')
    c.atomic(p,s);a.resume=True
    with pytest.raises(RuntimeError,match='uncertain'):c.run(a)
    assert len(calls)==2


def test_template_normalizes_constants_and_variables():
    from capx.training.campaign import template_hash
    assert template_hash('x = 1\ngoto_pose(x)')==template_hash('y = 2\ngoto_pose(y)')
    assert template_hash('open_gripper()')!=template_hash('close_gripper()')


def test_resume_configuration_cannot_change(tmp_path,monkeypatch):
    from capx.training import campaign as c
    fake_runtime(monkeypatch);a=args(tmp_path);c.run(a)
    a.resume=True;a.mock=False
    with pytest.raises(ValueError,match='configuration'):c.run(a)


def test_target_stops_without_extra_request(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    monkeypatch.setattr(c,'training_example',lambda row,mode:({'messages':[{'role':'assistant','content':f'fn_{row["seed"]}()'}]},None))
    monkeypatch.setattr(c,'export_data',lambda *a:None)
    state=c.run(args(tmp_path,max_requests=10))
    assert calls==[0,1]
    assert state['stop_reason']=='target_reached'
    assert state['stats']['accepted']['vision']==2


def test_template_quota_does_not_count_repeated_success(tmp_path,monkeypatch):
    from capx.training import campaign as c
    fake_runtime(monkeypatch)
    monkeypatch.setattr(c,'training_example',lambda row,mode:({'messages':[{'role':'assistant','content':f'x = {row["seed"]}\ngoto_pose(x)'}]},None))
    monkeypatch.setattr(c,'export_data',lambda *a:None)
    state=c.run(args(tmp_path,max_requests=4))
    assert state['stats']['accepted']['vision']==1
    assert state['stats']['rejections']['template_quota']==3


def test_request_ceiling_persists_across_resumes(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    a=args(tmp_path,max_requests=3);c.run(a);a.resume=True
    state=c.run(a)
    assert calls==[0,1,2]
    assert state['stop_reason']=='request_limit'


def test_cost_reservation_prevents_call(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    state=c.run(args(tmp_path,max_cost_usd=.5))
    assert calls==[] and state['stop_reason']=='cost_limit'


def test_completed_crash_window_reconciles_without_retry(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    a=args(tmp_path,min_free_gb=1.)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:Namespace(free=0 if calls else 10**12))
    c.run(a)
    p=Path(a.output)/'state.json';state=teacher.read(p)
    state['attempts'][0]['state']='started';state['scene_index']=0
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:Namespace(free=10**12))
    c.atomic(p,state);a.resume=True;c.run(a)
    assert calls==[0,1]


def test_lock_prevents_concurrent_collection(tmp_path,monkeypatch):
    import fcntl
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch);a=args(tmp_path);c.run(a);a.resume=True
    with (Path(a.output)/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already running'):c.run(a)
    assert calls==[0,1]


def test_native_transfer_same_scene_counts_once(tmp_path,monkeypatch):
    from capx.training import campaign as c
    raw=tmp_path/'raw';raw.mkdir()
    rows=[dict(id='native',mode='vision',source='teacher',status='success',snapshot_hash='same'),
          dict(id='transfer',mode='vision',source='transfer',status='success',snapshot_hash='same'),
          dict(id='privileged',mode='privileged',source='teacher',status='success',snapshot_hash='same')]
    for row in rows:teacher.append(raw/'records.jsonl',row)
    monkeypatch.setattr(c,'training_example',lambda row,mode:({'messages':[{'role':'assistant','content':row['id']+'()'}]},None))
    stats,selected=c.inventory([{'state':'done','raw':str(raw),'index':0}],20)
    assert stats['accepted']=={'privileged':1,'vision':1}
    assert stats['sft_eligible']['vision']==2
    assert stats['transfer_accepted']['vision']==0
    assert stats['rejections']['duplicate_scene_mode']==1


def test_cumulative_cost_stops_resume(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    original=c.collect
    def charged(a):
        original(a)
        c.atomic(Path(a.output)/'summary.json',dict(aborted=False,accounting_cost_usd=.6,http_attempts=1))
    monkeypatch.setattr(c,'collect',charged)
    a=args(tmp_path,max_cost_usd=1.5)
    c.run(a);a.resume=True
    state=c.run(a)
    assert calls==[0]
    assert state['stop_reason']=='cost_limit'
    assert state['http_attempts']==1


def test_aborted_attempt_not_automatically_retried(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch);original=c.collect
    def aborted(a):
        original(a)
        c.atomic(Path(a.output)/'summary.json',dict(aborted=True,accounting_cost_usd=0,http_attempts=0))
    monkeypatch.setattr(c,'collect',aborted)
    a=args(tmp_path)
    with pytest.raises(RuntimeError,match='aborted'):c.run(a)
    a.resume=True
    with pytest.raises(RuntimeError,match='blocked'):c.run(a)
    assert calls==[0]
    assert teacher.read(Path(a.output)/'state.json')['attempts'][0]['summary']['accounting_cost_usd']==1.


def test_continues_past_twenty_until_target(tmp_path,monkeypatch):
    from capx.training import campaign as c
    calls=fake_runtime(monkeypatch)
    monkeypatch.setattr(c,'training_example',lambda row,mode:({'messages':[{'role':'assistant','content':f'fn_{row["seed"]}()'}]},None))
    monkeypatch.setattr(c,'export_data',lambda *a:None)
    a=args(tmp_path,target_success=25,max_requests=30,max_scenes=30)
    state=c.run(a)
    assert len(calls)==25
    assert state['stop_reason']=='target_reached'


def test_legacy_checkpoint_compatibility_is_narrow():
    from capx.training.campaign import compatible_config
    new={'implementation_digest':'new','target_success':300}
    old={'implementation_digest':'3e7f0450409e878d79c7a50e2b6b47da869388db497101883476bd82818b417a','target_success':300}
    assert compatible_config(old,new)
    assert not compatible_config({**old,'target_success':100},new)
    assert not compatible_config({**old,'implementation_digest':'unknown'},new)
