import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from capx.web import experiments as web


def client(tmp_path,monkeypatch):
    monkeypatch.setattr(web,'ARTIFACTS',tmp_path)
    app=FastAPI();app.include_router(web.router);return TestClient(app)


def test_routes_do_not_expose_files(tmp_path,monkeypatch):
    c=client(tmp_path,monkeypatch)
    assert c.get('/api/experiments/episode',params={'path':'../../etc'}).status_code==400
    assert c.get('/api/experiments/video',params={'path':'.','name':'../secret.mp4'}).status_code==400
    assert c.post('/api/experiments/validate',json={'provider':'openai'}).status_code==400


def test_history_reports_actual_failure_and_no_fabricated_graph(tmp_path,monkeypatch):
    ep=tmp_path/'B0';ep.mkdir();(ep/'result.json').write_text(json.dumps({'reward':0,'raw_success':False,'group':'B0'}))
    c=client(tmp_path,monkeypatch)
    assert c.get('/api/experiments/episode',params={'path':'B0'}).json()['reward']==0
    assert c.get('/api/experiments/graph',params={'path':'B0'}).json()['nodes']=={}


def test_partial_trace_tolerated(tmp_path,monkeypatch):
    ep=tmp_path/'G1';ep.mkdir();(ep/'trace.jsonl').write_text('{"kind":"api","name":"wait"}\n{"kind":')
    c=client(tmp_path,monkeypatch)
    assert c.get('/api/experiments/trace',params={'path':'G1'}).json()['total']==1


def test_public_frame(tmp_path,monkeypatch):
    import numpy as np
    ep=tmp_path/'G1/observations';ep.mkdir(parents=True)
    np.savez(ep/'00000-camera.npz',rgb=np.zeros((12,12,3),dtype=np.uint8))
    c=client(tmp_path,monkeypatch)
    r=c.get('/api/experiments/frame',params={'path':'G1'})
    assert r.status_code==200 and r.headers['content-type']=='image/jpeg'


def test_stop_before_output_creation_does_not_break_exclusive_creation(tmp_path,monkeypatch):
    import types
    ident='a'*32;root=tmp_path/ident;root.mkdir()
    monkeypatch.setattr(web,'JOBS',tmp_path)
    monkeypatch.setattr(web,'processes',{ident:types.SimpleNamespace(poll=lambda:None)})
    assert web.stop(ident)['stopping']
    assert (root/'stop.pending').exists() and not (root/'run').exists()


def test_grounded_history_keeps_real_physics_and_program(tmp_path,monkeypatch):
    ep=tmp_path/'B0';ep.mkdir()
    (ep/'result.json').write_text(json.dumps({'group':'B0','reward':1.,'final_physics':{'raw_success':True}}))
    (ep/'block-0.json').write_text(json.dumps({'code':'print("example")','code_hash':'abc'}))
    c=client(tmp_path,monkeypatch)
    r=c.get('/api/experiments/episode',params={'path':'B0'}).json()
    assert r['raw_success'] is True and r['initial_code_hash']=='abc'
    assert c.get('/api/experiments/programs',params={'path':'B0'}).json()[0]['data']['code']=='print("example")'


def test_immediate_stop_terminates_owned_process_group(tmp_path,monkeypatch):
    import asyncio,subprocess,sys
    ident='b'*32;root=tmp_path/ident;(root/'run').mkdir(parents=True)
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    monkeypatch.setattr(web,'JOBS',tmp_path);monkeypatch.setattr(web,'processes',{ident:p})
    try:
        r=asyncio.run(web.terminate(ident))
        assert r['stopped'] and p.poll() is not None
        assert json.loads((root/'run/status.json').read_text())['reason']=='operator_terminated_incomplete_pair'
    finally:
        if p.poll() is None:p.kill();p.wait()


def test_reuse_statistics_count_only_recorded_reuses(tmp_path,monkeypatch):
    ep=tmp_path/'G1';ep.mkdir()
    events=[{'kind':'perception_request','reused':False},{'kind':'perception_request','reused':True},
            {'kind':'perception_geometry_reuse'},{'kind':'perception_plan_reuse','reused':False}]
    (ep/'trace.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    c=client(tmp_path,monkeypatch);r=c.get('/api/experiments/episode',params={'path':'G1'}).json()
    assert (r['semantic_reuses'],r['geometry_reuses'],r['plan_reuses'])==(1,1,0)


def test_force_history_keeps_timestamps_and_offline_boundary(tmp_path,monkeypatch):
    ep=tmp_path/'A';ep.mkdir()
    (ep/'result.json').write_text('{"force_feedback":false}')
    (ep/'force.jsonl').write_text('{"timestamp":1,"sequence":1,"external_base":[0,0,-1,0,0,0]}\n{"timestamp":1.05,"sequence":2}\n{"timestamp":')
    (ep/'trace.jsonl').write_text('{"kind":"public_capture","sim_time":1}\n{"kind":"force_load","status":"UNKNOWN"}\n{"kind":"skill_attachment_settle","sample":0}\n')
    c=client(tmp_path,monkeypatch);r=c.get('/api/experiments/force',params={'path':'A','limit':1}).json()
    assert r['total']==2 and r['rows'][0]['timestamp']==1.05 and r['offset']==1
    assert r['online_enabled'] is False and r['events'][0]['timestamp']==1
    assert c.get('/api/experiments/force',params={'path':'../../etc'}).status_code==400


def test_force_history_missing_is_not_fabricated_zero(tmp_path,monkeypatch):
    (tmp_path/'old').mkdir();c=client(tmp_path,monkeypatch)
    r=c.get('/api/experiments/force',params={'path':'old'}).json()
    assert r['total']==0 and r['rows']==[] and r['online_enabled'] is None
