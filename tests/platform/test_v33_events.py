import json
from fastapi.testclient import TestClient
from fastapi import FastAPI
from capx.web import experiments


def test_harness_history_absent_paged_and_path_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(experiments,'ARTIFACTS',tmp_path)
    app=FastAPI();app.include_router(experiments.router);c=TestClient(app)
    root=tmp_path/'trial';root.mkdir()
    assert c.get('/api/experiments/events',params={'path':'trial'}).json()['available'] is False
    h=root/'harness';h.mkdir()
    (h/'events.jsonl').write_text(json.dumps({'sequence':1,'type':'TaskStarted','data':{}})+'\n'+json.dumps({'sequence':2,'type':'RecoveryProposed','data':{'level':'R2'}})+'\n')
    r=c.get('/api/experiments/events',params={'path':'trial','offset':1,'limit':1}).json()
    assert r['available'] and r['total']==2 and r['rows'][0]['type']=='RecoveryProposed'
    assert c.get('/api/experiments/events',params={'path':'../secret'}).status_code==400
