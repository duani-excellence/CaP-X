"""HTTP job validation and CLI forwarding; no simulator or paid requests."""
from types import SimpleNamespace
import pytest
from fastapi import HTTPException


def test_web_collect_uses_cli_and_resume_keeps_limits(monkeypatch,tmp_path):
 from capx.web import multitask as web
 from capx.web import session_manager
 monkeypatch.setattr(web,'JOBS',tmp_path)
 monkeypatch.setattr(web,'processes',{})
 monkeypatch.setattr(web,'require_ready',lambda task:None)
 monkeypatch.setattr(session_manager,'get_session_manager',lambda:SimpleNamespace(list_sessions=lambda:[]))
 calls=[]
 def launch(command,**kwargs):
  calls.append(command)
  return SimpleNamespace(poll=lambda:0,pid=123)
 monkeypatch.setattr(web.subprocess,'Popen',launch)
 req=web.JobRequest(targets={'lift':2,'handover':3},action='collect',max_requests_per_task=7,max_cost_usd_per_task=2)
 result=web.start(req)
 assert '--targets' in calls[0] and 'lift=2' in calls[0] and 'handover=3' in calls[0]
 assert calls[0][calls[0].index('--max-requests-per-task')+1]=='7'
 assert not any('api-key' in arg.lower() for arg in calls[0])
 resumed=req.model_copy(update={'resume_id':result['id']})
 web.start(resumed)
 assert '--resume' in calls[1]
 with pytest.raises(HTTPException,match=''):
  web.start(resumed.model_copy(update={'max_requests_per_task':8}))
 assert len(calls)==2
 assert len(web.jobs())==1


def test_web_unverified_or_busy_never_launches(monkeypatch):
 from capx.web import multitask as web
 monkeypatch.setattr(web,'require_ready',lambda task: (_ for _ in ()).throw(ValueError('unverified')))
 with pytest.raises(HTTPException) as exc:web.start(web.JobRequest(targets={'wipe':1}))
 assert exc.value.status_code==400
 monkeypatch.setattr(web,'require_ready',lambda task:None)
 monkeypatch.setattr(web,'processes',{'busy':SimpleNamespace(poll=lambda:None)})
 with pytest.raises(HTTPException) as exc:web.start(web.JobRequest(targets={'lift':1}))
 assert exc.value.status_code==409


def test_web_stop_signals_only_owned_job(monkeypatch):
 from capx.web import multitask as web
 import signal
 ident='a'*32
 monkeypatch.setattr(web,'processes',{ident:SimpleNamespace(poll=lambda:None,pid=123)})
 sent=[];monkeypatch.setattr(web.os,'killpg',lambda *args:sent.append(args))
 assert web.stop(ident)=={'stopping':True}
 assert sent==[(123,signal.SIGINT)]
 with pytest.raises(HTTPException):web.stop('b'*32)
