import json
import pytest

@pytest.mark.parametrize('override,expected',[(None,32768),(16384,16384)])
def test_model_phase_propagates_shared_budget(tmp_path,monkeypatch,override,expected):
 import capx.experiments.pilot as p
 manifest=tmp_path/'manifest/scenes.json';manifest.parent.mkdir()
 manifest.write_text(json.dumps({'scenes':[{'state_hash':str(i),'seed':i} for i in range(5)]}))
 (manifest.parent/'gates.json').write_text(json.dumps({'privileged':True,'vision':False}))
 monkeypatch.setattr(p,'check_services',lambda _: {'8116':{'ok':True}})
 seen=[]
 def fake_trial(*args,**kwargs):
  seen.append((args[4],kwargs['max_tokens']))
  return {'status':'success'}
 monkeypatch.setattr(p,'run_trial',fake_trial)
 kwargs={} if override is None else {'max_tokens':override}
 p.run(manifest,'models',tmp_path,**kwargs)
 assert len(seen)==10 and {tokens for _,tokens in seen}=={expected}
 assert {provider for provider,_ in seen}=={'deepseek','openai'}
 assert json.loads((manifest.parent/'model_budget.json').read_text())['max_completion_tokens']==expected

def test_large_budget_is_reserved_before_request():
 from capx.experiments.pilot import CostLedger
 ledger=CostLedger()
 bound=ledger.reserve('openai',input_bytes=6000,max_tokens=32768)
 assert bound>1.6
 ledger.spent['openai']=2.0
 with pytest.raises(RuntimeError):ledger.reserve('openai',input_bytes=6000,max_tokens=32768)

@pytest.mark.parametrize('override,expected',[(None,32768),(16384,16384)])
def test_cli_passes_completion_budget(monkeypatch,override,expected):
 import runpy
 import sys
 import capx.experiments.pilot as p
 seen=[]
 monkeypatch.setattr(p,'run',lambda *args,**kwargs:seen.append(kwargs['max_tokens']))
 argv=['run.py','privileged','--snapshot-manifest','unused.json','--pilot-phase','models','--output-dir','unused']
 if override is not None:argv+=['--max-tokens',str(override)]
 monkeypatch.setattr(sys,'argv',argv)
 runpy.run_path('scripts/local/run.py',run_name='__main__')
 assert seen==[expected]
