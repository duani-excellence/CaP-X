from unittest.mock import patch
from threading import Event
import pytest
from fastapi.testclient import TestClient

@pytest.mark.parametrize('provider,model,key',[('deepseek','deepseek-v4-pro','ds-test'),('openai','gpt-6-astra','oa-test')])
@pytest.mark.parametrize('mode,privileged',[('vision',False),('privileged',True)])
def test_selection_routes_credentials_and_observation(monkeypatch,provider,model,key,mode,privileged):
 from capx.web.server import create_app
 monkeypatch.setenv('DEEPSEEK_API_KEY','ds-test');monkeypatch.setenv('OPENAI_API_KEY','oa-test')
 monkeypatch.setenv('DEEPSEEK_MODEL','deepseek-v4-pro');monkeypatch.setenv('OPENAI_MODEL','gpt-6-astra')
 monkeypatch.delenv('DEEPSEEK_SERVER_URL',raising=False);monkeypatch.delenv('OPENAI_SERVER_URL',raising=False)
 seen=[];captured=Event()
 async def capture(*,session,args):
  seen.append((session,args));captured.set()
 app=create_app();app.state.default_api_key='wrong-default';app.state.default_max_tokens=32768
 with patch('capx.web.server.run_trial_async',capture),TestClient(app) as client:
  response=client.post('/api/start-trial',json={'provider':provider,'observation_mode':mode,'config_path':'ignored','server_url':'https://wrong.invalid'})
  assert response.status_code==200,response.text
  assert captured.wait(2), "background task was not scheduled"
  session,args=seen[0]
  assert args.api_key==key and args.model==model
  assert args.server_url==f'https://api.{provider}.com/v1/chat/completions'
  assert args.max_tokens==32768 and args.temperature==.2
  assert session.env_factory['cfg']['privileged'] is privileged
  assert ('FrankaControlPrivilegedApi' in session.env_factory['cfg']['apis']) is privileged
  assert 'test' not in response.text and 'wrong-default' not in response.text

def test_options_contain_only_two_choices_and_no_secrets(monkeypatch):
 from capx.web.server import create_app
 monkeypatch.setenv('OPENAI_API_KEY','secret-not-for-browser')
 with TestClient(create_app()) as client:
  r=client.get('/api/trial-options')
  assert r.status_code==200
  assert len(r.json()['models'])==len(r.json()['observations'])==2
  assert 'secret-not-for-browser' not in r.text
