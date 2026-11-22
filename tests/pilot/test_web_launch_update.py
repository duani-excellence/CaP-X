import os
import runpy
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient

def test_web_cli_reuses_services(monkeypatch):
 import capx.envs.launch as launch
 monkeypatch.setenv('DEEPSEEK_API_KEY','mock-key')
 monkeypatch.delenv('GRASPNET_SERVICE_URL',raising=False)
 monkeypatch.setattr(sys,'argv',['run.py','web','--backend','deepseek'])
 with patch.object(launch,'main') as main:
  runpy.run_path('scripts/local/run.py',run_name='__main__')
 args=main.call_args.args[0]
 assert args.max_tokens==32768
 assert os.environ['GRASPNET_SERVICE_URL']=='http://127.0.0.1:8215'
 import yaml
 config=yaml.safe_load(open(args.config_path))
 assert config['api_servers']==[]

def test_page_load_does_not_start_paid_trial():
 from capx.web.server import create_app
 app=create_app();app.state.default_config_path='example.yaml'
 with TestClient(app) as client:
  assert client.get('/api/default-config').json()['auto_start'] is False
