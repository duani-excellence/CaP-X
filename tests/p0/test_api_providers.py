import json
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('name,prefix,host',[('deepseek','DEEPSEEK','api.deepseek.com'),('openai','OPENAI','api.openai.com'),('chatgpt','OPENAI','api.openai.com')])
def test_independent_credentials(name,prefix,host):
    from capx.llm.providers import provider_config
    env={'DEEPSEEK_API_KEY':'ds-test','OPENAI_API_KEY':'oa-test','OPENAI_MODEL':'account-model'}
    url,model,key=provider_config(name,env)
    assert host in url and key==env[prefix+'_API_KEY']
    assert model==('deepseek-v4-pro' if prefix=='DEEPSEEK' else 'account-model')

@pytest.mark.parametrize('env',[{}, {'OPENAI_API_KEY':'test'},{'OPENAI_MODEL':'x'}])
def test_missing_config_no_fallback(env):
    from capx.llm.providers import provider_config
    with pytest.raises(ValueError):provider_config('openai',env)

def test_official_openai_payload(monkeypatch):
    from capx.llm import client
    seen={}
    def post(url,**kwargs):
        seen.update(json.loads(kwargs['data']))
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':'pass'}}]})
    monkeypatch.setattr(client,'audited_post',post)
    args=client.ModelQueryArgs(model='account-model',server_url='https://api.openai.com/v1/chat/completions',api_key='test')
    assert client.query_model(args,[{'role':'user','content':'code'}])['content']=='pass'
    assert seen['max_completion_tokens']==4096
    assert 'temperature' not in seen and 'max_tokens' not in seen

def test_oracle_needs_no_cloud_key(monkeypatch):
    import importlib.util,sys,types
    spec=importlib.util.spec_from_file_location('local_run','scripts/local/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seen={}
    fake=types.ModuleType('capx.envs.launch')
    fake.LaunchArgs=lambda **kw:seen.update(kw) or kw
    fake.main=lambda args:None
    monkeypatch.setitem(sys.modules,'capx.envs.launch',fake)
    monkeypatch.setattr(sys,'argv',['run.py','oracle','--backend','openai'])
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('OPENAI_MODEL',raising=False)
    module.run()
    assert seen['use_oracle_code']
