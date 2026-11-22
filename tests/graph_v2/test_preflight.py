"""Read-only runtime inventory must not call models or leak credentials."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from capx.experiments import graph_preflight as p


def config():
    return {'services':{'sam3':'http://localhost:1','graspnet':'http://127.0.0.1:2','pyroki':'http://127.0.0.1:3'}}


def test_config_requires_explicit_three_service_urls():
    assert p.service_urls(config())==config()['services']
    with pytest.raises(ValueError):p.service_urls({'services':{'sam3':'http://localhost:1'}})


@pytest.mark.parametrize('url',['http://name:secret@localhost:1','http://localhost:1/?api_key=private','http://localhost:1/#private','file:///tmp/x'])
def test_service_urls_reject_embedded_credentials_and_non_http(url):
    c=config();c['services']['sam3']=url
    with pytest.raises(ValueError):p.service_urls(c)


def test_get_uses_only_get_and_records_hash_not_response_body(monkeypatch):
    seen=[]
    class Response:
        status=200;headers={'content-type':'application/json'}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):return b'{"private_response_value":"NEVER_SERIALIZE"}'
    def open_(request,timeout):seen.append(request);return Response()
    monkeypatch.setattr(p.urllib.request,'urlopen',open_)
    row,body=p.get_endpoint('http://localhost:1/openapi.json',2)
    assert seen[0].get_method()=='GET' and not seen[0].header_items()
    assert row['http_status']==200 and len(row['body_sha256'])==64
    assert 'NEVER_SERIALIZE' not in json.dumps(row)
    assert body['private_response_value']=='NEVER_SERIALIZE'


def test_service_inventory_schema_seed_and_health_only(monkeypatch):
    paths=[]
    doc={'paths':{'/healthz':{'get':{}},'/predict':{'post':{}}},'components':{'schemas':{
        'PlanRequest':{'properties':{'seed':{}}},'PlanResponse':{'properties':{'seed':{}}}}}}
    def get(url,timeout):paths.append(url);return {'http_status':200,'body_sha256':'hash'},doc
    monkeypatch.setattr(p,'get_endpoint',get)
    monkeypatch.setattr(p,'listener_pids',lambda url:([123],None))
    monkeypatch.setattr(p,'process_identity',lambda pid:{'pid':pid})
    row=p.inspect_service('graspnet','http://localhost:2',2)
    assert row['seed_request_response'] is True
    assert paths==['http://localhost:2/openapi.json','http://localhost:2/healthz']
    assert row['inference_reexecuted'] is False


def test_process_identity_omits_arbitrary_argv_and_environment(tmp_path):
    proc=tmp_path/'proc';(proc/'123').mkdir(parents=True)
    cwd=tmp_path/'project';cwd.mkdir();(cwd/'service.py').write_text('VALUE=1\n')
    (proc/'123/cwd').symlink_to(cwd);(proc/'123/exe').symlink_to('/usr/bin/python3')
    (proc/'123/cmdline').write_bytes(b'/usr/bin/python3\0service.py\0--api-key\0PRIVATE_SENTINEL\0')
    (proc/'123/environ').write_bytes(b'KEY=DO_NOT_READ_ENV')
    (proc/'123/stat').write_text('123 (python) '+ ' '.join(['S']+['0']*18+['100']+['0']*3))
    (proc/'stat').write_text('btime 1000\n')
    row=p.process_identity(123,proc_root=proc)
    encoded=json.dumps(row)
    assert 'PRIVATE_SENTINEL' not in encoded and 'DO_NOT_READ_ENV' not in encoded and 'api-key' not in encoded
    assert row['pid']==123 and row['cwd']==str(cwd)
    assert row['entrypoint']['kind']=='script'
    assert row['source_files'][str(cwd/'service.py')]['sha256']==p.sha256((cwd/'service.py').read_bytes())


def test_source_discovery_does_not_execute_module(tmp_path):
    file=tmp_path/'src/pkg/service.py';file.parent.mkdir(parents=True)
    file.write_text('raise RuntimeError("do not execute")\n')
    source=p.resolve_module('pkg.service',[tmp_path,tmp_path/'src'])
    assert source==file


def test_remote_service_identity_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(p,'run_command',lambda *a,**kw:pytest.fail('no local inspection for remote'))
    assert p.listener_pids('https://remote.example:8080')==([], 'remote_endpoint')


def test_inventory_refuses_existing_output_before_network(tmp_path,monkeypatch):
    c=tmp_path/'config.json';c.write_text(json.dumps(config()))
    monkeypatch.setattr(p,'inspect_service',lambda *a,**kw:pytest.fail('network started'))
    with pytest.raises(FileExistsError):p.collect(c,tmp_path,[])


def test_inventory_preserves_only_hashes_of_explicit_historical_evidence(tmp_path,monkeypatch):
    c=tmp_path/'config.json';c.write_text(json.dumps(config()))
    proof=tmp_path/'proof.json';proof.write_text('{"authorization":"DO_NOT_COPY"}')
    monkeypatch.setattr(p,'inspect_service',lambda name,url,timeout:{'name':name,'url':url,'openapi':{'http_status':200},'seed_request_response':True})
    monkeypatch.setattr(p,'runtime_inventory',lambda :{'python':'mock'})
    result=p.collect(c,tmp_path/'output',[proof])
    assert len(result['historical_evidence'])==1
    assert result['historical_evidence'][0]['sha256']==p.sha256(proof.read_bytes())
    assert 'DO_NOT_COPY' not in (tmp_path/'output/service-inventory.json').read_text()
    assert result['model_requests']==0 and result['inference_requests']==0


def test_cli_help_is_dependency_and_network_free(capsys):
    with pytest.raises(SystemExit) as stop:p.main(['--help'])
    assert stop.value.code==0
    assert '--config' in capsys.readouterr().out


def test_recursive_sources_follow_relative_imports_and_package_initializers(tmp_path):
    pkg=tmp_path/'pkg';pkg.mkdir();(pkg/'__init__.py').write_text('from . import side\n')
    (pkg/'main.py').write_text('from .helper import value\n');(pkg/'helper.py').write_text('from . import leaf\nvalue=1\n')
    (pkg/'leaf.py').write_text('VALUE=1\n');(pkg/'side.py').write_text('VALUE=2\n')
    found=p.source_closure([pkg/'main.py'],[tmp_path])
    assert {x.name for x in found}=={'main.py','helper.py','leaf.py','side.py','__init__.py'}


def test_runtime_inventory_only_records_explicit_safe_whitelist(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL','egl');monkeypatch.setenv('OPENAI_API_KEY','PRIVATE_SENTINEL')
    monkeypatch.setattr(p,'run_command',lambda *a,**kw:{'returncode':0,'output':''})
    out=p.runtime_inventory()
    assert out['runtime_environment']['MUJOCO_GL']=='egl'
    assert 'OPENAI_API_KEY' not in out['runtime_environment']
    assert 'PRIVATE_SENTINEL' not in json.dumps(out)


def test_process_start_uses_subsecond_boottime_not_truncated_btime(monkeypatch):
    monkeypatch.setattr(p.time,'time',lambda:2000.55)
    monkeypatch.setattr(p.time,'clock_gettime',lambda clock:1000.0)
    monkeypatch.setattr(p.os,'sysconf',lambda name:100)
    value=p.process_start_time(100,1000,Path('/proc'))
    assert value['started_at_unix']==pytest.approx(1001.55)
    assert value['started_at_timebase']=='CLOCK_BOOTTIME'
