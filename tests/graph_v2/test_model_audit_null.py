import json,time
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('body',[None,[],{'usage':None},{'usage':{'prompt_tokens':None,'completion_tokens':2}},{'usage':{'prompt_tokens':'bad','completion_tokens':2}},{'usage':{'prompt_tokens':-1,'completion_tokens':2}}])
def test_bad_usage_does_not_erase_transport_receipt(tmp_path,monkeypatch,body):
    from capx.experiments.layered_feedback import ModelBudget
    from capx.llm import providers,client
    monkeypatch.setattr(providers,'provider_config',lambda _:('https://unused.test','deepseek-v4-pro','test-not-a-secret'))
    def query(args,prompt):
        response=SimpleNamespace(status_code=200,headers={'x-request-id':'request-1'},json=lambda:body)
        args.response_observer(1,{'model':'deepseek-v4-pro'},response,None)
        raise RuntimeError('upstream payload rejected')
    monkeypatch.setattr(client,'query_model',query)
    budget=ModelBudget(tmp_path,1,10,64,time.monotonic()+60)
    with pytest.raises(RuntimeError,match='upstream payload rejected'):budget.query([], 'null')
    receipt=json.loads((tmp_path/'llm/001-null/receipt.json').read_text())
    http=json.loads((tmp_path/'llm/001-null/http.json').read_text())
    assert receipt['attempts']==1 and receipt['http_status']==200 and receipt['request_id']=='request-1'
    assert not receipt['usage_known'] and receipt['cost_upper_usd']>0
    assert receipt['usage']=={}  # downstream billing must not treat bad counts as measured usage
    assert http['response']==body and 'Authorization' not in json.dumps(http)
