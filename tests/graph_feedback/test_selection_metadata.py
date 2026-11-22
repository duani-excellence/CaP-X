"""No simulator or model: conservative fallback selection metadata only."""
import pytest
from capx.experiments import graph_feedback as runner


@pytest.mark.parametrize('status',['runtime_error','interrupted'])
def test_conservative_graph_fallback_reports_full_block_without_changing_prompt(status):
    code='first()\nsecond()'
    feedback={'status':'FAIL','diagnostic_source':code,
              'recovery_scope':{'policy':'all_unexecuted_suffix','fallback':True}}
    obs={'full_prompt':[{'role':'user','content':'Stack task'}]}
    prompt,meta=runner.graph_recovery_prompt(obs,code,feedback,status,'G1',api_doc='GRAPH DOC')
    expected=runner.recovery_prompt({'full_prompt':obs['full_prompt']+[{'role':'user','content':[{'type':'text','text':'GRAPH DOC'}]}]},code,feedback,status)
    assert prompt==expected
    assert meta['previous_code_hash']==runner.digest(code)
    assert meta['previous_code_selection']=='full_block'
    assert meta['diagnostic_source_fallback'] is True
