import json
from pathlib import Path
import pytest
from capx.experiments.graph_replay import RecordedResponses


def fixture(tmp_path, **overrides):
    source=tmp_path/'source';dest=tmp_path/'new';dest.mkdir()
    request=source/'llm'/'001';request.mkdir(parents=True)
    prompt=[{'role':'user','content':'public feedback'}]
    receipt=dict(label='replan-G1-seed-1-fault-none-1',http_status=200,
                 requested_model='deepseek-v4-pro',response_model='deepseek-v4-pro',finish_reason='stop')
    receipt.update(overrides)
    response=dict(content='```python\nread_robot()\n```',response_model='deepseek-v4-pro',finish_reason='stop')
    for name,data in [('receipt.json',receipt),('response.json',response),('prompt.json',prompt),('http.json',{'payload':{'messages':prompt}})]:
        (request/name).write_text(json.dumps(data))
    return source,dest,prompt,receipt['label'],request


def test_exact_historical_response_no_new_request_and_no_repeat(tmp_path):
    source,dest,prompt,label,path=fixture(tmp_path)
    replay=RecordedResponses(source,dest,120)
    answer,reference=replay.query(prompt,label)
    assert answer['content'].startswith('```python') and reference==str(path.resolve())
    assert replay.count==0 and replay.rows[0]['prompt_exact_match'] is True
    assert replay.rows[0]['new_llm_requests']==0
    with pytest.raises(ValueError,match='already consumed'):replay.query(prompt,label)


def test_prompt_drift_stops_before_consumption(tmp_path):
    source,dest,prompt,label,_=fixture(tmp_path);replay=RecordedResponses(source,dest,120)
    with pytest.raises(ValueError,match='prompt drift'):replay.query([{'role':'user','content':'changed'}],label)
    assert replay.rows==[] and replay.count==0


def test_missing_response_has_no_paid_fallback(tmp_path):
    source,dest,prompt,label,_=fixture(tmp_path);replay=RecordedResponses(source,dest,120)
    with pytest.raises(ValueError,match='Missing recorded'):replay.query(prompt,'missing')
    assert replay.rows==[]


@pytest.mark.parametrize('overrides',[{'http_status':500},{'finish_reason':'length'},
    {'requested_model':'other'},{'response_model':'other'}])
def test_bad_original_receipt_rejected(tmp_path,overrides):
    source,dest,prompt,label,_=fixture(tmp_path,**overrides)
    replay=RecordedResponses(source,dest,120)
    with pytest.raises(ValueError,match='receipt'):replay.query(prompt,label)


def test_transport_prompt_and_response_model_must_match(tmp_path):
    source,dest,prompt,label,path=fixture(tmp_path)
    (path/'http.json').write_text(json.dumps({'payload':{'messages':[]}}))
    with pytest.raises(ValueError,match='transport prompt'):RecordedResponses(source,dest,120).query(prompt,label)
    (path/'http.json').write_text(json.dumps({'payload':{'messages':prompt}}))
    response=json.loads((path/'response.json').read_text());response['response_model']='other'
    (path/'response.json').write_text(json.dumps(response))
    with pytest.raises(ValueError,match='response'):RecordedResponses(source,dest,120).query(prompt,label)


def test_duplicate_label_and_existing_record_refuse_overwrite(tmp_path):
    import shutil
    source,dest,prompt,label,path=fixture(tmp_path)
    shutil.copytree(path,path.parent/'002')
    with pytest.raises(ValueError,match='Duplicate'):RecordedResponses(source,dest,120)



def test_replay_module_does_not_import_simulator_before_service_configuration():
    import subprocess,sys
    checked=subprocess.run([sys.executable,'-B','-c',
        'import sys; import capx.experiments.graph_replay; assert "capx.envs" not in sys.modules'],
        capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
