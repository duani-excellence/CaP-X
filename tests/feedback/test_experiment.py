import pytest
from capx.experiments.layered_feedback import validated_program, initial_prompt, recovery_prompt

def test_reasoning_not_code():
    r={'content':'```python\nopen_gripper()\n```','reasoning_content':'env.secret','finish_reason':'stop','response_model':'deepseek-v4-pro'}
    assert validated_program(r)=='open_gripper()'

@pytest.mark.parametrize('field,value',[('finish_reason','length'),('response_model','another-model'),('content','')])
def test_bad_response_rejected(field,value):
    r={'content':'pass','finish_reason':'stop','response_model':'deepseek-v4-pro'};r[field]=value
    with pytest.raises(ValueError):validated_program(r)

def test_replan_context_excludes_evaluator():
    observation={'full_prompt':[{'role':'user','content':[{'type':'text','text':'API docs'}]}], 'raw_success':True, 'cube_poses':'SECRET'}
    prompt=recovery_prompt(observation,'pass',{'status':'UNCERTAIN'},'interrupted')
    assert 'SECRET' not in str(prompt) and 'raw_success' not in str(prompt)
    assert 'UNCERTAIN' in str(prompt) and 'REGENERATE' in str(prompt)

def test_explicit_deepseek_effort_does_not_change_default_requests(monkeypatch):
    import json
    from types import SimpleNamespace
    from capx.llm import client
    seen=[]
    def post(url,**kwargs):
        seen.append(json.loads(kwargs['data']))
        return SimpleNamespace(status_code=200,headers={},raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':'pass','reasoning_content':'not code'},'finish_reason':'stop'}],'model':'deepseek-v4-pro'})
    monkeypatch.setattr(client,'audited_post',post)
    args=client.ModelQueryArgs(model='deepseek-v4-pro',server_url='https://api.deepseek.com/v1/chat/completions',max_tokens=32768)
    client.query_model(args,[{'role':'user','content':'code'}])
    assert 'reasoning_effort' not in seen[-1]
    args.deepseek_reasoning_effort='low'
    answer=client.query_model(args,[{'role':'user','content':'code'}])
    assert seen[-1]['reasoning_effort']=='low' and seen[-1]['max_tokens']==32768
    assert answer['content']=='pass'
