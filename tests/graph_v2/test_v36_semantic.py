import copy
from pathlib import Path
import pytest
from capx.execution.dependency_graph import DependencyGraph
from capx.execution.semantic_capabilities import CapabilityRegistry, tool_manifest


def proposal():
    return dict(capability_id='measure_summary',version='1',purpose='Summarize a public measurement',
        code="def capability(label):\n    data = measure(label)\n    return {'value': data['value'], 'source': data['id']}\n",
        inputs={'label':'string'},outputs={'value':'number','source':'string'},
        tools=['measure'],expected_effects=[],physical_side_effects=[],
        output_projections={'value':{'call':0,'tool':'measure','path':['value']},
                            'source':{'call':0,'tool':'measure','path':['id']}},
        verification_refs=['return_schema'],semantic_dependencies=[],alternative_capabilities=[],
        validation_status='PROPOSED')


def test_harness_policy_is_separate_from_frozen_environment_spec(tmp_path):
    from types import SimpleNamespace as S
    from capx.execution.persistent_harness import PersistentHarness
    import time
    c=S(graph=DependencyGraph(),conditions={},definition={'goals':[]})
    skills=S(scene=S(source=None),held=None,phase='empty',motion_version=0)
    runtime=S(c=c,skills=skills,group='G0',emit=lambda *a,**k:None,
        state=lambda:{'conditions':{}},open_runtime=False)
    boundary=S(root=tmp_path,spec={'episode_seconds':100},started=time.monotonic(),
        calls=0,initial={'state_hash':'same'},call=lambda *a:None,actor=S(call_site=None))
    h=PersistentHarness(boundary,runtime,S(max_requests=0,count=0),request_limit=0,
        policy={'semantic_version':36})
    assert 'register_capability' in h.tools.functions()
    assert boundary.spec=={'episode_seconds':100}


def registry(tmp_path,**kw):
    def measure(label):return {'value':2.,'id':'public-1'}
    return CapabilityRegistry(tmp_path,{'measure':measure},graph=DependencyGraph(),
        tool_effects={'measure':[]},episode=kw.pop('episode','one'),**kw)


def test_registration_is_not_effect_or_permission(tmp_path):
    r=registry(tmp_path);p=proposal();value=r.register(p)
    assert value['status']=='VALIDATED' and value['permission'] is False
    assert r.query(value['id'])['physical_verified'] is False
    assert r.query(value['id'])['semantic_edges']==[]


@pytest.mark.parametrize('change',[
    {'tools':['invented']}, {'inputs':{'label':'secret'}}, {'permissions':['raw_robot']},
    {'validation_status':'VERIFIED'}, {'physical_side_effects':['release']},
    {'code':'def capability(label):\n    return env.step(label)'},
    {'code':'def capability(label):\n    return invented(label)'},
    {'expected_effects':['task_success'], 'verification_refs':['return_schema']},
])
def test_bad_proposals_do_not_mutate_valid_version(tmp_path,change):
    r=registry(tmp_path);good=r.register(proposal());bad=proposal();bad.update(change)
    response=r.register(bad)
    assert response['status']=='REJECTED' and response['errors']
    assert r.query(good['id'])['status']=='VALIDATED'


def test_real_bubblewrap_composition_and_reuse(tmp_path):
    r=registry(tmp_path);row=r.register(proposal());result=r.run(row['id'],{'label':'item'})
    assert result['status']=='VERIFIED' and result['output']['value']==2
    assert result['verification_scope']=='audited_output_projections' and not result['physical_verified']
    assert r.graph.nodes[result['node_id']]['data']['calls'][0]['name']=='measure'
    assert r.run(row['id'],{'label':3})['status']=='REJECTED'
    r2=registry(tmp_path,episode='two',callback=lambda name,args,kwargs:{'value':3.,'id':'public-2'})
    assert r2.query(row['id'])['status']=='VERIFIED'
    assert r2.run(row['id'],{'label':'another'})['status']=='VERIFIED'  # two cases do not certify the applicability domain


def test_identical_frozen_input_with_new_episode_label_not_reuse_validation(tmp_path):
    r=registry(tmp_path);row=r.register(proposal());r.run(row['id'],{'label':'x'})
    r2=registry(tmp_path,episode='different_analysis_directory')
    result=r2.run(row['id'],{'label':'x'})
    assert result['status']=='VERIFIED'


def test_version_drift_and_stale_evidence_block(tmp_path):
    r=registry(tmp_path,evidence_valid=lambda ref:False);row=r.register(proposal())
    assert r.run(row['id'],{'label':'x'},evidence_ids=['old'])['status']=='REJECTED'
    r.manifest['measure']['implementation_hash']='changed'
    assert r.run(row['id'],{'label':'x'})['status']=='DEPRECATED'


def test_semantic_edges_not_runtime_causal_edges(tmp_path):
    r=registry(tmp_path);p=proposal();p['alternative_capabilities']=['other_hypothesis']
    row=r.register(p);value=r.query(row['id'])
    assert value['semantic_edges'][0]['kind']=='SEMANTIC_DEPENDENCY'
    assert all(e['kind']!='PHYSICAL_CAUSAL_EVIDENCE' for e in value['semantic_edges'])
    assert not r.graph.edges


def test_schema_alone_does_not_verify_semantics(tmp_path):
    r=registry(tmp_path);p=proposal();p.pop('output_projections')
    p['code']="def capability(label):\n    data = measure(label)\n    return {'value': 99., 'source': 'invented'}"
    row=r.register(p);result=r.run(row['id'],{'label':'x'})
    assert result['output_schema_valid'] and result['status']=='EXECUTED'
    assert result['verification_scope']=='unverified'
    assert not r.query(row['id']).get('verified_episodes')


def test_false_projection_and_missing_call_cannot_verify(tmp_path):
    r=registry(tmp_path);p=proposal()
    p['code']="def capability(label):\n    data = measure(label)\n    return {'value': 99., 'source': data['id']}"
    row=r.register(p);result=r.run(row['id'],{'label':'x'})
    assert result['status']=='EXECUTED' and not result['projection_validation']['value']['matches']
    p['code']="def capability(label):\n    return {'value': 2., 'source': 'public-1'}"
    row=r.register(p);result=r.run(row['id'],{'label':'x'})
    assert result['status']=='EXECUTED' and not result['projection_validation']['value']['matches']


def test_legacy_schema_maturity_is_not_trusted(tmp_path):
    r=registry(tmp_path);row=r.register(proposal())
    r.event('execution',capability_id=row['id'],status='REUSABLE',physical_verified=False,
            verification_scope='return_schema_only',verified_episodes=['a','b'])
    record=r.query(row['id'])
    assert record['status']=='EXECUTED' and record['verified_episodes']==[]


def test_failed_nested_api_is_audited(tmp_path):
    def stop(name,args,kwargs):raise ValueError('public precondition missing')
    r=registry(tmp_path,callback=stop);row=r.register(proposal())
    result=r.run(row['id'],{'label':'x'})
    call=r.graph.nodes[result['node_id']]['data']['calls'][0]
    assert call['status']=='FAILED' and call['name']=='measure'
    assert 'precondition' in call['error'] and call['code_site']


def test_nested_projection_requires_all_output_leaves(tmp_path):
    r=registry(tmp_path);p=proposal();p['outputs']={'summary':'object'}
    p['code']="def capability(label):\n    data = measure(label)\n    return {'summary': {'value': data['value'], 'source': data['id']}}"
    p['output_projections']={'summary.value':{'call':0,'tool':'measure','path':['value']}}
    row=r.register(p);result=r.run(row['id'],{'label':'x'})
    assert result['status']=='EXECUTED' and result['output_schema_valid']
    p['output_projections']['summary.source']={'call':0,'tool':'measure','path':['id']}
    row=r.register(p);result=r.run(row['id'],{'label':'x'})
    assert result['status']=='VERIFIED' and not result['physical_verified']


def test_actual_tool_reference_and_projected_output_have_data_edges(tmp_path):
    def measure(label):return {'id':'measured-result','value':3}
    def inspect_result(ref):return {'id':'inspection-result','value':ref}
    r=CapabilityRegistry(tmp_path,{'measure':measure,'inspect_result':inspect_result},
        graph=DependencyGraph(),tool_effects={'measure':[],'inspect_result':[]},episode='one')
    p=proposal();p.update(code="def capability(label):\n    data=measure(label)\n    check=inspect_result(data['id'])\n    return {'source':check['value']}",
        outputs={'source':'string'},tools=['measure','inspect_result'],
        output_projections={'source':{'call':1,'tool':'inspect_result','path':['value']}})
    row=r.register(p);result=r.run(row['id'],{'label':'x'})
    calls=r.graph.nodes[result['node_id']]['data']['calls']
    assert {'source':calls[0]['result_node'],'target':calls[1]['call_node'],'kind':'data'} in r.graph.edges
    assert {'source':calls[1]['result_node'],'target':result['output_node'],'kind':'data'} in r.graph.edges
    assert r.graph.nodes[calls[0]['call_node']]['state']=='done'


def test_harness_event_sink_does_not_conflict_with_record_kind(tmp_path):
    events=[]
    r=registry(tmp_path,emit=lambda kind,**data:events.append((kind,data)))
    row=r.register(proposal())
    assert row['status']=='VALIDATED'
    assert events[0][0]=='SemanticCapabilityEvent'
    assert events[0][1]['record']['kind']=='registered'
