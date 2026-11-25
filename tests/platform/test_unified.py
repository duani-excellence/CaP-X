import json
from pathlib import Path
import pytest
from capx.experiments.unified import RunSpec, resolve
from capx.experiments.artifacts import safe_path, replay_graph, redact
from capx.execution.graph_journal import GraphJournal
from capx.execution.dependency_graph import DependencyGraph


def test_default_uses_existing_contract_runner():
    spec, cfg, family = resolve(RunSpec())
    assert family == 'contract' and spec.methods == ['G0', 'G1']
    assert cfg['tasks']['pick_place']['skill_recovery_version'] == 30

@pytest.mark.parametrize('changes', [dict(methods=['ASPIRE']),dict(task='pick_place',methods=['B0']),
    dict(provider='openai'),dict(observation='privileged'),dict(planner='curobo'),
    dict(environment='libero'),dict(methods=['LOGIV'],scenario='effect_displacement'),
    dict(ablation='no_memory',methods=['G0']),dict(skill='v23',task='pick_place')])
def test_unsupported_combinations_fail_before_execution(changes):
    with pytest.raises(ValueError):resolve(RunSpec(**changes))

def test_cube_b0_routes_to_legal_runner():
    s,c,f=resolve(RunSpec(task='stack',methods=['B0','G0','G1']))
    assert f=='grounded' and c['budgets']['generic_goals']


def test_override_not_silently_ignored():
    s,c,f=resolve(RunSpec(max_tokens=12345,api_budget=51,max_requests=3,skill='v29'))
    assert c['budgets']['max_tokens']==12345 and c['budgets']['max_requests']==3
    assert c['tasks']['pick_place']['api_budget']==51
    assert c['tasks']['pick_place']['skill_recovery_version']==29


def test_snapshot_cannot_silently_apply_to_other_seeds(tmp_path):
    p=tmp_path/'snap.json';p.write_text(json.dumps({'seed':21}))
    with pytest.raises(ValueError,match='Seed'):resolve(RunSpec(snapshot=str(p),seeds=[0]))


def test_journal_replay_uses_actual_invalidations(tmp_path):
    p=tmp_path/'events.jsonl'
    with GraphJournal(p):
        g=DependencyGraph();a=g.add('observation',{'pose':[1,2,3]},key='support')
        b=g.add('plan',{'target':[1,2,4]},parents=[a]);c=g.add('observation',{'unrelated':True})
        g.invalidate(a,'support moved')
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    before=replay_graph(rows,3);after=replay_graph(rows,4)
    assert before['nodes'][b]['valid'] and not after['nodes'][b]['valid']
    assert after['nodes'][c]['valid']
    assert after['edges']==g.edges


def test_two_graphs_do_not_collide(tmp_path):
    p=tmp_path/'e.jsonl'
    with GraphJournal(p):
        a=DependencyGraph();a.add('check',{'status':'PASS'})
        b=DependencyGraph();b.add('check',{'status':'FAIL'})
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    assert len({r['graph_id'] for r in rows})==2


def test_safe_path_rejects_traversal_and_symlink(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'private';outside.write_text('secret')
    (root/'link').symlink_to(outside)
    for name in ('../private','link','/etc/passwd'):
        with pytest.raises(ValueError):safe_path(root,name)


def test_secrets_not_returned():
    assert redact({'api_key':'secret','x':{'Authorization':'Bearer abc'},'text':'ok'}) == {'api_key':'[REDACTED]','x':{'Authorization':'[REDACTED]'},'text':'ok'}


def test_causal_edges_are_expanded_from_actual_record(tmp_path):
    from capx.experiments.artifacts import graph_view
    with GraphJournal(tmp_path/'graph-events.jsonl'):
        g=DependencyGraph();p=g.add('causal_plan',{'revision':'r1','edges':[{'source':'a','target':'b','kind':'resource'}]})
        a=g.add('causal_action',{'intent':'a'},parents=[p]);b=g.add('causal_action',{'intent':'b'},parents=[p])
    result=graph_view(tmp_path)
    assert any(e['source']==a and e['target']==b and e['kind']=='resource' and e['source_record']==p for e in result['edges'])


def test_checkpoint_preserves_numeric_arrays(tmp_path):
    import numpy as np
    from capx.experiments.unified import atomic
    atomic(tmp_path/'a.json',{'progress':np.array([1.,0.])})
    assert json.loads((tmp_path/'a.json').read_text())['progress']==[1.,0.]


def test_existing_output_is_never_overwritten(tmp_path):
    from capx.experiments.unified import execute
    p=tmp_path/'sentinel';p.write_text('keep')
    with pytest.raises(FileExistsError):execute(RunSpec(output=str(tmp_path)))
    assert p.read_text()=='keep'


def test_pair_boundary_stop_keeps_shared_initial_and_skips_next_scene(tmp_path,monkeypatch):
    import types
    from capx.experiments import unified, contract_models, graph_campaign
    class Budget:
        def __init__(self,*a):self.count=0;self.prefix='';self.ledger=types.SimpleNamespace(spent={'deepseek':0.})
    calls=[]
    out=tmp_path/'new'
    def episode(dest,spec,seed,group,budget,config,initial,expected,**kwargs):
        if initial is None:
            budget.count+=1;initial={'code':'real interface mock only','request':'test','code_hash':'same'}
        snap=expected or {'seed':seed,'state_hash':'same'}
        calls.append((group,id(initial),snap))
        if group=='G0':(out/'stop.request').touch()
        return {'group':group,'reward':0.},initial,snap
    monkeypatch.setattr(graph_campaign,'PersistentModelBudget',Budget)
    monkeypatch.setattr(contract_models,'run_episode',episode)
    monkeypatch.setattr(graph_campaign,'archive_inventory',lambda *a:{})
    unified.execute(RunSpec(output=str(out),seeds=[0,7]))
    s=json.loads((out/'status.json').read_text())
    assert [x[0] for x in calls]==['G0','G1']
    assert calls[0][1]==calls[1][1] and calls[0][2]==calls[1][2]
    assert s['state']=='stopped' and s['requests']==1 and not (out/'seed-7-repeat-0').exists()


def test_telemetry_failure_does_not_authorize_or_block_action(tmp_path):
    dest=tmp_path/'events';dest.mkdir()
    with GraphJournal(dest) as journal:
        g=DependencyGraph();n=g.add('plan');g.invalidate(n,'changed')
        with pytest.raises(RuntimeError):g.require_valid(n)
    assert journal.error in ('FileExistsError','IsADirectoryError')


def test_running_state_requires_dispatch_record_and_return_clears_it():
    rows=[{'event':{'action':'add','node':{'id':'n1','kind':'skill_intent','valid':True,'state':'pending','data':{'id':'a'}},'parents':[]}},
          {'event':{'action':'dispatch','node_id':'n1','intent':'a'}},
          {'event':{'action':'add','node':{'id':'n2','kind':'call','valid':True,'state':'done','data':{'intent':'a','returned':True}},'parents':[]}}]
    assert 'observed_execution' not in replay_graph(rows,1)['nodes']['n1']
    assert replay_graph(rows,2)['nodes']['n1']['observed_execution']=='running'
    assert replay_graph(rows,3)['nodes']['n1']['observed_execution']=='returned'
