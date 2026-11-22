"""Integration regression checks: public table evidence must control graph permits."""
import json
import pytest
from capx.execution.sandbox import ExecutionInterrupted
from capx.experiments.graph_campaign import json_value
from test_table_boundary import ready,checker


def test_invalidated_table_region_is_not_silently_reused_at_release():
    m,state,calls,raw,p,table=ready();m.table_evidence_fn=checker('PASS')
    m.graph.invalidate([table['id']],'earlier_clearance_failed')
    with pytest.raises(ExecutionInterrupted):m.invoke('open_gripper',[],{},raw)
    assert not calls and m.closed


def test_table_check_has_nonheld_size_as_graph_dependency():
    m,state,calls,raw,p,table=ready()
    geometry=m.graph.add('verification_fact',{'type':'public_size','object':'red cube','height_m':.04})
    def check(*a,**kw):
        r=checker('PASS')(*a,**kw);r['evidence']['geometry_source_ids']=[geometry];return r
    m.table_evidence_fn=check;m._current_table_check(table['id'])
    table_check=next(n['id'] for n in reversed(list(m.graph.nodes.values())) if n['data'].get('check')=='table_clearance')
    assert geometry in m.graph.ancestors([table_check])


def test_actual_release_is_downstream_of_current_table_clearance():
    m,state,calls,raw,p,table=ready();m.table_evidence_fn=checker('PASS')
    m.invoke('open_gripper',[],{},raw)
    check=next(n['id'] for n in reversed(list(m.graph.nodes.values())) if n['data'].get('check')=='table_clearance')
    assert calls[-1][0]=='open_gripper'
    assert check in m.graph.ancestors([m.call_history[-1]])


def test_public_table_validation_metrics_are_json_serializable():
    m,state,calls,raw,p,table=ready();m.table_evidence_fn=checker('PASS')
    m._current_table_check(table['id'])
    json.dumps(m.graph.snapshot(),default=json_value,allow_nan=False)
