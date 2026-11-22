from capx.execution.grounded_feedback import API_DOC


def test_source_record_and_id_contract_is_unambiguous():
    assert "dictionary" in API_DOC
    assert "grasp['id']" in API_DOC
    assert "robot['id']" in API_DOC
    assert 'Returns source ID' not in API_DOC
    assert 'Returns immutable parent source ID' not in API_DOC
    assert 'NEXT physical action' in API_DOC
