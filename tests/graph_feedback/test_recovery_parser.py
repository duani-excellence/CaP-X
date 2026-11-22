"""Recovery decisions must not turn valid code-only responses into FINISH."""
import pytest
from capx.experiments import graph_feedback as runner


@pytest.mark.parametrize('content', ['FINISH', '  FINISH\n'])
def test_only_explicit_standalone_finish_finishes(content):
    assert runner.parse_graph_recovery(content)==('finish',None,None)


@pytest.mark.parametrize('prefix', ['REGENERATE\n','REGENERATE'])
def test_regenerate_reuses_existing_program_validation(prefix):
    assert runner.parse_graph_recovery(prefix+'```python\nprobe()\n```')==('regenerate','probe()',None)


def test_complete_code_only_is_repaired_and_never_misclassified_finish():
    assert runner.parse_graph_recovery('```python\nprobe()\n```')==('regenerate','probe()','missing_regenerate_marker')


def test_finish_inside_python_string_is_not_a_finish_claim():
    assert runner.parse_graph_recovery('```python\nprint("FINISH")\n```')[0]=='regenerate'


@pytest.mark.parametrize('content', [None,'','done','probe()',
    'FINISH\n```python\nprobe()\n```',
    'REGENERATE\nFINISH\n```python\nprobe()\n```',
    '```python\nprobe()\n```\nFINISH',
    '```python\nprobe()\n```\n```python\nagain()\n```',
    '```python\nif True:\n```',
    'I think we are FINISH',
    'REGENERATEX\n```python\nprobe()\n```'])
def test_missing_ambiguous_conflicting_or_invalid_response_fails_closed(content):
    with pytest.raises((ValueError,SyntaxError)):
        runner.parse_graph_recovery(content)
