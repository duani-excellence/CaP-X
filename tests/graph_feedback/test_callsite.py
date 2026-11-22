"""Call-site metadata is provenance, never an API capability."""
import pytest

from capx.execution.sandbox import ExecutionInterrupted, SandboxProcess


def test_real_actor_direct_and_function_call_sites_preserve_callback_signature():
    seen = []
    worker = None
    def callback(name, args, kwargs):
        seen.append((name, args, kwargs, worker.call_site.copy()))
        return 7
    worker = SandboxProcess({'probe': None}, callback=callback)
    code = 'x = probe(1, mode="direct")\ndef nested():\n    return probe(2, mode="function")\ny = nested()\nprint(x + y)'
    try:
        result = worker.run(code, timeout=5)
        assert result['ok'], result
        assert result['stdout'].strip() == '14'
        assert seen == [
            ('probe', (1,), {'mode': 'direct'}, {'line': 1, 'source_line': 'x = probe(1, mode="direct")'}),
            ('probe', (2,), {'mode': 'function'}, {'line': 3, 'source_line': '    return probe(2, mode="function")'}),
        ]
        assert worker.run('pass', timeout=5)['ok']
        assert worker.call_site is None
        worker.call_site = {'line': 100, 'source_line': 'old'}
        assert worker.run('if :')['status'] == 'invalid_program'
        assert worker.call_site is None
    finally:
        worker.close()


def test_site_metadata_does_not_make_broker_interrupt_catchable():
    seen = []
    worker = None
    def callback(name, args, kwargs):
        seen.append((name, worker.call_site))
        raise ExecutionInterrupted('boundary blocked')
    worker = SandboxProcess({'release': None, 'retreat': None}, callback=callback)
    try:
        result = worker.run('try:\n    release()\nexcept Exception:\n    retreat()\n', timeout=5)
        assert result['status'] == 'interrupted'
        assert seen == [('release', {'line': 2, 'source_line': '    release()'})]
        assert worker.pid is None
    finally:
        worker.close()


@pytest.mark.parametrize('site', [None, True, False, 0, -1, 2, 1.0, '1', {}, []])
def test_invalid_metadata_is_ignored_without_changing_legacy_callback(monkeypatch, site):
    seen = []
    worker = SandboxProcess({'probe': None})
    worker.callback = lambda name, args, kwargs: seen.append((name, args, kwargs, worker.call_site))
    worker.call_site = {'line': 99, 'source_line': 'stale'}
    messages = iter([
        {'type': 'call', 'name': 'probe', 'args': [], 'kwargs': {}, 'site_line': site},
        {'type': 'done', 'ok': True},
    ])
    monkeypatch.setattr(worker, 'start', lambda: None)
    monkeypatch.setattr(worker, '_send', lambda _: None)
    monkeypatch.setattr(worker, '_receive', lambda _: next(messages))
    assert worker.run('probe()')['ok']
    assert seen == [('probe', [], {}, None)]


def test_absent_site_metadata_remains_compatible(monkeypatch):
    worker = SandboxProcess({'probe': lambda: 1})
    messages = iter([
        {'type': 'call', 'name': 'probe', 'args': [], 'kwargs': {}},
        {'type': 'done', 'ok': True},
    ])
    monkeypatch.setattr(worker, 'start', lambda: None)
    monkeypatch.setattr(worker, '_send', lambda _: None)
    monkeypatch.setattr(worker, '_receive', lambda _: next(messages))
    assert worker.run('probe()')['ok']
    assert worker.call_site is None


def test_valid_site_does_not_authorize_unknown_api(monkeypatch):
    seen = []
    worker = SandboxProcess({'probe': None}, callback=lambda *a: seen.append(a))
    monkeypatch.setattr(worker, 'start', lambda: None)
    monkeypatch.setattr(worker, '_send', lambda _: None)
    monkeypatch.setattr(worker, '_receive', lambda _: {
        'type': 'call', 'name': 'hidden_physics', 'args': [], 'kwargs': {}, 'site_line': 1,
    })
    result = worker.run('probe()')
    assert result['status'] == 'policy_violation'
    assert seen == []


@pytest.mark.parametrize('code', ['import sys\nprint(sys._getframe())', 'print(probe.__globals__)'])
def test_call_site_bootstrap_does_not_expose_private_state(code):
    worker = SandboxProcess({'probe': None})
    assert worker.run(code)['status'] == 'policy_violation'
