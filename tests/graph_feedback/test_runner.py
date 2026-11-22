"""Cheap runner checks: no model requests, rendering or simulation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from capx.experiments import graph_feedback as runner


def env_prompt():
    return SimpleNamespace(_full_prompt=[{'role': 'user', 'content': 'Stack task and original API docs'}])


def test_shared_initial_prompt_has_same_contract_without_group_or_truth():
    env = env_prompt()
    obs = {'robot_joint_pos': [0] * 7, 'robot_cartesian_pos': [0] * 8,
           'cube_poses': 'PRIVATE_EVALUATOR', 'raw_success': True}
    before = copy.deepcopy(env._full_prompt)
    g0 = runner.graph_initial_prompt(env, obs, api_budget=64, api_doc='GRAPH CONTRACT')
    g1 = runner.graph_initial_prompt(env, obs, api_budget=64, api_doc='GRAPH CONTRACT')
    assert g0 == g1
    assert env._full_prompt == before
    assert 'GRAPH CONTRACT' in str(g0)
    assert 'PRIVATE_EVALUATOR' not in str(g0) and 'raw_success' not in str(g0)


@pytest.mark.parametrize('mode,expected', [('G0', 'unrelated_old_call()'), ('G1', 'related_call()')])
def test_recovery_context_selection_and_same_graph_documentation(mode, expected):
    obs = {'full_prompt': env_prompt()._full_prompt, 'raw_success': True, 'cube_poses': 'PRIVATE'}
    feedback = {'diagnostic_source': 'related_call()', 'status': 'FAIL',
                'recovery_scope': {'recompute': ['place'], 'keep': ['calibration']}}
    prompt, meta = runner.graph_recovery_prompt(obs, 'unrelated_old_call()', feedback,
                                               'interrupted', mode, api_doc='GRAPH CONTRACT')
    assert expected in str(prompt)
    assert 'GRAPH CONTRACT' in str(prompt) and 'calibration' in str(prompt)
    assert 'PRIVATE' not in str(prompt) and 'raw_success' not in str(prompt)
    assert not meta['diagnostic_source_fallback']
    if mode == 'G1': assert 'unrelated_old_call()' not in str(prompt)


def test_graph_missing_dependency_source_falls_back_to_full_program():
    prompt, meta = runner.graph_recovery_prompt({'full_prompt': env_prompt()._full_prompt},
        'whole_program()', {'status': 'FAIL'}, 'interrupted', 'G1', api_doc='GRAPH CONTRACT')
    assert 'whole_program()' in str(prompt)
    assert meta['diagnostic_source_fallback']


@pytest.mark.parametrize('field', ['snapshot_hash', 'initial_prompt_hash', 'api_docs_hash', 'code_hash'])
def test_pair_hash_drift_is_rejected(field):
    hashes = dict(snapshot_hash='snapshot', initial_prompt_hash='prompt', api_docs_hash='docs', code_hash='code')
    runner.check_pair(hashes, hashes)
    mismatch = dict(hashes); mismatch[field] = 'changed'
    with pytest.raises(ValueError, match='Paired'):
        runner.check_pair(hashes, mismatch)


def test_budget_parser_defaults_and_duplicate_rejection(monkeypatch, tmp_path):
    for key in ['SAM3', 'GRASPNET', 'PYROKI']:
        monkeypatch.setenv(key + '_SERVICE_URL', 'http://localhost:1234')
    args = runner.parse_args(['--output', str(tmp_path/'new')])
    assert args.groups == ['G0', 'G1'] and args.max_blocks == 3
    assert args.max_tokens == 32768 and args.reasoning_effort == 'low'
    assert args.faults == ['none'] and args.request_timeout == 400
    for extra in [['--seeds', '2', '2'], ['--faults', 'none', 'none'],
                  ['--groups', 'G0', 'G0'], ['--api-budget', '0']]:
        with pytest.raises(SystemExit): runner.parse_args(['--output', str(tmp_path/'new')] + extra)


def test_cli_help_requires_no_service_or_simulator(monkeypatch, capsys):
    for key in ['SAM3', 'GRASPNET', 'PYROKI']:
        monkeypatch.delenv(key + '_SERVICE_URL', raising=False)
    with pytest.raises(SystemExit) as stop: runner.main(['--help'])
    assert stop.value.code == 0
    assert '--faults' in capsys.readouterr().out


def test_existing_output_is_rejected_before_any_experiment(monkeypatch, tmp_path):
    for key in ['SAM3', 'GRASPNET', 'PYROKI']:
        monkeypatch.setenv(key + '_SERVICE_URL', 'http://localhost:1234')
    monkeypatch.setattr(runner, 'environment', lambda *a: pytest.fail('environment started'))
    monkeypatch.setattr(runner, 'ModelBudget', lambda *a: pytest.fail('budget instantiated'))
    with pytest.raises(FileExistsError): runner.main(['--output', str(tmp_path)])


def test_real_response_parser_keeps_reasoning_out_of_program():
    answer = dict(content='```python\nprobe()\n```', reasoning_content='PRIVATE_REASONING',
                  finish_reason='stop', response_model='deepseek-v4-pro')
    assert runner.validated_program(answer) == 'probe()'
    answer['finish_reason'] = 'length'
    with pytest.raises(ValueError): runner.validated_program(answer)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('recover', [False, True])
@pytest.mark.parametrize('code_only', [False, True])
def test_two_groups_share_one_model_program_and_restart_actor_per_block(monkeypatch, tmp_path, recover, cached, code_only):
    import time
    import numpy as np
    import capx.envs.physical_evaluation as evaluation
    created = []
    class FakeEnv:
        def __init__(self):
            self.cfg = SimpleNamespace()
            self._full_prompt = env_prompt()._full_prompt
            self._apis = {'vision': SimpleNamespace(combined_doc=lambda: 'vision docs')}
            self._trace_path = tmp_path/'fake-trace.jsonl'
            self.closed = False
            self.actor_closes = 0
            self.executed = []
            self._api_calls = 0
            self._request_audit = SimpleNamespace(count=0)
            self._actor = SimpleNamespace(close=self.close_actor)
            data = SimpleNamespace(xpos=np.zeros((1, 3)), xmat=np.eye(3).reshape(1, 9))
            self.low_level_env = SimpleNamespace(base_link_idx=0,
                robosuite_env=SimpleNamespace(sim=SimpleNamespace(data=data),
                    cubeA=SimpleNamespace(size=[.02]*3), cubeB=SimpleNamespace(size=[.025]*3)))
        def close_actor(self): self.actor_closes += 1
        def reset(self, **kwargs): return {'full_prompt': self._full_prompt}, {'initial_state_hash': 'state'}
        def enable_video_capture(self, *a, **kw): pass
        def step(self, code):
            self.executed.append(code)
            self._api_calls += 1
            done = not recover or len(self.executed) == 2
            return {'full_prompt': self._full_prompt}, float(done), done, False, {'execution_status': 'ok' if done else 'interrupted', 'api_calls': self._api_calls}
        def get_video_frames(self): return []
        def close(self): self.closed = True
    def make_env(*args):
        obj = FakeEnv(); created.append(obj); return obj
    class FakeBudget:
        count = 0
        deadline = time.monotonic() + 100
        max_requests = 5
        def query(self, prompt, label):
            if cached and label.startswith('initial-'): pytest.fail('cached initial was regenerated')
            self.count += 1
            content = '```python\nprobe()\n```' if label.startswith('initial-') else 'REGENERATE\n```python\nrecover()\n```'
            if code_only and not label.startswith('initial-'): content = content.removeprefix('REGENERATE\n')
            return dict(content=content, finish_reason='stop', response_model='deepseek-v4-pro'), 'mock-provider-receipt'
    def fake_monitor(*args):
        return SimpleNamespace(begin_program=lambda code, block: None,
            finish_check=lambda: {'status': 'PASS'}, graph=SimpleNamespace(snapshot=lambda: {'nodes': []}, events=[]),
            recoveries=[], events=[], fault_fired=False,
            summary=lambda: {'diagnostic_source': 'probe()', 'status': 'FAIL'})
    monkeypatch.setattr(runner, 'environment', make_env)
    monkeypatch.setattr(runner, 'load_scene', lambda *a: {'state_hash': 'state'})
    monkeypatch.setattr(runner, '_api_doc', lambda explicit=None: 'GRAPH CONTRACT')
    monkeypatch.setattr(runner, 'instrument_tools', lambda *a: None)
    monkeypatch.setattr(runner, 'install_monitor', fake_monitor)
    monkeypatch.setattr(evaluation, 'physical_state', lambda *a: {'PRIVATE_EVALUATOR': True})
    args = SimpleNamespace(api_budget=64, max_local=2, max_blocks=3)
    budget = FakeBudget(); initial = None; rows = []
    if cached:
        initial = dict(code='probe()', code_hash=runner.digest('probe()'), request='historical-receipt',
            response=dict(content='```python\nprobe()\n```', finish_reason='stop', response_model='deepseek-v4-pro'),
            snapshot_hash='state', api_docs_hash=runner.digest(['vision docs', 'GRAPH CONTRACT']),
            initial_prompt_hash=runner.digest(runner.graph_initial_prompt(env_prompt(), {'full_prompt': env_prompt()._full_prompt}, max_blocks=3)),
            cache_source={'generated_in_this_run': False})
    entry = dict(seed=7, state_hash='state', model_hash='model')
    for group in ['G0', 'G1']:
        row, initial = runner.run_episode(tmp_path, tmp_path/'manifest.json', entry, group,
                                          initial, budget, args, fault='support_refresh')
        rows.append(row)
    assert budget.count == (0 if cached else 1) + (2 if recover else 0)
    assert [row['new_llm_requests'] for row in rows] == [(0 if cached else 1) + int(recover), int(recover)]
    assert all(row['initial_generated_in_this_run'] is (not cached) for row in rows)
    if cached: assert all(row['initial_request'] == 'historical-receipt' for row in rows)
    assert all(row['status'] == 'success' for row in rows)
    if recover:
        assert all(row['recovery_decisions'][0]['format_repair'] == ('missing_regenerate_marker' if code_only else None) for row in rows)
    runner.check_pair(rows[0], rows[1])
    expected = ['probe()', 'recover()'] if recover else ['probe()']
    assert all(env.executed == expected and env.actor_closes == len(expected) and env.closed for env in created)
    assert initial['code'] == 'probe()'


def stored_initial(tmp_path):
    source = tmp_path/'historical'
    path = source/'initial'/'seed-7-fault-none.json'
    path.parent.mkdir(parents=True)
    answer = dict(content='```python\nprobe()\n```', reasoning_content='never execute reasoning',
                  finish_reason='stop', response_model='deepseek-v4-pro')
    initial = dict(response=answer, code='probe()', code_hash=runner.digest('probe()'),
                   snapshot_hash='snapshot', initial_prompt_hash='prompt', api_docs_hash='docs',
                   request=str(source/'llm'/'001-initial-seed-7-fault-none'))
    path.write_text(json.dumps(initial))
    return source, path, initial


def test_load_initial_preserves_historical_source_and_does_not_query(monkeypatch, tmp_path):
    source, path, initial = stored_initial(tmp_path)
    output = tmp_path/'new'
    monkeypatch.setattr(runner.ModelBudget, 'query', lambda *a: pytest.fail('unexpected paid request'))
    cached = runner.load_initial_run(source, output, 7, 'none')
    assert cached['code'] == initial['code']
    assert cached['request'] == initial['request']
    assert cached['cache_source']['generated_in_this_run'] is False
    assert (output/'initial'/path.name).read_bytes() == path.read_bytes()
    provenance = json.loads((output/'initial'/'seed-7-fault-none-cache-source.json').read_text())
    assert provenance['initial_file'] == str(path.resolve())
    runner.check_pair(initial, cached)
    with pytest.raises(FileExistsError): runner.load_initial_run(source, output, 7, 'none')


@pytest.mark.parametrize('mutation', ['code', 'code_hash', 'content', 'finish_reason', 'response_model', 'snapshot_hash'])
def test_cached_initial_corruption_fails_closed_before_new_generation(monkeypatch, tmp_path, mutation):
    source, path, initial = stored_initial(tmp_path)
    if mutation == 'snapshot_hash': del initial['snapshot_hash']
    elif mutation in ('content', 'finish_reason', 'response_model'): initial['response'][mutation] = 'corrupt'
    else: initial[mutation] = 'corrupt'
    path.write_text(json.dumps(initial))
    monkeypatch.setattr(runner.ModelBudget, 'query', lambda *a: pytest.fail('unexpected paid request'))
    with pytest.raises((ValueError, KeyError)):
        runner.load_initial_run(source, tmp_path/'new', 7, 'none')
    assert not (tmp_path/'new'/'initial'/path.name).exists()


def test_missing_cached_initial_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError): runner.load_initial_run(tmp_path/'missing', tmp_path/'new', 7, 'none')


def test_source_archives_match_exact_original_bytes_and_are_exclusive(tmp_path):
    import hashlib
    repo = tmp_path/'repo'; output = tmp_path/'run'
    names = ['capx/execution/graph_feedback.py', 'capx/envs/physical_evaluation.py']
    for name in names:
        p = repo/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(b'# original bytes\n\x23\n')
    hashes = runner.archive_sources(repo, output, names)
    for name in names:
        assert (output/'sources'/name).read_bytes() == (repo/name).read_bytes()
        assert hashes[name] == hashlib.sha256((repo/name).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError): runner.archive_sources(repo, output, names)
    assert not (output/'sources'/'.env.local').exists()


def test_initial_run_flag_is_optional(monkeypatch, tmp_path):
    for key in ['SAM3', 'GRASPNET', 'PYROKI']:
        monkeypatch.setenv(key + '_SERVICE_URL', 'http://localhost:1234')
    assert runner.parse_args(['--output', str(tmp_path/'new')]).initial_run is None
    assert runner.parse_args(['--output', str(tmp_path/'new'), '--initial-run', str(tmp_path/'old')]).initial_run == tmp_path/'old'
