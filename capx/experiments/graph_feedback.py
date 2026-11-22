"""Paired graph/no-graph recovery using existing P0 execution and live DeepSeek."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from capx.experiments.pilot import dump, digest, environment, load_scene, prepare, one_program

PAIR_FIELDS = ('snapshot_hash', 'initial_prompt_hash', 'api_docs_hash', 'code_hash')

# Importing capx.envs registers simulators and caches service URL defaults.
# Keep CLI configuration ahead of those imports while exposing original objects
# (including ModelBudget.query) for callers and monkeypatch-based tests.
_LAYERED_EXPORTS = ('ModelBudget', 'initial_prompt', 'recovery_prompt',
                    'validated_program', 'instrument_tools')


def _layered_export(name):
    if name not in globals():
        from capx.experiments import layered_feedback
        globals()[name] = getattr(layered_feedback, name)
    return globals()[name]


def __getattr__(name):
    if name in _LAYERED_EXPORTS:
        return _layered_export(name)
    raise AttributeError(name)


def json_value(value):
    from capx.envs.reproducibility import json_value as encode
    return encode(value)


def _api_doc(explicit=None):
    if explicit is not None:
        return explicit
    from capx.execution.graph_feedback import GRAPH_API_DOC
    return GRAPH_API_DOC


def graph_initial_prompt(env, obs, *, api_budget=64, max_blocks=3, api_doc=None):
    prompt = _layered_export('initial_prompt')(env, obs)
    # Retain the mature prompt builder while matching the explicit episode budget.
    prompt[-1]['content'] = prompt[-1]['content'].replace(
        'Budget: 64 API calls per episode', f'Budget: {api_budget} API calls per episode')
    prompt.append({'role': 'user', 'content': _api_doc(api_doc) + f'\nEpisode budget: {api_budget} actual API calls including local recovery; {max_blocks} code blocks including initial program.'})
    return prompt


def graph_recovery_prompt(obs, code, feedback, status, mode, *, api_doc=None):
    if mode not in ('G0', 'G1'):
        raise ValueError('Expected G0 or G1')
    selected = feedback.get('diagnostic_source') if mode == 'G1' else code
    fallback = mode == 'G1' and (not isinstance(selected, str) or not selected.strip())
    if fallback:
        selected = code
    context = {'full_prompt': copy.deepcopy(obs['full_prompt'])}
    context['full_prompt'].append({'role': 'user', 'content': [{'type': 'text', 'text': _api_doc(api_doc)}]})
    prompt = _layered_export('recovery_prompt')(context, selected, feedback, status)
    conservative_fallback = feedback.get('recovery_scope', {}).get('fallback') is True
    return prompt, dict(diagnostic_source_fallback=fallback or conservative_fallback,
                        previous_code_selection='full_block' if mode == 'G0' or fallback or conservative_fallback else 'dependency_slice',
                        previous_code_hash=digest(selected))


def parse_graph_recovery(content):
    """Parse recovery without treating a missing marker as task completion.

    A complete lone fenced program is a recoverable formatting omission. Only
    standalone FINISH finishes; conflicting or malformed responses fail closed.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Empty graph recovery response')
    text = content.strip()
    if text == 'FINISH':
        return 'finish', None, None
    marker = 'REGENERATE'
    if text.startswith(marker):
        suffix = text[len(marker):]
        if not suffix or suffix[0].isspace() or suffix.startswith('```'):
            return 'regenerate', one_program(suffix.strip()), None
    if text.startswith('```'):
        return 'regenerate', one_program(text), 'missing_regenerate_marker'
    raise ValueError('Expected standalone FINISH or one complete recovery Python block')


def check_pair(reference, actual):
    for field in PAIR_FIELDS:
        if reference[field] != actual[field]:
            raise ValueError(f'Paired {field} drift')


def load_initial_run(source_run, output, seed, fault):
    """Reuse an explicitly selected historical response, never silently regenerate."""
    source = Path(source_run).resolve() / 'initial' / f'seed-{seed}-fault-{fault}.json'
    original_bytes = source.read_bytes()
    initial = json.loads(original_bytes)
    code = _layered_export('validated_program')(initial['response'])
    if code != initial.get('code') or digest(code) != initial.get('code_hash'):
        raise ValueError('Cached initial code/response/hash mismatch')
    for field in PAIR_FIELDS:
        if not isinstance(initial.get(field), str) or not initial[field]:
            raise ValueError('Cached initial missing ' + field)
    if not isinstance(initial.get('request'), str) or not initial['request']:
        raise ValueError('Cached initial lacks original model request reference')
    provenance = dict(source_run=str(Path(source_run).resolve()), initial_file=str(source),
                      initial_file_sha256=hashlib.sha256(original_bytes).hexdigest(),
                      original_request=initial['request'], generated_in_this_run=False,
                      source='reused_historical_DeepSeek_response', initial_requests_this_run=0)
    destination = Path(output) / 'initial' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(original_bytes)
    dump(destination.with_name(destination.stem + '-cache-source.json'), provenance)
    initial['cache_source'] = provenance
    return initial


def archive_sources(repo, output, names):
    """Hash and exclusively archive the same bytes; no environment/config secrets."""
    hashes = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Source archive paths must stay relative to the project')
        content = (Path(repo) / relative).read_bytes()
        target = Path(output) / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def install_monitor(env, group, dest, max_local, fault):
    from capx.execution.feedback import public_observation
    from capx.execution.graph_feedback import GraphFeedbackMonitor, GRAPH_APIS
    from capx.envs.physical_evaluation import physical_state
    count = [0]

    def observe():
        state = public_observation(env._get_observation())
        count[0] += 1
        arrays = {k: v for k, v in state.items() if not isinstance(v, dict)}
        for camera, value in state.items():
            if isinstance(value, dict):
                for key in ('intrinsics', 'pose_mat'):
                    arrays[camera + '_' + key] = value[key]
                for key, value in value['images'].items():
                    arrays[camera + '_' + key] = value
        path = dest / 'observations' / f'{count[0]:04d}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return state

    def emit(event):
        env._write('feedback', **event, observation_file=f'{count[0]:04d}.npz')
        # This separate parent-only ledger is NEVER an input to the monitor/model.
        env._write('feedback_evaluation', event_index=len(monitor.events) - 1,
                   state=physical_state(env.low_level_env))

    def graph_emit(event):
        env._write('graph_event', event=event, observation_file=f'{count[0]:04d}.npz')
        with (dest / 'graph-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, default=json_value, allow_nan=False) + '\n')

    def contract_only(*args, **kwargs):
        raise RuntimeError('Graph API must be handled by its parent monitor')

    monitor = GraphFeedbackMonitor(group, observe, emit=emit, graph_emit=graph_emit,
                                   max_local=max_local, fault=fault,
                                   site=lambda: env._actor.call_site)
    env._functions.update({name: contract_only for name in GRAPH_APIS})
    env._actor.apis.update({name: None for name in GRAPH_APIS})
    env._feedback_monitor = monitor
    return monitor


def run_episode(root, manifest, entry, group, initial, budget, args, *, fault='none'):
    """Execute one physical episode; initial is reused verbatim for the paired group."""
    from capx.utils.video_utils import _write_video
    from capx.envs.physical_evaluation import physical_state

    root, manifest = Path(root), Path(manifest)
    label = f'{group}-seed-{entry["seed"]}-fault-{fault}'
    dest = root / 'episodes' / label
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(group=group, seed=entry['seed'], fault=fault, online_vision=True,
                  model='deepseek-v4-pro', status='system_error', reward=None,
                  replans=0, replan_effective=False, blocks=[], recovery_decisions=[], initial_request=None,
                  interventions='API boundaries only; no in-flight motion cancellation')
    env = monitor = None
    start = time.monotonic()
    requests_start = budget.count
    try:
        env = environment('vision', dest / 'trace')
        env.cfg.feedback_api_budget = args.api_budget
        obs, info = env.reset(seed=entry['seed'], options={'snapshot': load_scene(manifest, entry)})
        if info['initial_state_hash'] != entry['state_hash']:
            raise ValueError('Snapshot state mismatch')
        docs = [api.combined_doc() for api in env._apis.values()] + [_api_doc()]
        prompt = graph_initial_prompt(env, obs, api_budget=args.api_budget, max_blocks=args.max_blocks)
        result.update(snapshot_hash=info['initial_state_hash'], trace_path=str(env._trace_path),
                      api_docs_hash=digest(docs), initial_prompt_hash=digest(prompt))
        dump(dest / 'api-docs.json', docs)
        dump(dest / 'initial-prompt.json', prompt)
        low = env.low_level_env
        sim = low.robosuite_env.sim
        dump(dest / 'evaluation-geometry.json', dict(
            red_half_size=low.robosuite_env.cubeA.size, green_half_size=low.robosuite_env.cubeB.size,
            base_position_world=sim.data.xpos[low.base_link_idx],
            base_rotation_world=sim.data.xmat[low.base_link_idx].reshape(3, 3),
            model_hash=entry['model_hash']))
        env.enable_video_capture(True, clear=True)
        _layered_export('instrument_tools')(env, dest)
        monitor = install_monitor(env, group, dest, args.max_local, fault)
        if initial is None:
            answer, source = budget.query(prompt, f'initial-seed-{entry["seed"]}-fault-{fault}')
            initial = dict(code=None, request=source, response=answer,
                           snapshot_hash=result['snapshot_hash'], initial_prompt_hash=digest(prompt),
                           api_docs_hash=result['api_docs_hash'])
            try:
                initial['code'] = _layered_export('validated_program')(answer)
            except (ValueError, SyntaxError) as exc:
                initial['code_error'] = str(exc)
            initial['code_hash'] = digest(initial['code'])
            dump(root / 'initial' / f'seed-{entry["seed"]}-fault-{fault}.json', initial)
        result.update(initial_request=initial['request'], code_hash=digest(initial['code']),
                      initial_cache_source=initial.get('cache_source'),
                      initial_generated_in_this_run='cache_source' not in initial)
        check_pair(initial, result)
        if initial['code'] is None:
            raise ValueError(initial['code_error'])
        code = initial['code']
        source = initial['request']
        for block in range(args.max_blocks):
            if time.monotonic() >= budget.deadline:
                result['stop_reason'] = 'global_wall_time'
                break
            dump(dest / f'block-{block}.json', dict(code=code, code_hash=digest(code),
                                                  model_request=source, block=block))
            # A new isolated actor after every boundary interruption; physical scene persists.
            env._actor.close()
            monitor.begin_program(code, block)
            started = time.monotonic()
            obs, reward, done, truncated, info = env.step(code)
            if info['execution_status'] in ('runtime_error', 'invalid_program', 'policy_violation'):
                monitor.end_block_failure(info['execution_status'], info.get('stderr', ''))
            check = monitor.finish_check() if not truncated else None
            block_result = dict(block=block, seconds=time.monotonic() - started, reward=reward,
                                execution_status=info['execution_status'], api_calls=info['api_calls'],
                                model_request=source, code_hash=digest(code), finish_check=check)
            result['blocks'].append(block_result)
            result.update(reward=reward, info=info, truncated=truncated,
                          status='success' if reward else info['execution_status']
                          if info['execution_status'] != 'ok' else 'task_failure')
            dump(dest / f'graph-block-{block}.json', monitor.graph.snapshot())
            # Preserve original P0 reward; graph interruption never sets success itself.
            if done or truncated or block + 1 >= args.max_blocks:
                if not done and not truncated and block + 1 >= args.max_blocks:
                    result['stop_reason'] = 'episode_code_block_budget'
                break
            if env._api_calls >= args.api_budget:
                result['stop_reason'] = 'episode_api_budget'
                break
            feedback = monitor.summary()
            feedback['remaining_api_calls'] = args.api_budget - env._api_calls
            feedback['remaining_code_blocks'] = args.max_blocks - block - 1
            prompt, selection = graph_recovery_prompt(obs, code, feedback, info['execution_status'], group)
            dump(dest / f'recovery-context-{block + 1}.json', dict(feedback=feedback, **selection))
            block_result['recovery_context'] = selection
            if budget.count >= budget.max_requests or time.monotonic() >= budget.deadline:
                result['stop_reason'] = 'global_model_or_wall_budget'
                break
            result['replans'] += 1
            answer, source = budget.query(prompt, f'replan-{label}-{block + 1}')
            if answer.get('finish_reason') != 'stop' or answer.get('response_model') != 'deepseek-v4-pro':
                raise ValueError('Incomplete or wrong-model recovery response')
            decision, new_code, format_repair = parse_graph_recovery(answer.get('content'))
            result['recovery_decisions'].append(dict(block=block + 1, model_request=source,
                                                     decision=decision, format_repair=format_repair))
            if decision == 'finish':
                result['finish_claim'] = answer['content']
                result['stop_reason'] = 'model_finish_claim_without_reward_change'
                break
            code = new_code
        result.update(replan_effective=bool(result['replans'] and result['reward']),
                      final_physics=physical_state(env.low_level_env))
    except (ValueError, SyntaxError) as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc)[:1000], status='invalid_program')
        if result['reward'] is None:
            result['reward'] = 0.0
    except Exception as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc)[:1000], status='system_error')
    finally:
        if monitor is not None:
            result.update(recoveries=monitor.recoveries, feedback_events=monitor.events,
                          graph_events=monitor.graph.events, fault_fired=monitor.fault_fired)
            dump(dest / 'graph-final.json', monitor.graph.snapshot())
        if env is not None:
            try:
                frames = env.get_video_frames()
                if frames:
                    _write_video(frames, str(dest), suffix=label,
                                 timestamps=list(env.low_level_env._frame_timestamps))
                    result['video'] = str(dest / f'video_{label}.mp4')
                result['actual_api_calls'] = env._api_calls
                result['http_requests'] = env._request_audit.count
            except Exception as exc:
                result.update(status='system_error', video_error=type(exc).__name__ + ': ' + str(exc))
            finally:
                env.close()
        result.update(total_seconds=time.monotonic() - start,
                      new_llm_requests=budget.count - requests_start,
                      logical_llm_requests=(1 if initial else 0) + result['replans'])
        dump(dest / 'result.json', result)
    print(label, result['status'], result['reward'], 'replans', result['replans'], flush=True)
    return result, initial


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot-manifest', type=Path)
    parser.add_argument('--initial-run', type=Path, help='Reuse validated historical DeepSeek initial responses; never fall back to a new initial request')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 21])
    parser.add_argument('--groups', nargs='+', choices=['G0', 'G1'], default=['G0', 'G1'])
    parser.add_argument('--faults', nargs='+', choices=['none', 'support_refresh'], default=['none'])
    parser.add_argument('--max-requests', type=int, default=15)
    parser.add_argument('--cost-limit-usd', type=float, default=5)
    parser.add_argument('--max-tokens', type=int, default=32768)
    parser.add_argument('--max-blocks', type=int, default=3)
    parser.add_argument('--api-budget', type=int, default=64)
    parser.add_argument('--max-local', type=int, default=2)
    parser.add_argument('--request-timeout', type=float, default=400)
    parser.add_argument('--wall-seconds', type=float, default=3600)
    parser.add_argument('--reasoning-effort', choices=['low', 'high', 'max'], default='low')
    parser.add_argument('--sam3-url', default=os.environ.get('SAM3_SERVICE_URL'))
    parser.add_argument('--graspnet-url', default=os.environ.get('GRASPNET_SERVICE_URL'))
    parser.add_argument('--pyroki-url', default=os.environ.get('PYROKI_SERVICE_URL'))
    args = parser.parse_args(argv)
    if not all([args.sam3_url, args.graspnet_url, args.pyroki_url]):
        parser.error('Set SAM3_SERVICE_URL, GRASPNET_SERVICE_URL and PYROKI_SERVICE_URL (or URL arguments)')
    if min(args.max_requests, args.cost_limit_usd, args.max_tokens, args.max_blocks,
           args.api_budget, args.request_timeout, args.wall_seconds) <= 0 or args.max_local < 0:
        parser.error('Budgets must be positive; max-local may be zero')
    for key in ('seeds', 'groups', 'faults'):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error('Duplicate ' + key)
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name, value in [('SAM3', args.sam3_url), ('GRASPNET', args.graspnet_url), ('PYROKI', args.pyroki_url)]:
        os.environ[name + '_SERVICE_URL'] = value
    repo = Path(__file__).resolve().parents[2]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    sources = ['capx/experiments/graph_feedback.py', 'capx/experiments/layered_feedback.py',
               'capx/execution/graph_feedback.py', 'capx/execution/dependency_graph.py', 'capx/execution/feedback.py',
               'capx/execution/sandbox.py', 'capx/execution/actor.py', 'capx/envs/tasks/audited_stack.py',
               'capx/envs/physical_evaluation.py']
    config['source_hashes'] = archive_sources(repo, root, sources)
    config['design'] = ('G0 and G1 share live DeepSeek initial code, snapshot, API documentation, fault, '
                        'validators, interception points and budgets; both may replan. '
                        'G0 conservatively discards the unfinished suffix; G1 selects dependency scope.')
    dump(root / 'experiment.json', config)
    started = time.monotonic()
    budget = _layered_export('ModelBudget')(root, args.max_requests, args.cost_limit_usd, args.max_tokens,
                         started + args.wall_seconds, args.reasoning_effort, args.request_timeout)
    rows = []
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    (root / 'gpu-before.txt').write_text(gpu.stdout + gpu.stderr)
    try:
        manifest = args.snapshot_manifest
        if manifest is None:
            manifest = root / 'manifest' / 'scenes.json'
            prepare(manifest, root, seeds=args.seeds)
        entries = {entry['seed']: entry for entry in json.loads(manifest.read_text())['scenes']}
        dump(root / 'snapshot-source.json', {'manifest': str(manifest.resolve()),
                                            'entries': [entries[seed] for seed in args.seeds]})
        # Validate/copy all explicitly requested caches before any new model request.
        cached = {(seed, fault): load_initial_run(args.initial_run, root, seed, fault)
                  for seed in args.seeds for fault in args.faults} if args.initial_run else {}
        for seed in args.seeds:
            for fault in args.faults:
                initial = copy.deepcopy(cached.get((seed, fault)))
                reference = None
                for group in args.groups:
                    row, initial = run_episode(root, manifest, entries[seed], group, initial, budget, args, fault=fault)
                    rows.append(row)
                    if args.initial_run and row.get('error', '').startswith('ValueError: Paired '):
                        raise RuntimeError('Cached initial no longer matches live paired inputs; refusing regeneration')
                    if all(field in row for field in PAIR_FIELDS):
                        hashes = {field: row[field] for field in PAIR_FIELDS}
                        if reference is not None:
                            check_pair(reference, hashes)
                        reference = hashes
                    if row['status'] in ('system_error', 'timeout') or row.get('stop_reason', '').startswith('global_'):
                        raise RuntimeError('Service/model/global budget failure: stop batch; inspect episode artifacts')
    finally:
        dump(root / 'summary.json', dict(results=rows, llm_requests=budget.rows,
             cost_upper_usd=budget.ledger.spent['deepseek'], seconds=time.monotonic() - started))
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        (root / 'gpu-after.txt').write_text(gpu.stdout + gpu.stderr)


if __name__ == '__main__':
    main()
