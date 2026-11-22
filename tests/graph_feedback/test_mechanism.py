"""Offline contracts with real isolated actors, never robot success measurements."""
import json
import pytest

from capx.experiments.graph_mechanism import run_case, main


@pytest.mark.parametrize('scenario', ['support_refresh', 'local_alignment'])
@pytest.mark.parametrize('mode', ['G0', 'G1'])
def test_real_actor_dependency_block_revalidation_and_no_replay(tmp_path, mode, scenario):
    row = run_case(tmp_path, mode, scenario)
    assert row['passed'], row
    assert row['online_vision'] is False and row['llm_requests'] == 0
    assert row['physical_reward'] is None
    assert row['initial']['status'] == 'interrupted'
    assert row['recovery']['status'] == 'ok'
    assert row['replay_attempt']['status'] == 'interrupted'
    assert row['old_place_invalid'] and row['old_retreat_invalid']
    assert row['unrelated_result_retained'] == (mode == 'G1')
    assert row['no_completed_motion_replayed']
    assert row['blocked_action_not_dispatched']
    assert len(set(row['actor_pids'])) == 3
    assert all(call['site']['line'] >= 1 and call['site']['source_line'] for call in row['broker_calls'])
    assert (tmp_path / mode / scenario / 'graph-events.jsonl').exists()


def test_command_refuses_existing_output_before_running(tmp_path):
    with pytest.raises(FileExistsError): main(['--output', str(tmp_path)])


def test_help_does_not_require_vision_or_model(capsys):
    with pytest.raises(SystemExit) as stop: main(['--help'])
    assert stop.value.code == 0
    assert '--scenarios' in capsys.readouterr().out
