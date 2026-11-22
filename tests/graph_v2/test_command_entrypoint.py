from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/local/graph_experiment.sh'


def test_entrypoint_help_from_outside_repository():
    result = subprocess.run(['bash', str(SCRIPT), '--help'], cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'pair' in result.stdout and 'prepare-development' in result.stdout
    assert 'resume' in result.stdout and 'analyze' in result.stdout


def test_pair_help_uses_existing_runner():
    result = subprocess.run(['bash', str(SCRIPT), 'pair', '--help'], cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--task' in result.stdout and '--snapshot' in result.stdout
    assert '--groups' in result.stdout and '--fault' in result.stdout


def test_invalid_command_fails_without_starting_experiment():
    result = subprocess.run(['bash', str(SCRIPT), 'train'], cwd='/tmp', capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Unknown command' in result.stderr
