import os
import time
import numpy as np
import pytest

from capx.execution.sandbox import SandboxProcess


@pytest.fixture
def sandbox():
    worker = SandboxProcess({'read_pose': lambda: (np.array([1., 2., 3.]), np.array([1., 0., 0., 0.]))})
    yield worker
    worker.close()


def test_numpy_rpc_and_persistent_variables(sandbox):
    result = sandbox.run('import numpy as np\np, q = read_pose()\nx = p + np.array([0., 0., 1.])\nprint(x.tolist())')
    assert result['ok'], result
    assert '[1.0, 2.0, 4.0]' in result['stdout']
    assert sandbox.pid != os.getpid()
    assert sandbox.run('print(x[2])')['stdout'].strip() == '4.0'


@pytest.mark.parametrize('code', [
    'print(env)', 'print(obs)', 'print(INPUTS)', 'print(APIS)',
    'print(globals())', 'print(read_pose.__globals__)',
    'import os\nprint(os.environ)', 'open("/etc/passwd").read()',
    'import numpy as np\nnp.load("/etc/passwd")',
    'getattr(read_pose, "__globals__")', 'import socket',
    'import numpy as np\nnp.ctypeslib.load_library("libc", "/lib")',
])
def test_hidden_state_and_host_access_rejected(sandbox, code):
    result = sandbox.run(code)
    assert not result['ok'] and result['status'] == 'policy_violation'


def test_timeout_kills_child_and_next_run_is_fresh(sandbox):
    assert sandbox.run('secret = 17')['ok']
    pid = sandbox.pid
    started = time.monotonic()
    result = sandbox.run('while True: pass', timeout=0.4)
    assert result['status'] == 'timeout'
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    result = sandbox.run('print(secret)')
    assert not result['ok'] and 'NameError' in result['stderr']
    assert sandbox.pid != pid


def test_unknown_api_and_syntax_are_not_system_errors(sandbox):
    assert sandbox.run('missing_api()')['status'] == 'runtime_error'
    assert sandbox.run('if :')['status'] == 'invalid_program'

def test_blocking_api_deadline():
    import time
    actor = SandboxProcess({'blocked': lambda: time.sleep(5)})
    start = time.monotonic()
    try:
        result = actor.run('blocked()', timeout=.3)
        assert result['status'] == 'timeout'
        assert time.monotonic() - start < 2
        assert actor.pid is None
    finally:
        actor.close()

def test_os_boundary_even_without_ast_guard(monkeypatch):
    import capx.execution.sandbox as module
    monkeypatch.setattr(module, 'validate', lambda code: None)
    actor=SandboxProcess({})
    try:
        result=actor.run("import os\nprint(os.path.exists('/home/hanyu/dqb'))\nprint(os.environ.get('DEEPSEEK_API_KEY'))")
        assert result['ok'] and result['stdout'].splitlines()==['False','None']
        result=actor.run("import socket\ns=socket.socket()\ns.settimeout(.1)\ns.connect(('127.0.0.1',8116))")
        assert not result['ok']
    finally:
        actor.close()
