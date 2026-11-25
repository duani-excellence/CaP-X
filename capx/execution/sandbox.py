"""Fail-closed bubblewrap actor, a JSON RPC broker, and a constrained Python subset.

AST checks are defense in depth. OS namespaces deny host files/network/processes;
only named, parent-owned callbacks are reachable. No fallback to in-process exec.
"""
from __future__ import annotations
import contextlib
import threading
import ast
import json
import os
from pathlib import Path
import resource
import select
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from .protocol import encode, decode

BLOCKED = {'env', 'obs', 'APIS', 'INPUTS', 'globals', 'locals', 'vars', 'dir',
           'getattr', 'setattr', 'delattr', 'hasattr', 'open', 'eval', 'exec',
           'compile', 'input', 'type', 'object', 'super', 'help', 'breakpoint',
           'memoryview', 'classmethod', 'staticmethod', 'property'}
NUMPY = {'array', 'asarray', 'zeros', 'ones', 'eye', 'arange', 'linspace', 'pi',
         'sin', 'cos', 'tan', 'sqrt', 'abs', 'clip', 'minimum', 'maximum', 'min',
         'max', 'mean', 'sum', 'concatenate', 'stack', 'cross', 'dot', 'isfinite',
         'all', 'any', 'allclose', 'float32', 'float64', 'int32', 'int64', 'linalg'}
METHODS = {'copy', 'tolist', 'reshape', 'shape', 'dtype', 'size', 'T', 'astype',
           'append', 'extend', 'items', 'keys', 'values', 'get', 'norm'}


def validate(code):
    if len(code) > 65536:
        raise ValueError('Program too large')
    tree = ast.parse(code)
    if not tree.body:
        raise SyntaxError('Empty program')
    aliases = {'np', 'numpy'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name != 'numpy' for a in node.names):
                raise ValueError('Only numpy imports are allowed')
            aliases.update(a.asname or a.name for a in node.names)
        if isinstance(node, (ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError('Unsupported Python construct')
        if isinstance(node, ast.Name) and (node.id in BLOCKED or (node.id.startswith('_') and node.id != '_')):
            raise ValueError('Forbidden name: ' + node.id)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError('Private attributes are forbidden')
            allowed = NUMPY if isinstance(node.value, ast.Name) and node.value.id in aliases else METHODS
            if node.attr not in allowed:
                raise ValueError('Attribute not allowed: ' + node.attr)
    return tree


@contextlib.contextmanager
def callback_deadline(deadline):
    # Workers are dedicated main-thread processes. Refuse an unbounded threaded broker.
    if threading.current_thread() is not threading.main_thread():
        raise ConnectionError('P0 broker must run on the worker main thread')
    def expired(signum, frame):
        raise TimeoutError('API execution deadline exceeded')
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-started))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0]-(time.monotonic()-started)), previous_timer[1])


def _limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))


class ExecutionInterrupted(Exception):
    """Parent monitor stopped this block; physical environment remains resumable."""


class SandboxProcess:
    def __init__(self, apis, *, callback=None):
        self.apis = dict(apis)
        self.callback = callback or (lambda name, args, kwargs: self.apis[name](*args, **kwargs))
        self.process = None
        self.buffer = b''
        self.calls = 0
        self.call_site = None

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def start(self):
        if self.process and self.process.poll() is None:
            return
        if shutil.which('bwrap') is None:
            raise RuntimeError('bubblewrap unavailable; refusing unsafe execution')
        python_root = Path(sys.executable).resolve().parent.parent
        site = Path(np.__file__).resolve().parent.parent
        runner = Path(__file__).parent
        cmd = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
               '--cap-drop', 'ALL', '--clearenv', '--setenv', 'OPENBLAS_NUM_THREADS', '1',
               '--setenv', 'OMP_NUM_THREADS', '1', '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
               '--ro-bind', str(python_root), '/python',
               '--ro-bind', '/usr/lib', '/usr/lib', '--ro-bind', '/lib', '/lib',
               '--ro-bind', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
               '--tmpfs', '/tmp', '--dir', '/site', '--dir', '/runner',
               '--ro-bind', str(site / 'numpy'), '/site/numpy',
               '--ro-bind', str(site / 'numpy.libs'), '/site/numpy.libs',
               '--ro-bind', str(runner / 'actor.py'), '/runner/actor.py',
               '--ro-bind', str(runner / 'protocol.py'), '/runner/protocol.py',
               '--chdir', '/tmp', '/python/bin/python3.12', '-I', '-S', '-B', '/runner/actor.py']
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
                                        preexec_fn=_limits, env={'PATH': '/usr/bin:/bin'})
        self.buffer = b''

    def _send(self, message):
        body = (json.dumps(message, allow_nan=False) + '\n').encode()
        self.process.stdin.write(body)
        self.process.stdin.flush()

    def _receive(self, deadline):
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('Actor execution deadline exceeded')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                detail = self.process.stderr.read(4096).decode(errors='replace')
                raise RuntimeError('Actor unavailable: ' + detail)
            self.buffer += chunk
            if len(self.buffer) > 1048576:
                raise ValueError('RPC response exceeds budget')
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def run(self, code, *, timeout=90.0):
        self.call_site = None
        calls = 0
        source_lines = code.splitlines()
        out = {'ok': False, 'stdout': '', 'stderr': '', 'status': 'runtime_error', 'api_calls': 0}
        try:
            validate(code)
        except SyntaxError as exc:
            return dict(out, status='invalid_program', stderr=str(exc))
        except ValueError as exc:
            return dict(out, status='policy_violation', stderr=str(exc))
        try:
            self.start()
            deadline = time.monotonic() + timeout
            self._send({'code': code, 'apis': list(self.apis)})
            calls = 0
            while True:
                message = self._receive(deadline)
                if message.get('type') == 'done':
                    return dict(out, ok=bool(message['ok']), stdout=message.get('stdout', ''),
                                stderr=message.get('stderr', ''), status='ok' if message['ok'] else 'runtime_error',
                                api_calls=calls)
                if message.get('type') != 'call' or message.get('name') not in self.apis:
                    raise ValueError('Invalid API request')
                calls += 1
                self.calls += 1
                if calls > 64 or self.calls > 256:
                    raise ValueError('API call budget exceeded')
                name = message['name']
                # Provenance is not authority: API allowlist and budgets were checked above.
                line = message.get('site_line')
                self.call_site = ({'line': line, 'source_line': source_lines[line - 1]}
                                  if type(line) is int and 1 <= line <= len(source_lines)
                                  else None)
                try:
                    with callback_deadline(deadline):
                        result = self.callback(name, decode(message['args']), decode(message['kwargs']))
                    self._send({'ok': True, 'result': encode(result)})
                except (TimeoutError, ConnectionError, ExecutionInterrupted):
                    raise
                except Exception as exc:
                    self._send({'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)[:2000]})
                if time.monotonic() >= deadline:
                    raise TimeoutError('API execution deadline exceeded')
        except ExecutionInterrupted as exc:
            self.close()
            return dict(out, status='interrupted', stderr=str(exc), api_calls=calls)
        except TimeoutError as exc:
            self.close()
            return dict(out, status='timeout', stderr=str(exc), api_calls=calls)
        except ValueError as exc:
            self.close()
            return dict(out, status='policy_violation', stderr=str(exc), api_calls=calls)
        except Exception as exc:
            self.close()
            return dict(out, status='system_error', stderr=str(exc), api_calls=calls)

    def close(self):
        if self.process:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()
        self.process = None
        self.buffer = b''
