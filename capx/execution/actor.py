"""Trusted bootstrap. Mounted alone with NumPy and stdlib in a bubblewrap root."""
import sys
sys.path.insert(0, '/site')
sys.path.insert(0, '/runner')
import builtins
import contextlib
import io
import json
import traceback
import numpy as np
from protocol import encode, decode

wire_in, wire_out = sys.stdin, sys.stdout

def send(message):
    wire_out.write(json.dumps(message, allow_nan=False) + '\n')
    wire_out.flush()


def proxy(name):
    def call(*args, **kwargs):
        # Trusted bootstrap provenance; the user namespace cannot access sys/private frames.
        site_line = sys._getframe(1).f_lineno
        send({'type': 'call', 'name': name, 'args': encode(args), 'kwargs': encode(kwargs),
              'site_line': site_line})
        response = json.loads(wire_in.readline())
        if not response['ok']:
            raise RuntimeError(response['error'])
        return decode(response['result'])
    return call

class BoundedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 65536:
            raise RuntimeError('stdout budget exceeded')
        return super().write(text)

namespace = {'__name__': 'policy', '__builtins__': builtins.__dict__}
for line in wire_in:
    command = json.loads(line)
    for name in command['apis']:
        namespace[name] = proxy(name)
    stdout, stderr = BoundedOutput(), BoundedOutput()
    ok = True
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(command['code'], '<policy>', 'exec'), namespace, namespace)
    except BaseException as exc:
        ok = False
        # Do not serialize traceback locals or host objects.
        stderr.write(type(exc).__name__ + ': ' + str(exc)[:2000])
    send({'type': 'done', 'ok': ok, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue()})
