"""Read-only service identity and dependency inventory, with no inference calls.

Only configured service URLs are loaded. Environment files and /proc/*/environ
are never read. Source hashes describe files currently on disk, not a proof of
which bytes a long-lived Python process loaded.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_BODY = 4 * 1024 * 1024
DEPENDENCIES = ('torch', 'mujoco', 'robosuite', 'numpy', 'jax', 'jaxlib', 'open3d', 'fastapi', 'uvicorn', 'sam3', 'pyroki', 'contact_graspnet')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def service_urls(config):
    services = config.get('services', {})
    result = {}
    for name in ('sam3', 'graspnet', 'pyroki'):
        url = services.get(name)
        if not isinstance(url, str):
            raise ValueError('configuration must explicitly specify all three service URLs')
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('service URLs must be HTTP(S) and contain no credentials, query, or fragment')
        _ = parts.port
        result[name] = url.rstrip('/')
    return result


def get_endpoint(url, timeout):
    start = time.monotonic()
    row = {'url': url, 'method': 'GET'}
    body = None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method='GET'), timeout=timeout) as response:
            raw = response.read(MAX_BODY + 1)
            row.update(http_status=response.status, body_bytes=len(raw), body_sha256=sha256(raw), body_truncated=len(raw) > MAX_BODY)
            if len(raw) <= MAX_BODY:
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeError):
                    row['json_parse_error'] = True
    except urllib.error.HTTPError as exc:
        row.update(http_status=exc.code, error_type=type(exc).__name__)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        row.update(http_status=None, error_type=type(exc).__name__)
    row['elapsed_seconds'] = round(time.monotonic() - start, 6)
    return row, body


def run_command(argv, timeout=10):
    try:
        r = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return {'returncode': r.returncode, 'output': r.stdout[:16384]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': None, 'error_type': type(exc).__name__, 'output': ''}


def listener_pids(url):
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in ('localhost', '127.0.0.1', '::1'):
        return [], 'remote_endpoint'
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    result = run_command(['ss', '-ltnpH'])
    pids = set()
    for line in result['output'].splitlines():
        fields = line.split()
        if len(fields) > 3 and fields[3].rsplit(':', 1)[-1] == str(port):
            pids.update(int(x) for x in re.findall(r'pid=(\d+)', line))
    return sorted(pids), None if pids else 'listener_pid_unavailable'


def resolve_module(module, roots):
    if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', module):
        return None
    relative = Path(*module.split('.'))
    for root in roots:
        for candidate in (Path(root) / relative.with_suffix('.py'), Path(root) / relative / '__init__.py'):
            if candidate.is_file():
                return candidate.resolve()
    return None



def source_closure(entries, roots):
    """Hashable local import closure without executing code or reading secrets.

    Includes package initializers and relative imports. Dynamic imports and
    external installed packages remain separately inventoried limitations.
    """
    roots=[Path(p).resolve() for p in roots];pending=[Path(p).resolve() for p in entries];found=set()
    while pending:
        path=pending.pop()
        if path in found or not path.is_file():continue
        found.add(path)
        if path.suffix!='.py':continue
        package=None
        for root in roots:
            try:parts=path.relative_to(root).with_suffix('').parts
            except ValueError:continue
            package=list(parts[:-1])
            for n in range(1,len(parts)):
                init=root.joinpath(*parts[:n],'__init__.py')
                if init.is_file() and init.resolve() not in found:pending.append(init.resolve())
            break
        try:tree=ast.parse(path.read_text())
        except (SyntaxError,UnicodeError):continue
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    if package is None or node.level>len(package)+1:continue
                    base='.'.join(package[:len(package)-node.level+1]+([base] if base else []))
                modules=([base] if base else [])+[base+'.'+a.name if base else a.name for a in node.names if a.name!='*']
            for module in modules:
                source=resolve_module(module,roots)
                if source and source not in found:pending.append(source)
    return sorted(found)


def source_record(path, started_at=None):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'sha256': sha256(path.read_bytes()), 'bytes': stat.st_size,
            'mtime_unix': stat.st_mtime,
            'modified_after_process_start': None if started_at is None else stat.st_mtime > started_at}


def package_versions(paths=None):
    result = {name: None for name in DEPENDENCIES}
    try:
        distributions = importlib.metadata.distributions(path=paths) if paths is not None else importlib.metadata.distributions()
        wanted = {x.replace('_', '-'): x for x in DEPENDENCIES}
        for dist in distributions:
            name = str(dist.metadata.get('Name', '')).lower().replace('_', '-')
            if name in wanted:
                result[wanted[name]] = dist.version
    except (OSError, ValueError):
        pass
    return result


def process_start_time(ticks, integer_boot, proc_root):
    """Avoid /proc/stat's truncated btime marking pre-launch files as edits.

    CLOCK_BOOTTIME includes suspend time, like Linux process start ticks. The
    measured wall-clock offset keeps subsecond precision. Synthetic proc trees
    and platforms without this clock use the conservative integer fallback.
    """
    hz=os.sysconf('SC_CLK_TCK')
    if Path(proc_root).resolve()==Path('/proc') and hasattr(time,'CLOCK_BOOTTIME'):
        before=time.time()
        elapsed=time.clock_gettime(time.CLOCK_BOOTTIME)
        after=time.time()
        return {'started_at_unix':(before+after)/2-elapsed+ticks/hz,
                'started_at_timebase':'CLOCK_BOOTTIME',
                'started_at_uncertainty_seconds':1/hz+(after-before)/2}
    return {'started_at_unix':integer_boot+ticks/hz,
            'started_at_timebase':'proc_stat_integer_btime',
            'started_at_uncertainty_seconds':1+1/hz}


def process_identity(pid, proc_root=Path('/proc')):
    proc_root = Path(proc_root)
    proc = proc_root / str(pid)
    row = {'pid': pid, 'environment_read': False,
           'source_hash_scope': 'current_disk_files; loaded process byte identity is not guaranteed'}
    try:
        cwd = (proc / 'cwd').resolve(strict=True)
        executable = (proc / 'exe').resolve(strict=True)
        raw = (proc / 'cmdline').read_bytes()
        argv = raw.decode('utf-8', errors='replace').rstrip('\0').split('\0')
        row.update(cwd=str(cwd), executable=str(executable), argv_sha256=sha256(raw))
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            ticks = int(fields[19])
            boot = int(re.search(r'^btime (\d+)$', (proc_root / 'stat').read_text(), re.M).group(1))
            row.update(process_start_time(ticks, boot, proc_root))
        except (OSError, ValueError, IndexError, AttributeError):
            row['started_at_unix'] = None
        invoked = Path(argv[0])
        if not invoked.is_absolute():
            invoked = cwd / invoked
        row['python_invocation'] = str(invoked)
        site_paths = list((invoked.parent.parent / 'lib').glob('python*/site-packages'))
        roots = [cwd, cwd / 'src', *site_paths]
        entry = None
        if '-m' in argv:
            i = argv.index('-m')
            if i + 1 < len(argv) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', argv[i + 1]):
                entry = resolve_module(argv[i + 1], roots)
                row['entrypoint'] = {'kind': 'module', 'module': argv[i + 1]}
        else:
            for arg in argv[1:]:
                if arg.startswith('-'):
                    continue
                if arg.endswith('.py'):
                    candidate = Path(arg) if Path(arg).is_absolute() else cwd / arg
                    entry = candidate.resolve() if candidate.is_file() else None
                    row['entrypoint'] = {'kind': 'script', 'path': str(candidate)}
                break
        row['source_files'] = {}
        row['source_discovery'] = 'recursive_local_ast_with_initializers'
        if entry:
            sources = source_closure([entry],roots[:2])
            for source in sources:
                row['source_files'][str(source)] = source_record(source, row['started_at_unix'])
        row['dependency_versions'] = package_versions([str(p) for p in site_paths])
    except (OSError, ValueError) as exc:
        row['error_type'] = type(exc).__name__
    return row


def inspect_service(name, url, timeout=5):
    schema_result, schema = get_endpoint(url + '/openapi.json', timeout)
    schema = schema if isinstance(schema, dict) else {}
    health = []
    for path in ('/healthz', '/health', '/readyz', '/ready'):
        if 'get' in schema.get('paths', {}).get(path, {}):
            health.append(get_endpoint(url + path, timeout)[0])
    schemas = schema.get('components', {}).get('schemas', {})
    seeded = all('seed' in schemas.get(k, {}).get('properties', {}) for k in ('PlanRequest', 'PlanResponse'))
    pids, reason = listener_pids(url)
    return {'name': name, 'url': url, 'openapi': schema_result, 'health': health,
            'health_endpoint_declared': bool(health), 'seed_request_response': seeded if name == 'graspnet' else None,
            'processes': [process_identity(pid) for pid in pids], 'process_identity_limit': reason,
            'inference_reexecuted': False}


def runtime_inventory():
    nvidia = run_command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'])
    kernel_module = Path('/proc/driver/nvidia/version')
    version = kernel_module.read_text()[:4096] if kernel_module.is_file() else None
    libraries = run_command(['ldconfig', '-p'])
    nvml = []
    for line in libraries['output'].splitlines():
        if 'libnvidia-ml.so' in line and '=>' in line:
            path = Path(line.split('=>', 1)[1].strip())
            nvml.append({'path': str(path), 'resolved_path': str(path.resolve())})
    return {'python': sys.version, 'executable': sys.executable, 'kernel': platform.release(),
            'platform': platform.platform(), 'dependencies': package_versions(),
            'nvidia_smi': nvidia, 'nvidia_kernel_module': version, 'nvml_libraries': nvml,
            'gpu_memory_readable': nvidia['returncode'] == 0,
            'runtime_environment': {k:os.environ.get(k) for k in ('MUJOCO_GL','LIBGL_ALWAYS_SOFTWARE','__EGL_VENDOR_LIBRARY_FILENAMES','MUJOCO_EGL_DEVICE_ID','JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'environment_variables_recorded': 'explicit_nonsecret_runtime_whitelist_only'}


def collect(config_path, output, evidence, timeout=5):
    output = Path(output)
    if output.exists():
        raise FileExistsError('output already exists; choose a fresh inventory directory')
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config=json.loads(config_bytes);urls = service_urls(config)
    from capx.experiments.graph_v2 import runtime_environment
    os.environ.update(runtime_environment(config))
    proofs = [source_record(Path(path)) for path in evidence]
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'capx.graph_service_inventory.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'mode': 'read_only_get_no_inference', 'model_requests': 0, 'inference_requests': 0,
              'config': {'path': str(config_path.resolve()), 'sha256': sha256(config_bytes)},
              'services': {name: inspect_service(name, url, timeout) for name, url in urls.items()},
              'runtime': runtime_inventory(), 'historical_evidence': proofs,
              'historical_evidence_reexecuted': False,
              'limitations': ['Schema and health GET do not prove inference availability.',
                              'Existing source files may differ from code loaded by a long-lived process.',
                              'Only the explicit nonsecret runtime whitelist is recorded; no private environment, authentication headers, or response bodies are stored.',
                              'Local AST import closure includes package initializers and relative imports; dynamic imports and installed packages are not claimed as complete loaded-byte provenance.']}
    with (output / 'service-inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    parser.add_argument('--timeout', type=float, default=5)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    result = collect(args.config, args.output, args.evidence, args.timeout)
    print(json.dumps({'inventory': str((args.output / 'service-inventory.json').resolve()),
                      'openapi_http_status': {name: row['openapi'].get('http_status') for name, row in result['services'].items()},
                      'graspnet_seed_schema': result['services']['graspnet']['seed_request_response'],
                      'gpu_memory_readable': result['runtime'].get('gpu_memory_readable'),
                      'model_requests': 0, 'inference_requests': 0}))
    return 0 if all(s['openapi'].get('http_status') == 200 for s in result['services'].values()) and result['services']['graspnet']['seed_request_response'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
