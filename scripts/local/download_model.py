import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--output', required=True)
p.add_argument('--size', type=int, required=True)
p.add_argument('--sha256', required=True)
p.add_argument('--workers', type=int, default=6)
a = p.parse_args()
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
partial = Path(str(out) + '.partial')
parts = Path(str(out) + '.parts')
parts.mkdir(exist_ok=True)
if out.exists():
    partial = out
prefix = partial.stat().st_size if partial.exists() else 0
if prefix > a.size:
    raise RuntimeError('Existing file exceeds expected size')
chunk = 128 * 1024 * 1024
jobs = [(start, min(start + chunk, a.size) - 1) for start in range(prefix, a.size, chunk)]

def download(job):
    start, end = job
    dest = parts / f'{start}-{end}'
    for attempt in range(6):
        have = dest.stat().st_size if dest.exists() else 0
        expected = end - start + 1
        if have == expected:
            return dest
        if have > expected:
            raise RuntimeError('Chunk exceeds expected size')
        offset = start + have
        try:
            req = urllib.request.Request(a.url, headers={'Range': f'bytes={offset}-{end}'})
            with urllib.request.urlopen(req, timeout=60) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status != 206 or content_range != f'bytes {offset}-{end}/{a.size}':
                    raise RuntimeError(f'Unexpected range response: {response.status}, {content_range}')
                with dest.open('ab') as f:
                    shutil.copyfileobj(response, f, 1024 * 1024)
            if dest.stat().st_size != expected:
                raise RuntimeError('Incomplete chunk')
            return dest
        except Exception as e:
            print(f'retry {attempt+1} range={offset}-{end}: {type(e).__name__}: {e}', flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))

start_time = time.time()
done = prefix
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
    fs = {ex.submit(download, j): j for j in jobs}
    for future in concurrent.futures.as_completed(fs):
        future.result()
        start, end = fs[future]
        done += end - start + 1
        elapsed = max(time.time() - start_time, 0.01)
        print(f'{out.name}: {done/a.size:.1%}, {(done-prefix)/elapsed/1e6:.1f} MB/s', flush=True)
with partial.open('ab') as f:
    for start, end in jobs:
        with (parts / f'{start}-{end}').open('rb') as src:
            shutil.copyfileobj(src, f, 1024 * 1024)
digest = hashlib.sha256()
with partial.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        digest.update(block)
if partial.stat().st_size != a.size or digest.hexdigest() != a.sha256:
    raise RuntimeError(f'Checksum mismatch: {digest.hexdigest()}')
os.replace(partial, out)
shutil.rmtree(parts)
print(f'VERIFIED {out} sha256={a.sha256}', flush=True)
