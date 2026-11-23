#!/usr/bin/env python3
"""Fetch pinned upstream task sources, optionally small reference demonstrations.

Never install dependencies, replace an existing checkout, or download models.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path


def verify_checkout(path,commit):
    path=Path(path)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise ValueError(f'Existing source differs; preserved without checkout: {path}')
    dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise ValueError(f'Existing source modified; preserved without reset: {path}')
    return dict(path=str(path.resolve()),commit=actual,tracked_source_clean=True)


def verify_reference(path,expected):
    path=Path(path)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=expected:raise ValueError(f'Reference hash mismatch; file preserved: {path}')
    return dict(path=str(path.resolve()),sha256=digest,bytes=path.stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--with-reference-data',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    for spec,path in [(cfg['robosuite_source'],cfg['source_paths'][0]),(cfg['upstream'],cfg['source_paths'][1])]:
        dest=Path(path)
        if dest.exists():
            print('verified',json.dumps(verify_checkout(dest,spec['commit'])));continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        for command in [['git','init',str(dest)],['git','-C',str(dest),'remote','add','origin',spec['url']],
                        ['git','-C',str(dest),'fetch','--depth','1','origin',spec['commit']],
                        ['git','-C',str(dest),'checkout','--detach','FETCH_HEAD']]:subprocess.run(command,check=True)
    if a.with_reference_data:
        import requests
        for row in cfg.get('reference_data',[]):
            path=Path(row['file']);path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with requests.get(row['url'],stream=True,timeout=60) as response:
                    response.raise_for_status()
                    with path.open('xb') as f:
                        for chunk in response.iter_content(1024*1024):f.write(chunk)
            print('verified_reference',json.dumps(verify_reference(path,row['sha256'])))

if __name__=='__main__':main()
