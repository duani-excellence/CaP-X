"""Fetch a fixed small official LIBERO-Recovery subset, no models or demos."""
import argparse,hashlib,json,random
from pathlib import Path,PurePosixPath
import requests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=2);p.add_argument('--suite',default='libero_10')
    p.add_argument('--revision',required=True,help='Pinned dataset commit, never implicitly latest')
    a=p.parse_args()
    if not 1<=a.count<=10:raise ValueError('Small subset only: 1..10')
    rows=[json.loads(x) for x in a.index.read_text().splitlines()]
    rows=sorted([x for x in rows if x['source']==a.suite],key=lambda x:x['scene_id'])
    chosen=random.Random(a.seed).sample(rows,min(a.count,len(rows)))
    a.output.mkdir(parents=True,exist_ok=False);manifest=[]
    # Save selection BEFORE attempting environment compatibility, including failures.
    (a.output/'selection.json').write_text(json.dumps(chosen,indent=2))
    for row in chosen:
        item=dict(scene=row,files={})
        for key in ['bddl_file','sim_state_file']:
            remote=PurePosixPath(row[key]).relative_to('benchmark_failure')
            if '..' in remote.parts:raise ValueError('Unsafe dataset path')
            dest=a.output/remote;dest.parent.mkdir(parents=True,exist_ok=True)
            response=requests.get('https://modelscope.cn/api/v1/datasets/ataier/LIBERO_Recovery_FailureScenes/repo',
                params={'Revision':a.revision,'FilePath':str(remote)},timeout=45)
            response.raise_for_status()
            if len(response.content)>8_000_000:raise ValueError('Unexpectedly large scene file')
            dest.write_bytes(response.content)
            item['files'][key]=dict(path=str(dest.resolve()),sha256=hashlib.sha256(response.content).hexdigest(),bytes=len(response.content))
        manifest.append(item)
        (a.output/'manifest.json').write_text(json.dumps(dict(dataset='ataier/LIBERO_Recovery_FailureScenes',revision=a.revision,
            index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest(),selection_seed=a.seed,suite=a.suite,scenes=manifest),indent=2))
    print(json.dumps({'downloaded_scenes':len(manifest),'output':str(a.output)}))

if __name__=='__main__':main()
