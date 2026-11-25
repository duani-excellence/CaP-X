"""Export current source and selected experiment artifacts, excluding credentials.

The source tree is copied from current working bytes, including uncommitted work.
Experiment text paths/credentials are redacted. Original artifacts stay intact.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,nargs='*',default=[],help='Optional experiment directories; omit for current-source-only export')
    p.add_argument('--exclude-record-dir',type=Path,action='append',default=[],help='Exclude e.g. isolated optional dependency installation; record exclusion in manifest')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];output=a.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    secrets=[]
    for name in ('.env.local','.env.api.local'):
        f=root/name
        if f.exists():
            for line in f.read_text().splitlines():
                if '=' not in line or line.lstrip().startswith('#'):continue
                key,value=line.split('=',1)
                if any(x in key.upper() for x in ('KEY','TOKEN','SECRET','PASSWORD')):
                    value=value.strip().strip('"\'')
                    if len(value)>8:secrets.append(value)
    def clean(raw,experiment):
        try:s=raw.decode('utf-8')
        except UnicodeDecodeError:return raw,False
        old=s
        for value in secrets:s=s.replace(value,'<REDACTED_CREDENTIAL>')
        s=re.sub(r'(?i)Bearer\s+[A-Za-z0-9._~+/-]{12,}','Bearer <REDACTED>',s)
        s=re.sub(r'sk-[A-Za-z0-9_-]{16,}','<REDACTED_API_KEY>',s)
        if experiment:
            s=re.sub(r'(?m)^self = environ\(.*$','<REDACTED_ENVIRONMENT_DUMP>',s)
            s=re.sub(r'(?i)("(?:authorization|api_key|access_token|password)"\s*:\s*)"[^"\n]*"',r'\1"<REDACTED>"',s)
            s=s.replace(str(root),'${CAPX_ROOT}').replace(str(Path.home()),'${USER_HOME}')
        return s.encode(),s!=old
    files=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'],cwd=root).decode().split('\0')
    manifest=[];excluded=[]
    excluded_dirs=[d.resolve() for d in a.exclude_record_dir]
    def add(tar,path,name,experiment=False):
        if path.is_symlink():excluded.append(dict(path=name,reason='symlink_not_followed'));return
        if not path.is_file():return
        raw=path.read_bytes();data,changed=clean(raw,experiment)
        for secret in secrets:
            if secret.encode() in data:raise RuntimeError('Credential redaction failed')
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o755 if path.stat().st_mode&0o111 else 0o644
        tar.addfile(info,io.BytesIO(data));manifest.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),
            source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(data),redacted=changed))
    with tarfile.open(output,'x:gz') as tar:
        for relative in sorted(set(files)):
            if not relative:continue
            path=Path(relative)
            if path.parts[0] in ('outputs','data','models','checkpoints','logs','.git') or any(x.startswith('.env') and x not in ('.env.example','.env.api.example') for x in path.parts):
                excluded.append(dict(path=relative,reason='runtime_data_or_private_config'));continue
            add(tar,root/path,'source/'+relative)
        for i,directory in enumerate(a.records):
            directory=directory.resolve()
            if root not in directory.parents or not directory.is_dir():raise ValueError('Records must be an existing project subdirectory')
            if directory in output.parents:raise ValueError('Export cannot be inside included records')
            for path in sorted(directory.rglob('*')):
                if any(d==path or d in path.parents for d in excluded_dirs):
                    if path in excluded_dirs:excluded.append(dict(path=str(path.relative_to(root)),reason='explicit_record_dependency_exclusion'))
                    continue
                if any(x.startswith('.env') or x in ('.venv','__pycache__','.git') for x in path.relative_to(directory).parts):continue
                if path.is_file():add(tar,path,f'records/{i}-{directory.name}/'+str(path.relative_to(directory)),True)
        metadata=dict(schema='capx.audit_export.v34',head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True),files=manifest,excluded=excluded,
            notes=['Current working-tree source, not only HEAD.','No .env credentials, virtualenvs, models or dependency installations.',
                   'Record redaction can alter original hash-chain text; original source_sha256 preserves archive provenance; do not interpret exported chain hashes as recomputed.'])
        data=json.dumps(metadata,ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    digest=hashlib.sha256(output.read_bytes()).hexdigest();output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(json.dumps(dict(path=str(output),sha256=digest,files=len(manifest),bytes=output.stat().st_size,redacted_files=sum(v['redacted'] for v in manifest)),indent=2))


if __name__=='__main__':main()
