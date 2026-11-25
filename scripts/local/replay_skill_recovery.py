"""Zero-new-LLM stage/adapter gate on an archived initial program + Snapshot."""
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
from capx.experiments.graph_v2 import configure_services,save
from capx.experiments.graph_campaign import archive_inventory,verify_source_hashes
from capx.experiments.contract_models import run_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--snapshot-group',default='G1')
    p.add_argument('--ik-runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--external-logiv',type=Path);p.add_argument('--group',choices=['G0','G1'],default='G1')
    a=p.parse_args();cfg=json.loads(a.config.read_text());cfg['services']['pyroki']=json.loads(a.ik_runtime.read_text())['url'];configure_services(cfg)
    common=json.loads((a.reference/'config.json').read_text())['tasks']['pick_place']
    initial=json.loads((a.reference/'initial.json').read_text());initial['historical_replay']=True
    snapshot=json.loads((a.reference/a.snapshot_group/'snapshot.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/'config.json',cfg)
    hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in cfg['sources']}
    save(a.output/'source-archive.json',archive_inventory(a.output,hashes));spec=copy.deepcopy(common);spec['skill_recovery_version']=30
    external=None
    if a.external_logiv:external=dict(path=str(a.external_logiv.resolve()),revision=subprocess.check_output(['git','-C',str(a.external_logiv),'rev-parse','HEAD'],text=True).strip())
    class NoNewLLM:
        count=0
        def query(self,*args):raise RuntimeError('Zero-new-LLM gate: model fallback disabled')
    result,_,_=run_episode(a.output/'episode',spec,snapshot['seed'],a.group,NoNewLLM(),cfg,initial,snapshot,environment_spec=common,external_method=external)
    save(a.output/'provenance.json',dict(mode='historical_program_live_vision_NOT_new_DeepSeek',new_llm_requests=0,initial=str(a.reference/'initial.json')))
    save(a.output/'source-postcheck.json',dict(changed=verify_source_hashes(hashes)))

if __name__=='__main__':main()
