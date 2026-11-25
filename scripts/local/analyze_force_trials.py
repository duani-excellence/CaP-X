"""Offline operator analysis of V32 artifacts; never used by online Skills."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from capx.experiments.artifacts import lines,episode_summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);summaries=[];probes=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for file in sorted(a.run.glob('seed-*-repeat-*/*/result.json')):
        root=file.parent;r=json.loads(file.read_text());summary=episode_summary(root)
        trace=list(lines(root/'trace.jsonl'));force=list(lines(root/'force.jsonl'))
        summary.update(path=str(root),seed=r.get('seed'),force_guard_stops=sum(e['kind']=='force_guard_stop' for e in trace),
            force_samples=len(force),force_decisions=r.get('force_decisions'),
            new_llm_requests=r.get('new_llm_requests'),force_feedback=r.get('force_feedback'),
            execution_seconds=r.get('execution_seconds', r.get('seconds')),
            end_to_end_seconds=r.get('end_to_end_seconds', r.get('seconds')),
            stop_reason=r.get('stop_reason'),last_error=r.get('last_error'))
        periods=np.diff([x['timestamp'] for x in force]);summary['median_sensor_period_s']=float(np.median(periods)) if len(periods) else None
        summaries.append(summary)
        last_motion=None
        for e in trace:
            if e['kind']=='api' and e['name']=='goto_pose':last_motion=e
            if e['kind']=='skill_attachment_probe' and last_motion:
                name=e['object'];before=last_motion['before']['objects'].get(name);after=last_motion['after']['objects'].get(name)
                if before and after:
                    rotation=lambda x:Rotation.from_quat(np.roll(x['quaternion'],-1))
                    probes.append(dict(variant=root.name,seed=r['seed'],object=name,public_check=e['check'],
                        offline_object_rotation_deg=float(np.degrees((rotation(after)*rotation(before).inv()).magnitude())),
                        offline_object_displacement_m=(np.array(after['position'])-before['position']).tolist(),
                        source_motion_time=last_motion['after']['sim_time'],scope='offline simulator diagnostic, not available to actor'))
        if force:
            t=np.array([x['timestamp'] for x in force]);w=np.array([x['external_base'] for x in force]);jaw=np.array([x['gripper_width_m'] for x in force])*1000;tcp=np.array([x['tcp'][:3] for x in force])
            fig,axs=plt.subplots(4,1,figsize=(11,9),sharex=True)
            for k,label in enumerate(['Fx','Fy','Fz']):axs[0].plot(t,w[:,k],label=label,lw=.8)
            for k,label in enumerate(['Tx','Ty','Tz']):axs[1].plot(t,w[:,k+3],label=label,lw=.8)
            axs[2].plot(t,jaw,label='jaw width',color='darkorange')
            for k,label in enumerate(['TCP x','TCP y','TCP z']):axs[3].plot(t,tcp[:,k],label=label,lw=.8)
            for ax,label in zip(axs,['Force / N','Torque / Nm','Jaw / mm','TCP / m']):
                ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(loc='upper right',ncol=3)
                for e in trace:
                    if e['kind']=='force_guard_stop':ax.axvline(e['sample']['timestamp'],color='red',ls=':',alpha=.7)
            axs[-1].set_xlabel('Simulation time / s')
            fig.suptitle(f"{root.name} | seed {r['seed']} | native goals {summary['native_goals']}/4 | success {r.get('raw_success')}")
            fig.tight_layout();fig.savefig(a.output/f"{root.parent.name}-{root.name}-force.png",dpi=150);plt.close(fig)
    (a.output/'summary.json').write_text(json.dumps(summaries,indent=2))
    (a.output/'offline-attachment-probes.json').write_text(json.dumps(probes,indent=2))
    requests={path for r in summaries for path in r.get('request_paths',[]) or []}
    paid=[json.loads((Path(path)/'receipt.json').read_text()) for path in requests if Path(path).is_relative_to(a.run.resolve())]
    (a.output/'unique-current-run-receipts.json').write_text(json.dumps(paid,indent=2))
    table=['| Seed | Arm | Goals | Native success | API / capture | SAM3 / CGN / IK | Attributed LLM / tokens | Execution / E2E s |',
           '|---|---|---|---|---|---|---|---|']
    for r in summaries:
        http=r['http_counts'];table.append(f"| {r['seed']} | {r['variant']} | {r['native_goals']}/4 | {r['raw_success']} | {r['api_calls']} / {r['captures']} | {http.get('sam3',0)} / {http.get('graspnet',0)} / {http.get('pyroki',0)} | {r['llm_requests']} / {(r.get('input_tokens') or 0)+(r.get('output_tokens') or 0)} | {r['execution_seconds']:.2f} / {r['end_to_end_seconds']:.2f} |")
    (a.output/'table.md').write_text('\n'.join(table)+'\n');print('\n'.join(table));print('Unique current-run receipts',len(paid))

if __name__=='__main__':main()
