"""OFFLINE paired error diagnostic, not a new online calibration or permit.

Common extrinsic translation cancels algebraically in simultaneous p_h-p_s.
The archived marginal envelopes do not identify its magnitude or the separate
attachment errors. Paired empirical errors demonstrate conservatism but cannot
alone certify unobserved support motion or dynamic attachment propagation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from calibrate_state_memory import samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    profile=json.loads(a.profile.read_text());rows=[]
    for e in profile['evidence']:
        episode=Path(e['path']).parents[2];data,_=samples(episode);frames={}
        for r in data:frames.setdefault(r['frame'],[]).append(r)
        for frame,values in frames.items():
            held=next((r for r in values if r['kind']=='attached'),None)
            support=next((r for r in values if r['kind']=='static'),None)
            if held and support:
                delta=np.asarray(held['error_xyz'])-support['error_xyz']
                gap=held['top_error']-held['height_error']-support['top_error']
                rows.append(dict(episode=str(episode),frame=frame,xy_error=float(np.linalg.norm(delta[:2])),
                    gap_error=abs(gap),held=held,support=support))
    summary=dict(kind='offline_empirical_diagnostic_NOT_release_authorization',paired_frames=len(rows),
        independent_scenes=False,new_llm_requests=0,
        maxima={k:max(r[k] for r in rows) for k in ('xy_error','gap_error')} if rows else {},
        online_envelope_changed=False,
        missing_validation=['independent dynamic attachment sequences','bounded hidden support motion',
            'separate extrinsic rotation/TCP encoder error','same-pair anchor lifetime and contradiction gates'])
    for name,value in [('rows',rows),('summary',summary)]:
        (a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
