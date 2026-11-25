"""Offline wrist calibration gate. No LLM, network or hardware access."""
import argparse,json
from pathlib import Path
import numpy as np
from capx.experiments.graph_v2 import configure_services
from capx.experiments.native_tasks import make_low,snapshot,save
from capx.integrations.sensors.wrist_ft import MujocoWristFT


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());configure_services(cfg);a.output.mkdir(parents=True,exist_ok=False)
    spec=cfg['tasks']['pick_place'];low=make_low(spec,0,render=False)
    try:
        f=MujocoWristFT(low,spec['force_sensor']);before=snapshot(low.robosuite_env,0,spec)
        initial=f.sample();after=snapshot(low.robosuite_env,0,spec)
        assert before==after,'Sensor recomputation mutated physics/controller/RNG'
        assert np.linalg.norm(initial['external_base'][:3])<.05,initial
        save(a.output/'calibration.json',f.calibration)
        empty=[]
        for _ in range(12):low._step_once();empty.append(f.sample())
        assert all(np.linalg.norm(x['external_base'][:3])<.05 for x in empty)
        # Offline known-load injection to sensor child robot body, not a task
        # contact oracle. Only this calibration fixture sees the imposed load.
        body=int(f.model.site_bodyid[f.site]);low.robosuite_env.sim.data.xfrc_applied[body,:3]=[0,0,-1]
        loaded=[]
        for _ in range(30):low._step_once();loaded.append(f.sample())
        err=abs(loaded[-1]['external_base'][2]+1)
        assert err<.08,loaded[-1]
        low.robosuite_env.sim.data.xfrc_applied[body,:]=0
        probes=[]
        # Independent loads across all axes, including values below the original
        # uncalibrated 0.15 N load detector. No task/object weights are inspected.
        for axis in range(6):
            for amplitude in ((.02,.05,.2) if axis<3 else (.002,.02)):
                imposed=np.zeros(6);imposed[axis]=amplitude
                low.robosuite_env.sim.data.xfrc_applied[body,:]=imposed
                for _ in range(30):low._step_once();row=f.sample()
                # xfrc is at body COM; predict that wrench at the observed TCP.
                from capx.integrations.sensors.wrist_ft import transform_wrench
                d=low.robosuite_env.sim.data;rb=d.body_xmat[low.base_link_idx].reshape(3,3)
                tcp=d.body_xpos[low.base_link_idx]+rb@np.asarray(row['tcp'][:3])
                expected=transform_wrench(imposed,rb.T,rb.T@(d._data.xipos[body]-tcp))
                delta=np.asarray(row['external_base'])-expected
                assert np.linalg.norm(delta[:3])<.005,(axis,amplitude,delta)
                assert np.linalg.norm(delta[3:])<.002,(axis,amplitude,delta)
                probes.append(dict(axis=axis,amplitude=amplitude,error=delta.tolist(),expected=expected.tolist(),sample=row))
        low.robosuite_env.sim.data.xfrc_applied[body,:]=0
        save(a.output/'result.json',dict(empty=empty,known_load=loaded,sensor_does_not_mutate_snapshot=before==after,known_force_error_n=err,initial=initial,axis_probes=probes,
             validated_error_bound_n=.005,validated_error_bound_nm=.002,scope='native ideal sensor; quasistatic loads at tested robot posture; no hardware or dynamic certification'))
        print(json.dumps(dict(empty_force_n=initial['external_base'][:3],known_force_error_n=err,snapshot_unchanged=True)))
    finally:low.close()
if __name__=='__main__':main()
