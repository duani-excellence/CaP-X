"""Opt-in integration against already loaded SAM3 and the P0 grasp service."""
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(not os.environ.get('CAPX_P0_VISION'),reason='requires existing SAM3 + P0 GraspNet service')


def test_seeded_visual_service_and_video():
    from capx.envs.tasks import get_config,get_exec_env
    import capx.integrations.vision.graspnet as grasp
    grasp.SERVICE_URL='http://127.0.0.1:8215'
    out=Path(os.environ['CAPX_TEST_OUTPUT'])/'vision'
    out.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(get_config('franka_robosuite_pick_place_vision_code_env'))
    cfg.enable_render=True;cfg.execution_timeout=90.;cfg.artifact_root=str(out/'episodes')
    env=get_exec_env('franka_robosuite_pick_place_vision_code_env')(cfg)
    rows=[]
    try:
        for repeat in range(2):
            obs,_=env.reset(seed=21)
            assert 'cube_poses' not in obs and 'object-state' not in obs
            assert all('segmentation' not in v.get('images',{}) for v in obs.values() if isinstance(v,dict))
            env.enable_video_capture(True,clear=True)
            _,reward,_,_,info=env.step("open_gripper()\np, q = sample_grasp_pose('red cube')\nprint(p, q)")
            samples=getattr(env.low_level_env,'grasp_sample',np.array([]))
            scores=getattr(env.low_level_env,'grasp_scores',np.array([]))
            timestamps=info['frame_timestamps']
            frames=env.get_video_frames()
            assert len(timestamps)==len(frames)>0
            assert np.all(np.diff(timestamps)>0)
            rows.append(dict(repeat=repeat,status=info['execution_status'],error=info['stderr'],initial_hash=info['initial_state_hash'],
                             candidate_hash=hashlib.sha256(np.asarray(samples).tobytes()+np.asarray(scores).tobytes()).hexdigest(),
                             candidates=len(samples),stdout=info['stdout'],timestamps=timestamps,trace_path=info['trace_path']))
            if repeat==0:
                import imageio.v3 as iio
                iio.imwrite(out/'video.mp4',np.asarray(frames),fps=4)
                (out/'video-timestamps.json').write_text(json.dumps([dict(frame=i,sim_time=t) for i,t in enumerate(timestamps)],indent=2))
        (out/'service-replay.json').write_text(json.dumps(rows,indent=2))
        assert all(r['status']=='ok' for r in rows), rows
        assert rows[0]['initial_hash']==rows[1]['initial_hash']
        assert rows[0]['candidate_hash']==rows[1]['candidate_hash']
        assert rows[0]['stdout']==rows[1]['stdout']
    finally:
        env.close()
