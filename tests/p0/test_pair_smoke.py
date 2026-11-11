"""One paired Oracle scene; task failure is a valid measured outcome."""
import copy
import json
import os
from pathlib import Path
import pytest

pytestmark=pytest.mark.skipif(not os.environ.get('CAPX_P0_VISION'),reason='requires existing SAM3 + P0 GraspNet service')


def test_paired_oracle_physical_accounting():
    from capx.envs.tasks import get_config,get_exec_env
    from capx.envs.tasks.franka.franka_pick_place import ORACLE_CODE
    from capx.utils.video_utils import _write_video
    import capx.integrations.vision.graspnet as grasp
    grasp.SERVICE_URL='http://127.0.0.1:8215'
    out=Path(os.environ['CAPX_TEST_OUTPUT'])/'paired-oracle'
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for label,source in [('A','franka_robosuite_pick_place_code_env'),('B','franka_robosuite_pick_place_vision_code_env')]:
        cfg=copy.deepcopy(get_config(source));cfg.artifact_root=str(out/label);cfg.enable_render=True
        env=get_exec_env(source)(cfg)
        try:
            _,reset=env.reset(seed=21)
            snapshot=env.low_level_env.capture_snapshot()
            (out/f'{label}-snapshot.json').write_text(json.dumps(snapshot,indent=2))
            env.enable_video_capture(True,clear=True)
            _,reward,_,truncated,info=env.step(ORACLE_CODE)
            rows.append(dict(group=label,seed=21,reward=reward,truncated=truncated,**info))
            _write_video(env.get_video_frames(),str(out),suffix=label,timestamps=info['frame_timestamps'])
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            assert info['execution_status'] not in ('system_error','timeout'),info
            assert bool(reward)==bool(info['physical_success'] and info['sandbox_rc']==0 and not truncated)
        finally:
            env.close()
    assert rows[0]['initial_state_hash']==rows[1]['initial_state_hash']
    assert rows[0]['reward']==1
