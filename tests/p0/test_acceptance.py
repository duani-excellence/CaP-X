"""Real simulation, existing local IK service; no LLM, weights, or training."""
import copy
import json
import os
from pathlib import Path
import pytest
from capx.envs.tasks import get_config, get_exec_env
from capx.envs.tasks.franka.franka_pick_place import ORACLE_CODE

OUT=Path(os.environ.get('CAPX_TEST_OUTPUT', '/tmp/capx-p0-acceptance'))


def test_reward_worker_single_sample():
    from verl_agent_reward import capx_franka_reward as worker
    OUT.mkdir(parents=True, exist_ok=True)
    source='franka_robosuite_pick_place_code_env'
    cfg=copy.deepcopy(get_config(source))
    cfg.artifact_root=str(OUT/'episodes')
    cfg.enable_render=False
    old=worker.get_config
    worker.get_config=lambda _:cfg
    try:
        result=worker.compute_score(source, ORACLE_CODE, {}, {'seed':0})
        (OUT/'reward-worker.json').write_text(json.dumps(result,indent=2))
        assert result['status'] != 'system_error', result
        assert result['score'] == 1 and result['won'], result
        assert result['physical_success'] and result['motion_calls']>0
    finally:
        worker._evict(source)
        worker.get_config=old


def test_snapshot_oracle_replay():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg=copy.deepcopy(get_config('franka_robosuite_pick_place_code_env'))
    cfg.enable_render=False; cfg.artifact_root=str(OUT/'replay')
    env=get_exec_env('franka_robosuite_pick_place_code_env')(cfg)
    rows=[]
    try:
        env.reset(seed=0)
        snap=env.low_level_env.capture_snapshot()
        for i in range(2):
            env.reset(seed=0)
            env.low_level_env.restore_snapshot(snap)
            _,reward,_,_,info=env.step(ORACLE_CODE)
            rows.append(dict(replay=i,initial_hash=info['initial_state_hash'], final_hash=env.low_level_env.physical_state_hash(), reward=reward,grasp_success=info['grasp_success'],slipped=info['slipped'],physical_success=info['physical_success'],trace_path=info['trace_path']))
        (OUT/'oracle-replay.json').write_text(json.dumps(rows,indent=2))
        assert rows[0]['initial_hash']==rows[1]['initial_hash']
        assert rows[0]['final_hash']==rows[1]['final_hash']
        assert rows[0]['reward']==rows[1]['reward']==1
    finally:
        env.close()

def test_real_reward_worker_rejections():
    from verl_agent_reward import capx_franka_reward as worker
    OUT.mkdir(parents=True,exist_ok=True)
    source='franka_robosuite_pick_place_code_env'
    cfg=copy.deepcopy(get_config(source));cfg.artifact_root=str(OUT/'negatives');cfg.execution_timeout=.3;cfg.enable_render=False
    old=worker.get_config;worker.get_config=lambda _:cfg
    rows=[]
    try:
        for code in ['pass','','for','FINISH','print("FINISH")','print(env)','while True: pass']:
            result=worker.compute_score(source,code,{}, {'seed':0})
            rows.append(dict(code=code,**result))
            assert result['score']==0 and not result['won'],result
        (OUT/'reward-worker-negatives.json').write_text(json.dumps(rows,indent=2))
        assert rows[-1]['status']=='timeout'
    finally:
        worker._evict(source);worker.get_config=old
