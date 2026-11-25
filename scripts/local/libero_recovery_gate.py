"""Official scene/protocol compatibility gate; NO policy/LLM/task-success claim.

Run in an existing isolated LIBERO interpreter. Imports only the four exact
upstream state-preparation helpers, avoiding its VLA model-client dependency.
Private state is confined to this evaluator-side gate, never sent to an actor.
"""
import argparse,ast,hashlib,json,os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--libero-config',type=Path,required=True);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--normal-suite',default='libero_10_swap');a=p.parse_args()
    os.environ['LIBERO_CONFIG_PATH']=str(a.libero_config.resolve())
    from libero.envs import OffScreenRenderEnv
    from libero import benchmark,get_libero_path
    a.output.mkdir(parents=True,exist_ok=False)
    names={'build_object_xy_qpos_adrs','perturb_object_xy','settle_gripper_to','reset_controller_goal_to_current'}
    tree=ast.parse(a.protocol.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    if len(nodes)!=len(names):raise ValueError('Unexpected official protocol')
    scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.protocol),'exec'),scope)
    result=[];rows=json.loads(a.manifest.read_text())['scenes'];rng=np.random.RandomState(a.seed)
    def environment(bddl,seed):
        error=None
        for attempt in range(12):
            np.random.seed(seed+attempt)
            try:return OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=128,camera_widths=128)
            except Exception as exc:
                error=exc
                if 'RandomizationError' not in type(exc).__name__:raise
        raise error
    for index,row in enumerate(rows):
        env=None;entry=dict(scene_id=row['scene']['scene_id'],task=row['scene']['task'],policy_executed=False)
        try:
            bddl=Path(row['files']['bddl_file']['path']);statefile=Path(row['files']['sim_state_file']['path'])
            before=hashlib.sha256(statefile.read_bytes()).hexdigest();original=np.load(statefile,allow_pickle=False)
            env=environment(bddl,a.seed+1_000_001*(index+1));env.seed(a.seed)
            adrs=scope['build_object_xy_qpos_adrs'](env);entry['trials']=[]
            for closed in [True,False]:
                env.reset();flat,deltas=scope['perturb_object_xy'](original,adrs,.02,rng)
                obs=env.set_init_state(flat);scope['reset_controller_goal_to_current'](env)
                scope['settle_gripper_to'](env,closed,20)
                current=env.env.sim.get_state().flatten();obs=env.env._get_observations()
                entry['trials'].append(dict(closed=closed,state_hash=hashlib.sha256(current.tobytes()).hexdigest(),
                    raw_native_success=bool(env.check_success()),applied_deltas=deltas,shape=list(current.shape),
                    preserved_input=hashlib.sha256(statefile.read_bytes()).hexdigest()==before))
                np.savez_compressed(a.output/f'scene-{index}-{closed}.npz',rgb=obs['agentview_image'],state=current)
            entry['status']='protocol_restore_and_settle_passed'
            entry['state_layout_audit']=dict(flat_size=len(original),nq=env.env.sim.model.nq,nv=env.env.sim.model.nv,
                perturbation_code_unchanged=True,possible_time_prefix_offset=len(original)==1+env.env.sim.model.nq+env.env.sim.model.nv)
        except Exception as exc:entry.update(status='blocked',error=type(exc).__name__+': '+str(exc))
        finally:
            if env is not None:env.close()
        result.append(entry);print(entry['scene_id'],entry['status'],entry.get('error',''),flush=True)
    env=None
    try:
        suite=benchmark.get_benchmark_dict()[a.normal_suite]();task=suite.get_task(0)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        env=environment(bddl,a.seed);env.seed(a.seed);env.reset()
        result.append(dict(suite=a.normal_suite,task=task.language,status='normal_long_environment_reset_passed',policy_executed=False))
    except Exception as exc:result.append(dict(suite=a.normal_suite,status='blocked',error=type(exc).__name__+': '+str(exc)))
    finally:
        if env is not None:env.close()
    (a.output/'result.json').write_text(json.dumps(dict(mode='official_protocol_compatibility_NOT_method_evaluation',
        protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),seed=a.seed,rows=result),indent=2))

if __name__=='__main__':main()
