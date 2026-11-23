#!/usr/bin/env python3
"""Probe downloaded upstream tasks in a process-local source environment.

No dependency installation, model download, LLM, or task-success substitution.
Runs upstream reset/step/_check_success and records private diagnostic evidence.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--demo',help='Optional downloaded upstream human HDF5; action replay only, no intermediate state resets')
    p.add_argument('--episode',default='demo_0');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    from fetch_upstream_tasks import verify_checkout,verify_reference
    provenance={'sources':[
        verify_checkout(path,spec['commit']) for path,spec in zip(cfg['source_paths'],[cfg['robosuite_source'],cfg['upstream']])],
        'gate_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'external_task_gate.py').write_bytes(Path(__file__).read_bytes())
    (out/'fetch_upstream_tasks.py').write_bytes(Path(__file__).with_name('fetch_upstream_tasks.py').read_bytes())
    if args.demo:
        matching=[r for r in cfg.get('reference_data',[]) if Path(r['file']).resolve()==Path(args.demo).resolve()]
        if len(matching)!=1:raise ValueError('Reference data must be declared with a pinned digest')
        provenance['reference']=verify_reference(args.demo,matching[0]['sha256'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for k,v in cfg['runtime_environment'].items():os.environ[k]=str(v)
    for directory in reversed(cfg['source_paths']):
        path=Path(directory).resolve()
        if not path.is_dir():raise FileNotFoundError(path)
        sys.path.insert(0,str(path))
    import numpy as np
    import robosuite
    import mimicgen
    # Process-local rendering compatibility only. The installed simulator and
    # upstream physics / success implementations are left untouched.
    if cfg.get('egl_context_source'):
        import importlib.util
        src=Path(cfg['egl_context_source']).resolve()
        module_spec=importlib.util.spec_from_file_location('capx_external_egl_compat',src)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        import robosuite.utils.binding_utils as binding
        binding.GLContext=module.EGLGLContext
        cfg['egl_context_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
    from capx.experiments.native_tasks import save,snapshot,physical_state,bind_rng
    save(out/'config.json',cfg);results=[]
    if args.demo:
        import h5py,imageio.v2 as imageio
        with h5py.File(args.demo,'r') as f:
            metadata=json.loads(f['data'].attrs['env_args']);demo=f['data'][args.episode]
            actions=demo['actions'][:];initial=demo['states'][0];xml=demo.attrs['model_file']
        kwargs=metadata['env_kwargs'];kwargs.update(has_renderer=False,has_offscreen_renderer=True,
            camera_names=['agentview'],camera_heights=256,camera_widths=256,camera_depths=True,use_object_obs=False)
        start=time.monotonic();env=None;frames=[];trace=[]
        result=dict(mode='upstream_human_action_replay_NOT_Oracle_NOT_DeepSeek',new_llm_requests=0,
                    task=metadata['env_name'],episode=args.episode,data_file=str(Path(args.demo).resolve()))
        try:
            np.random.seed(args.seed)
            env=robosuite.make(metadata['env_name'],**kwargs);env.reset()
            env.reset_from_xml_string(env.edit_model_xml(xml));env.sim.set_state_from_flattened(initial);env.sim.forward()
            result['initial_state_hash']=hashlib.sha256(env.sim.get_state().flatten().tobytes()).hexdigest()
            result['initial_success']=bool(env._check_success())
            save(out/'initial-physics.json',physical_state(env));save(out/'metadata.json',metadata)
            for i,action in enumerate(actions):
                obs,reward,done,info=env.step(action)
                if i%5==0:
                    frames.append(obs['agentview_image'][::-1]);state=physical_state(env)
                    if hasattr(env,'_get_partial_task_metrics'):state['partial_metrics']=env._get_partial_task_metrics()
                    trace.append(dict(action=i,physics=state))
            result.update(passed=bool(env._check_success()),actual_actions=len(actions),final_physics=physical_state(env),
                          initial_restore_only=True,intermediate_state_resets=0)
        except Exception as exc:
            import traceback
            result.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        finally:
            if frames:imageio.mimsave(out/'reference.mp4',frames,fps=4)
            if env is not None:env.close()
        result['seconds']=time.monotonic()-start;save(out/'reference-trace.json',trace);save(out/'reference-result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='final_physics'}),flush=True);return
    for task in args.tasks or cfg['tasks']:
        row=dict(task=task,seed=args.seed,mode='upstream_environment_gate_ONLY',new_llm_requests=0);start=time.monotonic();env=None
        try:
            states=[]
            for repeat in range(2):
                np.random.seed(args.seed)
                env=robosuite.make(task,robots='Panda',controller_configs=robosuite.load_controller_config(default_controller='JOINT_POSITION'),
                    has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,camera_names=['agentview'],
                    camera_heights=256,camera_widths=256,camera_depths=True,use_object_obs=False,
                    hard_reset=False,reward_shaping=False,horizon=100)
                bind_rng(env,args.seed);np.random.seed(args.seed);obs=env.reset()
                state=snapshot(env,args.seed,{'environment':task,'source':cfg['upstream']});states.append(state)
                save(out/f'{task}-snapshot-{repeat}.json',state)
                save(out/f'{task}-physics-{repeat}.json',physical_state(env))
                from PIL import Image
                Image.fromarray(obs['agentview_image'][::-1]).save(out/f'{task}-{repeat}.png')
                before=bool(env._check_success())
                for _ in range(3):env.step(np.zeros(env.action_dim))
                save(out/f'{task}-after-step-{repeat}.json',physical_state(env))
                env.close();env=None
            row.update(passed=states[0]==states[1] and not before,same_seed_initial_state=states[0]==states[1],
                       snapshot_scope='initial_reconstruction_only_NOT_mid_episode_controller_resume',
                       initial_success=before,step_executed=True)
        except Exception as e:
            import traceback
            row.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
        finally:
            if env is not None:env.close()
        row['seconds']=time.monotonic()-start;results.append(row);save(out/(task+'-gate.json'),row);print(json.dumps(row),flush=True)
    save(out/'summary.json',dict(robosuite=robosuite.__version__,mujoco=__import__('mujoco').__version__,results=results))

if __name__=='__main__':main()
