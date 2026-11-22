"""Real DeepSeek B0/G0/G1 runner over trusted Vision task adapters.

G0/G1 share initial code and Snapshot. B0 uses the original legal raw Vision API
and program-level console feedback. Strict response parsing is a common protocol
fix. Online monitors only receive public sensors; evaluator state is logged on a
separate parent-only channel. Imports that cache service URLs are deliberately lazy.
"""
from __future__ import annotations
import math
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from capx.experiments.graph_campaign import json_value, reject_secrets

TASKS=('stack','lift','restack')
GROUPS=('B0','G0','G1')
CONDITIONS={
    'stack':'Stack red cube on green cube, release, then retreat clear of the stack.',
    'lift':'Lift red cube and keep holding it stably. DO NOT release.',
    'restack':'Green cube initially rests on red cube. Move green to a clear table region and release/retreat first, then stack red onto green, release, and retreat.',
}
SOURCES=('capx/experiments/graph_v2.py','capx/experiments/vision_tasks.py',
         'capx/execution/grounded_feedback.py','capx/execution/grounded_recovery.py','capx/execution/grounded_targets.py','capx/execution/public_geometry.py',
         'capx/execution/graph_feedback.py','capx/execution/dependency_graph.py','capx/execution/feedback.py',
         'capx/execution/sandbox.py','capx/execution/actor.py','capx/envs/tasks/audited_stack.py',
         'capx/envs/physical_evaluation.py','capx/multitask/runtime.py','capx/experiments/graph_feedback.py',
         'capx/experiments/layered_feedback.py','capx/llm/providers.py','capx/llm/client.py',
         'capx/execution/grip_offset.py','capx/execution/support_geometry.py','capx/execution/grounded_control.py','capx/execution/grasp_geometry.py','capx/execution/control_frames.py','capx/execution/bounded_ik.py','capx/execution/public_camera.py','capx/execution/depth_sampling.py','capx/execution/robot_self_geometry.py','capx/execution/table_clearance.py',
         'capx/experiments/shared_initial.py','capx/experiments/graph_v2_analysis.py','capx/experiments/graph_power.py','capx/experiments/graph_preflight.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=json_value).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,default=json_value,ensure_ascii=False,indent=2,allow_nan=False)


def make_environment(*args,**kwargs):
    from capx.experiments.vision_tasks import make_environment as build
    return build(*args,**kwargs)


def prepare_scene(*args,**kwargs):
    from capx.experiments.vision_tasks import prepare_scene as prepare
    return prepare(*args,**kwargs)


def instrument_tools(*args):
    from capx.experiments.layered_feedback import instrument_tools as instrument
    return instrument(*args)


def validated_program(answer):
    from capx.experiments.layered_feedback import validated_program as validate
    return validate(answer)


def archive(output,config=None,*,initial_ref=None,campaign_ref=None):
    from capx.experiments.graph_feedback import archive_sources
    from capx.experiments.graph_campaign import source_inventory,archive_inventory
    archive_sources(Path(__file__).resolve().parents[2],output,SOURCES)
    cfg=dict(config or {})
    cfg.setdefault('sources',list(SOURCES))
    cfg.setdefault('prompt_files',['capx/execution/grounded_feedback.py'])
    cfg.setdefault('scoring_files',['capx/envs/physical_evaluation.py'])
    hashes=source_inventory(cfg,include_run_inputs=False)
    if campaign_ref is not None:
        frozen_path=Path(campaign_ref['root'])/'frozen.json'
        if hashlib.sha256(frozen_path.read_bytes()).hexdigest()!=campaign_ref['frozen_sha256']:
            raise ValueError('Campaign signature/source reference drift')
        expected=campaign_ref.get('_runtime_hashes')
        if expected is None:
            from capx.experiments.graph_campaign import load_campaign
            expected=load_campaign(campaign_ref['root'],check_sources=False)['source_hashes']
        if any(expected.get(raw)!=value for raw,value in hashes.items()):
            raise ValueError('Runtime source drift between campaign check and unit archive')
    if initial_ref is not None:
        from capx.experiments.shared_initial import selected_reference_files
        for raw,expected in selected_reference_files(initial_ref).items():
            if not Path(raw).is_file() or hashlib.sha256(Path(raw).read_bytes()).hexdigest()!=expected:
                raise ValueError('Shared initial selected input drift: '+raw)
            hashes[raw]=expected
    save(Path(output)/'source-archive.json',archive_inventory(output,hashes))
    if initial_ref is not None:
        frozen=str((Path(initial_ref['source_campaign'])/'frozen.json').resolve())
        expected=initial_ref['files'].get(frozen)
        if not expected or hashlib.sha256(Path(frozen).read_bytes()).hexdigest()!=expected:
            raise ValueError('Shared source campaign metadata drift')
        hashes[frozen]=expected
    if campaign_ref is not None:
        frozen=Path(campaign_ref['root'])/'frozen.json'
        if hashlib.sha256(frozen.read_bytes()).hexdigest()!=campaign_ref['frozen_sha256']:
            raise ValueError('Campaign signature/source reference drift')
        save(Path(output)/'campaign-reference.json',{k:v for k,v in campaign_ref.items() if not k.startswith('_')})
        # Guard the campaign metadata, without copying its global payload per unit.
        hashes[str(frozen.resolve())]=campaign_ref['frozen_sha256']
    return hashes


def eager_execution_imports():
    # These are local contract/broker modules, never model/service inference.
    # Load before B0 so a later graph arm cannot lazily load an edited monitor.
    from capx.experiments import vision_tasks
    from capx.execution import grounded_feedback,grounded_control,control_frames,sandbox
    return (vision_tasks,grounded_feedback,grounded_control,control_frames,sandbox)


RUNTIME_ENVIRONMENT=('MUJOCO_GL','LIBGL_ALWAYS_SOFTWARE','__EGL_VENDOR_LIBRARY_FILENAMES',
                     'MUJOCO_EGL_DEVICE_ID','JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')


def runtime_environment(config):
    values=config.get('runtime_environment',{})
    if not isinstance(values,dict) or not set(values)<=set(RUNTIME_ENVIRONMENT) or any(not isinstance(v,str) or not v or '\x00' in v for v in values.values()):
        raise ValueError('runtime_environment must contain only supported runtime variable names and nonempty string values')
    return values


def configure_services(config):
    reject_secrets(config)
    os.environ.update(runtime_environment(config))
    for key,var in (('sam3','SAM3_SERVICE_URL'),('graspnet','GRASPNET_SERVICE_URL'),('pyroki','PYROKI_SERVICE_URL')):
        if config.get('services',{}).get(key):os.environ[var]=config['services'][key]


def limits(config):
    b=dict(max_blocks=6,api_budget=128,max_local=3,max_episode_local=8,max_episode_views=2,block_seconds=90,episode_seconds=1800)
    b.update(config.get('budgets',{}))
    if type(b.get('bounded_recovery',False)) is not bool:raise ValueError('bounded_recovery must be boolean')
    for key in ('max_recovery_plans','max_recovery_no_progress'):
        if key in b and (type(b[key]) is not int or b[key]<1):raise ValueError('Invalid recovery budget '+key)
    for key in ('max_blocks','api_budget','block_seconds','episode_seconds'):
        if not isinstance(b[key],(int,float)) or b[key]<=0:raise ValueError('Invalid budget '+key)
    for key in ('max_local','max_episode_local'):
        if not isinstance(b[key],int) or b[key]<0:raise ValueError('Invalid local budget')
    if type(b['max_episode_views']) is not int or b['max_episode_views']<0:
        raise ValueError('Invalid episode view budget')
    return b


GRAPH_OPTIONS=('diagnostic_slice','selective_invalidation','selective_recovery')


def graph_options(config,group='G1'):
    explicit=config.get('graph_options',{})
    if not isinstance(explicit,dict) or not set(explicit)<=set(GRAPH_OPTIONS) or any(type(v) is not bool for v in explicit.values()):
        raise ValueError('graph_options must contain only the three boolean mechanism switches')
    if group=='B0':return None
    return {name:(explicit.get(name,True) if group=='G1' else False) for name in GRAPH_OPTIONS}


def ablation_name(config):
    disabled=[name for name,value in graph_options(config).items() if not value]
    return 'full' if not disabled else '+'.join('no_'+name for name in disabled)


def public_robot(obs):
    return {k:obs[k] for k in ('robot_joint_pos','robot_cartesian_pos') if k in obs}


def base_prompt(env,task,group,budgets):
    if group=='B0':
        prompt=copy.deepcopy(env._full_prompt)
        docs='\n'.join(api.combined_doc() for api in env._apis.values())
    else:
        from capx.execution.grounded_feedback import API_DOC
        docs=API_DOC
        prompt=[{'role':'system','content':'Generate Python using only the documented public robot and Vision APIs and numpy. No files, network, private attributes, env, obs or simulator state.'},
                {'role':'user','content':env._task_prompt+'\nAPI documentation:\n'+docs}]
    prompt.append({'role':'user','content':CONDITIONS[task]+'\nUse deployment-observable Vision/robot state only. '
        +f"Budget: {budgets['api_budget']} actual API calls per episode, {budgets['max_blocks']} complete code blocks, {budgets['block_seconds']} seconds per block. "
        +'The physical scene persists across blocks; Python variables are discarded. FINISH is only a claim, not physical success.'})
    return prompt,docs


def initial_prompt(base,obs):
    prompt=copy.deepcopy(base)
    prompt.append({'role':'user','content':'Return exactly one complete executable Python program in one fenced python block. Public robot observation: '+json.dumps(public_robot(obs),default=json_value)})
    return prompt


def b0_recovery_prompt(base,obs,code,info):
    from capx.utils.launch_utils import _build_multi_turn_decision_prompt
    context=copy.deepcopy(base)
    if isinstance(context[-1]['content'],str):context[-1]['content']=[{'type':'text','text':context[-1]['content']}]
    text=('Executed program:\n```python\n'+code+'\n```\nConsole stdout:\n'+str(info.get('stdout',''))
          +'\nConsole stderr:\n'+str(info.get('stderr',''))
          +'\nCurrent public robot observation: '+json.dumps(public_robot(obs),default=json_value)
          +'\nPhysical scene persists; write self-contained code and do not assume earlier Python variables exist.')
    return _build_multi_turn_decision_prompt({'full_prompt':context},text)


def install_monitor(env,group,task,dest,budgets,fault):
    import numpy as np
    from capx.execution.feedback import public_observation
    from capx.execution.grounded_feedback import GroundedMonitor,APIS
    from capx.execution.control_frames import transform_target
    from capx.execution.public_camera import robosuite_pixel_centers
    if group not in ('G0','G1'):raise ValueError('Static motion mapping is only installed for G0/G1')
    calibration=env.public_robot_calibration()
    assets=[]
    for source in calibration.get('sources',[]):
        path=Path(source['path']);data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
        if sha!=source['sha256']:raise ValueError('Robot calibration source drift: '+str(path))
        name=sha+'-'+path.name;archive_path=dest/'calibration-assets'/name
        archive_path.parent.mkdir(parents=True,exist_ok=True)
        if not archive_path.exists():
            with archive_path.open('xb') as stream:stream.write(data)
        assets.append(dict(source,archive_file=name))
    save(dest/'robot-calibration.json',calibration)
    save(dest/'calibration-assets'/'manifest.json',assets)
    # The existing IK service is a soft optimizer. Validate each solution
    # against local public robot kinematics before the control API can execute it.
    from capx.execution.bounded_ik import BoundedIK, IKSolveBudget
    urdf=next(row['path'] for row in calibration['sources'] if row['role']=='ik_urdf')
    api=env._apis['FrankaControlApi']
    api.ik_solve_fn=BoundedIK(api.ik_solve_fn,urdf_path=urdf,
        budget=IKSolveBudget(budgets['api_budget']*8),max_iterations=8,
        observe_joints=lambda:env._get_observation()['robot_joint_pos'],
        calibration_hash=calibration['calibration_hash'],
        emit=lambda row:env._write('ik_verification',**row))
    def motion_transform(position,quaternion_wxyz):
        mapped=transform_target(calibration,position,quaternion_wxyz)
        env._write('motion_frame_transform',calibration_hash=calibration['calibration_hash'],
            public_target=dict(position=list(position),quaternion_wxyz=list(quaternion_wxyz)),
            mapped_target=dict(position=mapped[0],quaternion_wxyz=mapped[1]))
        return mapped
    count=[0]
    def observe():
        raw=public_observation(env._get_observation())
        # The frame initializes/selects the current renderer before profiling.
        # Unknown resolve semantics fail closed; no scene-fitted ray offsets.
        state,camera_metadata=robosuite_pixel_centers(raw,
            depth_sampling=env.public_depth_sampling(),require_depth_sampling=True)
        camera_sources=monitor.register_camera_calibration(camera_metadata)
        count[0]+=1;arrays={}
        for key,value in state.items():
            if isinstance(value,dict):
                for prop in ('intrinsics','pose_mat'):
                    if prop in value:arrays[key+'_'+prop]=value[prop]
                for prop,im in value.get('images',{}).items():arrays[key+'_'+prop]=im
            else:arrays[key]=value
        p=dest/'observations'/f'{count[0]:05d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,**arrays)
        save(p.with_suffix('.camera.json'),{'calibration':camera_metadata,'graph_source_ids':camera_sources,
            'raw_tool_branch':'SAM3/GraspNet retains legacy camera interface; monitor surfaces use the declared corrected projection'})
        return state
    def emit(event):
        env._write('feedback',**event,observation_file=f'{count[0]:05d}.npz')
        env._write('feedback_evaluation',event_index=len(monitor.events)-1,state=env.evaluation_state(),
            phase=monitor.phase,held_object=monitor.held_object,
            placement_skill=(monitor.last_place or {}).get('skill'),local_attempts=monitor.local_attempts,
            hold_probes=monitor.hold_probes,program_block=getattr(monitor,'block',None))
    def graph_emit(event):
        env._write('graph_event',event=event)
        with (dest/'graph-events.jsonl').open('a') as f:f.write(json.dumps(event,default=json_value,allow_nan=False)+'\n')
    def contract_only(*a,**kw):raise RuntimeError('Grounded API must be dispatched by parent broker')
    monitor=GroundedMonitor(group,task,observe,emit=emit,graph_emit=graph_emit,max_local=budgets['max_local'],
        max_episode_local=budgets['max_episode_local'],max_episode_views=budgets.get('max_episode_views',2),
        fault=fault,site=lambda:env._actor.call_site,graph_options=budgets.get('graph_options'),motion_transform=motion_transform,
        ik_success_count=lambda:api.ik_solve_fn.successful_calls)
    if task=='restack':
        from capx.execution.robot_self_geometry import PandaSelfGeometry
        from capx.execution.table_clearance import current_table_evidence
        urdf=next(row['path'] for row in calibration['sources'] if row['role']=='ik_urdf')
        robot_geometry=PandaSelfGeometry(urdf,calibration)
        save(dest/'robot-geometry-assets.json',robot_geometry.metadata)
        for source in robot_geometry.sources:
            path=Path(source['path']);data=path.read_bytes();target=dest/'calibration-assets'/(source['sha256']+'-'+path.name)
            if hashlib.sha256(data).hexdigest()!=source['sha256']:
                raise ValueError('Robot geometry source drift: '+str(path))
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=source['sha256']:
                    raise ValueError('Robot geometry archive mismatch: '+str(target))
            else:
                with target.open('xb') as stream:stream.write(data)
        table_count=[0]
        def masks(arrays):
            table_count[0]+=1;path=dest/'table-masks'/f'{table_count[0]:05d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**arrays);return str(path)
        def table_evidence(state,region,held_name,held_prediction,**kwargs):
            return current_table_evidence(state,region,held_name,held_prediction,robot_geometry,
                env._apis['FrankaControlApi'].sam3_seg_fn,static_scene_assumption=True,artifact_callback=masks,
                reserve_semantic_refinement=monitor._reserve_semantic_refinement,**kwargs)
        monitor.table_evidence_fn=table_evidence
    env._functions.update({name:contract_only for name in APIS});env._actor.apis.update({name:None for name in APIS})
    env._feedback_monitor=monitor
    return monitor


class SourceChangedError(RuntimeError):
    pass


def exception_status(exc):
    if isinstance(exc,SourceChangedError):return 'source_changed'
    text=str(exc).lower()
    if isinstance(exc,RuntimeError) and any(word in text for word in ('budget exhausted','cost ceiling','wall time exhausted')):return 'budget_exceeded'
    return 'system_error'


def query(budget,prompt,label,episode_deadline):
    original=budget.deadline;budget.deadline=min(original,episode_deadline)
    started=time.monotonic()
    try:
        answer,source=budget.query(prompt,label)
        return answer,source,time.monotonic()-started
    finally:budget.deadline=original


def receipt_summary(paths):
    total=dict(input_tokens=0,output_tokens=0,reasoning_tokens=0,cost_usd=0.,usage_known_requests=0,usage_unknown_requests=0)
    for source in dict.fromkeys(paths):
        path=Path(source)/'receipt.json'
        if not path.exists():total['usage_unknown_requests']+=1;continue
        receipt=json.loads(path.read_text());usage=receipt.get('usage',{})
        if not all(isinstance(usage.get(k),(int,float)) for k in ('prompt_tokens','completion_tokens')):
            total['usage_unknown_requests']+=1;continue
        total['usage_known_requests']+=1;total['input_tokens']+=usage['prompt_tokens'];total['output_tokens']+=usage['completion_tokens']
        total['reasoning_tokens']+=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',usage.get('reasoning_tokens',0)) or 0
        total['cost_usd']+=receipt.get('cost_upper_usd',0.) or 0.
    return total


def simulator_termination(env):
    """Trusted runtime lifecycle only, never task truth or actor error text.

    Reset settling advances the native horizon before the wrapper zeroes its
    counter. That counter can lag a terminated simulator indefinitely.
    """
    low=getattr(env,'low_level_env',None)
    native=getattr(low,'robosuite_env',None)
    if getattr(native,'done',False) is not True:return None
    return dict(terminated=True,source='native_simulator_lifecycle',
        timestep=getattr(native,'timestep',None),horizon=getattr(native,'horizon',None),
        wrapper_steps=getattr(low,'_sim_step_count',None),wrapper_limit=getattr(low,'max_steps',None))


def run_episode(output,task,seed,repeat,snapshot,config,budget,group,initial=None,*,fault='none',source_hashes=None):
    from capx.experiments.graph_feedback import check_pair,parse_graph_recovery,graph_recovery_prompt
    dest=Path(output)/'episodes'/group;dest.mkdir(parents=True,exist_ok=False)
    b=limits(config);b['graph_options']=graph_options(config,group);start=time.monotonic();request_start=budget.count
    recovery=None;active_recipe=None;llm_seconds=0.
    if group!='B0' and b.get('bounded_recovery',False):
        from capx.execution.grounded_recovery import GroundedRecovery
        recovery=GroundedRecovery(max_attempts=b.get('max_recovery_plans',6),max_no_progress=b.get('max_recovery_no_progress',2))
    # A shared request saves real billing/wall time, not policy time for the
    # second arm. Charge the same initial latency before its recovery budget.
    shared_initial_seconds=float(initial.get('generation_seconds',0)) if initial else 0.
    if not math.isfinite(shared_initial_seconds) or shared_initial_seconds<0:raise ValueError('Invalid shared initial latency')
    episode_deadline=min(budget.deadline,start+b['episode_seconds']-shared_initial_seconds)
    result=dict(task=task,seed=seed,repeat=repeat,group=group,fault=fault,online_vision=True,model=config.get('model','deepseek-v4-pro'),
        status='system_error',reward=0.,policy_success=False,executed=False,replan_llm_requests=0,ablation=ablation_name(config),graph_options=graph_options(config,group),replans=0,blocks=[],recovery_decisions=[],request_paths=[],snapshot_hash=snapshot['state_hash'],
        interventions='API boundaries only; no in-flight motion cancellation',stop_reason=None,
        shared_initial_seconds_charged=shared_initial_seconds,episode_seconds_limit=b['episode_seconds'])
    env=monitor=None;generated_here=False
    def guard_sources():
        from capx.experiments.graph_campaign import verify_source_hashes
        changed=verify_source_hashes(source_hashes or {})
        if changed:
            result['source_changes']=changed
            raise SourceChangedError('Frozen local source changed')
    try:
        guard_sources()
        env=make_environment(task,dest/'trace',seed,snapshot,b['api_budget'])
        if env.reset_info['initial_state_hash']!=snapshot['state_hash']:raise ValueError('Snapshot replay hash mismatch')
        env.cfg.execution_timeout=b['block_seconds'];obs=env._get_observation();base,docs=base_prompt(env,task,group,b)
        if hasattr(env,'evaluation_geometry'):save(dest/'evaluation-geometry.json',env.evaluation_geometry())
        else:result['evaluation_geometry_missing']=True
        save(dest/'initial-evaluation-state.json',env.evaluation_state())
        prompt=initial_prompt(base,obs);result.update(initial_prompt_hash=digest(prompt),api_docs_hash=digest(docs),trace_path=str(env._trace_path))
        save(dest/'initial-prompt.json',prompt);save(dest/'api-docs.json',{'documentation':docs})
        env.enable_video_capture(True,clear=True);instrument_tools(env,dest)
        if group!='B0':
            monitor=install_monitor(env,group,task,dest,b,fault)
            # Validate only deployable RGB-D/robot geometry before paying for
            # code. This frame grants no skill permission: begin_program still
            # captures fresh evidence at its normal execution boundary.
            preflight_started=time.monotonic()
            result['public_preflight']={'status':'UNCERTAIN'}
            try:
                monitor.capture()
            except Exception as exc:
                result['public_preflight'].update(status='FAIL',error=type(exc).__name__)
                result['stop_reason']='public_observation_preflight_failed'
                # Geometry ValueError here is infrastructure, not generated code.
                raise RuntimeError('Public observation preflight failed') from exc
            else:
                result['public_preflight']['status']='PASS'
            finally:
                result['public_preflight']['seconds']=time.monotonic()-preflight_started
        guard_sources()
        if initial is None:
            generated_here=True
            initial=dict(code=None,response=None,request=None,request_count=0,generation_seconds=0.,snapshot_hash=result['snapshot_hash'],initial_prompt_hash=result['initial_prompt_hash'],api_docs_hash=result['api_docs_hash'])
            initial_request_start=budget.count
            try:
                answer,source,elapsed=query(budget,prompt,f'initial-{task}-seed-{seed}-repeat-{repeat}-'+('B0' if group=='B0' else 'shared-G0-G1'),episode_deadline)
                initial.update(response=answer,request=source,generation_seconds=elapsed)
                llm_seconds+=elapsed
                initial['code']=validated_program(answer)
            except (ValueError,SyntaxError) as exc:initial.update(code_error=type(exc).__name__+': '+str(exc),error_status='invalid_program')
            except Exception as exc:initial.update(code_error=type(exc).__name__,error_status=exception_status(exc))
            finally:initial['request_count']=budget.count-initial_request_start
            initial['code_hash']=digest(initial['code'])
            save(Path(output)/'initial'/('B0.json' if group=='B0' else 'G0-G1.json'),initial)
        result.update(initial_request=initial.get('request'),code_hash=initial['code_hash'],initial_generated_here=generated_here,initial_reused=bool(initial.get('cache_source')),initial_source=initial.get('cache_source'))
        check_pair(initial,result)
        if initial.get('request'):result['request_paths'].append(initial['request'])
        if initial['code'] is None:
            result.update(status=initial.get('error_status','invalid_program'),error=initial.get('code_error'),stop_reason='initial_generation_failed')
        else:
            code=initial['code'];source=initial['request']
            for block in range(b['max_blocks']):
                guard_sources()
                remaining=episode_deadline-time.monotonic()
                if remaining<=0:result.update(status='budget_exceeded',stop_reason='episode_or_global_wall_budget');break
                if env._api_calls>=b['api_budget']:result.update(status='budget_exceeded',stop_reason='episode_api_budget');break
                env.cfg.execution_timeout=min(b['block_seconds'],remaining)
                save(dest/f'block-{block}.json',dict(code=code,code_hash=digest(code),model_request=source,block=block,code_origin='bounded_skill_recipe' if active_recipe else 'deepseek',recovery_id=active_recipe))
                env._actor.close()
                if monitor:monitor.begin_program(code,block)
                # Independent private diagnostics. These objects never enter prompts/monitor.
                before_physics=env.evaluation_state()
                step_start=time.monotonic();result['executed']=True;result['current_block']=block
                try:
                    obs,reward,done,truncated,info=env.step(code)
                except Exception as exc:
                    lifecycle=simulator_termination(env)
                    if lifecycle is None:raise
                    env._write('runtime_termination',**lifecycle)
                    result.update(status='truncated',truncated=True,policy_success=False,
                        stop_reason='simulator_episode_terminated',runtime_termination=lifecycle,
                        error=type(exc).__name__,reward_is_last_completed_evaluation=True)
                    result['blocks'].append(dict(block=block,seconds=time.monotonic()-step_start,
                        execution_status='simulator_terminated',reward=None,api_calls=env._api_calls,
                        model_request=source,policy_success=False))
                    break
                lifecycle=simulator_termination(env)
                if lifecycle is not None:
                    truncated=True
                    result['runtime_termination']=lifecycle
                    env._write('runtime_termination',**lifecycle)
                result.update(reward=reward,info=info,truncated=truncated,policy_success=False,status=info['execution_status'] if info['execution_status']!='ok' else 'task_failure')
                save(dest/f'block-{block}-evaluation.json',dict(block=block,before=before_physics,after=env.evaluation_state(),
                    phase=getattr(monitor,'phase',None),held_object=getattr(monitor,'held_object',None)))
                if recovery:recovery.complete(monitor,info['execution_status'])
                if monitor and info['execution_status'] in ('runtime_error','invalid_program','policy_violation'):
                    monitor.end_block_failure(info['execution_status'],info.get('stderr',''))
                check=monitor.finish_check() if monitor and not truncated else None
                result['blocks'].append(dict(block=block,seconds=time.monotonic()-step_start,reward=reward,execution_status=info['execution_status'],api_calls=env._api_calls,finish_check=check,model_request=source))
                if monitor:save(dest/f'graph-block-{block}.json',monitor.graph.snapshot())
                # The evaluator's original reward is immutable. Policy completion
                # additionally requires a clean, nonterminal return within the
                # deadline AFTER independent evaluation and public verification.
                # Grounded policies must also finish their public skill contract
                # (e.g. release -> retreat); this does not alter physical reward.
                terminal=getattr(monitor,'terminal_reason',None)
                checks=dict(original_task_success=reward==1 and bool(done),execution_ok=info['execution_status']=='ok',
                    not_truncated=not truncated,no_terminal_exit=not terminal,within_wall_budget=time.monotonic()<episode_deadline)
                if monitor:
                    checks['public_stage_complete']=isinstance(check,dict) and check.get('status')=='PASS'
                result['completion_contract']='grounded_public_skill_phase' if monitor else 'unmonitored_original_api'
                result.update(policy_checks=checks,terminal_reason=terminal,policy_success=all(checks.values()),
                    policy_elapsed_seconds=time.monotonic()-start+shared_initial_seconds)
                result['blocks'][-1]['policy_success']=result['policy_success']
                if truncated:
                    if info['execution_status']=='ok' or lifecycle is not None:result['status']='truncated'
                    result['stop_reason']='simulator_episode_terminated' if lifecycle is not None else 'environment_timeout_or_truncation';break
                if terminal:
                    result.update(status='budget_exceeded' if 'budget' in terminal else 'controlled_exit',stop_reason=terminal);break
                if not checks['within_wall_budget']:
                    result.update(status='budget_exceeded',stop_reason='episode_or_global_wall_budget');break
                if result['policy_success']:
                    result.update(status='success',stop_reason='physical_success');break
                pending_public_completion=bool(monitor and checks['original_task_success']
                    and checks['execution_ok'] and not checks['public_stage_complete'])
                if done and not pending_public_completion:
                    result['stop_reason']='inconsistent_or_non_ok_environment_completion';break
                if block+1>=b['max_blocks']:result['stop_reason']='episode_code_block_budget';break
                if env._api_calls>=b['api_budget']:result['stop_reason']='episode_api_budget';break
                if time.monotonic()>=episode_deadline or budget.count>=budget.max_requests:result['stop_reason']='global_model_or_episode_wall_budget';break
                if recovery:
                    recipe=recovery.propose(monitor,execution_status=info['execution_status'])
                    if recipe:
                        save(dest/f'bounded-recovery-{block+1}.json',recipe)
                        code=recipe['code'];source=None;active_recipe=recipe['id']
                        continue
                active_recipe=None
                if group=='B0':
                    prompt=b0_recovery_prompt(base,obs,code,info)
                    context={'feedback_type':'original_program_console','stdout':info.get('stdout',''),'stderr':info.get('stderr','')}
                else:
                    feedback=monitor.summary();feedback.update(remaining_api_calls=b['api_budget']-env._api_calls,remaining_code_blocks=b['max_blocks']-block-1,
                        completion_check=check)
                    from capx.execution.grounded_feedback import API_DOC
                    prompt,selection=graph_recovery_prompt({'full_prompt':base},code,feedback,info['execution_status'],group,api_doc=API_DOC)
                    context=dict(feedback=feedback,**selection)
                save(dest/f'recovery-context-{block+1}.json',context);save(dest/f'recovery-prompt-{block+1}.json',prompt)
                guard_sources()
                result['replans']+=1
                replan_request_start=budget.count
                try:
                    answer,source,elapsed=query(budget,prompt,f'replan-{task}-{group}-seed-{seed}-repeat-{repeat}-block-{block+1}',episode_deadline)
                    llm_seconds+=elapsed
                finally:result['replan_llm_requests']+=budget.count-replan_request_start
                result['request_paths'].append(source)
                if answer.get('finish_reason')!='stop' or answer.get('response_model')!=config.get('model','deepseek-v4-pro'):raise ValueError('Incomplete or wrong-model recovery response')
                decision,new_code,repair=parse_graph_recovery(answer.get('content'))
                result['recovery_decisions'].append(dict(block=block+1,request=source,decision=decision,format_repair=repair))
                if decision=='finish':result['stop_reason']='model_finish_claim_without_reward_change';result['finish_claim']=answer['content'];break
                code=new_code
        result['final_physics']=env.evaluation_state()
        result['policy_elapsed_seconds']=time.monotonic()-start+shared_initial_seconds
        if result['policy_success'] and time.monotonic()>=episode_deadline:
            result.update(policy_success=False,status='budget_exceeded',stop_reason='episode_or_global_wall_budget')
            result['policy_checks']['within_wall_budget']=False
            if result['blocks']:result['blocks'][-1]['policy_success']=False
    except KeyboardInterrupt:
        result.update(status='interrupted_by_operator',policy_success=False,stop_reason='interrupted_by_operator',
                      reward_is_last_completed_evaluation=True,interrupted_block=result.get('current_block'),
                      operator_interruption_unix=time.time())
        if env:
            try:result['final_physics']=env.evaluation_state()
            except Exception as exc:result['interrupted_evaluation_error']=type(exc).__name__
        raise
    except (ValueError,SyntaxError) as exc:
        result.update(status='invalid_program',policy_success=False,error=type(exc).__name__+': '+str(exc)[:700],stop_reason=result['stop_reason'] or 'invalid_program')
    except Exception as exc:
        result.update(status=exception_status(exc),policy_success=False,error=type(exc).__name__,stop_reason=result['stop_reason'] or ('resource_budget_exhausted' if exception_status(exc)=='budget_exceeded' else 'frozen_source_drift' if isinstance(exc,SourceChangedError) else 'infrastructure_or_runtime_error'))
    finally:
        if recovery:
            # Includes interrupted final recipes; never label partial calls done.
            if recovery.pending:recovery.complete(monitor,result['status'])
            result['bounded_recovery_events']=recovery.events
        if monitor:
            result['component_timing']={**getattr(monitor,'timing',{}),'llm_seconds':llm_seconds+shared_initial_seconds,
                'note':'control includes IK; public observation includes RGB-D/render/geometry/storage; monitor is exclusive dispatch overhead; tool service timing is nested, not additive'}
            result['contract_calls']=getattr(monitor,'contract_calls',{})
            result.update(local_recoveries=monitor.recoveries,local_attempts=monitor.local_attempts,feedback_events=monitor.events,
                graph_events=monitor.graph.events,fault_fired=monitor.fault_fired,phase=monitor.phase,
                detection={'unknown':len(monitor.events)})
            save(dest/'graph-final.json',monitor.graph.snapshot())
        if env:
            result.update(actual_api_calls=env._api_calls,http_requests=env._request_audit.count)
            try:
                frames=env.get_video_frames()
                if frames:
                    from capx.utils.video_utils import _write_video
                    _write_video(frames,str(dest),suffix=group,timestamps=list(env.low_level_env._frame_timestamps))
                    result['video']=str(dest/f'video_{group}.mp4')
            except Exception as exc:result['video_error']=type(exc).__name__
            finally:env.close()
        # A final successful block can return without another loop boundary.
        # Check after environment/video cleanup as well; retain independently
        # measured physics, but never accept a policy result with source drift.
        try:
            guard_sources()
        except SourceChangedError:
            result.update(status='source_changed',policy_success=False,
                stop_reason='frozen_source_drift',error='SourceChangedError',
                source_postcheck=dict(unchanged=False,checked_files=len(source_hashes or {})))
        else:
            result['source_postcheck']=dict(unchanged=True,checked_files=len(source_hashes or {}))
        result['total_seconds']=time.monotonic()-start
        result['end_to_end_seconds']=result['total_seconds']+(0 if generated_here or not initial else initial.get('generation_seconds',0))
        result.update(new_llm_requests=budget.count-request_start,llm_requests=(initial.get('request_count',int(bool(initial.get('request')))) if initial else 0)+result['replan_llm_requests'])
        result.update(receipt_summary(result['request_paths']));save(dest/'result.json',result)
    print(task,seed,repeat,group,result['status'],result['reward'],result['stop_reason'],flush=True)
    return result,initial


def run_pair(output,task,seed,repeat,snapshot,config,request_budget,groups=GROUPS,fault='none',initial_ref=None,campaign_ref=None):
    if task not in TASKS or type(seed) is not int or not 0<=seed<2**32 or type(repeat) is not int or repeat<0:raise ValueError('Invalid task/scene identity')
    if not groups or len(set(groups))!=len(groups) or not set(groups)<=set(GROUPS):raise ValueError('Invalid groups')
    if fault!='none' and 'B0' in groups:raise ValueError('B0 has no equivalent evidence-fault injection; run stress G0/G1 separately')
    if config.get('model','deepseek-v4-pro')!='deepseek-v4-pro':raise ValueError('Only configured DeepSeek V4 Pro supported; no substitution')
    configure_services(config);limits(config);graph_options(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    eager_execution_imports()
    source_hashes=archive(output,config,initial_ref=initial_ref,campaign_ref=campaign_ref)
    try:
        if snapshot is None:entry,snapshot=prepare_scene(task,seed,output/'scene')
        else:
            if isinstance(snapshot,(str,Path)):snapshot=json.loads(Path(snapshot).read_text())
            entry={'task':task,'seed':seed,'state_hash':snapshot['state_hash']};save(output/'scene'/'snapshot.json',snapshot)
    except Exception as exc:
        rows=[dict(task=task,seed=seed,repeat=repeat,group=group,fault=fault,status='scene_preparation_error',
            reward=0.,executed=False,ablation=ablation_name(config),graph_options=graph_options(config,group),error=type(exc).__name__,snapshot_hash=None,online_vision=False,stop_reason='scene_preparation_failed') for group in groups]
        for row in rows:save(output/'episodes'/row['group']/'result.json',row)
        result=dict(episodes=rows,task=task,seed=seed,repeat=repeat,fault=fault)
        save(output/'summary.json',result);return result
    shared=None
    if initial_ref is not None:
        if 'B0' in groups:raise ValueError('Shared grounded initial cannot substitute B0 legal interface')
        from capx.experiments.shared_initial import load_initial_reference
        shared=load_initial_reference(initial_ref,config,task=task,seed=seed,repeat=repeat,fault=fault,snapshot=snapshot,_cache=(campaign_ref or {}).get('_initial_validation_cache'))
        save(output/'initial'/'G0-G1.json',shared)
        save(output/'initial'/'source-reference.json',initial_ref)
    save(output/'experiment.json',dict(task=task,seed=seed,repeat=repeat,groups=list(groups),fault=fault,config=config,source_hashes=source_hashes,snapshot_hash=snapshot['state_hash'],design='G0/G1 shared initial program and snapshot; B0 legal raw Vision API with original program-level multi-turn feedback'))
    save(output/'snapshot-source.json',{'entry':entry,'snapshot_hash':snapshot['state_hash']})
    rows=[]
    for group in groups:
        from capx.experiments.graph_campaign import verify_source_hashes
        changed=verify_source_hashes(source_hashes)
        if changed:
            row=dict(task=task,seed=seed,repeat=repeat,group=group,fault=fault,ablation=ablation_name(config),
                graph_options=graph_options(config,group),status='source_changed',stop_reason='frozen_source_drift',
                reward=None,executed=False,online_vision=False,source_changes=changed,snapshot_hash=snapshot['state_hash'],
                llm_requests=0,new_llm_requests=0,actual_api_calls=0,blocks=[],request_paths=[])
            save(output/'episodes'/group/'result.json',row);rows.append(row);continue
        row,initial=run_episode(output,task,seed,repeat,snapshot,config,request_budget,group,shared if group!='B0' else None,fault=fault,source_hashes=source_hashes)
        rows.append(row)
        if group!='B0':shared=initial
    result=dict(episodes=rows,task=task,seed=seed,repeat=repeat,fault=fault)
    save(output/'summary.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--groups',nargs='+',choices=GROUPS,default=list(GROUPS));p.add_argument('--fault',choices=['none','support_refresh','missing_baseline','target_shift'],default='none');p.add_argument('--snapshot',type=Path);p.add_argument('--initial-manifest',type=Path)
    a=p.parse_args(argv);config=json.loads(a.config.read_text());configure_services(config)
    if a.output.exists():raise FileExistsError('Use a new output directory; campaign handles resume')
    a.output.mkdir(parents=True)
    from capx.experiments.graph_campaign import PersistentModelBudget
    budget=PersistentModelBudget(a.output,config['budgets']);snapshot=json.loads(a.snapshot.read_text()) if a.snapshot else None
    initial_ref=None
    if a.initial_manifest:
        from capx.experiments.shared_initial import validate_initial_manifest
        refs=validate_initial_manifest(a.initial_manifest,config)
        initial_ref=refs.get((a.task,a.seed,a.repeat,a.fault))
        if initial_ref is None:raise ValueError('Shared initial manifest lacks requested identity; no fallback generation')
    run_pair(a.output,a.task,a.seed,a.repeat,snapshot,config,budget,tuple(a.groups),a.fault,initial_ref=initial_ref)

if __name__=='__main__':main()
