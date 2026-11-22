"""Bounded paired DeepSeek + live Vision experiment on the existing P0 runner."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from capx.experiments.pilot import dump, digest, one_program, environment, load_scene, prepare, CostLedger
from capx.envs.reproducibility import json_value


def validated_program(answer):
    if answer.get('finish_reason')!='stop':raise ValueError('Incomplete model response; refusing truncated code')
    if answer.get('response_model')!='deepseek-v4-pro':raise ValueError('Response model mismatch')
    return one_program(answer.get('content'))


def initial_prompt(env,obs):
    prompt=copy.deepcopy(env._full_prompt)
    public={k:obs[k] for k in ('robot_joint_pos','robot_cartesian_pos') if k in obs}
    prompt.append({'role':'user','content':'Return exactly one complete Python program in one fenced python block for the task. Use the documented Vision APIs; no files, network, private attributes, or simulator state. Budget: 64 API calls per episode, 90 seconds per code block. Public robot observation: '+json.dumps(public,default=json_value)})
    return prompt


def recovery_prompt(obs,code,feedback,status):
    from capx.utils.launch_utils import _build_multi_turn_decision_prompt
    text=('Previous program:\n```python\n'+code+'\n```\nExecution status: '+status+
          '\nStructured deployment-observable feedback:\n'+json.dumps(feedback,default=json_value)+
          '\nThe physical scene persists, but Python variables may have been discarded. '
          'Write self-contained recovery code. Do not blindly open a possibly loaded gripper: '
          'the release verifier will block unsafe release. You can re-perceive, realign, '
          'regrasp, or replan with the documented APIs. Reuse current robot state above. '
          'No hidden success flag or simulator object pose is available. FINISH is only a claim.')
    # Existing CaP-X multi-round format and decision parser, text-only Structured Vision.
    return _build_multi_turn_decision_prompt({'full_prompt':obs['full_prompt']},text)


class ModelBudget:
    def __init__(self,root,max_requests,cost_limit,max_tokens,deadline,reasoning_effort="low",request_timeout=400):
        self.root=root;self.max_requests=max_requests;self.max_tokens=max_tokens
        self.ledger=CostLedger(cost_limit);self.count=0;self.rows=[];self.deadline=deadline
        self.reasoning_effort=reasoning_effort;self.request_timeout=request_timeout

    def query(self,prompt,label):
        from capx.llm.providers import provider_config
        from capx.llm.client import ModelQueryArgs,query_model
        from capx.execution.sandbox import callback_deadline
        if self.count>=self.max_requests:raise RuntimeError('Global LLM request budget exhausted')
        remaining=self.deadline-time.monotonic()
        if remaining<10:raise RuntimeError('Global wall time exhausted')
        url,model,key=provider_config('deepseek')
        if model!='deepseek-v4-pro':raise ValueError('Expected configured deepseek-v4-pro; no substitution')
        bound=self.ledger.reserve('deepseek',input_bytes=len(json.dumps(prompt).encode()),max_tokens=self.max_tokens)
        self.count+=1;dest=self.root/'llm'/f'{self.count:03d}-{label}';dest.mkdir(parents=True,exist_ok=False)
        dump(dest/'prompt.json',prompt)
        row=dict(label=label,request_number=self.count,requested_model=model,max_tokens=self.max_tokens,
                 attempts=0,status='started',cost_upper_usd=bound)
        self.rows.append(row)
        def observe(attempt,payload,response,error):
            body={}
            if response is not None:
                try:body=response.json()
                except ValueError:pass
            usage=body.get('usage',{});it=usage.get('prompt_tokens');ot=usage.get('completion_tokens')
            cost=(it*1.32+ot*3.96)/1e6 if it is not None and ot is not None else bound
            self.ledger.spent['deepseek']+=cost
            row.update(attempts=attempt,http_status=getattr(response,'status_code',None),usage=usage,
                       cost_upper_usd=cost,error=error,request_id=(getattr(response,'headers',{}) or {}).get('x-request-id') or (getattr(response,'headers',{}) or {}).get('x-ds-request-id'), response_id=body.get('id'))
            dump(dest/'http.json',dict(payload=payload,response=body,**row))
        args=ModelQueryArgs(model=model,server_url=url,api_key=key,max_tokens=self.max_tokens,max_attempts=1,
                            request_timeout=min(self.request_timeout,remaining),response_observer=observe,deepseek_reasoning_effort=self.reasoning_effort)
        started=time.monotonic()
        try:
            with callback_deadline(started+min(self.request_timeout+5,remaining)):
                answer=query_model(args,prompt)
            row.update(status='received',finish_reason=answer['finish_reason'],response_model=answer['response_model'])
            dump(dest/'response.json',answer)
            return answer,str(dest)
        except Exception as exc:
            row.update(status='error',error=type(exc).__name__+': '+str(exc)[:500]);raise
        finally:
            row['seconds']=time.monotonic()-started
            if not row['attempts']:self.ledger.spent['deepseek']+=bound
            dump(dest/'receipt.json',row)


def instrument_tools(env,dest):
    """Record actual tool inputs and outputs separately from private physics."""
    for api in env._apis.values():
        for attr,label in [('sam3_seg_fn','sam3'),('grasp_net_plan_fn','graspnet'),('ik_solve_fn','ik')]:
            if not hasattr(api,attr):continue
            original=getattr(api,attr);counter=[0]
            def wrapped(*args,_fn=original,_label=label,_counter=counter,**kwargs):
                _counter[0]+=1;path=dest/'tools'/f'{_label}-{_counter[0]:03d}.json'
                start=time.monotonic();result=None;error=None
                try:result=_fn(*args,**kwargs);return result
                except Exception as exc:error=type(exc).__name__+': '+str(exc);raise
                finally:
                    # Arrays (RGB, masks, cloud candidates) live in a compressed bundle.
                    arrays={}
                    def pack(v):
                        if isinstance(v,np.ndarray):
                            key=f'array_{len(arrays)}';arrays[key]=v
                            return {'npz_key':key,'shape':v.shape,'dtype':str(v.dtype)}
                        if isinstance(v,dict):return {str(k):pack(x) for k,x in v.items()}
                        if isinstance(v,(tuple,list)):return [pack(x) for x in v]
                        return v
                    record=dict(tool=_label,args=pack(args),kwargs=pack(kwargs),response=pack(result),error=error,
                                elapsed_seconds=time.monotonic()-start)
                    dump(path,record)
                    if arrays:np.savez_compressed(path.with_suffix('.npz'),**arrays)
            setattr(api,attr,wrapped)


def install_monitor(env,group,dest,max_local):
    from capx.execution.feedback import FeedbackMonitor,public_observation
    from capx.envs.physical_evaluation import physical_state
    count=[0]
    def observe():
        state=public_observation(env._get_observation());count[0]+=1
        arrays={k:v for k,v in state.items() if not isinstance(v,dict)}
        for camera,value in state.items():
            if isinstance(value,dict):
                for k in ('intrinsics','pose_mat'):arrays[camera+'_'+k]=value[k]
                for k,v in value['images'].items():arrays[camera+'_'+k]=v
        p=dest/'observations'/f'{count[0]:04d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,**arrays)
        return state
    def emit(event):
        env._write('feedback',**event,observation_file=f'{count[0]:04d}.npz')
        # Private evaluator ledger is NEVER returned to the monitor or LLM.
        env._write('feedback_evaluation',event_index=len(monitor.events)-1,state=physical_state(env.low_level_env))
    monitor=FeedbackMonitor(group,observe,emit=emit,max_local=max_local)
    if group!='A':env._feedback_monitor=monitor
    return monitor


def run_episode(root,manifest,entry,group,initial,budget,args):
    from capx.utils.video_utils import _write_video
    from capx.utils.launch_utils import _parse_multi_turn_decision
    from capx.envs.physical_evaluation import physical_state
    label=f'{group}-seed-{entry["seed"]}';dest=root/'episodes'/label;dest.mkdir(parents=True,exist_ok=False)
    result=dict(group=group,seed=entry['seed'],online_vision=True,model='deepseek-v4-pro',status='system_error',
                reward=None,replans=0,replan_effective=False,blocks=[],initial_request=None)
    env=None;start=time.monotonic();requests_start=budget.count
    try:
        env=environment('vision',dest/'trace');env.cfg.feedback_api_budget=args.api_budget
        obs,info=env.reset(seed=entry['seed'],options={'snapshot':load_scene(manifest,entry)})
        assert info['initial_state_hash']==entry['state_hash']
        result.update(snapshot_hash=info['initial_state_hash'],trace_path=str(env._trace_path),api_docs_hash=digest([a.combined_doc() for a in env._apis.values()]))
        low=env.low_level_env;sim=low.robosuite_env.sim
        dump(dest/'evaluation-geometry.json',dict(red_half_size=low.robosuite_env.cubeA.size,
             green_half_size=low.robosuite_env.cubeB.size,base_position_world=sim.data.xpos[low.base_link_idx],
             base_rotation_world=sim.data.xmat[low.base_link_idx].reshape(3,3),model_hash=entry['model_hash']))
        prompt=initial_prompt(env,obs);result['initial_prompt_hash']=digest(prompt)
        env.enable_video_capture(True,clear=True);instrument_tools(env,dest)
        monitor=install_monitor(env,group,dest,args.max_local)
        if initial is None:
            answer,source=budget.query(prompt,f'initial-seed-{entry["seed"]}')
            initial={'code':None,'request':source,'prompt_hash':digest(prompt),'response':answer}
            try:initial['code']=validated_program(answer)
            except (ValueError,SyntaxError) as exc:initial['code_error']=str(exc)
            dump(root/'initial'/f'seed-{entry["seed"]}.json',initial)
        assert initial['prompt_hash']==digest(prompt),'Paired initial prompt drift'
        result['initial_request']=initial['request'];result['code_hash']=digest(initial['code'])
        if initial['code'] is None:raise ValueError(initial['code_error'])
        code=initial['code']
        for block in range(args.max_blocks):
            if time.monotonic()>=budget.deadline:raise RuntimeError('Global wall time exhausted')
            dump(dest/f'block-{block}.json',{'code':code})
            t=time.monotonic();obs,reward,done,truncated,info=env.step(code)
            check=monitor.finish_check() if group in ('D','E') and not truncated else None
            result['blocks'].append(dict(block=block,seconds=time.monotonic()-t,reward=reward,
                                         execution_status=info['execution_status'],api_calls=info['api_calls']))
            result.update(reward=reward,info=info,status='success' if reward else info['execution_status'] if info['execution_status']!='ok' else 'task_failure')
            # Evaluation can end a successful episode; evaluator details never enter prompts.
            if done or truncated or group!='E' or block+1>=args.max_blocks:break
            if info['execution_status']=='ok' and check and check['action']=='finish_candidate':break
            if env._api_calls>=args.api_budget:break
            feedback=monitor.summary();feedback['remaining_api_calls']=args.api_budget-env._api_calls
            feedback['remaining_code_blocks']=args.max_blocks-block-1
            prompt=recovery_prompt(obs,code,feedback,info['execution_status'])
            result['replans']+=1
            answer,source=budget.query(prompt,f'replan-{label}-{block+1}')
            if answer.get('finish_reason')!='stop' or answer.get('response_model')!='deepseek-v4-pro':
                raise ValueError('Incomplete or wrong-model recovery response')
            decision,new_code=_parse_multi_turn_decision(answer['content'])
            if decision=='finish':
                result['finish_claim']=answer['content'];break
            code=one_program(new_code)
        result.update(replan_effective=bool(result['replans'] and result['reward']),recoveries=monitor.recoveries,
                      feedback_events=monitor.events,final_physics=physical_state(env.low_level_env))
    except (ValueError,SyntaxError) as exc:
        result.update(error=type(exc).__name__+': '+str(exc)[:1000],status='invalid_program',reward=0.)
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc)[:1000],status='system_error')
    finally:
        if env is not None:
            frames=env.get_video_frames()
            if frames:
                _write_video(frames,str(dest),suffix=label,timestamps=list(env.low_level_env._frame_timestamps))
            result['actual_api_calls']=env._api_calls
            result['http_requests']=env._request_audit.count
            env.close()
        result.update(total_seconds=time.monotonic()-start,new_llm_requests=budget.count-requests_start)
        dump(dest/'result.json',result)
    print(label,result['status'],result['reward'],'replans',result['replans'],flush=True)
    return result,initial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot-manifest',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,21])
    p.add_argument('--groups',nargs='+',choices=list('ABCDE'),default=list('ABCDE'))
    p.add_argument('--max-requests',type=int,default=15);p.add_argument('--cost-limit-usd',type=float,default=5)
    p.add_argument('--max-tokens',type=int,default=32768);p.add_argument('--max-blocks',type=int,default=3)
    p.add_argument('--api-budget',type=int,default=64);p.add_argument('--max-local',type=int,default=2)
    p.add_argument('--wall-seconds',type=float,default=2400)
    p.add_argument('--reasoning-effort',choices=['low','high','max'],default='low')
    p.add_argument('--request-timeout',type=float,default=400)
    p.add_argument('--sam3-url',default=os.environ.get('SAM3_SERVICE_URL'))
    p.add_argument('--graspnet-url',default=os.environ.get('GRASPNET_SERVICE_URL'))
    p.add_argument('--pyroki-url',default=os.environ.get('PYROKI_SERVICE_URL'))
    a=p.parse_args()
    if not all([a.sam3_url,a.graspnet_url,a.pyroki_url]):p.error('Set SAM3_SERVICE_URL, GRASPNET_SERVICE_URL, PYROKI_SERVICE_URL or explicit URL arguments')
    if min(a.max_requests,a.cost_limit_usd,a.max_tokens,a.max_blocks,a.api_budget,a.wall_seconds)<=0 or a.max_local<0:p.error('Budgets must be positive')
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.groups))!=len(a.groups):p.error('Duplicate seeds/groups')
    for name,value in [('SAM3',a.sam3_url),('GRASPNET',a.graspnet_url),('PYROKI',a.pyroki_url)]:os.environ[name+'_SERVICE_URL']=value
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    config['source_hashes']={str(p):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('capx/execution/feedback.py'),Path('capx/execution/sandbox.py'),Path('capx/envs/tasks/audited_stack.py')]}
    config['design']='Same real DeepSeek initial code paired across groups; E alone may request recovery. No Oracle control code.'
    dump(root/'experiment.json',config)
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    (root/'gpu-before.txt').write_text(gpu.stdout+gpu.stderr)
    start=time.monotonic();budget=ModelBudget(root,a.max_requests,a.cost_limit_usd,a.max_tokens,start+a.wall_seconds,a.reasoning_effort,a.request_timeout)
    rows=[]
    try:
        manifest=a.snapshot_manifest
        if manifest is None:
            manifest=root/'manifest'/'scenes.json';prepare(manifest,root,seeds=a.seeds)
        entries={e['seed']:e for e in json.loads(manifest.read_text())['scenes']}
        dump(root/'snapshot-source.json',{'manifest':str(manifest.resolve()),'entries':[entries[s] for s in a.seeds]})
        for seed in a.seeds:
            initial=None
            for group in a.groups:
                row,initial=run_episode(root,manifest,entries[seed],group,initial,budget,a);rows.append(row)
                if row['status']=='system_error':raise RuntimeError('Infrastructure/model failure: stop batch; inspect episode result')
    finally:
        dump(root/'summary.json',{'results':rows,'llm_requests':budget.rows,'cost_upper_usd':budget.ledger.spent['deepseek'],'seconds':time.monotonic()-start})
        gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
        (root/'gpu-after.txt').write_text(gpu.stdout+gpu.stderr)

if __name__=='__main__':main()
