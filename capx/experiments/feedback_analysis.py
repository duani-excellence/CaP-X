"""Offline evaluation only. Simulator truth stays outside monitor and LLM input."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from capx.experiments.pilot import dump


def confusion(rows):
    counts=Counter()
    for r in rows:
        if r['status']=='UNCERTAIN':counts['uncertain']+=1
        elif r['truth_bad']:counts['tp' if r['status']=='FAIL' else 'fn']+=1
        else:counts['fp' if r['status']=='FAIL' else 'tn']+=1
    result={k:counts[k] for k in ('tp','tn','fp','fn','uncertain')}
    n=sum(result[k] for k in ('tp','tn','fp','fn'))
    result['accuracy_decided']=(result['tp']+result['tn'])/n if n else None
    return result


def truth_tracking(state,geometry,metrics):
    r=np.asarray(geometry['base_rotation_world']);b=np.asarray(geometry['base_position_world'])
    p=r.T@(np.asarray(state['tcp']['position'])-b)
    q=state['tcp']['quaternion_wxyz']
    actual=r.T@Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    target=np.asarray(metrics['target_tcp'])
    desired=Rotation.from_quat(target[3:][[1,2,3,0]]).as_matrix()
    distance=float(np.linalg.norm(p-target[:3]));angle=float(Rotation.from_matrix(actual.T@desired).magnitude())
    return dict(position_error_m=distance,orientation_error_rad=angle,truth_bad=distance>.025 or angle>np.deg2rad(20))


def extent(q,half):
    return abs(Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()) @ np.asarray(half)


def truth_release(state,geometry):
    red,green=state['objects']['primary'],state['objects']['secondary']
    rp,gp=np.array(red['position']),np.array(green['position'])
    re=extent(red['quaternion_wxyz'],geometry['red_half_size'])
    ge=extent(green['quaternion_wxyz'],geometry['green_half_size'])
    xy=float(np.linalg.norm(rp[:2]-gp[:2]));gap=float(rp[2]-re[2]-(gp[2]+ge[2]))
    # Diagnostic risk thresholds, NOT a modification to P0 reward.
    return dict(xy_error_m=xy,bottom_gap_m=gap,held=state['grasp_contact'],
                truth_bad=bool(xy>.010 or gap>.020 or gap<-.006 or not state['grasp_contact']))


def skill_counts(apis,states,geometry):
    from capx.envs.physical_evaluation import PhysicalCriteria
    counts=dict(grasp_attempts=0,grasp_contact_after_close=0,lift_attempts=0,lift_with_contact=0,
                release_attempts=0,stable_after_release=0)
    r=np.asarray(geometry['base_rotation_world']);b=np.asarray(geometry['base_position_world'])
    for i,api in enumerate(apis):
        before,after=api['before'],api['after']
        if api['name']=='close_gripper':
            counts['grasp_attempts']+=1;counts['grasp_contact_after_close']+=int(after['grasp_contact'])
        if api['name']=='goto_pose' and before['gripper_command']<.5:
            p=np.asarray(api['args'][0] if api['args'] else api['kwargs']['position'])
            old_tcp=r.T@(np.asarray(before['tcp']['position'])-b)
            if p[2]-old_tcp[2]>.05:
                counts['lift_attempts']+=1
                rise=after['objects']['primary']['position'][2]-before['objects']['primary']['position'][2]
                counts['lift_with_contact']+=int(rise>.035 and after['grasp_contact'])
        if api['name']=='open_gripper' and before['gripper_command']<.5:
            counts['release_attempts']+=1
            end=next((v['before']['sim_time'] for v in apis[i+1:] if v['name']=='close_gripper'),float('inf'))
            evaluator=PhysicalCriteria(before);success=False
            for state in states:
                if before['sim_time']<=state['sim_time']<end:
                    evaluator.update(state);success=success or evaluator.stable_placement
            counts['stable_after_release']+=int(success)
    return counts


def analyze_episode(path):
    result=json.loads(path.read_text());trace=Path(result['trace_path'])
    rows=[json.loads(l) for l in trace.read_text().splitlines()]
    states=[r['state'] for r in rows if r['kind']=='physics'];apis=[r for r in rows if r['kind']=='api']
    ev=[r for r in rows if r['kind']=='feedback'];private={r['event_index']:r['state'] for r in rows if r['kind']=='feedback_evaluation'}
    geometry_path=path.parent/'evaluation-geometry.json';geometry=json.loads(geometry_path.read_text()) if geometry_path.exists() else None
    start=apis[0]['before']['objects']['primary']['position'][2] if apis else None
    outcomes=dict(group=result['group'],seed=result['seed'],reward=result['reward'],status=result['status'],
                  execution_seconds=sum(b['seconds'] for b in result['blocks']),total_seconds=result['total_seconds'],
                  api_calls=result['actual_api_calls'],http_requests=result['http_requests'],tool_http_requests=result['http_requests']-result.get('new_llm_requests',0),replans=result['replans'],
                  replan_effective=result['replan_effective'],recoveries=result.get('recoveries',0),
                  grasp_success=result.get('info',{}).get('grasp_success'),slipped=result.get('info',{}).get('slipped'),
                  max_lift_m=max((s['objects']['primary']['position'][2]-start for s in states),default=0) if start is not None else 0,
                  failure_reasons=[e['reason'] for e in ev if e['status']=='FAIL'],checks=[])
    last_api=None;api_at_feedback=[]
    for row in rows:
        if row['kind']=='api':last_api=row
        if row['kind']=='feedback':api_at_feedback.append(last_api)
    for i,event in enumerate(ev):
        state=private.get(i)
        if state is None:continue
        item={'layer':event['layer'],'check':event['check'],'status':event['status'],'reason':event['reason'],'sim_time':state['sim_time'],'metrics':event['metrics']}
        if event['check'] in ('tcp_tracking','local_alignment_control') and geometry:
            item.update(truth_tracking(state,geometry,event['metrics']))
        elif event['check']=='get_object_pose' and geometry:
            api=api_at_feedback[i];response=api.get('response') if api else None
            if response and api['name']=='get_object_pose':
                label=api['args'][0] if api['args'] else api['kwargs'].get('object_name','')
                name='primary' if 'red' in label else 'secondary' if 'green' in label else None
                if name:
                    r=np.asarray(geometry['base_rotation_world']);b=np.asarray(geometry['base_position_world'])
                    truth=r.T@(np.asarray(state['objects'][name]['position'])-b)
                    error=np.asarray(response[0])-truth
                    item.update(position_error_m=error.tolist(),truth_bad=bool(np.linalg.norm(error[:2])>.025))
                    if response[2] is not None:
                        half=geometry['red_half_size' if name=='primary' else 'green_half_size']
                        q=state['objects'][name]['quaternion_wxyz']
                        rot=r.T@Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
                        ext=np.abs(rot)@(2*np.asarray(half));err=np.asarray(response[2])-ext
                        item['extent_error_m']=err.tolist()
                        item['truth_bad']=bool(item['truth_bad'] or np.max(np.abs(err))>.020)
        elif event['check']=='pre_release' and geometry:
            item.update(truth_release(state,geometry));item['attempt']=event['attempt']
        elif event['check'] in ('grasp','slip'):
            item['truth_bad']=not state['grasp_contact']
        elif event['check']=='lift':
            item['truth_bad']=not (state['grasp_contact'] and state['objects']['primary']['position'][2]>start+.05)
        elif event['check']=='place':
            # Instantaneous raw physical success; full stable reward is reported separately.
            item['truth_bad']=not state['raw_success']
        outcomes['checks'].append(item)
    recovery=[]
    for i,event in enumerate(ev):
        if event['check']!='recovery':continue
        reason=event['reason']
        wanted={'align':'pre_release','control_retry':'tcp_tracking','reperceive':event['metrics'].get('api')}.get(reason)
        next_index=next((j for j in range(i+1,len(ev)) if ev[j]['check']==wanted),None)
        attempt=dict(kind=reason,detector_cleared=next_index is not None and ev[next_index]['status']=='PASS')
        if reason=='align' and geometry and i in private and next_index in private:
            before=truth_release(private[i],geometry);after=truth_release(private[next_index],geometry)
            attempt.update(before=before,after=after,geometry_risk_cleared=before['truth_bad'] and not after['truth_bad'])
        recovery.append(attempt)
    outcomes['recovery_attempts']=recovery
    outcomes['immediately_successful_replans']=sum(b['block']>0 and b['reward']==1 for b in result['blocks'])
    request=result.get('initial_request')
    initial_receipt=Path(request)/'receipt.json' if request else None
    initial_seconds=json.loads(initial_receipt.read_text())['seconds'] if initial_receipt and initial_receipt.exists() else 0
    generated_here=result.get('new_llm_requests',0)>result['replans']
    outcomes['standalone_seconds_estimate']=result['total_seconds']+(0 if generated_here else initial_seconds)
    release=[a for a in apis if a['name']=='open_gripper' and a['before']['gripper_command']<.5]
    outcomes['skills']=skill_counts(apis,states,geometry) if geometry else {}
    outcomes['releases']=[dict(sim_time=a['before']['sim_time'],before=truth_release(a['before'],geometry) if geometry else None,
                              after_xy_m=float(np.linalg.norm(np.array(a['after']['objects']['primary']['position'])[:2]-np.array(a['after']['objects']['secondary']['position'])[:2]))) for a in release]
    final=result.get('final_physics');outcomes['final_xy_m']=float(np.linalg.norm(np.array(final['objects']['primary']['position'])[:2]-np.array(final['objects']['secondary']['position'])[:2])) if final else None
    return outcomes


def analyze(root,seeds=None):
    paths=sorted((root/'episodes').glob('*/result.json'))
    items=[analyze_episode(p) for p in paths]
    if seeds is not None:items=[i for i in items if i['seed'] in seeds]
    groups={}
    for group in 'ABCDE':
        subset=[i for i in items if i['group']==group]
        if not subset:continue
        checks=[c for i in subset for c in i['checks'] if 'truth_bad' in c]
        groups[group]=dict(n=len(subset),infrastructure_errors=sum(i['status']=='system_error' for i in subset),success=sum(i['reward']==1 for i in subset),
                           grasp_success=sum(bool(i['grasp_success']) for i in subset),
                           lift_success=sum(i['max_lift_m']>.05 for i in subset),
                           slipped=sum(bool(i['slipped']) for i in subset),
                           mean_execution_seconds=float(np.mean([i['execution_seconds'] for i in subset])),
                           api_calls=sum(i['api_calls'] for i in subset),http_requests=sum(i['http_requests'] for i in subset),
                           tool_http_requests=sum(i['tool_http_requests'] for i in subset),
                           mean_standalone_seconds_estimate=float(np.mean([i['standalone_seconds_estimate'] for i in subset])),
                           replans=sum(i['replans'] for i in subset),replan_episodes=sum(i['replans']>0 for i in subset),
                           successful_replan_episodes=sum(i['replan_effective'] for i in subset),
                           local_recoveries=sum(i['recoveries'] for i in subset),
                           recovery_attempts_by_kind=dict(Counter(a['kind'] for i in subset for a in i['recovery_attempts'])),
                           recovery_checks_cleared_by_kind=dict(Counter(a['kind'] for i in subset for a in i['recovery_attempts'] if a['detector_cleared'])),
                           immediately_successful_replans=sum(i['immediately_successful_replans'] for i in subset),
                           statuses=dict(Counter(i['status'] for i in subset)),
                           failure_reasons=dict(Counter(reason for i in subset for reason in set(i['failure_reasons']))),
                           skills=dict(sum((Counter(i['skills']) for i in subset),Counter())),
                           detection={name:confusion([c for c in checks if c['check']==name]) for name in sorted({c['check'] for c in checks})})
    receipts=[json.loads(p.read_text()) for p in sorted((root/'llm').glob('*/receipt.json'))]
    if seeds is not None:
        receipts=[r for r in receipts if any(r['label'].endswith(f'seed-{s}') or f'seed-{s}-' in r['label'] for s in seeds)]
    cost=0;input_tokens=output_tokens=reasoning_tokens=0
    for r in receipts:
        u=r.get('usage',{});input_tokens+=u.get('prompt_tokens',0);output_tokens+=u.get('completion_tokens',0)
        reasoning_tokens+=u.get('completion_tokens_details',{}).get('reasoning_tokens',0)
        hit=u.get('prompt_cache_hit_tokens',0);miss=u.get('prompt_cache_miss_tokens',u.get('prompt_tokens',0)-hit)
        cost+=(hit*.044+miss*1.32+u.get('completion_tokens',0)*3.96)/1e6
    return dict(groups=groups,episodes=items,llm_requests=len(receipts),tokens=dict(input=input_tokens,output=output_tokens,reasoning=reasoning_tokens),
                peak_cache_aware_cost_usd=cost,off_peak_cost_usd=cost/2,
                notes=['Exploratory paired initial programs and snapshots; no confidence claim from small n.',
                       'UNCERTAIN reported separately; not silently counted as correct detection.',
                       'Pre-release labels are geometric risk proxies: >10 mm XY, >20 mm vertical AABB gap, <-6 mm vertical AABB overlap, or no grasp contact. They do NOT establish whether a counterfactual release would succeed.',
                       'Place labels use instantaneous raw success, final success still uses unchanged P0 stability.',
                       'L0 pose label: GT XY centre error >25 mm or any extent error >20 mm. Partial surface consistency does not guarantee full geometry.',
                       'L1 label: physical TCP tracking >25 mm or >20 degrees, distinct from eventual task failure.',
                       'Lift outcome is height only; physical grasp_success additionally requires held contact duration.',
                       'Skill counts: close->bilateral contact; commanded upward move >50 mm->object rise >35 mm with contact; release->unchanged P0 stable criterion. Initial open is excluded.',
                       'Standalone latency adds shared initial-generation latency back to groups that reused it; wall time is not an isolated hardware benchmark.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+');a=p.parse_args()
    result=analyze(a.run,a.seeds);dump(a.output,result)
    print(json.dumps({'groups':result['groups'],'tokens':result['tokens'],'requests':result['llm_requests']},indent=2))
if __name__=='__main__':main()
