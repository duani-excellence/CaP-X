"""V25 capability gate on the existing boundary / contracts / Sandbox.

The monitor below sees public callbacks only; the final evaluator is deliberately
outside it. This deterministic reference run is NEVER labelled DeepSeek output.
"""
import json
import time
from pathlib import Path
import numpy as np
from .native_tasks import save,physical_state,physical_reward
from .native_vision import PublicScene
from capx.execution.vision_skills import RegionSkills
from capx.execution.skill_contracts import ContractGraph,ContractBroker
from capx.execution.graph_executive import GraphExecutive


def region_task(bindings,grounded=False):
    rows=[];effects=[]
    for i,obj in enumerate(bindings):
        held='held:'+obj;goal='placed:'+obj;effects.append(goal)
        rows.extend([
            dict(id=f'acquire-{i}',skill='acquire',object=obj,requires=['robot_ready'],effects=[held],invalidates=['robot_ready'],resources=['arm0'],after=[]),
            dict(id=f'deposit-{i}',skill='deposit',object=obj,target=str(i),requires=[held],effects=[goal],invalidates=[held],resources=['arm0'],after=[f'acquire-{i}'],allow_verified_predecessors=True)])
    result=dict(schema='capx.contract_task.v1',objects=list(bindings),resources=['arm0'],goals=effects,intents=rows)
    if grounded:
        specs={k:dict(observable=True) for k in ['robot_ready','workspace_geometry']+effects}
        for i,obj in enumerate(bindings):
            acquire,deposit=rows[2*i:2*i+2]
            acquire['requires'] += ['workspace_geometry','geometry:'+obj,'grasp_ready:'+obj]
            acquire['invalidates'] += ['geometry:'+obj,'grasp_ready:'+obj]
            deposit['requires'] += ['target_geometry:'+str(i)]
            deposit['effects'] += ['robot_ready']
            for k in acquire['requires']+deposit['requires']:specs[k]=dict(observable=True)
        result['condition_specs']=specs
    return result


def build_skills(boundary,spec):
    """Shared live sensor/control construction for reference and model runners."""
    from capx.execution.public_camera import robosuite_pixel_centers
    from capx.experiments.vision_tasks import public_depth_sampling
    from capx.execution.control_frames import transform_target
    from capx.execution.bounded_ik import IKConvergenceError
    from scipy.spatial.transform import Rotation
    profile_path=Path(spec['vision_skill']['error_profile']);profile=json.loads(profile_path.read_text())
    save(boundary.root/'error-profile.json',profile)
    def capture():
        obs,cal=robosuite_pixel_centers(boundary.capture(),depth_sampling=public_depth_sampling(boundary),require_depth_sampling=True)
        index=boundary.captures-1;save(boundary.root/'observations'/f'{index:05d}-calibration.json',cal)
        return obs,str(boundary.root/'observations'/f'{index:05d}-calibration.json')
    scene=PublicScene(capture,boundary.api.sam3_seg_fn,spec['vision_skill']['bindings'],emit=boundary.log,
        optimize=spec.get('perception_optimization',False))
    def reachable(p,q):
        pos,quat=transform_target(boundary.calibration,p,q)
        hand=np.asarray(pos)+Rotation.from_quat(np.roll(quat,-1)).apply(boundary.api._TCP_OFFSET)
        try:
            boundary.api.ik_solve_fn(target_pose_wxyz_xyz=np.r_[quat,hand],prev_cfg=boundary.api.cfg)
            return True
        except IKConvergenceError:return False
    skills=RegionSkills(scene,boundary.trusted_motion,boundary.api.grasp_net_plan_fn,reachable,profile,seed=boundary.seed,emit=boundary.log)
    return skills


def run_skills(boundary,spec,group='G1'):
    skills=build_skills(boundary,spec);scene=skills.scene
    definition=region_task(scene.bindings);c=ContractGraph(definition);decisions=[];exclusions=[]
    broker=ContractBroker(c,{'acquire':skills.acquire,'deposit':skills.deposit},skills.verify,
        max_calls=spec.get('skill_budget',24),seconds=spec['episode_seconds'],emit=lambda row:boundary.log('contract_skill',**row))
    # The actor only selects immutable parent-declared intent IDs. Raw motion
    # remains absent from its function table, including in the reference gate.
    boundary.functions['execute_skill']=broker.execute
    boundary.actor.close()
    from capx.execution.sandbox import SandboxProcess
    boundary.actor=SandboxProcess(boundary.functions,callback=boundary.call)
    error=None;actors=[];start=time.monotonic()
    try:
        skills.prepare()
        for _ in range(spec.get('skill_budget',24)*2):
            decision=GraphExecutive().next_contract(c,excluded=exclusions);decisions.append(decision)
            boundary.log('contract_decision',group=group,decision=decision)
            if decision['work']=='evidence':
                scope=c.recovery_scope(decision['missing'],group);broker.refresh(scope);continue
            if decision['work']=='verify_task':
                broker.refresh(c.definition['goals'])
                if not c.complete():boundary.log('task_evidence_revoked',reason='current_public_goal_verification_failed')
                break
            if decision['work']!='skill':break
            code=f"execute_skill({decision['intent']!r})"
            actor=boundary.actor.run(code,timeout=spec['episode_seconds']);actors.append(actor)
            if not actor['ok']:
                error=actor;exclusions.append(decision['intent'])
                if c.quarantined or skills.held is not None:break
        state=physical_state(boundary.low.robosuite_env)
        result=dict(schema='capx.native_vision_skill_gate.v25',group=group,mode='real_vision_reference_NOT_LLM',
            passed=bool(state['raw_success'] and c.complete() and not c.quarantined),raw_success=state['raw_success'],
            reward=physical_reward(c.complete(),boundary.motion,state['raw_success']),native_progress=state['native_progress'],
            graph_complete=c.complete(),completed_intents=list(c.completed),effects=skills.effects,
            error=error,actual_api=boundary.calls,captures=boundary.captures,skill_calls=broker.calls,
            seconds=time.monotonic()-start,new_llm_requests=0,complexity=c.complexity(),phase=skills.phase)
    except Exception as exc:
        state=physical_state(boundary.low.robosuite_env)
        result=dict(passed=False,error=type(exc).__name__+': '+str(exc),raw_success=state['raw_success'],
            reward=0.,native_progress=state['native_progress'],new_llm_requests=0,phase=skills.phase,
            actual_api=boundary.calls,captures=getattr(boundary,'captures',0))
    finally:
        save(boundary.root/'contract-graph.json',dict(nodes=c.graph.nodes,edges=c.graph.edges,events=c.graph.events,decisions=decisions))
        save(boundary.root/'actors.json',actors)
    save(boundary.root/'vision-skill-result.json',result);return result
