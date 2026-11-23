"""Gate 2 ONLY: physical reference capability driven by the generic executive.

This module intentionally has privileged evaluator access. Its results may prove
the continuous benchmark and contract plumbing, NEVER deployable Vision or LLM
success. No corresponding functions are registered in the Vision boundary.
"""
import time
from capx.execution.skill_contracts import ContractGraph,ContractBroker
from capx.execution.graph_executive import GraphExecutive
from capx.execution.dependency_graph import DependencyGraph
from .native_tasks import save,physical_state


def run_reference_graph(boundary,spec,program,group):
    if boundary.mode!='oracle':raise ValueError('Reference graph requires explicit privileged oracle mode')
    c=ContractGraph(spec['contract_task'],DependencyGraph(mode=group));decisions=[];checks=[]
    objects=spec['objects'];start=time.monotonic()
    def verify(keys):
        state=physical_state(boundary.low.robosuite_env)
        progress=state['native_progress'].get('objects_in_bins')
        if progress is None:raise ValueError('Reference region verifier not available')
        rows={}
        for key in keys:
            if key.startswith('goal:'):
                truth=bool(progress[objects.index(key.split(':',1)[1])])
            elif key=='task_public_verified':truth=state['raw_success']
            elif key.startswith(('visible:','geometry:')):
                # Reference geometry availability, explicitly PRIVATE, never a
                # claim that a current camera has observed this instance.
                obj=key.split(':',1)[1];truth=obj in state['objects']
            elif key.startswith('target_geometry:'):
                index=int(key.split(':',1)[1]);truth=0<=index<len(progress)
            elif key=='robot_ready':
                import numpy as np
                truth=bool(np.isfinite(boundary.public.get_observation()['robot_joint_pos']).all())
            else:truth=False
            rows[key]=dict(truth='TRUE' if truth else 'FALSE',evidence=dict(source='PRIVATE_ORACLE_ONLY',sim_time=state['sim_time']))
        checks.append(dict(keys=list(keys),sim_time=state['sim_time']));return rows
    def transfer(row):
        obj=row['object'];index=objects.index(obj)
        boundary.functions['oracle_bindings']=lambda:[(index,obj)]
        result=boundary.actor.run(program,timeout=spec['episode_seconds'])
        boundary.log('reference_skill',intent=row['id'],result=result)
        if not result['ok']:raise RuntimeError(result['stderr'])
        for _ in range(15):boundary.low._step_once()
    broker=ContractBroker(c,{'transfer_to_region':transfer,'verify_effects':lambda row:None},verify,
        max_calls=len(c.intents),seconds=spec['episode_seconds'],emit=lambda row:boundary.log('contract_skill',**row))
    error=None
    for _ in range(2*len(c.intents)+2):
        decision=GraphExecutive().next_contract(c);decisions.append(decision)
        try:
            if decision['work']=='evidence':
                scope=c.recovery_scope(decision['missing'],group)
                broker.refresh(scope);continue
            if decision['work']=='skill':broker.execute(decision['intent']);continue
            break
        except Exception as exc:error=type(exc).__name__+': '+str(exc);break
    state=physical_state(boundary.low.robosuite_env)
    result=dict(schema='capx.native_reference_graph.v24',mode='privileged_oracle_NOT_Vision_NOT_DeepSeek',group=group,
        passed=bool(state['raw_success'] and c.complete() and error is None),graph_complete=c.complete(),raw_success=state['raw_success'],error=error,
        native_progress=state['native_progress'],actual_api=boundary.calls,skill_calls=broker.calls,
        completed_intents=list(c.completed),decisions=decisions,condition_verifications=checks,
        seconds=time.monotonic()-start,new_llm_requests=0,complexity=c.complexity())
    save(boundary.root/'reference-graph.json',dict(nodes=c.graph.nodes,edges=c.graph.edges,events=c.graph.events))
    save(boundary.root/'reference-result.json',result)
    return result
