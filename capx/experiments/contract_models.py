"""Small real-DeepSeek paired experiments on existing native Vision Skills.

No reference program fallback. Safety, capability and score are shared; graph
policy changes evidence scope / scheduling hints, never a physical threshold.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from capx.experiments.graph_v2 import save,json_value

API_DOC = '''Only these parent APIs are available:
execute_skill(intent_id): execute an immutable declared skill intent. Raises and
interrupts the whole program at any unsafe/uncertain boundary. Calls which
completed must NEVER be replayed. Each primitive still checks live geometry.
execution_state(): return public condition evidence, completed/stopped calls,
remaining goals, attachment phase and unexecuted intents. No private reward.
refresh_evidence(intent_id): obtain current public evidence for an intent and
any relevant retained effects. Does not move the robot or claim completion.
repair_skill(intent_id): allocate a NEW id for a stopped/returned but unsatisfied
intent, with the same immutable contract; bounded by progress and retry limits.
Returns that id; call execute_skill on it only when its preconditions hold.
Existing unexecuted intents retain their IDs. A confirmed held object can proceed
to its declared deposit even when acquisition stopped after closing. Never try
another acquire while an attachment remains unknown. No raw motion, file,
network, simulator, private attributes or success flags are available.
Write a complete Python program in exactly one fenced python block; builtins and
simple loops/dicts are allowed, no import needed. Do not use the reserved names
env, obs, globals, getattr. Print a reason and stop if no safe work is possible.
The EXACT execution_state schema is:
{conditions: {condition_key: {truth: 'TRUE'|'FALSE'|'UNKNOWN', node: str,
evidence: dict}}, completed_calls: list[str], stopped_calls: list[str],
remaining_goals: list[str], phase: str, attachment: str|null, quarantined: bool,
last_intent: str|null, last_error: str|null, intents: list[contract], repair_count: int}.
An optional graph_proposal is only a suggestion, never permission.
Compare state['conditions'].get(key, {}).get('truth', 'UNKNOWN') == 'TRUE';
NEVER bool(condition_dict). Initial conditions may be empty: execute_skill
refreshes its live preconditions itself; or use refresh_evidence(intent_id).
refresh_evidence returns {scope:list[str], observed:{key:{truth,evidence}},
preserved:{key:node_id}, completed_calls:list[str], policy:str, intent:str}.
Allowed object methods are ONLY: get, items, keys, values, copy, append, extend,
tolist, reshape, astype. In particular NO lower, add, sort, startswith, index,
replace or arbitrary attributes. Use list membership and builtins sorted/range
instead. Avoid broad exception-catching; a safety stop aborts the whole block.
Keep the program short (at most 80 lines), direct and bounded; do not build an
interface-guessing framework or parse several hypothetical schemas.
Aim to complete ALL goals, preserving completed work and trying independent
feasible work after a local failure. FINISH is not physical success.'''


class ModelContractRuntime:
    """Parent API facade: policy has public callbacks and contracts only."""
    def __init__(self,skills,definition,*,group,limits,emit,ablation='none'):
        from capx.execution.skill_contracts import ContractGraph,ContractBroker
        from capx.execution.contract_recovery import ContractRecovery
        self.skills=skills;self.c=ContractGraph(definition);self.events=[];self.emit=emit
        self.ablation=ablation;self.group=group;self.last_intent=None;self.last_error=None
        self.refreshes=0;self.limit=limits.get('max_refreshes',12);self.reuse=[]
        self.evidence_frontiers={};self.terminal_reason=None
        self.broker=ContractBroker(self.c,{'acquire':skills.acquire,'deposit':skills.deposit},skills.verify,
            max_calls=limits['skill_budget'],seconds=limits['episode_seconds'],emit=self.record)
        self.recovery=ContractRecovery(self.c,self.broker,
            group='G0' if ablation=='no_selective' else group,
            max_repairs=limits.get('max_repairs',6),max_same_frontier=limits.get('max_same_frontier',2),emit=self.record)

    def record(self,row):
        self.events.append(copy.deepcopy(row));payload=copy.deepcopy(row)
        payload['event']=payload.pop('kind','skill_result');self.emit('model_contract',**payload)

    def state(self):
        c=self.c;conditions={k:dict(truth=c.truth(k),node=n,evidence=c.graph.nodes[n]['data']['evidence']) for k,n in c.conditions.items()}
        state=dict(conditions=conditions,completed_calls=list(c.completed),stopped_calls=list(c.stopped),
            remaining_goals=[k for k in c.definition['goals'] if c.truth(k)!='TRUE'],
            phase=self.skills.phase,attachment=self.skills.held['object'] if self.skills.held else None,
            quarantined=c.quarantined,last_intent=self.last_intent,last_error=self.last_error,
            intents=list(c.intents.values()),repair_count=self.recovery.repairs)
        if self.group=='G1' and self.ablation!='no_future':
            from capx.execution.graph_executive import GraphExecutive
            # Do not allocate plans just by reading status; an explicit repair API
            # owns new IDs. Exclude stopped physical calls from normal proposals.
            state['graph_proposal']=GraphExecutive().next_contract(c,excluded=list(c.stopped)+['__status_only__'])
        if self.group=='G1' and self.last_intent in c.intents:
            row=c.intents[self.last_intent]
            roots=[c.conditions[k] for k in row.get('requires',[])+row['effects'] if k in c.conditions]
            if roots:state['diagnosis']=c.graph.diagnose(roots,max_nodes=16)
        return state

    def refresh(self,intent):
        from capx.execution.sandbox import ExecutionInterrupted
        if self.refreshes>=self.limit:
            self.terminal_reason='evidence_budget';raise ExecutionInterrupted('Evidence refresh budget exhausted')
        self.refreshes+=1
        result=self.recovery.refresh_for(intent)
        if self.ablation=='no_memory':
            keys=[k for k in self.c.definition['goals'] if k in result['preserved']]
            if keys:
                if self.refreshes>=self.limit:
                    self.terminal_reason='evidence_budget';raise ExecutionInterrupted('Evidence refresh budget exhausted')
                self.refreshes+=1;self.broker.refresh(keys)
                self.record(dict(kind='memory_ablation_refresh',scope=keys))
            result['preserved']={k:v for k,v in result['preserved'].items() if k not in keys}
        signature=tuple((k,self.c.truth(k)) for k in sorted(self.c.conditions))
        count=self.evidence_frontiers.get(signature,0)+1;self.evidence_frontiers[signature]=count
        if count>2:
            self.terminal_reason='equivalent_evidence_no_progress'
            self.record(dict(kind='bounded_exit',reason=self.terminal_reason))
            raise ExecutionInterrupted('Equivalent observations have not changed executable conditions')
        self.reuse.append(result);return result

    def execute(self,intent):
        from capx.execution.sandbox import ExecutionInterrupted
        from capx.execution.skill_contracts import SkillBoundaryStop
        self.last_intent=intent;self.last_error=None
        try:
            if not self.skills.anchors:
                try:self.skills.prepare()
                except ValueError as exc:raise SkillBoundaryStop(str(exc),{}) from exc
            before=dict(self.c.conditions)
            result=self.broker.execute(intent)
            # Audit actual retained-goal use: the remaining-work decision after a
            # call consumes these conditions, not just a node-count statistic.
            retained={k:n for k,n in before.items() if k in self.c.definition['goals'] and
                      n==self.c.conditions.get(k) and self.c.truth(k)=='TRUE'}
            self.record(dict(kind='continuation',intent=intent,retained_goals_used=retained,
                             remaining=[k for k in self.c.definition['goals'] if self.c.truth(k)!='TRUE']))
            return result
        except (SkillBoundaryStop,ValueError) as exc:
            self.last_error=str(exc);self.record(dict(kind='boundary_interrupt',intent=intent,reason=str(exc)))
            raise ExecutionInterrupted(str(exc)) from exc


def prompt(definition):
    return [dict(role='system',content='You are the CaP-X robot coding agent. Use only deployment-observable tools and safe contracts.'),
            dict(role='user',content=API_DOC+'\nPublic task definition:\n'+json.dumps(definition)+
                 '\nFor this INITIAL program choose an order of the declared skills covering all goals, '
                 'and write direct execute_skill calls. Do not implement a scheduler, state machine, '
                 'generic interface adapter or retry loop. The parent handles interruption and will '
                 'ask you for new code with the actual state if needed. Keep this initial program under 25 lines.')]


def recovery_prompt(base,code,actor,state):
    return base+[dict(role='assistant',content='```python\n'+code+'\n```'),
        dict(role='user',content='Execution stopped; physical state persists and Python variables do not.\n'+
             json.dumps(dict(execution=actor,public_state=state))+
             '\nGenerate self-contained recovery/continuation code using the same APIs. Refresh only needed evidence via the parent API. '
             'Use new repair IDs only for previously attempted unsatisfied skills; do not replay completed work. '
             'If an intent is blocked while no object is held, consider another independent goal. '
             'Do not repeat a failed identical attempt without new information. Keep recovery under 35 lines; '
             'at most one refresh for each selected intent. If no safe continuation is supported, print the reason and stop.')]


def ready_continuation(c):
    """Common bounded-exit check, not a policy for selecting or executing work."""
    return not c.quarantined and any(i not in c.completed and i not in c.stopped and
        c.order_satisfied(row) and not all(c.truth(k)=='TRUE' for k in row['effects']) and
        all(c.truth(k)=='TRUE' for k in row.get('requires',[])) for i,row in c.intents.items())


def run_episode(dest,spec,seed,group,budget,config,initial,expected=None,*,study='normal',ablation='none'):
    from capx.experiments.native_tasks import NativeBoundary,physical_state,physical_reward
    from capx.experiments.native_vision_skills import build_skills,region_task
    from capx.experiments.layered_feedback import validated_program,instrument_tools
    from capx.experiments.graph_v2 import receipt_summary,digest
    from capx.execution.sandbox import SandboxProcess,ExecutionInterrupted
    from capx.utils.request_audit import RequestAudit
    boundary=None;start=time.monotonic();rows=[];requests=[];result={};runtime=None;injected=False;generated_here=initial is None;stop_reason='block_budget'
    try:
        with RequestAudit(dest.parent/(dest.name+'-http.jsonl')) as audit:
            boundary=NativeBoundary(spec,seed,dest,expected_snapshot=expected)
            instrument_tools(boundary,dest)
            skills=build_skills(boundary,spec);definition=region_task(skills.scene.bindings)
            runtime=ModelContractRuntime(skills,definition,group=group,limits=spec,emit=boundary.log,ablation=ablation)
            base=prompt(definition);save(dest/'api-docs.json',dict(text=API_DOC,definition=definition))
            if initial is None:
                answer,request=budget.query(base,f'initial-{seed}')
                code=validated_program(answer)
                initial=dict(code=code,request=request,code_hash=digest(code),prompt_hash=digest(base))
                save(dest.parent/'initial.json',initial)
            elif initial['prompt_hash']!=digest(base):raise ValueError('Shared initial prompt drift')
            requests.append(initial['request']);code=initial['code']
            functions=dict(execute_skill=runtime.execute,execution_state=runtime.state,
                           refresh_evidence=runtime.refresh,repair_skill=runtime.recovery.repair)
            def dispatch(intent):
                nonlocal injected
                value=runtime.execute(intent)
                # Evaluator-owned evidence-expiry stress. No fault label/hidden
                # state is passed to policy, and the physics is never mutated.
                if study=='evidence_expiry' and not injected:
                    completed_goals=[k for k in definition['goals'] if runtime.c.truth(k)=='TRUE']
                    if completed_goals:
                        key=completed_goals[0];runtime.c.graph.invalidate([runtime.c.conditions[key]],reason='evidence_expired')
                        injected=True;boundary.log('controlled_event',event='evidence_expired',condition=key)
                        runtime.last_error='A previously established goal no longer has valid evidence.'
                        raise ExecutionInterrupted(runtime.last_error)
                return value
            functions['execute_skill']=dispatch;boundary.functions=functions
            for block in range(config['budgets']['max_blocks']):
                if time.monotonic()-boundary.started>=spec['episode_seconds']:break
                boundary.actor.close();boundary.actor=SandboxProcess(functions,callback=boundary.call)
                runtime.c.register_program(code,requests[-1])
                save(dest/f'actor-{block:02d}-program.json',dict(code=code,request=requests[-1],origin='real_deepseek'))
                actor_start=time.monotonic()
                calls_before=runtime.broker.calls
                actor=boundary.actor.run(code,timeout=min(config['budgets']['block_seconds'],spec['episode_seconds']-(time.monotonic()-boundary.started)))
                actor['seconds']=time.monotonic()-actor_start
                rows.append(actor);save(dest/f'actor-{block:02d}-result.json',actor)
                save(dest/f'actor-{block:02d}-public.json',runtime.state())
                if runtime.terminal_reason:stop_reason=runtime.terminal_reason;break
                if runtime.c.quarantined or actor['status'] in ('timeout','system_error'):stop_reason='unknown_effect_or_timeout';break
                if runtime.c.complete():
                    runtime.broker.refresh(definition['goals'])
                    if runtime.c.complete():stop_reason='public_goals_verified';break
                if block>0 and actor['ok'] and runtime.broker.calls==calls_before and not ready_continuation(runtime.c):
                    stop_reason='model_stopped_without_admissible_continuation';break
                if block+1==config['budgets']['max_blocks']:break
                answer,path=budget.query(recovery_prompt(base,code,actor,runtime.state()),f'{dest.name}-seed-{seed}-recovery-{block+1}')
                requests.append(path);code=validated_program(answer)
            state=physical_state(boundary.low.robosuite_env)
            result=dict(raw_success=state['raw_success'],reward=physical_reward(runtime.c.complete() and not runtime.c.quarantined,boundary.motion,state['raw_success']),
                native_progress=state['native_progress'],graph_complete=runtime.c.complete(),quarantined=runtime.c.quarantined,
                api_calls=boundary.calls,captures=getattr(boundary,'captures',0),skill_calls=runtime.broker.calls,
                completed_calls=list(runtime.c.completed),last_error=runtime.last_error,phase=skills.phase,
                evidence_refreshes=runtime.refreshes,repairs=runtime.recovery.repairs,actual_http=audit.count,
                snapshot_hash=boundary.initial['state_hash'],model_hash=boundary.initial['model_hash'],fault_triggered=injected)
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc),reward=None)
        if boundary:
            state=physical_state(boundary.low.robosuite_env)
            result.update(raw_success=state['raw_success'],native_progress=state['native_progress'],api_calls=boundary.calls,captures=getattr(boundary,'captures',0))
    finally:
        if runtime:save(dest/'contract-graph.json',dict(nodes=runtime.c.graph.nodes,edges=runtime.c.graph.edges,events=runtime.c.graph.events,policy_events=runtime.events))
        snap=boundary.initial if boundary else expected
        if boundary:boundary.close()
        result.update(schema='capx.contract_models.v26',group=group,seed=seed,study=study,ablation=ablation,
            mode='real_deepseek_live_vision',initial_code_hash=initial['code_hash'] if initial else None,
            seconds=time.monotonic()-start,blocks=len(rows),request_paths=requests,llm_requests=len(requests),
            recovery_llm_requests=max(0,len(requests)-1),actors=rows,stop_reason=stop_reason)
        generation=0.
        if initial:
            generation=json.loads((Path(initial['request'])/'receipt.json').read_text()).get('seconds',0.)
        result.update(initial_generation_seconds=generation,
                      execution_seconds=result['seconds']-(generation if generated_here else 0.),
                      end_to_end_seconds=result['seconds']+(0. if generated_here else generation))
        if requests:result.update(receipt_summary(requests))
        save(dest/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('group','seed','study','ablation','raw_success','native_progress','api_calls','captures','llm_requests','seconds','error','last_error')},ensure_ascii=False,default=json_value),flush=True)
    return result,initial,snap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--study',choices=['normal','evidence_expiry'],default='normal')
    p.add_argument('--ablations',nargs='*',choices=['no_selective','no_future','no_memory'],default=[])
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    from capx.experiments.graph_v2 import configure_services
    configure_services(cfg)
    from capx.experiments.graph_campaign import PersistentModelBudget,archive_inventory,verify_source_hashes
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/'config.json',cfg)
    sources=set(cfg['sources'])|{str(a.config)}
    hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in sources}
    save(a.output/'source-archive.json',archive_inventory(a.output,hashes))
    budget=PersistentModelBudget(a.output,cfg['budgets']);initial=snap=None;results=[]
    for name,group,ablation in [('G0','G0','none'),('G1','G1','none')]+[('G1-'+x,'G1',x) for x in a.ablations]:
        if verify_source_hashes(hashes):raise RuntimeError('Source drift; stop at complete episode boundary')
        result,initial,snap=run_episode(a.output/name,cfg['tasks']['pick_place'],a.seed,group,budget,cfg,initial,snap,study=a.study,ablation=ablation)
        results.append(result)
        if initial is None:break
    save(a.output/'summary.json',dict(episodes=results,unique_requests=budget.count,estimated_cost_usd=budget.ledger.spent['deepseek']))
    save(a.output/'source-postcheck.json',dict(changed=verify_source_hashes(hashes)))


if __name__=='__main__':main()
