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
When recovery_scope.enforced is true, execute ONLY recovery_scope.allowed_intents.
Those IDs already include any necessary newly allocated repair. Do not allocate
another repair or try a failed predecessor. The parent refreshed and checked the
reported conditions before requesting recovery code.
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
        self.v27=limits.get('recovery_version',26)>=27
        self.graph_repair=self.v27 and group=='G1' and ablation!='no_graph_repair'
        self.condition_evidence=self.graph_repair or (limits.get('recovery_version',26)>=28 and group=='G1')
        self.costed_evidence=limits.get('recovery_version',26)>=28 and group=='G1' and ablation not in ('no_selective','no_cost')
        self.failed_frontiers={};self.work_scope=None;self.recovering=False
        self.actor_frontiers={};self.max_same_frontier=limits.get('max_same_frontier',2)
        self.v29=limits.get('recovery_version',26)>=29
        self.causal=None;self.local_attempts=0;self.local_limit=limits.get('max_repairs',6)
        self.deterministic_intent=None;self.local_frontiers={}
        from capx.execution.causal_plan import ground_library
        self.library=ground_library(definition) if self.v29 else []
        from capx.execution.contract_recovery import ConditionPlanner
        self.planner=ConditionPlanner(self.c)
        self.broker=ContractBroker(self.c,skills.capabilities() if hasattr(skills,'capabilities') else {'acquire':skills.acquire,'deposit':skills.deposit},skills.verify,
            max_calls=limits['skill_budget'],seconds=limits['episode_seconds'],emit=self.record,sync_invalidates=self.v27)
        self._sync_stages()
        self.recovery=ContractRecovery(self.c,self.broker,
            group='G0' if ablation=='no_selective' else group,
            max_repairs=limits.get('max_repairs',6),max_same_frontier=limits.get('max_same_frontier',2),emit=self.record)

    def compile_program(self,code,source='unattributed_test_program'):
        if not self.v29:return
        from capx.execution.graph_executive import GraphExecutive
        self.c.register_program(code,source)
        self.causal=GraphExecutive().compile_contract_program(self.c,code,automatic=self.group=='G1' and self.ablation!='no_causal')
        self.record(dict(kind='causal_compile',revision=self.causal.revision,order=self.causal.order,edges=self.causal.edges))

    def deterministic_code(self):
        if not self.v29 or self.ablation=='no_local_repair' or not self.deterministic_intent:return None
        return f"execute_skill({self.deterministic_intent!r})"

    def _prepare_local(self):
        from capx.execution.causal_plan import signature
        from capx.execution.graph_executive import GraphExecutive
        c=self.c
        if self.ablation=='no_local_repair' or self.causal is None:return None
        if self.recovery.repairs>=self.recovery.max_repairs:
            self.record(dict(kind='local_repair_exit',reason='local_budget'));return None
        if self.group=='G0':
            keys=self.recovery.scope(self.last_intent or next(iter(c.intents)))
            keys=[k for k in keys if self.planner.exhausted.get(k)!=c.truth(k)]
            if keys:self._planned_refresh(keys)
        failed=[k for edge in self.causal.live_support() for k in [edge['condition']] if k in c.conditions and c.truth(k)!=edge['expected']]
        forbidden=[signature(row) for i,row in c.intents.items() if self.failed_frontiers.get(self.frontier(i),0)]
        for _ in range(self.limit+1):
            decision=GraphExecutive().repair_causal_plan(self.causal,self.library,failed=failed,
                exhausted=self.planner.exhausted,forbidden=forbidden,check_continuation=self.ablation!='no_resume_validation')
            self.record(dict(kind='causal_repair_decision',decision=decision,live_support=self.causal.live_support()))
            if decision['status']=='evidence':
                keys=decision['keys']
                if self.costed_evidence:
                    scoped=dict(work='evidence',keys=keys,goals=c.definition['goals'])
                    costed=self.planner.costed_evidence(scoped,self.skills.evidence_features)
                    # Do not replace a local repair obligation with an unrelated
                    # skill frontier; only free evidence sharing is applicable.
                    if costed['work']=='evidence':keys=costed['keys']
                if not keys:return None
                self._planned_refresh(keys);continue
            if decision['status']!='ready':return None
            if not decision['fragment'] and not decision['retained']:
                self.terminal_reason='goals_verified' if c.complete() else 'missing_final_evidence';return False
            if self.recovery.repairs+len(decision['fragment'])>self.recovery.max_repairs:
                self.record(dict(kind='local_repair_exit',reason='shared_repair_budget'));return None
            self.causal=self.causal.commit(decision);self.local_attempts+=1
            self.recovery.repairs+=len(decision['fragment'])
            selected=self.causal.order[0]
            self.work_scope=dict(work='skill',allowed_intents=[selected],enforced=True,policy='causal_local_repair',
                affected=decision['affected'],retained=decision['retained'],replaced=decision['replaced'])
            self.deterministic_intent=selected
            self.record(dict(kind='deterministic_continuation',intent=selected,source='validated_causal_fragment_or_retained_suffix',
                             reused_intent=selected in decision['retained']))
            return True
        return None

    def _sync_stages(self):
        if hasattr(self.skills,'stage_conditions'):
            for key,value in self.skills.stage_conditions().items():
                if key not in self.c.conditions or self.c.truth(key)!=value['truth']:
                    self.c.observe(key,value['truth'],value['evidence'])

    def record(self,row):
        self._sync_stages()
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
        if self.v27:
            state['recovery_scope']=copy.deepcopy(self.work_scope)
            open_recovery=getattr(self,'agent_recovery',False)
            if open_recovery and state['recovery_scope'] is not None:
                state['recovery_scope'].update(enforced=False,role='graph_recommendation_only',
                    authority='current parent contracts, not this historical candidate list')
            state['graph_repair_enforced']=self.graph_repair and not open_recovery
            state['terminal_reason']=self.terminal_reason
        return state

    def frontier(self,intent):
        row=self.c.intents[intent]
        return (row.get('repair_of',intent),self.skills.phase,
            tuple((k,self.c.truth(k)) for k in sorted(set(row.get('requires',[])+row['effects']))),
            self.public_motion_frontier() if getattr(self,'harness_active',False) else 0)

    def public_motion_frontier(self):
        from capx.execution.persistent_harness import public_motion_signature
        return public_motion_signature(self.skills)

    def prepare_recovery(self):
        '''Parent selects legal work before paying for another actor. No private score.'''
        if not self.v27:return True
        self.recovering=True;c=self.c;self.deterministic_intent=None
        self.planner.motion_epoch(getattr(self.skills,'motion_version',0))
        if c.quarantined:self.terminal_reason='physical_effect_quarantined';return False
        remaining=[g for g in c.definition['goals'] if c.truth(g)!='TRUE']
        if self.v29:
            local=self._prepare_local()
            if local is not None:return local
        # Common capabilities and progress cap; only scope/condition regression differ.
        if not self.condition_evidence:
            intent=self.last_intent or next(iter(c.intents))
            scope=self.recovery.scope(intent)
            scope=[k for k in scope if self.planner.exhausted.get(k)!=c.truth(k)]
            if scope:self._planned_refresh(scope)
        for _ in range(self.limit+1):
            from capx.execution.graph_executive import GraphExecutive
            decision=GraphExecutive().repair_contract(c,self.planner)
            if self.costed_evidence:
                decision=self.planner.costed_evidence(decision,self.skills.evidence_features)
            if decision['work']=='evidence':
                if not self.condition_evidence:break
                keys=decision['keys']
                if self.ablation=='no_selective':
                    keys=self.recovery.scope(self.last_intent or next(iter(c.intents)))
                    keys=[k for k in keys if self.planner.exhausted.get(k)!=c.truth(k)]
                if not keys:break
                self.record(dict(kind='repair_decision',decision=decision,actual_scope=keys))
                self._planned_refresh(keys);continue
            break
        evidence_allowed=[]
        if self.graph_repair:
            allowed=decision.get('allowed_intents',[])
        else:
            # Strong conservative baseline can select every currently legal unfinished
            # branch. Completed sibling goals are kept; no graph-selected work lock.
            allowed=[]
            def needed(row,seen=()):
                if row['id'] in seen:return False
                goal_effects=[k for k in row['effects'] if k in c.definition['goals']]
                if goal_effects and all(c.truth(k)=='TRUE' for k in goal_effects):return False
                if any(k in remaining for k in row['effects']):return True
                return any(any(k in r.get('requires',[]) for k in row['effects']) and needed(r,seen+(row['id'],)) for r in c.definition['intents'])
            for row in c.definition['intents']:
                if needed(row) and c.order_satisfied(row) and all(c.truth(k)=='TRUE' for k in row.get('requires',[])) and not all(c.truth(k)=='TRUE' for k in row['effects']):
                    allowed.append(row['id'])
                if needed(row) and any(c.truth(k)=='UNKNOWN' and
                    c.definition.get('condition_specs',{}).get(k,{}).get('observable') and
                    self.planner.exhausted.get(k)!='UNKNOWN' for k in row.get('requires',[])):
                    evidence_allowed.append(row['id'])
        allowed=[i for i in allowed if self.failed_frontiers.get(self.frontier(i),0)==0]
        new=[]
        for i in allowed:
            if i in c.completed or i in c.stopped:
                try:i=self.recovery.repair(i)
                except ValueError:continue
            new.append(i)
        signature=(self.skills.phase,tuple((k,c.truth(k)) for k in sorted(c.conditions)),
            tuple(sorted(c.intents[i].get('repair_of',i) for i in new)),tuple(sorted(evidence_allowed)))
        attempts=self.actor_frontiers.get(signature,0)+1;self.actor_frontiers[signature]=attempts
        if (new or evidence_allowed) and attempts>self.max_same_frontier:
            self.terminal_reason='recovery_frontier_unchanged'
            self.record(dict(kind='bounded_exit',reason=self.terminal_reason,frontier=signature))
            return False
        self.work_scope=dict(decision,allowed_intents=new,evidence_allowed_intents=evidence_allowed,enforced=self.graph_repair,
            policy='condition_regression' if self.graph_repair else 'conservative_or_model_selection')
        self.record(dict(kind='repair_decision',decision=self.work_scope))
        if not new and not evidence_allowed:
            self.terminal_reason='goals_verified' if c.complete() else 'no_legal_recovery_capability'
            self.record(dict(kind='bounded_exit',reason=self.terminal_reason,decision=self.work_scope))
            return False
        if self.v29 and self.ablation!='no_local_repair' and new:
            self.deterministic_intent=new[0]
            self.record(dict(kind='deterministic_continuation',intent=new[0],source='existing_condition_frontier',reused_intent=new[0] in self.c.intents))
        return True

    def _planned_refresh(self,keys):
        if self.refreshes>=self.limit:
            self.planner.exhaust(keys);return {}
        self.refreshes+=1;before={k:self.c.truth(k) for k in keys};nodes=dict(self.c.conditions)
        started=time.monotonic();rows=self.broker.refresh(keys)
        self.planner.exhaust([k for k in keys if self.c.truth(k)==before[k]])
        self.record(dict(kind='planned_evidence',scope=keys,before=before,observed=rows,
            preserved={k:n for k,n in nodes.items() if k not in keys and self.c.truth(k)=='TRUE'},seconds=time.monotonic()-started))
        return rows

    def refresh(self,intent):
        from capx.execution.sandbox import ExecutionInterrupted
        if self.refreshes>=self.limit:
            self.terminal_reason='evidence_budget';raise ExecutionInterrupted('Evidence refresh budget exhausted')
        self.refreshes+=1
        before={k:self.c.truth(k) for k in self.c.conditions}
        result=self.recovery.refresh_for(intent)
        if self.v27:
            self.planner.motion_epoch(getattr(self.skills,'motion_version',0))
            self.planner.exhaust([k for k in result['scope'] if self.c.truth(k)==before.get(k,'UNKNOWN')])
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
            if self.v29 and self.causal is not None:
                from capx.execution.causal_plan import required
                for edge in self.causal.edges:
                    if edge['target']==intent and edge['kind'] in ('causal','data') and edge['source'] not in set(self.c.completed)|set(self.c.stopped)|{'START'}:
                        wanted=required(self.c.intents[intent]).get(edge['condition'],'TRUE')
                        if self.c.truth(edge['condition'])!=wanted:raise ValueError('Unexecuted causal predecessor lacks verified support')
            if self.v27:
                if self.failed_frontiers.get(self.frontier(intent),0):
                    raise ValueError('unchanged failed action frontier; no repeated capture or dispatch')
                if self.graph_repair and self.recovering and not getattr(self,'agent_recovery',False):
                    if self.work_scope is None:self.prepare_recovery()
                    if intent not in (self.work_scope or {}).get('allowed_intents',[]):
                        raise ValueError('Action outside parent graph recovery scope')
            if not self.v27 and not self.skills.anchors:
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
            self.work_scope=None
            return result
        except (SkillBoundaryStop,ValueError) as exc:
            if self.v27 and intent in self.c.intents:
                key=self.frontier(intent);self.failed_frontiers[key]=self.failed_frontiers.get(key,0)+1
                self.record(dict(kind='failed_frontier',intent=intent,signature=key,count=self.failed_frontiers[key]))
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


def shared_boundary_spec(variant,common):
    """Keep the strict full Snapshot gate; execution policy is separate metadata."""
    policy={'recovery_version','perception_optimization','skill_recovery_version','force_feedback','force_sensor','harness_version','harness_max_requests'}
    if {k:v for k,v in variant.items() if k not in policy}!={k:v for k,v in common.items() if k not in policy}:
        raise ValueError('Comparison changed non-policy environment or budget fields')
    return copy.deepcopy(common)


def run_episode(dest,spec,seed,group,budget,config,initial,expected=None,*,study='normal',ablation='none',environment_spec=None,external_method=None,recorded_recoveries=None):
    from capx.experiments.native_tasks import NativeBoundary,physical_state,physical_reward
    from capx.experiments.native_vision_skills import build_skills,region_task
    from capx.experiments.layered_feedback import validated_program,instrument_tools
    from capx.experiments.graph_v2 import receipt_summary,digest
    from capx.execution.sandbox import SandboxProcess,ExecutionInterrupted
    from capx.utils.request_audit import RequestAudit
    before_llm=getattr(budget,'count',0)
    recorded=list(recorded_recoveries or [])
    if recorded and (spec.get('harness_version')!=33 or any(row.get('origin') not in ('archived_model_program','controlled_public_probe_NOT_LLM') for row in recorded)):
        raise ValueError('Recorded recovery requires persistent Harness and explicit provenance')
    boundary=None;harness=None;start=time.monotonic();rows=[];requests=[];result={};runtime=None;injected=False;generated_here=initial is None;stop_reason='block_budget'
    try:
        with RequestAudit(dest.parent/(dest.name+'-http.jsonl')) as audit:
            boundary=NativeBoundary(shared_boundary_spec(spec,environment_spec or spec),seed,dest,expected_snapshot=expected,sensor_config=spec.get('force_sensor'))
            save(dest/'runtime-spec.json',spec)
            instrument_tools(boundary,dest)
            skills=build_skills(boundary,spec);definition=region_task(skills.scene.bindings,grounded=spec.get('recovery_version',26)>=27,causal=(environment_spec or spec).get('recovery_version',26)>=29)
            # Same initial public intent/prompt across old/new Skills. Recovery
            # capabilities are parent-owned and added after initial generation.
            initial_definition=copy.deepcopy(definition)
            if spec.get('skill_recovery_version',29)>=30:
                from capx.execution.skill_recovery import recovery_library
                recovery_library(definition)
            runtime=ModelContractRuntime(skills,definition,group=group,limits=spec,emit=boundary.log,ablation=ablation)
            if hasattr(skills,'force_graph'):skills.force_graph=runtime.c
            base=prompt(initial_definition);save(dest/'api-docs.json',dict(text=API_DOC,definition=definition))
            if initial is None:
                answer,request=budget.query(base,f'initial-{seed}')
                code=validated_program(answer)
                initial=dict(code=code,request=request,code_hash=digest(code),prompt_hash=digest(base))
                save(dest.parent/'initial.json',initial)
            elif initial['prompt_hash']!=digest(base):raise ValueError('Shared initial prompt drift')
            requests.append(initial['request']);code=initial['code'];code_origin='real_deepseek'
            if external_method is None:runtime.compile_program(code,initial['request'])
            else:runtime.c.register_program(code,initial['request'])
            functions=dict(execute_skill=runtime.execute,execution_state=runtime.state,
                           refresh_evidence=runtime.refresh,repair_skill=runtime.recovery.repair)
            def dispatch(intent):
                nonlocal injected
                value=runtime.execute(intent)
                # Evaluator-owned evidence-expiry stress. No fault label/hidden
                # state is passed to policy, and the physics is never mutated.
                if study in ('evidence_expiry','effect_displacement') and not injected:
                    completed_goals=[k for k in definition['goals'] if runtime.c.truth(k)=='TRUE']
                    if completed_goals:
                        key=completed_goals[0]
                        if study=='effect_displacement':
                            from capx.experiments.native_faults import perturb_completed_effect
                            perturb_completed_effect(boundary,key.split(':',1)[1],config['physical_disturbance'])
                        runtime.c.graph.invalidate([runtime.c.conditions[key]],reason='evidence_expired')
                        injected=True;boundary.log('controlled_event',event='evidence_expired',condition=key)
                        runtime.last_error='A previously established goal no longer has valid evidence.'
                        raise ExecutionInterrupted(runtime.last_error)
                return value
            functions['execute_skill']=dispatch
            if spec.get('harness_version')==33:
                from capx.execution.persistent_harness import PersistentHarness
                harness=PersistentHarness(boundary,runtime,budget,request_limit=spec.get('harness_max_requests',2))
                functions.update(harness.tools.functions())
            boundary.functions=functions
            if external_method:
                if study!='normal':raise ValueError('External adapter natural scenarios only; no private fault labels')
                from capx.experiments.external_logiv import execute_official
                rows,stop_reason=execute_official(runtime,boundary,code,config,external_method)
            for block in range(0 if external_method else config['budgets']['max_blocks']):
                if time.monotonic()-boundary.started>=spec['episode_seconds']:break
                boundary.actor.close();boundary.actor=SandboxProcess(functions,callback=boundary.call)
                runtime.c.register_program(code,requests[-1] if code_origin=='real_deepseek' else code_origin)
                save(dest/f'actor-{block:02d}-program.json',dict(code=code,request=requests[-1] if code_origin=='real_deepseek' else None,origin=code_origin))
                if harness:harness.program(block,code,code_origin,requests[-1] if code_origin=='real_deepseek' else None)
                actor_start=time.monotonic()
                calls_before=runtime.broker.calls
                actor=boundary.actor.run(code,timeout=min(config['budgets']['block_seconds'],spec['episode_seconds']-(time.monotonic()-boundary.started)))
                actor['seconds']=time.monotonic()-actor_start
                rows.append(actor);save(dest/f'actor-{block:02d}-result.json',actor)
                save(dest/f'actor-{block:02d}-public.json',runtime.state())
                if not harness and runtime.terminal_reason:stop_reason=runtime.terminal_reason;break
                if not harness and (runtime.c.quarantined or actor['status'] in ('timeout','system_error')):stop_reason='unknown_effect_or_timeout';break
                if runtime.c.complete():
                    runtime.broker.refresh(definition['goals'])
                    if runtime.c.complete():stop_reason='public_goals_verified';break
                if not harness and block>0 and actor['ok'] and runtime.broker.calls==calls_before and not ready_continuation(runtime.c):
                    stop_reason='model_stopped_without_admissible_continuation';break
                if block+1==config['budgets']['max_blocks']:break
                if harness:
                    decision,continuation=harness.next(actor,recorded_programs=len(recorded))
                    if decision['level']=='R4':stop_reason=decision['reason'];break
                    if decision['level']=='R1':
                        code=continuation;code_origin='deterministic_verified_plan';continue
                    if recorded:
                        entry=recorded.pop(0);code=entry['code'];code_origin=entry['origin']
                        harness.store.append('RecoveryProposed',dict(level='R2',source=code_origin,reference=entry.get('reference'),new_llm_request=False))
                        continue
                    answer,path=harness.controller.generate(base,harness.context(actor),f'{dest.name}-seed-{seed}-recovery-{block+1}')
                    requests.append(path);code_origin='real_deepseek'
                    try:code=validated_program(answer)
                    except ValueError as exc:
                        # Invalid responses never execute motion; a subsequent
                        # bounded actor reports a program error to the controller.
                        harness.store.append('RecoveryRejected',dict(level='R2',reason=str(exc),request=path))
                        code='raise ValueError('+repr('Rejected model response: '+str(exc))+')'
                    continue
                if runtime.v27 and not runtime.prepare_recovery():stop_reason=runtime.terminal_reason;break
                continuation=runtime.deterministic_code()
                if continuation:
                    code=continuation;code_origin='deterministic_verified_plan';continue
                code_origin='real_deepseek'
                answer,path=budget.query(recovery_prompt(base,code,actor,runtime.state()),f'{dest.name}-seed-{seed}-recovery-{block+1}')
                requests.append(path);code=validated_program(answer)
            state=physical_state(boundary.low.robosuite_env)
            result=dict(raw_success=state['raw_success'],reward=physical_reward(runtime.c.complete() and not runtime.c.quarantined,boundary.motion,state['raw_success']),
                native_progress=state['native_progress'],graph_complete=runtime.c.complete(),quarantined=runtime.c.quarantined,
                api_calls=boundary.calls,captures=getattr(boundary,'captures',0),skill_calls=runtime.broker.calls,
                completed_calls=list(runtime.c.completed),last_error=runtime.last_error,phase=skills.phase,
                evidence_refreshes=runtime.refreshes,repairs=runtime.recovery.repairs,actual_http=audit.count,
                snapshot_hash=boundary.initial['state_hash'],model_hash=boundary.initial['model_hash'],fault_triggered=injected)
            result['force_feedback']=bool(spec.get('force_feedback',False))
            result['force_decisions']=len(getattr(skills,'force_decisions',[]))
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc),reward=None)
        if boundary:
            state=physical_state(boundary.low.robosuite_env)
            result.update(raw_success=state['raw_success'],native_progress=state['native_progress'],api_calls=boundary.calls,captures=getattr(boundary,'captures',0))
    finally:
        if runtime:save(dest/'contract-graph.json',dict(nodes=runtime.c.graph.nodes,edges=runtime.c.graph.edges,events=runtime.c.graph.events,policy_events=runtime.events))
        snap=boundary.initial if boundary else expected
        if harness:
            result['stop_reason']=stop_reason
            harness.finish(result)
            result['harness']=dict(version=33,events=harness.store.sequence,recovery_requests=harness.supervisor.requests,
                no_progress=harness.supervisor.no_progress,unresolved_calls=harness.store.unresolved())
        if boundary:boundary.close()
        result.update(schema='capx.contract_models.v'+str(max(min(29,spec.get('recovery_version',26)),spec.get('skill_recovery_version',0))),group=group,seed=seed,study=study,ablation=ablation,
            mode=('historical_initial_live_vision_real_recovery' if harness and getattr(budget,'count',0)>before_llm else 'historical_program_live_vision') if initial and initial.get('historical_replay') else 'real_deepseek_live_vision',new_llm_requests=getattr(budget,'count',0)-before_llm,method='LOGIV-official-core-common-skills-adapter' if external_method else 'capx',initial_code_hash=initial['code_hash'] if initial else None,
            seconds=time.monotonic()-start,blocks=len(rows),request_paths=requests,llm_requests=len(requests),
            recovery_llm_requests=max(0,len(requests)-1),actors=rows,stop_reason=stop_reason)
        generation=0.
        if initial:
            generation=json.loads((Path(initial['request'])/'receipt.json').read_text()).get('seconds',0.)
        result.update(initial_generation_seconds=generation,
                      execution_seconds=result['seconds']-(generation if generated_here else 0.),
                      end_to_end_seconds=result['seconds']+(0. if generated_here else generation))
        if recorded_recoveries:result['mode']='recorded_recovery_live_vision_NOT_new_model_experiment'
        if requests:result.update(receipt_summary(requests))
        save(dest/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('group','seed','study','ablation','raw_success','native_progress','api_calls','captures','llm_requests','seconds','error','last_error')},ensure_ascii=False,default=json_value),flush=True)
    return result,initial,snap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--study',choices=['normal','evidence_expiry','effect_displacement'],default='normal')
    p.add_argument('--ablations',nargs='*',choices=['no_selective','no_future','no_memory','no_graph_repair','no_cost','no_causal','no_local_repair','no_resume_validation'],default=[])
    p.add_argument('--perception-controls',action='store_true',help='Add paired V27 G0/G1 without common perception reuse or cost planning')
    p.add_argument('--external-logiv',type=Path,help='Revision-pinned official LOGIV checkout for adapted external comparison')
    p.add_argument('--v29-skills-control',action='store_true',help='Same initial program and graph policy with original V29 Skills')
    p.add_argument('--v28-control',action='store_true',help='Paired V28 execution policy on shared current contracts and skills')
    p.add_argument('--ik-runtime',type=Path,help='Existing start_feedback_ik runtime JSON; override only the IK service URL')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.ik_runtime:cfg['services']['pyroki']=json.loads(a.ik_runtime.read_text())['url']
    if a.external_logiv:
        import subprocess
        cfg['external_logiv']=dict(path=str(a.external_logiv.resolve()),revision=subprocess.check_output(['git','-C',str(a.external_logiv),'rev-parse','HEAD'],text=True).strip())
    from capx.experiments.graph_v2 import configure_services
    configure_services(cfg)
    from capx.experiments.graph_campaign import PersistentModelBudget,archive_inventory,verify_source_hashes
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/'config.json',cfg)
    sources=set(cfg['sources'])|{str(a.config)}
    if a.ik_runtime:sources.add(str(a.ik_runtime))
    hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in sources}
    save(a.output/'source-archive.json',archive_inventory(a.output,hashes))
    budget=PersistentModelBudget(a.output,cfg['budgets']);initial=snap=None;results=[]
    groups=[('G0','G0','none'),('G1','G1','none')]+[('G1-'+x,'G1',x) for x in a.ablations]
    if a.external_logiv:groups=[('LOGIV-adapted','G0','none')]+groups
    if a.v29_skills_control:groups += [('V29-Skills-G1','G1','none')]
    if a.v28_control:groups += [('V28-G1','G1','none')]
    if a.perception_controls:groups += [('V27-G0','G0','none'),('V27-G1','G1','none')]
    for name,group,ablation in groups:
        if verify_source_hashes(hashes):raise RuntimeError('Source drift; stop at complete episode boundary')
        spec=copy.deepcopy(cfg['tasks']['pick_place'])
        if name=='V29-Skills-G1':spec.update(skill_recovery_version=29)
        if name=='V28-G1':spec.update(recovery_version=28)
        if name.startswith('V27-'):spec.update(recovery_version=27,perception_optimization=False)
        result,initial,snap=run_episode(a.output/name,spec,a.seed,group,budget,cfg,initial,snap,study=a.study,ablation=ablation,environment_spec=cfg['tasks']['pick_place'],external_method=cfg.get('external_logiv') if name=='LOGIV-adapted' else None)
        results.append(result)
        if initial is None:break
    save(a.output/'summary.json',dict(episodes=results,unique_requests=budget.count,estimated_cost_usd=budget.ledger.spent['deepseek']))
    save(a.output/'source-postcheck.json',dict(changed=verify_source_hashes(hashes)))


if __name__=='__main__':main()
