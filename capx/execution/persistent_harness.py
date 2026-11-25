"""V33 lifecycle services used by the existing Contract Runner.

No physics reset, alternate execution loop or unguarded robot API lives here.
The supervisor owns limits/completion; the controller owns program versions;
the monitor owns public events; the orchestrator only selects recovery levels.
"""
import copy
import json
import time
from .agent_events import EventStore,plain,digest
from .open_recovery import OpenRecoveryAPIs,API_DOC


def public_motion_signature(skills):
    obs=getattr(getattr(skills,'scene',None),'obs',{})
    # Measured motion only; incrementing a counter or changing source IDs cannot
    # resurrect a failed physical frontier. Millimetre / encoder resolution.
    return tuple((k,tuple(round(float(x),3) for x in obs[k])) for k in
        ('robot_cartesian_pos','robot_joint_pos','robot_gripper_joint_pos') if k in obs)


STOP_FAILURES={'UNKNOWN_PHYSICAL_EFFECT','HARD_SAFETY_VIOLATION','ENVIRONMENT_FAILURE'}


class ProgressTracker:
    """Semantic best-so-far progress. Motion, new IDs and sensor noise are not gains."""
    def __init__(self,search_limit=4):
        self.best={};self.seen_states=set();self.search=set();self.search_limit=search_limit;self.no_progress=0
        self.previous={};self.reestablished={};self.transition_limit=2

    def update(self,state,information,*,robot=None,search=()):
        key=digest([sorted(state.get('remaining_goals',[])),
            sorted((k,v.get('truth')) for k,v in state.get('conditions',{}).items())])
        # Only new verified TRUE facts count as positive physical/task progress.
        true=set(k for k,v in state.get('conditions',{}).items() if v.get('truth')=='TRUE')
        prior=self.best.setdefault('verified',set());task=bool(true-prior);prior.update(true)
        # Recovery may restore an old fact or establish a negative precondition.
        # Source IDs do not matter. A bounded semantic transition allowance is
        # separate from the ever-verified statistics, so chatter cannot renew it.
        wanted={k:'TRUE' for k in state.get('remaining_goals',[])}
        for row in state.get('intents',[]):
            wanted.update({k:'TRUE' for k in row.get('requires',[])+row.get('data_requires',[])})
            wanted.update({k:'FALSE' for k in row.get('requires_false',[])})
        wanted.update({k:'TRUE' for k in prior})
        current={k:v.get('truth','UNKNOWN') for k,v in state.get('conditions',{}).items()}
        restored=[]
        for k,expected in wanted.items():
            if current.get(k)==expected and self.previous.get(k)!=expected:
                transition=(k,expected)
                if self.reestablished.get(transition,0)<self.transition_limit:
                    self.reestablished[transition]=self.reestablished.get(transition,0)+1
                    restored.append(k)
        task |= bool(restored)
        self.previous=current
        first=not self.seen_states;self.seen_states.add(key)
        information_progress=False
        for name,row in information.items():
            if not isinstance(row,dict):continue
            for metric in ('coverage','wall_coverage','observable_fraction','constrained_axes'):
                value=row.get(metric)
                if not isinstance(value,(float,int)) or not 0<=value<=3:continue
                # Quantize by one percentage point to avoid noise/source churn.
                q=int(value*100);k=(name,metric);old=self.best.get(k,-1)
                if q>old:
                    information_progress |= old>=0 and q-old>=1
                    self.best[k]=q
            for metric in ('residual_95_m','error_bound_m','position_error_m','goal_error_m','innovation_m'):
                value=row.get(metric)
                if not isinstance(value,(int,float)) or value<0:continue
                q=round(value,4);k=(name,metric);old=self.best.get(k)
                if old is None or q<old:
                    information_progress |= old is not None and old-q>=.0001
                    self.best[k]=q
        search_progress=False
        for row in search:
            if not row.get('outcome'):continue
            from .semantic_capabilities import semantic_case
            k=digest(semantic_case({x:row.get(x) for x in ('strategy','parameters','outcome','candidate')}))
            if k not in self.search and len(self.search)<self.search_limit:
                self.search.add(k);search_progress=True
        changed=first or task or information_progress or search_progress
        self.no_progress=0 if changed else self.no_progress+1
        return dict(changed=changed,task_or_verified_physical_progress=task,information_progress=information_progress,
            search_progress=search_progress,no_progress=self.no_progress,signature=key,robot_motion_alone_is_progress=False)

    def best_summary(self):
        return dict(verified_condition_count=len(self.best.get('verified',())),distinct_supported_searches=len(self.search),
            no_progress=self.no_progress,metrics={str(k):v for k,v in self.best.items() if k!='verified'})


def classify_failure(actor,unknown=False):
    if unknown:return 'UNKNOWN_PHYSICAL_EFFECT'
    status=actor.get('status','');msg=str(actor.get('error',''))+' '+str(actor.get('stderr',''))
    if actor.get('failure_class') in STOP_FAILURES:return actor['failure_class']
    if status=='system_error':return 'ENVIRONMENT_FAILURE'
    if status in ('invalid_program','policy_violation') or any(x in msg for x in ('SyntaxError','NameError','TypeError','AttributeError','KeyError','IndexError','ZeroDivisionError','UnboundLocalError')):return 'PROGRAM_ERROR'
    if any(x in msg.lower() for x in ('connection','http','service unavailable')):return 'TRANSIENT_TOOL_ERROR'
    if any(x in msg.lower() for x in ('unknown','geometry','evidence','visible','boundary','uncertain')):return 'PERCEPTION_UNCERTAIN'
    if any(x in msg.lower() for x in ('dependency','invalidated','predecessor')):return 'DEPENDENCY_INVALIDATION'
    if status=='ok':return 'PERCEPTION_UNCERTAIN'  # Program return is not task completion.
    return 'PHYSICAL_EXECUTION_FAILURE'


class RecoveryOrchestrator:
    def __init__(self,limit=2):self.limit=limit
    def decide(self,failure,deterministic,*,requests_left,no_progress):
        if failure in STOP_FAILURES:return dict(level='R4',reason=failure.lower())
        if no_progress>=self.limit:return dict(level='R4',reason='no_information_progress')
        if deterministic:return dict(level='R1',reason='verified_existing_continuation')
        if requests_left<=0:return dict(level='R4',reason='recovery_request_budget')
        return dict(level='R2',reason='deterministic_capability_exhausted')


class TaskSupervisor:
    def __init__(self,boundary,runtime,budget,request_limit):
        self.b=boundary;self.r=runtime;self.budget=budget;self.request_limit=request_limit
        self.requests=0;self.no_progress=0;self.last_progress=None
        self.tracker=ProgressTracker();self.open_runtime=bool(getattr(runtime,'open_runtime',False) or (boundary and boundary.spec.get('harness_version',0)>=34))

    def remaining(self):
        return dict(api=max(0,self.b.spec.get('api_budget',256)-self.b.calls),
            skill=max(0,self.r.broker.max_calls-self.r.broker.calls),
            evidence=max(0,getattr(self.r,'limit',0)-getattr(self.r,'refreshes',0)),
            repairs=max(0,getattr(getattr(self.r,'recovery',None),'max_repairs',0)-getattr(getattr(self.r,'recovery',None),'repairs',0)),
            seconds=max(0,self.b.spec['episode_seconds']-(time.monotonic()-self.b.started)),
            recovery_requests=max(0,min(self.request_limit-self.requests,
                getattr(self.budget,'max_requests',0)-getattr(self.budget,'count',0))))

    def progress(self):
        # New source IDs, camera noise, clock ticks and a new program are NOT
        # progress. A changed executable condition or verified physical move is.
        state=self.r.state()
        if self.open_runtime:
            tools=getattr(self.r,'open_tools',None)
            result=self.tracker.update(state,getattr(tools,'information',{}),search=getattr(tools,'search_results',()))
            self.no_progress=result['no_progress'];return result
        key=digest([state['phase'],state['attachment'],state['remaining_goals'],
            {k:v['truth'] for k,v in state['conditions'].items()},public_motion_signature(self.r.skills)])
        same=key==self.last_progress
        self.no_progress=self.no_progress+1 if same else 0;self.last_progress=key
        return dict(changed=not same,no_progress=self.no_progress,signature=key)


class ExecutionMonitor:
    PHYSICAL={'goto_pose','follow_trajectory','open_gripper','close_gripper','wait','home_pose'}
    READONLY={'capture_observation','inspect_region','graph_query','execution_state','verify_conditions','refresh_evidence'}

    def __init__(self,store,observation,boundary):
        self.store=store;self.observation=observation;self.boundary=boundary;self.version=0;self.counter=0;self.last_failure=None
        self.effect_lifecycle=getattr(boundary,'execution_version',getattr(boundary,'spec',{}).get('harness_version',0))>=34
        self.pending_motion=[];self.graph=None
        self.contracts=None;self.call_stack=[]

    def observe_returned(self,data):
        if not self.effect_lifecycle:return
        for key,row in list(self.store.actions.items()):
            if row['state']!='RETURNED' or row['name'] in ('goto_pose','follow_trajectory'):continue
            # A fresh encoder sample observes the effect, but close-command
            # return does NOT establish grasping; a Skill verifier must do that.
            result=dict(row['result'],effect_observation=data,verification_pending=True)
            self.store.finish_action(key,'OBSERVED',result)
            node=result.get('graph_node')
            if node and self.graph:self.graph.nodes[node]['data'].update(lifecycle='observed',effect_evidence=data, effect_verified=False)

    def verify_motion(self,p,q,observation,passed):
        if not self.effect_lifecycle:return
        if not self.pending_motion:raise ValueError('No returned motion to verify')
        key=self.pending_motion[-1];row=self.store.actions[key]
        graph_node=row['result'].get('graph_node')
        if observation.get('source')==row['result']['before'].get('source'):raise ValueError('Effect verification requires fresh observation')
        data=dict(target=list(p),quaternion=list(q),evidence_source=observation.get('source'),observation=observation,verified=bool(passed))
        self.store.finish_action(key,'OBSERVED',data)
        self.store.finish_action(key,'DONE' if passed else 'FAILED',data);self.pending_motion.pop()
        if self.graph:
            node=self.graph.add('verification_fact',dict(action_id=key,**data),parents=[(graph_node,'attestation')] if graph_node else [])
            if graph_node:
                self.graph.nodes[graph_node]['data'].update(effect_verified=bool(passed),verification_node=node)
                if passed:self.graph.nodes[graph_node]['state']='done';self.graph._event('done',node_id=graph_node)
                else:self.graph.invalidate([graph_node],reason='fresh TCP convergence failed')
            self.store.append('VerificationResult',dict(action_id=key,graph_node=node,**data))

    def invoke(self,name,args,kwargs,function):
        from .skill_contracts import SkillBoundaryStop
        self.counter+=1;key=f'p{self.version}:call{self.counter}'
        physical=name in self.PHYSICAL
        site=copy.deepcopy(getattr(self.boundary.actor,'call_site',None))
        self.store.start_action(key,name,dict(args=args,kwargs=kwargs),physical=physical,site=site)
        before=self.observation()
        receipt=getattr(self.boundary,'dispatch_receipt',None)
        dispatched_before=receipt() if receipt else None
        prior_actions=set(self.store.actions)
        node=None
        if self.effect_lifecycle and self.graph:
            def ids(value):
                if isinstance(value,str):return [value] if value in self.graph.nodes else []
                if isinstance(value,dict):return [x for v in value.values() for x in ids(v)]
                if isinstance(value,(tuple,list)):return [x for v in value for x in ids(v)]
                return []
            parents=list(dict.fromkeys(ids(args)+ids(kwargs)))
            if self.call_stack:parents.append((self.call_stack[-1],'attestation'))
            if self.contracts and self.contracts.program_node:parents.append((self.contracts.program_node,'attestation'))
            node=self.graph.add('api_call',dict(name=name,action_id=key,inputs=plain(dict(args=args,kwargs=kwargs)),site=site,
                interface_returned=False,effect_verified=False if physical else None,lifecycle='running'),parents=parents)
            self.call_stack.append(node)
        try:
            try:result=function(*args,**kwargs)
            except (ConnectionError,TimeoutError) as exc:
                # Tool-read retries only, never a Skill or motion invocation.
                if name not in self.READONLY:raise
                self.store.append('RecoveryExecuted',dict(level='R0',action_id=key,retry=1,reason=type(exc).__name__))
                result=function(*args,**kwargs)
        except BaseException as exc:
            # Known synchronous guarded boundary stops are reconciled partial
            # effects. An arbitrary error after low-level dispatch is UNKNOWN.
            dispatch_state='LEGACY_UNKNOWN'
            dispatched_after=receipt() if receipt else None
            if receipt:
                dispatched=dispatched_after['started']>dispatched_before['started']
                dispatch_state='DISPATCHED_UNRECONCILED' if dispatched else 'NOT_DISPATCHED'
                reconcile=getattr(self.boundary,'reconcile_dispatch',None)
                children=[v for k,v in self.store.actions.items() if k not in prior_actions and v.get('physical')]
                covered=sum((v.get('result',{}).get('dispatch_after') or {}).get('started',0)-
                            (v.get('result',{}).get('dispatch_before') or {}).get('started',0) for v in children)
                nested_known=(not physical and bool(children) and not self.store.unresolved() and
                              covered==dispatched_after['started']-dispatched_before['started'])
                if dispatched and (isinstance(exc,SkillBoundaryStop) or nested_known) and reconcile and reconcile():
                    dispatch_state='DISPATCHED_RECONCILED'
                state='UNKNOWN' if physical and dispatch_state=='DISPATCHED_UNRECONCILED' else 'FAILED'
            else:state='UNKNOWN' if physical and not isinstance(exc,SkillBoundaryStop) else 'FAILED'
            data=dict(dispatch_state=dispatch_state,dispatch_before=dispatched_before,dispatch_after=dispatched_after,error=type(exc).__name__+': '+str(exc),before=before,after=self.observation(),site=site)
            self.store.finish_action(key,state,data)
            self.last_failure=dict(action_id=key,name=name,state=state,**data)
            self.store.append('FailureDetected',self.last_failure)
            if node:
                self.graph.nodes[node]['data'].update(error=str(exc),lifecycle=state)
                self.graph.invalidate([node],reason='API failed: '+str(exc))
                self.call_stack.pop()
            raise
        if node:
            self.graph.nodes[node]['data'].update(interface_returned=True,result=plain(result))
            self.graph.nodes[node]['data']['lifecycle']='returned'
            if not physical:self.graph.nodes[node]['state']='done';self.graph._event('done',node_id=node)
            self.call_stack.pop()
        self.store.finish_action(key,'RETURNED' if self.effect_lifecycle and physical else 'DONE',dict(dispatch_before=dispatched_before,dispatch_after=receipt() if receipt else None,value=plain(result),before=before,after=self.observation(),graph_node=node))
        if self.effect_lifecycle and name in ('goto_pose','follow_trajectory'):self.pending_motion.append(key)
        return result


class AgentController:
    def __init__(self,store,budget,supervisor):self.store=store;self.budget=budget;self.supervisor=supervisor
    def generate(self,base,context,label):
        self.supervisor.requests+=1
        if self.supervisor.open_runtime:
            from .open_recovery import runtime_api_doc
            OPEN_API_DOC=runtime_api_doc(preplanning=hasattr(self.supervisor.r.skills,'preplanner'),semantic=hasattr(self.supervisor.r.open_tools,'capabilities'))
            return self.budget.query(base+[dict(role='user',content=OPEN_API_DOC+'\nCurrent state persists, Python variables do not. Generate a new self-contained program with calculations, branches and bounded loops. Select a different meaningful strategy when prior work gained no information. Parent safety still applies.\n'+json.dumps(context,ensure_ascii=False))],label)
        messages=base+[dict(role='user',content='RECOVERY API EDITION 33 supersedes the INITIAL tool-list and work-list restrictions. The candidate list is advice, not an action permit. An empty list does not forbid the documented open recovery APIs. Every action is still checked by the parent.\n'+API_DOC+'\nRecovery context (current physics persists; no Python variables persist):\n'+json.dumps(context,ensure_ascii=False)+
            '\nWrite a self-contained Python recovery program. Query tools if needed. Use fresh operation/repair IDs; never replay completed calls. '
            'Preserve completed goals. Only documented APIs exist. Do not silently repeat a no-information attempt. '
            'Use at most one deliberate observation-motion strategy in this program; verify its result, then continue eligible goals. '
            'If no permitted strategy exists, print the specific missing capability. Return a python fenced block.')]
        return self.budget.query(messages,label)


class PersistentHarness:
    def __init__(self,boundary,runtime,budget,*,request_limit=2,policy=None):
        policy=boundary.spec if policy is None else policy
        self.b=boundary;self.r=runtime;self.store=EventStore(boundary.root/'harness')
        boundary.execution_version=34 if getattr(runtime,'open_runtime',False) else boundary.spec.get('harness_version',33)
        self.tools=OpenRecoveryAPIs(runtime,self.store,open_runtime=boundary.execution_version>=34)
        runtime.open_tools=self.tools
        self.supervisor=TaskSupervisor(boundary,runtime,budget,request_limit)
        self.monitor=ExecutionMonitor(self.store,self.tools.observation,boundary)
        self.controller=AgentController(self.store,budget,self.supervisor)
        if policy.get('semantic_version',0)>=36:
            from .semantic_capabilities import CapabilityRegistry
            functions=self.tools.functions()
            functions['execution_state']=runtime.state
            if hasattr(runtime,'execute'):functions['execute_skill']=runtime.execute
            motion={'run_plan','move_tcp','execute_skill'}
            effects={name:[] if name not in motion else ['parent_guarded_possible_physical_effect'] for name in functions}
            def evidence_valid(ref):
                return ref in runtime.c.graph.nodes and runtime.c.graph.nodes[ref]['valid']
            self.tools.capabilities=CapabilityRegistry(
                policy.get('semantic_store',boundary.root/'semantic-capabilities'),functions,
                graph=runtime.c.graph,tool_effects=effects,episode=str(boundary.root.resolve()),
                callback=boundary.call,evidence_valid=evidence_valid,
                condition_state=lambda:runtime.state()['conditions'],qualify=getattr(runtime,'qualify_capability',None),
                emit=lambda kind,**data:self.store.append(kind,data),
                seconds=lambda:self.supervisor.remaining()['seconds'])
        self.orchestrator=RecoveryOrchestrator(limit=4 if self.supervisor.open_runtime else 2);self.history=[];self.original=None;self.current=None;self.version=-1
        if self.supervisor.open_runtime:
            self.monitor.graph=runtime.c.graph;self.monitor.contracts=runtime.c
            self.tools.monitor=self.monitor
            runtime.skills.motion_receipt=getattr(boundary,'dispatch_receipt',None)
            runtime.skills.motion_invalidate=self.tools.motion_changed
            def motion_verified(p,q,passed):
                self.monitor.verify_motion(p,q,self.tools.observation(),passed)
            runtime.skills.effect_monitor=motion_verified
            from .belief_state import BeliefState
            self.tools.belief=BeliefState(runtime.c.graph,lambda event_name,**data:self.store.append(event_name,data))
            # Index existing public scene/skill estimates. No second perception
            # implementation, evaluator handle, or truth-to-permission bridge.
            for owner in (runtime.skills,runtime.skills.scene):
                prior=getattr(owner,'emit',lambda *a,**k:None)
                def record(kind,_prior=prior,**data):
                    _prior(kind,**data)
                    source=data.get('source',getattr(runtime.skills.scene,'source',None))
                    self.tools.belief.ingest(kind,data,source)
                    if kind=='region_geometry_diagnostic':self.tools.information['region:'+data['role']]=data['fit']
                    if kind in ('skill_hold_check','skill_hold_evidence'):self.tools.information['attachment:'+data['object']]=data['registration']
                    if kind=='skill_control_check':
                        self.tools.information['tcp_target:'+digest(data.get('target'))]=dict(position_error_m=data['position_error_m'])
                    if kind=='placement_quality_observed':self.tools.information['quality:'+data['object']]=data['quality']
                owner.emit=record
        boundary.action_observer=self.monitor.invoke
        def captured(data):
            self.store.append('ObservationCaptured',data)
            if self.supervisor.open_runtime:
                self.monitor.observe_returned(data)
                self.tools.belief.record('robot',dict(data['robot'],status='PASS',
                    occupied_resources=list(runtime.c.locks),free_space='UNKNOWN except explicitly checked motion samples'),source=data.get('files',{}))
        boundary.public_capture_sink=captured
        runtime.harness_active=True
        previous=runtime.emit
        def emit(kind,**data):
            previous(kind,**data)
            self.store.append('ConditionChanged',dict(channel=kind,**data))
        runtime.emit=emit
        self.store.append('TaskStarted',dict(goals=runtime.c.definition['goals'],group=runtime.group,
            snapshot_hash=boundary.initial['state_hash'],request_limit=request_limit,physics_resume=False))

    def program(self,version,code,origin,request):
        self.version=version;self.monitor.version=version;self.current=code
        if self.original is None:self.original=code
        self.store.program(version,code,origin,request)
        if version:self.store.append('RecoveryExecuted',dict(version=version,origin=origin,phase=self.r.skills.phase))

    def context(self,actor):
        state=self.r.state()
        if self.supervisor.open_runtime:
            # Compact decision packet. Programs, full graphs and image archives
            # remain queryable; they are not duplicated in every LLM request.
            failure=self.monitor.last_failure or {}
            if not str(failure.get('action_id','')).startswith(f'p{self.version}:'):failure={}
            import re
            lines=(self.current or '').splitlines()
            hits=re.findall(r'line (\d+)',str(actor.get('stderr','')))
            line=int(hits[-1]) if hits else (failure.get('site') or {}).get('line')
            excerpt=[dict(line=i+1,code=lines[i]) for i in range(max(0,(line or 1)-3),min(len(lines),(line or 1)+2))]
            program_error=dict(classification=classify_failure(actor),line=line,excerpt=excerpt,
                error=str(actor.get('error') or actor.get('stderr',''))[-2400:],
                completed_calls=list(self.r.c.completed),not_replayable=True)
            if getattr(self.r.skills,'decision_encoding',None)=='encoded':
                packet=self.tools.decision_state()
                packet.update(remaining_budget=self.supervisor.remaining(),
                    actor={k:str(actor.get(k,''))[:2500] for k in ('status','stderr','stdout')},
                    failed_api={k:failure.get(k) for k in ('action_id','name','state','error','site')},
                    program_version=self.version,program_failure=program_error,unknown_calls=self.store.unresolved(),
                    detail_tools='planning_query(id), belief_query(key), history_query(version)')
                return plain(packet)
            beliefs=self.tools.belief.query()
            relevant={k:v for k,v in beliefs.items() if k!='robot'}
            return plain(dict(task_goals=self.r.c.definition['goals'],remaining_goals=state['remaining_goals'],
                phase=state['phase'],attachment=state['attachment'],conditions={k:{'truth':v['truth']} for k,v in state['conditions'].items()},
                relevant_belief=relevant,robot=self.tools.observation()['robot'],
                actor={k:(str(actor.get(k,''))[:4000] if k in ('stderr','stdout') else actor.get(k)) for k in ('status','stderr','stdout','seconds')},
                failed_api={k:failure.get(k) for k in ('action_id','name','state','error','site')},
                completed_calls=list(self.r.c.completed),unknown_calls=self.store.unresolved(),program_version=self.version,program_failure=program_error,
                recent_decisions=self.history[-3:],recent_search=self.tools.search_results[-4:],
                remaining_budget=self.supervisor.remaining(),information_progress=self.supervisor.tracker.best_summary() if hasattr(self.supervisor.tracker,'best_summary') else {'no_progress':self.supervisor.no_progress},
                detailed_history='history_query(version), belief_query(key, history=True), graph_query',
                world_during_llm='MuJoCo paused between control calls; not a real-hardware safe-hold validation'))
        return plain(dict(task_goals=self.r.c.definition['goals'],public_state=state,
            original_program=self.original,current_program=self.current,program_version=self.version,
            actor=actor,failed_api=self.monitor.last_failure,current_observation=self.tools.observation(),
            completed_physical_calls=[dict(id=k,name=v['name'],inputs=v['inputs']) for k,v in self.store.actions.items() if v.get('physical') and v['state']=='DONE'],
            unknown_physical_calls=self.store.unresolved(),recovery_history=self.history[-6:],
            remaining_budget=self.supervisor.remaining(),progress=dict(no_progress=self.supervisor.no_progress),
            graph_reference='contract-graph.json',evidence_reference='observations/',motion_scope=self.tools.observation()['motion_scope']))

    def checkpoint(self,actor):
        return self.store.checkpoint(self.context(actor))

    def next(self,actor,*,recorded_programs=0):
        # Called after each actual actor, before requesting another program.
        failure=classify_failure(actor,
            bool(self.store.unresolved()) or self.r.c.quarantined)
        if self.r.terminal_reason in ('evidence_budget',):failure='PERCEPTION_UNCERTAIN'
        prepared=False
        if failure not in STOP_FAILURES and failure!='PROGRAM_ERROR':
            self.r.terminal_reason=None
            try:prepared=self.r.prepare_recovery()
            except Exception as exc:
                self.store.append('RecoveryRejected',dict(level='R1',reason=type(exc).__name__+': '+str(exc)))
                if self.store.unresolved() or self.r.c.quarantined:failure='UNKNOWN_PHYSICAL_EFFECT'
                elif isinstance(exc,(TimeoutError,ConnectionError)):failure='TRANSIENT_TOOL_ERROR'
                else:failure='ENVIRONMENT_FAILURE'
        progress=self.supervisor.progress();self.store.append('VerificationResult',dict(progress=progress,scope='post_reconciliation_public_task_conditions'))
        code=self.r.deterministic_code() if prepared else None
        decision=self.orchestrator.decide(failure,bool(code),requests_left=max(recorded_programs,self.supervisor.remaining()['recovery_requests']),no_progress=self.supervisor.no_progress)
        if any(self.supervisor.remaining()[k]<=0 for k in ('api','skill','seconds')):decision=dict(level='R4',reason='execution_budget')
        decision.update(program_source='recorded_non_llm' if recorded_programs and decision['level']=='R2' else 'runtime',failure=failure,deterministic_reason=self.r.terminal_reason,progress=progress)
        self.history.append(decision);self.store.append('RecoveryProposed',decision)
        self.checkpoint(actor)
        if decision['level']=='R2':
            self.r.terminal_reason=None;self.r.agent_recovery=True
            # Work-list advice is not a safety permit. Contract preconditions,
            # completed-call guards, actor budgets and physical checks remain.
        elif decision['level']=='R1':self.r.agent_recovery=False
        return decision,code

    def finish(self,result):
        success=result.get('raw_success') is True and result.get('reward')==1
        kind='TaskCompleted' if success else 'TaskSuspended' if not result.get('error') else 'TaskTerminated'
        self.store.append(kind,dict(reason=result.get('stop_reason'),native_progress=result.get('native_progress'),
            reward=result.get('reward'),raw_success=result.get('raw_success'),channel='independent_evaluator_not_agent'))
        self.checkpoint(dict(status=kind))
