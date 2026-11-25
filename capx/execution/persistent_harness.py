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


def classify_failure(actor,unknown=False):
    if unknown:return 'UNKNOWN_PHYSICAL_EFFECT'
    status=actor.get('status','');msg=str(actor.get('error',''))+' '+str(actor.get('stderr',''))
    if actor.get('failure_class') in STOP_FAILURES:return actor['failure_class']
    if status=='system_error':return 'ENVIRONMENT_FAILURE'
    if status in ('invalid_program','policy_violation') or any(x in msg for x in ('SyntaxError','NameError','TypeError')):return 'PROGRAM_ERROR'
    if any(x in msg.lower() for x in ('connection','http','service unavailable')):return 'TRANSIENT_TOOL_ERROR'
    if any(x in msg.lower() for x in ('unknown','geometry','evidence','visible','boundary','uncertain')):return 'PERCEPTION_UNCERTAIN'
    if any(x in msg.lower() for x in ('dependency','invalidated','predecessor')):return 'DEPENDENCY_INVALIDATION'
    if status=='ok':return 'PERCEPTION_UNCERTAIN'  # Program return is not task completion.
    return 'PHYSICAL_EXECUTION_FAILURE'


class RecoveryOrchestrator:
    def decide(self,failure,deterministic,*,requests_left,no_progress):
        if failure in STOP_FAILURES:return dict(level='R4',reason=failure.lower())
        if no_progress>=2:return dict(level='R4',reason='no_information_progress')
        if deterministic:return dict(level='R1',reason='verified_existing_continuation')
        if requests_left<=0:return dict(level='R4',reason='recovery_request_budget')
        return dict(level='R2',reason='deterministic_capability_exhausted')


class TaskSupervisor:
    def __init__(self,boundary,runtime,budget,request_limit):
        self.b=boundary;self.r=runtime;self.budget=budget;self.request_limit=request_limit
        self.requests=0;self.no_progress=0;self.last_progress=None

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
        key=digest([state['phase'],state['attachment'],state['remaining_goals'],
            {k:v['truth'] for k,v in state['conditions'].items()},public_motion_signature(self.r.skills)])
        same=key==self.last_progress
        self.no_progress=self.no_progress+1 if same else 0;self.last_progress=key
        return dict(changed=not same,no_progress=self.no_progress,signature=key)


class ExecutionMonitor:
    PHYSICAL={'goto_pose','open_gripper','close_gripper','wait','home_pose'}
    READONLY={'capture_observation','inspect_region','graph_query','execution_state','verify_conditions','refresh_evidence'}

    def __init__(self,store,observation,boundary):
        self.store=store;self.observation=observation;self.boundary=boundary;self.version=0;self.counter=0;self.last_failure=None

    def invoke(self,name,args,kwargs,function):
        from .skill_contracts import SkillBoundaryStop
        self.counter+=1;key=f'p{self.version}:call{self.counter}'
        physical=name in self.PHYSICAL
        site=copy.deepcopy(getattr(self.boundary.actor,'call_site',None))
        self.store.start_action(key,name,dict(args=args,kwargs=kwargs),physical=physical,site=site)
        before=self.observation()
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
            state='UNKNOWN' if physical and not isinstance(exc,SkillBoundaryStop) else 'FAILED'
            data=dict(error=type(exc).__name__+': '+str(exc),before=before,after=self.observation(),site=site)
            self.store.finish_action(key,state,data)
            self.last_failure=dict(action_id=key,name=name,state=state,**data)
            self.store.append('FailureDetected',self.last_failure)
            raise
        self.store.finish_action(key,'DONE',dict(value=plain(result),before=before,after=self.observation()))
        return result


class AgentController:
    def __init__(self,store,budget,supervisor):self.store=store;self.budget=budget;self.supervisor=supervisor
    def generate(self,base,context,label):
        self.supervisor.requests+=1
        messages=base+[dict(role='user',content='RECOVERY API EDITION 33 supersedes the INITIAL tool-list and work-list restrictions. The candidate list is advice, not an action permit. An empty list does not forbid the documented open recovery APIs. Every action is still checked by the parent.\n'+API_DOC+'\nRecovery context (current physics persists; no Python variables persist):\n'+json.dumps(context,ensure_ascii=False)+
            '\nWrite a self-contained Python recovery program. Query tools if needed. Use fresh operation/repair IDs; never replay completed calls. '
            'Preserve completed goals. Only documented APIs exist. Do not silently repeat a no-information attempt. '
            'Use at most one deliberate observation-motion strategy in this program; verify its result, then continue eligible goals. '
            'If no permitted strategy exists, print the specific missing capability. Return a python fenced block.')]
        return self.budget.query(messages,label)


class PersistentHarness:
    def __init__(self,boundary,runtime,budget,*,request_limit=2):
        self.b=boundary;self.r=runtime;self.store=EventStore(boundary.root/'harness')
        self.tools=OpenRecoveryAPIs(runtime,self.store)
        self.supervisor=TaskSupervisor(boundary,runtime,budget,request_limit)
        self.monitor=ExecutionMonitor(self.store,self.tools.observation,boundary)
        self.controller=AgentController(self.store,budget,self.supervisor)
        self.orchestrator=RecoveryOrchestrator();self.history=[];self.original=None;self.current=None;self.version=-1
        boundary.action_observer=self.monitor.invoke
        boundary.public_capture_sink=lambda data:self.store.append('ObservationCaptured',data)
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
        progress=self.supervisor.progress();self.store.append('VerificationResult',dict(progress=progress,scope='public_task_conditions'))
        failure=classify_failure(dict(actor,error=self.r.last_error or actor.get('error','')),
            bool(self.store.unresolved()) or self.r.c.quarantined)
        if self.r.terminal_reason in ('evidence_budget',):failure='PERCEPTION_UNCERTAIN'
        prepared=False
        if failure not in STOP_FAILURES:
            self.r.terminal_reason=None
            try:prepared=self.r.prepare_recovery()
            except Exception as exc:
                self.store.append('RecoveryRejected',dict(level='R1',reason=type(exc).__name__+': '+str(exc)))
                if self.store.unresolved() or self.r.c.quarantined:failure='UNKNOWN_PHYSICAL_EFFECT'
                elif isinstance(exc,(TimeoutError,ConnectionError)):failure='TRANSIENT_TOOL_ERROR'
                else:failure='ENVIRONMENT_FAILURE'
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
